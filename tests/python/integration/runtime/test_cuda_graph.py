"""A captured graph owns its private pool, not its context's stream."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from uniserve.nn import Linear
from uniserve.runtime import CUDAStream, ExecutionContext
from uniserve.runtime.cuda_graph import CUDAGraph, CUDAGraphError

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def test_capture_after_warmup_on_another_thread():
    """Numerical warmup does not require capture on the same host thread."""
    device = torch.device("cuda:0")
    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    module = Linear(256, 256, bias=False).to(
        device=device, dtype=torch.bfloat16
    )
    context = ExecutionContext(module, stream=stream)
    context.prepare(None)
    x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    graph = CUDAGraph(context=context)
    try:
        with context.activate():
            expected = module(x).clone()
        torch.cuda.synchronize(device)
        with ThreadPoolExecutor(max_workers=1) as thread:
            thread.submit(graph.capture, lambda: module(x)).result()
        actual = graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        graph.close()
        torch.cuda.synchronize(device)
        context.close()
        stream.close()


@torch.inference_mode()
def test_closing_a_graph_keeps_its_siblings_replayable():
    """Closing a graph preserves its siblings' eager-equivalent results."""
    device = torch.device("cuda:0")
    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    module = Linear(256, 256, bias=False).to(
        device=device, dtype=torch.bfloat16
    )
    context = ExecutionContext(module, stream=stream)
    context.prepare(None)
    x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    pool = torch.cuda.MemPool()
    graphs = []
    try:
        with context.activate():
            expected = module(x).clone()
        for _ in range(2):
            graph = CUDAGraph(context=context, pools={device: pool})
            graph.capture(lambda: module(x))
            graphs.append(graph)
        torch.cuda.synchronize(device)

        graphs[0].close()

        actual = graphs[1].replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        for graph in graphs:
            graph.close()
        torch.cuda.synchronize(device)
        context.close()
        stream.close()


@torch.inference_mode()
def test_capture_can_retry_after_a_failed_call():
    device = torch.device("cuda:0")
    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    module = Linear(256, 256, bias=False).to(
        device=device, dtype=torch.bfloat16
    )
    context = ExecutionContext(module, stream=stream)
    context.prepare(None)
    x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    pool = torch.cuda.MemPool()
    graph = CUDAGraph(context=context, pools={device: pool})
    failure = ValueError("numerical call failed")

    def rejected():
        module(x)
        raise failure

    try:
        with context.activate():
            expected = module(x).clone()

        with pytest.raises(CUDAGraphError) as raised:
            graph.capture(rejected)
        assert raised.value.__cause__ is failure

        graph.capture(lambda: module(x))
        actual = graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        graph.close()
        torch.cuda.synchronize(device)
        context.close()
        stream.close()


@torch.inference_mode()
def test_replay_advances_the_captured_random_generator():
    """Successive replays match eager draws after resetting the same seed."""
    device = torch.device("cuda:0")
    module = torch.nn.Linear(1, 1).to(device)
    with ExecutionContext(module) as context:
        context.prepare(None)
        with CUDAGraph(context=context) as graph:

            def draw():
                return torch.rand(128, device=device)

            with context.activate():
                draw()
            graph.capture(draw)
            torch.cuda.manual_seed(1234)
            with context.activate():
                actual = [graph.replay().clone() for _ in range(2)]
                torch.cuda.manual_seed(1234)
                expected = [draw() for _ in range(2)]
            torch.cuda.synchronize(device)
            for value, wanted in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("pending", [False, True])
@torch.inference_mode()
def test_caller_failure_keeps_inputs_only_while_device_work_is_pending(pending):
    device = torch.device("cuda:0")
    module = torch.nn.Linear(1, 1).to(device)
    with ExecutionContext(module) as context:
        context.prepare(None)
        torch.cuda.synchronize(device)
        source = torch.ones(4 << 20, device=device)
        input_bytes = source.numel() * source.element_size()
        graph = CUDAGraph(context=context)
        graph.capture(lambda source=source: source.add_(1))
        del source
        allocated = torch.cuda.memory_allocated(device)

        try:
            with pytest.raises(ValueError, match="caller"):
                with graph:
                    if pending:
                        torch.cuda._sleep(1 << 30)
                    graph.replay()
                    if not pending:
                        torch.cuda.synchronize(device)
                    raise ValueError("caller ended the invocation")

            # Measure releases across close: numerical libraries can retain
            # their own stream workspaces beyond an individual graph's life.
            released = allocated - torch.cuda.memory_allocated(device)
            assert released == 0 if pending else released >= input_bytes
        finally:
            torch.cuda.synchronize(device)
