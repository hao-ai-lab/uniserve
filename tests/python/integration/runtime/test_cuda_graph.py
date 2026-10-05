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
