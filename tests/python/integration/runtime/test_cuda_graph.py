"""A captured graph owns its private pool, not its context's stream."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from uniserve.nn import Linear
from uniserve.runtime import CUDAStream, ExecutionContext
from uniserve.runtime.cuda import CUDAError, verify_graph_context
from uniserve.runtime.cuda_graph import CUDAGraph

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
def test_graph_context_validation_rejects_foreign_compute():
    """Validation must inspect compute even when the graph is very small."""
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    values = torch.ones(16, device=device)
    output = torch.empty_like(values)
    with torch.cuda.stream(stream):
        torch.add(values, 1, out=output)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(graph, stream=stream):
            torch.add(values, 1, out=output)
        # No real CUDA context has the null handle. The captured computation
        # therefore lies outside this declared set of allowed contexts.
        with pytest.raises(CUDAError, match="escaped its owning context"):
            verify_graph_context(graph, frozenset({0}))
    finally:
        graph.reset()
