"""A captured graph owns its private pool, not its context's stream."""

import pytest
import torch

from uniserve.nn import Linear
from uniserve.runtime import ExecutionContext
from uniserve.runtime.cuda_graph import CUDAGraph

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def test_closing_a_graph_keeps_its_siblings_replayable():
    """Closing a graph preserves its siblings' eager-equivalent results."""
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
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
