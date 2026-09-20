"""A captured graph owns its private pool, not its context's stream."""

import pytest
import torch

from uniserve.nn import Linear
from uniserve.runtime import ExecutionContext
from uniserve.runtime.cuda_graph import CUDAGraph

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _shared_allocations(
    stream: torch.cuda.Stream,
) -> frozenset[tuple[int, int]]:
    """Return the live allocations made on ``stream`` outside any graph pool."""
    allocations = set()
    for segment in torch.cuda.memory_snapshot():
        if segment["stream"] != stream.cuda_stream:
            continue
        if segment["segment_pool_id"] != (0, 0):
            continue
        offset = segment["address"]
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                allocations.add((offset, block["size"]))
            offset += block["size"]
    return frozenset(allocations)


@torch.inference_mode()
def test_closing_a_graph_keeps_its_siblings_replayable():
    """Closing one graph releases only that graph's own storage.

    Several graphs captured on one context share the context's stream and the
    stream's library workspaces, so closing one must leave every allocation
    the stream made outside the graph pools in place and the other graphs
    replayable with eager values.
    """
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

        shared = _shared_allocations(stream)
        graphs[0].close()
        torch.cuda.empty_cache()
        assert _shared_allocations(stream) == shared

        actual = graphs[1].replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        for graph in graphs:
            graph.close()
        torch.cuda.synchronize(device)
        context.close()
