"""Green Context execution-partition resource and causality proofs."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.batch import Domain
from uniserve_worker.execution.lane import (
    ExecutionPartitionRuntime,
    create_green_contexts,
    verify_graph_context,
)
from uniserve_worker.foundation.runtime_config import LaneConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


class _GraphOwner:
    def __init__(self, graph: torch.cuda.CUDAGraph | None = None) -> None:
        self.graph = graph

    def close(self) -> None:
        if self.graph is not None:
            self.graph.reset()
            self.graph = None


def _gb200_lanes() -> tuple[LaneConfig, LaneConfig]:
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    if int(properties.multi_processor_count) != 152:
        pytest.skip("the exact 64/88 lane proof targets a 152-SM GB200")
    return (
        LaneConfig("decode", 64, (Domain.DECODE,)),
        LaneConfig("compute", 88, (Domain.PREFILL, Domain.FLOW)),
    )


def test_green_context_lanes_are_exact_disjoint_and_event_ordered() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    contexts = create_green_contexts(_gb200_lanes(), device)
    runtimes = tuple(
        ExecutionPartitionRuntime(
            lane=context.lane,
            device=device,
            stream=context.stream,
            sm_count=context.sm_count,
            buffer=object(),
            graphs=_GraphOwner(),
            green=context,
        )
        for context in contexts
    )
    try:
        assert tuple(runtime.sm_count for runtime in runtimes) == (64, 88)
        assert len({runtime.green_context_handle for runtime in runtimes}) == 2
        for runtime in runtimes:
            runtime.verify_stream()

        value = torch.zeros(4096, device=device)
        producer = torch.cuda.Event(blocking=False)
        reader = torch.cuda.Event(blocking=False)
        with torch.cuda.stream(runtimes[0].stream):
            value.fill_(7)
            producer.record(runtimes[0].stream)
        with torch.cuda.stream(runtimes[1].stream):
            runtimes[1].stream.wait_event(producer)
            result = value + 5
            reader.record(runtimes[1].stream)
        reader.synchronize()
        torch.testing.assert_close(result, torch.full_like(result, 12))

        completions = [torch.cuda.Event(blocking=False) for _ in runtimes]
        for runtime, completion in zip(runtimes, completions, strict=True):
            with torch.cuda.nvtx.range(f"independent_lane:{runtime.lane_id}"):
                with torch.cuda.stream(runtime.stream):
                    torch.cuda._sleep(20_000_000)
                    completion.record(runtime.stream)
        for completion in completions:
            completion.synchronize()
    finally:
        for runtime in reversed(runtimes):
            runtime.close()


def test_graph_compute_nodes_keep_the_owning_green_context() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    contexts = create_green_contexts(_gb200_lanes(), device)
    graphs = tuple(torch.cuda.CUDAGraph(keep_graph=True) for _ in contexts)
    runtimes = tuple(
        ExecutionPartitionRuntime(
            lane=context.lane,
            device=device,
            stream=context.stream,
            sm_count=context.sm_count,
            buffer=object(),
            graphs=_GraphOwner(graph),
            green=context,
        )
        for context, graph in zip(contexts, graphs, strict=True)
    )
    try:
        values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for context, graph in zip(contexts, graphs, strict=True):
            with torch.cuda.stream(context.stream):
                source = torch.arange(4096, device=device, dtype=torch.float32)
                output = source.square()
            context.stream.synchronize()
            with torch.cuda.graph(graph, stream=context.stream):
                output.copy_(source.square())
            graph.instantiate()
            assert verify_graph_context(graph, int(context.context)) > 0
            values.append((source, output))
        for runtime, graph in zip(runtimes, graphs, strict=True):
            with torch.cuda.nvtx.range(f"graph_replay_lane:{runtime.lane_id}"):
                with torch.cuda.stream(runtime.stream):
                    graph.replay()
        for runtime, (source, output) in zip(runtimes, values, strict=True):
            runtime.stream.synchronize()
            torch.testing.assert_close(output, source.square())
    finally:
        for runtime in reversed(runtimes):
            runtime.close()
