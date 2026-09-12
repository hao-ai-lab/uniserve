"""Bound communication preserves logical members through Green Context graph replay."""

from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from uniserve_worker.config import LaneConfig
from uniserve_worker.execution.batch import COMPUTATIONS, ForwardMode
from uniserve_worker.execution.graph.full import FullCudaGraphBackend
from uniserve_worker.execution.lane import create_green_contexts
from uniserve_worker.nn.collective import stream_collective_scope
from uniserve_worker.nn.parallel import ParallelConfig
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _run_collectives(rank: int, rendezvous: str):
    device = torch.device("cuda", rank)
    with init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    ) as environment:
        mesh = initialize_model_parallel(environment, {"model": ((1, 0), ParallelConfig(2))})[
            "model"
        ]
        group = mesh.get_group("tp")
        greens = create_green_contexts(
            (
                LaneConfig("decode", 64, (ForwardMode.DECODE, ForwardMode.VERIFY)),
                LaneConfig(
                    "compute",
                    88,
                    tuple(
                        kind
                        for kind in COMPUTATIONS
                        if kind not in {ForwardMode.DECODE, ForwardMode.VERIFY}
                    ),
                ),
            ),
            device,
        )
        try:
            for green in greens:
                bindings = environment.stream_collectives(green.stream)
                graph = FullCudaGraphBackend(
                    device=device, stream=green.stream, expected_context=int(green.context)
                )
                value = torch.arange(8, dtype=torch.float32, device=device).view(2, 4) + rank * 10

                def execute():
                    with stream_collective_scope(bindings):
                        total = group.all_reduce(value.clone())
                        maximum = group.all_reduce_max(value.clone())
                        minimum = group.all_reduce_min(value.clone())
                        gathered = torch.empty((4, 4), device=device)
                        group.all_gather_into_tensor(gathered, value)
                        broadcast = group.broadcast(value.clone(), src=0)
                        reduced = group.reduce_scatter(value)
                        exchanged = torch.empty_like(value)
                        group.all_to_all_single_into(exchanged, value, [1, 1], [1, 1])
                        destination = torch.empty((2, 2, 4), device=device) if rank == 1 else None
                        group.gather_into_tensor(destination, value, dst=0)
                        peer = torch.empty_like(value)
                        group.send_recv(
                            value, peer, dst=1 - group.rank_in_group, src=1 - group.rank_in_group
                        )
                        return (
                            total,
                            maximum,
                            minimum,
                            gathered,
                            broadcast,
                            reduced,
                            exchanged,
                            destination,
                            peer,
                        )

                try:
                    graph.capture_one("collectives", execute, keepalive=(value,))
                    for iteration in range(2):
                        with torch.cuda.stream(green.stream):
                            value.copy_(
                                torch.arange(8, device=device).view(2, 4) + rank * 10 + iteration
                            )
                            actual = graph.replay("collectives")
                        green.stream.synchronize()
                        base = (
                            torch.arange(8, dtype=torch.float32, device=device).view(2, 4)
                            + iteration
                        )
                        peers = [base + member * 10 for member in group.ranks]
                        expected = (
                            base * 2 + 10,
                            base + 10,
                            base,
                            torch.cat(peers),
                            base + 10,
                            (base * 2 + 10).chunk(2)[group.rank_in_group],
                            torch.cat([peer.chunk(2)[group.rank_in_group] for peer in peers]),
                            torch.stack(peers) if rank == 1 else None,
                            base + (1 - rank) * 10,
                        )
                        for result, reference in zip(actual, expected, strict=True):
                            if reference is None:
                                assert result is None
                            else:
                                torch.testing.assert_close(result, reference, rtol=0, atol=0)
                finally:
                    green.stream.synchronize()
                    graph.close()
                    for binding in bindings.values():
                        binding.close()
        finally:
            for green in greens:
                green.close()


def test_stream_collectives_preserve_values_and_rank_order_under_capture(tmp_path: Path):
    mp.spawn(_run_collectives, ((tmp_path / "world").as_uri(),), nprocs=2, join=True)
