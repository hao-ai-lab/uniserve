"""Public execution contexts bind direct communicators.

The communicators bind to Green Context streams.
"""

import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.model import TextSize
from uniserve.nn import ColumnParallelLinear
from uniserve.nn.attention import AttentionParallelConfig, Ulysses
from uniserve.runtime import CUDAGraph, ExecutionContext, partition_streams
from uniserve.runtime.execution import close_stream_collectives
from uniserve.runtime.process_groups import initialize_process_groups

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

#: How long the peer keeps its communicator bound while the failing rank
#: releases. A collective retirement cannot complete within it, so a release
#: that returns well inside it did not wait for the peer.
PEER_HOLD_SECONDS = 30.0


class _Collectives(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        group = mesh.get_group("tokens")
        self.group = group
        self.register_buffer("reference", torch.empty(0, device=group.device))
        self.projection = ColumnParallelLinear(
            4, 4, bias=False, device=group.device
        )
        parallelize_(
            self.projection,
            mesh,
            attention=AttentionParallelConfig(heads=Ulysses("tokens")),
        )
        self.projection.weight = nn.Parameter(
            torch.eye(4, device=group.device), requires_grad=False
        )

    def forward(self, value):
        group = self.group
        half = value.shape[0] // 2
        destination = (
            value.new_empty((2 * value.shape[0], value.shape[1]))
            if group.rank == 0
            else None
        )
        projected = value.new_empty(
            (group.size * value.shape[0], value.shape[1])
        )
        total = None
        start = group.rank * value.shape[0]
        for interval, chunk in self.projection.forward_chunks(
            value,
            token_slice=slice(start, start + value.shape[0]),
            num_tokens=projected.shape[0],
        ):
            # A numerical consumer can invoke a collective while its outer
            # projection still retains unread remote rows.
            if total is None:
                total = group.all_reduce(value, out=torch.empty_like(value))
            projected[interval].copy_(chunk)
        return (
            group.all_reduce(value, out=torch.empty_like(value)),
            group.all_reduce(value, op="max", out=torch.empty_like(value)),
            group.all_reduce(value, op="min", out=torch.empty_like(value)),
            group.all_gather(value),
            group.broadcast(value, src=0, out=torch.empty_like(value)),
            group.reduce_scatter(value),
            group.all_to_all(
                value, input_splits=(half, half), output_splits=(half, half)
            ),
            group.gather(value, dst=0, out=destination),
            group.send_recv(
                value,
                dst=1 - group.rank,
                src=1 - group.rank,
                out=torch.empty_like(value),
            ),
            projected,
            total,
        )


@torch.inference_mode()
def _run_collectives(rank: int, rendezvous: str):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    ) as environment:
        mesh = environment.bind(
            DeviceMesh(ranks=(1, 0), shape=(2,), axes=("tokens",), rank=rank),
            device=device,
        )
        group = mesh.get_group("tokens")
        module = _Collectives(mesh)
        greens = partition_streams(device, (64, 88))
        try:
            for green in greens:
                with ExecutionContext(module, stream=green.stream) as context:
                    # Replacing a preparation must retire its registrations and
                    # plans. The same numerical module remains callable with a
                    # different row capacity after every graph reader retires.
                    for rows in (2, 4):
                        context.prepare(TextSize(2 * rows, 1))
                        value = (
                            torch.arange(
                                rows * 4, dtype=torch.float32, device=device
                            ).view(rows, 4)
                            + rank * 10
                        )
                        module(value)
                        with CUDAGraph(context=context) as graph:
                            graph.capture(lambda: module(value))
                            for iteration in range(2):
                                value.copy_(
                                    torch.arange(rows * 4, device=device).view(
                                        rows, 4
                                    )
                                    + rank * 10
                                    + iteration
                                )
                                actual = graph.replay()
                                green.stream.synchronize()
                                base = (
                                    torch.arange(
                                        rows * 4,
                                        dtype=torch.float32,
                                        device=device,
                                    ).view(rows, 4)
                                    + iteration
                                )
                                peers = [
                                    base + member * 10 for member in group.ranks
                                ]
                                expected = (
                                    base * 2 + 10,
                                    base + 10,
                                    base,
                                    torch.cat(peers),
                                    base + 10,
                                    (base * 2 + 10).chunk(2)[group.rank],
                                    torch.cat(
                                        [
                                            peer.chunk(2)[group.rank]
                                            for peer in peers
                                        ]
                                    ),
                                    torch.cat(peers)
                                    if group.rank == 0
                                    else None,
                                    base + (1 - rank) * 10,
                                    torch.cat(peers),
                                    base * 2 + 10,
                                )
                                for result, reference in zip(
                                    actual, expected, strict=True
                                ):
                                    if reference is None:
                                        assert result is None
                                    else:
                                        torch.testing.assert_close(
                                            result, reference, rtol=0, atol=0
                                        )
                        green.stream.synchronize()
        finally:
            for green in greens:
                green.close()


def test_stream_collectives_preserve_values_and_rank_order_under_capture(
    tmp_path: Path,
):
    mp.spawn(
        _run_collectives, ((tmp_path / "world").as_uri(),), nprocs=2, join=True
    )


@torch.inference_mode()
def _run_release_after_failure(rank: int, rendezvous: str):
    """Release a bound communicator on one rank while its peer keeps it.

    Rank 0 stands in for a rank that has failed and is unwinding; rank 1 stands
    in for a peer that is still serving and will never join a collective
    retirement.
    """
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    ) as environment:
        mesh = environment.bind(
            DeviceMesh(ranks=(0, 1), shape=(2,), axes=("tokens",), rank=rank),
            device=device,
        )
        module = _Collectives(mesh)
        green = partition_streams(device, (64,))[0]
        try:
            with ExecutionContext(module, stream=green.stream) as context:
                context.prepare(TextSize(4, 1))
                module(
                    torch.arange(8, dtype=torch.float32, device=device).view(
                        2, 4
                    )
                )
            green.stream.synchronize()

            # Both ranks hold a bound communicator before either moves on.
            dist.barrier()

            if rank == 0:
                started = time.monotonic()
                close_stream_collectives(green.stream, aborted=True)
                elapsed = time.monotonic() - started
                assert elapsed < PEER_HOLD_SECONDS / 3, (
                    "releasing after a failure waited for a peer that is "
                    f"still holding its communicator: {elapsed:.1f}s"
                )
            else:
                # Outlive rank 0's release, so that release cannot be
                # completed by this rank going away.
                time.sleep(PEER_HOLD_SECONDS)
                close_stream_collectives(green.stream, aborted=True)
        finally:
            green.close()


def test_release_after_failure_does_not_wait_for_a_serving_peer(
    tmp_path: Path,
):
    mp.spawn(
        _run_release_after_failure,
        ((tmp_path / "failure").as_uri(),),
        nprocs=2,
        join=True,
    )
