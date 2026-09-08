"""Attention head shards return to sequence owners through registered storage."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from uniserve_worker.execution.bounded_storage import BoundedTensorStorage, TensorSchema
from uniserve_worker.nn.parallel import ParallelConfig, SequenceParallel
from uniserve_worker.nn.parallel_attention import AttentionRowExchange, ParallelAttention
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _run_exchange(rank: int, rendezvous: str) -> None:
    device = torch.device("cuda", rank)
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=str(device),
        backend="nccl",
        init_method=rendezvous,
    )
    parallel = ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (2,)))
    meshes = initialize_model_parallel(
        environment,
        {"ordered": ((0, 1), parallel), "reversed": ((1, 0), parallel)},
    )
    try:
        for shape, mesh, chunked in (
            (shape, mesh, chunked)
            for shape in ((64, 4, 128), (32784, 16, 128))
            for mesh in meshes.values()
            for chunked in (False, True)
        ):
            group = mesh.get_group("ulysses")
            attention = ParallelAttention(mesh=mesh)
            storage = BoundedTensorStorage.allocate(
                {
                    name: TensorSchema(shape, torch.bfloat16, memory="symmetric", group=group)
                    for name in ("outgoing", "incoming", "staging")
                },
                device,
                environment=environment,
            )
            outgoing, incoming = (storage.capacity[name] for name in ("outgoing", "incoming"))
            staging = storage.capacity["staging"]
            rows = (torch.arange(outgoing.numel(), device=device) % 13).view(shape)
            outgoing.copy_(rows + rank * 64)
            local_rows = shape[0] // group.world_size
            begin = group.rank_in_group * local_rows
            expected = torch.cat(
                [rows[begin : begin + local_rows] + member * 64 for member in group.ranks], dim=1
            ).to(torch.bfloat16)

            def execute():
                exchange = AttentionRowExchange(attention, outgoing, staging)
                if not chunked:
                    return exchange.materialize()
                result = torch.empty_like(expected)
                for interval, values in exchange.chunks(incoming):
                    result[interval].copy_(values)
                return result

            actual = execute()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.cuda.synchronize(device)
            outgoing.copy_(rows + rank * 64)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = execute()
            outgoing.copy_(rows + rank * 64 + 1)
            graph.replay()
            torch.testing.assert_close(
                actual,
                expected + 1,
                rtol=0,
                atol=0,
                msg=lambda message: (
                    f"rank={rank}, members={group.ranks}, shape={shape}, chunked={chunked}\n{message}"
                ),
            )
            graph.reset()
            del actual, outgoing, incoming, staging, storage
    finally:
        environment.close()
        dist.destroy_process_group()


def test_attention_row_exchange_replays_updated_values_in_logical_rank_order(tmp_path):
    mp.spawn(_run_exchange, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
