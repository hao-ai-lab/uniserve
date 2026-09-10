"""Attention head shards return to sequence owners through registered storage."""

import pytest
import torch
import torch.multiprocessing as mp

from uniserve_worker.execution.bounded_storage import BoundedTensorStorage, TensorSchema
from uniserve_worker.nn.attention import RadixAttention
from uniserve_worker.nn.parallel import ParallelConfig, SequenceParallel
from uniserve_worker.nn.parallel_attention import (
    AttentionRowExchange,
    HeadRowPreparation,
    ParallelAttention,
)
from uniserve_worker.nn.parallel_sequence import SequencePartition
from uniserve_worker.runtime.attention_storage import allocate_attention_exchange_storage
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
        for shape, mesh, mode in (
            (shape, mesh, mode)
            for shape in ((64, 4, 128), (32784, 16, 128))
            for mesh in meshes.values()
            for mode in ("whole", "chunked", "produced")
        ):
            group = mesh.get_group("ulysses")
            attention = ParallelAttention(mesh=mesh)
            storage = BoundedTensorStorage.allocate(
                {
                    name: TensorSchema(
                        (shape[0] * 2, *shape[1:]) if name == "incoming" else shape,
                        torch.bfloat16,
                        memory="symmetric",
                        group=group,
                    )
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

            def execute(outgoing=outgoing, incoming=incoming, staging=staging):
                def produce(interval, destinations):
                    for owner, destination in enumerate(destinations):
                        destination.copy_(outgoing.view(2, local_rows, *shape[1:])[owner, interval])

                exchange = (
                    AttentionRowExchange(attention, staging, outgoing, produce)
                    if mode == "produced"
                    else AttentionRowExchange(attention, outgoing, staging)
                )
                if mode == "whole":
                    return exchange.materialize()
                result = torch.empty_like(expected)
                _, spare = exchange.partition_workspace(incoming, outgoing)
                for interval, values in exchange.chunks(incoming):
                    spare.fill_(17)
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
                    f"rank={rank}, members={group.ranks}, shape={shape}, mode={mode}\n{message}"
                ),
            )
            graph.reset()
            del execute, actual, outgoing, incoming, staging, storage
    finally:
        environment.close()


def test_attention_row_exchange_replays_updated_values_in_logical_rank_order(tmp_path):
    mp.spawn(_run_exchange, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


def _run_head_rows(rank, rendezvous):
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
        environment, {"ordered": ((0, 1), parallel), "reversed": ((1, 0), parallel)}
    )
    try:
        for mesh in meshes.values():
            attention = ParallelAttention(mesh=mesh)
            for rows in (1, 3, 19, 131075):
                partition = SequencePartition(rows, mesh.get_group("ulysses"))
                for kv_heads in (1, 2):
                    widths = (4 * 128, kv_heads * 128, kv_heads * 128)
                    complete = (
                        torch.arange(rows * sum(widths), device=device)
                        .remainder(17)
                        .reshape(rows, sum(widths))
                        .bfloat16()
                    )
                    local = partition.local(complete).clone()
                    storage = allocate_attention_exchange_storage(
                        (RadixAttention(4, kv_heads, 128, sequence=partition.group),),
                        environment,
                        max_tokens=max(19, rows),
                        dtype=torch.bfloat16,
                        scope=("head_rows", kv_heads),
                    )[partition.group]

                    independent_storage = allocate_attention_exchange_storage(
                        (RadixAttention(4, kv_heads, 128, sequence=partition.group),),
                        environment,
                        max_tokens=max(19, rows),
                        dtype=torch.bfloat16,
                        scope=("independent_head_rows", kv_heads),
                    )[partition.group]

                    def execute(storage=storage):
                        prepared = HeadRowPreparation(attention, partition, storage)
                        boundary = min(65536 if rows > 65536 else 8, partition.count)
                        for interval in (slice(0, boundary), slice(boundary, partition.count)):
                            if interval.start == interval.stop and partition.count > 0:
                                continue
                            values = tuple(
                                value.reshape(value.shape[0], width // 128, 128)
                                for value, width in zip(
                                    local[interval].split(widths, dim=1), widths
                                )
                            )
                            prepared.append(interval, values)
                            if partition.count == 0:
                                break
                        result = prepared.finish()
                        return result.query, result.key, result.value

                    def expected(offset):
                        result = []
                        for value, width in zip(complete.split(widths, dim=1), widths):
                            count, begin = attention.head_region(width // 128)
                            result.append(
                                (
                                    value.reshape(rows, width // 128, 128)[:, begin : begin + count]
                                    + offset
                                )
                            )
                        return result

                    actual = execute()
                    for value, wanted in zip(actual, expected(0)):
                        torch.testing.assert_close(value, wanted, rtol=0, atol=0)
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = execute()
                    local.add_(1)
                    graph.replay()
                    for value, wanted in zip(actual, expected(1)):
                        torch.testing.assert_close(value, wanted, rtol=0, atol=0)
                    local.add_(1)
                    independent = execute(independent_storage)
                    for value, wanted in zip(independent, expected(2)):
                        torch.testing.assert_close(value, wanted, rtol=0, atol=0)
                    for value, wanted in zip(actual, expected(1)):
                        torch.testing.assert_close(value, wanted, rtol=0, atol=0)
                    graph.reset()
                    del execute, storage, independent_storage, actual, independent, value, wanted
    finally:
        environment.close()


def test_projected_head_rows_preserve_global_order_and_gqa_under_replay(tmp_path):
    mp.spawn(_run_head_rows, args=((tmp_path / "head_rows").as_uri(),), nprocs=2, join=True)
