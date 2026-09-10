"""Row gathering and projection preserve values through CUDA Graph replay."""

import pytest
import torch
import torch.multiprocessing as mp

from uniserve_worker.execution.bounded_storage import BoundedTensorStorage, TensorSchema
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.linear import LinearBase
from uniserve_worker.nn.mesh import Communicator
from uniserve_worker.nn.parallel import ParallelConfig, SequenceParallel
from uniserve_worker.nn.row_pipeline import ProjectedRows, RowStage, run_row_pipeline
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.no_grad()
def _run_gather(rank: int, rendezvous: str) -> None:
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
        for mesh in meshes.values():
            group = mesh.get_group("sp")
            for dtype in (torch.bfloat16, torch.uint8):
                storage = BoundedTensorStorage.allocate(
                    {"rows": TensorSchema((8192,), dtype, memory="symmetric", group=group)},
                    device,
                    environment=environment,
                )
                gathered = storage.capacity["rows"]
                local = (torch.arange(4096, device=device) % 31 + rank * 64).to(dtype)
                expected = torch.cat(
                    [
                        (torch.arange(4096, device=device) % 31 + member * 64).to(dtype)
                        for member in group.ranks
                    ]
                )
                group.all_gather_into_tensor(gathered, local)
                torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    group.all_gather_into_tensor(gathered, local)
                local.add_(1)
                graph.replay()
                torch.testing.assert_close(gathered, expected + 1, rtol=0, atol=0)
                graph.reset()
                del gathered, storage
            for local_rows, width in ((128, 512), (16392, 2048)):
                storage = BoundedTensorStorage.allocate(
                    {
                        "rows": TensorSchema(
                            (local_rows * width * 4,),
                            torch.uint8,
                            memory="symmetric",
                            group=group,
                        )
                    },
                    device,
                    environment=environment,
                )
                columns = (torch.arange(width, device=device) % 17).bfloat16() / 16
                rows = columns.repeat(local_rows, 1).add_(rank)
                rows.add_((torch.arange(local_rows, device=device) % 7).bfloat16()[:, None] / 16)
                expected_rows = torch.cat([rows - rank + member for member in group.ranks])
                for use_bias in (False, True):
                    layer = LinearBase(
                        width,
                        256,
                        layer_config=LayerConfig(Communicator(), None),
                        bias=use_bias,
                        sequence_group=group,
                    ).to(device=device, dtype=torch.bfloat16)
                    layer.weight.copy_(
                        (torch.arange(256 * width, device=device).reshape(256, width) % 11) / 16
                    )
                    if local_rows > 128:
                        # The transport-tail case selects columns exactly, so
                        # GEMM reduction heuristics cannot alter its oracle.
                        layer.weight.zero_()
                        columns = torch.arange(256, device=device)
                        layer.weight[columns, columns * 7 % width] = 1
                    if layer.bias is not None:
                        layer.bias.copy_(torch.arange(256, device=device) / 16)
                    expected = torch.nn.functional.linear(expected_rows, layer.weight, layer.bias)
                    actual = layer.forward_sequence_parallel(rows, storage.capacity["rows"])
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=0,
                        atol=0,
                        msg=f"ranks={group.ranks}, rows={local_rows}, bias={use_bias}",
                    )
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = layer.forward_sequence_parallel(rows, storage.capacity["rows"])
                    rows.add_(1)
                    expected_rows.add_(1)
                    graph.replay()
                    expected = torch.nn.functional.linear(expected_rows, layer.weight, layer.bias)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    graph.reset()
                    actual = layer.forward_sequence_parallel(
                        rows.view(8, local_rows // 8, width), storage.capacity["rows"]
                    )
                    torch.testing.assert_close(
                        actual, expected.view(16, local_rows // 8, 256), rtol=0, atol=0
                    )

                    def produce(workspace=storage.capacity["rows"]):
                        def consume(interval, projected):
                            positions = torch.arange(interval.start, interval.stop, device=device)
                            projected.add_(positions.remainder(3).to(projected.dtype)[:, None])

                        projection = layer.stream_sequence_parallel(
                            local_rows, workspace, row_consumer=consume
                        )
                        interval_rows = 3072 if local_rows > 128 else 48
                        for start in range(0, local_rows, interval_rows):
                            projection.append(start, rows[start : start + interval_rows])
                        return projection.finish()

                    actual = produce()
                    positions = torch.arange(local_rows * 2, device=device)
                    row_bias = positions.remainder(3).to(expected.dtype)[:, None]
                    torch.testing.assert_close(actual, expected + row_bias, rtol=0, atol=0)
                    torch.cuda.synchronize(device)
                    with torch.cuda.graph(graph):
                        actual = produce()
                    rows.add_(1)
                    expected_rows.add_(1)
                    graph.replay()
                    expected = torch.nn.functional.linear(expected_rows, layer.weight, layer.bias)
                    torch.testing.assert_close(actual, expected + row_bias, rtol=0, atol=0)
                    graph.reset()
                del storage
            rows = torch.arange(128 * 128, device=device).view(128, 128).remainder(7).bfloat16()
            rows.add_(rank)
            layers = [
                LinearBase(
                    128,
                    128,
                    layer_config=LayerConfig(Communicator(), None),
                    bias=False,
                    sequence_group=group,
                ).to(device=device, dtype=torch.bfloat16)
                for _ in range(3)
            ]
            for layer in layers:
                layer.weight.copy_(torch.eye(128, device=device))
            workspaces = [
                torch.empty(2 * 2 * 128 * 128, device=device, dtype=torch.bfloat16) for _ in layers
            ]

            def bind_stage(index, layer):
                def operation(hidden, *, prepared_projection, row_consumer):
                    projected = (
                        layer.forward_sequence_parallel(hidden, workspaces[index])
                        if prepared_projection is None
                        else prepared_projection.finish()[0]
                    )
                    result = projected.view(2, 128, 128).sum(dim=0)
                    if row_consumer is not None:
                        for start in range(0, 128, 48):
                            interval = slice(start, min(start + 48, 128))
                            row_consumer(interval, result[interval])
                    return result

                def prepare(hidden):
                    return ProjectedRows(
                        layer.stream_sequence_parallel(hidden.shape[0], workspaces[index]), 0
                    )

                return RowStage(operation, True, prepare)

            stages = tuple(bind_stage(index, layer) for index, layer in enumerate(layers))
            actual = run_row_pipeline(rows, stages)
            expected = (rows - rank) * 8 + 4
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run_row_pipeline(rows, stages)
            rows.add_(1)
            graph.replay()
            torch.testing.assert_close(actual, expected + 8, rtol=0, atol=0)
            graph.reset()
    finally:
        environment.close()


def test_row_gather_and_projection_replay_updated_values_in_logical_rank_order(tmp_path):
    mp.spawn(_run_gather, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


@torch.inference_mode()
def _run_quantized_gather(rank: int, rendezvous: str) -> None:
    from uniserve_worker.nn.quant import DynamicW8A8Fp8LinearMethod, DynamicW8A8MxFp8LinearMethod

    device = torch.device("cuda", rank)
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=str(device),
        backend="nccl",
        init_method=rendezvous,
    )
    mesh = initialize_model_parallel(
        environment,
        {"model": ((1, 0), ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (2,))))},
    )["model"]
    group = mesh.get_group("sp")
    try:
        local_rows, width, outputs = 193, 512, 256
        torch.manual_seed(811)
        full = (torch.rand((local_rows * 2, width), device=device) + 0.25).bfloat16()
        magnitudes = torch.tensor([0.03125, 0.25, 2, 16], device=device)
        full.mul_(magnitudes[torch.arange(local_rows * 2, device=device) % 4, None])
        local = full.chunk(2)[group.rank_in_group].clone()
        storage = BoundedTensorStorage.allocate(
            {
                "rows": TensorSchema(
                    (4 * 128 * width,), torch.bfloat16, memory="symmetric", group=group
                )
            },
            device,
            environment=environment,
        )
        for method in (DynamicW8A8Fp8LinearMethod(), DynamicW8A8MxFp8LinearMethod()):
            layer = LinearBase(
                width,
                outputs,
                layer_config=LayerConfig(Communicator(), None),
                quant_method=method,
                sequence_group=group,
            ).to(device=device, dtype=torch.bfloat16)
            layer.weight.copy_((torch.rand_like(layer.weight) + 0.125) / 16)
            layer.bias.fill_(0.5)
            layer.finalize_weights()

            def execute():
                projection = layer.stream_sequence_parallel(local_rows, storage.capacity["rows"])
                for start in range(0, local_rows, 48):
                    projection.append(start, local[start : start + 48])
                return projection.finish()

            expected = layer(full)
            actual = layer.forward_sequence_parallel(local, storage.capacity["rows"])
            # Full and streamed projection use identical row/block scales.
            # The existing quantized projection contract allows BF16 rounding
            # differences from GEMM shape selection, bounded by 2**-7.
            torch.testing.assert_close(actual, expected, rtol=2**-7, atol=0)
            torch.testing.assert_close(execute(), expected, rtol=2**-7, atol=0)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = execute()
            full.mul_(2)
            local.mul_(2)
            graph.replay()
            torch.testing.assert_close(actual, layer(full), rtol=2**-7, atol=0)
            graph.reset()
    finally:
        environment.close()


def test_row_and_block_quantized_projection_preserves_scale_domains_under_replay(tmp_path):
    mp.spawn(_run_quantized_gather, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


@torch.inference_mode()
def _run_routed_scales(rank: int, rendezvous: str) -> None:
    from uniserve_worker.execution.forward_batch import ExpertRoute, RouteSpan
    from uniserve_worker.nn.expert_routing import RoutedTensor
    from uniserve_worker.nn.mlp import GatedMLP
    from uniserve_worker.nn.parallel_sequence import SequencePartition
    from uniserve_worker.nn.quant.base import process_quantized_modules
    from uniserve_worker.nn.quant.fp8 import DynamicW8A8Fp8LinearMethod
    from uniserve_worker.nn.quant.mxfp8 import DynamicW8A8MxFp8LinearMethod
    from uniserve_worker.nn.quant.nvfp4 import DynamicW4A4NvFp4LinearMethod

    device = torch.device("cuda", rank)
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    )
    mesh = initialize_model_parallel(
        environment,
        {"model": ((1, 0), ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (2,))))},
    )["model"]
    group = mesh.get_group("sp")
    graph = None
    try:
        for method in (
            DynamicW8A8Fp8LinearMethod(tensorwise=True),
            DynamicW8A8Fp8LinearMethod(),
            DynamicW8A8MxFp8LinearMethod(),
            DynamicW4A4NvFp4LinearMethod(),
        ):
            local_experts, full_experts = [], []
            for expert in range(2):
                local = GatedMLP(
                    512,
                    1024,
                    layer_config=LayerConfig(Communicator(), None, sequence=group),
                    quant_method=method,
                ).to(device=device, dtype=torch.bfloat16)
                full = GatedMLP(
                    512,
                    1024,
                    layer_config=LayerConfig(Communicator(), None),
                    quant_method=method,
                ).to(device=device, dtype=torch.bfloat16)
                for index, (left, right) in enumerate(
                    zip(local.parameters(), full.parameters(), strict=True)
                ):
                    values = (
                        (
                            torch.arange(left.numel(), device=device).reshape(left.shape)
                            + expert
                            + index
                        )
                        % 3
                        + 1
                    ) / 1024
                    left.copy_(values)
                    right.copy_(values)
                process_quantized_modules(local.modules())
                process_quantized_modules(full.modules())
                local_experts.append(local)
                full_experts.append(full)
            for rows in (1, 129):
                values = (
                    torch.arange(rows * 512, device=device).reshape(rows, 512) % 7 + 1
                ).bfloat16() / 16
                values.mul_((torch.arange(rows, device=device) % 3 + 1).unsqueeze(-1))
                split = (rows + 1) // 2
                spans = (RouteSpan(ExpertRoute.TEXT, 0, split),)
                if rows > split:
                    spans += (RouteSpan(ExpertRoute.FLOW, split, rows - split),)
                routes = frozenset(span.route for span in spans)
                partition = SequencePartition(rows, group)
                local_values = partition.local(values)
                local_spans = partition.routes(spans)

                def execute():
                    output = (
                        RoutedTensor.from_packed(local_values, local_spans, routes=routes)
                        .apply(
                            text=local_experts[0],
                            flow=local_experts[1],
                        )
                        .packed(local_spans)
                    )
                    return partition.gather(output)

                def reference():
                    return (
                        RoutedTensor.from_packed(values, spans)
                        .apply(
                            text=full_experts[0],
                            flow=full_experts[1],
                        )
                        .packed(spans)
                    )

                # The established quantized row projection contract permits
                # one BF16 ULP for GEMM row-shape selection, with no absolute floor.
                torch.testing.assert_close(execute(), reference(), rtol=2**-7, atol=0)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = execute()
                values.mul_(2)
                # The local slice may alias values when its physical layout already matches.
                local_values.copy_(partition.local(values))
                graph.replay()
                torch.testing.assert_close(actual, reference(), rtol=2**-7, atol=0)
                graph.reset()
                graph = None
    finally:
        if graph is not None:
            graph.reset()
        environment.close()


def test_quantized_routes_share_complete_scale_domains_with_empty_sequence_shards(tmp_path):
    mp.spawn(_run_routed_scales, args=((tmp_path / "routed-scales").as_uri(),), nprocs=2, join=True)
