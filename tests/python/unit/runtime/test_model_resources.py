"""Model execution and worker resource behavior."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.model_execution import TEST_MODEL, TEST_WORKER_CONFIG
from uniserve_worker.bootstrap.capacity import (
    latent_trajectory_bytes,
    model_arena_capacity,
    request_tensor_window,
    tensor_slot_capacity,
)
from uniserve_worker.bootstrap.worker_info_builder import build_worker_layout
from uniserve_worker.execution.batch import OpCode
from uniserve_worker.execution.bounded_storage import TensorSchema
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.foundation.math import ceil_div
from uniserve_worker.nn.mesh import Communicator

pytestmark = pytest.mark.unit


def test_request_capacity_charges_only_device_storage_against_device_budget():
    schema = {
        "state": TensorSchema((128,), torch.float32),
        "initial_values": TensorSchema((1024,), torch.float32, memory="pinned"),
    }
    assert (
        tensor_slot_capacity(
            schema,
            Communicator(),
            maximum=8,
            minimum=2,
            available_bytes=2048,
            auxiliary_bytes=lambda slots: slots * 512,
        )
        == 2
    )
    with pytest.raises(RuntimeError, match="insufficient device memory"):
        tensor_slot_capacity(
            schema,
            Communicator(),
            maximum=8,
            minimum=2,
            available_bytes=2047,
            auxiliary_bytes=lambda slots: slots * 512,
        )


def test_request_capacity_accounts_for_the_candidate_output_horizon():
    schema = {"state": TensorSchema((128,), torch.float32)}
    # At depth 12, two, three, and four requests retain 10, 9, and 8
    # output batches respectively. Smaller counts need more product storage.
    count = tensor_slot_capacity(
        schema,
        Communicator(),
        maximum=4,
        minimum=2,
        available_bytes=10_240,
        auxiliary_bytes=lambda slots: slots * request_tensor_window(12, slots) * 1024,
    )
    assert count == 4


def test_worker_info_projects_model_behavior_and_resource_geometry():
    layout = build_worker_layout(
        TEST_MODEL,
        TEST_WORKER_CONFIG,
        model_name="test-model",
    )
    info = layout.info

    assert OpCode.AR_EXTEND in info.supported_ops
    assert OpCode.DIFFUSION_STEP in info.supported_ops
    assert layout.max_vision_feature_bytes == (
        int(TEST_MODEL.max_vit_grid_tokens) * int(TEST_MODEL.hidden_size) * 2
    )
    assert info.kv_cache is not None
    assert info.kv_cache.num_layers == TEST_MODEL.cache_geometry.num_layers
    assert info.model_name == "test-model"


def test_latent_capacity_rounds_to_complete_scheduler_pages() -> None:
    flow = TEST_MODEL.generation
    assert flow is not None
    worker_config = replace(
        TEST_WORKER_CONFIG,
        kv_token_capacity=int(flow.max_latent_tokens) + 1,
    )

    layout = build_worker_layout(TEST_MODEL, worker_config)

    expected_pages = ceil_div(
        int(flow.max_latent_tokens) + 1,
        int(worker_config.block_size),
    )
    assert layout.info.latent_pages == expected_pages + 1
    assert layout.info.latent_capacity_units == expected_pages * int(worker_config.block_size)


def test_persistent_buffer_capacity_includes_active_encoder_output() -> None:
    layout = build_worker_layout(TEST_MODEL, TEST_WORKER_CONFIG)
    feature_bytes = max(
        layout.max_latent_feature_bytes,
        layout.max_vision_feature_bytes,
    )
    assert (
        layout.info.buffer_pool_bytes
        == (int(TEST_MODEL.resource_geometry.encoder_cache_entries) + 1) * feature_bytes
    )


def test_transfer_capacity_covers_one_maximum_float32_trajectory_per_ticket() -> None:
    worker_config = replace(TEST_WORKER_CONFIG, model_dtype="float32")
    layout = build_worker_layout(TEST_MODEL, worker_config)
    flow = TEST_MODEL.generation
    assert flow is not None
    assert layout.max_latent_feature_bytes == latent_trajectory_bytes(
        int(flow.max_vae_grid_tokens),
        int(flow.latent_channels) * int(flow.latent_patch_size) ** 2,
        4,
    )
    arena = model_arena_capacity(
        TEST_MODEL,
        worker_config,
        pipeline_depth=1,
        completion_payload_bytes=1024,
        num_blocks=2,
        request_pool_size=4,
        num_latent_pages=5,
        latent_page_units=4,
        latent_width=1024,
        max_latent_feature_bytes=1,
        max_vision_feature_bytes=1,
        bytes_per_token=1,
    )
    expected = latent_trajectory_bytes(int(flow.max_latent_tokens), 1024, 4)
    assert arena.transfer_bytes == expected * arena.transfer_tickets


@pytest.mark.parametrize(
    "worker_config",
    [
        lambda: replace(TEST_WORKER_CONFIG, rank=1, world_size=1),
        lambda: replace(TEST_WORKER_CONFIG, world_size=0),
        lambda: replace(TEST_WORKER_CONFIG, block_size=0),
        lambda: replace(TEST_WORKER_CONFIG, model_dtype="bf16"),
    ],
)
def test_worker_worker_config_rejects_invalid_runtime_geometry(worker_config):
    with pytest.raises(WorkerError):
        worker_config()


def test_worker_reserves_declared_tensor_results_for_every_request() -> None:
    import torch

    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.batch import (
        BufferAllocation,
        DeviceDim,
        DType,
        PointRange,
        ProductKind,
        ProductRef,
        RequestKey,
        ShapeBound,
        StaticDim,
        StorageClass,
        TensorSpec,
    )
    from uniserve_worker.execution.bounded_storage import TensorSchema
    from uniserve_worker.models.runtime import (
        ExecutionModel,
        ResourceGeometry,
    )
    from uniserve_worker.runtime.persistent_buffers import PersistentBuffers

    model = ExecutionModel()
    model.architecture = "TensorEntryModel"
    model.supported_work = frozenset((OpCode.ENCODER_TEXT, OpCode.DIFFUSION_STEP))
    model.resource_geometry = ResourceGeometry(
        kv=False, request_tensors={"state": TensorSchema((4,), torch.float32)}
    )
    model.entry_outputs = {
        "text_encoder": (
            TensorSpec("conditioning", DType.F32, ShapeBound((DeviceDim(3), StaticDim(7)))),
        ),
        "denoiser": (TensorSpec("latent", DType.F32, ShapeBound((StaticDim(5), StaticDim(11)))),),
    }
    config = WorkerConfig(device="cpu", max_request_pool_size=2)
    info = build_worker_layout(model, config, queue_depth=8, completion_payload_bytes=1024).info
    arena = PersistentBuffers(byte_capacity=info.buffer_pool_bytes, devices=("cpu",))
    bindings = []
    offset = 0
    try:
        for request_id in range(1, info.request_slots + 1):
            for op_id, outputs in enumerate(model.entry_outputs.values(), start=1):
                output = outputs[0]
                shape = tuple(
                    dim.extent if isinstance(dim, StaticDim) else dim.bound
                    for dim in output.shape_bound.dims
                )
                product = ProductRef(
                    RequestKey(1, request_id, 1),
                    op_id,
                    0,
                    1,
                    ProductKind.TENSOR,
                    StorageClass.DEVICE_TENSOR,
                    output.dtype,
                    output.shape_bound,
                    PointRange(),
                )
                binding = arena.bind(
                    product,
                    BufferAllocation(product.buffer_id, offset, product.max_bytes),
                    device="cpu",
                    dtype=torch.float32,
                    shape=shape,
                )
                binding.tensor.fill_(request_id * 10 + op_id)
                bindings.append((binding, request_id * 10 + op_id))
                offset += ((product.max_bytes + 255) // 256) * 256
        for binding, expected in bindings:
            torch.testing.assert_close(
                binding.tensor, torch.full_like(binding.tensor, expected), rtol=0, atol=0
            )
    finally:
        for binding, _expected in bindings:
            arena.release(binding)
        arena.close()


def test_cuda_capacity_query_failure_is_not_an_empty_budget(monkeypatch) -> None:
    from uniserve_worker.bootstrap.capacity import device_total_bytes

    def unavailable(_device):
        raise RuntimeError("CUDA device is unavailable")

    monkeypatch.setattr(torch.cuda, "mem_get_info", unavailable)
    assert device_total_bytes("cpu") == 0
    with pytest.raises(RuntimeError, match="CUDA device is unavailable"):
        device_total_bytes("cuda:0")


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_product_capacity_accounts_for_remote_consumers_and_streamed_units(rank):
    from uniserve_worker.bootstrap.capacity import local_product_storage_bytes
    from uniserve_worker.execution.batch import DeviceDim, DType, ShapeBound, StaticDim, TensorSpec
    from uniserve_worker.models.video import MediaExecutionPlan, MediaPlanRepeat, MediaPlanStage
    from uniserve_worker.nn.mesh import DeviceMesh, EntryBindings
    from uniserve_worker.nn.parallel import EntryConfig

    entries = {
        "encode": EntryConfig((2,)),
        "predict": EntryConfig((1,)),
        "decode": EntryConfig((1, 3), distribution="temporal_units", units_per_rank=2),
        "assemble": EntryConfig((0,)),
    }
    bindings = EntryBindings(
        entries,
        {
            name: DeviceMesh(config.ranks, rank, config.parallel_config, torch.device("cpu"))
            for name, config in entries.items()
            if config.distribution is None and rank in config.ranks
        },
        Communicator((0, 1, 2, 3), rank),
    )
    outputs = {
        "encode": (TensorSpec("embedding", DType.F32, ShapeBound((StaticDim(128),))),),
        "predict": (TensorSpec("latents", DType.F32, ShapeBound((StaticDim(256),))),),
        "decode": (TensorSpec("frames", DType.F32, ShapeBound((DeviceDim(20), StaticDim(128)))),),
    }
    plan = MediaExecutionPlan(
        (
            MediaPlanStage("conditioning", OpCode.ENCODER_TEXT, "encode"),
            MediaPlanStage("denoise", OpCode.DIFFUSION_STEP, "predict", input_from="conditioning"),
            MediaPlanStage(
                "decode",
                OpCode.DIFFUSION_DECODE,
                "decode",
                input_from="denoise",
                repeat=MediaPlanRepeat.VIDEO_UNITS,
            ),
            MediaPlanStage(
                "write",
                OpCode.MEDIA_APPEND,
                "assemble",
                input_from="decode",
                repeat=MediaPlanRepeat.VIDEO_UNITS,
            ),
        )
    )
    # Two outstanding groups each contain four 512-byte units. The output
    # assembler imports both groups although it executes neither producer.
    expected = {0: 4096, 1: 512 + 1024 + 4096, 2: 512, 3: 1024 + 4096}
    assert (
        local_product_storage_bytes(outputs, bindings=bindings, plan=plan, max_unresolved_ops=2)
        == expected[rank]
    )
    # A horizon beyond the complete trajectory never reserves extra units.
    expected_full = {0: 10_240, 1: 512 + 1024 + 10_240, 2: 512, 3: 1024 + 10_240}
    assert (
        local_product_storage_bytes(outputs, bindings=bindings, plan=plan, max_unresolved_ops=8)
        == expected_full[rank]
    )
