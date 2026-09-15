"""Model execution and worker resource behavior."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.worker_config import stub_worker_config
from uniserve.distributed.mesh import Communicator
from uniserve.math import ceil_div
from uniserve.tensors import BufferConfig
from uniserve_models.stub import Model, image_processor
from uniserve_worker.bootstrap.capacity import (
    latent_trajectory_bytes,
    model_arena_capacity,
    request_tensor_window,
    tensor_slot_capacity,
)
from uniserve_worker.bootstrap.config import ComponentConfig
from uniserve_worker.bootstrap.worker_info_builder import build_worker_layout
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.protocol.operation import ForwardMode, PipelineStage

TEST_MODEL = Model()
TEST_WORKER_CONFIG = stub_worker_config(64, max_batch_tokens=8192)

pytestmark = pytest.mark.unit


def test_declared_scratch_borrows_compact_views():
    from uniserve.runtime.tensor_buffers import TensorBuffers

    storage = TensorBuffers.allocate(
        {"rows": BufferConfig((2, 3), torch.float32, capacity_shape=(4, 6))}, device="cpu"
    )
    storage.view({"rows": BufferConfig((4, 6), torch.float32)})["rows"].copy_(
        torch.arange(24).reshape(4, 6)
    )
    views = storage.view({"rows": BufferConfig((2, 3), torch.float32)})
    torch.testing.assert_close(views["rows"], torch.arange(6).float().reshape(2, 3))
    views["rows"].add_(10)
    expected = torch.arange(24).float()
    expected[:6] += 10
    torch.testing.assert_close(
        storage.view({"rows": BufferConfig((4, 6), torch.float32)})["rows"], expected.reshape(4, 6)
    )
    larger = storage.view({"rows": BufferConfig((4, 6), torch.float32)})
    torch.testing.assert_close(larger["rows"], expected.reshape(4, 6))
    with pytest.raises(ValueError, match="capacity"):
        storage.view({"rows": BufferConfig((5, 6), torch.float32)})
    incompatible = TensorBuffers.from_tensors({"rows": torch.empty((4, 6), dtype=torch.int32)})
    with pytest.raises(ValueError, match="dtype"):
        incompatible.view({"rows": BufferConfig((2, 3), torch.float32)})


def test_tensor_capacity_covers_packed_selections_that_move_between_ranks():
    from uniserve.runtime.tensor_buffers import TensorBuffers

    def fields(prompt):
        # Six feature rows follow the prompt in a two-way sequence partition.
        # Changing the prompt moves the feature boundary across rank zero.
        rows = prompt + 6
        count = max(0, rows // 2 - prompt)
        return {
            "features": BufferConfig(
                (count, 2), torch.float32, capacity_shape=(min(6, rows // 2), 2)
            )
        }

    storage = TensorBuffers.allocate(fields(16), device="cpu")
    for prompt, expected in ((16, []), (0, [[0, 1], [2, 3], [4, 5]]), (2, [[0, 1], [2, 3]])):
        features = storage.view(fields(prompt))["features"]
        features.copy_(torch.arange(features.numel()).reshape_as(features))
        torch.testing.assert_close(
            features, torch.tensor(expected, dtype=torch.float32).reshape(-1, 2)
        )


def test_closed_request_storage_rejects_admission_and_borrowing():
    from tests.python.fixtures.depth_one import ar_params
    from uniserve_worker.runtime.request import RequestPool

    admission = replace(ar_params(71, block_ids=(0,)), request_pool_idx=1)
    pool = RequestPool(2, state_buffers={"state": BufferConfig((4,), torch.float32)}, device="cpu")
    pool.start(admission)
    pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.tensors(admission.request_pool_idx)
    with pytest.raises(RuntimeError, match="closed"):
        pool.start(admission)


def test_request_capacity_charges_only_device_storage_against_device_budget():
    schema = {
        "state": BufferConfig((128,), torch.float32),
        "initial_values": BufferConfig((1024,), torch.float32, host=True),
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
    schema = {"state": BufferConfig((128,), torch.float32)}
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


def test_stateless_ranks_still_budget_request_products_and_arenas():
    assert (
        tensor_slot_capacity(
            {},
            Communicator(),
            maximum=8,
            minimum=2,
            available_bytes=3072,
            auxiliary_bytes=lambda slots: slots * 1024,
        )
        == 3
    )


def test_worker_info_projects_model_behavior_and_resource_geometry():
    layout = build_worker_layout(
        TEST_MODEL,
        TEST_WORKER_CONFIG,
        image_processor=image_processor(),
        model_name="test-model",
    )
    info = layout.info

    assert ForwardMode.PREFILL in info.supported_ops
    assert PipelineStage.DENOISING in info.supported_ops
    assert layout.max_vision_feature_bytes == ((512 // 16) ** 2 * 4 * 2)
    assert info.kv_cache is not None
    assert info.kv_cache.num_layers == len(TEST_MODEL.cache_config.layers)
    assert info.model_name == "test-model"


def test_latent_capacity_rounds_to_complete_scheduler_pages() -> None:
    worker_config = replace(
        TEST_WORKER_CONFIG,
        kv_token_capacity=1024 + 1,
    )

    layout = build_worker_layout(TEST_MODEL, worker_config, image_processor=image_processor())

    expected_pages = ceil_div(
        1024 + 1,
        int(worker_config.block_size),
    )
    assert layout.info.latent_pages == expected_pages + 1
    assert layout.info.latent_capacity_units == expected_pages * int(worker_config.block_size)


def test_persistent_buffer_capacity_includes_active_encoder_output() -> None:
    layout = build_worker_layout(TEST_MODEL, TEST_WORKER_CONFIG, image_processor=image_processor())
    feature_bytes = max(
        layout.max_latent_feature_bytes,
        layout.max_vision_feature_bytes,
    )
    assert (
        layout.info.buffer_pool_bytes
        == (TEST_WORKER_CONFIG.encoder_cache_entries + 1) * feature_bytes
    )


def test_transfer_capacity_covers_one_maximum_float32_trajectory_per_ticket() -> None:
    worker_config = replace(TEST_WORKER_CONFIG, model_dtype="float32")
    layout = build_worker_layout(TEST_MODEL, worker_config, image_processor=image_processor())
    assert layout.max_latent_feature_bytes == latent_trajectory_bytes(
        1024,
        3 * 16**2,
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
    expected = latent_trajectory_bytes(1024, 1024, 4)
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


@pytest.mark.parametrize(
    "dtype,axes,message",
    [(torch.float64, (), "dtype"), (torch.float32, (0, 1), "dynamic axes")],
)
def test_result_publication_rejects_unrepresentable_numerical_outputs(dtype, axes, message):
    from tests.python.fixtures.encoding import Model as EncodedModel
    from uniserve.model import TextEncoder
    from uniserve.tensors import OutputLayout
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.runtime.results import resolve_outputs

    class FeatureEncoder(TextEncoder):
        def output_layout(self, num_tokens):
            return {
                "conditioning": OutputLayout(
                    (2, 3), dtype, (slice(0, 2), slice(0, 3)), variable_axes=axes
                )
            }

    model = EncodedModel()
    model.text_encoder = FeatureEncoder(model.text_encoder.network, (0,))
    with pytest.raises(ValueError, match=message):
        resolve_outputs(model, WorkerConfig())


def test_worker_reserves_declared_tensor_results_for_every_request() -> None:
    from tests.python.fixtures.encoding import Config
    from tests.python.fixtures.encoding import Model as EncodedModel
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.protocol.batch import BufferAllocation
    from uniserve_worker.protocol.identity import ComputationId, RequestKey
    from uniserve_worker.protocol.tensor import StaticDim, TensorRef
    from uniserve_worker.runtime.buffer_pool import BufferPool

    model = EncodedModel(Config(hidden_size=7))
    config = WorkerConfig(device="cpu", max_sequence_tokens=3, max_request_pool_size=2)
    info = build_worker_layout(
        model,
        config,
        queue_depth=8,
        completion_payload_bytes=1024,
        components=(("text_encoder", ComponentConfig((0,))),),
    ).info
    arena = BufferPool(byte_capacity=info.buffer_pool_bytes, devices=("cpu",))
    bindings = []
    offset = 0
    try:
        for request_id in range(1, info.request_slots + 1):
            for op_id, entry in enumerate(info.components, start=1):
                output = entry.outputs[0]
                shape = tuple(
                    dim.extent if isinstance(dim, StaticDim) else dim.bound
                    for dim in output.shape_bound.dims
                )
                product = TensorRef(
                    RequestKey(1, request_id, 1),
                    ComputationId(op_id, 0),
                    0,
                    1,
                    output.dtype,
                    output.shape_bound,
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
@pytest.mark.parametrize(
    ("streamed_width", "window_bytes", "full_bytes", "import_bytes"),
    [(128, 4096, 10_240, 10_240), (130, 4608, 11_520, 15_360)],
)
@pytest.mark.parametrize(
    "worker_entries",
    [
        ("encode", "predict", "decode", "assemble"),
        ("encode",),
        ("predict",),
        ("decode",),
        ("assemble",),
        ("decode", "assemble"),
    ],
)
def test_product_capacity_accounts_for_remote_consumers_and_streamed_units(
    rank, worker_entries, streamed_width, window_bytes, full_bytes, import_bytes
):
    from uniserve.distributed.mesh import DeviceMesh
    from uniserve_worker.bootstrap.capacity import local_product_storage_bytes
    from uniserve_worker.bootstrap.config import ComponentConfig
    from uniserve_worker.execution.model_entry import ModelEntry
    from uniserve_worker.protocol.operation import PipelineStage
    from uniserve_worker.protocol.tensor import (
        DeviceDim,
        DType,
        OutputInfo,
        ShapeBound,
        StaticDim,
    )

    entries = {
        "encode": ComponentConfig((2,)),
        "predict": ComponentConfig((1,)),
        "decode": ComponentConfig((1, 3), distribution="temporal_units", units_per_rank=2),
        "assemble": ComponentConfig((0,)),
    }
    group = Communicator((0, 1, 2, 3), rank)
    bindings = {
        name: ModelEntry(
            name,
            config,
            group,
            DeviceMesh(
                ranks=config.ranks,
                rank=rank,
                shape=tuple(size for _, size in config.parallel_config.dimensions),
                axes=tuple(axis for axis, _ in config.parallel_config.dimensions),
            )
            if config.distribution is None and rank in config.ranks
            else None,
            group.device,
        )
        for name, config in entries.items()
        if name in worker_entries
    }
    outputs = {
        "encode": (OutputInfo("embedding", DType.F32, ShapeBound((StaticDim(128),))),),
        "predict": (OutputInfo("latents", DType.F32, ShapeBound((StaticDim(256),))),),
        "decode": (
            OutputInfo("frames", DType.F32, ShapeBound((DeviceDim(20), StaticDim(streamed_width)))),
        ),
    }
    components = {
        PipelineStage.TEXT_ENCODING: "encode",
        PipelineStage.LATENT_PREPARATION: "predict",
        PipelineStage.DENOISING: "predict",
        PipelineStage.VIDEO_DECODING: "decode",
        PipelineStage.VIDEO_ENCODING: "assemble",
    }
    # Two outstanding groups each contain four units. Every allocation is
    # aligned to 256 bytes, including imported units from a remote worker.
    expected = {
        ("encode", "predict", "decode", "assemble"): {
            0: window_bytes,
            1: 1536 + window_bytes,
            2: 512,
            3: 1024 + window_bytes,
        },
        ("encode",): {2: 512},
        ("predict",): {1: 1536},
        ("decode",): {1: 1024 + window_bytes, 3: 1024 + window_bytes},
        # An assembler in another worker may import a group spanning every
        # declared unit; the producer's local grouping cannot bound it.
        ("assemble",): {0: import_bytes},
        ("decode", "assemble"): {
            0: window_bytes,
            1: 1024 + window_bytes,
            3: 1024 + window_bytes,
        },
    }[worker_entries]
    assert local_product_storage_bytes(
        outputs, bindings=bindings, pipeline_components=components, max_unresolved_ops=2
    ) == expected.get(rank, 0)
    # A horizon beyond the complete trajectory never reserves extra units.
    expected_full = {
        ("encode", "predict", "decode", "assemble"): {
            0: full_bytes,
            1: 1536 + full_bytes,
            2: 512,
            3: 1024 + full_bytes,
        },
        ("encode",): {2: 512},
        ("predict",): {1: 1536},
        ("decode",): {1: 1024 + full_bytes, 3: 1024 + full_bytes},
        ("assemble",): {0: import_bytes},
        ("decode", "assemble"): {0: full_bytes, 1: 1024 + full_bytes, 3: 1024 + full_bytes},
    }[worker_entries]
    assert local_product_storage_bytes(
        outputs, bindings=bindings, pipeline_components=components, max_unresolved_ops=8
    ) == expected_full.get(rank, 0)


@pytest.mark.parametrize("start,stop", [(2, 5), (5, 5)])
def test_tensor_output_preserves_global_slice_and_borrows_storage(start, stop):
    from uniserve.tensors import OutputLayout, TensorOutput

    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    result = TensorOutput(
        source[start:stop],
        OutputLayout((6, 4), torch.float32, (slice(start, stop), slice(0, 4))),
    )
    assert result.layout.local_slice == (slice(start, stop), slice(0, 4))
    source.add_(1)
    torch.testing.assert_close(result.tensor, source[start:stop])
    with pytest.raises(ValueError, match="shape"):
        TensorOutput(source, result.layout)
    with pytest.raises(ValueError, match="dtype"):
        TensorOutput(source[start:stop].double(), result.layout)


@pytest.mark.parametrize(
    "region",
    [
        (slice(None, 2), slice(0, 4)),
        (slice(-1, 2), slice(0, 4)),
        (slice(3, 2), slice(0, 4)),
        (slice(0, 2, 2), slice(0, 4)),
        (slice(0, 7), slice(0, 4)),
        (slice(0, 6),),
    ],
)
def test_output_layout_rejects_ambiguous_or_out_of_bounds_slices(region):
    from uniserve.tensors import OutputLayout

    with pytest.raises(ValueError):
        OutputLayout((6, 4), torch.float32, region)
