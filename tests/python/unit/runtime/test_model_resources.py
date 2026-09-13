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
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.foundation.math import ceil_div
from uniserve_worker.nn.mesh import Communicator
from uniserve_worker.nn.parallel import ComponentConfig
from uniserve_worker.protocol.batch import ForwardMode, PipelineStage
from uniserve_worker.runtime.tensor_buffers import TensorSchema

pytestmark = pytest.mark.unit


def test_attention_output_bindings_isolate_callers_and_restore_after_errors():
    from uniserve_worker.nn.mesh import DeviceMesh
    from uniserve_worker.nn.parallel_attention import ParallelAttention, output_scope
    from uniserve_worker.runtime.attention_storage import allocate_output_storage

    attention = ParallelAttention(mesh=DeviceMesh.trivial())
    callers = tuple(
        allocate_output_storage((attention,), rows=8, heads=2, head_dim=4, dtype=torch.float32)
        for _ in range(2)
    )
    query = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)

    def publish(value):
        destinations = attention.output_views(value)
        destinations[0].copy_(value)
        return attention.finish_output(destinations)

    with output_scope(callers[0].views):
        first = publish(query)
        with pytest.raises(ValueError, match="caller exit"):
            with output_scope(callers[1].views):
                second = publish(query + 7)
                raise ValueError("caller exit")
        torch.testing.assert_close(first, query)
        restored = publish(query - 2)
        torch.testing.assert_close(restored, query - 2)
        torch.testing.assert_close(second, query + 7)
        for invalid in (torch.zeros(9, 2, 4), query.double()):
            with pytest.raises(ValueError, match="bound tensor storage"):
                publish(invalid)
            torch.testing.assert_close(restored, query - 2)

    with pytest.raises(RuntimeError, match="bound output buffers"):
        publish(query)


def test_call_state_borrows_its_fields_and_stages_zero_padded_values():
    from uniserve_worker.modeling.components import Call
    from uniserve_worker.modeling.geometry import MediaShape
    from uniserve_worker.modeling.model import Model
    from uniserve_worker.modeling.resources import TensorNeeds
    from uniserve_worker.modeling.resources import TensorSchema as NumericalTensor
    from uniserve_worker.runtime.tensor_buffers import TensorBuffers
    from uniserve_worker.runtime.tensors import bind_state, stage_tensor

    class ConditionedModel(Model):
        def tensor_specs(self, call, shape):
            return TensorNeeds(
                state={"condition": NumericalTensor((shape.height, shape.width), torch.float32)}
            )

    backing = torch.full((4, 3), -1.0)
    overlap = torch.full((2,), 7.0)
    storage = TensorBuffers({"condition": backing, "overlap": overlap})
    state = bind_state(ConditionedModel(), Call.DIFFUSION, MediaShape(3, 3), storage)
    source = torch.arange(6).float().reshape(2, 3)
    stage_tensor(source, state["condition"])
    expected = torch.cat((source, torch.zeros(1, 3), torch.full((1, 3), -1.0)))
    torch.testing.assert_close(backing, expected)
    torch.testing.assert_close(source, torch.arange(6).float().reshape(2, 3))
    torch.testing.assert_close(overlap, torch.full((2,), 7.0))
    # Invalid production values cannot partially overwrite existing request state.
    for invalid in (torch.ones(4, 3), torch.ones(3, 3, dtype=torch.float64), torch.ones(9)):
        with pytest.raises(ValueError, match="representation"):
            stage_tensor(invalid, state["condition"])
        torch.testing.assert_close(backing, expected)


def test_declared_scratch_borrows_compact_views():
    from uniserve_worker.modeling.components import Call
    from uniserve_worker.modeling.geometry import MediaShape
    from uniserve_worker.modeling.model import Model
    from uniserve_worker.modeling.resources import TensorNeeds
    from uniserve_worker.modeling.resources import TensorSchema as NumericalTensor
    from uniserve_worker.runtime.tensor_buffers import TensorBuffers
    from uniserve_worker.runtime.tensors import bind_scratch

    class Projection(Model):
        def tensor_specs(self, call, shape):
            return TensorNeeds(
                scratch={"rows": NumericalTensor((shape.height, shape.width), torch.float32)}
            )

    storage = TensorBuffers.allocate(
        {"rows": TensorSchema((4, 6), torch.float32)},
        "cpu",
    )
    storage.capacity["rows"].copy_(torch.arange(24).reshape(4, 6))
    model = Projection()
    views = bind_scratch(model, Call.DIFFUSION, MediaShape(2, 3), storage)
    torch.testing.assert_close(views["rows"], torch.arange(6).float().reshape(2, 3))
    views["rows"].add_(10)
    expected = torch.arange(24).float()
    expected[:6] += 10
    torch.testing.assert_close(storage.capacity["rows"], expected.reshape(4, 6))
    larger = bind_scratch(model, Call.DIFFUSION, MediaShape(4, 6), storage)
    torch.testing.assert_close(larger["rows"], expected.reshape(4, 6))
    with pytest.raises(ValueError, match="capacity"):
        bind_scratch(model, Call.DIFFUSION, MediaShape(5, 6), storage)
    incompatible = TensorBuffers({"rows": torch.empty((4, 6), dtype=torch.int32)})
    with pytest.raises(ValueError, match="dtype"):
        bind_scratch(model, Call.DIFFUSION, MediaShape(2, 3), incompatible)


@pytest.mark.parametrize("kind", ["state", "scratch"])
def test_tensor_capacity_covers_packed_selections_that_move_between_ranks(kind):
    from uniserve_worker.modeling.components import Call
    from uniserve_worker.modeling.geometry import MediaShape
    from uniserve_worker.modeling.model import Model
    from uniserve_worker.modeling.resources import PackedAxis, TensorNeeds
    from uniserve_worker.modeling.resources import TensorSchema as NumericalTensor
    from uniserve_worker.runtime.tensor_buffers import TensorBuffers
    from uniserve_worker.runtime.tensors import bind_scratch, bind_state, resolve_resources

    class PackedFeatures(Model):
        def tensor_specs(self, call, shape):
            # Six feature rows follow a variable prompt in a two-way sequence
            # partition. This caller owns the first half of that sequence.
            rows = shape.prompt_tokens + 6
            count = max(0, rows // 2 - shape.prompt_tokens)
            return TensorNeeds(
                **{
                    kind: {
                        "features": NumericalTensor(
                            (count, 2), torch.float32, partition=PackedAxis(0, 6, rows, 2)
                        )
                    }
                }
            )

    model = PackedFeatures()
    maximum = MediaShape(1, 1, prompt_tokens=16)
    resources = resolve_resources(model, ((Call.DIFFUSION, maximum),))
    storage = TensorBuffers.allocate(getattr(resources, kind), "cpu")
    bind = bind_state if kind == "state" else bind_scratch
    for prompt, expected in ((16, []), (0, [[0, 1], [2, 3], [4, 5]]), (2, [[0, 1], [2, 3]])):
        shape = MediaShape(1, 1, prompt_tokens=prompt)
        features = bind(model, Call.DIFFUSION, shape, storage)["features"]
        features.copy_(torch.arange(features.numel()).reshape_as(features))
        torch.testing.assert_close(
            features, torch.tensor(expected, dtype=torch.float32).reshape(-1, 2)
        )


def test_closed_request_storage_rejects_admission_and_borrowing():
    from tests.python.fixtures.depth_one import ar_params
    from uniserve_worker.runtime.request import RequestPool

    admission = replace(ar_params(71, block_ids=(0,)), request_pool_idx=1)
    pool = RequestPool(2, tensor_schema={"state": TensorSchema((4,), torch.float32)}, device="cpu")
    pool.start(admission)
    pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.tensors(admission.request_pool_idx)
    with pytest.raises(RuntimeError, match="closed"):
        pool.start(admission)


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
        model_name="test-model",
    )
    info = layout.info

    assert ForwardMode.PREFILL in info.supported_ops
    assert PipelineStage.DENOISING in info.supported_ops
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
        == (TEST_WORKER_CONFIG.encoder_cache_entries + 1) * feature_bytes
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


@pytest.mark.parametrize(
    "dtype,axes,message",
    [(torch.float64, (), "dtype"), (torch.float32, (0, 1), "dynamic axes")],
)
def test_result_publication_rejects_unrepresentable_numerical_outputs(dtype, axes, message):
    from uniserve_worker.modeling.components import Call
    from uniserve_worker.modeling.geometry import MediaShape
    from uniserve_worker.modeling.model import Model
    from uniserve_worker.modeling.resources import TensorNeeds
    from uniserve_worker.modeling.resources import TensorSchema as NumericalTensor
    from uniserve_worker.runtime.results import resolve_outputs

    class Features(Model):
        output_shapes = {"encoder": (Call.ENCODE_VISION, MediaShape(2, 3))}

        def tensor_specs(self, call, shape):
            return TensorNeeds(
                outputs={"features": NumericalTensor((2, 3), dtype, variable_axes=axes)}
            )

    with pytest.raises(ValueError, match=message):
        resolve_outputs(Features())


def test_worker_reserves_declared_tensor_results_for_every_request() -> None:
    import torch

    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.modeling.components import Call, CallSpec, ComponentSpec
    from uniserve_worker.modeling.geometry import MediaShape, TextShape
    from uniserve_worker.modeling.model import (
        Model,
    )
    from uniserve_worker.modeling.resources import TensorNeeds
    from uniserve_worker.modeling.resources import TensorSchema as NumericalTensor
    from uniserve_worker.protocol.batch import (
        BufferAllocation,
        RequestKey,
        StaticDim,
        TensorRef,
    )
    from uniserve_worker.runtime.buffer_pool import BufferPool
    from uniserve_worker.runtime.tensor_buffers import TensorSchema

    class TensorModel(Model):
        output_shapes = {
            "text_encoder": (Call.ENCODE_TEXT, TextShape(1)),
            "denoiser": (Call.DIFFUSION, MediaShape(1, 1)),
        }

        def tensor_specs(self, call, shape):
            return TensorNeeds(
                outputs=(
                    {"conditioning": NumericalTensor((3, 7), torch.float32, variable_axes=(0,))}
                    if call is Call.ENCODE_TEXT
                    else {"latent": NumericalTensor((5, 11), torch.float32)}
                )
            )

        @classmethod
        def components(cls, config):
            return (
                ComponentSpec("text_encoder", (CallSpec(Call.ENCODE_TEXT),)),
                ComponentSpec("denoiser", (CallSpec(Call.DIFFUSION),)),
            )

    model = TensorModel()
    model.architecture = "TensorEntryModel"
    config = WorkerConfig(device="cpu", max_request_pool_size=2)
    info = build_worker_layout(
        model,
        config,
        queue_depth=8,
        completion_payload_bytes=1024,
        state_schema={"state": TensorSchema((4,), torch.float32)},
        components=(
            ("text_encoder", ComponentConfig((0,))),
            ("denoiser", ComponentConfig((0,))),
        ),
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
                    op_id,
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
    from uniserve_worker.bootstrap.capacity import local_product_storage_bytes
    from uniserve_worker.execution.model_entry import ModelEntry
    from uniserve_worker.nn.mesh import DeviceMesh
    from uniserve_worker.nn.parallel import ComponentConfig
    from uniserve_worker.protocol.batch import (
        DeviceDim,
        DType,
        PipelineStage,
        ShapeBound,
        StaticDim,
        TensorSpec,
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
            DeviceMesh(config.ranks, rank, config.parallel_config, group.device)
            if config.distribution is None and rank in config.ranks
            else None,
            group.device,
        )
        for name, config in entries.items()
        if name in worker_entries
    }
    outputs = {
        "encode": (TensorSpec("embedding", DType.F32, ShapeBound((StaticDim(128),))),),
        "predict": (TensorSpec("latents", DType.F32, ShapeBound((StaticDim(256),))),),
        "decode": (
            TensorSpec("frames", DType.F32, ShapeBound((DeviceDim(20), StaticDim(streamed_width)))),
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
