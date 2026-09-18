"""Build WorkerInfo from a loaded model and worker resource settings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn

from uniserve.distributed.mesh import Communicator
from uniserve.math import ceil_div
from uniserve.media import image
from uniserve.model import CausalLM, PatchEncoder, VideoPostprocessor
from uniserve.processing import ImageProcessor
from uniserve.quantization import QuantizedTensor
from uniserve.tensors import BufferConfig
from uniserve_worker.config import WorkerConfig
from uniserve_worker.protocol.call import CALL_KINDS, CallKind

from ..config import graph_memory_budget_bytes, graph_padding_block_count
from ..execution.input_buffers import InputBufferConfig
from ..execution.model_entry import ModelEntry
from ..execution.resources import media_state_buffers
from ..foundation.errors import unsupported_setup
from ..protocol.transfer import WorkerEndpoint
from ..runtime.block_tables import BlockTables
from ..runtime.cache_imports import cache_transfer_workspace_bytes
from ..runtime.decode_state import DecodeState
from ..runtime.results import resolve_outputs
from .cache import cache_info, resize_cache
from .capacity import (
    ArenaCapacity,
    active_latent_capacity_tokens,
    call_window,
    derive_runtime_kv_capacity,
    device_total_bytes,
    input_buffer_config,
    latent_pool_capacity_bytes,
    local_product_storage_bytes,
    model_arena_capacity,
    product_storage_bytes,
    request_tensor_window,
    vision_tokens,
)
from .components import media_components, supported_calls
from .config import ComponentConfig
from .inputs import capability, image_builder, media_builder
from .worker_info import EntryInfo, WorkerInfo

__all__ = [
    "WorkerLayout",
    "build_worker_layout",
    "configuration_identity",
]


@dataclass(frozen=True, slots=True)
class WorkerLayout:
    """Worker-local model dimensions paired with the public capacity report."""

    info: WorkerInfo
    arena: ArenaCapacity
    input_config: InputBufferConfig | None
    fixed_device_bytes: tuple[tuple[str, int], ...]
    physical_buffer_pool_bytes: int
    latent_width: int
    latent_dtype: str
    latent_downsample: int
    max_vae_grid_tokens: int
    max_vit_grid_tokens: int
    max_latent_feature_bytes: int
    max_vision_feature_bytes: int
    commit_marker_tokens: int
    gen_rope_advance: int
    max_cfg_branches: int
    encoder_cache_entries: int
    incremental_kv_publication: bool
    model_dtype: str


def configuration_identity(
    model: nn.Module,
    worker_config: WorkerConfig,
    layout: WorkerLayout,
    components: tuple[tuple[str, ComponentConfig], ...],
    attention_identity: str | None,
) -> str:
    """Identify resolved params, numerical storage, operators, and shape bounds.

    This describes resolved execution dimensions and numerical policy, not
    a hash of weight contents. Graph and arena objects retain their own
    lifetimes and cannot be reused by a differently initialized worker.
    """
    numerical = {
        name: asdict(value.quantizer)
        for name, value in model.named_parameters()
        if isinstance(value, QuantizedTensor)
    }

    layout_description = asdict(layout)
    layout_description["info"].pop("endpoint")
    layout_description["info"].pop("configuration_id")
    if layout.input_config is not None:
        layout_description["input_config"]["embedding_dtype"] = str(
            layout.input_config.embedding_dtype
        )

    # The observed grant changes with transient allocations and other
    # processes. Resolved capacities live in layout; free bytes are not an
    # execution identity.
    execution_config = asdict(worker_config)
    execution_config.pop("pool_memory_bytes")

    encoded = json.dumps(
        {
            "worker_config": execution_config,
            "layout": layout_description,
            "components": {
                name: component.to_dict() for name, component in components
            },
            "entry_outputs": {
                name: [output.to_mapping() for output in outputs]
                for name, outputs in resolve_outputs(
                    model, worker_config
                ).items()
            },
            "attention": attention_identity,
            "model": f"{type(model).__module__}.{type(model).__qualname__}",
            "numerical": numerical,
            "parameters": [
                (name, tuple(value.shape), str(value.dtype))
                for name, value in model.named_parameters()
            ],
            "buffers": [
                (name, tuple(value.shape), str(value.dtype))
                for name, value in model.named_buffers()
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(encoded.encode()).hexdigest()


def build_worker_layout(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    image_processor: ImageProcessor | None = None,
    model_name: str | None = None,
    queue_depth: int = 1,
    completion_payload_bytes: int = 1 << 20,
    endpoint: WorkerEndpoint | None = None,
    capacity_group: Communicator | None = None,
    allowed_work_variants: frozenset[CallKind] | None = None,
    transfer_backends: tuple[str, ...] = ("local",),
    components: tuple[tuple[str, ComponentConfig], ...] = (),
    attention_identity: str | None = None,
    bindings: Mapping[str, ModelEntry] | None = None,
    state_buffers: Mapping[str, BufferConfig] | None = None,
) -> WorkerLayout:
    """Resolve resource dimensions and the capacity report used by the worker.

    Model dimensions determine storage reservations. Admission additionally
    obeys the configured call set and every lane's bounds; these
    restrictions do not shrink the storage needed by warmup and graph
    capture.
    """
    if queue_depth <= 0:
        raise unsupported_setup("worker pipeline depth must be positive")

    if completion_payload_bytes <= 0:
        raise ValueError("completion payload capacity must be positive")

    supported_ops = supported_calls(model)
    if allowed_work_variants is not None:
        supported_ops = supported_ops & allowed_work_variants

    if not supported_ops:
        raise unsupported_setup(
            "worker model implements none of the requested work variants"
        )

    model_name = (
        f"{type(model).__module__}.{type(model).__qualname__}"
        if model_name is None
        else model_name
    )
    endpoint = endpoint or WorkerEndpoint.local(rank=int(worker_config.rank))

    if state_buffers is None:
        state_buffers = media_state_buffers(
            model, bindings or {}, worker_config
        )

    # Media workers hold fixed request tensors; token workers size paged KV,
    # latent pools, and text staging instead.
    if capability(model, VideoPostprocessor) is not None or state_buffers:
        layout = _request_tensor_worker_layout(
            model,
            worker_config,
            state_buffers=state_buffers,
            bindings=bindings,
            model_name=model_name,
            queue_depth=queue_depth,
            completion_payload_bytes=completion_payload_bytes,
            endpoint=endpoint,
        )
    else:
        layout = _token_worker_layout(
            model,
            worker_config,
            image_processor=image_processor,
            model_name=model_name,
            queue_depth=queue_depth,
            completion_payload_bytes=completion_payload_bytes,
            endpoint=endpoint,
            capacity_group=capacity_group,
        )

    # A shared admission limit must be safe on every eligible lane. Keep it
    # in the layout so runtime allocation and the IPC handshake read the
    # same value.
    max_calls = layout.info.max_batch_ops
    max_tokens = layout.info.max_batch_tokens

    for lane in worker_config.lanes:
        max_calls = min(max_calls, lane.max_batch_calls or max_calls)
        max_tokens = min(max_tokens, lane.max_batch_tokens or max_tokens)

    outputs = resolve_outputs(model, worker_config)
    info = replace(
        layout.info,
        supported_ops=tuple(
            code for code in CALL_KINDS if code in supported_ops
        ),
        transfer_backends=transfer_backends,
        max_batch_ops=max_calls,
        max_batch_tokens=max_tokens,
        encoder_cache_entries=layout.encoder_cache_entries,
        encoder_entry_bytes=max(
            layout.max_latent_feature_bytes, layout.max_vision_feature_bytes
        )
        if layout.encoder_cache_entries
        else 0,
        components=tuple(
            EntryInfo(name, entry, outputs.get(name, ()))
            for name, entry in components
        ),
    )
    layout = replace(layout, info=info)

    # Hash the final advertised limits and component bindings, excluding the
    # process incarnation and the identity field itself.
    identity = configuration_identity(
        model, worker_config, layout, components, attention_identity
    )

    return replace(layout, info=replace(info, configuration_id=identity))


def _token_worker_layout(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    image_processor: ImageProcessor | None,
    model_name: str,
    queue_depth: int,
    completion_payload_bytes: int,
    endpoint: WorkerEndpoint,
    capacity_group: Communicator | None,
) -> WorkerLayout:
    """Size token inputs, paged KV, and latent storage before admission."""
    encoder_cache_entries = (
        worker_config.encoder_cache_entries
        if image_processor is not None
        else 0
    )
    text = capability(model, CausalLM)
    owns_kv = text is not None
    cache = (
        None if text is None else cache_info(text, worker_config, num_blocks=1)
    )
    bytes_per_token = 0 if cache is None else cache.bytes_per_token

    flow = image_builder(model)
    requested_latent_units = (
        active_latent_capacity_tokens(
            flow.max_tokens,
            worker_config.kv_token_capacity,
        )
        if flow is not None
        else 0
    )
    latent_page_units = int(worker_config.block_size) if flow is not None else 0
    num_latent_pages = (
        ceil_div(requested_latent_units, latent_page_units) + 1
        if flow is not None
        else 0
    )
    latent_width = (
        flow.denoiser.latent_channels * flow.denoiser.patch_size**2
        if flow is not None
        else 0
    )
    model_dtype_bytes = _model_dtype_bytes(worker_config.model_dtype)
    latent_pool_bytes = (
        latent_pool_capacity_bytes(
            request_pool_size=int(worker_config.max_request_pool_size),
            num_pages=num_latent_pages,
            page_units=latent_page_units,
            latent_width=latent_width,
            dtype_bytes=model_dtype_bytes,
        )
        if flow is not None
        else 0
    )

    max_vit_grid_tokens = vision_tokens(model, image_processor)
    vision = capability(model, PatchEncoder)
    # Encoder representation remains meaningful on a worker without a text
    # decoder. Query the numerical output instead of borrowing language width.
    if vision is None or not max_vit_grid_tokens:
        max_vision_feature_bytes = 0
    else:
        stride = vision.patch_size * vision.downsample
        feature = vision.output_layout(image.Config(stride, stride))["features"]
        max_vision_feature_bytes = (
            max_vit_grid_tokens * feature.shape[1] * feature.dtype.itemsize
        )
    max_latent_feature_bytes = (
        0
        if flow is None
        else flow.max_tokens * latent_width * model_dtype_bytes
    )

    capacity = None
    unresolved_window = call_window(
        int(queue_depth), int(worker_config.max_batch_calls)
    )
    buffer_pool_bytes = (encoder_cache_entries + 1) * max(
        max_latent_feature_bytes, max_vision_feature_bytes
    )
    # Standalone encoders also publish persistent tensors. Their declared
    # products retain one bounded allocation per resident request.
    buffer_pool_bytes += (
        worker_config.max_request_pool_size
        * product_storage_bytes(resolve_outputs(model, worker_config))
    )

    arena_args = {
        "queue_depth": int(queue_depth),
        "completion_payload_bytes": int(completion_payload_bytes),
        "request_pool_size": int(worker_config.max_request_pool_size),
        "num_latent_pages": num_latent_pages,
        "latent_page_units": latent_page_units,
        "latent_width": latent_width,
        "max_latent_feature_bytes": max_latent_feature_bytes,
        "max_vision_feature_bytes": max_vision_feature_bytes,
        "bytes_per_token": bytes_per_token,
    }
    arena = model_arena_capacity(
        model,
        worker_config,
        num_blocks=0,
        bindings=None,
        state_buffers=None,
        **arena_args,
    )
    input_config = (
        input_buffer_config(model, worker_config, processor=image_processor)
        if owns_kv
        else None
    )

    devices = tuple(
        dict.fromkeys(
            (
                worker_config.device,
                worker_config.generation_device or worker_config.device,
            )
        )
    )
    # Every device has its own persistent and product backing. Request tables
    # and continuation rows are shared across lanes on the primary device.
    fixed_bytes = dict.fromkeys(
        devices, buffer_pool_bytes + arena.device_product_bytes // len(devices)
    )
    fixed_bytes[worker_config.generation_device or worker_config.device] += (
        latent_pool_bytes
    )

    if text is not None:
        assert cache is not None and input_config is not None
        input_bytes = sum(
            field.nbytes for field in input_config.buffers().values()
        )
        for device in devices:
            fixed_bytes[device] += input_bytes * max(
                1, len(worker_config.lanes)
            )

        schemas = (
            BlockTables.buffers(
                group_count=1,
                request_pool_size=worker_config.max_request_pool_size,
                max_blocks_per_request=input_config.max_blocks_per_row,
            ),
            DecodeState.buffers(
                request_pool_size=worker_config.max_request_pool_size,
                vocab_size=text.backbone.vocab_size,
                continuation_width=1,
                logits_dtype=getattr(torch, worker_config.model_dtype),
            ),
        )
        fixed_bytes[worker_config.device] += sum(
            field.nbytes for schema in schemas for field in schema.values()
        ) + cache_transfer_workspace_bytes(
            num_layers=int(cache.num_layers),
            page_size=int(worker_config.block_size),
            num_kv_heads=int(cache.num_kv_heads),
            head_dim=int(cache.head_dim),
            capacity=unresolved_window,
        )

        resident_copies, co_resident_blocks = _kv_residency_shape(
            worker_config,
            bytes_per_token=bytes_per_token,
            co_resident_bytes=fixed_bytes[worker_config.device],
        )
        capacity = derive_runtime_kv_capacity(
            block_size=int(worker_config.block_size),
            kv_token_capacity=worker_config.kv_token_capacity,
            bytes_per_token=bytes_per_token,
            device=worker_config.device,
            available_bytes=worker_config.pool_memory_bytes,
            resident_copies=resident_copies,
            co_resident_blocks=co_resident_blocks,
        )

        # Ranks sharing one logical pool must agree on its page count; the
        # minimum keeps every member within its own memory grant.
        if capacity_group is not None and capacity_group.size > 1:
            pages = torch.tensor(
                capacity.num_blocks,
                dtype=torch.int64,
                device=capacity_group.device,
            )
            capacity_group.all_reduce(pages, op="min")
            blocks = int(pages.item())
            capacity = replace(
                capacity,
                num_blocks=blocks,
                token_capacity=blocks * capacity.block_size,
            )

    supported_ops = tuple(
        code for code in CALL_KINDS if code in supported_calls(model)
    )
    info = WorkerInfo(
        model_name=model_name,
        endpoint=endpoint,
        device=str(worker_config.device),
        world_size=int(worker_config.world_size),
        supported_ops=supported_ops,
        queue_depth=int(queue_depth),
        max_batch_ops=int(worker_config.max_batch_calls),
        max_batch_tokens=int(worker_config.max_batch_tokens),
        request_slots=int(worker_config.max_request_pool_size),
        kv_cache=(
            resize_cache(cache, capacity.num_blocks)
            if capacity is not None and cache is not None
            else None
        ),
        latent_page_units=latent_page_units,
        latent_pages=num_latent_pages,
        buffer_pool_bytes=buffer_pool_bytes,
        max_unresolved_ops=unresolved_window,
        pipeline_components=dict(media_components(model)),
        num_inference_steps=0,
        host_lane_capacity=1,
    )
    arena = model_arena_capacity(
        model,
        worker_config,
        bindings=None,
        state_buffers=None,
        num_blocks=0 if capacity is None else capacity.num_blocks,
        **arena_args,
    )
    return WorkerLayout(
        # The rank's host executor owns the host lane, so its advertised
        # capacity is the one the arena reserved.
        info=replace(info, host_lane_capacity=int(arena.host_lane_inflight)),
        arena=arena,
        input_config=input_config,
        fixed_device_bytes=tuple(fixed_bytes.items()),
        physical_buffer_pool_bytes=buffer_pool_bytes,
        latent_width=latent_width,
        latent_dtype=worker_config.model_dtype if flow is not None else "",
        latent_downsample=flow.denoiser.downsample if flow is not None else 1,
        max_vae_grid_tokens=flow.max_tokens + flow.framing
        if flow is not None
        else 0,
        max_vit_grid_tokens=max_vit_grid_tokens,
        max_latent_feature_bytes=max_latent_feature_bytes,
        max_vision_feature_bytes=max_vision_feature_bytes,
        commit_marker_tokens=0 if flow is None else flow.framing,
        gen_rope_advance=2 if flow is None else flow.rope_advance,
        max_cfg_branches=1 if flow is None else 3,
        encoder_cache_entries=encoder_cache_entries,
        incremental_kv_publication=owns_kv,
        model_dtype=worker_config.model_dtype,
    )


def _request_tensor_worker_layout(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    state_buffers: Mapping[str, BufferConfig],
    model_name: str,
    queue_depth: int,
    completion_payload_bytes: int,
    endpoint: WorkerEndpoint,
    bindings: Mapping[str, ModelEntry] | None,
) -> WorkerLayout:
    """Describe request tensors, products, and persistent capacity."""
    slots = int(worker_config.max_request_pool_size)
    depth = int(queue_depth)
    unresolved_window = request_tensor_window(depth, slots)
    max_calls = min(slots, int(worker_config.max_batch_calls))

    info = WorkerInfo(
        model_name=model_name,
        endpoint=endpoint,
        device=str(worker_config.device),
        world_size=int(worker_config.world_size),
        supported_ops=tuple(
            code for code in CALL_KINDS if code in supported_calls(model)
        ),
        queue_depth=depth,
        max_batch_ops=max_calls,
        max_batch_tokens=max_calls,
        request_slots=slots,
        kv_cache=None,
        latent_page_units=0,
        latent_pages=0,
        buffer_pool_bytes=slots
        * product_storage_bytes(resolve_outputs(model, worker_config)),
        max_unresolved_ops=unresolved_window,
        pipeline_components=dict(media_components(model)),
        num_inference_steps=media_builder(model, worker_config).num_steps,
        host_lane_capacity=1,
    )
    arena = model_arena_capacity(
        model,
        worker_config,
        bindings=bindings,
        state_buffers=state_buffers,
        queue_depth=depth,
        completion_payload_bytes=completion_payload_bytes,
        num_blocks=0,
        request_pool_size=slots,
        num_latent_pages=0,
        latent_page_units=0,
        latent_width=0,
        max_latent_feature_bytes=0,
        max_vision_feature_bytes=0,
        bytes_per_token=0,
    )

    return WorkerLayout(
        # The rank's host executor owns the host lane, so its advertised
        # capacity is the one the arena reserved.
        info=replace(info, host_lane_capacity=int(arena.host_lane_inflight)),
        arena=arena,
        input_config=None,
        fixed_device_bytes=(),
        physical_buffer_pool_bytes=slots
        * local_product_storage_bytes(
            resolve_outputs(model, worker_config),
            bindings=bindings or {},
            pipeline_components=media_components(model),
            max_unresolved_ops=unresolved_window,
        ),
        latent_width=1,
        latent_dtype="float32",
        latent_downsample=1,
        max_vae_grid_tokens=0,
        max_vit_grid_tokens=0,
        max_latent_feature_bytes=0,
        max_vision_feature_bytes=0,
        commit_marker_tokens=0,
        gen_rope_advance=1,
        max_cfg_branches=1,
        encoder_cache_entries=0,
        incremental_kv_publication=False,
        model_dtype="bfloat16",
    )


def _model_dtype_bytes(dtype: str) -> int:
    """Return the byte width of a supported model activation dtype."""
    width = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }.get(str(dtype).removeprefix("torch.").lower())
    if width is None:
        raise ValueError(f"unsupported model dtype {dtype!r}")
    return width


def _kv_residency_shape(
    worker_config: WorkerConfig,
    *,
    bytes_per_token: int,
    co_resident_bytes: int,
) -> tuple[int, int]:
    """Describe the single KV pool and co-resident fixed allocations."""
    block_size = int(worker_config.block_size)
    padding_blocks = graph_padding_block_count(block_size)
    # Captured executables are held for the worker's lifetime, so they occupy
    # the same static budget as the KV pools and are reserved before the
    # request pool is sized.
    graph_blocks = ceil_div(
        graph_memory_budget_bytes(device_total_bytes(worker_config.device)),
        block_size * max(1, int(bytes_per_token)),
    )

    fixed_owner_blocks = ceil_div(
        max(0, int(co_resident_bytes)),
        block_size * max(1, int(bytes_per_token)),
    )

    return 1, padding_blocks + graph_blocks + fixed_owner_blocks
