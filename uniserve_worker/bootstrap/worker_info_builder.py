"""Build WorkerInfo from a loaded model and worker geometry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import cast

import torch

from uniserve_worker.config import WorkerConfig
from uniserve_worker.nn.mesh import Communicator
from uniserve_worker.nn.parallel import EntryConfig

from ..config import graph_memory_budget_bytes, graph_padding_block_count
from ..execution.batch import OpCode, WorkerEndpoint
from ..execution.input_buffers import InputGeometry
from ..foundation.errors import invalid_descriptor
from ..foundation.math import ceil_div
from ..models.generation import GenerationPipeline, LatentLayout
from ..models.runtime import ExecutionModel, active_latent_capacity_tokens
from ..nn.linear import LinearBase
from ..nn.quant.base import LinearMethod
from ..runtime.cache_transfer import cache_transfer_workspace_bytes
from ..runtime.req_to_token_pool import ReqToTokenPool
from ..runtime.runtime_states import RuntimeStates
from .capacity import (
    ArenaCapacity,
    derive_runtime_kv_capacity,
    device_total_bytes,
    latent_pool_capacity_bytes,
    local_product_storage_bytes,
    model_arena_capacity,
    operation_window,
    packed_input_geometry,
    request_tensor_window,
)
from .worker_info import (
    KvCacheConfig,
    KvGroup,
    KvGroupKind,
    WorkerInfo,
)

__all__ = ["WorkerLayout", "build_worker_info", "build_worker_layout", "configuration_identity"]


@dataclass(frozen=True, slots=True)
class WorkerLayout:
    """Worker-local model geometry paired with the public capacity report."""

    info: WorkerInfo
    arena: ArenaCapacity
    input_geometry: InputGeometry | None
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
    model: ExecutionModel,
    worker_config: WorkerConfig,
    layout: WorkerLayout,
    components: tuple[tuple[str, EntryConfig], ...],
    attention_identity: str | None,
) -> str:
    """Identify resolved params, numerical storage, operators, and shape bounds.

    Weight contents have their separate version. This identity describes the
    initialized worker configuration; graph and arena objects retain their own
    lifetimes and cannot be reused by a differently initialized worker.
    """

    numerical = {}
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, LinearMethod):
            numerical[name] = {
                "linear": f"{type(module).__module__}.{type(module).__qualname__}",
                "method": f"{type(method).__module__}.{type(method).__qualname__}",
                "input_scale_domain": method.input_scale_domain,
                "weight_scale_domain": method.weight_scale_domain,
                "weight_shard_axis": (
                    module.weight_shard_axis if isinstance(module, LinearBase) else None
                ),
            }
    layout_description = asdict(layout)
    layout_description["info"].pop("endpoint")
    execution_config = asdict(worker_config)
    # The observed grant changes with transient allocations and other processes.
    # Resolved capacities live in layout; free bytes are not an execution identity.
    execution_config.pop("pool_memory_bytes")
    encoded = json.dumps(
        {
            "worker_config": execution_config,
            "layout": layout_description,
            "components": {name: component.to_dict() for name, component in components},
            "entry_outputs": {
                name: [output.to_mapping() for output in outputs]
                for name, outputs in model.entry_outputs.items()
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


def build_worker_info(
    model: ExecutionModel,
    worker_config: WorkerConfig,
    *,
    model_name: str | None = None,
    weight_version: int = 0,
    queue_depth: int = 1,
    completion_payload_bytes: int = 1 << 20,
    endpoint: WorkerEndpoint | None = None,
) -> WorkerInfo:
    """Build the post-load worker capacity handshake."""

    return build_worker_layout(
        model,
        worker_config,
        model_name=model_name,
        weight_version=weight_version,
        queue_depth=queue_depth,
        completion_payload_bytes=completion_payload_bytes,
        endpoint=endpoint,
    ).info


def build_worker_layout(
    model: ExecutionModel,
    worker_config: WorkerConfig,
    *,
    model_name: str | None = None,
    weight_version: int = 0,
    queue_depth: int = 1,
    completion_payload_bytes: int = 1 << 20,
    endpoint: WorkerEndpoint | None = None,
    capacity_group: Communicator | None = None,
) -> WorkerLayout:
    """Build worker-local model geometry and its public capacity projection."""

    model_name = model.architecture if model_name is None else model_name
    endpoint = endpoint or WorkerEndpoint.local(rank=int(worker_config.rank))

    if bool(model.resource_geometry.request_tensors):
        return _request_tensor_worker_layout(
            model,
            worker_config,
            model_name=model_name,
            weight_version=weight_version,
            queue_depth=queue_depth,
            completion_payload_bytes=completion_payload_bytes,
            endpoint=endpoint,
        )

    resources = model.resource_geometry
    owns_kv = bool(resources.kv)
    bytes_per_token = _kv_bytes_per_token(model, worker_config) if owns_kv else 0
    flow = model.generation
    if flow is not None and not isinstance(flow, GenerationPipeline):
        raise invalid_descriptor("model generation behavior has an invalid type")
    requested_latent_units = (
        active_latent_capacity_tokens(
            int(flow.max_latent_tokens),
            worker_config.kv_token_capacity,
        )
        if flow is not None
        else 0
    )
    latent_page_units = int(worker_config.block_size) if flow is not None else 0
    num_latent_pages = (
        ceil_div(requested_latent_units, latent_page_units) + 1 if flow is not None else 0
    )
    latent_width = (
        int(flow.latent_channels) * int(flow.latent_patch_size) ** 2 if flow is not None else 0
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
    cache = model.cache_geometry if owns_kv else None
    max_vit_grid_tokens = int(model.max_vit_grid_tokens)
    hidden_elements = int(getattr(model, "hidden_size", 0))
    max_vision_feature_bytes = max_vit_grid_tokens * hidden_elements * model_dtype_bytes
    max_latent_feature_bytes = (
        0
        if flow is None
        else (
            int(flow.max_vae_grid_tokens)
            * int(flow.latent_channels)
            * int(flow.latent_patch_size)
            * int(flow.latent_patch_size)
            * model_dtype_bytes
        )
    )
    capacity = None
    unresolved_window = operation_window(int(queue_depth), int(worker_config.max_batch_operations))
    buffer_pool_bytes = (int(resources.encoder_cache_entries) + 1) * max(
        max_latent_feature_bytes, max_vision_feature_bytes
    )
    arena_args = dict(
        pipeline_depth=int(queue_depth),
        completion_payload_bytes=int(completion_payload_bytes),
        request_pool_size=int(worker_config.max_request_pool_size),
        num_latent_pages=num_latent_pages,
        latent_page_units=latent_page_units,
        latent_width=latent_width,
        max_latent_feature_bytes=max_latent_feature_bytes,
        max_vision_feature_bytes=max_vision_feature_bytes,
        bytes_per_token=bytes_per_token,
    )
    arena = model_arena_capacity(model, worker_config, num_blocks=0, **arena_args)
    input_geometry = packed_input_geometry(model, worker_config) if owns_kv else None
    devices = tuple(
        dict.fromkeys(
            (worker_config.device, worker_config.generation_device or worker_config.device)
        )
    )
    # Every device has its own persistent and product backing. Request tables
    # and continuation rows are shared across lanes on the primary device.
    fixed_bytes = dict.fromkeys(
        devices, buffer_pool_bytes + arena.device_product_bytes // len(devices)
    )
    fixed_bytes[worker_config.generation_device or worker_config.device] += latent_pool_bytes
    if owns_kv:
        assert cache is not None and input_geometry is not None
        input_bytes = sum(field.nbytes for field in input_geometry.tensor_schema().values())
        for device in devices:
            fixed_bytes[device] += input_bytes * max(1, len(worker_config.lanes))
        schemas = (
            ReqToTokenPool.tensor_schema(
                group_count=1,
                request_pool_size=worker_config.max_request_pool_size,
                max_blocks_per_request=input_geometry.max_blocks_per_row,
            ),
            RuntimeStates.tensor_schema(
                request_pool_size=worker_config.max_request_pool_size,
                vocab_size=model.vocab_size,
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
        if capacity_group is not None and capacity_group.world_size > 1:
            pages = torch.tensor(
                capacity.num_blocks, dtype=torch.int64, device=capacity_group.device
            )
            capacity_group.all_reduce_min(pages)
            blocks = int(pages.item())
            capacity = replace(
                capacity, num_blocks=blocks, token_capacity=blocks * capacity.block_size
            )
    if int(completion_payload_bytes) < 1:
        raise ValueError("completion payload capacity must be positive")
    supported_ops = tuple(code for code in OpCode if code in model.supported_work)
    info = WorkerInfo(
        model_name=model_name,
        weight_version=weight_version,
        endpoint=endpoint,
        device=str(worker_config.device),
        world_size=int(worker_config.world_size),
        supported_ops=supported_ops,
        queue_depth=int(queue_depth),
        max_batch_ops=int(worker_config.max_batch_operations),
        max_batch_tokens=int(worker_config.max_batch_tokens),
        request_slots=int(worker_config.max_request_pool_size),
        kv_cache=(
            KvCacheConfig(
                block_size=int(worker_config.block_size),
                num_blocks=int(capacity.num_blocks),
                num_layers=int(cache.num_layers),
                total_layers=cast(int, cache.total_layers),
                layer_offset=int(cache.layer_offset),
                num_kv_heads=int(cache.num_kv_heads),
                total_kv_heads=int(cache.total_kv_heads),
                kv_head_offset=int(cache.kv_head_offset),
                head_dim=int(cache.head_dim),
                bytes_per_token=bytes_per_token,
                groups=(
                    KvGroup(
                        num_blocks=int(capacity.num_blocks),
                        kind=KvGroupKind.FULL,
                        window=0,
                        sink=0,
                    ),
                ),
                dtype=_kv_dtype(model, worker_config),
            )
            if capacity is not None and cache is not None
            else None
        ),
        latent_page_units=latent_page_units,
        latent_pages=num_latent_pages,
        buffer_pool_bytes=buffer_pool_bytes,
        max_unresolved_ops=unresolved_window,
        media_plan=getattr(model, "media_plan", None),
    )
    arena = model_arena_capacity(
        model,
        worker_config,
        num_blocks=0 if capacity is None else capacity.num_blocks,
        **arena_args,
    )
    return WorkerLayout(
        info=info,
        arena=arena,
        input_geometry=input_geometry,
        fixed_device_bytes=tuple(fixed_bytes.items()),
        physical_buffer_pool_bytes=buffer_pool_bytes,
        latent_width=latent_width,
        latent_dtype=worker_config.model_dtype if flow is not None else "",
        latent_downsample=int(flow.latent_downsample) if flow is not None else 1,
        max_vae_grid_tokens=int(flow.max_vae_grid_tokens) if flow is not None else 0,
        max_vit_grid_tokens=max_vit_grid_tokens,
        max_latent_feature_bytes=max_latent_feature_bytes,
        max_vision_feature_bytes=max_vision_feature_bytes,
        commit_marker_tokens=(
            int(flow.commit_marker_tokens)
            if flow is not None and flow.latent_layout is LatentLayout.PATCH_TOKENS
            else 0
        ),
        gen_rope_advance=int(flow.rope_advance) if flow is not None else 2,
        max_cfg_branches=int(flow.max_cfg_branches) if flow is not None else 1,
        encoder_cache_entries=int(resources.encoder_cache_entries),
        incremental_kv_publication=owns_kv,
        model_dtype=worker_config.model_dtype,
    )


def _request_tensor_worker_layout(
    model: ExecutionModel,
    worker_config: WorkerConfig,
    *,
    model_name: str,
    weight_version: int,
    queue_depth: int,
    completion_payload_bytes: int,
    endpoint: WorkerEndpoint,
) -> WorkerLayout:
    """Describe request tensors, products, and persistent capacity."""

    if int(completion_payload_bytes) < 1:
        raise ValueError("completion payload capacity must be positive")
    if not model.resource_geometry.request_tensors:
        raise RuntimeError("request tensor worker info requires declared tensor storage")
    slots = int(worker_config.max_request_pool_size)
    depth = int(queue_depth)
    unresolved_window = request_tensor_window(depth, slots)
    max_operations = min(slots, int(worker_config.max_batch_operations))
    info = WorkerInfo(
        model_name=model_name,
        weight_version=weight_version,
        endpoint=endpoint,
        device=str(worker_config.device),
        world_size=int(worker_config.world_size),
        supported_ops=tuple(code for code in OpCode if code in model.supported_work),
        queue_depth=depth,
        max_batch_ops=max_operations,
        max_batch_tokens=max_operations,
        request_slots=slots,
        kv_cache=None,
        latent_page_units=0,
        latent_pages=0,
        buffer_pool_bytes=slots * model.product_storage_bytes,
        max_unresolved_ops=unresolved_window,
        media_plan=getattr(model, "media_plan", None),
    )
    return WorkerLayout(
        info=info,
        arena=model_arena_capacity(
            model,
            worker_config,
            pipeline_depth=depth,
            completion_payload_bytes=completion_payload_bytes,
            num_blocks=0,
            request_pool_size=slots,
            num_latent_pages=0,
            latent_page_units=0,
            latent_width=0,
            max_latent_feature_bytes=0,
            max_vision_feature_bytes=0,
            bytes_per_token=0,
        ),
        input_geometry=None,
        fixed_device_bytes=(),
        physical_buffer_pool_bytes=slots
        * local_product_storage_bytes(
            model.entry_outputs,
            bindings=model.bindings,
            plan=model.media_plan,
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


def _kv_dtype(model: ExecutionModel, worker_config: WorkerConfig) -> str:
    """Resolve the advertised KV storage dtype from model and worker_config policy."""

    override = worker_config.kv_cache_dtype
    if override is None:
        cache = model.cache_geometry
        return str(cache.store_dtype or cache.dtype).removeprefix("torch.")
    return str(override).removeprefix("torch.")


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


def _kv_bytes_per_token(model: ExecutionModel, worker_config: WorkerConfig) -> int:
    """Compute rank-local key-and-value bytes retained for one cached token."""

    width = {
        "float8_e4m3fn": 1,
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }.get(_kv_dtype(model, worker_config).lower())
    if width is None:
        raise ValueError(f"unsupported KV dtype {_kv_dtype(model, worker_config)!r}")
    cache = model.cache_geometry
    values = 2 * int(width) * int(cache.num_layers) * int(cache.num_kv_heads) * int(cache.head_dim)
    scales = ceil_div(8 * int(cache.num_layers), int(worker_config.block_size)) if width == 1 else 0
    return values + scales


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
