"""Build WorkerInfo from a loaded model and worker geometry."""

from __future__ import annotations

from dataclasses import dataclass

from ..execution.batch import logical_op_kinds
from ..foundation.errors import invalid_descriptor
from ..foundation.math import ceil_div
from ..models.generation import GenerationPipeline, LatentLayout
from ..models.runtime import ExecutionModel, WorkerDeployment, active_latent_capacity_tokens
from .capacity import (
    derive_runtime_kv_capacity,
    device_total_bytes,
    latent_pool_capacity_bytes,
    operation_window,
)
from .execution_config import graph_memory_budget_bytes, graph_padding_block_count
from .worker_info import (
    KvCacheConfig,
    KvGroup,
    KvGroupKind,
    RankInfo,
    WorkerInfo,
)

__all__ = ["WorkerLayout", "build_worker_info", "build_worker_layout"]


@dataclass(frozen=True, slots=True)
class WorkerLayout:
    """Worker-local model geometry paired with the public capacity report."""

    info: WorkerInfo
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


def build_worker_info(
    model: ExecutionModel,
    deployment: WorkerDeployment,
    *,
    model_name: str | None = None,
    weight_version: int = 0,
    queue_depth: int = 1,
    completion_payload_bytes: int = 1 << 20,
) -> WorkerInfo:
    """Build the post-load worker capacity handshake."""

    return build_worker_layout(
        model,
        deployment,
        model_name=model_name,
        weight_version=weight_version,
        queue_depth=queue_depth,
        completion_payload_bytes=completion_payload_bytes,
    ).info


def build_worker_layout(
    model: ExecutionModel,
    deployment: WorkerDeployment,
    *,
    model_name: str | None = None,
    weight_version: int = 0,
    queue_depth: int = 1,
    completion_payload_bytes: int = 1 << 20,
) -> WorkerLayout:
    """Build worker-local model geometry and its public capacity projection."""

    model_name = model.architecture if model_name is None else model_name

    if model.dedicated_state_geometry is not None:
        return _dedicated_state_worker_layout(
            model,
            deployment,
            model_name=model_name,
            weight_version=weight_version,
            queue_depth=queue_depth,
            completion_payload_bytes=completion_payload_bytes,
        )

    resources = model.resource_geometry
    owns_kv = bool(resources.kv)
    bytes_per_token = _kv_bytes_per_token(model, deployment) if owns_kv else 0
    flow = model.generation
    if flow is not None and not isinstance(flow, GenerationPipeline):
        raise invalid_descriptor("model generation behavior has an invalid type")
    requested_latent_units = (
        active_latent_capacity_tokens(
            int(flow.max_latent_tokens),
            deployment.kv_token_capacity,
        )
        if flow is not None
        else 0
    )
    latent_page_units = int(deployment.block_size) if flow is not None else 0
    num_latent_pages = (
        ceil_div(requested_latent_units, latent_page_units) + 1 if flow is not None else 0
    )
    latent_width = (
        int(flow.latent_channels) * int(flow.latent_patch_size) ** 2 if flow is not None else 0
    )
    model_dtype_bytes = _model_dtype_bytes(deployment.model_dtype)
    latent_pool_bytes = (
        latent_pool_capacity_bytes(
            request_pool_size=int(deployment.max_request_pool_size),
            num_pages=num_latent_pages,
            page_units=latent_page_units,
            latent_width=latent_width,
            dtype_bytes=model_dtype_bytes,
        )
        if flow is not None
        else 0
    )
    cache = model.cache_geometry if owns_kv else None
    max_vit_grid_tokens = int(getattr(model, "max_vit_grid_tokens", 0))
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
    if owns_kv:
        resident_copies, co_resident_blocks = _kv_residency_shape(
            deployment,
            bytes_per_token=bytes_per_token,
            co_resident_bytes=(
                latent_pool_bytes
                if deployment.generation_device in {None, deployment.device}
                else 0
            ),
        )
        capacity = derive_runtime_kv_capacity(
            block_size=int(deployment.block_size),
            kv_token_capacity=deployment.kv_token_capacity,
            bytes_per_token=bytes_per_token,
            device=deployment.device,
            memory_fraction=float(deployment.kv_memory_fraction),
            resident_copies=resident_copies,
            co_resident_blocks=co_resident_blocks,
        )
    if int(completion_payload_bytes) < 1:
        raise ValueError("completion payload capacity must be positive")
    unresolved_window = operation_window(
        int(queue_depth),
        int(deployment.max_batch_operations),
    )
    supported_ops = logical_op_kinds(tuple(model.supported_work))
    info = WorkerInfo(
        model_name=model_name,
        weight_version=weight_version,
        rank=RankInfo(tp_rank=int(deployment.tp_rank), tp_size=int(deployment.tp_size)),
        supported_ops=supported_ops,
        queue_depth=int(queue_depth),
        max_batch_ops=int(deployment.max_batch_operations),
        max_batch_tokens=int(deployment.max_batch_tokens),
        request_slots=int(deployment.max_request_pool_size),
        kv_cache=(
            KvCacheConfig(
                block_size=int(deployment.block_size),
                num_blocks=int(capacity.num_blocks),
                num_layers=int(cache.num_layers),
                num_kv_heads=int(cache.num_kv_heads),
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
                dtype=_kv_dtype(model, deployment),
            )
            if capacity is not None and cache is not None
            else None
        ),
        latent_page_units=latent_page_units,
        latent_pages=num_latent_pages,
        buffer_pool_bytes=(int(resources.encoder_cache_entries) + 1)
        * max(max_latent_feature_bytes, max_vision_feature_bytes),
        max_unresolved_ops=unresolved_window,
    )
    return WorkerLayout(
        info=info,
        latent_width=latent_width,
        latent_dtype=deployment.model_dtype if flow is not None else "",
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
        model_dtype=deployment.model_dtype,
    )


def _dedicated_state_worker_layout(
    model: ExecutionModel,
    deployment: WorkerDeployment,
    *,
    model_name: str,
    weight_version: int,
    queue_depth: int,
    completion_payload_bytes: int,
) -> WorkerLayout:
    if int(completion_payload_bytes) < 1:
        raise ValueError("completion payload capacity must be positive")
    geometry = model.dedicated_state_geometry
    if geometry is None:
        raise RuntimeError("dedicated state worker info requires declared state geometry")
    slots = int(geometry.slot_count)
    depth = int(queue_depth)
    unresolved_window = depth // slots - 1
    if unresolved_window < 2 or depth < slots * (unresolved_window + 1):
        raise invalid_descriptor(
            "dedicated-state pipeline depth does not provide two unresolved outputs per state slot"
        )
    max_operations = min(slots, int(deployment.max_batch_operations))
    info = WorkerInfo(
        model_name=model_name,
        weight_version=weight_version,
        rank=RankInfo(tp_rank=int(geometry.rank), tp_size=int(geometry.size)),
        supported_ops=logical_op_kinds(tuple(model.supported_work)),
        queue_depth=depth,
        max_batch_ops=max_operations,
        max_batch_tokens=max_operations,
        request_slots=slots,
        kv_cache=None,
        latent_page_units=int(geometry.persistent_units),
        latent_pages=slots + 1,
        buffer_pool_bytes=0,
        max_unresolved_ops=unresolved_window,
    )
    return WorkerLayout(
        info=info,
        latent_width=1,
        latent_dtype="float32",
        latent_downsample=1,
        max_vae_grid_tokens=int(geometry.max_vae_grid_tokens),
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


def _kv_dtype(model: ExecutionModel, deployment: WorkerDeployment) -> str:
    override = deployment.kv_cache_dtype
    if override is None:
        cache = model.cache_geometry
        return str(cache.store_dtype or cache.dtype).removeprefix("torch.")
    return str(override).removeprefix("torch.")


def _model_dtype_bytes(dtype: str) -> int:
    width = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }.get(str(dtype).removeprefix("torch.").lower())
    if width is None:
        raise ValueError(f"unsupported model dtype {dtype!r}")
    return width


def _kv_bytes_per_token(model: ExecutionModel, deployment: WorkerDeployment) -> int:
    width = {
        "float8_e4m3fn": 1,
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }.get(_kv_dtype(model, deployment).lower())
    if width is None:
        raise ValueError(f"unsupported KV dtype {_kv_dtype(model, deployment)!r}")
    cache = model.cache_geometry
    return 2 * int(width) * int(cache.num_layers) * int(cache.num_kv_heads) * int(cache.head_dim)


def _kv_residency_shape(
    deployment: WorkerDeployment,
    *,
    bytes_per_token: int,
    co_resident_bytes: int,
) -> tuple[int, int]:
    """Describe the single KV pool and co-resident fixed allocations."""

    block_size = int(deployment.block_size)
    padding_blocks = graph_padding_block_count(block_size)
    # Captured executables are held for the worker's lifetime, so they occupy
    # the same static budget as the KV pools and are reserved before the
    # request pool is sized.
    graph_blocks = ceil_div(
        graph_memory_budget_bytes(device_total_bytes(deployment.device)),
        block_size * max(1, int(bytes_per_token)),
    )
    fixed_owner_blocks = ceil_div(
        max(0, int(co_resident_bytes)),
        block_size * max(1, int(bytes_per_token)),
    )
    return 1, padding_blocks + graph_blocks + fixed_owner_blocks
