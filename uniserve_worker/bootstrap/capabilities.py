"""Resolve WorkerInfo from one loaded model and worker geometry."""

from __future__ import annotations

from ..batch import ForwardMode, SamplingOwnership
from ..capabilities import (
    KvGroupKind,
    KvGroupSpec,
    RankInfo,
    RequestKind,
    ResourceClass,
    WorkerInfo,
)
from ..foundation.errors import invalid_descriptor
from ..foundation.math import ceil_div
from ..models.generation import GenerationPipeline, LatentLayout
from ..models.minimax_h3 import MiniMaxH3Model
from ..models.runtime import ExecutionModel, WorkerDeployment, active_latent_capacity_tokens
from .capacity import (
    derive_runtime_kv_capacity,
    device_total_bytes,
    latent_pool_capacity_bytes,
    operation_window,
)
from .execution_config import graph_memory_budget_bytes, graph_padding_block_count

__all__ = ["resolve_capabilities"]


def resolve_capabilities(
    model: ExecutionModel | MiniMaxH3Model,
    deployment: WorkerDeployment,
    *,
    architecture_digest: str | None = None,
    weight_digest: str | None = None,
    pipeline_depth: int = 1,
    completion_payload_bytes: int = 1 << 20,
) -> WorkerInfo:
    """Build the post-load worker handshake from model-owned behavior."""

    if isinstance(model, MiniMaxH3Model):
        return _h3_capabilities(
            model,
            deployment,
            architecture_digest=architecture_digest,
            weight_digest=weight_digest,
            pipeline_depth=pipeline_depth,
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
    hidden_elements = (
        int(cache.num_attention_heads) * int(cache.head_dim)
        if cache is not None
        else int(getattr(model, "hidden_size", 0))
    )
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
        int(pipeline_depth),
        int(deployment.max_batch_operations),
    )
    controls = [
        RequestKind.DROP_SESSION,
        RequestKind.RELEASE_PRODUCTS,
    ]
    supported_work = tuple(variant for variant in ForwardMode if variant in model.supported_work)
    sampling_ownership = SamplingOwnership.DESIGNATED_RANK
    return WorkerInfo(
        block_size=int(deployment.block_size) if owns_kv else 0,
        num_blocks=int(capacity.num_blocks) if capacity is not None else 0,
        num_layers=int(cache.num_layers) if cache is not None else 0,
        num_kv_heads=int(cache.num_kv_heads) if cache is not None else 0,
        head_dim=int(cache.head_dim) if cache is not None else 0,
        supported_work=supported_work,
        latent_page_units=latent_page_units,
        num_latent_pages=num_latent_pages,
        latent_width=latent_width,
        latent_dtype=deployment.model_dtype if flow is not None else "",
        latent_downsample=int(flow.latent_downsample) if flow is not None else 1,
        bytes_per_token=bytes_per_token,
        supported_controls=tuple(controls),
        max_batch_operations=int(deployment.max_batch_operations),
        max_batch_tokens=int(deployment.max_batch_tokens),
        max_request_pool_size=int(deployment.max_request_pool_size),
        max_unresolved_window=unresolved_window,
        incremental_kv_publication=owns_kv,
        mixed_buckets=(),
        sampling_ownership=sampling_ownership,
        resource_classes=tuple(ResourceClass(value) for value in resources.classes()),
        kv_dtype=_kv_dtype(model, deployment) if owns_kv else "",
        model_dtype=deployment.model_dtype,
        encoder_cache_budget=int(resources.encoder_cache_entries),
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
        rank=RankInfo(tp_rank=int(deployment.tp_rank), tp_size=int(deployment.tp_size)),
        pipeline_depth=int(pipeline_depth),
        groups=(
            (
                KvGroupSpec(
                    num_blocks=int(capacity.num_blocks),
                    kind=KvGroupKind.FULL,
                    window=0,
                    sink=0,
                ),
            )
            if capacity is not None
            else ()
        ),
        model_identity=architecture_digest or "",
        weight_digest=weight_digest or "",
    )


def _h3_capabilities(
    model: MiniMaxH3Model,
    deployment: WorkerDeployment,
    *,
    architecture_digest: str | None,
    weight_digest: str | None,
    pipeline_depth: int,
    completion_payload_bytes: int,
) -> WorkerInfo:
    if int(completion_payload_bytes) < 1:
        raise ValueError("completion payload capacity must be positive")
    slots = int(model.states.slot_count)
    depth = int(pipeline_depth)
    unresolved_window = depth // slots - 1
    if unresolved_window < 2 or depth < slots * (unresolved_window + 1):
        raise invalid_descriptor(
            "MiniMax H3 pipeline depth does not provide two unresolved outputs per state slot"
        )
    max_operations = min(slots, int(deployment.max_batch_operations))
    layout = model.layout
    return WorkerInfo(
        block_size=0,
        num_blocks=0,
        num_layers=0,
        num_kv_heads=0,
        head_dim=0,
        supported_work=tuple(
            variant for variant in ForwardMode if variant in model.supported_work
        ),
        latent_page_units=int(layout.persistent_units),
        num_latent_pages=slots + 1,
        latent_width=1,
        latent_dtype="float32",
        latent_downsample=1,
        max_vae_grid_tokens=int(layout.packed.video_indices.numel()),
        max_vit_grid_tokens=0,
        max_latent_feature_bytes=0,
        max_vision_feature_bytes=0,
        commit_marker_tokens=0,
        gen_rope_advance=1,
        max_cfg_branches=1,
        bytes_per_token=0,
        groups=(),
        kv_dtype="",
        model_dtype="bfloat16",
        rank=RankInfo(tp_rank=model.mesh.coord("sp"), tp_size=model.mesh.size("sp")),
        pipeline_depth=depth,
        encoder_cache_budget=0,
        supported_controls=(RequestKind.DROP_SESSION, RequestKind.RELEASE_PRODUCTS),
        max_batch_operations=max_operations,
        max_batch_tokens=max_operations,
        max_request_pool_size=slots,
        max_unresolved_window=unresolved_window,
        incremental_kv_publication=False,
        mixed_buckets=(),
        sampling_ownership=SamplingOwnership.DESIGNATED_RANK,
        resource_classes=(ResourceClass.IMAGE_LATENT,),
        model_identity=architecture_digest or "",
        weight_digest=weight_digest or "",
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
