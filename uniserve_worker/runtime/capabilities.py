"""Resolve scheduler capabilities from one loaded model and worker geometry."""

from __future__ import annotations

from collections.abc import Sequence

from ..batch import SamplingOwnership, WorkVariant
from ..capabilities import (
    RankInfo,
    RequestKind,
    ResourceClass,
    WorkerCapabilities,
    configured_work_variants,
)
from ..foundation.errors import invalid_descriptor
from ..foundation.runtime_config import (
    graph_memory_budget_bytes,
    graph_padding_block_count,
)
from ..foundation.sizing import ceil_div, derive_runtime_kv_capacity, device_total_bytes
from ..models.generation import GenerationPipeline
from ..models.runtime import (
    ExecutionModel,
    LoweredStage,
    RowKind,
    WorkerDeployment,
    active_latent_capacity_tokens,
)
from .arena_capacity import operation_window

__all__ = ["prove_depth_one_lowering", "resolve_capabilities"]


def prove_depth_one_lowering(
    model: ExecutionModel,
    supported_work: Sequence[WorkVariant],
) -> dict[WorkVariant, LoweredStage | None]:
    """Prove that each advertised leaf lowers to at most one neural primary."""

    primary_by_variant: dict[WorkVariant, LoweredStage | None] = {}
    for variant in supported_work:
        primaries = tuple(
            stage for stage in model.lower(variant) if not stage.publishes_state
        )
        if len(primaries) > 1:
            raise invalid_descriptor(
                f"work variant {variant.value!r} declares multiple primary stages, "
                "breaking one-variant-to-one-route lowering"
            )
        if not primaries:
            primary_by_variant[variant] = None
            continue
        primary_by_variant[variant] = primaries[0]
    return primary_by_variant


def _route_tensorized_mixed(
    model: ExecutionModel,
    primary_by_variant: dict[WorkVariant, LoweredStage | None],
) -> bool:
    primary_rows: dict[str, set[RowKind]] = {}
    for stage in primary_by_variant.values():
        if stage is not None:
            primary_rows.setdefault(stage.route, set()).add(stage.row)
    return any(
        len(rows) > 1 and model.allows_mixed(stage.route, frozenset(rows))
        for stage in primary_by_variant.values()
        if stage is not None
        for rows in (primary_rows[stage.route],)
    )


def resolve_capabilities(
    model: ExecutionModel,
    deployment: WorkerDeployment,
    *,
    architecture_digest: str | None = None,
    weight_digest: str | None = None,
    pipeline_depth: int = 1,
    completion_payload_bytes: int = 1 << 20,
) -> WorkerCapabilities:
    """Build the complete capability snapshot from model-owned behavior."""

    resources = model.resource_geometry
    bytes_per_token = _kv_bytes_per_token(model, deployment)
    flow = model.generation
    if flow is not None and not isinstance(flow, GenerationPipeline):
        raise invalid_descriptor("model generation behavior has an invalid type")
    max_latent_size = (
        active_latent_capacity_tokens(
            int(flow.max_latent_tokens),
            deployment.kv_token_capacity,
        )
        if flow is not None
        else 0
    )
    cache = model.cache_geometry
    max_vit_grid_tokens = int(getattr(model, "max_vit_grid_tokens", 0))
    hidden_elements = int(cache.num_attention_heads) * int(cache.head_dim)
    max_vision_feature_bytes = max_vit_grid_tokens * hidden_elements * 2
    max_latent_feature_bytes = (
        0
        if flow is None
        else (
            int(flow.max_vae_grid_tokens)
            * int(flow.latent_channels)
            * int(flow.latent_patch_size)
            * int(flow.latent_patch_size)
            * 2
        )
    )
    resident_copies, co_resident_blocks = _kv_residency_shape(
        deployment,
        resources,
        max_latent_size=max_latent_size,
        bytes_per_token=bytes_per_token,
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
    scratch_capacity = _scratch_capacity_tokens(
        deployment,
        resources,
        num_blocks=int(capacity.num_blocks),
        max_latent_size=max_latent_size,
    )
    if int(completion_payload_bytes) < 1:
        raise ValueError("completion payload capacity must be positive")
    unresolved_window = operation_window(
        int(pipeline_depth),
        int(deployment.max_batch_operations),
    )
    controls = [
        RequestKind.DROP_SESSION,
        RequestKind.COPY_KV,
        RequestKind.RELEASE_PRODUCTS,
    ]
    supported_work = configured_work_variants(model.supported_work)
    primary_by_variant = prove_depth_one_lowering(model, supported_work)
    tensorized_mixed = _route_tensorized_mixed(model, primary_by_variant)
    sampling_ownership = SamplingOwnership.DESIGNATED_RANK
    return WorkerCapabilities(
        block_size=int(deployment.block_size),
        num_blocks=int(capacity.num_blocks),
        num_layers=int(cache.num_layers),
        num_kv_heads=int(cache.num_kv_heads),
        head_dim=int(cache.head_dim),
        scratch_capacity_tokens=scratch_capacity,
        supported_work=supported_work,
        max_latent_size=max_latent_size,
        latent_downsample=int(flow.latent_downsample) if flow is not None else 1,
        bytes_per_token=bytes_per_token,
        supported_controls=tuple(controls),
        max_batch_operations=int(deployment.max_batch_operations),
        max_unresolved_window=unresolved_window,
        incremental_kv_publication=True,
        tensorized_mixed=tensorized_mixed,
        sampling_ownership=sampling_ownership,
        resource_classes=tuple(ResourceClass(value) for value in resources.classes()),
        attention_backend=deployment.attention_backend or "auto",
        kv_dtype=_kv_dtype(model, deployment),
        model_dtype=deployment.model_dtype,
        encoder_cache_budget=int(resources.encoder_cache_entries),
        max_vae_grid_tokens=int(flow.max_vae_grid_tokens) if flow is not None else 0,
        max_vit_grid_tokens=max_vit_grid_tokens,
        max_latent_feature_bytes=max_latent_feature_bytes,
        max_vision_feature_bytes=max_vision_feature_bytes,
        commit_marker_tokens=int(flow.commit_marker_tokens) if flow is not None else 2,
        gen_rope_advance=int(flow.rope_advance) if flow is not None else 2,
        max_cfg_branches=int(flow.max_cfg_branches) if flow is not None else 1,
        rank=RankInfo(tp_rank=int(deployment.tp_rank), tp_size=int(deployment.tp_size)),
        pipeline_depth=int(pipeline_depth),
        groups=(),
        quantization=None,
        model_identity=architecture_digest or "",
        weight_digest=weight_digest or "",
    )


def _kv_dtype(model: ExecutionModel, deployment: WorkerDeployment) -> str:
    override = deployment.kv_cache_dtype
    if override is None:
        cache = model.cache_geometry
        return str(cache.store_dtype or cache.dtype).removeprefix("torch.")
    return str(override).removeprefix("torch.")


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
    resources: object,
    *,
    max_latent_size: int,
    bytes_per_token: int,
) -> tuple[int, int]:
    """Describe every KV pool that shares the deployment memory budget.

    The first element counts the copies of the request pool held resident at
    once; the second counts the additional blocks provisioned alongside them.
    Together they let capacity sizing reserve room for the scratch residency
    that :func:`_scratch_capacity_tokens` goes on to declare.
    """

    block_size = int(deployment.block_size)
    padding_blocks = graph_padding_block_count(block_size)
    # Captured executables are held for the worker's lifetime, so they occupy
    # the same static budget as the KV pools and are reserved before the
    # request pool is sized.
    graph_blocks = ceil_div(
        graph_memory_budget_bytes(device_total_bytes(deployment.device)),
        block_size * max(1, int(bytes_per_token)),
    )
    scratch = getattr(resources, "scratch", None)
    if scratch is None:
        return 1, padding_blocks + graph_blocks
    fixed_blocks = ceil_div(int(scratch.fixed_tokens), block_size)
    latent_blocks = ceil_div(max_latent_size * int(scratch.latent_copies), block_size)
    return (
        2 if scratch.mirror_kv else 1,
        2 * padding_blocks
        + graph_blocks
        + max(int(scratch.minimum_blocks), fixed_blocks + latent_blocks),
    )


def _scratch_capacity_tokens(
    deployment: WorkerDeployment,
    resources: object,
    *,
    num_blocks: int,
    max_latent_size: int,
) -> int:
    scratch = getattr(resources, "scratch", None)
    if scratch is None:
        return 0
    block_size = int(deployment.block_size)
    blocks = max(
        int(scratch.minimum_blocks),
        ceil_div(int(scratch.fixed_tokens), block_size)
        + (num_blocks if scratch.mirror_kv else 0)
        + ceil_div(max_latent_size * int(scratch.latent_copies), block_size),
    )
    return int(blocks * block_size)
