"""Resolve scheduler capabilities from immutable model and deployment specs."""

from __future__ import annotations

from ..capabilities import (
    AdapterMode,
    EngineCaps,
    ExecutionConstraints,
    RankInfo,
    RequestKind,
    ResourceClass,
)
from ..foundation.sizing import ceil_div, derive_runtime_kv_capacity
from ..spec import DeploymentOverlay, ModelSpec, active_latent_capacity_tokens

__all__ = ["resolve_capabilities"]


def resolve_capabilities(
    spec: ModelSpec,
    deployment: DeploymentOverlay,
    *,
    model_spec_digest: str | None = None,
    weight_digest: str | None = None,
) -> EngineCaps:
    """Build the complete capability snapshot without consulting model code."""

    resources = deployment.resources
    bytes_per_token = _kv_bytes_per_token(spec, deployment)
    capacity = derive_runtime_kv_capacity(
        block_size=int(deployment.block_size),
        kv_token_capacity=deployment.kv_token_capacity,
        bytes_per_token=bytes_per_token,
        device=deployment.device,
        memory_fraction=float(deployment.kv_memory_fraction),
    )
    flow = spec.flow
    max_latent_size = (
        active_latent_capacity_tokens(
            int(flow.max_latent_tokens),
            deployment.kv_token_capacity,
        )
        if flow is not None
        else 0
    )
    scratch_capacity = _scratch_capacity_tokens(
        deployment,
        num_blocks=int(capacity.num_blocks),
        max_latent_size=max_latent_size,
    )
    controls = [RequestKind.DROP_SESSION, RequestKind.COPY_KV, RequestKind.RESET_PREFIX_CACHE]
    if resources.encoder_output is not None:
        controls.append(RequestKind.RELEASE_PRODUCTS)
    if resources.adapter is not None:
        controls.extend((RequestKind.LOAD_ADAPTER, RequestKind.UNLOAD_ADAPTER))
    return EngineCaps(
        block_size=int(deployment.block_size),
        num_blocks=int(capacity.num_blocks),
        num_layers=int(spec.cache.num_layers),
        scratch_capacity_tokens=scratch_capacity,
        supported_operation_types=tuple(operation.kind for operation in spec.operations),
        max_latent_size=max_latent_size,
        latent_downsample=int(flow.latent_downsample) if flow is not None else 1,
        bytes_per_token=bytes_per_token,
        supported_controls=tuple(controls),
        adapter_mode=AdapterMode(deployment.adapter_mode),
        execution_constraints=ExecutionConstraints(
            max_batch_operations=int(deployment.max_batch_operations)
        ),
        resource_classes=tuple(ResourceClass(value) for value in resources.classes()),
        attention_backend=deployment.attention_backend or "auto",
        kv_dtype=_kv_dtype(spec, deployment),
        model_dtype=deployment.model_dtype,
        encoder_cache_budget=int(spec.inputs.encoder_cache_budget),
        max_vae_grid_tokens=int(flow.max_vae_grid_tokens) if flow is not None else 0,
        max_vit_grid_tokens=int(spec.inputs.max_vit_grid_tokens),
        commit_marker_tokens=int(flow.commit_marker_tokens) if flow is not None else 2,
        gen_rope_advance=int(flow.rope_advance) if flow is not None else 2,
        max_cfg_branches=int(flow.max_cfg_branches) if flow is not None else 1,
        rank=RankInfo(tp_rank=int(deployment.tp_rank), tp_size=int(deployment.tp_size)),
        pipeline_depth=1,
        groups=(),
        quantization=None,
        model_spec_digest=model_spec_digest or "",
        weight_digest=weight_digest or "",
    )


def _kv_dtype(spec: ModelSpec, deployment: DeploymentOverlay) -> str:
    override = deployment.kv_cache_dtype
    if override is None:
        return str(spec.cache.store_dtype or spec.cache.dtype).removeprefix("torch.")
    return str(override).removeprefix("torch.")


def _kv_bytes_per_token(spec: ModelSpec, deployment: DeploymentOverlay) -> int:
    width = {
        "float8_e4m3fn": 1,
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }.get(_kv_dtype(spec, deployment).lower())
    if width is None:
        raise ValueError(f"unsupported KV dtype {_kv_dtype(spec, deployment)!r}")
    cache = spec.cache
    return (
        2
        * int(width)
        * int(cache.num_layers)
        * int(cache.num_kv_heads)
        * int(cache.head_dim)
    )


def _scratch_capacity_tokens(
    deployment: DeploymentOverlay,
    *,
    num_blocks: int,
    max_latent_size: int,
) -> int:
    scratch = deployment.resources.scratch
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
