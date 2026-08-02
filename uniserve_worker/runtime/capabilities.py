"""Resolve scheduler capabilities from immutable model and deployment specs."""

from __future__ import annotations

from ..batch import SamplingOwnership
from ..capabilities import (
    AdapterMode,
    CreditVector,
    EngineCaps,
    ExecutionConstraints,
    RankInfo,
    RequestKind,
    ResourceClass,
    RouteCreditLimits,
    RouteExecutionCapability,
    work_variants_for_operation_types,
)
from ..foundation.runtime_config import (
    graph_memory_budget_bytes,
    graph_padding_block_count,
)
from ..foundation.sizing import ceil_div, derive_runtime_kv_capacity, device_total_bytes
from ..spec import DeploymentOverlay, ModelSpec, active_latent_capacity_tokens
from .latent_capacity import latent_store_capacity_bytes
from .product_capacity import device_product_arena_bytes

__all__ = ["resolve_capabilities"]


def resolve_capabilities(
    spec: ModelSpec,
    deployment: DeploymentOverlay,
    *,
    model_spec_digest: str | None = None,
    weight_digest: str | None = None,
    pipeline_depth: int = 1,
    completion_payload_bytes: int = 1 << 20,
) -> EngineCaps:
    """Build the complete capability snapshot without consulting model code."""

    resources = deployment.resources
    bytes_per_token = _kv_bytes_per_token(spec, deployment)
    flow = spec.flow
    max_latent_size = (
        active_latent_capacity_tokens(
            int(flow.max_latent_tokens),
            deployment.kv_token_capacity,
        )
        if flow is not None
        else 0
    )
    hidden_elements = int(spec.cache.num_attention_heads) * int(spec.cache.head_dim)
    max_vision_feature_bytes = int(spec.inputs.max_vit_grid_tokens) * hidden_elements * 2
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
        num_blocks=int(capacity.num_blocks),
        max_latent_size=max_latent_size,
    )
    credit_limits = _route_credit_limits(
        spec,
        deployment,
        pipeline_depth=int(pipeline_depth),
        completion_payload_bytes=int(completion_payload_bytes),
        num_blocks=int(capacity.num_blocks),
        scratch_capacity_tokens=int(scratch_capacity),
        max_latent_size=int(max_latent_size),
        max_latent_feature_bytes=int(max_latent_feature_bytes),
        max_vision_feature_bytes=int(max_vision_feature_bytes),
        bytes_per_token=int(bytes_per_token),
    )
    controls = [
        RequestKind.DROP_SESSION,
        RequestKind.COPY_KV,
        RequestKind.RESET_PREFIX_CACHE,
        RequestKind.RELEASE_PRODUCTS,
    ]
    if resources.adapter is not None:
        controls.extend((RequestKind.LOAD_ADAPTER, RequestKind.UNLOAD_ADAPTER))
    supported_work = work_variants_for_operation_types(
        tuple(operation.kind for operation in spec.operations)
    )
    return EngineCaps(
        block_size=int(deployment.block_size),
        num_blocks=int(capacity.num_blocks),
        num_layers=int(spec.cache.num_layers),
        scratch_capacity_tokens=scratch_capacity,
        supported_work=supported_work,
        max_latent_size=max_latent_size,
        latent_downsample=int(flow.latent_downsample) if flow is not None else 1,
        bytes_per_token=bytes_per_token,
        supported_controls=tuple(controls),
        adapter_mode=AdapterMode(deployment.adapter_mode),
        execution_constraints=ExecutionConstraints(
            max_batch_operations=int(deployment.max_batch_operations),
            max_speculative_points=17,
            device_sequence_lengths=True,
            device_append_offsets=True,
            incremental_kv_publication=True,
            route_capabilities=(
                RouteExecutionCapability(
                    route=0,
                    supported_work=supported_work,
                    tensorized_mixed=any(bool(route.mixed_combinations) for route in spec.routes),
                    sampling_ownership=SamplingOwnership.DESIGNATED_RANK,
                    preemptible=False,
                    credits=credit_limits,
                ),
            ),
        ),
        resource_classes=tuple(ResourceClass(value) for value in resources.classes()),
        attention_backend=deployment.attention_backend or "auto",
        kv_dtype=_kv_dtype(spec, deployment),
        model_dtype=deployment.model_dtype,
        encoder_cache_budget=int(spec.inputs.encoder_cache_budget),
        max_vae_grid_tokens=int(flow.max_vae_grid_tokens) if flow is not None else 0,
        max_vit_grid_tokens=int(spec.inputs.max_vit_grid_tokens),
        max_latent_feature_bytes=max_latent_feature_bytes,
        max_vision_feature_bytes=max_vision_feature_bytes,
        commit_marker_tokens=int(flow.commit_marker_tokens) if flow is not None else 2,
        gen_rope_advance=int(flow.rope_advance) if flow is not None else 2,
        max_cfg_branches=int(flow.max_cfg_branches) if flow is not None else 1,
        rank=RankInfo(tp_rank=int(deployment.tp_rank), tp_size=int(deployment.tp_size)),
        pipeline_depth=int(pipeline_depth),
        groups=(),
        quantization=None,
        model_spec_digest=model_spec_digest or "",
        weight_digest=weight_digest or "",
    )


def _route_credit_limits(
    spec: ModelSpec,
    deployment: DeploymentOverlay,
    *,
    pipeline_depth: int,
    completion_payload_bytes: int,
    num_blocks: int,
    scratch_capacity_tokens: int,
    max_latent_size: int,
    max_latent_feature_bytes: int,
    max_vision_feature_bytes: int,
    bytes_per_token: int,
) -> RouteCreditLimits:
    if pipeline_depth < 1 or completion_payload_bytes < 1:
        raise ValueError("credit sizing requires positive pipeline and payload bounds")
    max_operations = int(deployment.max_batch_operations)
    slots = pipeline_depth * max_operations
    window = min(slots, max(2, pipeline_depth))
    max_speculative_points = 17
    products_per_operation = 5
    transfer_tickets = min(slots, 256)
    block_size = int(deployment.block_size)
    scratch_pages = ceil_div(int(scratch_capacity_tokens), block_size)
    kv_pages = int(num_blocks) + scratch_pages
    max_route_tokens = max(
        (int(route.shape.max_tokens_per_row) for route in spec.routes), default=1
    )
    max_transfer_per_operation = max(
        int(num_blocks) * block_size * int(bytes_per_token),
        int(max_latent_feature_bytes),
        int(max_vision_feature_bytes),
        1,
    )
    flow = spec.flow
    latent_capacity_bytes = 0
    artifact_bytes = 0
    if flow is not None:
        latent_capacity_bytes = latent_store_capacity_bytes(
            int(max_latent_size),
            int(flow.latent_channels),
            int(flow.latent_patch_size),
        )
        raw_image_bytes = int(flow.max_vae_grid_tokens) * int(flow.latent_downsample) ** 2 * 3
        artifact_bytes = ((2 * raw_image_bytes + (1 << 20) + 2) // 3) * 4
    product_bytes = max(1, artifact_bytes, max_latent_feature_bytes, max_vision_feature_bytes)
    latent_artifact_bytes = latent_capacity_bytes + products_per_operation * window * product_bytes
    device_product_slots = products_per_operation * slots
    device_count = len(
        {
            str(deployment.device),
            str(deployment.generation_device or deployment.device),
        }
    )
    worker_product_bytes = device_product_arena_bytes(
        device_product_slots,
        device_count,
        max_speculative_points=max_speculative_points,
        max_product_bytes=product_bytes,
    )
    max_event_bytes = max(4096, artifact_bytes)
    output_journal_bytes = 64 * max_event_bytes
    per_operation_staging = (
        int(completion_payload_bytes) + max_route_tokens * 32 + kv_pages * 8 + 256
    )
    completion_words = 4 * max_operations + (int(completion_payload_bytes) + 3) // 4
    completion_arena_bytes = pipeline_depth * completion_words * 8
    per_request = CreditVector(
        registered_operations=window,
        execution_slots=window,
        completion_slots=window,
        device_products=products_per_operation * window,
        kv_pages=kv_pages,
        rollback_deltas=max_speculative_points * window,
        latent_artifact_bytes=latent_artifact_bytes,
        pinned_completion_staging_bytes=per_operation_staging * window,
        transfer_bytes=max_transfer_per_operation * window,
        transfer_tickets=min(window, transfer_tickets),
        cpu_tasks=2,
        output_journal_bytes=output_journal_bytes,
    )
    worker = CreditVector(
        registered_operations=slots,
        execution_slots=slots,
        completion_slots=slots,
        device_products=products_per_operation * slots,
        kv_pages=kv_pages,
        rollback_deltas=max_speculative_points * slots,
        latent_artifact_bytes=latent_capacity_bytes + worker_product_bytes,
        pinned_completion_staging_bytes=max(
            completion_arena_bytes,
            per_operation_staging * window,
        ),
        transfer_bytes=max_transfer_per_operation * transfer_tickets,
        transfer_tickets=transfer_tickets,
        cpu_tasks=256,
        output_journal_bytes=output_journal_bytes * max_operations,
    )
    return RouteCreditLimits(per_request=per_request, worker=worker)


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
    return 2 * int(width) * int(cache.num_layers) * int(cache.num_kv_heads) * int(cache.head_dim)


def _kv_residency_shape(
    deployment: DeploymentOverlay,
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
    scratch = deployment.resources.scratch
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
