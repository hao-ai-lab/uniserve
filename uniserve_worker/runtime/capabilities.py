"""Resolve scheduler capabilities from immutable model and deployment specs."""

from __future__ import annotations

from collections.abc import Sequence

from ..batch import SamplingOwnership, WorkVariant
from ..capabilities import (
    CreditVector,
    EngineCaps,
    ExecutionConstraints,
    RankInfo,
    RequestKind,
    ResourceClass,
    RouteCreditLimits,
    RouteExecutionCapability,
    configured_work_variants,
)
from ..foundation.errors import invalid_descriptor
from ..foundation.runtime_config import (
    graph_memory_budget_bytes,
    graph_padding_block_count,
)
from ..foundation.sizing import ceil_div, derive_runtime_kv_capacity, device_total_bytes
from ..spec import (
    DeploymentOverlay,
    FlowConditioningKind,
    ModelSpec,
    OperationStagePurpose,
    OperationStageSpec,
    RouteRowKind,
    active_latent_capacity_tokens,
)
from .latent_capacity import latent_store_capacity_bytes
from .product_capacity import device_product_arena_bytes

# KV, latent, and scratch residency hold committed request state across the
# operation boundary, so a route sharing them advances state in place and is not
# safe to preempt.
_STATEFUL_RESOURCE_CLASSES = frozenset({"kv_block", "image_latent", "scratch"})

# Sampler-processor, RNG-layout, and processor-order constants mirroring the
# worker-wire capability constants byte-for-byte. The full sampler-processor set
# is the fourteen bits of the canonical processor order; the legal-continuation
# feature set is the device-representable subset (temperature, top-k, top-p,
# min-p, typical) for which a successor may relay before host observation.
PROCESSOR_ORDER_REVISION = 1
_ALL_SAMPLER_PROCESSORS = (1 << 14) - 1
_LEGAL_CONTINUATION_FEATURES = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
_RNG_TARGET_SAMPLING = 1 << 0
_RNG_FLOW_NOISE = 1 << 2
_ROUTE_ROW_KIND_BIT = {kind: 1 << index for index, kind in enumerate(RouteRowKind)}

__all__ = ["prove_depth_one_lowering", "resolve_capabilities"]


def prove_depth_one_lowering(
    spec: ModelSpec,
    supported_work: Sequence[WorkVariant],
) -> dict[WorkVariant, OperationStageSpec | None]:
    """Prove every advertised work variant lowers to one declared model route.

    Each admitted variant resolves through the single primary stage of its
    ``OperationSpec`` to one ``RouteSpec`` the model declares. A stage-less
    system-only operation, and a materialization that carries no neural primary
    stage, lower to no route — the same model-free frame lowering the executor
    performs in ``_operation_stages_for`` — and map to ``None`` rather than
    failing. Admission fails deterministically when a variant names no
    operation, a variant declares
    more than one primary stage (which would break one-variant-to-one-route), or
    a primary stage names a route the model does not declare.
    """

    route_names = {route.name for route in spec.routes}
    primary_by_variant: dict[WorkVariant, OperationStageSpec | None] = {}
    for variant in supported_work:
        operation = spec.operation(variant)
        primaries = tuple(
            stage
            for stage in operation.stages
            if stage.purpose is OperationStagePurpose.PRIMARY
        )
        if len(primaries) > 1:
            raise invalid_descriptor(
                f"work variant {variant.value!r} declares multiple primary stages, "
                "breaking one-variant-to-one-route lowering"
            )
        if not primaries:
            # A variant with no primary neural stage lowers onto no compute
            # route: transfer and KV state-publication ops carry only STATE
            # stages, a materialized frame is model-free, and system ops are
            # stage-less. They map to no route rather than failing.
            primary_by_variant[variant] = None
            continue
        stage = primaries[0]
        if stage.route not in route_names:
            raise invalid_descriptor(
                f"work variant {variant.value!r} lowers onto undeclared route {stage.route!r}"
            )
        primary_by_variant[variant] = stage
    return primary_by_variant


def _route_tensorized_mixed(
    spec: ModelSpec,
    primary_by_variant: dict[WorkVariant, OperationStageSpec | None],
) -> bool:
    """Whether any declared route admits a mixed row combination reachable by the
    admitted work: every row kind of a declared combination must be produced by a
    primary stage the advertised variants lower onto."""

    primary_rows: dict[str, set[RouteRowKind]] = {}
    for stage in primary_by_variant.values():
        if stage is not None:
            primary_rows.setdefault(stage.route, set()).add(stage.row)
    return any(
        any(
            set(combination) <= primary_rows.get(route.name, set())
            for combination in route.mixed_combinations
        )
        for route in spec.routes
    )


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
        RequestKind.RELEASE_PRODUCTS,
    ]
    supported_work = configured_work_variants(
        operation.kind for operation in spec.operations
    )
    # Startup lowering proof: every advertised variant must reduce to one
    # declared route via its primary stage before the capability is published.
    primary_by_variant = prove_depth_one_lowering(spec, supported_work)
    # Speculative acceptance needs an advertised drafter; a serving route
    # advertises none (DRAFT carries no depth-one route), so the declared
    # acceptance window is a single verified point.
    max_speculative_points = 2 if WorkVariant.DRAFT in supported_work else 1
    tensorized_mixed = _route_tensorized_mixed(spec, primary_by_variant)
    # A route holding paged KV, latent, or scratch residency advances committed
    # request state in place and is not safe to preempt mid-operation.
    preemptible = not _STATEFUL_RESOURCE_CLASSES.intersection(resources.classes())
    # Sampling reduces full-vocabulary logits on one designated rank; no route
    # declares a deterministically sharded sampler.
    sampling_ownership = SamplingOwnership.DESIGNATED_RANK
    # DR-072 route facts derived from the model spec: captured-graph eligibility,
    # the exact tensorized mixed row combinations (each a bitset of row kinds),
    # the Gen conditioning form, and the RNG draw spaces the route addresses.
    route_graph_eligible = all(route.graph_eligible for route in spec.routes)
    mixed_row_combinations = tuple(
        sorted(
            {
                sum(_ROUTE_ROW_KIND_BIT[kind] for kind in combination)
                for route in spec.routes
                for combination in route.mixed_combinations
            }
        )
    )
    gen_conditioning = (
        list(FlowConditioningKind).index(flow.conditioning) if flow is not None else 0
    )
    rng_layouts = _RNG_TARGET_SAMPLING | (_RNG_FLOW_NOISE if flow is not None else 0)
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
        execution_constraints=ExecutionConstraints(
            max_batch_operations=int(deployment.max_batch_operations),
            max_speculative_points=max_speculative_points,
            max_unresolved_window=int(credit_limits.per_request.registered_operations),
            device_sequence_lengths=True,
            device_append_offsets=True,
            incremental_kv_publication=True,
            route_capabilities=(
                RouteExecutionCapability(
                    route=0,
                    supported_work=supported_work,
                    tensorized_mixed=tensorized_mixed,
                    sampling_ownership=sampling_ownership,
                    preemptible=preemptible,
                    credits=credit_limits,
                    max_unresolved_window=int(
                        credit_limits.per_request.registered_operations
                    ),
                    legal_feature_bitset=_LEGAL_CONTINUATION_FEATURES,
                    sampler_processors=_ALL_SAMPLER_PROCESSORS,
                    processor_order_revision=PROCESSOR_ORDER_REVISION,
                    rng_layouts=rng_layouts,
                    graph_eligible=route_graph_eligible,
                    gen_conditioning=gen_conditioning,
                    max_points_per_operation=max_speculative_points,
                    mixed_row_combinations=mixed_row_combinations,
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
    max_speculative_points = 1
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
