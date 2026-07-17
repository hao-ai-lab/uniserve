"""Dormant family cache-schema registrations for the target cache runtime.

The "Family Mappings" of ``specs/unified_kv_attention_runtime.md`` expressed as
:mod:`~uniserve_worker.contracts.cache_schema` data. These factories prove the
acceptance obligation of the "Closed contracts and sessions" work package —
Qwen3, BAGEL, and SenseNova schemas validate without model callbacks — and give
the future ``ModelRegistration`` one authoritative construction site.

Geometry (layer count, head counts, head dims) comes from the resolved
checkpoint configuration, so the factories take it as arguments rather than
baking one checkpoint's shape into the contract. Nothing here is consumed by
production execution yet.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts.cache_schema import (
    AttentionPattern,
    AttentionSiteSpec,
    BranchCloseExpr,
    BranchOpenExpr,
    CacheDomainSpec,
    CacheEffect,
    CacheLifetime,
    CacheRegionSpec,
    CacheRoleSelector,
    CacheRoleSpec,
    CacheStateKind,
    CommitExpr,
    CommitExprKind,
    DTypeTag,
    ExtentExpr,
    ExtentOperand,
    FamilyCacheSchema,
    GridFieldTag,
    MhaHistoryLayout,
    OperationTag,
    PageOrder,
    PlacementConstraint,
    PositionExpr,
    PositionSchema,
    PrefixPublication,
    RegionMultiplicity,
    RetentionPolicy,
    RoleInitialization,
    RoleRelease,
    RouteRunSpec,
    Shareability,
    StoreFormat,
    validate_registration,
)

__all__ = [
    "FamilyCacheRegistration",
    "PRIMARY_ROLE",
    "IMAGE_UNCONDITIONAL_ROLE",
    "TEXT_UNCONDITIONAL_ROLE",
    "bagel_cache_registration",
    "qwen3_cache_registration",
    "sensenova_cache_registration",
]

PRIMARY_ROLE = 1
TEXT_UNCONDITIONAL_ROLE = 2
IMAGE_UNCONDITIONAL_ROLE = 3

_DECODER_DOMAIN = 1
_TEXT_ROUTE = 0
_GENERATION_ROUTE = 1

_TOKENS = ExtentExpr.operand(ExtentOperand.INPUT_TOKEN_COUNT)
_CANDIDATES = ExtentExpr.operand(ExtentOperand.CANDIDATE_COUNT)
_IMAGE_TOKENS = ExtentExpr.operand(ExtentOperand.IMAGE_TOKEN_COUNT)


@dataclass(frozen=True, slots=True)
class FamilyCacheRegistration:
    """One family's validated static cache registration."""

    domains: tuple[CacheDomainSpec, ...]
    sites: tuple[AttentionSiteSpec, ...]
    schema: FamilyCacheSchema

    def validate(self) -> "FamilyCacheRegistration":
        validate_registration(self.domains, self.sites, self.schema)
        return self


def _decoder_domain(
    *,
    layer_count: int,
    kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    page_tokens: int,
    shareability: Shareability,
) -> CacheDomainSpec:
    return CacheDomainSpec(
        domain_id=_DECODER_DOMAIN,
        kind=CacheStateKind.MHA_HISTORY,
        layer_ids=tuple(range(layer_count)),
        layout=MhaHistoryLayout(
            kv_heads=kv_heads,
            qk_head_dim=qk_head_dim,
            value_head_dim=value_head_dim,
            compute_dtype=DTypeTag.BF16,
            store_format=StoreFormat.BF16,
            page_order=PageOrder.LAYER_MAJOR,
        ),
        page_tokens=page_tokens,
        retention=RetentionPolicy.FULL_HISTORY,
        placement=PlacementConstraint.engine_primary(),
        shareability=shareability,
    )


def _text_region(
    local_region_id: int,
    *,
    route_id: int = _TEXT_ROUTE,
) -> CacheRegionSpec:
    """Ordinary text over the primary sequence: reserve and commit every row."""

    return CacheRegionSpec(
        operation_tag=OperationTag.SEQUENCE_STEP,
        local_region_id=local_region_id,
        multiplicity=RegionMultiplicity.ONE,
        role=CacheRoleSelector.fixed(PRIMARY_ROLE),
        route_runs=(RouteRunSpec(route_id=route_id, extent=_TOKENS),),
        query_rows=_TOKENS,
        reserve_rows=_TOKENS,
        commit_rows=CommitExpr.all_reserved(),
        cache_effect=CacheEffect.PERSISTENT_APPEND,
        attention_pattern=AttentionPattern.CAUSAL_PREFIX,
        position=PositionExpr.linear_from_cursor(0),
        logical_position_commit=CommitExpr.all_reserved(),
    )


def _verification_region(local_region_id: int) -> CacheRegionSpec:
    """Target verification: reserve the full candidate tail, commit the
    accepted prefix."""

    return CacheRegionSpec(
        operation_tag=OperationTag.SEQUENCE_STEP,
        local_region_id=local_region_id,
        multiplicity=RegionMultiplicity.ONE,
        role=CacheRoleSelector.fixed(PRIMARY_ROLE),
        route_runs=(RouteRunSpec(route_id=_TEXT_ROUTE, extent=_CANDIDATES),),
        query_rows=_CANDIDATES,
        reserve_rows=_CANDIDATES,
        commit_rows=CommitExpr.accepted_candidate_prefix(),
        cache_effect=CacheEffect.TENTATIVE_APPEND,
        attention_pattern=AttentionPattern.CAUSAL_PREFIX,
        position=PositionExpr.linear_from_cursor(0),
        logical_position_commit=CommitExpr.accepted_candidate_prefix(),
    )


def _branch_role(role_id: int) -> CacheRoleSpec:
    """CFG conditioning history: opened before the first flow step, closed
    after the materialize commit or session termination."""

    return CacheRoleSpec(
        role_id=role_id,
        domain_id=_DECODER_DOMAIN,
        lifetime=CacheLifetime.BRANCH,
        initialization=RoleInitialization.empty_when(
            BranchOpenExpr(operation_tag=OperationTag.FLOW_STEP)
        ),
        release=RoleRelease.release_when(
            BranchCloseExpr(operation_tag=OperationTag.MATERIALIZE_STEP)
        ),
        publication=PrefixPublication.NEVER,
    )


def qwen3_cache_registration(
    *,
    layer_count: int,
    query_heads: int,
    kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    page_tokens: int = 16,
) -> FamilyCacheRegistration:
    """One full-history MHA domain, one causal site, one primary role."""

    domain = _decoder_domain(
        layer_count=layer_count,
        kv_heads=kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        page_tokens=page_tokens,
        shareability=Shareability.EXACT_COMPLETE_PAGES,
    )
    site = AttentionSiteSpec(
        site_id=1,
        cache_domain_id=domain.domain_id,
        query_heads=query_heads,
        kv_heads=kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        scale=qk_head_dim ** -0.5,
        position_schema=PositionSchema.LINEAR,
        visibility_domain=frozenset({AttentionPattern.CAUSAL_PREFIX}),
    )
    schema = FamilyCacheSchema(
        roles=(
            CacheRoleSpec(
                role_id=PRIMARY_ROLE,
                domain_id=domain.domain_id,
                lifetime=CacheLifetime.REQUEST,
                initialization=RoleInitialization.adopt_exact_prefix(0),
                release=RoleRelease.on_request_drop(),
                publication=PrefixPublication.EXACT_COMPLETE_PAGES,
            ),
        ),
        regions=(
            _text_region(0),
            _verification_region(1),
        ),
    )
    return FamilyCacheRegistration(
        domains=(domain,),
        sites=(site,),
        schema=schema,
    ).validate()


def _image_generation_schema(
    *,
    marker_runs: bool,
    position: PositionExpr,
    feedback_position: PositionExpr,
) -> FamilyCacheSchema:
    """Shared BAGEL/SenseNova lowering: text decode, CFG denoise overlays, and
    persistent generated-image feedback."""

    branch_roles = (
        CacheRoleSelector.current_cfg_branch(
            (PRIMARY_ROLE, TEXT_UNCONDITIONAL_ROLE, IMAGE_UNCONDITIONAL_ROLE)
        )
    )
    denoise_runs: tuple[RouteRunSpec, ...]
    if marker_runs:
        marker = ExtentExpr.literal(1)
        denoise_runs = (
            RouteRunSpec(route_id=_TEXT_ROUTE, extent=marker),
            RouteRunSpec(route_id=_GENERATION_ROUTE, extent=_IMAGE_TOKENS),
            RouteRunSpec(route_id=_TEXT_ROUTE, extent=marker),
        )
        denoise_rows = ExtentExpr(constant=2, terms=_IMAGE_TOKENS.terms)
    else:
        denoise_runs = (
            RouteRunSpec(route_id=_GENERATION_ROUTE, extent=_IMAGE_TOKENS),
        )
        denoise_rows = _IMAGE_TOKENS
    return FamilyCacheSchema(
        roles=(
            CacheRoleSpec(
                role_id=PRIMARY_ROLE,
                domain_id=_DECODER_DOMAIN,
                lifetime=CacheLifetime.REQUEST,
                initialization=RoleInitialization.empty_on_admission(),
                release=RoleRelease.on_request_drop(),
                publication=PrefixPublication.NEVER,
            ),
            _branch_role(TEXT_UNCONDITIONAL_ROLE),
            _branch_role(IMAGE_UNCONDITIONAL_ROLE),
        ),
        regions=(
            _text_region(0),
            CacheRegionSpec(
                operation_tag=OperationTag.FLOW_STEP,
                local_region_id=0,
                multiplicity=RegionMultiplicity.CFG_BRANCHES,
                role=branch_roles,
                route_runs=denoise_runs,
                query_rows=denoise_rows,
                reserve_rows=denoise_rows,
                commit_rows=CommitExpr.zero(),
                cache_effect=CacheEffect.TRANSIENT_OVERLAY,
                attention_pattern=AttentionPattern.FULL_QUERY_PREFIX,
                position=position,
                logical_position_commit=CommitExpr.zero(),
            ),
            CacheRegionSpec(
                operation_tag=OperationTag.SEQUENCE_STEP,
                local_region_id=1,
                multiplicity=RegionMultiplicity.ONE,
                role=CacheRoleSelector.fixed(PRIMARY_ROLE),
                route_runs=(
                    RouteRunSpec(route_id=_GENERATION_ROUTE, extent=_IMAGE_TOKENS),
                ),
                query_rows=_IMAGE_TOKENS,
                reserve_rows=_IMAGE_TOKENS,
                commit_rows=CommitExpr.all_reserved(),
                cache_effect=CacheEffect.PERSISTENT_APPEND,
                attention_pattern=AttentionPattern.FULL_QUERY_PREFIX,
                position=feedback_position,
                logical_position_commit=CommitExpr(
                    kind=CommitExprKind.RESULT_AFFINE,
                    result_affine=ExtentExpr.literal(1),
                ),
            ),
        ),
    )


def bagel_cache_registration(
    *,
    layer_count: int,
    query_heads: int,
    kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    page_tokens: int = 16,
) -> FamilyCacheRegistration:
    """One decoder domain shared by understanding and generation routes,
    one-axis positions, and marker/body/marker denoise route runs."""

    domain = _decoder_domain(
        layer_count=layer_count,
        kv_heads=kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        page_tokens=page_tokens,
        shareability=Shareability.PRIVATE,
    )
    site = AttentionSiteSpec(
        site_id=1,
        cache_domain_id=domain.domain_id,
        query_heads=query_heads,
        kv_heads=kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        scale=qk_head_dim ** -0.5,
        position_schema=PositionSchema.LINEAR,
        visibility_domain=frozenset(
            {AttentionPattern.CAUSAL_PREFIX, AttentionPattern.FULL_QUERY_PREFIX}
        ),
    )
    schema = _image_generation_schema(
        marker_runs=True,
        position=PositionExpr.constant_from_cursor(0),
        feedback_position=PositionExpr.constant_from_cursor(0),
    )
    return FamilyCacheRegistration(
        domains=(domain,),
        sites=(site,),
        schema=schema,
    ).validate()


def sensenova_cache_registration(
    *,
    layer_count: int,
    query_heads: int,
    kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    page_tokens: int = 16,
) -> FamilyCacheRegistration:
    """Compatible decoder domain with three-axis temporal-grid positions and
    visible-end coverage; full-history attention on every decoder layer per the
    frozen conformance profile."""

    domain = _decoder_domain(
        layer_count=layer_count,
        kv_heads=kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        page_tokens=page_tokens,
        shareability=Shareability.PRIVATE,
    )
    site = AttentionSiteSpec(
        site_id=1,
        cache_domain_id=domain.domain_id,
        query_heads=query_heads,
        kv_heads=kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        scale=qk_head_dim ** -0.5,
        position_schema=PositionSchema.TEMPORAL_GRID,
        visibility_domain=frozenset(
            {
                AttentionPattern.CAUSAL_PREFIX,
                AttentionPattern.FULL_QUERY_PREFIX,
                AttentionPattern.VISIBLE_END,
            }
        ),
    )
    grid = PositionExpr.temporal_grid_from_operation(0, 1, 2, GridFieldTag.IMAGE_GRID)
    schema = _image_generation_schema(
        marker_runs=False,
        position=grid,
        feedback_position=grid,
    )
    return FamilyCacheRegistration(
        domains=(domain,),
        sites=(site,),
        schema=schema,
    ).validate()
