"""Closed cache-domain, attention-site, and family cache-lowering contracts.

These types extend model-family registration with declarative cache data:

* :class:`CacheDomainSpec` — static physical compatibility of one cache state
  family (layers, layout, page geometry, retention, placement, shareability).
* :class:`AttentionSiteSpec` — the static binding of resident attention layers
  to one cache domain, geometry, position schema, and visibility domain.
* :class:`FamilyCacheSchema` — closed, bounded lowering data
  (:class:`CacheRoleSpec` / :class:`CacheRegionSpec`) interpreted by one engine
  lowerer. It is declarative data, never a callback or execution IR.
* :class:`CacheSequenceRef` — the engine-private versioned reference to one
  logical cache history. It never crosses the worker wire.

The module is torch-free so validation and fingerprinting run anywhere.

Every enum carries canonical integer tags shared with protocol peers;
:func:`registration_fingerprint` serializes declarations by field order, so a
tag or field-order change is a contract change by construction.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, fields
from enum import IntEnum

from .execution import EngineRef, SessionRef
from .operations import OperationTag

__all__ = [
    "NO_CACHE_DOMAIN",
    "AttentionPattern",
    "AttentionSiteSpec",
    "BranchCloseExpr",
    "BranchOpenExpr",
    "CacheDomainSpec",
    "CacheEffect",
    "CacheLifetime",
    "CacheRegionSpec",
    "CacheRoleSelector",
    "CacheRoleSpec",
    "CacheSchemaError",
    "CacheSequenceRef",
    "CacheStateKind",
    "CommitExpr",
    "CommitExprKind",
    "DTypeTag",
    "EngineRef",
    "ExtentExpr",
    "ExtentOperand",
    "FamilyCacheRegistration",
    "FamilyCacheSchema",
    "GridFieldTag",
    "MhaHistoryLayout",
    "OperationTag",
    "PageOrder",
    "PlacementConstraint",
    "PlacementKind",
    "PositionExpr",
    "PositionExprKind",
    "PositionSchema",
    "PrefixPublication",
    "RegionMultiplicity",
    "ResultPredicateTag",
    "RetentionPolicy",
    "RoleInitialization",
    "RoleInitializationKind",
    "RoleRelease",
    "RoleReleaseKind",
    "RoleSelectorKind",
    "RouteRunSpec",
    "SessionRef",
    "Shareability",
    "StoreFormat",
    "registration_fingerprint",
    "validate_registration",
]


class CacheSchemaError(ValueError):
    """A cache registration violates the closed contract."""


# --------------------------------------------------------------------------- #
# Closed enums (canonical integer tags; shared with the Rust contract).
# --------------------------------------------------------------------------- #


class CacheStateKind(IntEnum):
    MHA_HISTORY = 1


class RetentionPolicy(IntEnum):
    FULL_HISTORY = 1


class DTypeTag(IntEnum):
    BF16 = 1
    FP16 = 2
    FP32 = 3


class StoreFormat(IntEnum):
    BF16 = 1
    FP16 = 2
    FP8_E4M3_FROZEN_BLOCK_SCALE = 3


class PageOrder(IntEnum):
    LAYER_MAJOR = 1
    PAGE_MAJOR = 2


class Shareability(IntEnum):
    PRIVATE = 1
    EXACT_COMPLETE_PAGES = 2


class PrefixPublication(IntEnum):
    NEVER = 1
    EXACT_COMPLETE_PAGES = 2


class AttentionPattern(IntEnum):
    CAUSAL_PREFIX = 1
    FULL_QUERY_PREFIX = 2
    VISIBLE_END = 3


class CacheEffect(IntEnum):
    READ_ONLY = 0
    PERSISTENT_APPEND = 1
    TRANSIENT_OVERLAY = 2
    TENTATIVE_APPEND = 3


class CacheLifetime(IntEnum):
    REQUEST = 1
    BRANCH = 2


class PositionSchema(IntEnum):
    LINEAR = 1
    CONSTANT = 2
    TEMPORAL_GRID = 3
    PRODUCT_TABLE = 4


class RegionMultiplicity(IntEnum):
    ONE = 1
    CFG_BRANCHES = 2


class GridFieldTag(IntEnum):
    """Closed operation-field tags naming a typed temporal grid."""

    IMAGE_GRID = 1


class ResultPredicateTag(IntEnum):
    """Closed predicates declared by parent result variants."""

    FLOW_SCHEDULE_COMPLETE = 1


class PlacementKind(IntEnum):
    ENGINE_PRIMARY = 1
    ROUTE_COORDINATE = 2


NO_CACHE_DOMAIN = 0


# --------------------------------------------------------------------------- #
# Static registration data.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PlacementConstraint:
    """Compatibility requirement inside one logical engine, not a device."""

    kind: PlacementKind
    mesh_axis_id: int = 0
    coordinate: int = 0

    @classmethod
    def engine_primary(cls) -> "PlacementConstraint":
        return cls(kind=PlacementKind.ENGINE_PRIMARY)

    @classmethod
    def route_coordinate(cls, mesh_axis_id: int, coordinate: int) -> "PlacementConstraint":
        return cls(
            kind=PlacementKind.ROUTE_COORDINATE,
            mesh_axis_id=int(mesh_axis_id),
            coordinate=int(coordinate),
        )


@dataclass(frozen=True, slots=True)
class MhaHistoryLayout:
    """The initial ``CacheLayout`` variant: full-history MHA pages.

    Layout contains no capacity, allocator, or provider — those belong to
    residency and the configured attention backend.
    """

    kv_heads: int
    qk_head_dim: int
    value_head_dim: int
    compute_dtype: DTypeTag
    store_format: StoreFormat
    page_order: PageOrder


CacheLayout = MhaHistoryLayout


@dataclass(frozen=True, slots=True)
class CacheDomainSpec:
    """Static description of one physically compatible cache state family."""

    domain_id: int
    kind: CacheStateKind
    layer_ids: tuple[int, ...]
    layout: CacheLayout
    page_tokens: int
    retention: RetentionPolicy
    placement: PlacementConstraint
    shareability: Shareability


@dataclass(frozen=True, slots=True)
class AttentionSiteSpec:
    """Static binding of resident attention layers to one cache domain.

    ``cache_domain_id`` is :data:`NO_CACHE_DOMAIN` only when the site has no
    persistent or transient cache.
    """

    site_id: int
    cache_domain_id: int
    query_heads: int
    kv_heads: int
    qk_head_dim: int
    value_head_dim: int
    scale: float
    position_schema: PositionSchema
    visibility_domain: frozenset[AttentionPattern]


# --------------------------------------------------------------------------- #
# Closed expression algebra.
# --------------------------------------------------------------------------- #


class ExtentOperand(IntEnum):
    """Typed operation fields an extent expression may reference."""

    INPUT_TOKEN_COUNT = 1
    IMAGE_TOKEN_COUNT = 2
    CANDIDATE_COUNT = 3
    PRODUCT_ROW_COUNT = 4
    CFG_BRANCH_COUNT = 5


@dataclass(frozen=True, slots=True)
class ExtentExpr:
    """Bounded affine expression over typed operation fields.

    ``constant + sum(coefficient * operand)``. It cannot inspect a model,
    session dictionary, tensor value, provider, or physical allocation.
    """

    constant: int = 0
    terms: tuple[tuple[ExtentOperand, int], ...] = ()

    @classmethod
    def literal(cls, value: int) -> "ExtentExpr":
        return cls(constant=int(value))

    @classmethod
    def operand(cls, operand: ExtentOperand, coefficient: int = 1) -> "ExtentExpr":
        return cls(terms=((operand, int(coefficient)),))

    def evaluate(self, operands: dict[ExtentOperand, int]) -> int:
        total = int(self.constant)
        for operand, coefficient in self.terms:
            if operand not in operands:
                raise CacheSchemaError(
                    f"extent expression references missing operand {operand.name}"
                )
            total += int(coefficient) * int(operands[operand])
        return total

    def canonical_terms(self) -> dict[ExtentOperand, int]:
        merged: dict[ExtentOperand, int] = {}
        for operand, coefficient in self.terms:
            merged[operand] = merged.get(operand, 0) + int(coefficient)
        return {operand: value for operand, value in merged.items() if value != 0}


ZERO_EXTENT = ExtentExpr()


class CommitExprKind(IntEnum):
    ZERO = 1
    ALL_RESERVED = 2
    ACCEPTED_CANDIDATE_PREFIX = 3
    RESULT_AFFINE = 4


@dataclass(frozen=True, slots=True)
class CommitExpr:
    """Closed result-dependent commit expression.

    Evaluated only after result validation; for every result
    ``0 <= commit_rows <= reserve_rows``.
    """

    kind: CommitExprKind
    result_affine: ExtentExpr | None = None

    @classmethod
    def zero(cls) -> "CommitExpr":
        return cls(kind=CommitExprKind.ZERO)

    @classmethod
    def all_reserved(cls) -> "CommitExpr":
        return cls(kind=CommitExprKind.ALL_RESERVED)

    @classmethod
    def accepted_candidate_prefix(cls) -> "CommitExpr":
        return cls(kind=CommitExprKind.ACCEPTED_CANDIDATE_PREFIX)


class PositionExprKind(IntEnum):
    LINEAR_FROM_CURSOR = 1
    CONSTANT_FROM_CURSOR = 2
    TEMPORAL_GRID_FROM_OPERATION = 3
    PRODUCT_POSITION_TABLE = 4


@dataclass(frozen=True, slots=True)
class PositionExpr:
    """Closed position lowering with canonical integer tags and fixed fields."""

    kind: PositionExprKind
    axis: int = 0
    step: int = 0
    t_axis: int = 0
    h_axis: int = 0
    w_axis: int = 0
    grid_field_tag: GridFieldTag | None = None
    product_slot: int = 0
    axis_count: int = 0

    @classmethod
    def linear_from_cursor(cls, axis: int, step: int = 1) -> "PositionExpr":
        return cls(kind=PositionExprKind.LINEAR_FROM_CURSOR, axis=int(axis), step=int(step))

    @classmethod
    def constant_from_cursor(cls, axis: int) -> "PositionExpr":
        return cls(kind=PositionExprKind.CONSTANT_FROM_CURSOR, axis=int(axis))

    @classmethod
    def temporal_grid_from_operation(
        cls,
        t_axis: int,
        h_axis: int,
        w_axis: int,
        grid_field_tag: GridFieldTag,
    ) -> "PositionExpr":
        return cls(
            kind=PositionExprKind.TEMPORAL_GRID_FROM_OPERATION,
            t_axis=int(t_axis),
            h_axis=int(h_axis),
            w_axis=int(w_axis),
            grid_field_tag=grid_field_tag,
        )

    @classmethod
    def product_position_table(cls, product_slot: int, axis_count: int) -> "PositionExpr":
        return cls(
            kind=PositionExprKind.PRODUCT_POSITION_TABLE,
            product_slot=int(product_slot),
            axis_count=int(axis_count),
        )

    def position_schema(self) -> PositionSchema:
        return {
            PositionExprKind.LINEAR_FROM_CURSOR: PositionSchema.LINEAR,
            PositionExprKind.CONSTANT_FROM_CURSOR: PositionSchema.CONSTANT,
            PositionExprKind.TEMPORAL_GRID_FROM_OPERATION: PositionSchema.TEMPORAL_GRID,
            PositionExprKind.PRODUCT_POSITION_TABLE: PositionSchema.PRODUCT_TABLE,
        }[self.kind]


# --------------------------------------------------------------------------- #
# Role lifecycle expressions.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BranchOpenExpr:
    """``ROLE_ABSENT_AND_OPERATION(operation_tag)``."""

    operation_tag: OperationTag


@dataclass(frozen=True, slots=True)
class BranchCloseExpr:
    """``COMMITTED_OPERATION`` or ``VALIDATED_RESULT`` over parent results."""

    operation_tag: OperationTag
    result_predicate: ResultPredicateTag | None = None


class RoleInitializationKind(IntEnum):
    EMPTY_ON_ADMISSION = 1
    ADOPT_EXACT_PREFIX = 2
    FORK_ROLE = 3
    EMPTY_WHEN = 4


@dataclass(frozen=True, slots=True)
class RoleInitialization:
    kind: RoleInitializationKind
    input_lease_slot: int = 0
    source_role_id: int = 0
    open_expr: BranchOpenExpr | None = None

    @classmethod
    def empty_on_admission(cls) -> "RoleInitialization":
        return cls(kind=RoleInitializationKind.EMPTY_ON_ADMISSION)

    @classmethod
    def adopt_exact_prefix(cls, input_lease_slot: int) -> "RoleInitialization":
        return cls(
            kind=RoleInitializationKind.ADOPT_EXACT_PREFIX,
            input_lease_slot=int(input_lease_slot),
        )

    @classmethod
    def fork_role(cls, source_role_id: int) -> "RoleInitialization":
        return cls(kind=RoleInitializationKind.FORK_ROLE, source_role_id=int(source_role_id))

    @classmethod
    def empty_when(cls, open_expr: BranchOpenExpr) -> "RoleInitialization":
        return cls(kind=RoleInitializationKind.EMPTY_WHEN, open_expr=open_expr)


class RoleReleaseKind(IntEnum):
    ON_REQUEST_DROP = 1
    ON_SESSION_TERMINAL = 2
    RELEASE_WHEN = 3


@dataclass(frozen=True, slots=True)
class RoleRelease:
    kind: RoleReleaseKind
    close_expr: BranchCloseExpr | None = None

    @classmethod
    def on_request_drop(cls) -> "RoleRelease":
        return cls(kind=RoleReleaseKind.ON_REQUEST_DROP)

    @classmethod
    def on_session_terminal(cls) -> "RoleRelease":
        return cls(kind=RoleReleaseKind.ON_SESSION_TERMINAL)

    @classmethod
    def release_when(cls, close_expr: BranchCloseExpr) -> "RoleRelease":
        return cls(kind=RoleReleaseKind.RELEASE_WHEN, close_expr=close_expr)


# --------------------------------------------------------------------------- #
# Family cache schema.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CacheRoleSpec:
    """Family-level semantic cache role; it is not a pool name."""

    role_id: int
    domain_id: int
    lifetime: CacheLifetime
    initialization: RoleInitialization
    release: RoleRelease
    publication: PrefixPublication


class RoleSelectorKind(IntEnum):
    FIXED = 1
    CURRENT_CFG_BRANCH = 2
    NO_CACHE = 3


@dataclass(frozen=True, slots=True)
class CacheRoleSelector:
    kind: RoleSelectorKind
    role_id: int = 0
    branch_role_ids: tuple[int, ...] = ()

    @classmethod
    def fixed(cls, role_id: int) -> "CacheRoleSelector":
        return cls(kind=RoleSelectorKind.FIXED, role_id=int(role_id))

    @classmethod
    def current_cfg_branch(cls, branch_role_ids: tuple[int, ...]) -> "CacheRoleSelector":
        return cls(
            kind=RoleSelectorKind.CURRENT_CFG_BRANCH,
            branch_role_ids=tuple(int(role) for role in branch_role_ids),
        )

    @classmethod
    def no_cache(cls) -> "CacheRoleSelector":
        return cls(kind=RoleSelectorKind.NO_CACHE)

    def referenced_role_ids(self) -> tuple[int, ...]:
        if self.kind is RoleSelectorKind.FIXED:
            return (self.role_id,)
        if self.kind is RoleSelectorKind.CURRENT_CFG_BRANCH:
            return self.branch_role_ids
        return ()


@dataclass(frozen=True, slots=True)
class RouteRunSpec:
    """One statically named resident route span inside an attention region."""

    route_id: int
    extent: ExtentExpr


@dataclass(frozen=True, slots=True)
class CacheRegionSpec:
    """Cache, route-run, position, visibility, and cursor consequences of one
    operation region. The same expressions derive segments, capacity demand,
    cache reservation, and expected session deltas."""

    operation_tag: OperationTag
    local_region_id: int
    multiplicity: RegionMultiplicity
    role: CacheRoleSelector
    route_runs: tuple[RouteRunSpec, ...]
    query_rows: ExtentExpr
    reserve_rows: ExtentExpr
    commit_rows: CommitExpr
    cache_effect: CacheEffect
    attention_pattern: AttentionPattern
    position: PositionExpr
    logical_position_commit: CommitExpr


@dataclass(frozen=True, slots=True)
class FamilyCacheSchema:
    roles: tuple[CacheRoleSpec, ...]
    regions: tuple[CacheRegionSpec, ...]


# --------------------------------------------------------------------------- #
# Logical cache sequences (engine-private; never serialized to the scheduler).
# Engine/session identity and the sealed operation tag are owned by
# `contracts.execution` (the parent host contract) and re-exported here for
# the cache companion's public API.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CacheSequenceRef:
    """Engine-private versioned reference to one logical cache history.

    ``sequence_id`` is not a physical page, pool row, pointer, or allocator
    index. ``generation`` prevents stale reuse after release or replacement.
    """

    engine: EngineRef
    session: SessionRef
    sequence_id: int
    generation: int
    domain_id: int
    role_id: int
    committed_rows: int
    lifetime: CacheLifetime


# --------------------------------------------------------------------------- #
# Registration validation.
# --------------------------------------------------------------------------- #


def validate_registration(
    domains: tuple[CacheDomainSpec, ...],
    sites: tuple[AttentionSiteSpec, ...],
    schema: FamilyCacheSchema,
) -> None:
    """Prove the closed registration invariants the contract level can check.

    Raises :class:`CacheSchemaError` on the first violation. Provider coverage,
    residency completeness, rank fingerprint agreement, and capacity dominance
    are engine/startup obligations validated at their owning seams.
    """

    domain_by_id = _validate_domains(domains)
    _validate_sites(sites, domain_by_id)
    role_by_id = _validate_roles(schema.roles, domain_by_id)
    _validate_regions(schema.regions, role_by_id, domain_by_id)


def _validate_domains(
    domains: tuple[CacheDomainSpec, ...],
) -> dict[int, CacheDomainSpec]:
    domain_by_id: dict[int, CacheDomainSpec] = {}
    owned_layers: dict[int, int] = {}
    for domain in domains:
        if domain.domain_id <= NO_CACHE_DOMAIN:
            raise CacheSchemaError(f"cache domain ids must be positive; got {domain.domain_id}")
        if domain.domain_id in domain_by_id:
            raise CacheSchemaError(f"duplicate cache domain id {domain.domain_id}")
        if domain.page_tokens <= 0:
            raise CacheSchemaError(f"cache domain {domain.domain_id} page_tokens must be positive")
        if not domain.layer_ids:
            raise CacheSchemaError(f"cache domain {domain.domain_id} owns no layers")
        for layer_id in domain.layer_ids:
            if layer_id in owned_layers:
                raise CacheSchemaError(
                    f"layer {layer_id} belongs to domains "
                    f"{owned_layers[layer_id]} and {domain.domain_id}; "
                    "every cache-writing layer has exactly one domain"
                )
            owned_layers[layer_id] = domain.domain_id
        domain_by_id[domain.domain_id] = domain
    return domain_by_id


def _validate_sites(
    sites: tuple[AttentionSiteSpec, ...],
    domain_by_id: dict[int, CacheDomainSpec],
) -> None:
    seen_sites: set[int] = set()
    for site in sites:
        if site.site_id in seen_sites:
            raise CacheSchemaError(f"duplicate attention site id {site.site_id}")
        seen_sites.add(site.site_id)
        if not site.visibility_domain:
            raise CacheSchemaError(f"attention site {site.site_id} advertises no attention pattern")
        if site.cache_domain_id == NO_CACHE_DOMAIN:
            continue
        domain = domain_by_id.get(site.cache_domain_id)
        if domain is None:
            raise CacheSchemaError(
                f"attention site {site.site_id} references unknown cache domain "
                f"{site.cache_domain_id}"
            )
        layout = domain.layout
        if (
            site.kv_heads != layout.kv_heads
            or site.qk_head_dim != layout.qk_head_dim
            or site.value_head_dim != layout.value_head_dim
        ):
            raise CacheSchemaError(
                f"attention site {site.site_id} geometry does not match cache "
                f"domain {domain.domain_id} layout"
            )


def _validate_roles(
    roles: tuple[CacheRoleSpec, ...],
    domain_by_id: dict[int, CacheDomainSpec],
) -> dict[int, CacheRoleSpec]:
    role_by_id: dict[int, CacheRoleSpec] = {}
    for role in roles:
        if role.role_id in role_by_id:
            raise CacheSchemaError(f"duplicate cache role id {role.role_id}")
        domain = domain_by_id.get(role.domain_id)
        if domain is None:
            raise CacheSchemaError(
                f"cache role {role.role_id} references unknown domain {role.domain_id}"
            )
        if (
            role.publication is PrefixPublication.EXACT_COMPLETE_PAGES
            and domain.shareability is not Shareability.EXACT_COMPLETE_PAGES
        ):
            raise CacheSchemaError(
                f"cache role {role.role_id} publishes prefixes but domain "
                f"{domain.domain_id} is not shareable"
            )
        if role.lifetime is CacheLifetime.BRANCH:
            if role.initialization.kind not in (
                RoleInitializationKind.EMPTY_WHEN,
                RoleInitializationKind.FORK_ROLE,
            ):
                raise CacheSchemaError(
                    f"branch-lifetime role {role.role_id} needs an open expression or fork source"
                )
            if role.release.kind is not RoleReleaseKind.RELEASE_WHEN:
                raise CacheSchemaError(
                    f"branch-lifetime role {role.role_id} needs a close expression"
                )
        if (
            role.initialization.kind is RoleInitializationKind.EMPTY_WHEN
            and role.initialization.open_expr is None
        ):
            raise CacheSchemaError(
                f"role {role.role_id} EMPTY_WHEN initialization has no open expression"
            )
        if role.release.kind is RoleReleaseKind.RELEASE_WHEN and role.release.close_expr is None:
            raise CacheSchemaError(
                f"role {role.role_id} RELEASE_WHEN release has no close expression"
            )
        role_by_id[role.role_id] = role
    for role in roles:
        if role.initialization.kind is RoleInitializationKind.FORK_ROLE:
            if role.initialization.source_role_id not in role_by_id:
                raise CacheSchemaError(
                    f"role {role.role_id} forks unknown role {role.initialization.source_role_id}"
                )
    return role_by_id


def _validate_regions(
    regions: tuple[CacheRegionSpec, ...],
    role_by_id: dict[int, CacheRoleSpec],
    domain_by_id: dict[int, CacheDomainSpec],
) -> None:
    seen_regions: set[tuple[OperationTag, int]] = set()
    for region in regions:
        key = (region.operation_tag, region.local_region_id)
        if key in seen_regions:
            raise CacheSchemaError(
                f"duplicate region {region.local_region_id} for operation "
                f"{region.operation_tag.name}"
            )
        seen_regions.add(key)
        _validate_region_roles(region, role_by_id)
        _validate_region_effect(region)
        _validate_route_partition(region)


def _validate_region_roles(
    region: CacheRegionSpec,
    role_by_id: dict[int, CacheRoleSpec],
) -> None:
    if region.role.kind is RoleSelectorKind.NO_CACHE:
        if region.cache_effect is not CacheEffect.READ_ONLY:
            raise CacheSchemaError(
                f"region {region.local_region_id}: NO_CACHE regions must be READ_ONLY"
            )
        return
    for role_id in region.role.referenced_role_ids():
        if role_id not in role_by_id:
            raise CacheSchemaError(
                f"region {region.local_region_id} references unknown role {role_id}"
            )
    if (
        region.role.kind is RoleSelectorKind.CURRENT_CFG_BRANCH
        and region.multiplicity is not RegionMultiplicity.CFG_BRANCHES
    ):
        raise CacheSchemaError(
            f"region {region.local_region_id}: branch role selection requires "
            "CFG_BRANCHES multiplicity"
        )


def _validate_region_effect(region: CacheRegionSpec) -> None:
    effect = region.cache_effect
    if effect is CacheEffect.READ_ONLY:
        if region.reserve_rows.canonical_terms() or region.reserve_rows.constant != 0:
            raise CacheSchemaError(
                f"region {region.local_region_id}: READ_ONLY regions reserve no rows"
            )
        if region.commit_rows.kind is not CommitExprKind.ZERO:
            raise CacheSchemaError(
                f"region {region.local_region_id}: READ_ONLY regions commit zero rows"
            )
    elif effect is CacheEffect.TRANSIENT_OVERLAY:
        if region.commit_rows.kind is not CommitExprKind.ZERO:
            raise CacheSchemaError(
                f"region {region.local_region_id}: transient effects commit zero cache rows"
            )
    elif effect is CacheEffect.TENTATIVE_APPEND:
        if region.commit_rows.kind not in (
            CommitExprKind.ACCEPTED_CANDIDATE_PREFIX,
            CommitExprKind.RESULT_AFFINE,
        ):
            raise CacheSchemaError(
                f"region {region.local_region_id}: tentative effects use "
                "ACCEPTED_CANDIDATE_PREFIX or a bounded result expression"
            )
    elif effect is CacheEffect.PERSISTENT_APPEND:
        if region.commit_rows.kind is CommitExprKind.ACCEPTED_CANDIDATE_PREFIX:
            raise CacheSchemaError(
                f"region {region.local_region_id}: persistent appends commit "
                "reserved or result-derived rows, not candidate prefixes"
            )
    if (
        region.commit_rows.kind is CommitExprKind.RESULT_AFFINE
        and region.commit_rows.result_affine is None
    ):
        raise CacheSchemaError(
            f"region {region.local_region_id}: RESULT_AFFINE commit has no expression"
        )


def _validate_route_partition(region: CacheRegionSpec) -> None:
    """Route-run extents must exactly partition the region's query rows."""

    if not region.route_runs:
        raise CacheSchemaError(f"region {region.local_region_id} declares no route runs")
    total_constant = sum(run.extent.constant for run in region.route_runs)
    total_terms: dict[ExtentOperand, int] = {}
    for run in region.route_runs:
        for operand, coefficient in run.extent.canonical_terms().items():
            total_terms[operand] = total_terms.get(operand, 0) + coefficient
    total_terms = {op: value for op, value in total_terms.items() if value != 0}
    if (
        total_constant != region.query_rows.constant
        or total_terms != region.query_rows.canonical_terms()
    ):
        raise CacheSchemaError(
            f"region {region.local_region_id}: route runs do not partition query rows"
        )


@dataclass(frozen=True, slots=True)
class FamilyCacheRegistration:
    """One family's validated static cache registration.

    The record shape is family-neutral contract data; concrete family
    instances are authored beside their family implementations in
    ``models/cache_registrations.py``.
    """

    domains: tuple[CacheDomainSpec, ...]
    sites: tuple[AttentionSiteSpec, ...]
    schema: FamilyCacheSchema

    def validate(self) -> "FamilyCacheRegistration":
        validate_registration(self.domains, self.sites, self.schema)
        return self


# --------------------------------------------------------------------------- #
# Registration fingerprint.
# --------------------------------------------------------------------------- #


def _canonical_encode(value: object, out: list[str]) -> None:
    if isinstance(value, IntEnum):
        out.append(f"e{value.__class__.__name__}:{int(value)}")
    elif isinstance(value, bool):
        out.append(f"b{int(value)}")
    elif isinstance(value, int):
        out.append(f"i{value}")
    elif isinstance(value, float):
        out.append(f"f{value!r}")
    elif isinstance(value, str):
        out.append(f"s{len(value)}:{value}")
    elif value is None:
        out.append("n")
    elif isinstance(value, frozenset):
        out.append(f"F{len(value)}[")
        for item in sorted(value, key=int):
            _canonical_encode(item, out)
        out.append("]")
    elif isinstance(value, tuple):
        out.append(f"T{len(value)}[")
        for item in value:
            _canonical_encode(item, out)
        out.append("]")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        out.append(f"D{value.__class__.__name__}[")
        for field in fields(value):
            _canonical_encode(getattr(value, field.name), out)
        out.append("]")
    else:
        raise CacheSchemaError(
            f"value of type {type(value).__name__} cannot join a registration fingerprint"
        )


def registration_fingerprint(
    domains: tuple[CacheDomainSpec, ...],
    sites: tuple[AttentionSiteSpec, ...],
    schema: FamilyCacheSchema,
) -> str:
    """Serialize every declaration by field order into one stable digest.

    Domain, role, lowering, layout, placement, position, and publication
    declarations join the all-rank configuration hash through this value.
    """

    out: list[str] = []
    _canonical_encode(tuple(domains), out)
    _canonical_encode(tuple(sites), out)
    _canonical_encode(schema, out)
    return hashlib.sha256("".join(out).encode("utf-8")).hexdigest()
