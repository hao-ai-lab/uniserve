"""Schema-driven lowering: sealed operations into segments and reservations.

Dormant Stage 4 deliverable from ``specs/unified_forward_execution.md``,
implementing the cache companion's central rule: *the same region
specifications derive segments, capacity demand, cache reservation, and
expected session deltas* — one closed data source
(:class:`~uniserve_worker.contracts.cache_schema.FamilyCacheRegistration`)
interpreted by one engine lowerer, never per-family formulas that could
disagree.

Given one transaction's rows (sealed operations plus each row's role-to-
sequence bindings), lowering:

1. evaluates the family's closed extent expressions against typed operation
   operands (token counts, candidate counts, latent extents from product
   leases, CFG branch counts);
2. expands region multiplicity (one instance per CFG branch), resolves cache
   roles to engine-private sequence references, and stacks same-sequence
   writing regions onto consecutive provisional extents;
3. emits canonical segments in row order and local semantic order, one
   `SequenceBinding` per region instance in the same order; and
4. packs the closed `SegmentTableArrays` and computes the aggregate capacity
   demand used for graph-bucket dominance selection.

The output is validated by the same device-contract validators the graph
buckets use, so host lowering and device schema cannot drift. Nothing in
production routes through this module until the vertical slice activates.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts.cache_schema import (
    CacheEffect,
    CacheRegionSpec,
    ExtentOperand,
    FamilyCacheRegistration,
    RoleSelectorKind,
)
from ..contracts.execution import (
    EncodeStep,
    ExecuteRow,
    FlowStep,
    MaterializeStep,
    OperationTag,
    SequenceStep,
    operation_tag,
)
from ..contracts.residency_batch import ResidencyBatchCapacity
from ..contracts.segment_table import GraphCapacity, SegmentTableArrays
from ..runtime.transactional_residency import (
    ReservationPlan,
    RowDemand,
    SequenceBinding,
)

__all__ = [
    "LoweredBatch",
    "LoweredSegment",
    "LoweringError",
    "RoleSequences",
    "lower_rows",
    "select_capacity",
]


class LoweringError(ValueError):
    """A row cannot be lowered under the family's closed cache schema."""


@dataclass(frozen=True, slots=True)
class RoleSequences:
    """One row's cache-role resolution: role id to current sequence reference.

    The engine owns this mapping (roles live in `RequestSession` leases); the
    lowerer only consumes it. Branch-lifetime roles appear once per role id.
    """

    sequences: dict[int, object]  # role_id -> CacheSequenceRef

    def resolve(self, role_id: int):
        sequence = self.sequences.get(role_id)
        if sequence is None:
            raise LoweringError(f"row resolves no sequence for role {role_id}")
        return sequence


@dataclass(frozen=True, slots=True)
class LoweredSegment:
    """One canonical segment draft (packed spans assigned at fill time)."""

    row_id: int
    operation_tag: OperationTag
    route_id: int
    query_count: int
    context_length: int
    attention_pattern: int
    attention_region_id: int
    kv_group: int
    binding_index: int
    cache_effect: CacheEffect
    cache_write_count: int
    branch_id: int
    branch_count: int
    candidate_count: int
    result_slot: int


@dataclass(frozen=True, slots=True)
class LoweredBatch:
    """Segments, reservation plan, and aggregate demand for one transaction."""

    segments: tuple[LoweredSegment, ...]
    plan: ReservationPlan
    rows: int
    tokens: int
    branches: int
    candidate_tokens: int
    write_token_begins: tuple[int, ...]

    def demand(self, *, page_tokens: int) -> GraphCapacity:
        """The aggregate capacity vector this transaction requires."""

        bindings = sum(len(row.bindings) for row in self.plan.rows)
        page_references = 0
        for row in self.plan.rows:
            for binding in row.bindings:
                total = binding.sequence.committed_rows + binding.reserve_rows
                page_references += -(-total // page_tokens)
        return GraphCapacity(
            rows=self.rows,
            segments=len(self.segments),
            tokens=self.tokens,
            branches=self.branches,
            candidate_tokens=self.candidate_tokens,
            position_axes=1,
            visibility_payload_entries=0,
            residency=ResidencyBatchCapacity(
                bindings=bindings,
                page_references=page_references,
                tokens=self.tokens,
            ),
        )

    def fill_segment_table(self, capacity: GraphCapacity) -> SegmentTableArrays:
        """Pack the canonical table for one bucket (validated by the caller)."""

        if len(self.segments) > capacity.segments:
            raise LoweringError("transaction exceeds the bucket segment capacity")
        columns: dict[str, list[int]] = {
            name: [0] * capacity.segments
            for name in SegmentTableArrays.__dataclass_fields__
        }
        token_cursor = 0
        for index, segment in enumerate(self.segments):
            columns["segment_active"][index] = 1
            columns["row_id"][index] = segment.row_id
            columns["operation_tag"][index] = int(segment.operation_tag)
            columns["route_id"][index] = segment.route_id
            columns["token_begin"][index] = token_cursor
            columns["token_count"][index] = segment.query_count
            columns["query_begin"][index] = token_cursor
            columns["query_count"][index] = segment.query_count
            columns["context_length"][index] = segment.context_length
            columns["position_begin"][index] = token_cursor
            columns["position_count"][index] = segment.query_count
            columns["branch_id"][index] = segment.branch_id
            columns["branch_count"][index] = segment.branch_count
            columns["attention_pattern"][index] = segment.attention_pattern
            columns["attention_region_id"][index] = segment.attention_region_id
            columns["kv_group"][index] = segment.kv_group
            columns["kv_read_index"][index] = segment.binding_index
            columns["kv_write_index"][index] = (
                segment.binding_index
                if segment.cache_effect is not CacheEffect.READ_ONLY
                else 0
            )
            columns["cache_effect"][index] = int(segment.cache_effect)
            columns["cache_write_count"][index] = segment.cache_write_count
            columns["candidate_begin"][index] = 0
            columns["candidate_count"][index] = segment.candidate_count
            columns["result_slot"][index] = segment.result_slot
            token_cursor += segment.query_count
        local: dict[int, int] = {}
        for index, segment in enumerate(self.segments):
            columns["local_segment_id"][index] = local.get(segment.row_id, 0)
            local[segment.row_id] = local.get(segment.row_id, 0) + 1
        return SegmentTableArrays(**columns)


def lower_rows(
    rows: tuple[tuple[ExecuteRow, RoleSequences], ...],
    registration: FamilyCacheRegistration,
) -> LoweredBatch:
    """Lower one transaction's rows under the family's closed cache schema."""

    domain_by_role = {
        role.role_id: role.domain_id for role in registration.schema.roles
    }
    segments: list[LoweredSegment] = []
    demands: list[RowDemand] = []
    write_token_begins: list[int] = []
    token_cursor = 0
    region_counter = 0
    max_branches = 0
    candidate_tokens = 0
    for row, roles in rows:
        operands = _operand_values(row)
        tag = operation_tag(row.operation)
        regions = [
            region
            for region in registration.schema.regions
            if region.operation_tag is tag
        ]
        if not regions:
            raise LoweringError(
                f"row {row.row_id}: the family schema lowers no {tag.name} regions"
            )
        bindings: list[SequenceBinding] = []
        stacked_rows: dict[int, int] = {}
        emitted = 0
        for region in regions:
            query_rows = region.query_rows.evaluate(operands)
            if query_rows == 0:
                continue
            instances = _instances(region, operands)
            max_branches = max(max_branches, len(instances) if len(instances) > 1 else 0)
            for branch_id, role_id in instances:
                emitted += 1
                region_counter += 1
                reserve_rows = region.reserve_rows.evaluate(operands)
                if role_id is None:
                    binding_index = 0
                    kv_group = 0
                    context_length = 0
                else:
                    sequence = roles.resolve(role_id)
                    stacked = stacked_rows.get(role_id, 0)
                    context_length = sequence.committed_rows + stacked
                    if region.cache_effect in (
                        CacheEffect.PERSISTENT_APPEND,
                        CacheEffect.TENTATIVE_APPEND,
                    ):
                        stacked_rows[role_id] = stacked + reserve_rows
                    bindings.append(
                        SequenceBinding(
                            sequence=sequence,
                            effect=region.cache_effect,
                            reserve_rows=(
                                reserve_rows
                                if region.cache_effect is not CacheEffect.READ_ONLY
                                else 0
                            ),
                        )
                    )
                    binding_index = _global_binding_index(demands, bindings)
                    kv_group = domain_by_role[role_id]
                    write_token_begins.append(token_cursor)
                is_tentative = region.cache_effect is CacheEffect.TENTATIVE_APPEND
                if is_tentative:
                    candidate_tokens = max(candidate_tokens, query_rows)
                run_cursor = 0
                for run in region.route_runs:
                    extent = run.extent.evaluate(operands)
                    if extent == 0:
                        continue
                    segments.append(
                        LoweredSegment(
                            row_id=row.row_id,
                            operation_tag=tag,
                            route_id=run.route_id,
                            query_count=extent,
                            context_length=context_length,
                            attention_pattern=int(region.attention_pattern),
                            attention_region_id=region_counter - 1,
                            kv_group=kv_group,
                            binding_index=binding_index,
                            cache_effect=region.cache_effect,
                            cache_write_count=(
                                extent
                                if region.cache_effect is not CacheEffect.READ_ONLY
                                else 0
                            ),
                            branch_id=branch_id if len(instances) > 1 else 0,
                            branch_count=(
                                len(instances) if len(instances) > 1 else 0
                            ),
                            candidate_count=extent if is_tentative else 0,
                            result_slot=row.row_id,
                        )
                    )
                    run_cursor += extent
                if run_cursor != query_rows:
                    raise LoweringError(
                        f"row {row.row_id}: route runs cover {run_cursor} of "
                        f"{query_rows} query rows"
                    )
                token_cursor += query_rows
        if emitted == 0:
            raise LoweringError(
                f"row {row.row_id}: no region evaluates to any query rows"
            )
        demands.append(RowDemand(row_id=row.row_id, bindings=tuple(bindings)))
    return LoweredBatch(
        segments=tuple(segments),
        plan=ReservationPlan(rows=tuple(demands)),
        rows=len(rows),
        tokens=token_cursor,
        branches=max_branches,
        candidate_tokens=candidate_tokens,
        write_token_begins=tuple(write_token_begins),
    )


def select_capacity(
    demand: GraphCapacity,
    capacities: tuple[GraphCapacity, ...],
) -> GraphCapacity:
    """Smallest configured capacity that dominates the demand (spec rule)."""

    dominating = [
        capacity for capacity in capacities if capacity.dominates(demand)
    ]
    if not dominating:
        raise LoweringError("no configured graph capacity dominates the demand")
    return min(
        dominating,
        key=lambda capacity: (
            capacity.tokens,
            capacity.segments,
            capacity.rows,
            capacity.residency.page_references,
        ),
    )


# --------------------------------------------------------------------------- #


def _operand_values(row: ExecuteRow) -> dict[ExtentOperand, int]:
    operation = row.operation
    values = {
        ExtentOperand.INPUT_TOKEN_COUNT: 0,
        ExtentOperand.CANDIDATE_COUNT: 0,
        ExtentOperand.IMAGE_TOKEN_COUNT: 0,
        ExtentOperand.PRODUCT_ROW_COUNT: 0,
        ExtentOperand.CFG_BRANCH_COUNT: 0,
    }
    if isinstance(operation, SequenceStep):
        values[ExtentOperand.INPUT_TOKEN_COUNT] = len(operation.input_tokens)
        if operation.verification is not None:
            values[ExtentOperand.CANDIDATE_COUNT] = len(
                operation.verification.candidate_tokens
            )
        if row.product_leases:
            # Generated-image feedback appends an explicitly encoded region:
            # a sequence row carrying exactly one conditioning product lends
            # that product's extent to the image-token operand.
            if len(row.product_leases) != 1:
                raise LoweringError(
                    f"row {row.row_id}: sequence rows carry at most one "
                    "conditioning product"
                )
            values[ExtentOperand.IMAGE_TOKEN_COUNT] = row.product_leases[
                0
            ].extent_rows
    elif isinstance(operation, FlowStep):
        values[ExtentOperand.CFG_BRANCH_COUNT] = len(operation.branch_coefficients)
        extent = _product_extent(row, operation.input_product)
        values[ExtentOperand.IMAGE_TOKEN_COUNT] = extent
        values[ExtentOperand.PRODUCT_ROW_COUNT] = extent
    elif isinstance(operation, EncodeStep):
        tokens = operation.grid[0] * operation.grid[1] * operation.grid[2]
        values[ExtentOperand.IMAGE_TOKEN_COUNT] = tokens
        values[ExtentOperand.PRODUCT_ROW_COUNT] = tokens
    elif isinstance(operation, MaterializeStep):
        values[ExtentOperand.PRODUCT_ROW_COUNT] = _product_extent(
            row, operation.input_product
        )
    return values


def _product_extent(row: ExecuteRow, lease_id: int) -> int:
    for lease in row.product_leases:
        if lease.lease_id == lease_id:
            return lease.extent_rows
    raise LoweringError(
        f"row {row.row_id} names input product {lease_id} without its lease"
    )


def _instances(
    region: CacheRegionSpec,
    operands: dict[ExtentOperand, int],
) -> list[tuple[int, int | None]]:
    """``(branch_id, role_id-or-None)`` per region instance."""

    selector = region.role
    if selector.kind is RoleSelectorKind.NO_CACHE:
        return [(0, None)]
    if selector.kind is RoleSelectorKind.FIXED:
        return [(0, selector.role_id)]
    branch_count = operands[ExtentOperand.CFG_BRANCH_COUNT]
    if branch_count <= 0:
        raise LoweringError("branch-role regions require a CFG branch count")
    if branch_count > len(selector.branch_role_ids):
        raise LoweringError(
            f"{branch_count} CFG branches exceed the declared branch roles"
        )
    return [
        (branch, selector.branch_role_ids[branch])
        for branch in range(branch_count)
    ]


def _global_binding_index(
    demands: list[RowDemand],
    bindings: list[SequenceBinding],
) -> int:
    """1-based plan-order binding index (0 is the sink binding)."""

    return sum(len(row.bindings) for row in demands) + len(bindings)
