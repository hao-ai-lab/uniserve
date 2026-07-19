"""Schema-driven row lowering and its closed segment-table device schema.

This module is torch-free. One family cache registration is interpreted into
segments, aggregate graph demand, residency reservations, and expected session
effects, so host planning and device metadata share one structural source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

from uniserve_worker.contracts.cache_schema import (
    AttentionPattern,
    CacheEffect,
    CacheRegionSpec,
    CommitExpr,
    ExtentOperand,
    FamilyCacheRegistration,
    RoleSelectorKind,
)
from uniserve_worker.contracts.execution import (
    EncodeStep,
    ExecuteRow,
    MaterializeStep,
    OperationTag,
    SequenceStep,
    operation_tag,
)
from uniserve_worker.contracts.execution import (
    FlowStep as FlowOperation,
)
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.runtime.transactional_residency import (
    ProductDemand,
    ReservationPlan,
    RowDemand,
    SequenceBinding,
)

__all__ = [
    "AttentionLayerSpec",
    "GraphCapacity",
    "LoweredBatch",
    "LoweredSegment",
    "LoweringError",
    "RoleSequences",
    "SegmentTableArrays",
    "SegmentTableError",
    "lower_rows",
    "select_capacity",
]


class SegmentTableError(ValueError):
    """A populated segment table violates the closed device schema."""


@dataclass(frozen=True, slots=True)
class AttentionLayerSpec:
    """Narrow immutable layer description handed to the attention backend.

    The model-facing attention contract: family code passes Q/K/V and this
    static description to the shared attention layer; providers own cache
    writes, metadata, and kernels behind it.
    """

    layer_id: int
    site_id: int
    domain_id: int
    query_heads: int
    kv_heads: int
    qk_head_dim: int
    value_head_dim: int
    scale: float


@dataclass(frozen=True, slots=True)
class GraphCapacity:
    """One captured bucket's finite capacity vector.

    Buckets reserve pointer-stable metadata at these bounds; capacity is
    metadata reservation, not KV ownership. Selection picks the smallest
    configured capacity that dominates a transaction's aggregate demand.
    """

    rows: int
    segments: int
    tokens: int
    branches: int
    candidate_tokens: int
    position_axes: int
    visibility_payload_entries: int
    residency: ResidencyBatchCapacity

    def __post_init__(self) -> None:
        scalar_axes = (
            self.rows,
            self.segments,
            self.tokens,
            self.branches,
            self.candidate_tokens,
            self.position_axes,
            self.visibility_payload_entries,
        )
        if any(axis < 0 for axis in scalar_axes):
            raise SegmentTableError("capacity axes must be non-negative")

    def dominates(self, other: "GraphCapacity") -> bool:
        """True when every axis of ``other`` fits inside this capacity."""

        return (
            self.rows >= other.rows
            and self.segments >= other.segments
            and self.tokens >= other.tokens
            and self.branches >= other.branches
            and self.candidate_tokens >= other.candidate_tokens
            and self.position_axes >= other.position_axes
            and self.visibility_payload_entries >= other.visibility_payload_entries
            and self.residency.bindings >= other.residency.bindings
            and self.residency.page_references >= other.residency.page_references
            and self.residency.tokens >= other.residency.tokens
        )


_COLUMNS = (
    "segment_active",
    "row_id",
    "operation_tag",
    "route_id",
    "local_segment_id",
    "token_begin",
    "token_count",
    "query_begin",
    "query_count",
    "context_length",
    "position_begin",
    "position_count",
    "branch_id",
    "branch_count",
    "attention_pattern",
    "attention_region_id",
    "kv_group",
    "kv_read_index",
    "kv_write_index",
    "cache_effect",
    "cache_write_count",
    "input_product_index",
    "output_product_index",
    "overlay_slot",
    "candidate_begin",
    "candidate_count",
    "result_slot",
)


@dataclass(frozen=True, slots=True)
class SegmentTableArrays:
    """One populated segment table, expressed as host integer sequences.

    Column set is the parent's mandatory schema plus the cache companion's
    refinements (attention region, position count, cache effect and write
    count). All columns share capacity ``segments``.
    """

    segment_active: Sequence[int]
    row_id: Sequence[int]
    operation_tag: Sequence[int]
    route_id: Sequence[int]
    local_segment_id: Sequence[int]
    token_begin: Sequence[int]
    token_count: Sequence[int]
    query_begin: Sequence[int]
    query_count: Sequence[int]
    context_length: Sequence[int]
    position_begin: Sequence[int]
    position_count: Sequence[int]
    branch_id: Sequence[int]
    branch_count: Sequence[int]
    attention_pattern: Sequence[int]
    attention_region_id: Sequence[int]
    kv_group: Sequence[int]
    kv_read_index: Sequence[int]
    kv_write_index: Sequence[int]
    cache_effect: Sequence[int]
    cache_write_count: Sequence[int]
    input_product_index: Sequence[int]
    output_product_index: Sequence[int]
    overlay_slot: Sequence[int]
    candidate_begin: Sequence[int]
    candidate_count: Sequence[int]
    result_slot: Sequence[int]

    def validate(self, capacity: GraphCapacity) -> None:
        """Prove the structural device invariants for one populated table."""

        self._validate_shapes(capacity)
        active = self._validate_active_prefix()
        self._validate_scheduler_order(active, capacity)
        self._validate_semantics(active, capacity)
        self._validate_inactive_tail(active, capacity)

    # ------------------------------------------------------------------ #

    def _validate_shapes(self, capacity: GraphCapacity) -> None:
        for field in fields(self):
            column = getattr(self, field.name)
            if len(column) != capacity.segments:
                raise SegmentTableError(
                    f"column {field.name} has {len(column)} entries; capacity "
                    f"requires {capacity.segments}"
                )

    def _validate_active_prefix(self) -> int:
        active = 0
        seen_inactive = False
        for index, flag in enumerate(self.segment_active):
            if flag not in (0, 1):
                raise SegmentTableError(f"segment_active[{index}] must be 0 or 1")
            if flag == 1:
                if seen_inactive:
                    raise SegmentTableError("active segments must occupy one prefix of the table")
                active += 1
            else:
                seen_inactive = True
        return active

    def _validate_scheduler_order(self, active: int, capacity: GraphCapacity) -> None:
        expected_token_begin = 0
        expected_query_begin = 0
        previous_row = -1
        expected_local = 0
        for segment in range(active):
            row = self.row_id[segment]
            if row < previous_row:
                raise SegmentTableError(f"segment {segment} breaks scheduler row order")
            if row > previous_row:
                if row != previous_row + 1:
                    raise SegmentTableError(f"segment {segment} skips row {previous_row + 1}")
                expected_local = 0
            if self.local_segment_id[segment] != expected_local:
                raise SegmentTableError(
                    f"segment {segment} breaks local segment order for row {row}"
                )
            if self.token_begin[segment] != expected_token_begin:
                raise SegmentTableError(f"segment {segment} token span is not packed canonically")
            if self.query_begin[segment] != expected_query_begin:
                raise SegmentTableError(f"segment {segment} query span is not packed canonically")
            if self.token_count[segment] < 0 or self.query_count[segment] < 0:
                raise SegmentTableError(f"segment {segment} has negative span extents")
            expected_token_begin += self.token_count[segment]
            expected_query_begin += self.query_count[segment]
            previous_row = row
            expected_local += 1
        if active and self.row_id[active - 1] + 1 > capacity.rows:
            raise SegmentTableError("active rows exceed the bucket row capacity")
        if expected_token_begin > capacity.tokens:
            raise SegmentTableError("packed tokens exceed the bucket token capacity")

    def _validate_semantics(self, active: int, capacity: GraphCapacity) -> None:
        operation_tags = {int(tag) for tag in OperationTag}
        attention_patterns = {int(pattern) for pattern in AttentionPattern}
        cache_effects = {int(effect) for effect in CacheEffect}
        for segment in range(active):
            if self.operation_tag[segment] not in operation_tags:
                raise SegmentTableError(f"segment {segment} carries an unsealed operation tag")
            if self.attention_pattern[segment] not in attention_patterns:
                raise SegmentTableError(f"segment {segment} carries an unknown attention pattern")
            if self.cache_effect[segment] not in cache_effects:
                raise SegmentTableError(f"segment {segment} carries an unknown cache effect")
            branch_count = self.branch_count[segment]
            if branch_count > capacity.branches:
                raise SegmentTableError(f"segment {segment} branch count exceeds capacity")
            if branch_count > 0 and not 0 <= self.branch_id[segment] < branch_count:
                raise SegmentTableError(
                    f"segment {segment} branch identity is outside its multiplicity"
                )
            bindings = capacity.residency.bindings
            if not 0 <= self.kv_read_index[segment] <= bindings:
                raise SegmentTableError(f"segment {segment} kv_read_index exceeds binding capacity")
            if not 0 <= self.kv_write_index[segment] <= bindings:
                raise SegmentTableError(
                    f"segment {segment} kv_write_index exceeds binding capacity"
                )
            effect = self.cache_effect[segment]
            if effect == int(CacheEffect.READ_ONLY) and (
                self.kv_write_index[segment] != 0 or self.cache_write_count[segment] != 0
            ):
                raise SegmentTableError(
                    f"segment {segment}: READ_ONLY segments own no write binding"
                )
            if effect != int(CacheEffect.READ_ONLY) and (
                self.kv_write_index[segment] == 0 or self.cache_write_count[segment] <= 0
            ):
                raise SegmentTableError(
                    f"segment {segment}: writing effects need an active write "
                    "binding and a positive write count"
                )
            if self.kv_group[segment] == 0 and (
                self.kv_read_index[segment] != 0
                or self.kv_write_index[segment] != 0
                or self.context_length[segment] != 0
                or effect != int(CacheEffect.READ_ONLY)
            ):
                raise SegmentTableError(
                    f"segment {segment}: NO_CACHE_DOMAIN segments are read-only "
                    "with zero bindings and context"
                )
            if self.candidate_count[segment] > 0 and self.operation_tag[segment] != int(
                OperationTag.SEQUENCE_STEP
            ):
                raise SegmentTableError(
                    f"segment {segment}: candidate spans belong to sequence steps"
                )
            if self.candidate_count[segment] > capacity.candidate_tokens:
                raise SegmentTableError(f"segment {segment} candidate span exceeds capacity")
            if self.position_count[segment] != self.query_count[segment]:
                raise SegmentTableError(f"segment {segment}: position_count must equal query_count")
            if not 0 <= self.result_slot[segment] < max(capacity.rows, 1):
                raise SegmentTableError(
                    f"segment {segment} result slot exceeds the bucket row capacity"
                )

    def _validate_inactive_tail(self, active: int, capacity: GraphCapacity) -> None:
        for segment in range(active, capacity.segments):
            for name in _COLUMNS:
                if getattr(self, name)[segment] != 0:
                    raise SegmentTableError(
                        f"inactive segment {segment} column {name} must hold the zero sentinel"
                    )


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
    """Segments, reservation plan, and aggregate demand for one transaction.

    ``binding_commits`` and ``binding_row_ids`` align with the plan's binding
    order: the declared commit expression and owning row of each reserved
    binding, so delta derivation consumes the same closed schema source that
    produced the reservation.
    """

    segments: tuple[LoweredSegment, ...]
    plan: ReservationPlan
    rows: int
    tokens: int
    branches: int
    candidate_tokens: int
    write_token_begins: tuple[int, ...]
    binding_commits: tuple[CommitExpr, ...]
    binding_row_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]

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
            name: [0] * capacity.segments for name in SegmentTableArrays.__dataclass_fields__
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
                segment.binding_index if segment.cache_effect is not CacheEffect.READ_ONLY else 0
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

    domain_by_role = {role.role_id: role.domain_id for role in registration.schema.roles}
    segments: list[LoweredSegment] = []
    demands: list[RowDemand] = []
    write_token_begins: list[int] = []
    binding_commits: list[CommitExpr] = []
    binding_row_ids: list[int] = []
    token_ids: list[int] = []
    positions: list[int] = []
    token_cursor = 0
    region_counter = 0
    max_branches = 0
    candidate_tokens = 0
    for row, roles in rows:
        operands = _operand_values(row)
        tag = operation_tag(row.operation)
        regions = [region for region in registration.schema.regions if region.operation_tag is tag]
        if not regions:
            raise LoweringError(f"row {row.row_id}: the family schema lowers no {tag.name} regions")
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
                    binding_commits.append(region.commit_rows)
                    binding_row_ids.append(row.row_id)
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
                                extent if region.cache_effect is not CacheEffect.READ_ONLY else 0
                            ),
                            branch_id=branch_id if len(instances) > 1 else 0,
                            branch_count=(len(instances) if len(instances) > 1 else 0),
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
                ids, region_positions = _region_payload(
                    row.operation, region, query_rows, context_length
                )
                token_ids.extend(ids)
                positions.extend(region_positions)
                token_cursor += query_rows
        if emitted == 0:
            raise LoweringError(f"row {row.row_id}: no region evaluates to any query rows")
        demands.append(
            RowDemand(
                row_id=row.row_id,
                bindings=tuple(bindings),
                products=_published_products(row, operands),
                input_products=tuple(lease.lease_id for lease in row.product_leases),
            )
        )
    return LoweredBatch(
        segments=tuple(segments),
        plan=ReservationPlan(rows=tuple(demands)),
        rows=len(rows),
        tokens=token_cursor,
        branches=max_branches,
        candidate_tokens=candidate_tokens,
        write_token_begins=tuple(write_token_begins),
        binding_commits=tuple(binding_commits),
        binding_row_ids=tuple(binding_row_ids),
        token_ids=tuple(token_ids),
        positions=tuple(positions),
    )


def select_capacity(
    demand: GraphCapacity,
    capacities: tuple[GraphCapacity, ...],
) -> GraphCapacity:
    """Smallest configured capacity that dominates the demand (spec rule)."""

    dominating = [capacity for capacity in capacities if capacity.dominates(demand)]
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
            values[ExtentOperand.CANDIDATE_COUNT] = len(operation.verification.candidate_tokens)
        if row.product_leases:
            # Generated-image feedback appends an explicitly encoded region:
            # a sequence row carrying exactly one conditioning product lends
            # that product's extent to the image-token operand.
            if len(row.product_leases) != 1:
                raise LoweringError(
                    f"row {row.row_id}: sequence rows carry at most one conditioning product"
                )
            values[ExtentOperand.IMAGE_TOKEN_COUNT] = row.product_leases[0].extent_rows
    elif isinstance(operation, FlowOperation):
        values[ExtentOperand.CFG_BRANCH_COUNT] = len(operation.branch_coefficients)
        extent = _product_extent(row, operation.input_product)
        values[ExtentOperand.IMAGE_TOKEN_COUNT] = extent
        values[ExtentOperand.PRODUCT_ROW_COUNT] = extent
    elif isinstance(operation, EncodeStep):
        tokens = operation.grid[0] * operation.grid[1] * operation.grid[2]
        values[ExtentOperand.IMAGE_TOKEN_COUNT] = tokens
        values[ExtentOperand.PRODUCT_ROW_COUNT] = tokens
    elif isinstance(operation, MaterializeStep):
        values[ExtentOperand.PRODUCT_ROW_COUNT] = _product_extent(row, operation.input_product)
    return values


def _region_payload(
    operation: object,
    region: CacheRegionSpec,
    query_rows: int,
    context_length: int,
) -> tuple[list[int], list[int]]:
    """Packed token ids and positions for one region instance.

    Sequence regions carry the operation's real token spans and linear
    positions; verification regions carry the candidate span at candidate
    positions; non-token regions pack zero ids with region-local positions.
    """

    if isinstance(operation, SequenceStep):
        if (
            region.cache_effect is CacheEffect.TENTATIVE_APPEND
            and operation.verification is not None
        ):
            return (
                list(operation.verification.candidate_tokens),
                list(operation.verification.candidate_positions),
            )
        if operation.input_tokens:
            begin = operation.position_begin
            return (
                list(operation.input_tokens),
                list(range(begin, begin + len(operation.input_tokens))),
            )
    return [0] * query_rows, list(range(context_length, context_length + query_rows))


def _published_products(
    row: ExecuteRow,
    operands: dict[ExtentOperand, int],
) -> tuple[ProductDemand, ...]:
    """Encode and materialize rows publish one typed product at commit."""

    operation = row.operation
    if isinstance(operation, EncodeStep):
        return (
            ProductDemand(
                schema_id=operation.output_schema,
                rows=operands[ExtentOperand.IMAGE_TOKEN_COUNT],
                producer=row.session,
            ),
        )
    if isinstance(operation, MaterializeStep):
        return (
            ProductDemand(
                schema_id=operation.output_schema,
                rows=operands[ExtentOperand.PRODUCT_ROW_COUNT],
                producer=row.session,
            ),
        )
    return ()


def _product_extent(row: ExecuteRow, lease_id: int) -> int:
    for lease in row.product_leases:
        if lease.lease_id == lease_id:
            return lease.extent_rows
    raise LoweringError(f"row {row.row_id} names input product {lease_id} without its lease")


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
        raise LoweringError(f"{branch_count} CFG branches exceed the declared branch roles")
    return [(branch, selector.branch_role_ids[branch]) for branch in range(branch_count)]


def _global_binding_index(
    demands: list[RowDemand],
    bindings: list[SequenceBinding],
) -> int:
    """1-based plan-order binding index (0 is the sink binding)."""

    return sum(len(row.bindings) for row in demands) + len(bindings)
