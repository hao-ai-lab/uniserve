"""Closed ``SegmentTable`` device schema and ``GraphCapacity`` axes.

Stage 1 device-contract deliverable from
``specs/unified_forward_execution.md``, refined by the cache companion
(``specs/unified_kv_attention_runtime.md``). The segment table is the sole
structural source for attention planning, route dispatch, cache reads and
writes, overlay selection, and result projection; no second host-side
structure may disagree with it.

Like :mod:`~uniserve_worker.contracts.residency_batch`, this module is
torch-free: the schema is expressed over host integer sequences so graph
buckets, providers, and contract tests validate one description without a
GPU. The production `contracts.forward_batch.ForwardBatch` remains
authoritative until the vertical cutover replaces it.

Sentinel semantics: active segments occupy one prefix of the table; every
inactive entry has ``segment_active == 0`` and zero in every other column.
No provider or adapter may infer active structure from stale tail contents.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

from .cache_schema import AttentionPattern, CacheEffect
from .execution import OperationTag
from .residency_batch import ResidencyBatchCapacity

__all__ = [
    "AttentionLayerSpec",
    "GraphCapacity",
    "SegmentTableArrays",
    "SegmentTableError",
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
                    raise SegmentTableError(
                        "active segments must occupy one prefix of the table"
                    )
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
                raise SegmentTableError(
                    f"segment {segment} breaks scheduler row order"
                )
            if row > previous_row:
                if row != previous_row + 1:
                    raise SegmentTableError(
                        f"segment {segment} skips row {previous_row + 1}"
                    )
                expected_local = 0
            if self.local_segment_id[segment] != expected_local:
                raise SegmentTableError(
                    f"segment {segment} breaks local segment order for row {row}"
                )
            if self.token_begin[segment] != expected_token_begin:
                raise SegmentTableError(
                    f"segment {segment} token span is not packed canonically"
                )
            if self.query_begin[segment] != expected_query_begin:
                raise SegmentTableError(
                    f"segment {segment} query span is not packed canonically"
                )
            if self.token_count[segment] < 0 or self.query_count[segment] < 0:
                raise SegmentTableError(
                    f"segment {segment} has negative span extents"
                )
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
                raise SegmentTableError(
                    f"segment {segment} carries an unsealed operation tag"
                )
            if self.attention_pattern[segment] not in attention_patterns:
                raise SegmentTableError(
                    f"segment {segment} carries an unknown attention pattern"
                )
            if self.cache_effect[segment] not in cache_effects:
                raise SegmentTableError(
                    f"segment {segment} carries an unknown cache effect"
                )
            branch_count = self.branch_count[segment]
            if branch_count > capacity.branches:
                raise SegmentTableError(
                    f"segment {segment} branch count exceeds capacity"
                )
            if branch_count > 0 and not 0 <= self.branch_id[segment] < branch_count:
                raise SegmentTableError(
                    f"segment {segment} branch identity is outside its multiplicity"
                )
            bindings = capacity.residency.bindings
            if not 0 <= self.kv_read_index[segment] <= bindings:
                raise SegmentTableError(
                    f"segment {segment} kv_read_index exceeds binding capacity"
                )
            if not 0 <= self.kv_write_index[segment] <= bindings:
                raise SegmentTableError(
                    f"segment {segment} kv_write_index exceeds binding capacity"
                )
            effect = self.cache_effect[segment]
            if effect == int(CacheEffect.READ_ONLY) and (
                self.kv_write_index[segment] != 0
                or self.cache_write_count[segment] != 0
            ):
                raise SegmentTableError(
                    f"segment {segment}: READ_ONLY segments own no write binding"
                )
            if effect != int(CacheEffect.READ_ONLY) and (
                self.kv_write_index[segment] == 0
                or self.cache_write_count[segment] <= 0
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
            if self.candidate_count[segment] > 0 and self.operation_tag[
                segment
            ] != int(OperationTag.SEQUENCE_STEP):
                raise SegmentTableError(
                    f"segment {segment}: candidate spans belong to sequence steps"
                )
            if self.candidate_count[segment] > capacity.candidate_tokens:
                raise SegmentTableError(
                    f"segment {segment} candidate span exceeds capacity"
                )
            if self.position_count[segment] != self.query_count[segment]:
                raise SegmentTableError(
                    f"segment {segment}: position_count must equal query_count"
                )
            if not 0 <= self.result_slot[segment] < max(capacity.rows, 1):
                raise SegmentTableError(
                    f"segment {segment} result slot exceeds the bucket row capacity"
                )

    def _validate_inactive_tail(self, active: int, capacity: GraphCapacity) -> None:
        for segment in range(active, capacity.segments):
            for name in _COLUMNS:
                if getattr(self, name)[segment] != 0:
                    raise SegmentTableError(
                        f"inactive segment {segment} column {name} must hold "
                        "the zero sentinel"
                    )
