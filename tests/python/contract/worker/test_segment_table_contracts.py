"""Closed SegmentTable device schema and GraphCapacity (Stage 1, dormant)."""

from __future__ import annotations

import dataclasses

import pytest

from uniserve_worker.contracts.cache_schema import AttentionPattern, CacheEffect
from uniserve_worker.contracts.execution import OperationTag
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.execution import (
    GraphCapacity,
    SegmentTableArrays,
    SegmentTableError,
)

pytestmark = pytest.mark.contract


def _capacity() -> GraphCapacity:
    return GraphCapacity(
        rows=4,
        segments=6,
        tokens=64,
        branches=3,
        candidate_tokens=8,
        position_axes=3,
        visibility_payload_entries=16,
        residency=ResidencyBatchCapacity(bindings=4, page_references=32, tokens=64),
    )


def _table() -> SegmentTableArrays:
    """Two rows: a causal decode segment and a two-segment denoise region."""

    def column(*active: int) -> list[int]:
        return [*active] + [0] * (6 - len(active))

    return SegmentTableArrays(
        segment_active=column(1, 1, 1),
        row_id=column(0, 1, 1),
        operation_tag=column(
            int(OperationTag.SEQUENCE_STEP),
            int(OperationTag.FLOW_STEP),
            int(OperationTag.FLOW_STEP),
        ),
        route_id=column(0, 1, 0),
        local_segment_id=column(0, 0, 1),
        token_begin=column(0, 1, 17),
        token_count=column(1, 16, 2),
        query_begin=column(0, 1, 17),
        query_count=column(1, 16, 2),
        context_length=column(12, 40, 40),
        position_begin=column(0, 1, 17),
        position_count=column(1, 16, 2),
        branch_id=column(0, 1, 1),
        branch_count=column(0, 3, 3),
        attention_pattern=column(
            int(AttentionPattern.CAUSAL_PREFIX),
            int(AttentionPattern.FULL_QUERY_PREFIX),
            int(AttentionPattern.FULL_QUERY_PREFIX),
        ),
        attention_region_id=column(0, 1, 1),
        kv_group=column(1, 1, 1),
        kv_read_index=column(1, 2, 2),
        kv_write_index=column(1, 2, 2),
        cache_effect=column(
            int(CacheEffect.PERSISTENT_APPEND),
            int(CacheEffect.TRANSIENT_OVERLAY),
            int(CacheEffect.TRANSIENT_OVERLAY),
        ),
        cache_write_count=column(1, 16, 2),
        input_product_index=column(0, 1, 1),
        output_product_index=column(0, 1, 1),
        overlay_slot=column(0, 0, 0),
        candidate_begin=column(0, 0, 0),
        candidate_count=column(0, 0, 0),
        result_slot=column(0, 1, 1),
    )


def test_canonical_table_validates_and_capacity_dominance_orders_buckets():
    _table().validate(_capacity())
    small = _capacity()
    large = dataclasses.replace(small, tokens=128, rows=8)
    assert large.dominates(small)
    assert not small.dominates(large)


def test_active_segments_occupy_one_prefix():
    holed = dataclasses.replace(_table(), segment_active=[1, 0, 1, 0, 0, 0])
    with pytest.raises(SegmentTableError, match="prefix"):
        holed.validate(_capacity())


def test_rows_stay_in_scheduler_order_without_regrouping():
    reordered = dataclasses.replace(_table(), row_id=[0, 1, 0, 0, 0, 0])
    with pytest.raises(SegmentTableError, match="row order"):
        reordered.validate(_capacity())
    skipping = dataclasses.replace(_table(), row_id=[0, 2, 2, 0, 0, 0])
    with pytest.raises(SegmentTableError, match="skips row"):
        skipping.validate(_capacity())


def test_token_spans_pack_canonically():
    gapped = dataclasses.replace(_table(), token_begin=[0, 2, 18, 0, 0, 0])
    with pytest.raises(SegmentTableError, match="packed canonically"):
        gapped.validate(_capacity())


def test_read_only_segments_own_no_write_binding():
    table = _table()
    effects = list(table.cache_effect)
    effects[0] = int(CacheEffect.READ_ONLY)
    with pytest.raises(SegmentTableError, match="READ_ONLY"):
        dataclasses.replace(table, cache_effect=effects).validate(_capacity())


def test_no_cache_domain_segments_are_fully_neutral():
    table = _table()
    groups = list(table.kv_group)
    groups[0] = 0
    with pytest.raises(SegmentTableError, match="NO_CACHE_DOMAIN"):
        dataclasses.replace(table, kv_group=groups).validate(_capacity())


def test_branch_identity_stays_inside_its_multiplicity():
    table = _table()
    branches = list(table.branch_id)
    branches[1] = 3
    with pytest.raises(SegmentTableError, match="multiplicity"):
        dataclasses.replace(table, branch_id=branches).validate(_capacity())


def test_candidate_spans_belong_to_sequence_steps():
    table = _table()
    candidates = list(table.candidate_count)
    candidates[1] = 2
    with pytest.raises(SegmentTableError, match="sequence steps"):
        dataclasses.replace(table, candidate_count=candidates).validate(_capacity())


def test_stale_tail_contents_are_rejected():
    table = _table()
    tail = list(table.context_length)
    tail[5] = 40
    with pytest.raises(SegmentTableError, match="zero sentinel"):
        dataclasses.replace(table, context_length=tail).validate(_capacity())


def test_capacity_axes_bound_the_active_extent():
    tiny = dataclasses.replace(_capacity(), tokens=8)
    with pytest.raises(SegmentTableError, match="token capacity"):
        _table().validate(tiny)
