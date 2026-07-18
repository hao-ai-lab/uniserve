"""Schema-driven lowering composition (Stage 4, dormant).

Proves the companion's central rule end to end: one family cache schema
derives segments, capacity demand, and cache reservations; the packed device
tables produced from real reservations pass the same contract validators the
graph buckets will use.
"""

from __future__ import annotations

import pytest

from uniserve_worker.contracts.cache_schema import CacheEffect, CacheLifetime
from uniserve_worker.contracts.execution import (
    CandidateVerification,
    EngineRef,
    ExecuteRow,
    FlowStep,
    OperationTag,
    ProductLease,
    SequenceStep,
    SessionRef,
)
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.contracts.segment_table import GraphCapacity
from uniserve_worker.execution.engine import (
    LoweringError,
    RoleSequences,
    lower_rows,
    select_capacity,
)
from uniserve_worker.models.cache_registrations import (
    IMAGE_UNCONDITIONAL_ROLE,
    PRIMARY_ROLE,
    TEXT_UNCONDITIONAL_ROLE,
    qwen3_cache_registration,
    sensenova_cache_registration,
)
from uniserve_worker.runtime.transactional_residency import (
    ArenaConfig,
    ProductDemand,
    ProductStoreConfig,
    ReservationPlan,
    Residency,
    RowDemand,
)

pytestmark = pytest.mark.unit

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_SESSION = SessionRef(_ENGINE, request_id=41, incarnation=1, session_version=2)

_QWEN3 = qwen3_cache_registration(
    layer_count=4, query_heads=8, kv_heads=2, qk_head_dim=64, value_head_dim=64
)
_SENSENOVA = sensenova_cache_registration(
    layer_count=4, query_heads=8, kv_heads=2, qk_head_dim=64, value_head_dim=64
)


def _capacity(**overrides) -> GraphCapacity:
    values = dict(
        rows=4,
        segments=8,
        tokens=128,
        branches=3,
        candidate_tokens=8,
        position_axes=3,
        visibility_payload_entries=0,
        residency=ResidencyBatchCapacity(bindings=6, page_references=64, tokens=128),
    )
    values.update(overrides)
    return GraphCapacity(**values)


def _residency() -> Residency:
    return Residency(
        _ENGINE,
        (ArenaConfig(domain_id=1, page_count=65, page_tokens=16),),
        product_stores=(ProductStoreConfig(schema_id=7, row_capacity=256),),
    )


def _row(operation, *, row_id: int = 0, product_leases=()) -> ExecuteRow:
    return ExecuteRow(
        row_id=row_id,
        session=_SESSION,
        operation=operation,
        admission=None,
        cache_leases=(),
        product_leases=tuple(product_leases),
        scheduler_op_id=row_id,
    )


def _publish_latent(residency: Residency, rows: int) -> ProductLease:
    """Publish one committed latent product (an encode row's outcome)."""

    reservation = residency.reserve(
        ReservationPlan(
            rows=(
                RowDemand(
                    row_id=0,
                    bindings=(),
                    products=(ProductDemand(schema_id=7, rows=rows, producer=_SESSION),),
                ),
            )
        )
    )
    (lease,) = reservation.commit(())
    return lease


def test_qwen3_extend_and_verification_stack_one_sequence():
    residency = _residency()
    primary = residency.create_sequence(
        _SESSION, domain_id=1, role_id=PRIMARY_ROLE, lifetime=CacheLifetime.REQUEST
    )
    operation = SequenceStep(
        input_tokens=(5, 6, 7),
        history_length=0,
        position_begin=0,
        requested_outputs=1,
        verification=CandidateVerification((11, 12), (3, 4)),
    )
    lowered = lower_rows(
        ((_row(operation), RoleSequences({PRIMARY_ROLE: primary})),),
        _QWEN3,
    )
    # Two regions -> two segments on one row; verification stacks after text.
    assert [segment.query_count for segment in lowered.segments] == [3, 2]
    assert [segment.context_length for segment in lowered.segments] == [0, 3]
    assert [segment.candidate_count for segment in lowered.segments] == [0, 2]
    assert [segment.cache_effect for segment in lowered.segments] == [
        CacheEffect.PERSISTENT_APPEND,
        CacheEffect.TENTATIVE_APPEND,
    ]
    assert lowered.tokens == 5

    capacity = _capacity()
    table = lowered.fill_segment_table(capacity)
    table.validate(capacity)

    reservation = residency.reserve(lowered.plan)
    arrays = reservation.batch_arrays(capacity.residency, lowered.write_token_begins)
    arrays.validate(capacity.residency, page_tokens=16)
    # Text rows commit fully; one accepted candidate publishes one more row.
    reservation.commit((3, 1))
    assert residency.sequence_ref(primary.sequence_id).committed_rows == 4


def test_sensenova_flow_lowers_one_segment_per_cfg_branch():
    residency = _residency()
    roles = RoleSequences(
        {
            role_id: residency.create_sequence(
                _SESSION,
                domain_id=1,
                role_id=role_id,
                lifetime=CacheLifetime.BRANCH,
            )
            for role_id in (
                PRIMARY_ROLE,
                TEXT_UNCONDITIONAL_ROLE,
                IMAGE_UNCONDITIONAL_ROLE,
            )
        }
    )
    latent = _publish_latent(residency, 32)
    operation = FlowStep(
        schedule_id=1,
        step_index=3,
        total_steps=50,
        input_product=latent.lease_id,
        branch_coefficients=(4.0, 1.0, 1.0),
        conditioning_products=(),
        output_schema=7,
    )
    lowered = lower_rows(
        ((_row(operation, product_leases=(latent,)), roles),),
        _SENSENOVA,
    )
    assert len(lowered.segments) == 3
    assert [segment.branch_id for segment in lowered.segments] == [0, 1, 2]
    assert all(segment.branch_count == 3 for segment in lowered.segments)
    assert all(
        segment.cache_effect is CacheEffect.TRANSIENT_OVERLAY for segment in lowered.segments
    )
    assert lowered.tokens == 96

    capacity = _capacity(tokens=128, candidate_tokens=0)
    table = lowered.fill_segment_table(capacity)
    table.validate(capacity)

    reservation = residency.reserve(lowered.plan)
    arrays = reservation.batch_arrays(capacity.residency, lowered.write_token_begins)
    arrays.validate(capacity.residency, page_tokens=16)
    # Flow commits zero rows everywhere; overlay pages return.
    free_before = residency.pressure()[1]["free_pages"]
    reservation.commit((0, 0, 0))
    assert residency.pressure()[1]["free_pages"] == free_before + 6


def test_sensenova_feedback_appends_the_encoded_image_region():
    residency = _residency()
    primary = residency.create_sequence(
        _SESSION, domain_id=1, role_id=PRIMARY_ROLE, lifetime=CacheLifetime.REQUEST
    )
    feedback = SequenceStep(
        input_tokens=(),
        history_length=0,
        position_begin=7,
        requested_outputs=1,
    )
    lowered = lower_rows(
        (
            (
                _row(feedback, product_leases=(_publish_latent(residency, 48),)),
                RoleSequences({PRIMARY_ROLE: primary}),
            ),
        ),
        _SENSENOVA,
    )
    assert len(lowered.segments) == 1
    segment = lowered.segments[0]
    assert segment.query_count == 48
    assert segment.cache_effect is CacheEffect.PERSISTENT_APPEND
    reservation = residency.reserve(lowered.plan)
    reservation.commit((48,))
    assert residency.sequence_ref(primary.sequence_id).committed_rows == 48


def test_demand_drives_smallest_dominating_capacity():
    residency = _residency()
    primary = residency.create_sequence(
        _SESSION, domain_id=1, role_id=PRIMARY_ROLE, lifetime=CacheLifetime.REQUEST
    )
    operation = SequenceStep((1, 2, 3, 4), 0, 0, 1)
    lowered = lower_rows(
        ((_row(operation), RoleSequences({PRIMARY_ROLE: primary})),),
        _QWEN3,
    )
    demand = lowered.demand(page_tokens=16)
    small = _capacity(tokens=16, segments=2, rows=1, candidate_tokens=0)
    large = _capacity(tokens=128)
    assert select_capacity(demand, (large, small)) == small
    tiny = _capacity(tokens=2, segments=1, rows=1, candidate_tokens=0)
    with pytest.raises(LoweringError, match="dominates"):
        select_capacity(demand, (tiny,))


def test_zero_row_operations_are_rejected():
    residency = _residency()
    primary = residency.create_sequence(
        _SESSION, domain_id=1, role_id=PRIMARY_ROLE, lifetime=CacheLifetime.REQUEST
    )
    empty = SequenceStep((), 0, 0, 1)
    with pytest.raises(LoweringError, match="no region"):
        lower_rows(
            ((_row(empty), RoleSequences({PRIMARY_ROLE: primary})),),
            _QWEN3,
        )


def test_operation_tags_without_regions_fail_closed():
    residency = _residency()
    primary = residency.create_sequence(
        _SESSION, domain_id=1, role_id=PRIMARY_ROLE, lifetime=CacheLifetime.REQUEST
    )
    latent = _publish_latent(residency, 8)
    flow = FlowStep(1, 0, 50, latent.lease_id, (1.0,), (), 7)
    with pytest.raises(LoweringError, match="no FLOW_STEP regions|lowers no"):
        lower_rows(
            (
                (
                    _row(flow, product_leases=(latent,)),
                    RoleSequences({PRIMARY_ROLE: primary}),
                ),
            ),
            _QWEN3,
        )
    assert OperationTag.FLOW_STEP not in {region.operation_tag for region in _QWEN3.schema.regions}
