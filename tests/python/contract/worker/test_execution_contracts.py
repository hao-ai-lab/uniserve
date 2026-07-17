"""Canonical host execution contracts (unified forward execution, Stage 1).

Covers the Stage 1 test obligations that are decidable on the Python value
layer: canonical row order, duplicate incarnation rejection, admission
version rules, sealed/advertised operation membership, per-variant bounds,
canonical fingerprint stability, and the acknowledgement exclusion from
transaction identity. Cross-language byte agreement is pinned separately by
the shared fingerprint vectors in ``crates/protocol/vocab``.
"""

from __future__ import annotations

import dataclasses

import pytest

from uniserve_worker.contracts.execution import (
    CacheLease,
    CandidateVerification,
    EngineRef,
    ExecuteBatch,
    ExecuteRow,
    ExecutionContractError,
    FlowStep,
    NewSession,
    OperationTag,
    SamplingSpec,
    SequenceStep,
    SessionRef,
    canonical_payload_fingerprint,
    operation_tag,
    validate_execute_batch,
)

pytestmark = pytest.mark.contract

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_ALL_OPS = frozenset(OperationTag)


def _session(request_id: int, *, version: int = 3, incarnation: int = 1) -> SessionRef:
    return SessionRef(
        engine=_ENGINE,
        request_id=request_id,
        incarnation=incarnation,
        session_version=version,
    )


def _sequence(tokens: tuple[int, ...] = (5, 6, 7)) -> SequenceStep:
    return SequenceStep(
        input_tokens=tokens,
        history_length=12,
        position_begin=12,
        requested_outputs=1,
    )


def _row(row_id: int, request_id: int, **overrides) -> ExecuteRow:
    values = dict(
        row_id=row_id,
        session=_session(request_id),
        operation=_sequence(),
        admission=None,
        cache_leases=(),
        product_leases=(),
        scheduler_op_id=row_id + 100,
    )
    values.update(overrides)
    return ExecuteRow(**values)


def _batch(*rows: ExecuteRow, step_id: int = 9, ack: int = 4) -> ExecuteBatch:
    return ExecuteBatch(
        engine_epoch=_ENGINE.engine_epoch,
        step_id=step_id,
        acknowledged_through=ack,
        rows=tuple(rows),
    )


def test_valid_batch_passes_and_rows_stay_in_scheduler_order():
    batch = _batch(_row(0, 41), _row(1, 42))
    validate_execute_batch(batch, advertised_operations=_ALL_OPS)
    assert [row.row_id for row in batch.rows] == [0, 1]


def test_row_id_must_equal_position():
    batch = _batch(_row(0, 41), _row(0, 42))
    with pytest.raises(ExecutionContractError, match="position"):
        validate_execute_batch(batch, advertised_operations=_ALL_OPS)


def test_duplicate_request_incarnation_is_rejected():
    batch = _batch(_row(0, 41), _row(1, 41))
    with pytest.raises(ExecutionContractError, match="incarnation"):
        validate_execute_batch(batch, advertised_operations=_ALL_OPS)


def test_same_request_new_incarnation_is_legal():
    second = _row(1, 41, session=_session(41, incarnation=2))
    validate_execute_batch(
        _batch(_row(0, 41), second),
        advertised_operations=_ALL_OPS,
    )


def test_admission_only_at_version_zero():
    admission = NewSession(
        request_id=41,
        incarnation=1,
        sampling=SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0),
        base_seed=42,
        max_history_tokens=4096,
    )
    versioned = _row(0, 41, admission=admission)
    with pytest.raises(ExecutionContractError, match="version 0"):
        validate_execute_batch(_batch(versioned), advertised_operations=_ALL_OPS)
    fresh = _row(0, 41, session=_session(41, version=0), admission=admission)
    validate_execute_batch(_batch(fresh), advertised_operations=_ALL_OPS)
    unadmitted = _row(0, 41, session=_session(41, version=0))
    with pytest.raises(ExecutionContractError, match="unadmitted"):
        validate_execute_batch(_batch(unadmitted), advertised_operations=_ALL_OPS)


def test_operations_outside_the_advertised_set_are_rejected():
    sequence_only = frozenset({OperationTag.SEQUENCE_STEP})
    flow = FlowStep(
        schedule_id=1,
        step_index=0,
        total_steps=50,
        input_product=3,
        branch_coefficients=(4.0, 1.0),
        conditioning_products=(1, 2),
        output_schema=5,
    )
    batch = _batch(_row(0, 41, operation=flow))
    with pytest.raises(ExecutionContractError, match="advertised"):
        validate_execute_batch(batch, advertised_operations=sequence_only)
    assert operation_tag(flow) is OperationTag.FLOW_STEP


def test_sequence_step_bounds():
    empty = _row(0, 41, operation=_sequence(()))
    with pytest.raises(ExecutionContractError, match="no positions"):
        validate_execute_batch(_batch(empty), advertised_operations=_ALL_OPS)

    misaligned = _row(
        0,
        41,
        operation=SequenceStep(
            input_tokens=(),
            history_length=8,
            position_begin=8,
            requested_outputs=1,
            verification=CandidateVerification(
                candidate_tokens=(1, 2, 3),
                candidate_positions=(8, 9),
            ),
        ),
    )
    with pytest.raises(ExecutionContractError, match="disagree"):
        validate_execute_batch(_batch(misaligned), advertised_operations=_ALL_OPS)


def test_flow_step_coordinate_and_coefficients_are_bounded():
    out_of_schedule = FlowStep(
        schedule_id=1,
        step_index=50,
        total_steps=50,
        input_product=3,
        branch_coefficients=(1.0,),
        conditioning_products=(),
        output_schema=5,
    )
    with pytest.raises(ExecutionContractError, match="schedule"):
        validate_execute_batch(
            _batch(_row(0, 41, operation=out_of_schedule)),
            advertised_operations=_ALL_OPS,
        )
    non_finite = dataclasses.replace(
        out_of_schedule,
        step_index=0,
        branch_coefficients=(float("nan"),),
    )
    with pytest.raises(ExecutionContractError, match="finite"):
        validate_execute_batch(
            _batch(_row(0, 41, operation=non_finite)),
            advertised_operations=_ALL_OPS,
        )


def test_cache_leases_pin_epoch_and_digest_shape():
    lease = CacheLease(
        lease_id=5,
        engine_epoch=_ENGINE.engine_epoch + 1,
        identity_digest=bytes(32),
        identity_schema=1,
        charge=128,
        version=2,
        residency_handle=77,
    )
    with pytest.raises(ExecutionContractError, match="foreign epoch"):
        validate_execute_batch(
            _batch(_row(0, 41, cache_leases=(lease,))),
            advertised_operations=_ALL_OPS,
        )
    short_digest = dataclasses.replace(
        lease,
        engine_epoch=_ENGINE.engine_epoch,
        identity_digest=b"short",
    )
    with pytest.raises(ExecutionContractError, match="32 bytes"):
        validate_execute_batch(
            _batch(_row(0, 41, cache_leases=(short_digest,))),
            advertised_operations=_ALL_OPS,
        )


# --------------------------------------------------------------------------- #
# Canonical payload fingerprint.
# --------------------------------------------------------------------------- #


def test_fingerprint_is_stable_and_row_order_sensitive():
    batch = _batch(_row(0, 41), _row(1, 42))
    again = _batch(_row(0, 41), _row(1, 42))
    assert canonical_payload_fingerprint(batch) == canonical_payload_fingerprint(again)
    swapped = _batch(
        _row(0, 42),
        _row(1, 41),
    )
    assert canonical_payload_fingerprint(batch) != canonical_payload_fingerprint(
        swapped
    )


def test_fingerprint_excludes_cumulative_acknowledgement():
    batch = _batch(_row(0, 41), ack=4)
    retried = _batch(_row(0, 41), ack=9)
    assert canonical_payload_fingerprint(batch) == canonical_payload_fingerprint(
        retried
    )


def test_fingerprint_covers_operation_and_lease_values():
    base = _batch(_row(0, 41))
    other_tokens = _batch(_row(0, 41, operation=_sequence((5, 6, 8))))
    assert canonical_payload_fingerprint(base) != canonical_payload_fingerprint(
        other_tokens
    )
    lease = CacheLease(
        lease_id=5,
        engine_epoch=_ENGINE.engine_epoch,
        identity_digest=bytes(32),
        identity_schema=1,
        charge=128,
        version=2,
        residency_handle=77,
    )
    with_lease = _batch(_row(0, 41, cache_leases=(lease,)))
    assert canonical_payload_fingerprint(base) != canonical_payload_fingerprint(
        with_lease
    )


def test_fingerprint_distinguishes_session_versions():
    base = _batch(_row(0, 41))
    stale = _batch(_row(0, 41, session=_session(41, version=2)))
    assert canonical_payload_fingerprint(base) != canonical_payload_fingerprint(stale)
