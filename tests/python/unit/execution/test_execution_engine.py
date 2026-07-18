"""Target ExecutionEngine transaction semantics (Stage 8 skeleton, dormant).

Drives the exactly-once machine with a stub transaction executor: duplicate
steps return the retained receipt, payload conflicts poison, acknowledgement
evicts, stale and out-of-order steps reject, the replay window applies
backpressure, pre-launch rejection leaves no record, post-acceptance failure
poisons, results stay in scheduler order, and commands respect transaction
boundaries.
"""

from __future__ import annotations

import dataclasses

import pytest

from uniserve_worker.contracts.execution import (
    DropSession,
    EngineRef,
    ExecuteBatch,
    ExecuteRow,
    ExecutionContractError,
    NewSession,
    OperationTag,
    Quiesce,
    Resume,
    RowResult,
    RowStatus,
    SamplingSpec,
    SequenceStep,
    SessionDelta,
    SessionRef,
    TerminalStatus,
)
from uniserve_worker.execution.engine import (
    AdminOutcome,
    EngineBackpressure,
    EnginePoisoned,
    EngineState,
    ExecutionEngine,
    PreLaunchRejection,
    StaleStep,
)

pytestmark = pytest.mark.unit

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_SAMPLING = SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0)


class StubExecutor:
    """Deterministic model-side stub: emits one token per row."""

    def __init__(self) -> None:
        self.calls = 0
        self.fail_next = False
        self.reject_prepare = False

    def prepare(self, batch, sessions):
        if self.reject_prepare:
            raise PreLaunchRejection("injected reservation exhaustion")
        return (batch, sessions)

    def launch(self, prepared):
        batch, sessions = prepared
        self.calls += 1
        if self.fail_next:
            raise RuntimeError("injected device failure")
        results = []
        deltas = []
        for row, session in zip(batch.rows, sessions):
            results.append(
                RowResult(
                    row_id=row.row_id,
                    request_id=row.session.request_id,
                    incarnation=row.session.incarnation,
                    status=RowStatus.OK,
                    sampled_tokens=(int(session.history_length) + 1,),
                    accepted_candidates=0,
                    logprobs=(),
                    terminal=TerminalStatus.ACTIVE,
                )
            )
            deltas.append(
                SessionDelta(
                    source=session.ref,
                    next_version=session.ref.session_version + 1,
                    history_length_after=session.history_length + 1,
                    flow_coordinate_after=0,
                    rng_advance=1,
                    history_append=(int(session.history_length) + 1,),
                    cache_leases_added=(),
                    cache_leases_released=(),
                    product_leases_added=(),
                    product_leases_released=(),
                    terminal=TerminalStatus.ACTIVE,
                )
            )
        return tuple(results), tuple(deltas)


def _engine(window: int = 4) -> tuple[ExecutionEngine, StubExecutor]:
    executor = StubExecutor()
    engine = ExecutionEngine(
        engine=_ENGINE,
        executor=executor,
        advertised_operations=frozenset(OperationTag),
        replay_window=window,
    )
    return engine, executor


def _admission_row(request_id: int, *, row_id: int = 0) -> ExecuteRow:
    return ExecuteRow(
        row_id=row_id,
        session=SessionRef(_ENGINE, request_id, 1, 0),
        operation=SequenceStep((11, 12), 0, 0, 1),
        admission=NewSession(request_id, 1, _SAMPLING, 42, 64),
        cache_leases=(),
        product_leases=(),
        scheduler_op_id=row_id,
    )


def _advance_row(request_id: int, version: int, *, row_id: int = 0) -> ExecuteRow:
    return ExecuteRow(
        row_id=row_id,
        session=SessionRef(_ENGINE, request_id, 1, version),
        operation=SequenceStep((13,), version, version, 1),
        admission=None,
        cache_leases=(),
        product_leases=(),
        scheduler_op_id=row_id,
    )


def _batch(step_id: int, *rows: ExecuteRow, ack: int = -1) -> ExecuteBatch:
    return ExecuteBatch(
        engine_epoch=_ENGINE.engine_epoch,
        step_id=step_id,
        acknowledged_through=ack,
        rows=tuple(rows),
    )


def test_admission_transaction_commits_and_result_preserves_row_order():
    engine, executor = _engine()
    receipt = engine.execute(_batch(0, _admission_row(41), _admission_row(42, row_id=1)))
    assert receipt.ready()
    result = receipt.result()
    assert [row.row_id for row in result.row_results] == [0, 1]
    assert [delta.next_version for delta in result.session_deltas] == [1, 1]
    assert executor.calls == 1
    # The committed snapshots are now resolvable at version 1.
    engine.execute(_batch(1, _advance_row(41, 1), _advance_row(42, 1, row_id=1)))


def test_duplicate_step_returns_retained_result_without_reexecution():
    engine, executor = _engine()
    batch = _batch(0, _admission_row(41))
    first = engine.execute(batch).result()
    duplicate = engine.execute(batch).result()
    assert duplicate == first
    assert executor.calls == 1


def test_duplicate_with_conflicting_payload_poisons_the_epoch():
    engine, _ = _engine()
    engine.execute(_batch(0, _admission_row(41)))
    conflicting = _batch(0, _admission_row(43))
    with pytest.raises(EnginePoisoned):
        engine.execute(conflicting)
    assert engine.state is EngineState.POISONED
    with pytest.raises(EnginePoisoned):
        engine.execute(_batch(1, _advance_row(41, 1)))


def test_acknowledgement_evicts_and_stale_steps_cannot_reexecute():
    engine, _ = _engine(window=2)
    engine.execute(_batch(0, _admission_row(41)))
    engine.execute(_batch(1, _advance_row(41, 1)))
    # Window full without acknowledgement.
    with pytest.raises(EngineBackpressure):
        engine.execute(_batch(2, _advance_row(41, 2)))
    # Acknowledging both frees the window; the acknowledged step is stale
    # (an older envelope on the retry is a harmless no-op).
    engine.execute(_batch(2, _advance_row(41, 2), ack=1))
    with pytest.raises(StaleStep):
        engine.execute(_batch(1, _advance_row(41, 2), ack=0))


def test_first_unseen_step_must_be_the_next_sequence_number():
    engine, _ = _engine()
    engine.execute(_batch(0, _admission_row(41)))
    with pytest.raises(StaleStep, match="first unseen"):
        engine.execute(_batch(5, _advance_row(41, 1)))


def test_acknowledgement_never_regresses_and_batches_cannot_ack_themselves():
    engine, _ = _engine()
    engine.execute(_batch(0, _admission_row(41)))
    engine.execute(_batch(1, _advance_row(41, 1), ack=0))
    with pytest.raises(ExecutionContractError, match="own or a future"):
        engine.execute(_batch(2, _advance_row(41, 2), ack=2))
    # A stale (older) envelope is a no-op, not a regression.
    engine.execute(_batch(2, _advance_row(41, 2), ack=-1))
    with pytest.raises(StaleStep):
        engine.execute(_batch(0, _advance_row(41, 3)))


def test_prelaunch_rejection_is_noncommitted_and_retryable():
    engine, executor = _engine()
    stale_version = _batch(0, _advance_row(41, 3))
    with pytest.raises(Exception, match="no committed session"):
        engine.execute(stale_version)
    assert executor.calls == 0
    assert engine.state is EngineState.READY
    # The same step id retries successfully after correction.
    receipt = engine.execute(_batch(0, _admission_row(41)))
    assert receipt.ready()


def test_post_acceptance_failure_poisons_the_epoch():
    engine, executor = _engine()
    executor.fail_next = True
    with pytest.raises(EnginePoisoned, match="after acceptance"):
        engine.execute(_batch(0, _admission_row(41)))
    assert engine.state is EngineState.POISONED


def test_executor_prepare_rejection_is_noncommitted_and_retryable():
    engine, executor = _engine()
    executor.reject_prepare = True
    with pytest.raises(PreLaunchRejection):
        engine.execute(_batch(0, _admission_row(41)))
    assert engine.state is EngineState.READY
    assert executor.calls == 0
    executor.reject_prepare = False
    # The same step retries once the pre-launch condition clears.
    receipt = engine.execute(_batch(0, _admission_row(41)))
    assert receipt.ready()


def test_commands_execute_at_transaction_boundaries_only():
    engine, _ = _engine()
    engine.execute(_batch(0, _admission_row(41)))
    stale_drop = DropSession(_ENGINE, 41, incarnation=2, min_committed_version=1)
    assert engine.apply(stale_drop) is AdminOutcome.STALE
    exact_drop = DropSession(_ENGINE, 41, incarnation=1, min_committed_version=1)
    assert engine.apply(exact_drop) is AdminOutcome.APPLIED
    assert engine.apply(Quiesce(_ENGINE)) is AdminOutcome.APPLIED
    with pytest.raises(EngineBackpressure, match="QUIESCED"):
        engine.execute(_batch(1, _admission_row(44)))
    assert engine.apply(Resume(_ENGINE)) is AdminOutcome.APPLIED
    engine.execute(_batch(1, _admission_row(44)))


def test_foreign_epoch_batches_are_rejected():
    engine, _ = _engine()
    foreign = dataclasses.replace(_batch(0, _admission_row(41)), engine_epoch=99)
    with pytest.raises(ExecutionContractError, match="epoch"):
        engine.execute(foreign)
