"""Exactly-once transaction lifecycle for model-backed execution.

The engine owns execution identity, replay retention, session admission and
commit, administrative state, and epoch poisoning. Numerical preparation and
device work remain behind the torch-free TransactionExecutor protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from uniserve_worker.contracts.execution import (
    DropSession,
    EngineCommand,
    EngineRef,
    ExecuteBatch,
    ExecuteResult,
    ExecuteRow,
    ExecutionContractError,
    OperationTag,
    Quiesce,
    Resume,
    RowResult,
    SessionDelta,
    canonical_payload_fingerprint,
    validate_execute_batch,
)
from uniserve_worker.runtime.immutable_session import DropOutcome, RequestSession, SessionRegistry


class EngineState(IntEnum):
    READY = 1
    QUIESCED = 2
    POISONED = 3


class EngineExecutionError(RuntimeError):
    """Base class for typed engine failures."""


class EngineBackpressure(EngineExecutionError):
    """Retryable: the replay window or the single transaction slot is busy."""


class StaleStep(EngineExecutionError):
    """The step is at or below cumulative acknowledgement, or out of order."""


class PayloadConflict(EngineExecutionError):
    """A retained step arrived with a different canonical fingerprint."""


class EnginePoisoned(EngineExecutionError):
    """The epoch is poisoned; the engine requires controlled reconstruction."""


class PreLaunchRejection(EngineExecutionError):
    """Typed noncommitted pre-launch failure: no record, retry permitted."""


class TransactionExecutor(Protocol):
    """Internal seam for the model-backed half of one transaction.

    ``prepare`` performs every fallible step — lowering, capacity selection,
    residency reservation, lease pinning, metadata refresh — and may raise
    :class:`PreLaunchRejection` or :class:`EngineBackpressure`; the engine
    treats those as noncommitted (no replay record, the step may retry).

    ``launch`` replays the prepared transaction and returns row results and
    authoritative deltas in row order. Acceptance has happened by then: any
    exception from ``launch`` poisons the epoch, and the executor must
    abort its reservation before raising.
    """

    def prepare(
        self,
        batch: ExecuteBatch,
        sessions: tuple[RequestSession, ...],
    ) -> object: ...

    def launch(
        self,
        prepared: object,
    ) -> tuple[tuple[RowResult, ...], tuple[SessionDelta, ...]]: ...


class _RecordState(IntEnum):
    SUBMITTED = 1
    COMMITTED = 2


@dataclass
class _ReplayRecord:
    step_id: int
    fingerprint: str
    state: _RecordState
    result: ExecuteResult | None


class _Receipt:
    """Engine-owned receipt: no callbacks, sessions, or graph state."""

    def __init__(self, engine: "ExecutionEngine", step_id: int) -> None:
        self._engine = engine
        self._step_id = step_id

    def ready(self) -> bool:
        record = self._engine._records.get(self._step_id)
        return record is not None and record.state is _RecordState.COMMITTED

    def result(self) -> ExecuteResult:
        record = self._engine._records.get(self._step_id)
        if record is None or record.result is None:
            raise EngineExecutionError(f"step {self._step_id} has no durable result in the window")
        return record.result


class AdminOutcome(IntEnum):
    APPLIED = 1
    STALE = 2
    UNSUPPORTED = 3


class ExecutionEngine:
    """The sole model-backed data-plane interface.

    ``execute`` is the only execution method and ``apply`` the only
    administrative seam; there are no public methods for planning, staging,
    per-operation forwarding, graph selection, or postprocessing.
    """

    def __init__(
        self,
        *,
        engine: EngineRef,
        executor: TransactionExecutor,
        advertised_operations: frozenset[OperationTag],
        replay_window: int = 8,
    ) -> None:
        if replay_window <= 0:
            raise ExecutionContractError("replay window needs at least one record")
        self.engine = engine
        self.sessions = SessionRegistry(engine)
        self._executor = executor
        self._advertised = advertised_operations
        self._window_capacity = replay_window
        self._records: dict[int, _ReplayRecord] = {}
        # -1 is the "none acknowledged" sentinel; acknowledgement is a step id.
        self._acknowledged_through = -1
        self._next_step = 0
        self._state = EngineState.READY
        self._unresolved_step: int | None = None

    # ------------------------------------------------------------------ #
    # Data plane.
    # ------------------------------------------------------------------ #

    def execute(self, batch: ExecuteBatch) -> _Receipt:
        self._require_state(EngineState.READY)
        if batch.engine_epoch != self.engine.engine_epoch:
            raise ExecutionContractError(
                f"batch names epoch {batch.engine_epoch}; engine is {self.engine.engine_epoch}"
            )
        if batch.acknowledged_through >= batch.step_id:
            raise ExecutionContractError("a batch cannot acknowledge its own or a future step")
        self._apply_acknowledgement(batch.acknowledged_through)
        retained = self._records.get(batch.step_id)
        if retained is not None:
            fingerprint = canonical_payload_fingerprint(batch)
            if fingerprint != retained.fingerprint:
                self._poison(f"step {batch.step_id} retried with a conflicting payload")
            return _Receipt(self, batch.step_id)
        if batch.step_id <= self._acknowledged_through:
            raise StaleStep(
                f"step {batch.step_id} is at or below acknowledgement "
                f"{self._acknowledged_through} and cannot re-execute"
            )
        if batch.step_id != self._next_step:
            raise StaleStep(f"first unseen step must be {self._next_step}; got {batch.step_id}")
        if self._unresolved_step is not None:
            raise EngineBackpressure(f"transaction {self._unresolved_step} is still unresolved")
        if len(self._records) >= self._window_capacity:
            raise EngineBackpressure("replay window is full; acknowledge completed steps first")
        # Pre-launch phase: every failure through prepare() is noncommitted
        # and leaves no record, so the scheduler may retry the same step
        # after correction (or after backpressure clears).
        validate_execute_batch(batch, advertised_operations=self._advertised)
        provisional, snapshots = self._resolve_sessions(batch.rows)
        prepared = self._executor.prepare(batch, snapshots)
        # Acceptance: from here the identity is retained and never returns to
        # an unobserved state within the epoch.
        fingerprint = canonical_payload_fingerprint(batch)
        record = _ReplayRecord(
            step_id=batch.step_id,
            fingerprint=fingerprint,
            state=_RecordState.SUBMITTED,
            result=None,
        )
        self._records[batch.step_id] = record
        self._next_step = batch.step_id + 1
        self._unresolved_step = batch.step_id
        try:
            row_results, deltas = self._executor.launch(prepared)
            result = self._finalize(batch, row_results, deltas, provisional)
        except EnginePoisoned:
            raise
        except Exception as error:  # noqa: BLE001 — any post-acceptance failure
            self._poison(f"transaction {batch.step_id} failed after acceptance: {error}")
        record.result = result
        record.state = _RecordState.COMMITTED
        self._unresolved_step = None
        return _Receipt(self, batch.step_id)

    # ------------------------------------------------------------------ #
    # Administration (transaction boundaries only).
    # ------------------------------------------------------------------ #

    def apply(self, command: EngineCommand) -> AdminOutcome:
        if self._state is EngineState.POISONED:
            raise EnginePoisoned("poisoned epochs accept no commands")
        if self._unresolved_step is not None:
            raise EngineBackpressure("commands execute at transaction boundaries")
        if isinstance(command, DropSession):
            outcome = self.sessions.drop(command)
            return AdminOutcome.APPLIED if outcome is DropOutcome.DROPPED else AdminOutcome.STALE
        if isinstance(command, Quiesce):
            self._state = EngineState.QUIESCED
            return AdminOutcome.APPLIED
        if isinstance(command, Resume):
            if self._state is EngineState.QUIESCED:
                self._state = EngineState.READY
            return AdminOutcome.APPLIED
        # This engine instance has no residency-backed command executor.
        return AdminOutcome.UNSUPPORTED

    @property
    def state(self) -> EngineState:
        return self._state

    # ------------------------------------------------------------------ #
    # Internal transitions.
    # ------------------------------------------------------------------ #

    def _require_state(self, required: EngineState) -> None:
        if self._state is EngineState.POISONED:
            raise EnginePoisoned("the epoch is poisoned; reconstruct the engine")
        if self._state is not required:
            raise EngineBackpressure(f"engine is {self._state.name}, not READY")

    def _apply_acknowledgement(self, acknowledged_through: int) -> None:
        # Effective acknowledgement never regresses: a retry carrying an older
        # envelope is a no-op, so repeating a retained step stays harmless.
        if acknowledged_through <= self._acknowledged_through:
            return
        for step_id in sorted(self._records):
            if step_id > acknowledged_through:
                break
            record = self._records[step_id]
            if record.state is not _RecordState.COMMITTED:
                raise ExecutionContractError(
                    f"acknowledgement cannot release unresolved step {step_id}"
                )
            del self._records[step_id]
        self._acknowledged_through = acknowledged_through

    def _resolve_sessions(
        self,
        rows: tuple[ExecuteRow, ...],
    ) -> tuple[dict[int, RequestSession], tuple[RequestSession, ...]]:
        provisional: dict[int, RequestSession] = {}
        snapshots: list[RequestSession] = []
        for row in rows:
            if row.admission is not None:
                snapshot = self.sessions.admit(row.admission)
                provisional[row.row_id] = snapshot
            else:
                snapshot = self.sessions.resolve(row.session)
            snapshots.append(snapshot)
        return provisional, tuple(snapshots)

    def _finalize(
        self,
        batch: ExecuteBatch,
        row_results: tuple[RowResult, ...],
        deltas: tuple[SessionDelta, ...],
        provisional: dict[int, RequestSession],
    ) -> ExecuteResult:
        if len(row_results) != len(batch.rows) or len(deltas) != len(batch.rows):
            raise ExecutionContractError("executor must return one result and one delta per row")
        for index, (row, result) in enumerate(zip(batch.rows, row_results)):
            if result.row_id != index:
                raise ExecutionContractError(f"row result {index} is out of scheduler order")
            if (
                result.request_id != row.session.request_id
                or result.incarnation != row.session.incarnation
            ):
                raise ExecutionContractError(
                    f"row result {index} names a foreign request incarnation"
                )
        for row, delta in zip(batch.rows, deltas):
            self.sessions.apply(delta, provisional=provisional.get(row.row_id))
        return ExecuteResult(
            engine_epoch=batch.engine_epoch,
            step_id=batch.step_id,
            row_results=row_results,
            session_deltas=deltas,
        )

    def _poison(self, reason: str) -> None:
        self._state = EngineState.POISONED
        self._unresolved_step = None
        raise EnginePoisoned(reason)
