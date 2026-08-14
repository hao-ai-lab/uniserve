"""Structured, content-free observations of canonical execution phases."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

logger = logging.getLogger("uniserve.execution")


class ExecutionPhase(StrEnum):
    PROTOCOL_VALIDATION = "protocol_validation"
    CANDIDATE_STAGE = "candidate_stage"
    PLAN_CREATION = "plan_creation"
    ROUTE_EXECUTION = "route_execution"
    FORWARD_COMPLETION = "forward_completion"
    POSTPROCESS = "postprocess"
    COMMIT = "commit"
    CANDIDATE_DISCARD = "candidate_discard"
    REPLAY = "replay"
    CLEANUP = "cleanup"


@dataclass(frozen=True, slots=True)
class OperationTrace:
    session_id: int
    epoch: int
    op_id: int
    version: int


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    phase: ExecutionPhase
    operations: tuple[OperationTrace, ...]
    candidate_digest: str
    timestamp_ns: int
    duration_us: int | None = None
    route: str | None = None
    row_kind_counts: tuple[tuple[str, int], ...] = ()
    error_class: str | None = None
    execution_path: str | None = None

    def to_wire(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "operations": [
                {
                    "session_id": value.session_id,
                    "epoch": value.epoch,
                    "op_id": value.op_id,
                    "version": value.version,
                }
                for value in self.operations
            ],
            "candidate_digest": self.candidate_digest,
            "timestamp_ns": self.timestamp_ns,
            "duration_us": self.duration_us,
            "route": self.route,
            "row_kind_counts": dict(self.row_kind_counts),
            "error_class": self.error_class,
            "execution_path": self.execution_path,
        }


class ExecutionTrace:
    """Worker-owned bounded event stream and structured-log publisher."""

    def __init__(self, candidate_digest: str, *, capacity: int = 4096) -> None:
        if int(capacity) < 1:
            raise ValueError("execution trace capacity must be positive")
        self.candidate_digest = str(candidate_digest)
        self._events: deque[ExecutionEvent] = deque(maxlen=int(capacity))
        self._lock = RLock()

    def emit(
        self,
        phase: ExecutionPhase,
        operations: Sequence[OperationTrace],
        *,
        duration_us: int | None = None,
        route: str | None = None,
        row_kind_counts: Mapping[str, int] | None = None,
        error: BaseException | None = None,
        execution_path: str | None = None,
    ) -> ExecutionEvent:
        counts = tuple(
            sorted(
                (str(name), int(count))
                for name, count in ({} if row_kind_counts is None else row_kind_counts).items()
            )
        )
        event = ExecutionEvent(
            phase=phase,
            operations=tuple(operations),
            candidate_digest=self.candidate_digest,
            timestamp_ns=time.time_ns(),
            duration_us=None if duration_us is None else int(duration_us),
            route=None if route is None else str(route),
            row_kind_counts=counts,
            error_class=None if error is None else type(error).__name__,
            execution_path=execution_path,
        )
        with self._lock:
            self._events.append(event)
        # One record per execution phase runs tens of thousands of times in a
        # single multi-image request, so emit the wire trace only when DEBUG is
        # actually enabled. The guard also skips the ``to_wire`` serialization,
        # keeping this off the hot path at the default INFO level. The in-memory
        # trace above remains the durable record.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("model execution phase", extra={"execution_event": event.to_wire()})
        return event

    def snapshot(self) -> tuple[ExecutionEvent, ...]:
        with self._lock:
            return tuple(self._events)


__all__ = ["ExecutionEvent", "ExecutionPhase", "ExecutionTrace", "OperationTrace"]
