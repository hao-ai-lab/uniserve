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
    """Identifies the validation, staging, execution, commit, discard, and cleanup phases of a run."""

    INPUT_VALIDATION = "input_validation"
    CANDIDATE_STAGE = "candidate_stage"
    ROUTE_EXECUTION = "route_execution"
    FORWARD_COMPLETION = "forward_completion"
    COMMIT = "commit"
    CANDIDATE_DISCARD = "candidate_discard"
    CLEANUP = "cleanup"


@dataclass(frozen=True, slots=True)
class OperationTrace:
    """Identifies one operation version without recording request content."""

    authority_id: int
    request_id: int
    epoch: int
    op_id: int
    version: int


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """Records a content-free execution phase, affected operations, route, timing, and failure class."""

    phase: ExecutionPhase
    operations: tuple[OperationTrace, ...]
    model_name: str
    timestamp_ns: int
    duration_us: int | None = None
    route: str | None = None
    row_kind_counts: tuple[tuple[str, int], ...] = ()
    error_class: str | None = None
    execution_path: str | None = None

    def to_mapping(self) -> dict[str, object]:
        """Serialize content-free phase identity, operation keys, route, timing, and error class."""

        return {
            "phase": self.phase.value,
            "operations": [
                {
                    "authority_id": value.authority_id,
                    "request_id": value.request_id,
                    "epoch": value.epoch,
                    "op_id": value.op_id,
                    "version": value.version,
                }
                for value in self.operations
            ],
            "model_name": self.model_name,
            "timestamp_ns": self.timestamp_ns,
            "duration_us": self.duration_us,
            "route": self.route,
            "row_kind_counts": dict(self.row_kind_counts),
            "error_class": self.error_class,
            "execution_path": self.execution_path,
        }


class ExecutionTrace:
    """Worker-owned bounded event stream and structured-log publisher."""

    def __init__(self, model_name: str, *, capacity: int = 4096) -> None:
        """Create a thread-safe bounded event trace for one model execution root."""

        if int(capacity) < 1:
            raise ValueError("execution trace capacity must be positive")
        self.model_name = str(model_name)
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
        """Append one bounded content-free execution event when tracing is enabled."""

        counts = tuple(
            sorted(
                (str(name), int(count))
                for name, count in ({} if row_kind_counts is None else row_kind_counts).items()
            )
        )
        event = ExecutionEvent(
            phase=phase,
            operations=tuple(operations),
            model_name=self.model_name,
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
        # single multi-image request, so emit the IPC trace only when DEBUG is
        # actually enabled. The guard also skips the ``to_mapping`` serialization,
        # keeping this off the hot path at the default INFO level. The in-memory
        # trace above remains the durable record.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("model execution phase", extra={"execution_event": event.to_mapping()})
        return event

    def snapshot(self) -> tuple[ExecutionEvent, ...]:
        """Copy the bounded event sequence under the trace lock."""

        with self._lock:
            return tuple(self._events)


__all__ = ["ExecutionEvent", "ExecutionPhase", "ExecutionTrace", "OperationTrace"]
