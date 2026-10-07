"""Python exception values classified by the native worker.

The exception classes carry request context for Python callers. Rust owns
classification, fatality policy and the failure path through execution and IPC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from uniserve_worker._uniserve_ipc import classify_error as classify
from uniserve_worker._uniserve_ipc import (
    should_capture_trace as should_capture_trace,
)
from uniserve_worker._uniserve_ipc import worker_error_mapping

if TYPE_CHECKING:
    from uniserve_worker.protocol.identity import CallId

__all__ = [
    "WorkerErrorCode",
    "WorkerError",
    "InputError",
    "ComputeError",
    "ResourceError",
    "classify",
    "should_capture_trace",
    "invalid_descriptor",
    "unsupported_setup",
    "unsupported_call",
    "resource_error",
]


class WorkerErrorCode(StrEnum):
    """Stable worker error identifiers.

    Members are strings because the IPC error response carries the value
    (for example ``"InvariantViolation"``) as its ``code`` field. The engine
    matches on these values (``WorkerGroup::join_rank_errors``, for example,
    escalates ``SchedulerBug`` and ``InvariantViolation`` as it does fatal
    errors), so changing a value is a wire-protocol change.
    """

    UNSUPPORTED_CALL = "UnsupportedCall"
    UNSUPPORTED_CONTROL = "UnsupportedControl"
    INVALID_DESCRIPTOR = "InvalidDescriptor"
    UNSUPPORTED_SETUP = "UnsupportedSetup"
    RESOURCE_LEASE_VIOLATION = "ResourceLeaseViolation"
    INPUT_ERROR = "InputError"
    COMPUTE_ERROR = "ComputeError"
    RESOURCE_ERROR = "ResourceError"
    INVARIANT_VIOLATION = "InvariantViolation"
    FATAL_WORKER_FAILURE = "FatalWorkerFailure"
    SCHEDULER_BUG = "SchedulerBug"


@dataclass
class WorkerError(Exception):
    """A classified worker error.

    Raise it directly or build via ``classify``. Only ``code``, ``message``,
    ``fatal``, ``phase``, ``route`` and ``calls`` cross the IPC boundary (see
    ``to_mapping``); ``req_id``, ``call_id``, ``call_kind`` and ``details``
    stay in this process for diagnostic logging.

    Attributes:
        phase: Worker phase that failed, such as ``"batch registration"``.
        route: Call kind of the failed batch, such as ``"prefill"``, or the
            forward mode of the failed numerical call within it. None when
            the failure belongs to no batch with calls: an undecodable
            request, a submission refused at admission, or a lifecycle-only
            batch.
        calls: Affected calls as ``(engine_id, request_id, request_epoch,
            call_id)`` tuples.
    """

    code: WorkerErrorCode
    message: str
    fatal: bool = False
    req_id: int | None = None
    call_id: CallId | None = None
    call_kind: str | None = None
    phase: str | None = None
    route: str | None = None
    calls: tuple[tuple[int, int, int, CallId], ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set the exception arguments to ``"<code>: <message>"``.

        The dataclass-generated ``__init__`` does not call
        ``Exception.__init__``, so this is what gives ``str(error)`` its text.
        """
        Exception.__init__(self, f"{self.code}: {self.message}")

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the native error's code, message and affected calls."""
        return worker_error_mapping(self)


class InputError(WorkerError):
    """An execution input is invalid for its declared route."""

    def __init__(self, message: str, **kw: Any) -> None:
        """Create a request-scoped input failure.

        Keyword arguments attach request context; fatality defaults to false.
        """
        super().__init__(
            code=WorkerErrorCode.INPUT_ERROR, message=message, **kw
        )


class ComputeError(WorkerError):
    """Neural execution or raw-output validation failed."""

    def __init__(self, message: str, **kw: Any) -> None:
        """Create a request-scoped execution failure.

        Keyword arguments attach request context; fatality defaults to false.
        """
        super().__init__(
            code=WorkerErrorCode.COMPUTE_ERROR, message=message, **kw
        )


class ResourceError(WorkerError):
    """Graph, device, allocation, communication, or residency failed."""

    def __init__(self, message: str, **kw: Any) -> None:
        """Create a resource failure.

        Keyword arguments attach request context; fatality defaults to false.
        """
        super().__init__(
            code=WorkerErrorCode.RESOURCE_ERROR, message=message, **kw
        )


def unsupported_call(kind: str, req_id: int | None = None) -> WorkerError:
    """Create a classified error for a call kind this worker cannot run."""
    return WorkerError(
        WorkerErrorCode.UNSUPPORTED_CALL,
        f"call kind {kind!r} is not supported by this worker",
        req_id=req_id,
        call_kind=kind,
    )


def invalid_descriptor(message: str, **kw: Any) -> WorkerError:
    """Create a classified error for malformed scheduler or transport input."""
    return WorkerError(WorkerErrorCode.INVALID_DESCRIPTOR, message, **kw)


def unsupported_setup(message: str, **kw: Any) -> WorkerError:
    """Create a classified error for a setup the runtime cannot provide.

    Used for launch configuration and for missing worker resources that a
    call requires, such as request block tables.
    """
    return WorkerError(WorkerErrorCode.UNSUPPORTED_SETUP, message, **kw)


def resource_error(message: str, **kw: Any) -> ResourceError:
    """Create a classified error for exhausted or unavailable resources."""
    return ResourceError(message, **kw)
