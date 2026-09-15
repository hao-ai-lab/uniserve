"""Typed worker error taxonomy.

Every failure is classified into a stable error class with ``code``, ``message``,
``retryable``, and ``fatal`` (whether the worker process must be torn down).

``to_mapping()`` produces ``{"kind": "error", "message", "code", "retryable",
"fatal", ...}``. Message fields are scalars and short strings only — never tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple

from uniserve.runtime import EventPoolError

if TYPE_CHECKING:
    from uniserve_worker.protocol.identity import ComputationId

__all__ = [
    "WorkerErrorCode",
    "ErrorPolicy",
    "WorkerError",
    "InputError",
    "ComputeError",
    "ResourceError",
    "classify",
    "should_capture_trace",
    "invalid_descriptor",
    "unsupported_setup",
    "unsupported_operation",
    "unsupported_control",
    "compute_error",
    "resource_error",
]


class WorkerErrorCode(StrEnum):
    """Stable worker error identifiers.

    Members are strings because the IPC reply carries their names.
    """

    UNSUPPORTED_OPERATION = "UnsupportedOperation"
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


class ErrorPolicy(NamedTuple):
    """Define the handling policy for an error class.

    Fatal errors leave the worker unsafe for further requests and require host
    teardown; non-fatal errors fail only the offending request or operation.
    ``capture_trace`` marks errors whose log record includes a stack trace.
    """

    retryable: bool
    fatal: bool
    capture_trace: bool


# ``capture_trace`` here is the single source for "which errors warrant a
# traceback" so request-handling code never re-lists that set by hand.
_DEFAULT_POLICY = ErrorPolicy(retryable=False, fatal=False, capture_trace=False)

_POLICY: dict[WorkerErrorCode, ErrorPolicy] = {
    WorkerErrorCode.UNSUPPORTED_OPERATION: ErrorPolicy(False, False, False),
    WorkerErrorCode.UNSUPPORTED_CONTROL: ErrorPolicy(False, False, False),
    WorkerErrorCode.INVALID_DESCRIPTOR: ErrorPolicy(False, False, False),
    WorkerErrorCode.UNSUPPORTED_SETUP: ErrorPolicy(False, False, False),
    WorkerErrorCode.RESOURCE_LEASE_VIOLATION: ErrorPolicy(False, False, False),
    WorkerErrorCode.INPUT_ERROR: ErrorPolicy(False, False, False),
    WorkerErrorCode.COMPUTE_ERROR: ErrorPolicy(False, False, True),
    WorkerErrorCode.RESOURCE_ERROR: ErrorPolicy(True, False, True),
    WorkerErrorCode.INVARIANT_VIOLATION: ErrorPolicy(False, True, True),
    WorkerErrorCode.FATAL_WORKER_FAILURE: ErrorPolicy(False, True, True),
    WorkerErrorCode.SCHEDULER_BUG: ErrorPolicy(False, False, False),
}


def should_capture_trace(code: WorkerErrorCode) -> bool:
    """Return whether an error of this class warrants a stack trace in its log line.

    The single source for that decision so the request handler shares one policy table.
    """
    return _POLICY.get(code, _DEFAULT_POLICY).capture_trace


@dataclass
class WorkerError(Exception):
    """A classified worker error. Raise it directly or build via ``classify``."""

    code: WorkerErrorCode
    message: str
    retryable: bool = False
    fatal: bool = False
    req_id: int | None = None
    op_id: ComputationId | None = None
    op_kind: str | None = None
    phase: str | None = None
    route: str | None = None
    operations: tuple[tuple[int, int, int, ComputationId], ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the exception message from the classified worker error fields."""

        Exception.__init__(self, f"{self.code}: {self.message}")

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the stable error code, message, fatal flag, and optional operation context."""

        # Only the fields modeled on the Rust WorkerResponse cross the IPC boundary.
        # Richer context (req_id, op_id, op_kind, details) stays local
        # for logging and metrics.
        return {
            "kind": "error",
            "code": str(self.code),
            "message": self.message,
            "retryable": bool(self.retryable),
            "fatal": bool(self.fatal),
            "phase": self.phase,
            "route": self.route,
            "operations": [
                {
                    "request_key": {
                        "engine_id": engine_id,
                        "request_id": request_id,
                        "request_epoch": request_epoch,
                    },
                    "op_id": op_id.to_mapping(),
                }
                for engine_id, request_id, request_epoch, op_id in self.operations
            ],
        }


class InputError(WorkerError):
    """A staged execution input is invalid for its declared route."""

    def __init__(self, message: str, **kw: Any) -> None:
        """Create a request-scoped input failure with the configured retry policy."""

        policy = _POLICY[WorkerErrorCode.INPUT_ERROR]
        kw.setdefault("retryable", policy.retryable)
        kw.setdefault("fatal", policy.fatal)
        super().__init__(code=WorkerErrorCode.INPUT_ERROR, message=message, **kw)


class ComputeError(WorkerError):
    """Neural execution or raw-output validation failed."""

    def __init__(self, message: str, **kw: Any) -> None:
        """Create a request-scoped execution failure with the configured retry policy."""

        policy = _POLICY[WorkerErrorCode.COMPUTE_ERROR]
        kw.setdefault("retryable", policy.retryable)
        kw.setdefault("fatal", policy.fatal)
        super().__init__(code=WorkerErrorCode.COMPUTE_ERROR, message=message, **kw)


class ResourceError(WorkerError):
    """Graph, device, allocation, communication, or residency failed."""

    def __init__(self, message: str, **kw: Any) -> None:
        """Create a resource failure with the configured retry and fatality policy."""

        policy = _POLICY[WorkerErrorCode.RESOURCE_ERROR]
        kw.setdefault("retryable", policy.retryable)
        kw.setdefault("fatal", policy.fatal)
        super().__init__(code=WorkerErrorCode.RESOURCE_ERROR, message=message, **kw)


def _make(code: WorkerErrorCode, message: str, **kw: Any) -> WorkerError:
    """Construct a classified worker error with shared contextual fields."""

    policy = _POLICY.get(code, _DEFAULT_POLICY)
    kw.setdefault("retryable", policy.retryable)
    kw.setdefault("fatal", policy.fatal)
    return WorkerError(code=code, message=message, **kw)


# Error classification cannot import the GPU runtime. Detect OOM structurally
# through exception hierarchy names, then recognize plain runtime errors by
# their stable allocation-failure messages.
_OOM_TYPE_TOKENS = ("outofmemory",)
_OOM_TEXT_TOKENS = ("out of memory", "cuda oom", "cublas_status_alloc_failed")

# Context-corrupting CUDA failures leave the worker unsafe for further requests
# and require host teardown. Match their messages because the leaf exception is
# usually a plain RuntimeError.
_FATAL_CUDA_TEXT_TOKENS = (
    "illegal memory access",
    "cudaerrorillegaladdress",
    "an illegal memory access",
    "device-side assert",
    "device-side assertion",
    "cudaerrorlaunchfailure",
    "unspecified launch failure",
    "unrecoverable",
    "uncorrectable ecc",
    "misaligned address",
)


def _looks_like_oom(exc: BaseException, lowered_msg: str) -> bool:
    """Return whether an exception type or message denotes resource exhaustion."""

    for cls in type(exc).__mro__:
        lname = cls.__name__.lower()
        if any(tok in lname for tok in _OOM_TYPE_TOKENS):
            return True
    return any(tok in lowered_msg for tok in _OOM_TEXT_TOKENS)


def _looks_like_fatal_cuda(lowered_msg: str) -> bool:
    """Return whether an error message denotes an unrecoverable CUDA context failure."""

    return any(tok in lowered_msg for tok in _FATAL_CUDA_TEXT_TOKENS)


def unsupported_control(name: str) -> WorkerError:
    """Create a classified error for an unrecognized control-plane request."""

    return _make(
        WorkerErrorCode.UNSUPPORTED_CONTROL,
        f"control {name!r} is not supported by this worker",
    )


def unsupported_operation(kind: str, req_id: int | None = None) -> WorkerError:
    """Create a classified error for an operation kind unavailable on this worker."""

    return _make(
        WorkerErrorCode.UNSUPPORTED_OPERATION,
        f"op kind {kind!r} is not supported by this worker",
        req_id=req_id,
        op_kind=kind,
    )


def invalid_descriptor(message: str, **kw: Any) -> WorkerError:
    """Create a classified error for malformed scheduler or transport input."""

    return _make(WorkerErrorCode.INVALID_DESCRIPTOR, message, **kw)


def unsupported_setup(message: str, **kw: Any) -> WorkerError:
    """Create a classified error for launch configuration the runtime cannot provide."""

    return _make(WorkerErrorCode.UNSUPPORTED_SETUP, message, **kw)


def compute_error(message: str, **kw: Any) -> ComputeError:
    """Create a nonfatal classified error for model execution failure."""

    return ComputeError(message, **kw)


def resource_error(message: str, **kw: Any) -> ResourceError:
    """Create a classified error for exhausted or unavailable runtime resources."""

    return ResourceError(message, **kw)


# Ordered exception -> code rules, evaluated top to bottom; the first matching
# predicate wins, so precedence is positional. Each predicate takes the
# exception and its lowered message. The ordering is load-bearing:
#   - fatal-CUDA before OOM: a context-corrupting CUDA error is FATAL even when
#     it also mentions "out of memory" (the worker cannot serve further work).
#   - the typed checks (NotImplementedError / decode errors / AssertionError)
#     follow the text/hierarchy heuristics.
# An exception matching no rule falls through to ComputeError below.
_CLASSIFY_RULES: list[tuple[Any, WorkerErrorCode]] = [
    (lambda exc, lowered: _looks_like_fatal_cuda(lowered), WorkerErrorCode.FATAL_WORKER_FAILURE),
    (lambda exc, lowered: _looks_like_oom(exc, lowered), WorkerErrorCode.RESOURCE_ERROR),
    (
        lambda exc, lowered: isinstance(exc, EventPoolError),
        WorkerErrorCode.INVARIANT_VIOLATION,
    ),
    (
        lambda exc, lowered: isinstance(exc, NotImplementedError),
        WorkerErrorCode.UNSUPPORTED_OPERATION,
    ),
    # malformed operation or descriptor decoded from IPC
    (
        lambda exc, lowered: isinstance(exc, (KeyError, IndexError, TypeError, ValueError)),
        WorkerErrorCode.INPUT_ERROR,
    ),
    (
        lambda exc, lowered: isinstance(exc, AssertionError),
        WorkerErrorCode.INVARIANT_VIOLATION,
    ),
]


def classify(exc: BaseException, *, context: str | None = None, **kw: Any) -> WorkerError:
    """Map an arbitrary exception onto the taxonomy.

    Already-classified ``WorkerError``s pass through (callers may enrich ids).
    """
    if isinstance(exc, WorkerError):
        for k, v in kw.items():
            if getattr(exc, k, None) is None and v is not None:
                setattr(exc, k, v)
        return exc

    name = type(exc).__name__
    msg = str(exc) or name
    if context:
        msg = f"{context}: {msg}"

    # torch CUDA OOM (avoid importing torch here; match by class hierarchy + text).
    lowered = msg.lower()
    code = WorkerErrorCode.COMPUTE_ERROR
    for predicate, rule_code in _CLASSIFY_RULES:
        if predicate(exc, lowered):
            code = rule_code
            break
    if code is WorkerErrorCode.INPUT_ERROR:
        return InputError(msg, **kw)
    if code is WorkerErrorCode.COMPUTE_ERROR:
        return ComputeError(msg, **kw)
    if code is WorkerErrorCode.RESOURCE_ERROR:
        return ResourceError(msg, **kw)
    return _make(code, msg, **kw)
