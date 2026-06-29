"""Typed worker error taxonomy.

Every failure is classified into a stable error class with ``code``, ``message``,
``retryable``, ``cleanup`` (whether the host must drop request state), and
``fatal`` (whether the worker process must be torn down).

``to_wire()`` produces ``{"kind": "error", "message", "code", "retryable",
"fatal", ...}``. Wire fields are scalars and short strings only — never tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NamedTuple

__all__ = [
    "ErrorCode",
    "ErrorPolicy",
    "WorkerError",
    "classify",
    "should_capture_trace",
    "invalid_descriptor",
    "capability_mismatch",
    "unsupported_operation",
    "unsupported_control",
    "model_execution_error",
    "resource_lease_violation",
    "scheduler_bug",
    "distributed_setup_error",
]


class ErrorCode(StrEnum):
    """Stable wire-string error class identifiers.

    Members are ``str`` values: they compare equal to and hash like their wire
    string, so existing ``== "UnsupportedOperation"`` checks, dict lookups by
    raw string, and serialization continue to work.
    """

    UNSUPPORTED_OPERATION = "UnsupportedOperation"
    UNSUPPORTED_CONTROL = "UnsupportedControl"
    INVALID_DESCRIPTOR = "InvalidDescriptor"
    CAPABILITY_MISMATCH = "CapabilityMismatch"
    RESOURCE_LEASE_VIOLATION = "ResourceLeaseViolation"
    WORKER_OOM = "WorkerOOM"
    BACKEND_OOM = "BackendOOM"
    MODEL_EXECUTION_ERROR = "ModelExecutionError"
    INVARIANT_VIOLATION = "InvariantViolation"
    TRANSIENT_WORKER_FAILURE = "TransientWorkerFailure"
    FATAL_WORKER_FAILURE = "FatalWorkerFailure"
    USER_INPUT_ERROR = "UserInputError"
    SCHEDULER_BUG = "SchedulerBug"


class ErrorPolicy(NamedTuple):
    """Per-class behavior. ``fatal`` means the worker can no longer be trusted to
    serve further requests and the host should tear it down; non-fatal errors
    fail only the offending request/op. ``capture_trace`` marks the severe subset
    whose log line should include a stack trace.
    """

    retryable: bool
    fatal: bool
    capture_trace: bool


# ``capture_trace`` here is the single source for "which errors warrant a
# traceback" so request-handling code never re-lists that set by hand.
_DEFAULT_POLICY = ErrorPolicy(retryable=False, fatal=False, capture_trace=False)

_POLICY: dict[ErrorCode, ErrorPolicy] = {
    ErrorCode.UNSUPPORTED_OPERATION: ErrorPolicy(False, False, False),
    ErrorCode.UNSUPPORTED_CONTROL: ErrorPolicy(False, False, False),
    ErrorCode.INVALID_DESCRIPTOR: ErrorPolicy(False, False, False),
    ErrorCode.CAPABILITY_MISMATCH: ErrorPolicy(False, False, False),
    ErrorCode.RESOURCE_LEASE_VIOLATION: ErrorPolicy(False, False, False),
    ErrorCode.WORKER_OOM: ErrorPolicy(True, False, True),
    ErrorCode.BACKEND_OOM: ErrorPolicy(True, False, True),
    ErrorCode.MODEL_EXECUTION_ERROR: ErrorPolicy(False, False, True),
    ErrorCode.INVARIANT_VIOLATION: ErrorPolicy(False, True, True),
    ErrorCode.TRANSIENT_WORKER_FAILURE: ErrorPolicy(True, False, False),
    ErrorCode.FATAL_WORKER_FAILURE: ErrorPolicy(False, True, True),
    ErrorCode.USER_INPUT_ERROR: ErrorPolicy(False, False, False),
    ErrorCode.SCHEDULER_BUG: ErrorPolicy(False, False, False),
}


def should_capture_trace(code: str) -> bool:
    """Whether an error of this class warrants a stack trace in its log line.

    The single source for that decision so the request handler shares one policy table.
    """
    return _POLICY.get(code, _DEFAULT_POLICY).capture_trace


@dataclass
class WorkerError(Exception):
    """A classified worker error. Raise it directly or build via ``classify``."""

    code: str
    message: str
    retryable: bool = False
    fatal: bool = False
    cleanup: bool = True
    req_id: int | None = None
    op_id: int | None = None
    op_kind: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code}: {self.message}")

    def to_wire(self) -> dict[str, Any]:
        # Only the fields modeled on the Rust WorkerResponse cross the wire.
        # Richer context (cleanup, req_id, op_id, op_kind, details) stays local
        # for logging/metrics. ``code`` crosses as a plain ``str`` so the wire
        # bytes are independent of whether it was built from an ``ErrorCode``.
        return {
            "kind": "error",
            "code": str(self.code),
            "message": self.message,
            "retryable": bool(self.retryable),
            "fatal": bool(self.fatal),
        }


def _make(code: str, message: str, **kw: Any) -> WorkerError:
    policy = _POLICY.get(code, _DEFAULT_POLICY)
    kw.setdefault("retryable", policy.retryable)
    kw.setdefault("fatal", policy.fatal)
    return WorkerError(code=code, message=message, **kw)


# OOM signals. We cannot import torch here (errors must classify without GPU
# deps), so OOM is detected structurally first — by walking the exception's
# class hierarchy names, which catches ``torch.cuda.OutOfMemoryError`` and any
# subclass/alias regardless of the leaf name — and only then by message text as
# a fallback for the common case where CUDA OOM surfaces as a plain
# ``RuntimeError`` carrying "CUDA out of memory" / "out of memory".
_OOM_TYPE_TOKENS = ("outofmemory",)
_OOM_TEXT_TOKENS = ("out of memory", "cuda oom", "cublas_status_alloc_failed")

# Context-corrupting CUDA failures: once the CUDA context hits one of these the
# worker can no longer be trusted to serve any further request, so it is
# classified FATAL (host tears the worker down) rather than failing only the
# offending op. Matched by message text since the leaf exception is usually a
# plain RuntimeError.
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
    for cls in type(exc).__mro__:
        lname = cls.__name__.lower()
        if any(tok in lname for tok in _OOM_TYPE_TOKENS):
            return True
    return any(tok in lowered_msg for tok in _OOM_TEXT_TOKENS)


def _looks_like_fatal_cuda(lowered_msg: str) -> bool:
    return any(tok in lowered_msg for tok in _FATAL_CUDA_TEXT_TOKENS)


def unsupported_control(name: str) -> WorkerError:
    return _make(
        ErrorCode.UNSUPPORTED_CONTROL,
        f"control {name!r} is not supported by this worker",
        cleanup=False,
    )


def unsupported_operation(kind: str, req_id: int | None = None) -> WorkerError:
    return _make(
        ErrorCode.UNSUPPORTED_OPERATION,
        f"op kind {kind!r} is not supported by this worker",
        req_id=req_id,
        op_kind=kind,
    )


def invalid_descriptor(message: str, **kw: Any) -> WorkerError:
    return _make(ErrorCode.INVALID_DESCRIPTOR, message, **kw)


def capability_mismatch(message: str, **kw: Any) -> WorkerError:
    return _make(ErrorCode.CAPABILITY_MISMATCH, message, **kw)


def model_execution_error(message: str, **kw: Any) -> WorkerError:
    """A failure during model execution / internal worker state.

    This is the class :func:`classify` assigns to an otherwise-unclassified
    exception, so routing a bare ``RuntimeError`` through here is wire-equivalent
    while removing the un-taxonomized raise.
    """
    return _make(ErrorCode.MODEL_EXECUTION_ERROR, message, **kw)


def resource_lease_violation(message: str, **kw: Any) -> WorkerError:
    return _make(ErrorCode.RESOURCE_LEASE_VIOLATION, message, **kw)


def scheduler_bug(message: str, **kw: Any) -> WorkerError:
    return _make(ErrorCode.SCHEDULER_BUG, message, **kw)


def distributed_setup_error(message: str, **kw: Any) -> WorkerError:
    """Tensor-parallel/distributed initialization failure.

    Routed through the taxonomy as a capability mismatch: the worker cannot
    provide the requested multi-rank topology (missing torch.distributed, a
    misconfigured world size, or no rendezvous address).
    """
    return _make(ErrorCode.CAPABILITY_MISMATCH, message, **kw)


# Ordered exception -> code rules, evaluated top to bottom; the first matching
# predicate wins, so precedence is positional. Each predicate takes the
# exception and its lowered message. The ordering is load-bearing:
#   - fatal-CUDA before OOM: a context-corrupting CUDA error is FATAL even when
#     it also mentions "out of memory" (the worker cannot serve further work).
#   - the typed checks (NotImplementedError / wire-decode errors / AssertionError)
#     follow the text/hierarchy heuristics.
# An exception matching no rule falls through to MODEL_EXECUTION_ERROR below.
_CLASSIFY_RULES: list[tuple[Any, str]] = [
    (lambda exc, lowered: _looks_like_fatal_cuda(lowered), ErrorCode.FATAL_WORKER_FAILURE),
    (lambda exc, lowered: _looks_like_oom(exc, lowered), ErrorCode.WORKER_OOM),
    (lambda exc, lowered: isinstance(exc, NotImplementedError), ErrorCode.UNSUPPORTED_OPERATION),
    # malformed op/descriptor decoded from the wire
    (
        lambda exc, lowered: isinstance(exc, (KeyError, IndexError, TypeError, ValueError)),
        ErrorCode.INVALID_DESCRIPTOR,
    ),
    (lambda exc, lowered: isinstance(exc, AssertionError), ErrorCode.INVARIANT_VIOLATION),
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
    code = ErrorCode.MODEL_EXECUTION_ERROR
    for predicate, rule_code in _CLASSIFY_RULES:
        if predicate(exc, lowered):
            code = rule_code
            break
    return _make(code, msg, **kw)
