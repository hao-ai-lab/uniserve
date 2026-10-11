"""Typed worker error taxonomy.

Every worker failure is represented as a `WorkerError` carrying a stable
``code`` (`ErrorCode`), a ``message``, and ``fatal`` (whether the
worker process is unsafe for further work and must be torn down). Worker code
raises one directly through the constructors here, and `classify` maps any
other exception onto the taxonomy where failures are caught (in `Service`,
`Executor`, `ModelExecutor`, and `uniserve_worker.execution.step`).

``WorkerError.to_mapping`` produces the IPC error response fields that
`uniserve_worker.protocol.messages.error_response` sends to the engine, where
the PyO3 extension decodes them into ``WorkerResponseError``. Those fields
are scalars, short strings, and call identities, never tensors. The codes are
the ones the call completions carry, the extension's own ``ErrorCode`` (the
worker-ipc schema's enum). `_POLICY` defines each code's default fatality and
whether its log record carries a stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from uniserve.runtime import EventPoolError
from uniserve_worker._uniserve_ipc import ErrorCode

if TYPE_CHECKING:
    from uniserve_worker.protocol.identity import CallId

__all__ = [
    "ErrorCode",
    "ErrorPolicy",
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


class ErrorPolicy(NamedTuple):
    """Define the handling policy for an error class.

    Fatal errors leave the worker unsafe for further requests and require host
    teardown; non-fatal errors fail only the offending request or call.
    ``capture_trace`` marks errors whose log record includes a stack trace.

    ``fatal`` is only a default: `_make`, the `WorkerError` subclasses, and
    `classify` (for an exception it wraps) apply it unless the caller passes
    ``fatal``. Constructing `WorkerError` directly does not consult the
    policy and defaults to non-fatal.
    """

    fatal: bool
    capture_trace: bool


# ``capture_trace`` here is the single source for "which errors warrant a
# traceback" so request-handling code never re-lists that set by hand.
# `_DEFAULT_POLICY` applies to a code missing from `_POLICY`.
_DEFAULT_POLICY = ErrorPolicy(fatal=False, capture_trace=False)

_POLICY: dict[ErrorCode, ErrorPolicy] = {
    ErrorCode.UNSUPPORTED_CALL: ErrorPolicy(False, False),
    ErrorCode.INVALID_DESCRIPTOR: ErrorPolicy(False, False),
    ErrorCode.UNSUPPORTED_SETUP: ErrorPolicy(False, False),
    ErrorCode.INPUT_ERROR: ErrorPolicy(False, False),
    ErrorCode.COMPUTE_ERROR: ErrorPolicy(False, True),
    ErrorCode.RESOURCE_ERROR: ErrorPolicy(False, True),
    ErrorCode.INVARIANT_VIOLATION: ErrorPolicy(True, True),
    ErrorCode.FATAL_WORKER_FAILURE: ErrorPolicy(True, True),
}


def should_capture_trace(code: ErrorCode) -> bool:
    """Return whether an error of this class warrants a stack trace.

    `uniserve_worker.profiling.record_failure` and the batch failure log in
    `uniserve_worker.execution.step` consult this to choose between logging
    with a traceback and a warning without one.
    """
    return _POLICY.get(code, _DEFAULT_POLICY).capture_trace


@dataclass
class WorkerError(Exception):
    """A classified worker error.

    Raise it directly or build via ``classify``. Only ``code``, ``message``,
    ``fatal``, ``phase``, ``route`` and ``calls`` cross the IPC boundary (see
    ``to_mapping``); ``req_id``, ``call_id``, ``call_kind`` and ``details``
    stay in this process, where `record_failure` logs the first three.

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

    code: ErrorCode
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
        """Serialize the fields of an IPC error response.

        Returns:
            A mapping with ``kind`` (always ``"error"``), ``code``,
            ``message``, ``fatal``, ``phase``, ``route`` and ``calls``, each
            call as its request key and call id mapping. `error_response`
            drops ``kind`` in favor of the response envelope's own.
        """
        # Only the fields of the Rust ``WorkerResponseError`` cross the IPC
        # boundary; richer context (req_id, call_id, call_kind, details)
        # stays in this process.
        return {
            "kind": "error",
            "code": self.code,
            "message": self.message,
            "fatal": bool(self.fatal),
            "phase": self.phase,
            "route": self.route,
            "calls": [
                {
                    "request_key": {
                        "engine_id": engine_id,
                        "request_id": request_id,
                        "request_epoch": request_epoch,
                    },
                    "call_id": call_id.to_mapping(),
                }
                for (
                    engine_id,
                    request_id,
                    request_epoch,
                    call_id,
                ) in self.calls
            ],
        }


class InputError(WorkerError):
    """A staged execution input is invalid for its declared route."""

    def __init__(self, message: str, **kw: Any) -> None:
        """Create a request-scoped input failure.

        ``fatal`` defaults to the `_POLICY` entry for ``INPUT_ERROR`` unless
        the caller passes it; other keyword arguments set `WorkerError`
        fields.
        """
        policy = _POLICY[ErrorCode.INPUT_ERROR]
        kw.setdefault("fatal", policy.fatal)
        super().__init__(code=ErrorCode.INPUT_ERROR, message=message, **kw)


class ComputeError(WorkerError):
    """Neural execution or raw-output validation failed."""

    def __init__(self, message: str, **kw: Any) -> None:
        """Create a request-scoped execution failure.

        ``fatal`` defaults to the `_POLICY` entry for ``COMPUTE_ERROR`` unless
        the caller passes it; other keyword arguments set `WorkerError`
        fields.
        """
        policy = _POLICY[ErrorCode.COMPUTE_ERROR]
        kw.setdefault("fatal", policy.fatal)
        super().__init__(code=ErrorCode.COMPUTE_ERROR, message=message, **kw)


class ResourceError(WorkerError):
    """Graph, device, allocation, communication, or residency failed."""

    def __init__(self, message: str, **kw: Any) -> None:
        """Create a resource failure.

        ``fatal`` defaults to the `_POLICY` entry for ``RESOURCE_ERROR`` unless
        the caller passes it; other keyword arguments set `WorkerError`
        fields.
        """
        policy = _POLICY[ErrorCode.RESOURCE_ERROR]
        kw.setdefault("fatal", policy.fatal)
        super().__init__(code=ErrorCode.RESOURCE_ERROR, message=message, **kw)


def _make(code: ErrorCode, message: str, **kw: Any) -> WorkerError:
    """Construct a `WorkerError` whose ``fatal`` defaults to its policy.

    Keyword arguments set the other `WorkerError` fields, and an explicit
    ``fatal`` overrides the policy.
    """
    policy = _POLICY.get(code, _DEFAULT_POLICY)
    kw.setdefault("fatal", policy.fatal)
    return WorkerError(code=code, message=message, **kw)


# OOM is recognized by name rather than by exception type: any class in the
# exception's MRO whose lowercased name contains a type token (such as torch's
# ``OutOfMemoryError``), or a lowercased message of any exception type that
# contains a text token (these cover allocation failures raised as plain
# runtime errors).
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
    """Return whether an exception denotes memory exhaustion.

    True when a class in the exception's MRO has a name containing an
    `_OOM_TYPE_TOKENS` entry, or when ``lowered_msg`` contains an
    `_OOM_TEXT_TOKENS` entry.
    """
    for cls in type(exc).__mro__:
        lname = cls.__name__.lower()
        if any(tok in lname for tok in _OOM_TYPE_TOKENS):
            return True
    return any(tok in lowered_msg for tok in _OOM_TEXT_TOKENS)


def _looks_like_fatal_cuda(lowered_msg: str) -> bool:
    """Return whether a message denotes a context-corrupting CUDA failure.

    ``lowered_msg`` must already be lowercased; any `_FATAL_CUDA_TEXT_TOKENS`
    substring matches.
    """
    return any(tok in lowered_msg for tok in _FATAL_CUDA_TEXT_TOKENS)


def unsupported_call(kind: str, req_id: int | None = None) -> WorkerError:
    """Create a classified error for a call kind this worker cannot run."""
    return _make(
        ErrorCode.UNSUPPORTED_CALL,
        f"call kind {kind!r} is not supported by this worker",
        req_id=req_id,
        call_kind=kind,
    )


def invalid_descriptor(message: str, **kw: Any) -> WorkerError:
    """Create a classified error for malformed scheduler or transport input."""
    return _make(ErrorCode.INVALID_DESCRIPTOR, message, **kw)


def unsupported_setup(message: str, **kw: Any) -> WorkerError:
    """Create a classified error for a setup the runtime cannot provide.

    Used for launch configuration and for missing worker resources that a
    call requires, such as request block tables.
    """
    return _make(ErrorCode.UNSUPPORTED_SETUP, message, **kw)


def resource_error(message: str, **kw: Any) -> ResourceError:
    """Create a classified error for exhausted or unavailable resources."""
    return ResourceError(message, **kw)


# Ordered exception -> code rules, evaluated top to bottom; the first matching
# predicate wins, so precedence is positional. Each predicate takes the
# exception and its lowered message. The ordering is load-bearing:
#   - fatal-CUDA before OOM: a context-corrupting CUDA error is FATAL even when
#     it also mentions "out of memory" (the worker cannot serve further work).
#   - the typed checks (EventPoolError / NotImplementedError / the standard
#     lookup, type and value errors / AssertionError) follow the
#     text/hierarchy heuristics, so for example a ValueError whose message
#     mentions "out of memory" classifies as RESOURCE_ERROR.
# An exception matching no rule falls through to ComputeError in `classify`.
_CLASSIFY_RULES: list[tuple[Any, ErrorCode]] = [
    (
        lambda exc, lowered: _looks_like_fatal_cuda(lowered),
        ErrorCode.FATAL_WORKER_FAILURE,
    ),
    (
        lambda exc, lowered: _looks_like_oom(exc, lowered),
        ErrorCode.RESOURCE_ERROR,
    ),
    (
        lambda exc, lowered: isinstance(exc, EventPoolError),
        ErrorCode.INVARIANT_VIOLATION,
    ),
    (
        lambda exc, lowered: isinstance(exc, NotImplementedError),
        ErrorCode.UNSUPPORTED_CALL,
    ),
    # Treated as a malformed call or descriptor decoded from IPC. The rule
    # applies wherever the exception was raised, including during execution.
    (
        lambda exc, lowered: isinstance(
            exc, (KeyError, IndexError, TypeError, ValueError)
        ),
        ErrorCode.INPUT_ERROR,
    ),
    (
        lambda exc, lowered: isinstance(exc, AssertionError),
        ErrorCode.INVARIANT_VIOLATION,
    ),
]


def classify(
    exc: BaseException, *, context: str | None = None, **kw: Any
) -> WorkerError:
    """Map an arbitrary exception onto the taxonomy.

    An already-classified ``WorkerError`` is returned as the same object:
    each non-None ``kw`` value fills a field that is currently None, fields
    already set are kept (always so for ``fatal``, ``calls`` and ``details``,
    whose defaults are not None), and ``context`` is ignored.

    Any other exception becomes a new error whose message is the exception
    text (or its type name when empty), prefixed with ``"<context>: "`` when
    ``context`` is given; the first matching `_CLASSIFY_RULES` entry picks its
    code, ``COMPUTE_ERROR`` otherwise, and ``kw`` sets the remaining fields.
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

    # Rules match the context-prefixed message, not the bare exception text.
    lowered = msg.lower()
    code = ErrorCode.COMPUTE_ERROR
    for predicate, rule_code in _CLASSIFY_RULES:
        if predicate(exc, lowered):
            code = rule_code
            break
    if code is ErrorCode.INPUT_ERROR:
        return InputError(msg, **kw)
    if code is ErrorCode.COMPUTE_ERROR:
        return ComputeError(msg, **kw)
    if code is ErrorCode.RESOURCE_ERROR:
        return ResourceError(msg, **kw)
    return _make(code, msg, **kw)
