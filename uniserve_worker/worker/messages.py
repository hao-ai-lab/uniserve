"""IPC request dependencies and protocol response envelopes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..bootstrap.worker_info import RequestKind, ResponseKind
from ..execution.batch_state import BatchState
from ..foundation.errors import WorkerError, invalid_descriptor
from ..protocol.batch import ScheduleBatch
from ..protocol.output import BatchOutput


def response(kind: ResponseKind, **payload: Any) -> dict[str, Any]:
    """Build a protocol response.

    With the canonical kind tag and payload fields.
    """
    # The envelope carries every field the wire protocol allows; payloads may
    # only override declared fields, leaving the rest at their null defaults.
    response: dict[str, Any] = {
        "kind": kind.value,
        "call_id": None,
        "info": None,
        "result": None,
        "message": None,
        "code": None,
        "retryable": None,
        "fatal": None,
        "phase": None,
        "route": None,
        "operations": [],
    }

    unknown = set(payload) - set(response)
    if unknown:
        raise invalid_descriptor(
            f"worker response contains unknown fields {sorted(unknown)!r}"
        )
    response.update(payload)
    return response


def required(request: Mapping[str, Any], field: str, kind: RequestKind) -> Any:
    """Return a required request field.

    Raise a classified descriptor error when it is absent.
    """
    value = request.get(field)
    if value is None:
        raise invalid_descriptor(
            f"request {kind.value!r} is missing required field {field!r}",
            op_kind=kind.value,
        )
    return value


def integer(request: Mapping[str, Any], field: str, kind: RequestKind) -> int:
    """Decode a required request field as a non-negative integer."""
    value = required(request, field, kind)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(
            f"request {kind.value!r} field {field!r} must be a "
            "non-negative integer",
            op_kind=kind.value,
        )
    return value


def request_kind(request: Mapping[str, Any]) -> RequestKind:
    """Decode and validate the request kind discriminator."""
    raw = request.get("kind")
    if not isinstance(raw, str):
        raise invalid_descriptor("worker request kind must be a string")
    try:
        return RequestKind(raw)
    except ValueError:
        raise invalid_descriptor(
            f"unknown worker request kind {raw!r}"
        ) from None


def run_requests(run: ScheduleBatch) -> frozenset[int]:
    """Collect request identifiers referenced by a run.

    Admissions, operations, and commands each reference request keys.
    """
    keys = (
        *(admission.request_key for admission in run.admissions),
        *(operation.request_key for operation in run.operations),
        *(command.request_key for command in run.commands),
    )
    return frozenset(int(key.request_id) for key in keys)


def raw_request_ids(request: Mapping[str, Any]) -> frozenset[int]:
    """Extract the request identifiers a command touches.

    Covers submit and lifecycle commands.
    """
    requests: set[int] = set()
    run = request.get("run")
    if isinstance(run, ScheduleBatch):
        return run_requests(run) | requests
    if not isinstance(run, Mapping):
        return frozenset(requests)

    # Walk the raw wire form: operation and command items may wrap their
    # payload in a "value" key, and a request key may nest under a "request"
    # payload (as in admission commands).
    groups: list[object] = [run.get("operations", ()), run.get("commands", ())]
    for group in groups:
        if not isinstance(group, Sequence):
            continue

        for item in group:
            if not isinstance(item, Mapping):
                continue

            value = item.get("value", item)
            if not isinstance(value, Mapping):
                continue

            key = value.get("request_key")
            request_value = value.get("request")
            if key is None and isinstance(request_value, Mapping):
                key = request_value.get("request_key")

            request_id = (
                key.get("request_id") if isinstance(key, Mapping) else None
            )
            if isinstance(request_id, int) and not isinstance(request_id, bool):
                requests.add(int(request_id))

    return frozenset(requests)


def finalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an in-memory run result into its transport mapping."""
    finalized = dict(response)
    report = finalized.get("result")
    if isinstance(report, BatchOutput):
        finalized["result"] = report.to_mapping()
    return finalized


@dataclass(slots=True)
class ServiceRequest:
    """Track one decoded IPC request.

    Covers its dependencies, successors, run, and release state.
    """

    sequence: int
    request: dict[str, Any]
    requests: frozenset[int]
    kind: RequestKind
    run: ScheduleBatch | None = None
    dependencies: int = 0
    successors: list[ServiceRequest] = field(default_factory=list)
    released: bool = False


@dataclass(slots=True)
class PendingResponse:
    """Pairs an ordered IPC response with the run that determines readiness."""

    sequence: int
    requests: frozenset[int]
    response: dict[str, Any]
    run: BatchState | None = None


def with_call_id(
    response: dict[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy the caller correlation identifier onto a response when present."""
    call_id = request.get("call_id")
    if call_id is not None:
        response["call_id"] = call_id
    return response


def error_response(
    error: WorkerError, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Encode a classified error.

    Preserve the request correlation identifier on the response.
    """
    fields = error.to_mapping()
    fields.pop("kind", None)
    return with_call_id(response(ResponseKind.ERROR, **fields), request)
