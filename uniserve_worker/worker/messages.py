"""IPC request dependencies and protocol response envelopes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..bootstrap.worker_info import RequestKind, ResponseKind
from ..execution.batch_state import BatchState
from ..foundation.errors import WorkerError, invalid_descriptor
from ..protocol.batch import Batch
from ..protocol.output import BatchOutput


def response(kind: ResponseKind, **payload: Any) -> dict[str, Any]:
    """Build a protocol response.

    With the canonical kind tag and payload fields.
    """
    # The envelope carries every field the wire protocol allows; payloads may
    # only override declared fields, leaving the rest at their null defaults.
    response: dict[str, Any] = {
        "kind": kind.value,
        "message_id": None,
        "info": None,
        "result": None,
        "message": None,
        "code": None,
        "fatal": None,
        "phase": None,
        "route": None,
        "calls": [],
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


def batch_requests(batch: Batch) -> frozenset[int]:
    """Collect request identifiers referenced by a batch.

    Admissions, calls, and commands each reference request keys.
    """
    keys = (
        *(admission.request_key for admission in batch.admissions),
        *(call.request_key for call in batch.calls),
        *(command.request_key for command in batch.commands),
    )
    return frozenset(int(key.request_id) for key in keys)


def raw_request_ids(request: Mapping[str, Any]) -> frozenset[int]:
    """Extract the request identifiers a command touches.

    Covers submit and lifecycle commands.
    """
    requests: set[int] = set()
    batch = request.get("batch")
    if isinstance(batch, Batch):
        return batch_requests(batch) | requests
    if not isinstance(batch, Mapping):
        return frozenset(requests)

    # Walk the raw wire form: call and command items may wrap their
    # payload in a "value" key, and a request key may nest under a "request"
    # payload (as in admission commands).
    groups: list[object] = [
        batch.get("calls", ()),
        batch.get("commands", ()),
    ]
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
    """Convert an in-memory batch result into its transport mapping."""
    finalized = dict(response)
    report = finalized.get("result")
    if isinstance(report, BatchOutput):
        finalized["result"] = report.to_mapping()
    return finalized


@dataclass(slots=True)
class ServiceRequest:
    """Track one decoded IPC request, its batch and its release state."""

    sequence: int
    request: dict[str, Any]
    requests: frozenset[int]
    kind: RequestKind
    batch: Batch | None = None
    released: bool = False


@dataclass(slots=True)
class PendingResponse:
    """Pairs an ordered IPC response with the batch that gates it."""

    sequence: int
    requests: frozenset[int]
    response: dict[str, Any]
    batch: BatchState | None = None


def with_message_id(
    response: dict[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy the caller correlation identifier onto a response when present."""
    message_id = request.get("message_id")
    if message_id is not None:
        response["message_id"] = message_id
    return response


def error_response(
    error: WorkerError, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Encode a classified error.

    Preserve the request correlation identifier on the response.
    """
    fields = error.to_mapping()
    fields.pop("kind", None)
    return with_message_id(response(ResponseKind.ERROR, **fields), request)
