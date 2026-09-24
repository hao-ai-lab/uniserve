"""Worker IPC request decoding helpers and response envelopes.

`uniserve_worker.service.Service` uses these to read a request's kind and
fields and to build the response mapping it hands to the endpoint. The PyO3
transport (`crates/worker-ipc-py`) decodes result and error responses strictly,
so the envelope shape built here is part of the wire contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from uniserve_worker.errors import WorkerError, invalid_descriptor
from uniserve_worker.protocol.batch import Batch
from uniserve_worker.protocol.output import BatchOutput
from uniserve_worker.protocol.worker_info import RequestKind, ResponseKind


def response(kind: ResponseKind, **payload: Any) -> dict[str, Any]:
    """Build a response envelope with the kind tag and payload fields.

    Raises:
        WorkerError: `payload` names a field the envelope does not declare.
    """
    # The envelope carries every field the wire protocol allows; payloads may
    # only override declared fields, leaving the rest at their null defaults.
    # The null defaults matter: the transport rejects a result response whose
    # info or error fields (message, code, fatal, phase, route, calls) carry
    # data, and an error response whose info or result is set.
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

    Raises:
        WorkerError: The field is absent or None.
    """
    value = request.get(field)
    if value is None:
        raise invalid_descriptor(
            f"request {kind.value!r} is missing required field {field!r}",
            call_kind=kind.value,
        )
    return value


def integer(request: Mapping[str, Any], field: str, kind: RequestKind) -> int:
    """Decode a required request field as a non-negative integer.

    Raises:
        WorkerError: The field is absent, None, a bool, not an int, or
            negative.
    """
    value = required(request, field, kind)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(
            f"request {kind.value!r} field {field!r} must be a "
            "non-negative integer",
            call_kind=kind.value,
        )
    return value


def request_kind(request: Mapping[str, Any]) -> RequestKind:
    """Decode the request kind discriminator.

    Raises:
        WorkerError: The kind is not a string or names no `RequestKind`.
    """
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
    """Collect the request ids a batch references.

    Admissions, calls, and commands each carry a request key; only its
    `request_id` is collected, not the engine id or epoch.
    """
    keys = (
        *(admission.request_key for admission in batch.admissions),
        *(call.request_key for call in batch.calls),
        *(command.request_key for command in batch.commands),
    )
    return frozenset(int(key.request_id) for key in keys)


def raw_request_ids(request: Mapping[str, Any]) -> frozenset[int]:
    """Extract the request ids a raw request's batch references.

    Accepts either a decoded `Batch` or its wire mapping under ``"batch"``.
    Malformed entries are skipped rather than rejected, and a request without
    a batch yields an empty set.
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
    """Replace a `BatchOutput` result with its wire mapping.

    The transport decodes a result response only from plain mappings and
    lists, so `Service` passes every queued response through this before
    sending it. Returns a shallow copy; the input mapping is not modified.
    """
    finalized = dict(response)
    report = finalized.get("result")
    if isinstance(report, BatchOutput):
        finalized["result"] = report.to_mapping()
    return finalized


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
    """Encode a classified error, keeping the request's message id."""
    # WorkerError.to_mapping includes its own "kind", which would collide with
    # the envelope's kind argument.
    fields = error.to_mapping()
    fields.pop("kind", None)
    return with_message_id(response(ResponseKind.ERROR, **fields), request)
