"""Worker IPC request decoding helpers and response envelopes.

`uniserve_worker.service.Service` uses these to read a request's kind and
fields and to build the response mapping it hands to the endpoint. The PyO3
transport (`crates/worker-ipc-py`) decodes result and error responses strictly,
so the envelope shape built here is part of the wire contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from uniserve_worker.errors import WorkerError, invalid_descriptor
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
