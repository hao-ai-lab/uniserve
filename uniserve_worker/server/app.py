"""Typed worker-protocol dispatch around one assembled execution worker."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from ..batch import Batch, ExecutionResult
from ..capabilities import RequestKind, ResponseKind
from ..foundation.env import env_int
from ..foundation.errors import (
    WorkerError,
    classify,
    invalid_descriptor,
    should_capture_trace,
    unsupported_control,
)
from ..runtime.snapshot_store import SnapshotRef
from ..worker.protocol import Worker
from .metrics import MetricsService
from .process import WorkerIpcTransport
from .profiler import WorkerProfiler

__all__ = ["WorkerServer", "dispatch"]

logger = logging.getLogger(__name__)


def _response(kind: ResponseKind, **payload: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "kind": kind.value,
        "call_id": None,
        "capabilities": None,
        "result": None,
        "metrics": None,
        "pressure": None,
        "message": None,
        "code": None,
        "retryable": None,
        "fatal": None,
        "phase": None,
        "route": None,
        "operations": [],
        "snapshot": None,
    }
    unknown = set(payload) - set(response)
    if unknown:
        raise invalid_descriptor(f"worker response contains unknown fields {sorted(unknown)!r}")
    response.update(payload)
    return response


def _required(request: Mapping[str, Any], field: str, kind: RequestKind) -> Any:
    value = request.get(field)
    if value is None:
        raise invalid_descriptor(
            f"request {kind.value!r} is missing required field {field!r}",
            op_kind=kind.value,
        )
    return value


def _integer(request: Mapping[str, Any], field: str, kind: RequestKind) -> int:
    value = _required(request, field, kind)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(
            f"request {kind.value!r} field {field!r} must be a non-negative integer",
            op_kind=kind.value,
        )
    return value


def _string(request: Mapping[str, Any], field: str, kind: RequestKind) -> str:
    value = _required(request, field, kind)
    if not isinstance(value, str) or not value:
        raise invalid_descriptor(
            f"request {kind.value!r} field {field!r} must be a non-empty string",
            op_kind=kind.value,
        )
    return value


def _integer_pairs(
    request: Mapping[str, Any], field: str, kind: RequestKind
) -> tuple[tuple[int, int], ...]:
    value = _required(request, field, kind)
    if not isinstance(value, (list, tuple)):
        raise invalid_descriptor(
            f"request {kind.value!r} field {field!r} must be a list",
            op_kind=kind.value,
        )
    pairs: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or any(
                not isinstance(member, int) or isinstance(member, bool) or member < 0
                for member in item
            )
        ):
            raise invalid_descriptor(
                f"request {kind.value!r} field {field}[{index}] must contain two non-negative integers",
                op_kind=kind.value,
            )
        pairs.append((int(item[0]), int(item[1])))
    return tuple(pairs)


def _integers(
    request: Mapping[str, Any], field: str, kind: RequestKind
) -> tuple[int, ...]:
    value = _required(request, field, kind)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value
    ):
        raise invalid_descriptor(
            f"request {kind.value!r} field {field!r} must be a list of non-negative integers",
            op_kind=kind.value,
        )
    return tuple(int(item) for item in value)


def _request_kind(request: Mapping[str, Any]) -> RequestKind:
    raw = request.get("kind")
    if not isinstance(raw, str):
        raise invalid_descriptor("worker request kind must be a string")
    try:
        return RequestKind(raw)
    except ValueError:
        raise invalid_descriptor(f"unknown worker request kind {raw!r}") from None


def _execute(worker: Worker, request: Mapping[str, Any], metrics: MetricsService) -> dict[str, Any]:
    raw_batch = _required(request, "batch", RequestKind.EXECUTE)
    batch = raw_batch if isinstance(raw_batch, Batch) else Batch.from_wire(raw_batch)
    supported = frozenset(worker.contract.capabilities.supported_operation_types)
    unsupported = tuple(
        operation.operation_type
        for operation in batch.operations
        if operation.operation_type not in supported
    )
    if unsupported:
        names = sorted({value.value for value in unsupported})
        raise invalid_descriptor(
            f"execution batch contains operation types outside worker capabilities: {names!r}"
        )
    started = metrics.now_ns()
    result = worker.execute(batch)
    duration = metrics.now_ns() - started
    result.validate_for(batch)
    operation_types = [value.operation_type.value for value in batch.operations]
    metrics.record_execute(duration, operation_types)
    metrics.record_forward_stats(result.forward_stats)
    return _response(
        ResponseKind.RESULT,
        result=result if _result_has_deferred_tokens(result) else result.to_wire(),
    )


def _result_deferred_tokens(result: ExecutionResult) -> tuple[object, ...]:
    tokens: list[object] = []
    for operation in result.operations:
        delta = operation.delta
        effect = getattr(delta, "effect", None)
        if effect is None:
            effect = getattr(delta, "sequence", None)
        if effect is not None:
            tokens.extend(
                token
                for token in effect.sampled_token_ids
                if callable(getattr(token, "finalize", None))
            )
    return tuple(tokens)


def _result_has_deferred_tokens(result: ExecutionResult) -> bool:
    return bool(_result_deferred_tokens(result))


def _response_ready(response: Mapping[str, Any]) -> bool:
    result = response.get("result")
    if not isinstance(result, ExecutionResult):
        return True
    return all(
        bool(ready())
        for token in _result_deferred_tokens(result)
        if callable(ready := getattr(token, "ready", None))
    )


def _finalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
    finalized = dict(response)
    result = finalized.get("result")
    if isinstance(result, ExecutionResult):
        finalized["result"] = result.to_wire()
    return finalized


def _control(
    worker: Worker, kind: RequestKind, request: Mapping[str, Any]
) -> dict[str, Any] | None:
    supported = frozenset(worker.contract.capabilities.supported_controls)
    if kind not in supported:
        raise unsupported_control(kind.value)
    if kind is RequestKind.DROP_SESSION:
        worker.drop_session(_integer(request, "session_id", kind))
    elif kind is RequestKind.COPY_KV:
        worker.copy_kv(_integer_pairs(request, "copies", kind))
    elif kind is RequestKind.LOAD_ADAPTER:
        worker.load_adapter(
            _integer(request, "adapter_id", kind),
            _string(request, "adapter_path", kind),
        )
    elif kind is RequestKind.UNLOAD_ADAPTER:
        worker.unload_adapter(_integer(request, "adapter_id", kind))
    elif kind is RequestKind.RELEASE_PRODUCTS:
        worker.release_products(_integers(request, "product_handles", kind))
    elif kind is RequestKind.RESET_PREFIX_CACHE:
        worker.reset_prefix_cache()
    elif kind is RequestKind.SNAPSHOT_SESSION:
        reference = worker.snapshot_session(_integer(request, "session_id", kind))
        return _response(ResponseKind.SNAPSHOT, snapshot=reference.to_wire())
    elif kind is RequestKind.RESTORE_SESSION:
        worker.restore_session(
            SnapshotRef.from_wire(_required(request, "snapshot", kind))
        )
    else:
        raise unsupported_control(kind.value)
    return None


def dispatch(
    worker: Worker,
    request: Mapping[str, Any],
    metrics: MetricsService | None = None,
) -> dict[str, Any]:
    """Dispatch one validated protocol-v3 request without performing transport I/O."""

    service = metrics or MetricsService()
    kind = _request_kind(request)
    if kind is RequestKind.GET_CAPABILITIES:
        return _response(
            ResponseKind.CAPABILITIES,
            capabilities=worker.contract.capabilities.to_wire(),
        )
    if kind is RequestKind.EXECUTE:
        return _execute(worker, request, service)
    if kind is RequestKind.GET_METRICS:
        return _response(ResponseKind.METRICS, metrics=service.snapshot())
    if kind is RequestKind.GET_PRESSURE:
        return _response(ResponseKind.PRESSURE, pressure=worker.resource_pressure())
    if kind is RequestKind.SHUTDOWN:
        return _response(ResponseKind.OK)
    response = _control(worker, kind, request)
    return _response(ResponseKind.OK) if response is None else response


class WorkerServer:
    """Classify, observe, and transport canonical worker requests and responses."""

    def __init__(
        self,
        worker: Worker,
        ipc_endpoint: WorkerIpcTransport | None,
        metrics: MetricsService | None = None,
    ) -> None:
        self.worker = worker
        self.ipc_endpoint = ipc_endpoint
        self.metrics = metrics or MetricsService()
        self.profiler = WorkerProfiler.from_env()
        self._execute_count = 0
        self._terminate_after = env_int("UNISERVE_STUB_DIE_AFTER", default=0)
        self.pipeline_depth = max(1, int(worker.contract.capabilities.pipeline_depth))

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raw_kind = request.get("kind")
        if raw_kind == RequestKind.EXECUTE.value and self._terminate_after:
            self._execute_count += 1
            if self._execute_count > self._terminate_after:
                os._exit(1)
        try:
            if raw_kind == RequestKind.EXECUTE.value:
                with self.profiler.step("uniserve.worker.execute"):
                    response = dispatch(self.worker, request, self.metrics)
            else:
                response = dispatch(self.worker, request, self.metrics)
            try:
                kind = RequestKind(str(raw_kind))
            except ValueError:
                kind = None
            if kind not in {
                None,
                RequestKind.GET_CAPABILITIES,
                RequestKind.EXECUTE,
                RequestKind.GET_METRICS,
                RequestKind.GET_PRESSURE,
                RequestKind.SHUTDOWN,
            }:
                self.metrics.record_control(kind.value, True)
        except WorkerError as error:
            self._record_failure(raw_kind, error)
            fields = error.to_wire()
            fields.pop("kind", None)
            response = _response(ResponseKind.ERROR, **fields)
        except Exception as error:  # noqa: BLE001 - every process-boundary failure is typed.
            classified = classify(error, context=str(raw_kind) if raw_kind is not None else None)
            self._record_failure(raw_kind, classified, unexpected=True)
            fields = classified.to_wire()
            fields.pop("kind", None)
            response = _response(ResponseKind.ERROR, **fields)
        call_id = request.get("call_id")
        if call_id is not None:
            response["call_id"] = call_id
        return response

    def _record_failure(
        self,
        raw_kind: object,
        error: WorkerError,
        *,
        unexpected: bool = False,
    ) -> None:
        try:
            kind = RequestKind(str(raw_kind))
        except ValueError:
            kind = None
        if kind not in {
            None,
            RequestKind.GET_CAPABILITIES,
            RequestKind.EXECUTE,
            RequestKind.GET_METRICS,
            RequestKind.GET_PRESSURE,
            RequestKind.SHUTDOWN,
        }:
            self.metrics.record_control(kind.value, False)
        self.metrics.record_error(str(error.code))
        include_trace = unexpected or should_capture_trace(str(error.code))
        log = logger.exception if include_trace else logger.warning
        log(
            "worker request %r failed: %s [code=%s session_id=%s op_id=%s operation=%s]",
            raw_kind,
            error.message,
            error.code,
            error.req_id,
            error.op_id,
            error.op_kind,
        )

    def respond(self, response: dict[str, Any]) -> None:
        if self.ipc_endpoint is None:
            raise RuntimeError("worker server has no IPC endpoint")
        started = self.metrics.now_ns()
        self.ipc_endpoint.respond(_finalize_response(response))
        self.metrics.record_pipeline("send", self.metrics.now_ns() - started)

    def serve(self) -> None:
        from .process import WorkerServeLoop

        if self.ipc_endpoint is None:
            raise RuntimeError("worker server has no IPC endpoint")
        WorkerServeLoop(self, self.ipc_endpoint).run()
