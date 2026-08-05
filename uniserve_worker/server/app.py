"""Typed worker-protocol dispatch around one assembled execution worker."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from ..batch import Batch, CompletionReport, PartitionCompletion, SnapshotRef
from ..capabilities import RequestKind, ResponseKind, operation_type
from ..execution.executor import finalize_completion_report, partition_completion_ready
from ..foundation.env import env_int
from ..foundation.errors import (
    WorkerError,
    classify,
    invalid_descriptor,
    should_capture_trace,
    unsupported_control,
)
from ..worker.protocol import Worker
from .metrics import MetricsService
from .process import WorkerIpcTransport
from .profiler import WorkerProfiler

__all__ = ["WorkerServer", "dispatch"]

logger = logging.getLogger(__name__)


class _PendingExecution:
    def __init__(
        self,
        worker: Worker,
        prepared: object,
        operation_types: list[str],
        metrics: MetricsService,
        started: int,
    ) -> None:
        self.worker = worker
        self.prepared = prepared
        self.operation_types = operation_types
        self.metrics = metrics
        self.started = started

    def ready(self) -> bool:
        query = getattr(self.prepared, "ready", None)
        if not callable(query):
            raise invalid_descriptor("prepared execution has no readiness query")
        return bool(query())

    def resolve(self) -> CompletionReport:
        if not self.ready():
            raise RuntimeError("pending execution was observed before transfer readiness")
        execute = getattr(self.worker, "execute_prepared", None)
        if not callable(execute):
            raise invalid_descriptor("worker cannot execute prepared transfer inputs")
        result = execute(self.prepared)
        if not isinstance(result, CompletionReport):
            raise RuntimeError("prepared worker execution returned an invalid report")
        self.metrics.record_execute(
            self.metrics.now_ns() - self.started,
            self.operation_types,
        )
        return result

    def record_failure(self, error: BaseException) -> WorkerError:
        classified = classify(error, context="execute")
        self.metrics.record_error(str(classified.code))
        include_trace = should_capture_trace(str(classified.code))
        log = logger.exception if include_trace else logger.warning
        log(
            "deferred worker execution failed: %s [code=%s session_id=%s op_id=%s operation=%s]",
            classified.message,
            classified.code,
            classified.req_id,
            classified.op_id,
            classified.op_kind,
        )
        return classified


def _response(kind: ResponseKind, **payload: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "kind": kind.value,
        "call_id": None,
        "capabilities": None,
        "completion_report": None,
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


def _integers(request: Mapping[str, Any], field: str, kind: RequestKind) -> tuple[int, ...]:
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
    supported = frozenset(worker.contract.capabilities.supported_work)
    unsupported = tuple(
        operation.work.variant
        for operation in batch.operations
        if operation.work.variant not in supported
    )
    if unsupported:
        names = sorted({value.value for value in unsupported})
        raise invalid_descriptor(
            f"execution batch contains work variants outside worker capabilities: {names!r}"
        )
    started = metrics.now_ns()
    operation_types = [operation_type(value).value for value in batch.operations]
    prepare = getattr(worker, "prepare_execute", None)
    prepared = prepare(batch) if callable(prepare) else None
    if prepared is not None:
        pending = _PendingExecution(worker, prepared, operation_types, metrics, started)
        if not pending.ready():
            return _response(ResponseKind.RESULT, completion_report=pending)
        result = pending.resolve()
    else:
        result = worker.execute(batch)
        metrics.record_execute(metrics.now_ns() - started, operation_types)
    # Carry the report object so the progress loop can query its completion
    # events and serialize only records whose pinned copies are ready.
    return _response(ResponseKind.RESULT, completion_report=result)


def _response_ready(response: Mapping[str, Any]) -> bool:
    """Whether a response can be serialized without stalling on a deferred token.

    A completion report's committed tokens and semantic digest are read only
    after its pinned copy events report ready. A response polled before that
    stays pending rather than blocking the transport thread.
    """

    result = response.get("completion_report")
    if isinstance(result, _PendingExecution):
        pending = result
        if not pending.ready():
            return False
        if not isinstance(response, dict):
            raise RuntimeError("pending execution response is not mutable")
        try:
            result = pending.resolve()
            response["completion_report"] = result
        except BaseException as error:
            classified = pending.record_failure(error)
            fields = classified.to_wire()
            fields.pop("kind", None)
            call_id = response.get("call_id")
            response.clear()
            response.update(_response(ResponseKind.ERROR, **fields))
            response["call_id"] = call_id
            return True
    if isinstance(result, CompletionReport):
        for partition in result.partitions:
            if partition_completion_ready(partition):
                return True
        return not result.partitions
    return True


def _finalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
    finalized = dict(response)
    result = finalized.get("completion_report")
    if isinstance(result, CompletionReport):
        finalized["completion_report"] = finalize_completion_report(result).to_wire()
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
        worker.restore_session(SnapshotRef.from_wire(_required(request, "snapshot", kind)))
    else:
        raise unsupported_control(kind.value)
    return None


def dispatch(
    worker: Worker,
    request: Mapping[str, Any],
    metrics: MetricsService | None = None,
) -> dict[str, Any]:
    """Dispatch one validated worker-protocol request without performing transport I/O."""

    service = metrics or MetricsService()
    kind = _request_kind(request)
    if kind is RequestKind.GET_CAPABILITIES:
        return _response(
            ResponseKind.CAPABILITIES,
            capabilities=worker.contract.capabilities.to_wire(),
        )
    if kind is RequestKind.EXECUTE:
        return _execute(worker, request, service)
    if kind is RequestKind.POLL_COMPLETIONS:
        raise invalid_descriptor("completion polling is owned by the worker server")
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
        self._pending_completion_reports: dict[int, CompletionReport] = {}

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raw_kind = request.get("kind")
        if raw_kind == RequestKind.EXECUTE.value and self._terminate_after:
            self._execute_count += 1
            if self._execute_count > self._terminate_after:
                os._exit(1)
        try:
            if raw_kind == RequestKind.POLL_COMPLETIONS.value:
                step_id = request.get("step_id")
                if not isinstance(step_id, int) or isinstance(step_id, bool) or step_id < 0:
                    raise invalid_descriptor("poll_completions requires an unsigned step id")
                report = self._pending_completion_reports.get(step_id)
                if report is None:
                    raise invalid_descriptor(
                        f"poll_completions names step {step_id} with no pending partitions"
                    )
                response = _response(ResponseKind.RESULT, completion_report=report)
            elif raw_kind == RequestKind.EXECUTE.value:
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
                RequestKind.POLL_COMPLETIONS,
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
            RequestKind.POLL_COMPLETIONS,
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
        result = response.get("completion_report")
        if isinstance(result, CompletionReport):
            ready_partitions: list[PartitionCompletion] = []
            pending_partitions: list[PartitionCompletion] = []
            for partition in result.partitions:
                target = (
                    ready_partitions
                    if partition_completion_ready(partition)
                    else pending_partitions
                )
                target.append(partition)
            ready = tuple(ready_partitions)
            pending = tuple(pending_partitions)
            if result.partitions and not ready:
                raise RuntimeError(
                    "completion response was selected before any partition was ready"
                )
            if pending:
                self._pending_completion_reports[result.step_id] = CompletionReport(
                    step_id=result.step_id,
                    partitions=pending,
                )
            else:
                self._pending_completion_reports.pop(result.step_id, None)
            response = dict(response)
            response["completion_report"] = CompletionReport(
                step_id=result.step_id,
                partitions=ready,
            )
        started = self.metrics.now_ns()
        self.ipc_endpoint.respond(_finalize_response(response))
        self.metrics.record_pipeline("send", self.metrics.now_ns() - started)

    def serve(self) -> None:
        from .process import WorkerServeLoop

        if self.ipc_endpoint is None:
            raise RuntimeError("worker server has no IPC endpoint")
        WorkerServeLoop(self, self.ipc_endpoint).run()
