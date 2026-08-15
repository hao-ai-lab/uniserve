"""Typed worker-protocol dispatch around one assembled execution worker."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from ..batch import (
    Batch,
    CacheCopy,
    CompletionReport,
    RecoveryPlacement,
    SnapshotRef,
)
from ..capabilities import RequestKind, ResponseKind
from ..foundation.env import env_int, env_optional_int
from ..foundation.errors import (
    WorkerError,
    classify,
    invalid_descriptor,
    should_capture_trace,
    unsupported_control,
)
from ..worker import Worker
from .process import WorkerIpcTransport
from .profiler import WorkerProfiler, profile_range
from .replay import CompletionDelivery, ReplayCoordinator

__all__ = ["WorkerServer", "dispatch"]

logger = logging.getLogger(__name__)


class _PendingExecution:
    def __init__(
        self,
        worker: Worker,
        prepared: object,
    ) -> None:
        self.worker = worker
        self.prepared = prepared

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
        return result

    def record_failure(self, error: BaseException) -> WorkerError:
        classified = classify(error, context="execute")
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


def _execute(
    worker: Worker,
    request: Mapping[str, Any],
    replay: ReplayCoordinator,
) -> dict[str, Any]:
    raw_batch = _required(request, "batch", RequestKind.EXECUTE)
    batch = raw_batch if isinstance(raw_batch, Batch) else Batch.from_wire(raw_batch)
    supported = frozenset(worker.capabilities.supported_work)
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
    if not batch.operations:
        result = worker.execute(batch)
        return _response(ResponseKind.RESULT, completion_report=result)
    registration = replay.register(batch)
    if not registration.execute:
        return _response(
            ResponseKind.RESULT,
            completion_report=registration.delivery,
        )
    try:
        prepare = getattr(worker, "prepare_execute", None)
        prepared = prepare(batch) if callable(prepare) else None
        if prepared is not None:
            source: object = _PendingExecution(
                worker,
                prepared,
            )
        else:
            source = worker.execute(batch)
        replay.attach(registration, source)
    except BaseException as error:
        replay.abort(registration, error)
        raise
    return _response(
        ResponseKind.RESULT,
        completion_report=registration.delivery,
    )


def _response_ready(response: dict[str, Any]) -> bool:
    """Whether a response can be serialized without stalling on a deferred token.

    A completion report's committed tokens and semantic digest are read only
    after its pinned copy events report ready. A response polled before that
    stays pending rather than blocking the transport thread.
    """

    result = response.get("completion_report")
    if isinstance(result, CompletionDelivery):
        try:
            return result.ready()
        except BaseException as error:
            _record_delivery_failure(response, result, error)
            return True
    if isinstance(result, CompletionReport):
        return True
    return True


def _response_execution_complete(response: dict[str, Any]) -> bool:
    """Advance a response source until its request-state publication point."""

    if response.get("kind") == ResponseKind.ERROR.value:
        return False
    result = response.get("completion_report")
    if not isinstance(result, CompletionDelivery):
        return True
    try:
        return result.execution_complete()
    except BaseException as error:
        _record_delivery_failure(response, result, error)
        return False


def _record_delivery_failure(
    response: dict[str, Any],
    result: CompletionDelivery,
    error: BaseException,
) -> None:
    source = result.source
    classified = (
        source.record_failure(error)
        if isinstance(source, _PendingExecution)
        else classify(error, context="completion materialization")
    )
    fields = classified.to_wire()
    fields.pop("kind", None)
    call_id = response.get("call_id")
    response.clear()
    response.update(_response(ResponseKind.ERROR, **fields))
    response["call_id"] = call_id


def _finalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
    finalized = dict(response)
    result = finalized.get("completion_report")
    if isinstance(result, CompletionReport):
        finalized["completion_report"] = result.to_wire()
    return finalized


def _control(
    worker: Worker,
    kind: RequestKind,
    request: Mapping[str, Any],
    replay: ReplayCoordinator | None,
) -> dict[str, Any] | None:
    supported = frozenset(worker.capabilities.supported_controls)
    if kind not in supported:
        raise unsupported_control(kind.value)
    if kind is RequestKind.DROP_SESSION:
        session_id = _integer(request, "session_id", kind)
        if replay is not None:
            replay.ensure_session_idle(session_id)
        worker.drop_session(session_id)
        if replay is not None:
            replay.drop_session(session_id)
    elif kind is RequestKind.COPY_KV:
        raw_copies = _required(request, "copies", kind)
        if not isinstance(raw_copies, list):
            raise invalid_descriptor("copy_kv copies must be a list")
        worker.copy_kv(
            tuple(
                CacheCopy.from_wire(value, f"copy_kv copies[{index}]")
                for index, value in enumerate(raw_copies)
            )
        )
    elif kind is RequestKind.RELEASE_PRODUCTS:
        worker.release_products(_integers(request, "product_handles", kind))
    elif kind is RequestKind.SNAPSHOT_SESSION:
        reference = worker.snapshot_session(
            RecoveryPlacement.from_wire(_required(request, "recovery_placement", kind))
        )
        return _response(ResponseKind.SNAPSHOT, snapshot=reference.to_wire())
    elif kind is RequestKind.RESTORE_SESSION:
        worker.restore_session(
            SnapshotRef.from_wire(_required(request, "snapshot", kind)),
            RecoveryPlacement.from_wire(_required(request, "recovery_placement", kind)),
        )
    else:
        raise unsupported_control(kind.value)
    return None


def dispatch(
    worker: Worker,
    request: Mapping[str, Any],
    replay: ReplayCoordinator | None = None,
) -> dict[str, Any]:
    """Dispatch one validated worker-protocol request without performing transport I/O."""

    kind = _request_kind(request)
    if kind is RequestKind.GET_CAPABILITIES:
        return _response(
            ResponseKind.CAPABILITIES,
            capabilities=worker.capabilities.to_wire(),
        )
    if kind is RequestKind.EXECUTE:
        if replay is None:
            raise invalid_descriptor("execution dispatch requires a server replay coordinator")
        return _execute(worker, request, replay)
    if kind is RequestKind.POLL_COMPLETIONS:
        raise invalid_descriptor("completion polling is owned by the worker server")
    if kind is RequestKind.GET_PRESSURE:
        return _response(ResponseKind.PRESSURE, pressure=worker.resource_pressure())
    if kind is RequestKind.SHUTDOWN:
        return _response(ResponseKind.OK)
    response = _control(worker, kind, request, replay)
    return _response(ResponseKind.OK) if response is None else response


class WorkerServer:
    """Classify, observe, and transport canonical worker requests and responses."""

    def __init__(
        self,
        worker: Worker,
        ipc_endpoint: WorkerIpcTransport | None,
        *,
        replay_capacity: int | None = None,
    ) -> None:
        self.worker = worker
        self.ipc_endpoint = ipc_endpoint
        self.profiler = WorkerProfiler.from_env()
        self._execute_count = 0
        self._terminate_after = env_int("UNISERVE_STUB_DIE_AFTER", default=0)
        self._terminate_rank = env_optional_int("UNISERVE_STUB_DIE_RANK")
        if self._terminate_rank is not None and self._terminate_rank < 0:
            raise ValueError("UNISERVE_STUB_DIE_RANK must be non-negative")
        self.pipeline_depth = max(1, int(worker.capabilities.pipeline_depth))
        max_operations = max(1, int(worker.capabilities.max_batch_operations))
        completed_capacity = (
            env_int(
                "UNISERVE_WORKER_REPLAY_CAPACITY",
                default=max(4096, max_operations),
                strict=True,
            )
            if replay_capacity is None
            else int(replay_capacity)
        )
        if completed_capacity < max_operations:
            raise ValueError(
                "completed replay capacity must hold one maximum-sized submission"
            )
        self.replay = ReplayCoordinator(
            in_flight_capacity=self.pipeline_depth * max_operations,
            completed_capacity=completed_capacity,
        )
        self._pending_completion_reports: dict[int, CompletionDelivery] = {}

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raw_kind = request.get("kind")
        terminate_this_rank = self._terminate_rank in {
            None,
            int(self.worker.capabilities.rank.tp_rank),
        }
        if (
            raw_kind == RequestKind.EXECUTE.value
            and self._terminate_after
            and terminate_this_rank
        ):
            self._execute_count += 1
            if self._execute_count > self._terminate_after:
                os._exit(1)
        try:
            if raw_kind == RequestKind.EXECUTE.value:
                with self.profiler.step("uniserve.worker.execute"):
                    response = dispatch(
                        self.worker,
                        request,
                        self.replay,
                    )
            else:
                request_name = raw_kind if isinstance(raw_kind, str) else "unknown"
                with profile_range(f"uniserve.worker.{request_name}"):
                    if raw_kind == RequestKind.POLL_COMPLETIONS.value:
                        step_id = request.get("step_id")
                        if (
                            not isinstance(step_id, int)
                            or isinstance(step_id, bool)
                            or step_id < 0
                        ):
                            raise invalid_descriptor(
                                "poll_completions requires an unsigned step id"
                            )
                        delivery = self._pending_completion_reports.get(step_id)
                        if delivery is None:
                            raise invalid_descriptor(
                                f"poll_completions names step {step_id} with no pending partitions"
                            )
                        response = _response(
                            ResponseKind.RESULT,
                            completion_report=delivery,
                        )
                    else:
                        response = dispatch(
                            self.worker,
                            request,
                            self.replay,
                        )
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
        with profile_range("uniserve.worker.respond"):
            self._respond(response)

    def _respond(self, response: dict[str, Any]) -> None:
        if self.ipc_endpoint is None:
            raise RuntimeError("worker server has no IPC endpoint")
        result = response.get("completion_report")
        if isinstance(result, CompletionDelivery):
            ready = result.take_ready()
            existing = self._pending_completion_reports.get(result.step_id)
            if result.pending():
                if existing is None:
                    self._pending_completion_reports[result.step_id] = result
                elif existing.submission_token != result.submission_token:
                    raise RuntimeError(
                        "one execution step has multiple pending completion submissions"
                    )
            elif existing is result:
                self._pending_completion_reports.pop(result.step_id, None)
            response = dict(response)
            response["completion_report"] = ready
        with profile_range("uniserve.worker.finalize_response"):
            finalized = _finalize_response(response)
        self.ipc_endpoint.respond(finalized)

    def serve(self) -> None:
        from .process import WorkerServeLoop

        if self.ipc_endpoint is None:
            raise RuntimeError("worker server has no IPC endpoint")
        WorkerServeLoop(self, self.ipc_endpoint).run()
