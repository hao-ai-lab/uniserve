"""Event-driven worker IPC serving with exact-once step execution."""

from __future__ import annotations

import gc
import logging
import os
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..worker_info import RequestKind, ResponseKind
from ..execution.batch import (
    Batch,
    CompletionReport,
    ForwardMode,
    PartitionCompletion,
)
from ..execution.rows import PreparedExecution
from ..foundation.env import env_int, env_optional_int
from ..foundation.errors import (
    WorkerError,
    classify,
    invalid_descriptor,
    resource_error,
    should_capture_trace,
    unsupported_control,
)
from ..worker import Worker
from .completion import (
    CompletedStepCache,
    StepOutputs,
    _completion_payload_ready,
    _record_ready,
    finalize_completion_report,
    partition_completion_ready,
)
from .profiler import WorkerProfiler, profile_range

if TYPE_CHECKING:
    from .ipc import WorkerIpcEndpoint

__all__ = [
    "InflightStep",
    "PendingRequest",
    "PendingResponse",
    "TerminalStep",
    "WorkerServer",
    "dispatch",
]

logger = logging.getLogger(__name__)

_OperationKey = tuple[int, int, int]
_EpochKey = tuple[int, int]


def _response(kind: ResponseKind, **payload: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "kind": kind.value,
        "call_id": None,
        "info": None,
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


def _operation_key(operation: object) -> _OperationKey:
    request = getattr(operation, "request_key")
    return (
        int(request.session_id),
        int(request.epoch),
        int(getattr(operation, "op_id")),
    )


def _batch_lineage(batch: Batch) -> tuple[frozenset[int], frozenset[_EpochKey]]:
    request_keys = tuple(
        (
            *(admission.request_key for admission in batch.admissions),
            *(operation.request_key for operation in batch.operations),
            *(control.request_key for control in batch.controls),
        )
    )
    epochs = frozenset((int(request.session_id), int(request.epoch)) for request in request_keys)
    return frozenset(session_id for session_id, _epoch in epochs), epochs


def _raw_request_sessions(request: Mapping[str, Any]) -> frozenset[int]:
    sessions: set[int] = set()
    direct = request.get("session_id")
    if isinstance(direct, int) and not isinstance(direct, bool):
        sessions.add(int(direct))
    batch = request.get("batch")
    if isinstance(batch, Batch):
        return _batch_lineage(batch)[0] | sessions
    if not isinstance(batch, Mapping):
        return frozenset(sessions)
    groups: list[object] = [
        batch.get("admissions", ()),
        batch.get("controls", ()),
    ]
    partitions = batch.get("partitions", ())
    if isinstance(partitions, Sequence):
        groups.extend(
            partition.get("operations", ())
            for partition in partitions
            if isinstance(partition, Mapping)
        )
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
            session_id = key.get("session_id") if isinstance(key, Mapping) else None
            if isinstance(session_id, int) and not isinstance(session_id, bool):
                sessions.add(int(session_id))
    return frozenset(sessions)


def _report_partition_order(batch: Batch) -> tuple[int, ...]:
    return tuple(int(partition.partition_id) for partition in batch.partitions)


def _report_operation_keys(batch: Batch) -> tuple[tuple[_OperationKey, ...], ...]:
    return tuple(
        tuple(_operation_key(operation) for operation in partition.operations)
        for partition in batch.partitions
    )


def _validate_report_shape(
    report: CompletionReport,
    *,
    step_id: int,
    partition_order: tuple[int, ...],
    partition_operation_keys: tuple[tuple[_OperationKey, ...], ...],
) -> None:
    if int(report.step_id) != int(step_id):
        raise invalid_descriptor("terminal report step identity does not match its submission")
    actual_order = tuple(int(partition.partition_id) for partition in report.partitions)
    if actual_order != partition_order:
        raise invalid_descriptor("terminal report partition identity does not match its submission")
    for partition, expected_keys in zip(
        report.partitions,
        partition_operation_keys,
        strict=True,
    ):
        actual_keys = tuple(
            (
                int(record.request_key.session_id),
                int(record.request_key.epoch),
                int(record.op_id),
            )
            for record in partition.completions
        )
        if actual_keys != expected_keys:
            raise invalid_descriptor("terminal report operations do not align with their partition")
        expected = set(expected_keys)
        for product in partition.products:
            reference = product.product
            key = (
                int(reference.request_key.session_id),
                int(reference.request_key.epoch),
                int(reference.producer_op_id),
            )
            if key not in expected:
                raise invalid_descriptor(
                    "terminal product does not belong to its completion partition"
                )


def _materialize_partition(step_id: int, partition: PartitionCompletion) -> PartitionCompletion:
    if not partition_completion_ready(partition):
        raise RuntimeError("completion partition was materialized before query readiness")
    report = finalize_completion_report(
        CompletionReport(step_id=int(step_id), partitions=(partition,))
    )
    if len(report.partitions) != 1:
        raise RuntimeError("completion materialization changed partition cardinality")
    materialized = report.partitions[0]
    if any(
        type(token) is not int
        for record in materialized.completions
        for token in record.committed_tokens
    ):
        raise RuntimeError("materialized completion carries an unresolved committed token")
    if any(type(product.payload) is not bytes for product in materialized.products):
        raise RuntimeError("materialized completion contains a non-byte product payload")
    return materialized


@dataclass(slots=True)
class TerminalStep:
    step_id: int
    batch: Batch
    session_ids: frozenset[int]
    epochs: frozenset[_EpochKey]
    partition_order: tuple[int, ...]
    weight: int
    report: CompletionReport | None
    error: WorkerError | None
    partial_partitions: tuple[PartitionCompletion, ...]
    active_cursors: int = 0

    @property
    def complete(self) -> bool:
        return True

    @property
    def source(self) -> None:
        return None

    def current(self) -> TerminalStep:
        return self

    def advance_execution(self) -> bool:
        return True

    def advance_materialization(self) -> None:
        return None

    def materialized_partitions(self) -> tuple[PartitionCompletion, ...]:
        if self.report is not None:
            return self.report.partitions
        return self.partial_partitions


class InflightStep:
    """One exact-once execution source and its host materialization progress."""

    def __init__(
        self,
        batch: Batch,
        *,
        on_terminal: Any,
    ) -> None:
        sessions, epochs = _batch_lineage(batch)
        self.step_id = int(batch.step_id)
        self.batch = batch
        self.session_ids = sessions
        self.epochs = epochs
        self.partition_order = _report_partition_order(batch)
        self.partition_operation_keys = _report_operation_keys(batch)
        self.weight = max(1, len(batch.operations))
        self.active_cursors = 0
        self.state = "QUEUED"
        self.error: WorkerError | None = None
        self.source: CompletionReport | PreparedExecution | None = None
        self._raw_report: CompletionReport | None = None
        self._materialized: dict[int, PartitionCompletion] = {}
        self._ready_cursor: dict[int, tuple[int, int]] = {}
        self._terminal: TerminalStep | None = None
        self._on_terminal = on_terminal

    @property
    def complete(self) -> bool:
        return self._terminal is not None

    def current(self) -> InflightStep | TerminalStep:
        return self if self._terminal is None else self._terminal

    def attach(self, source: CompletionReport | PreparedExecution) -> None:
        if self.source is not None or self._terminal is not None:
            raise RuntimeError("execution step already has an execution source")
        self.source = source
        self.state = "RUNNING"

    def fail(self, error: BaseException, *, context: str = "execute") -> TerminalStep:
        if self._terminal is not None:
            return self._terminal
        source = self.source
        classified = (
            error
            if isinstance(error, WorkerError)
            else source.record_failure(error)
            if isinstance(source, PreparedExecution)
            else classify(error, context=context)
        )
        return self._terminalize(error=classified)

    def advance_execution(self) -> bool:
        if self._terminal is not None:
            return True
        if self._raw_report is not None:
            return True
        source = self.source
        if source is None:
            return False
        try:
            if isinstance(source, CompletionReport):
                report = source
            else:
                if not source.ready():
                    return False
                report = source.resolve()
            _validate_report_shape(
                report,
                step_id=self.step_id,
                partition_order=self.partition_order,
                partition_operation_keys=self.partition_operation_keys,
            )
            self._raw_report = report
            self.source = None
            self.state = "MATERIALIZING"
            return True
        except BaseException as error:
            self.fail(error, context="execute")
            return True

    def advance_materialization(self) -> None:
        if self._terminal is not None or not self.advance_execution():
            return
        report = self._raw_report
        if report is None:
            return
        try:
            for partition in report.partitions:
                partition_id = int(partition.partition_id)
                if partition_id in self._materialized:
                    continue
                if self._partition_ready(partition_id, partition):
                    self._materialized[partition_id] = _materialize_partition(
                        self.step_id,
                        partition,
                    )
            if len(self._materialized) != len(self.partition_order):
                return
            ordered = tuple(
                self._materialized[partition_id] for partition_id in self.partition_order
            )
            host_report = CompletionReport(step_id=self.step_id, partitions=ordered)
            _validate_report_shape(
                host_report,
                step_id=self.step_id,
                partition_order=self.partition_order,
                partition_operation_keys=self.partition_operation_keys,
            )
            self._terminalize(report=host_report)
        except BaseException as error:
            self.fail(error, context="completion materialization")

    def _partition_ready(
        self,
        partition_id: int,
        partition: PartitionCompletion,
    ) -> bool:
        record_cursor, product_cursor = self._ready_cursor.get(partition_id, (0, 0))
        while record_cursor < len(partition.completions) and _record_ready(
            partition.completions[record_cursor]
        ):
            record_cursor += 1
        while product_cursor < len(partition.products) and _completion_payload_ready(
            partition.products[product_cursor].payload
        ):
            product_cursor += 1
        self._ready_cursor[partition_id] = (record_cursor, product_cursor)
        return record_cursor == len(partition.completions) and product_cursor == len(
            partition.products
        )

    def materialized_partitions(self) -> tuple[PartitionCompletion, ...]:
        if self._terminal is not None:
            return self._terminal.materialized_partitions()
        return tuple(
            self._materialized[partition_id]
            for partition_id in self.partition_order
            if partition_id in self._materialized
        )

    def _terminalize(
        self,
        *,
        report: CompletionReport | None = None,
        error: WorkerError | None = None,
    ) -> TerminalStep:
        if self._terminal is not None:
            return self._terminal
        partial = tuple(
            self._materialized[partition_id]
            for partition_id in self.partition_order
            if partition_id in self._materialized
        )
        terminal = TerminalStep(
            step_id=self.step_id,
            batch=self.batch,
            session_ids=self.session_ids,
            epochs=self.epochs,
            partition_order=self.partition_order,
            weight=self.weight,
            report=report,
            error=error,
            partial_partitions=partial,
            active_cursors=self.active_cursors,
        )
        self.error = error
        self._raw_report = None
        self.source = None
        self.state = "TERMINAL"
        self._terminal = terminal
        self._on_terminal(self, terminal)
        return terminal


@dataclass(slots=True)
class PendingRequest:
    sequence: int
    request: dict[str, Any]
    sessions: frozenset[int]
    kind: RequestKind
    batch: Batch | None = None
    early_launch: bool = False


@dataclass(slots=True)
class PendingResponse:
    sequence: int
    sessions: frozenset[int]
    response: dict[str, Any]
    cursor: StepOutputs | None = None
    origin: RequestKind | None = None


def _finalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
    finalized = dict(response)
    report = finalized.get("completion_report")
    if isinstance(report, CompletionReport):
        finalized["completion_report"] = report.to_mapping()
    return finalized


def _control(
    worker: Worker, kind: RequestKind, request: Mapping[str, Any]
) -> dict[str, Any] | None:
    supported = frozenset(worker.info.supported_controls)
    if kind not in supported:
        raise unsupported_control(kind.value)
    if kind is RequestKind.DROP_SESSION:
        worker.drop_session(_integer(request, "session_id", kind))
    elif kind is RequestKind.RELEASE_PRODUCTS:
        worker.release_products(_integers(request, "product_handles", kind))
    else:
        raise unsupported_control(kind.value)
    return None


def dispatch(worker: Worker, request: Mapping[str, Any]) -> dict[str, Any]:
    """Pure request-to-response dispatch for non-execution worker controls."""

    kind = _request_kind(request)
    if kind is RequestKind.GET_INFO:
        return _response(
            ResponseKind.INFO,
            info=worker.info.to_mapping(),
        )
    if kind is RequestKind.GET_PRESSURE:
        return _response(ResponseKind.PRESSURE, pressure=worker.resource_pressure())
    if kind is RequestKind.SHUTDOWN:
        return _response(ResponseKind.OK)
    if kind in {RequestKind.EXECUTE, RequestKind.POLL_COMPLETIONS}:
        raise invalid_descriptor(f"{kind.value} is owned by the worker server")
    response = _control(worker, kind, request)
    return _response(ResponseKind.OK) if response is None else response


class WorkerServer:
    """Own transport credit, execution steps, ordering, and response transmission."""

    def __init__(
        self,
        worker: Worker,
        ipc_endpoint: WorkerIpcEndpoint | None,
        *,
        step_cache_capacity: int | None = None,
    ) -> None:
        self.worker = worker
        self.ipc_endpoint = ipc_endpoint
        self.profiler = WorkerProfiler.from_env()
        self.pipeline_depth = max(1, int(worker.info.pipeline_depth))
        max_operations = max(1, int(worker.info.max_batch_operations))
        capacity = (
            env_int(
                "UNISERVE_WORKER_STEP_CACHE_CAPACITY",
                default=max(4096, max_operations),
                strict=True,
            )
            if step_cache_capacity is None
            else int(step_cache_capacity)
        )
        if capacity < max_operations:
            raise ValueError("completed step cache capacity must hold one maximum-sized submission")
        self.completed_steps = CompletedStepCache(capacity)
        self.steps: dict[int, InflightStep | TerminalStep] = {}
        self.poll_outputs: dict[int, StepOutputs] = {}
        self.waiting_requests: deque[PendingRequest] = deque()
        self.pending_responses: deque[PendingResponse] = deque()
        self._ended_epochs: set[_EpochKey] = set()
        self._next_sequence = 1
        self._shutdown_response: dict[str, Any] | None = None
        self._accepting_closed = False
        self._fatal_shutdown = False
        self._launch_reorder = int(worker.info.rank.tp_size) == 1
        self._execute_count = 0
        self._terminate_after = env_int("UNISERVE_STUB_DIE_AFTER", default=0)
        self._terminate_rank = env_optional_int("UNISERVE_STUB_DIE_RANK")
        if self._terminate_rank is not None and self._terminate_rank < 0:
            raise ValueError("UNISERVE_STUB_DIE_RANK must be non-negative")
        profile_dir = os.environ.get("UNISERVE_SERVE_LOOP_CPROFILE_DIR")
        self._profile_state: tuple[Any, float | None, str] | None = None
        self._profile_executes = 0
        self._profile_start_execute = int(
            os.environ.get("UNISERVE_SERVE_LOOP_CPROFILE_START_EXECUTE", "1")
        )
        if profile_dir:
            import cProfile

            self._profile_state = (cProfile.Profile(), None, profile_dir)
        if ipc_endpoint is not None:
            wake = getattr(ipc_endpoint, "wake", None)
            wake_on_stream = getattr(ipc_endpoint, "wake_on_stream", None)
            if callable(wake) and callable(wake_on_stream):
                worker.set_completion_wake(wake, wake_on_stream)

    def _profile_tick(self) -> None:
        if self._profile_state is None:
            return
        profiler, deadline, directory = self._profile_state
        if deadline is None:
            window = float(os.environ.get("UNISERVE_SERVE_LOOP_CPROFILE_SECONDS", "15"))
            self._profile_state = (profiler, time.monotonic() + window, directory)
            profiler.enable()
            return
        if time.monotonic() < deadline:
            return
        from pathlib import Path

        profiler.disable()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(target / f"serve-loop-{os.getpid()}.cprofile"))
        self._profile_state = None

    def _call_id(self, request: Mapping[str, Any]) -> object:
        return request.get("call_id")

    def _with_call_id(
        self,
        response: dict[str, Any],
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        call_id = self._call_id(request)
        if call_id is not None:
            response["call_id"] = call_id
        return response

    def _error_response(
        self,
        error: WorkerError,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        fields = error.to_mapping()
        fields.pop("kind", None)
        return self._with_call_id(_response(ResponseKind.ERROR, **fields), request)

    def _record_failure(
        self,
        raw_kind: object,
        error: WorkerError,
        *,
        unexpected: bool = False,
    ) -> None:
        include_trace = unexpected or should_capture_trace(error.code)
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

    def _boxed_error(
        self,
        request: Mapping[str, Any],
        error: BaseException,
    ) -> dict[str, Any]:
        classified = (
            error
            if isinstance(error, WorkerError)
            else classify(error, context=str(request.get("kind", "unknown")))
        )
        self._record_failure(
            request.get("kind"),
            classified,
            unexpected=not isinstance(error, WorkerError),
        )
        return self._error_response(classified, request)

    def _accept(self, request: dict[str, Any]) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1
        sessions = _raw_request_sessions(request)
        try:
            kind = _request_kind(request)
            if kind is RequestKind.SHUTDOWN:
                self._accepting_closed = True
                self._shutdown_response = self._with_call_id(
                    dispatch(self.worker, request),
                    request,
                )
                return sequence
            batch: Batch | None = None
            early = False
            if kind is RequestKind.EXECUTE:
                self._profile_executes += 1
                if self._profile_state is not None and (
                    self._profile_executes >= self._profile_start_execute
                ):
                    self._profile_tick()
                terminate_this_rank = self._terminate_rank in {
                    None,
                    int(self.worker.info.rank.tp_rank),
                }
                self._execute_count += 1
                if (
                    self._terminate_after
                    and terminate_this_rank
                    and self._execute_count > self._terminate_after
                ):
                    os._exit(1)
                raw_batch = _required(request, "batch", kind)
                raw_step_id = (
                    int(raw_batch.step_id)
                    if isinstance(raw_batch, Batch)
                    else int(raw_batch.get("step_id", -1))
                    if isinstance(raw_batch, Mapping)
                    else -1
                )
                with profile_range(self._profile_name("batch_decode", step_id=raw_step_id)):
                    batch = (
                        raw_batch if isinstance(raw_batch, Batch) else Batch.from_mapping(raw_batch)
                    )
                sessions = _batch_lineage(batch)[0]
                early = self._starts_early(batch)
            self.waiting_requests.append(
                PendingRequest(
                    sequence=sequence,
                    request=request,
                    sessions=sessions,
                    kind=kind,
                    batch=batch,
                    early_launch=early,
                )
            )
        except BaseException as error:
            self.pending_responses.append(
                PendingResponse(
                    sequence=sequence,
                    sessions=sessions,
                    response=self._boxed_error(request, error),
                )
            )
        return sequence

    def _starts_early(self, batch: Batch) -> bool:
        if not self._launch_reorder:
            return False
        return any(
            operation.work.encode_mode is not None or operation.work is ForwardMode.TOKEN_EXTEND
            for operation in batch.operations
        )

    def _transport_window_open(self) -> bool:
        return len(self.waiting_requests) + len(self.pending_responses) < self.pipeline_depth

    def _live_execution_count(self) -> int:
        return sum(isinstance(step, InflightStep) for step in self.steps.values())

    def _is_execution_miss(self, pending: PendingRequest) -> bool:
        if pending.kind is not RequestKind.EXECUTE:
            return False
        batch = pending.batch
        if batch is None:
            return False
        return int(batch.step_id) not in self.steps

    def _session_gate(self, index: int, pending: PendingRequest) -> bool:
        for earlier in tuple(self.waiting_requests)[:index]:
            if not earlier.sessions.isdisjoint(pending.sessions):
                return False
        if not self._launch_reorder:
            self._advance_execution_order()
            return all(
                not isinstance(step, InflightStep)
                or step.session_ids.isdisjoint(pending.sessions)
                or step.source is None
                for step in self.steps.values()
            )
        for step in tuple(self.steps.values()):
            if not isinstance(step, InflightStep):
                continue
            if step.session_ids.isdisjoint(pending.sessions):
                continue
            if not step.advance_execution():
                return False
        return True

    def _launch_one_ready_request(self) -> bool:
        if not self.waiting_requests:
            return False
        items = tuple(self.waiting_requests)
        misses = [index for index, pending in enumerate(items) if self._is_execution_miss(pending)]
        non_sources = [index for index, pending in enumerate(items) if index not in set(misses)]
        for index in non_sources:
            pending = items[index]
            if self._session_gate(index, pending):
                self._remove_waiting(index)
                self._launch(pending)
                return True
        if self._live_execution_count() >= self.pipeline_depth or not misses:
            return False
        if self._launch_reorder:
            ordered_misses = [index for index in misses if items[index].early_launch] + [
                index for index in misses if not items[index].early_launch
            ]
        else:
            ordered_misses = misses[:1]
        for index in ordered_misses:
            pending = items[index]
            if self._session_gate(index, pending):
                self._remove_waiting(index)
                self._launch(pending)
                return True
        return False

    def _remove_waiting(self, index: int) -> None:
        items = tuple(self.waiting_requests)
        self.waiting_requests = deque(
            item for position, item in enumerate(items) if position != index
        )

    def _launch(self, pending: PendingRequest) -> None:
        try:
            if pending.kind is RequestKind.EXECUTE:
                self._launch_execute(pending)
            elif pending.kind is RequestKind.POLL_COMPLETIONS:
                self._launch_poll(pending)
            else:
                self._launch_control(pending)
        except BaseException as error:
            self.pending_responses.append(
                PendingResponse(
                    sequence=pending.sequence,
                    sessions=pending.sessions,
                    response=self._boxed_error(pending.request, error),
                )
            )

    def _launch_execute(self, pending: PendingRequest) -> None:
        batch = pending.batch
        if batch is None:
            raise RuntimeError("accepted execute request lost its batch")
        step_id = int(batch.step_id)
        existing = self.steps.get(step_id)
        if existing is not None:
            current = existing.current()
            if current is not existing:
                self.steps[step_id] = current
                existing = current
            if existing.batch != batch:
                raise invalid_descriptor(
                    f"execution step {step_id} conflicts with its submitted batch"
                )
            self.completed_steps.touch(step_id)
            cursor = self._new_cursor(existing)
        else:
            step = InflightStep(batch, on_terminal=self._step_terminal)
            self.steps[step_id] = step
            cursor = self._new_cursor(step)
            self._start_execution(step, batch)
        response = self._with_call_id(
            _response(ResponseKind.RESULT, completion_report=cursor),
            pending.request,
        )
        self.pending_responses.append(
            PendingResponse(
                sequence=pending.sequence,
                sessions=pending.sessions | cursor.session_ids,
                response=response,
                cursor=cursor,
                origin=RequestKind.EXECUTE,
            )
        )

    def _start_execution(self, step: InflightStep, batch: Batch) -> None:
        try:
            supported = frozenset(self.worker.info.supported_work)
            unsupported = tuple(
                operation.work for operation in batch.operations if operation.work not in supported
            )
            if unsupported:
                names = sorted({value.value for value in unsupported})
                raise invalid_descriptor(
                    f"execution batch contains work variants unsupported by this worker: {names!r}"
                )
            prepared = self.worker.prepare_execute(batch) if batch.operations else None
            if prepared is not None:
                source: CompletionReport | PreparedExecution = prepared
            else:
                with profile_range(self._profile_name("model_execute", step_id=int(batch.step_id))):
                    source = self.worker.execute(batch)
            step.attach(source)
        except BaseException as error:
            step.fail(error)

    def _launch_poll(self, pending: PendingRequest) -> None:
        step_id = _integer(pending.request, "step_id", RequestKind.POLL_COMPLETIONS)
        cursor = self.poll_outputs.pop(step_id, None)
        if cursor is None:
            raise invalid_descriptor(
                f"poll_completions names step {step_id} with no pending partitions"
            )
        response = self._with_call_id(
            _response(ResponseKind.RESULT, completion_report=cursor),
            pending.request,
        )
        self.pending_responses.append(
            PendingResponse(
                sequence=pending.sequence,
                sessions=pending.sessions | cursor.session_ids,
                response=response,
                cursor=cursor,
                origin=RequestKind.POLL_COMPLETIONS,
            )
        )

    def _launch_control(self, pending: PendingRequest) -> None:
        if pending.kind is RequestKind.DROP_SESSION:
            session_id = _integer(pending.request, "session_id", pending.kind)
            self._ensure_session_idle(session_id)
            response = dispatch(self.worker, pending.request)
            self._drop_session_steps(session_id)
        else:
            response = dispatch(self.worker, pending.request)
        self.pending_responses.append(
            PendingResponse(
                sequence=pending.sequence,
                sessions=pending.sessions,
                response=self._with_call_id(response, pending.request),
            )
        )

    def _new_cursor(self, step: InflightStep | TerminalStep) -> StepOutputs:
        current = step.current()
        if isinstance(current, TerminalStep):
            self.completed_steps.take(current.step_id)
        current.active_cursors += 1
        return StepOutputs(current, self._cursor_closed)

    def _cursor_closed(self, cursor: StepOutputs) -> None:
        current = cursor._current()
        if current.active_cursors < 1:
            raise RuntimeError("step cursor ownership underflow")
        current.active_cursors -= 1
        if isinstance(current, TerminalStep) and current.active_cursors == 0:
            self._retain_terminal(current)

    def _step_terminal(self, inflight: InflightStep, terminal: TerminalStep) -> None:
        if self.steps.get(inflight.step_id) is inflight:
            self.steps[inflight.step_id] = terminal
        if terminal.active_cursors == 0:
            self._retain_terminal(terminal)

    def _retain_terminal(self, terminal: TerminalStep) -> None:
        if terminal.epochs and terminal.epochs.issubset(self._ended_epochs):
            if self.steps.get(terminal.step_id) is terminal:
                del self.steps[terminal.step_id]
            self.completed_steps.remove(terminal.step_id)
            self._prune_ended_epochs()
            return
        evicted = self.completed_steps.put(terminal)
        for victim in evicted:
            if self.steps.get(victim.step_id) is victim:
                del self.steps[victim.step_id]
        self._prune_ended_epochs()

    def _ensure_session_idle(self, session_id: int) -> None:
        target = int(session_id)
        if any(
            isinstance(step, InflightStep) and target in step.session_ids
            for step in self.steps.values()
        ):
            raise resource_error(f"session {target} still has an in-flight execution step")

    def _drop_session_steps(self, session_id: int) -> None:
        target = int(session_id)
        self._ended_epochs.update(
            epoch for step in self.steps.values() for epoch in step.epochs if epoch[0] == target
        )
        for step_id, step in tuple(self.steps.items()):
            if not isinstance(step, TerminalStep) or step.active_cursors:
                continue
            if step.epochs and step.epochs.issubset(self._ended_epochs):
                self.completed_steps.remove(step_id)
                del self.steps[step_id]
        self._prune_ended_epochs()

    def _prune_ended_epochs(self) -> None:
        referenced = {
            epoch
            for step in self.steps.values()
            if isinstance(step, TerminalStep)
            for epoch in step.epochs
        }
        self._ended_epochs.intersection_update(referenced)

    def _pending_ready(self, pending: PendingResponse) -> bool:
        cursor = pending.cursor
        if cursor is None:
            return True
        try:
            with profile_range(self._profile_name("completion", step_id=int(cursor.step_id))):
                return cursor.ready()
        except BaseException as error:
            pending.response = self._boxed_error(pending.response, error)
            cursor.close()
            pending.cursor = None
            return True

    def _advance_execution_order(self) -> None:
        inflight = (step for step in self.steps.values() if isinstance(step, InflightStep))
        ordered = sorted(
            inflight,
            key=lambda step: (
                min(
                    (partition.collective_seq for partition in step.batch.partitions),
                    default=0,
                ),
                step.step_id,
            ),
        )
        for step in ordered:
            if not step.advance_execution():
                return

    def _send_one_ready_response(self) -> bool:
        if not self._launch_reorder:
            self._advance_execution_order()
        earlier_sessions: set[int] = set()
        for index, pending in enumerate(self.pending_responses):
            lineage_ready = earlier_sessions.isdisjoint(pending.sessions)
            execution_ready = (
                self._launch_reorder or pending.cursor is None or pending.cursor.source is None
            )
            if lineage_ready and execution_ready and self._pending_ready(pending):
                del self.pending_responses[index]
                self._send_pending(pending)
                return True
            earlier_sessions.update(pending.sessions)
        return False

    def _send_pending(self, pending: PendingResponse) -> None:
        response = dict(pending.response)
        cursor = pending.cursor
        step_id = int(cursor.step_id) if cursor is not None else None
        if cursor is not None:
            if cursor.error is not None:
                error = cursor.take_error()
                response = self._error_response(error, pending.response)
            else:
                response["completion_report"] = cursor.take_ready()
            if cursor.pending():
                existing = self.poll_outputs.setdefault(cursor.step_id, cursor)
                if existing is not cursor:
                    cursor.close()
            else:
                if self.poll_outputs.get(cursor.step_id) is cursor:
                    del self.poll_outputs[cursor.step_id]
                cursor.close()
        fatal = bool(response.get("fatal"))
        with profile_range(self._profile_name("finalize_response", step_id=step_id)):
            finalized = _finalize_response(response)
        self._transport_respond(finalized)
        if fatal:
            self._accepting_closed = True
            self._fatal_shutdown = True

    def _transport_respond(self, response: dict[str, Any]) -> None:
        if self.ipc_endpoint is None:
            raise RuntimeError("worker server has no IPC endpoint")
        with profile_range("uniserve.worker.respond"):
            self.ipc_endpoint.respond(response)

    def _profile_name(self, boundary: str, *, step_id: int | None = None) -> str:
        name = f"uniserve.worker.{boundary} rank={int(self.worker.info.rank.tp_rank)}"
        return f"{name} step={step_id}" if step_id is not None and step_id >= 0 else name

    def _reap_device_events(self) -> None:
        self.worker.device_events.reap()

    def _drain_continuations(self) -> None:
        for step_id, cursor in tuple(self.poll_outputs.items()):
            try:
                cursor.ready()
            except BaseException:
                pass
            if cursor.complete:
                del self.poll_outputs[step_id]
                cursor.close()

    def _drained(self) -> bool:
        return (
            not self.waiting_requests
            and not self.pending_responses
            and not any(isinstance(step, InflightStep) for step in self.steps.values())
            and not self.poll_outputs
        )

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Accept and launch one request without performing transport I/O."""

        sequence = self._accept(dict(request))
        while self._launch_one_ready_request():
            if any(item.sequence == sequence for item in self.pending_responses):
                break
        for index, pending in enumerate(self.pending_responses):
            if pending.sequence == sequence:
                del self.pending_responses[index]
                return pending.response
        if self._shutdown_response is not None:
            return dict(self._shutdown_response)
        raise RuntimeError("worker request did not become launchable")

    def respond(self, response: dict[str, Any]) -> None:
        cursor = response.get("completion_report")
        pending = PendingResponse(
            sequence=0,
            sessions=frozenset() if not isinstance(cursor, StepOutputs) else cursor.session_ids,
            response=response,
            cursor=cursor if isinstance(cursor, StepOutputs) else None,
            origin=RequestKind.EXECUTE,
        )
        if not self._launch_reorder:
            self._advance_execution_order()
            if pending.cursor is not None and pending.cursor.source is not None:
                raise RuntimeError("worker response is not query-ready")
        if not self._pending_ready(pending):
            raise RuntimeError("worker response is not query-ready")
        self._send_pending(pending)

    def serve(self) -> None:
        if self.ipc_endpoint is None:
            raise RuntimeError("worker server has no IPC endpoint")
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            while True:
                self._reap_device_events()
                if self._send_one_ready_response():
                    continue
                if self._launch_one_ready_request():
                    continue
                if self._accepting_closed:
                    self._drain_continuations()
                    if self._drained():
                        if self._shutdown_response is not None:
                            self._transport_respond(self._shutdown_response)
                        return
                if not self._accepting_closed and self._transport_window_open():
                    try_receive = getattr(self.ipc_endpoint, "try_recv", None)
                    if callable(try_receive):
                        request = try_receive()
                        if request is not None:
                            self._accept(request)
                            continue
                if (
                    self.waiting_requests
                    or self.pending_responses
                    or self.poll_outputs
                    or any(isinstance(step, InflightStep) for step in self.steps.values())
                ):
                    self.ipc_endpoint.wait_incoming(60_000_000)
                    continue
                if self._accepting_closed:
                    return
                self._accept(self.ipc_endpoint.recv())
        finally:
            if gc_was_enabled:
                gc.enable()
            self.profiler.close()
            self.worker.close()
