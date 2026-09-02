"""Worker process queues, execution runs, replay, and IPC responses."""

from __future__ import annotations

import gc
import logging
import os
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import TYPE_CHECKING, Any

from .execution.batch import Batch, CompletionReport, ForwardMode
from .execution.run import BatchReader, BatchRun, ReplayWindow
from .execution.rows import PreparedExecution
from .foundation.env import env_int, env_optional_int
from .foundation.errors import (
    WorkerError,
    classify,
    invalid_descriptor,
    should_capture_trace,
    unsupported_control,
)
from .profiling import WorkerProfiler, profile_range
from .worker import Worker
from .bootstrap.worker_info import RequestKind, ResponseKind

if TYPE_CHECKING:
    from .bootstrap.ipc import WorkerIpcEndpoint

__all__ = ["WorkerProcess", "dispatch"]

logger = logging.getLogger(__name__)


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


def _batch_sessions(batch: Batch) -> frozenset[int]:
    keys = (
        *(admission.request_key for admission in batch.admissions),
        *(operation.request_key for operation in batch.operations),
        *(control.request_key for control in batch.controls),
    )
    return frozenset(int(key.session_id) for key in keys)


def _raw_request_sessions(request: Mapping[str, Any]) -> frozenset[int]:
    sessions: set[int] = set()
    direct = request.get("session_id")
    if isinstance(direct, int) and not isinstance(direct, bool):
        sessions.add(int(direct))
    batch = request.get("batch")
    if isinstance(batch, Batch):
        return _batch_sessions(batch) | sessions
    if not isinstance(batch, Mapping):
        return frozenset(sessions)
    groups: list[object] = [batch.get("admissions", ()), batch.get("controls", ())]
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


def _finalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
    finalized = dict(response)
    report = finalized.get("completion_report")
    if isinstance(report, CompletionReport):
        finalized["completion_report"] = report.to_mapping()
    return finalized


def _control(worker: Worker, kind: RequestKind, request: Mapping[str, Any]) -> None:
    if kind not in frozenset(worker.info.supported_controls):
        raise unsupported_control(kind.value)
    if kind is RequestKind.DROP_SESSION:
        worker.drop_session(_integer(request, "session_id", kind))
    elif kind is RequestKind.RELEASE_PRODUCTS:
        worker.release_products(_integers(request, "product_handles", kind))
    else:
        raise unsupported_control(kind.value)


def dispatch(worker: Worker, request: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one non-execution control without transport I/O."""

    kind = _request_kind(request)
    if kind is RequestKind.GET_INFO:
        return _response(ResponseKind.INFO, info=worker.info.to_mapping())
    if kind is RequestKind.GET_PRESSURE:
        return _response(ResponseKind.PRESSURE, pressure=worker.resource_pressure())
    if kind is RequestKind.SHUTDOWN:
        return _response(ResponseKind.OK)
    if kind in {RequestKind.EXECUTE, RequestKind.POLL_COMPLETIONS}:
        raise invalid_descriptor(f"{kind.value} is owned by the worker process")
    _control(worker, kind, request)
    return _response(ResponseKind.OK)


@dataclass(slots=True)
class _Request:
    sequence: int
    request: dict[str, Any]
    sessions: frozenset[int]
    kind: RequestKind
    batch: Batch | None = None
    early: bool = False
    dependencies: int = 0
    successors: list[_Request] = field(default_factory=list)
    released: bool = False


@dataclass(slots=True)
class _PendingResponse:
    sequence: int
    sessions: frozenset[int]
    response: dict[str, Any]
    reader: BatchReader | None = None


class WorkerProcess:
    """Own the bounded transport, dependency, run, ready, and replay queues."""

    def __init__(
        self,
        worker: Worker,
        ipc_endpoint: WorkerIpcEndpoint | None,
        *,
        replay_capacity: int | None = None,
    ) -> None:
        self.worker = worker
        self.ipc_endpoint = ipc_endpoint
        self.profiler = WorkerProfiler.from_env()
        self.pipeline_depth = max(1, int(worker.info.pipeline_depth))
        max_operations = max(1, int(worker.info.max_batch_operations))
        capacity = (
            env_int(
                "UNISERVE_WORKER_REPLAY_CAPACITY",
                default=max(4096, max_operations),
                strict=True,
            )
            if replay_capacity is None
            else int(replay_capacity)
        )
        if capacity < max_operations:
            raise ValueError("replay window must hold one maximum-sized submission")
        self.replay = ReplayWindow(capacity)
        self.runs: dict[int, BatchRun] = {}
        self.poll_readers: dict[int, BatchReader] = {}
        self.pending_responses: deque[_PendingResponse] = deque()
        self._waiting_responses: dict[int, list[_PendingResponse]] = {}
        self._waiting_response_count = 0
        self._runnable_controls: deque[_Request] = deque()
        self._runnable_early: deque[_Request] = deque()
        self._runnable: deque[_Request] = deque()
        self._tasks: dict[int, _Request] = {}
        self._session_tails: dict[int, _Request] = {}
        self._run_waiters: dict[int, list[_Request]] = {}
        self._preparation_ready: SimpleQueue[BatchRun] = SimpleQueue()
        self._device_fifo: deque[BatchRun] = deque()
        self._active_runs: set[int] = set()
        self._ended_epochs: set[tuple[int, int]] = set()
        self._next_sequence = 1
        self._transport_occupancy = 0
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

    @staticmethod
    def _call_id(request: Mapping[str, Any]) -> object:
        return request.get("call_id")

    def _with_call_id(
        self, response: dict[str, Any], request: Mapping[str, Any]
    ) -> dict[str, Any]:
        call_id = self._call_id(request)
        if call_id is not None:
            response["call_id"] = call_id
        return response

    def _error_response(
        self, error: WorkerError, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        fields = error.to_mapping()
        fields.pop("kind", None)
        return self._with_call_id(_response(ResponseKind.ERROR, **fields), request)

    def _record_failure(
        self, raw_kind: object, error: WorkerError, *, unexpected: bool = False
    ) -> None:
        log = logger.exception if unexpected or should_capture_trace(error.code) else logger.warning
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
        self, request: Mapping[str, Any], error: BaseException
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
        self._transport_occupancy += 1
        sessions = _raw_request_sessions(request)
        try:
            kind = _request_kind(request)
            if kind is RequestKind.SHUTDOWN:
                self._accepting_closed = True
                self._shutdown_response = self._with_call_id(dispatch(self.worker, request), request)
                return sequence
            batch: Batch | None = None
            early = False
            if kind is RequestKind.EXECUTE:
                self._profile_executes += 1
                if self._profile_state is not None and self._profile_executes >= self._profile_start_execute:
                    self._profile_tick()
                terminate_this_rank = self._terminate_rank in {
                    None,
                    int(self.worker.info.rank.tp_rank),
                }
                self._execute_count += 1
                if self._terminate_after and terminate_this_rank and self._execute_count > self._terminate_after:
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
                    batch = raw_batch if isinstance(raw_batch, Batch) else Batch.from_mapping(raw_batch)
                sessions = _batch_sessions(batch)
                early = self._launch_reorder and any(
                    operation.work.encode_mode is not None
                    or operation.work is ForwardMode.TOKEN_EXTEND
                    for operation in batch.operations
                )
            pending = _Request(
                sequence=sequence,
                request=request,
                sessions=sessions,
                kind=kind,
                batch=batch,
                early=early,
            )
            predecessors = {
                id(predecessor): predecessor
                for session in sessions
                if (predecessor := self._session_tails.get(session)) is not None
            }
            pending.dependencies = len(predecessors)
            for predecessor in predecessors.values():
                predecessor.successors.append(pending)
            for session in sessions:
                self._session_tails[session] = pending
            self._tasks[sequence] = pending
            if pending.dependencies == 0:
                self._enqueue(pending)
        except BaseException as error:
            self.pending_responses.append(
                _PendingResponse(sequence, sessions, self._boxed_error(request, error))
            )
        return sequence

    def _enqueue(self, pending: _Request) -> None:
        if pending.kind is not RequestKind.EXECUTE:
            self._runnable_controls.append(pending)
        elif pending.early:
            self._runnable_early.append(pending)
        else:
            self._runnable.append(pending)

    def _release(self, pending: _Request) -> None:
        if pending.released:
            return
        pending.released = True
        self._tasks.pop(pending.sequence, None)
        for session in pending.sessions:
            if self._session_tails.get(session) is pending:
                del self._session_tails[session]
        for successor in pending.successors:
            successor.dependencies -= 1
            if successor.dependencies == 0:
                self._enqueue(successor)
        pending.successors.clear()

    def _transport_window_open(self) -> bool:
        return self._transport_occupancy < self.pipeline_depth

    def _launch_one_ready_request(self) -> bool:
        if self._runnable_controls:
            pending = self._runnable_controls.popleft()
        else:
            queue = self._runnable_early if self._runnable_early else self._runnable
            if not queue:
                return False
            candidate = queue[0]
            batch = candidate.batch
            new_run = batch is not None and int(batch.step_id) not in self.runs
            if new_run and len(self._active_runs) >= self.pipeline_depth:
                return False
            pending = queue.popleft()
        try:
            if pending.kind is RequestKind.EXECUTE:
                self._launch_execute(pending)
            elif pending.kind is RequestKind.POLL_COMPLETIONS:
                self._launch_poll(pending)
                self._release(pending)
            else:
                self._launch_control(pending)
                self._release(pending)
        except BaseException as error:
            self.pending_responses.append(
                _PendingResponse(
                    pending.sequence,
                    pending.sessions,
                    self._boxed_error(pending.request, error),
                )
            )
            self._release(pending)
        return True

    def _launch_execute(self, pending: _Request) -> None:
        batch = pending.batch
        if batch is None:
            raise RuntimeError("accepted execute request lost its batch")
        step_id = int(batch.step_id)
        run = self.runs.get(step_id)
        if run is not None:
            if run.batch != batch:
                raise invalid_descriptor(f"execution step {step_id} conflicts with its submitted batch")
            self.replay.touch(step_id)
            waiters = self._run_waiters.get(step_id)
            if run.complete or waiters is None:
                self._release(pending)
            else:
                waiters.append(pending)
        else:
            run = BatchRun(
                batch,
                on_successors_ready=self._run_successors_ready,
                on_ready=self._run_ready,
                on_terminal=self._run_terminal,
            )
            self.runs[step_id] = run
            self._active_runs.add(step_id)
            self._run_waiters[step_id] = [pending]
            self._start_execution(run)
        reader = self._new_reader(run)
        response = self._with_call_id(
            _response(ResponseKind.RESULT, completion_report=reader), pending.request
        )
        self._queue_response(
            _PendingResponse(
                pending.sequence,
                pending.sessions | reader.session_ids,
                response,
                reader,
            )
        )

    def _start_execution(self, run: BatchRun) -> None:
        try:
            supported = frozenset(self.worker.info.supported_work)
            unsupported = tuple(
                operation.work
                for operation in run.batch.operations
                if operation.work not in supported
            )
            if unsupported:
                names = sorted({value.value for value in unsupported})
                raise invalid_descriptor(
                    f"execution batch contains work variants unsupported by this worker: {names!r}"
                )
            prepared = self.worker.prepare_execute(run.batch) if run.batch.operations else None
            if prepared is None:
                with profile_range(self._profile_name("model_execute", step_id=run.step_id)):
                    source = self.worker.execute(run.batch)
            else:
                source = prepared
            run.attach(source)
            if run.advance_execution():
                if not run.complete:
                    self._device_fifo.append(run)
            else:
                if not isinstance(source, PreparedExecution):
                    raise RuntimeError("pending execution source has no readiness owner")
                source.on_transfer_completion(lambda: self._preparation_ready.put(run))
        except BaseException as error:
            run.fail(error)

    def _launch_poll(self, pending: _Request) -> None:
        step_id = _integer(pending.request, "step_id", RequestKind.POLL_COMPLETIONS)
        reader = self.poll_readers.pop(step_id, None)
        if reader is None:
            raise invalid_descriptor(
                f"poll_completions names step {step_id} with no pending partitions"
            )
        response = self._with_call_id(
            _response(ResponseKind.RESULT, completion_report=reader), pending.request
        )
        self._queue_response(
            _PendingResponse(
                pending.sequence,
                pending.sessions | reader.session_ids,
                response,
                reader,
            )
        )

    def _launch_control(self, pending: _Request) -> None:
        if pending.kind is RequestKind.DROP_SESSION:
            session_id = _integer(pending.request, "session_id", pending.kind)
            self._ensure_session_idle(session_id)
            response = dispatch(self.worker, pending.request)
            self._drop_session_runs(session_id)
        else:
            response = dispatch(self.worker, pending.request)
        self.pending_responses.append(
            _PendingResponse(
                pending.sequence,
                pending.sessions,
                self._with_call_id(response, pending.request),
            )
        )

    def _new_reader(self, run: BatchRun) -> BatchReader:
        if run.complete:
            self.replay.take(run.step_id)
        run.active_readers += 1
        return BatchReader(run, self._reader_closed)

    def _reader_closed(self, reader: BatchReader) -> None:
        run = reader.run
        if run.active_readers < 1:
            raise RuntimeError("batch reader ownership underflow")
        run.active_readers -= 1
        if run.complete and run.active_readers == 0:
            self._retain_terminal(run)

    def _queue_response(self, pending: _PendingResponse) -> None:
        reader = pending.reader
        if reader is None or self._pending_ready(pending):
            self.pending_responses.append(pending)
            return
        self._waiting_responses.setdefault(reader.step_id, []).append(pending)
        self._waiting_response_count += 1

    def _run_ready(self, run: BatchRun) -> None:
        waiting = self._waiting_responses.pop(run.step_id, ())
        self._waiting_response_count -= len(waiting)
        for pending in waiting:
            if self._pending_ready(pending):
                self.pending_responses.append(pending)
            else:
                self._waiting_responses.setdefault(run.step_id, []).append(pending)
                self._waiting_response_count += 1

    def _run_successors_ready(self, run: BatchRun) -> None:
        for pending in self._run_waiters.pop(run.step_id, ()):
            self._release(pending)

    def _run_terminal(self, run: BatchRun) -> None:
        self._active_runs.discard(run.step_id)
        self._run_ready(run)
        for pending in self._run_waiters.pop(run.step_id, ()):
            self._release(pending)
        if run.active_readers == 0:
            self._retain_terminal(run)

    def _retain_terminal(self, run: BatchRun) -> None:
        if run.epochs and run.epochs.issubset(self._ended_epochs):
            if self.runs.get(run.step_id) is run:
                del self.runs[run.step_id]
            self.replay.remove(run.step_id)
            self._prune_ended_epochs()
            return
        for victim in self.replay.put(run):
            if self.runs.get(victim.step_id) is victim:
                del self.runs[victim.step_id]
        self._prune_ended_epochs()

    def _ensure_session_idle(self, session_id: int) -> None:
        target = int(session_id)
        if any(not run.complete and target in run.session_ids for run in self.runs.values()):
            raise invalid_descriptor(f"session {target} still has an in-flight execution step")

    def _drop_session_runs(self, session_id: int) -> None:
        target = int(session_id)
        self._ended_epochs.update(
            epoch for run in self.runs.values() for epoch in run.epochs if epoch[0] == target
        )
        for step_id, run in tuple(self.runs.items()):
            if not run.complete or run.active_readers:
                continue
            if run.epochs and run.epochs.issubset(self._ended_epochs):
                self.replay.remove(step_id)
                del self.runs[step_id]
        self._prune_ended_epochs()

    def _prune_ended_epochs(self) -> None:
        referenced = {
            epoch for run in self.runs.values() if run.complete for epoch in run.epochs
        }
        self._ended_epochs.intersection_update(referenced)

    def _advance_device_fifo(self) -> bool:
        advanced = False
        if not self._preparation_ready.empty():
            run = self._preparation_ready.get_nowait()
            before = run.state
            if not run.complete:
                run.advance_execution()
            if not run.complete:
                self._device_fifo.append(run)
            advanced = run.complete or run.state != before
        while self._device_fifo:
            run = self._device_fifo[0]
            if run.complete:
                self._device_fifo.popleft()
                advanced = True
                continue
            before = run.state
            run.advance()
            if run.complete:
                self._device_fifo.popleft()
                advanced = True
                continue
            return advanced or run.state != before
        return advanced

    def _pending_ready(self, pending: _PendingResponse) -> bool:
        reader = pending.reader
        if reader is None:
            return True
        try:
            with profile_range(self._profile_name("completion", step_id=reader.step_id)):
                return reader.ready()
        except BaseException as error:
            pending.response = self._boxed_error(pending.response, error)
            reader.close()
            pending.reader = None
            return True

    def _send_one_ready_response(self) -> bool:
        if not self.pending_responses:
            return False
        self._send_pending(self.pending_responses.popleft())
        return True

    def _send_pending(self, pending: _PendingResponse) -> None:
        response = dict(pending.response)
        reader = pending.reader
        step_id = reader.step_id if reader is not None else None
        if reader is not None:
            if reader.error is not None:
                response = self._error_response(reader.take_error(), pending.response)
            else:
                response["completion_report"] = reader.take_ready()
            if reader.pending():
                existing = self.poll_readers.setdefault(reader.step_id, reader)
                if existing is not reader:
                    reader.close()
            else:
                if self.poll_readers.get(reader.step_id) is reader:
                    del self.poll_readers[reader.step_id]
                reader.close()
        fatal = bool(response.get("fatal"))
        with profile_range(self._profile_name("finalize_response", step_id=step_id)):
            finalized = _finalize_response(response)
        self._transport_respond(finalized)
        if pending.sequence > 0:
            self._transport_occupancy -= 1
            if self._transport_occupancy < 0:
                raise RuntimeError("worker transport occupancy underflow")
        if fatal:
            self._accepting_closed = True
            self._fatal_shutdown = True

    def _transport_respond(self, response: dict[str, Any]) -> None:
        if self.ipc_endpoint is None:
            raise RuntimeError("worker process has no IPC endpoint")
        with profile_range("uniserve.worker.respond"):
            self.ipc_endpoint.respond(response)

    def _profile_name(self, boundary: str, *, step_id: int | None = None) -> str:
        name = f"uniserve.worker.{boundary} rank={int(self.worker.info.rank.tp_rank)}"
        return f"{name} step={step_id}" if step_id is not None and step_id >= 0 else name

    def _drain_continuations(self) -> None:
        for step_id, reader in tuple(self.poll_readers.items()):
            try:
                reader.ready()
            except BaseException:
                pass
            if reader.complete:
                del self.poll_readers[step_id]
                reader.close()

    def _drained(self) -> bool:
        return (
            not self._tasks
            and not self.pending_responses
            and not self._waiting_responses
            and not self._active_runs
            and not self.poll_readers
        )

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Accept and launch one request without transport I/O."""

        sequence = self._accept(dict(request))
        while self._launch_one_ready_request():
            if any(item.sequence == sequence for item in self.pending_responses):
                break
        for index, pending in enumerate(self.pending_responses):
            if pending.sequence == sequence:
                del self.pending_responses[index]
                self._transport_occupancy -= 1
                return pending.response
        for step_id, waiting in tuple(self._waiting_responses.items()):
            for index, pending in enumerate(waiting):
                if pending.sequence != sequence:
                    continue
                del waiting[index]
                self._waiting_response_count -= 1
                self._transport_occupancy -= 1
                if not waiting:
                    del self._waiting_responses[step_id]
                return pending.response
        if self._shutdown_response is not None:
            return dict(self._shutdown_response)
        raise RuntimeError("worker request did not become launchable")

    def respond(self, response: dict[str, Any]) -> None:
        reader = response.get("completion_report")
        pending = _PendingResponse(
            0,
            frozenset() if not isinstance(reader, BatchReader) else reader.session_ids,
            response,
            reader if isinstance(reader, BatchReader) else None,
        )
        self._advance_device_fifo()
        if not self._pending_ready(pending):
            raise RuntimeError("worker response is not query-ready")
        self._send_pending(pending)

    def serve(self) -> None:
        if self.ipc_endpoint is None:
            raise RuntimeError("worker process has no IPC endpoint")
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            while True:
                self.worker.device_events.reap()
                if self._advance_device_fifo():
                    continue
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
                    request = self.ipc_endpoint.try_recv()
                    if request is not None:
                        self._accept(request)
                        continue
                if self._tasks or self.pending_responses or self.poll_readers or self._active_runs:
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
