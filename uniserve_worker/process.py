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

from .bootstrap.worker_info import RequestKind, ResponseKind
from .execution.batch import Finish, Run, RunKind, RunResult
from .execution.rows import PreparedExecution
from .execution.run import ReplayWindow, RunReader, WorkerRun
from .foundation.env import env_int, env_optional_int
from .foundation.errors import (
    WorkerError,
    classify,
    invalid_descriptor,
    should_capture_trace,
)
from .profiling import WorkerProfiler, profile_range
from .worker import Worker

if TYPE_CHECKING:
    from .bootstrap.ipc import WorkerIpcEndpoint

__all__ = ["WorkerProcess", "dispatch"]

logger = logging.getLogger(__name__)


def _response(kind: ResponseKind, **payload: Any) -> dict[str, Any]:
    """Build a protocol response with the canonical kind tag and payload fields."""

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
        raise invalid_descriptor(f"worker response contains unknown fields {sorted(unknown)!r}")
    response.update(payload)
    return response


def _required(request: Mapping[str, Any], field: str, kind: RequestKind) -> Any:
    """Return a required request field or raise a classified descriptor error."""

    value = request.get(field)
    if value is None:
        raise invalid_descriptor(
            f"request {kind.value!r} is missing required field {field!r}",
            op_kind=kind.value,
        )
    return value


def _integer(request: Mapping[str, Any], field: str, kind: RequestKind) -> int:
    """Decode a required request field as a non-negative integer."""

    value = _required(request, field, kind)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(
            f"request {kind.value!r} field {field!r} must be a non-negative integer",
            op_kind=kind.value,
        )
    return value


def _request_kind(request: Mapping[str, Any]) -> RequestKind:
    """Decode and validate the request kind discriminator."""

    raw = request.get("kind")
    if not isinstance(raw, str):
        raise invalid_descriptor("worker request kind must be a string")
    try:
        return RequestKind(raw)
    except ValueError:
        raise invalid_descriptor(f"unknown worker request kind {raw!r}") from None


def _run_requests(run: Run) -> frozenset[int]:
    """Collect request identifiers referenced by a run's admissions, operations, and commands."""

    keys = (
        *(admission.request_key for admission in run.admissions),
        *(operation.request_key for operation in run.operations),
        *(command.request_key for command in run.commands),
    )
    return frozenset(int(key.request_id) for key in keys)


def _raw_request_ids(request: Mapping[str, Any]) -> frozenset[int]:
    """Extract the request identifiers touched by a submit or lifecycle command."""

    requests: set[int] = set()
    run = request.get("run")
    if isinstance(run, Run):
        return _run_requests(run) | requests
    if not isinstance(run, Mapping):
        return frozenset(requests)
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
            request = value.get("request")
            if key is None and isinstance(request, Mapping):
                key = request.get("request_key")
            request_id = key.get("request_id") if isinstance(key, Mapping) else None
            if isinstance(request_id, int) and not isinstance(request_id, bool):
                requests.add(int(request_id))
    return frozenset(requests)


def _finalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an in-memory run result into its transport mapping."""

    finalized = dict(response)
    report = finalized.get("result")
    if isinstance(report, RunResult):
        finalized["result"] = report.to_mapping()
    return finalized


def dispatch(worker: Worker, request: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one non-run request without transport I/O."""

    kind = _request_kind(request)
    if kind is RequestKind.INFO:
        return _response(ResponseKind.INFO, info=worker.info.to_mapping())
    if kind is RequestKind.CLOSE:
        return _response(ResponseKind.OK)
    if kind in {RequestKind.SUBMIT, RequestKind.POLL}:
        raise invalid_descriptor(f"{kind.value} is owned by the worker process")
    raise invalid_descriptor(f"unsupported worker request {kind.value!r}")


@dataclass(slots=True)
class _Request:
    """Tracks one decoded IPC request, its dependencies, successors, run, and release state."""

    sequence: int
    request: dict[str, Any]
    requests: frozenset[int]
    kind: RequestKind
    run: Run | None = None
    early: bool = False
    dependencies: int = 0
    successors: list[_Request] = field(default_factory=list)
    released: bool = False


@dataclass(slots=True)
class _PendingResponse:
    """Pairs an ordered IPC response with the run reader that determines readiness."""

    sequence: int
    requests: frozenset[int]
    response: dict[str, Any]
    reader: RunReader | None = None


class WorkerProcess:
    """Own the bounded transport, dependency, run, ready, and replay queues."""

    def __init__(
        self,
        worker: Worker,
        ipc_endpoint: WorkerIpcEndpoint | None,
        *,
        replay_capacity: int | None = None,
    ) -> None:
        """Bind one worker to bounded IPC admission, dependency, replay, and response queues."""

        self.worker = worker
        self.ipc_endpoint = ipc_endpoint
        self.profiler = WorkerProfiler.from_env()
        self.pipeline_depth = max(1, int(worker.info.queue_depth))
        max_operations = max(1, int(worker.info.max_batch_ops))
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
        self.runs: dict[int, WorkerRun] = {}
        self.poll_readers: dict[int, RunReader] = {}
        self.pending_responses: deque[_PendingResponse] = deque()
        self._waiting_responses: dict[int, list[_PendingResponse]] = {}
        self._waiting_response_count = 0
        self._runnable_admin: deque[_Request] = deque()
        self._runnable_early: deque[_Request] = deque()
        self._runnable: deque[_Request] = deque()
        self._tasks: dict[int, _Request] = {}
        self._request_tails: dict[int, _Request] = {}
        self._run_waiters: dict[int, list[_Request]] = {}
        self._preparation_ready: SimpleQueue[WorkerRun] = SimpleQueue()
        self._device_fifo: deque[WorkerRun] = deque()
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
        """Advance profiler state at an execution boundary and apply the configured window."""

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
        """Return the optional caller correlation identifier."""

        return request.get("call_id")

    def _with_call_id(
        self, response: dict[str, Any], request: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Copy the caller correlation identifier onto a response when present."""

        call_id = self._call_id(request)
        if call_id is not None:
            response["call_id"] = call_id
        return response

    def _error_response(
        self, error: WorkerError, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Encode a classified error and preserve the request correlation identifier."""

        fields = error.to_mapping()
        fields.pop("kind", None)
        return self._with_call_id(_response(ResponseKind.ERROR, **fields), request)

    def _record_failure(
        self, raw_kind: object, error: WorkerError, *, unexpected: bool = False
    ) -> None:
        """Log a classified request failure at the severity required by its error code."""

        log = logger.exception if unexpected or should_capture_trace(error.code) else logger.warning
        log(
            "worker request %r failed: %s [code=%s request_id=%s op_id=%s operation=%s]",
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
        """Classify an exception, record it, and encode the protocol error response."""

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
        """Assign transport order, decode the request, and link per-request dependencies."""

        # Sequence and occupancy are assigned before parsing so even malformed
        # requests produce an ordered response and release one transport slot.
        sequence = self._next_sequence
        self._next_sequence += 1
        self._transport_occupancy += 1
        requests = _raw_request_ids(request)
        try:
            kind = _request_kind(request)

            # Close commands stop admission immediately but retain their
            # sequence position until all earlier responses have drained.
            if kind is RequestKind.CLOSE:
                self._accepting_closed = True
                self._shutdown_response = self._with_call_id(dispatch(self.worker, request), request)
                return sequence
            run: Run | None = None
            early = False
            if kind is RequestKind.SUBMIT:
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
                raw_run = _required(request, "run", kind)
                raw_run_id = (
                    int(raw_run.run_id)
                    if isinstance(raw_run, Run)
                    else int(raw_run.get("run_id", -1))
                    if isinstance(raw_run, Mapping)
                    else -1
                )
                with profile_range(self._profile_name("run_decode", run_id=raw_run_id)):
                    run = raw_run if isinstance(raw_run, Run) else Run.from_mapping(raw_run)
                requests = _run_requests(run)
                early = self._launch_reorder and any(
                    operation.kind.encode_mode is not None
                    or operation.kind is RunKind.AR_EXTEND
                    for operation in run.operations
                )

            # Each request identifier forms a FIFO dependency chain. Multi-key
            # work waits once per distinct predecessor to avoid double counts.
            pending = _Request(
                sequence=sequence,
                request=request,
                requests=requests,
                kind=kind,
                run=run,
                early=early,
            )
            predecessors = {
                id(predecessor): predecessor
                for request in requests
                if (predecessor := self._request_tails.get(request)) is not None
            }
            pending.dependencies = len(predecessors)
            for predecessor in predecessors.values():
                predecessor.successors.append(pending)
            for request in requests:
                self._request_tails[request] = pending
            self._tasks[sequence] = pending
            if pending.dependencies == 0:
                self._enqueue(pending)
        except BaseException as error:
            # Parse and admission failures enter the same ordered response queue
            # as successfully launched requests.
            self.pending_responses.append(
                _PendingResponse(sequence, requests, self._boxed_error(request, error))
            )
        return sequence

    def _enqueue(self, pending: _Request) -> None:
        """Place a dependency-ready request into its priority execution queue."""

        if pending.kind is not RequestKind.SUBMIT:
            self._runnable_admin.append(pending)
        elif pending.early:
            self._runnable_early.append(pending)
        else:
            self._runnable.append(pending)

    def _release(self, pending: _Request) -> None:
        """Release one completed request and wake successors whose dependencies reach zero."""

        if pending.released:
            return
        pending.released = True
        self._tasks.pop(pending.sequence, None)
        for request in pending.requests:
            if self._request_tails.get(request) is pending:
                del self._request_tails[request]
        for successor in pending.successors:
            successor.dependencies -= 1
            if successor.dependencies == 0:
                self._enqueue(successor)
        pending.successors.clear()

    def _transport_window_open(self) -> bool:
        """Return whether the IPC pipeline can admit another request."""

        return self._transport_occupancy < self.pipeline_depth

    def _launch_one_ready_request(self) -> bool:
        """Select and launch one dependency-ready administrative or execution request."""

        if self._runnable_admin:
            pending = self._runnable_admin.popleft()
        else:
            queue = self._runnable_early if self._runnable_early else self._runnable
            if not queue:
                return False
            candidate = queue[0]
            run = candidate.run
            new_run = run is not None and int(run.run_id) not in self.runs
            if new_run and len(self._active_runs) >= self.pipeline_depth:
                return False
            pending = queue.popleft()
        try:
            if pending.kind is RequestKind.SUBMIT:
                self._launch_execute(pending)
            elif pending.kind is RequestKind.POLL:
                self._launch_poll(pending)
                self._release(pending)
            else:
                self._launch_admin(pending)
                self._release(pending)
        except BaseException as error:
            self.pending_responses.append(
                _PendingResponse(
                    pending.sequence,
                    pending.requests,
                    self._boxed_error(pending.request, error),
                )
            )
            self._release(pending)
        return True

    def _launch_admin(self, pending: _Request) -> None:
        """Dispatch an administrative request and queue its ordered response."""

        response = dispatch(self.worker, pending.request)
        self.pending_responses.append(
            _PendingResponse(
                pending.sequence,
                pending.requests,
                self._with_call_id(response, pending.request),
            )
        )

    def _launch_execute(self, pending: _Request) -> None:
        """Start one submitted run and attach ordered response completion handling."""

        submitted = pending.run
        if submitted is None:
            raise RuntimeError("accepted execute request lost its run")
        run_id = int(submitted.run_id)
        planned = self.worker.plan_run(submitted)
        run = self.runs.get(run_id)
        if run is not None:
            if run.run != planned:
                raise invalid_descriptor(f"run id {run_id} conflicts with its submitted run")
            self.replay.touch(run_id)
            waiters = self._run_waiters.get(run_id)
            if run.complete or waiters is None:
                self._release(pending)
            else:
                waiters.append(pending)
        else:
            run = WorkerRun(
                planned,
                on_successors_ready=self._run_successors_ready,
                on_ready=self._run_ready,
                on_terminal=self._run_terminal,
            )
            self.runs[run_id] = run
            self._active_runs.add(run_id)
            self._run_waiters[run_id] = [pending]
            self._start_execution(run)
        reader = self._new_reader(run)
        response = self._with_call_id(
            _response(ResponseKind.RESULT, result=reader), pending.request
        )
        self._queue_response(
            _PendingResponse(
                pending.sequence,
                pending.requests | reader.request_ids,
                response,
                reader,
            )
        )

    def _start_execution(self, run: WorkerRun) -> None:
        """Prepare transfers and predicates, then execute now or return a readiness-gated future."""

        try:
            unsupported = tuple(
                operation.kind
                for operation in run.run.operations
                if not self.worker.supports_run_kind(operation.kind)
            )
            if unsupported:
                names = sorted({value.value for value in unsupported})
                raise invalid_descriptor(
                    f"execution run contains work variants unsupported by this worker: {names!r}"
                )
            prepared = self.worker.prepare_execute(run.run) if run.run.operations else None
            if prepared is None:
                with profile_range(self._profile_name("model_execute", run_id=run.run_id)):
                    source = self.worker.execute(run.run)
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
        """Execute one administrative poll command and queue its ordered response."""

        run_id = _integer(pending.request, "run_id", RequestKind.POLL)
        reader = self.poll_readers.pop(run_id, None)
        if reader is None:
            raise invalid_descriptor(
                f"poll names run {run_id} with no pending results"
            )
        response = self._with_call_id(
            _response(ResponseKind.RESULT, result=reader), pending.request
        )
        self._queue_response(
            _PendingResponse(
                pending.sequence,
                pending.requests | reader.request_ids,
                response,
                reader,
            )
        )

    def _new_reader(self, run: WorkerRun) -> RunReader:
        """Acquire one reader lease over a run's completion stream."""

        if run.complete:
            self.replay.take(run.run_id)
        run.active_readers += 1
        return RunReader(run, self._reader_closed)

    def _reader_closed(self, reader: RunReader) -> None:
        """Release a run reader and retain terminal replay state after the final reader."""

        run = reader.run
        if run.active_readers < 1:
            raise RuntimeError("run reader ownership underflow")
        run.active_readers -= 1
        if run.complete and run.active_readers == 0:
            self._retain_terminal(run)

    def _queue_response(self, pending: _PendingResponse) -> None:
        """Queue a response immediately or hold it until its run reader is ready."""

        reader = pending.reader
        if reader is None or self._pending_ready(pending):
            self.pending_responses.append(pending)
            return
        self._waiting_responses.setdefault(reader.run_id, []).append(pending)
        self._waiting_response_count += 1

    def _run_ready(self, run: WorkerRun) -> None:
        """Move newly readable responses for a run into the transport-ready queue."""

        waiting = self._waiting_responses.pop(run.run_id, ())
        self._waiting_response_count -= len(waiting)
        for pending in waiting:
            if self._pending_ready(pending):
                self.pending_responses.append(pending)
            else:
                self._waiting_responses.setdefault(run.run_id, []).append(pending)
                self._waiting_response_count += 1

    def _run_successors_ready(self, run: WorkerRun) -> None:
        """Release requests waiting for a run's successor-visible products."""

        for pending in self._run_waiters.pop(run.run_id, ()):
            self._release(pending)

    def _run_terminal(self, run: WorkerRun) -> None:
        """Retire a terminal run, release its waiters, and retain replay state."""

        self._active_runs.discard(run.run_id)
        self._ended_epochs.update(
            (int(command.request_key.request_id), int(command.request_key.epoch))
            for command in run.run.commands
            if isinstance(command, Finish)
        )
        self._run_ready(run)
        for pending in self._run_waiters.pop(run.run_id, ()):
            self._release(pending)
        if run.active_readers == 0:
            self._retain_terminal(run)

    def _retain_terminal(self, run: WorkerRun) -> None:
        """Retain request identities whose terminal releases must wait for response delivery."""

        retains_finish = any(
            isinstance(command, Finish) for command in run.run.commands
        )
        if (
            run.epochs
            and run.epochs.issubset(self._ended_epochs)
            and not retains_finish
        ):
            if self.runs.get(run.run_id) is run:
                del self.runs[run.run_id]
            self.replay.remove(run.run_id)
            self._prune_ended_epochs()
            return
        for victim in self.replay.put(run):
            if self.runs.get(victim.run_id) is victim:
                del self.runs[victim.run_id]
        self._prune_ended_epochs()

    def _prune_ended_epochs(self) -> None:
        """Retain terminal epochs still referenced by completed resident runs."""

        referenced = {
            epoch for run in self.runs.values() if run.complete for epoch in run.epochs
        }
        self._ended_epochs.intersection_update(referenced)

    def _advance_device_fifo(self) -> bool:
        """Retire device work in submission order once its completion events become ready."""

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
        """Return whether device completion and all CPU artifacts for a response are ready."""

        reader = pending.reader
        if reader is None:
            return True
        try:
            with profile_range(self._profile_name("completion", run_id=reader.run_id)):
                return reader.ready()
        except BaseException as error:
            pending.response = self._boxed_error(pending.response, error)
            reader.close()
            pending.reader = None
            return True

    def _send_one_ready_response(self) -> bool:
        """Send the oldest transport-ready response if one exists."""

        if not self.pending_responses:
            return False
        self._send_pending(self.pending_responses.popleft())
        return True

    def _send_pending(self, pending: _PendingResponse) -> None:
        """Serialize and send one ready response while preserving transport sequence order."""

        response = dict(pending.response)
        reader = pending.reader
        run_id = reader.run_id if reader is not None else None
        if reader is not None:
            if reader.error is not None:
                response = self._error_response(reader.take_error(), pending.response)
            else:
                response["result"] = reader.take_ready()
            if reader.pending():
                existing = self.poll_readers.setdefault(reader.run_id, reader)
                if existing is not reader:
                    reader.close()
            else:
                if self.poll_readers.get(reader.run_id) is reader:
                    del self.poll_readers[reader.run_id]
                reader.close()
        fatal = bool(response.get("fatal"))
        with profile_range(self._profile_name("finalize_response", run_id=run_id)):
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
        """Send one finalized response through the bound IPC endpoint."""

        if self.ipc_endpoint is None:
            raise RuntimeError("worker process has no IPC endpoint")
        with profile_range("uniserve.worker.respond"):
            self.ipc_endpoint.respond(response)

    def _profile_name(self, boundary: str, *, run_id: int | None = None) -> str:
        """Build a rank- and run-qualified profiler range name."""

        name = f"uniserve.worker.{boundary} rank={int(self.worker.info.rank.tp_rank)}"
        return f"{name} run={run_id}" if run_id is not None and run_id >= 0 else name

    def _drain_continuations(self) -> None:
        """Close completed continuation readers retained for polling."""

        for run_id, reader in tuple(self.poll_readers.items()):
            try:
                reader.ready()
            except BaseException:
                pass
            if reader.complete:
                del self.poll_readers[run_id]
                reader.close()

    def _drained(self) -> bool:
        """Return whether all tasks, responses, runs, and readers have drained."""

        return (
            not self._tasks
            and not self.pending_responses
            and not self._waiting_responses
            and not self._active_runs
            and not self.poll_readers
        )

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Accept one request and synchronously resolve its query-ready response."""

        sequence = self._accept(dict(request))
        while self._launch_one_ready_request():
            if any(item.sequence == sequence for item in self.pending_responses):
                break
        for index, pending in enumerate(self.pending_responses):
            if pending.sequence == sequence:
                del self.pending_responses[index]
                self._transport_occupancy -= 1
                return pending.response
        for run_id, waiting in tuple(self._waiting_responses.items()):
            for index, pending in enumerate(waiting):
                if pending.sequence != sequence:
                    continue
                del waiting[index]
                self._waiting_response_count -= 1
                self._transport_occupancy -= 1
                if not waiting:
                    del self._waiting_responses[run_id]
                return pending.response
        if self._shutdown_response is not None:
            return dict(self._shutdown_response)
        raise RuntimeError("worker request did not become launchable")

    def respond(self, response: dict[str, Any]) -> None:
        """Send a response after resolving any device-backed run reader it contains."""

        reader = response.get("result")
        pending = _PendingResponse(
            0,
            frozenset() if not isinstance(reader, RunReader) else reader.request_ids,
            response,
            reader if isinstance(reader, RunReader) else None,
        )
        self._advance_device_fifo()
        if not self._pending_ready(pending):
            raise RuntimeError("worker response is not query-ready")
        self._send_pending(pending)

    def serve(self) -> None:
        """Drive request admission, device completion, and ordered IPC responses until shutdown."""

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
