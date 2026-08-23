"""Blocking request loop for one canonical worker process."""

from __future__ import annotations

import gc
import os
from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from ..capabilities import RequestKind

if TYPE_CHECKING:
    from .app import WorkerServer

__all__ = ["WorkerIpcTransport", "WorkerServeLoop"]


def _request_session_ids(request: Mapping[str, Any]) -> frozenset[int]:
    sessions: set[int] = set()
    direct_session = request.get("session_id")
    if isinstance(direct_session, int) and not isinstance(direct_session, bool):
        sessions.add(direct_session)
    batch = request.get("batch")
    groups: tuple[object, ...]
    if isinstance(batch, Mapping):
        raw_partitions = batch.get("partitions", ())
        partition_operations = (
            tuple(
                operation
                for partition in raw_partitions
                if isinstance(partition, Mapping)
                for operation in partition.get("operations", ())
            )
            if isinstance(raw_partitions, Sequence)
            else ()
        )
        groups = (
            batch.get("admissions", ()),
            partition_operations,
            batch.get("controls", ()),
        )
    elif batch is not None:
        groups = (
            getattr(batch, "admissions", ()),
            getattr(batch, "operations", ()),
            getattr(batch, "controls", ()),
        )
    else:
        groups = ()
    for group in groups:
        if not isinstance(group, Sequence):
            continue
        for item in group:
            value = item.get("value", item) if isinstance(item, Mapping) else item
            request_key = (
                value.get("request_key")
                if isinstance(value, Mapping)
                else getattr(value, "request_key", None)
            )
            session_id = (
                request_key.get("session_id")
                if isinstance(request_key, Mapping)
                else getattr(request_key, "session_id", None)
            )
            if isinstance(session_id, int) and not isinstance(session_id, bool):
                sessions.add(session_id)
    return frozenset(sessions)


def _response_session_ids(response: Mapping[str, Any]) -> frozenset[int]:
    report = response.get("completion_report")
    session_ids = getattr(report, "session_ids", None)
    if isinstance(session_ids, frozenset):
        return frozenset(int(session_id) for session_id in session_ids)
    completions = getattr(report, "completions", ())
    return frozenset(int(completion.request_key.session_id) for completion in completions)


class WorkerIpcTransport(Protocol):
    def recv(self) -> dict[str, Any]: ...

    def try_recv(self) -> dict[str, Any] | None: ...

    def respond(self, response: dict[str, Any]) -> None: ...

    def wait_incoming(self, timeout_us: int) -> None: ...

    def wake(self) -> None: ...

    def wake_on_stream(self, stream: int) -> None: ...


class WorkerServeLoop:
    """Launch a bounded request pipeline and finalize responses when ready."""

    def __init__(self, worker_server: WorkerServer, ipc_endpoint: WorkerIpcTransport) -> None:
        self.worker_server = worker_server
        self.ipc_endpoint = ipc_endpoint
        self.inflight: deque[dict[str, Any]] = deque()
        self.waiting: deque[tuple[dict[str, Any], frozenset[int], bool]] = deque()
        self._inflight_sessions: dict[int, frozenset[int]] = {}
        self.shutdown: dict[str, Any] | None = None
        self._launch_reorder = int(worker_server.worker.capabilities.rank.tp_size) == 1
        profile_dir = os.environ.get("UNISERVE_SERVE_LOOP_CPROFILE_DIR")
        self._profile_state: tuple[Any, float | None, str] | None = None
        self._profile_executes = 0
        self._profile_start_execute = int(
            os.environ.get("UNISERVE_SERVE_LOOP_CPROFILE_START_EXECUTE", "1")
        )
        if profile_dir:
            import cProfile

            self._profile_state = (cProfile.Profile(), None, profile_dir)
        wake = getattr(ipc_endpoint, "wake", None)
        wake_on_stream = getattr(ipc_endpoint, "wake_on_stream", None)
        install = getattr(worker_server.worker, "set_completion_wake", None)
        if callable(wake) and callable(wake_on_stream) and callable(install):
            install(wake, wake_on_stream)

    def _append_inflight(
        self,
        request_sessions: frozenset[int],
        response: dict[str, Any],
    ) -> None:
        self.inflight.append(response)
        self._inflight_sessions[id(response)] = request_sessions | _response_session_ids(
            response
        )

    def run(self) -> None:
        self._run()

    def _profile_tick(self) -> None:
        import time

        if self._profile_state is None:
            return
        profiler, deadline, directory = self._profile_state
        if deadline is None:
            from pathlib import Path

            window = float(os.environ.get("UNISERVE_SERVE_LOOP_CPROFILE_SECONDS", "15"))
            self._profile_state = (profiler, time.monotonic() + window, directory)
            profiler.enable()
            return
        if time.monotonic() >= deadline:
            from pathlib import Path

            profiler.disable()
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(target / f"serve-loop-{os.getpid()}.cprofile"))
            self._profile_state = None

    def _run(self) -> None:
        # The loaded model and model-runner graph live for the worker's full
        # lifetime. Do not collect or freeze that initialized graph: both
        # operations traverse every model, CUDA graph, and attention-wrapper
        # object, making readiness scale with the resident graph catalog.
        # Request state is ownership-structured and reference-counted, so keep
        # cyclic scans off the serving path and restore the caller's GC mode at
        # shutdown.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            while True:
                if self._respond_ready():
                    continue
                if self._start_waiting():
                    continue
                self._refill()
                if self._start_waiting():
                    continue
                if self._respond_ready():
                    continue
                if self.shutdown is not None and not self.inflight and not self.waiting:
                    self.worker_server.respond(self.shutdown)
                    return
                if self.inflight or self.waiting:
                    self.ipc_endpoint.wait_incoming(60_000_000)
                    continue
                self._accept(self.ipc_endpoint.recv())
        finally:
            if gc_was_enabled:
                gc.enable()
            self.worker_server.profiler.close()
            close = getattr(self.worker_server.worker, "close", None)
            if callable(close):
                close()

    def _refill(self) -> None:
        while (
            self.shutdown is None
            and len(self.inflight) + len(self.waiting) < self.worker_server.pipeline_depth
        ):
            try_receive = getattr(self.ipc_endpoint, "try_recv", None)
            if not callable(try_receive):
                return
            request = try_receive()
            if request is None:
                return
            self._accept(request)

    def _accept(self, request: dict[str, Any]) -> None:
        if self._profile_state is not None:
            if request.get("kind") == RequestKind.EXECUTE.value:
                self._profile_executes += 1
            if self._profile_executes >= self._profile_start_execute:
                self._profile_tick()
        if request.get("kind") == RequestKind.SHUTDOWN.value:
            self.shutdown = self.worker_server.handle(request)
            return
        self.waiting.append(
            (request, _request_session_ids(request), self._starts_early(request))
        )

    def _starts_early(self, request: Mapping[str, Any]) -> bool:
        """Whether this request may launch ahead of older session-disjoint work.

        Prompt batches are admission-latency critical, and their device work is
        ordered by launch order, so a single-rank worker launches them before
        queued decode submissions whenever every session-lineage constraint
        still holds. Multi-rank workers execute strictly in submission order
        because collective positions are assigned at submission.
        """

        if not self._launch_reorder:
            return False
        batch = request.get("batch")
        operations = getattr(batch, "operations", None)
        if not operations:
            return False
        return any(
            work.kind == "encode" or (work.kind == "token" and work.mode == "extend")
            for operation in operations
            for work in (operation.work,)
        )

    def _start_waiting(self) -> bool:
        from .app import _response_execution_complete

        if not self.waiting:
            return False
        started = False
        items = list(self.waiting)
        order = [index for index, item in enumerate(items) if item[2]]
        order += [index for index, item in enumerate(items) if not item[2]]
        launched: set[int] = set()
        for index in order:
            request, sessions, _early = items[index]
            # A request may not overtake earlier unlaunched work that shares a
            # session, and may only start once every overlapping in-flight
            # execution has published its request state.
            ordered_before = (
                item[1]
                for position, item in enumerate(items[:index])
                if position not in launched
            )
            if (
                any(not earlier.isdisjoint(sessions) for earlier in ordered_before)
                or not all(
                    self._inflight_sessions[id(response)].isdisjoint(sessions)
                    or _response_execution_complete(response)
                    for response in self.inflight
                )
            ):
                continue
            response = self.worker_server.handle(request)
            self._append_inflight(sessions, response)
            launched.add(index)
            started = True
        self.waiting = deque(
            item for position, item in enumerate(items) if position not in launched
        )
        return started

    def _respond_ready(self) -> bool:
        from .app import _response_ready

        earlier_sessions: set[int] = set()
        for index, response in enumerate(self.inflight):
            response_id = id(response)
            request_sessions = self._inflight_sessions[response_id]
            lineage_ready = earlier_sessions.isdisjoint(request_sessions)
            if lineage_ready and _response_ready(response):
                del self.inflight[index]
                del self._inflight_sessions[response_id]
                self.worker_server.respond(response)
                return True
            earlier_sessions.update(request_sessions)
        return False
