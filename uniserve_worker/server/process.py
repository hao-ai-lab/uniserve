"""Blocking request loop for one canonical worker process."""

from __future__ import annotations

import gc
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
    completions = getattr(report, "completions", ())
    return frozenset(
        int(completion.request_key.session_id)
        for completion in completions
    )


class WorkerIpcTransport(Protocol):
    def recv(self) -> dict[str, Any]: ...

    def try_recv(self) -> dict[str, Any] | None: ...

    def respond(self, response: dict[str, Any]) -> None: ...

    def wait_incoming(self, timeout_us: int) -> None: ...


class WorkerServeLoop:
    """Launch a bounded request pipeline and finalize responses when ready."""

    def __init__(self, worker_server: WorkerServer, ipc_endpoint: WorkerIpcTransport) -> None:
        self.worker_server = worker_server
        self.ipc_endpoint = ipc_endpoint
        self.inflight: deque[tuple[dict[str, Any], dict[str, Any]]] = deque()
        self._inflight_sessions: dict[int, frozenset[int]] = {}
        self.shutdown: tuple[dict[str, Any], dict[str, Any]] | None = None

    def _append_inflight(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        self.inflight.append((request, response))
        self._inflight_sessions[id(response)] = _request_session_ids(
            request
        ) | _response_session_ids(response)

    def run(self) -> None:
        # The loaded model and executor graph live for the worker's full
        # lifetime. Excluding that initialized graph from later cyclic scans
        # keeps request-time full collections proportional to request state;
        # objects allocated while serving remain in the ordinary generations.
        gc.collect()
        gc.freeze()
        try:
            while True:
                self._refill()
                if self._respond_ready():
                    continue
                if self.shutdown is not None and not self.inflight:
                    self.worker_server.respond(self.shutdown[1])
                    return
                if self.inflight:
                    # Device work is in flight but none is query-ready. Park on
                    # the IPC command wake so a new submission advances the loop
                    # immediately; the bounded timeout is the query-only
                    # completion-readiness re-check floor, since device readiness
                    # carries no operating-system wake.
                    self._wait_incoming()
                    continue
                request = self._receive()
                response = self.worker_server.handle(request)
                if request.get("kind") == RequestKind.SHUTDOWN.value:
                    self.worker_server.respond(response)
                    return
                self._append_inflight(request, response)
        finally:
            gc.unfreeze()
            self.worker_server.profiler.close()
            close = getattr(self.worker_server.worker, "close", None)
            if callable(close):
                close()

    # 50-microsecond floor between query-only completion-readiness re-checks
    # while device work is in flight; an inbound command wake returns earlier.
    _INFLIGHT_WAIT_US = 50

    def _wait_incoming(self) -> None:
        self.ipc_endpoint.wait_incoming(self._INFLIGHT_WAIT_US)

    def _refill(self) -> None:
        while self.shutdown is None and len(self.inflight) < self.worker_server.pipeline_depth:
            started = self.worker_server.metrics.now_ns()
            try_receive = getattr(self.ipc_endpoint, "try_recv", None)
            if not callable(try_receive):
                return
            request = try_receive()
            self.worker_server.metrics.record_pipeline(
                "receive", self.worker_server.metrics.now_ns() - started
            )
            if request is None:
                return
            response = self.worker_server.handle(request)
            if request.get("kind") == RequestKind.SHUTDOWN.value:
                self.shutdown = (request, response)
                return
            self._append_inflight(request, response)

    def _respond_ready(self) -> bool:
        from .app import _response_ready

        earlier_sessions: set[int] = set()
        for index, (request, response) in enumerate(self.inflight):
            response_id = id(response)
            request_sessions = self._inflight_sessions.get(response_id)
            if request_sessions is None:
                # Preserve direct test/debug injection into ``inflight`` while
                # keeping the normal polling path allocation-free.
                request_sessions = _request_session_ids(request) | _response_session_ids(
                    response
                )
                self._inflight_sessions[response_id] = request_sessions
            lineage_ready = earlier_sessions.isdisjoint(request_sessions)
            if lineage_ready and _response_ready(response):
                del self.inflight[index]
                del self._inflight_sessions[response_id]
                self.worker_server.respond(response)
                return True
            earlier_sessions.update(request_sessions)
        return False

    def _receive(self) -> dict[str, Any]:
        started = self.worker_server.metrics.now_ns()
        request = self.ipc_endpoint.recv()
        self.worker_server.metrics.record_pipeline(
            "receive", self.worker_server.metrics.now_ns() - started
        )
        return request
