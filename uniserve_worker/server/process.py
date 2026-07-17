"""Pipelined request serving for one already-running worker process."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any, Protocol, cast

from .execution_pipeline import PendingResult

if TYPE_CHECKING:
    from .app import WorkerServer

__all__ = [
    "WorkerIpcTransport",
    "WorkerServeLoop",
]


class WorkerIpcTransport(Protocol):
    def recv(self) -> dict[str, Any]: ...
    def try_recv(self) -> dict[str, Any] | None: ...
    def respond(self, response: dict[str, Any]) -> None: ...


class WorkerServeLoop:
    """Owns pipelined receive/dispatch/finalize scheduling."""

    def __init__(
        self,
        worker_server: WorkerServer,
        ipc_endpoint: WorkerIpcTransport,
    ) -> None:
        self.worker_server = worker_server
        self.ipc_endpoint = ipc_endpoint
        self.pending_results: deque[tuple[int | None, PendingResult]] = deque()
        self.shutdown_response: dict[str, Any] | None = None
        self.draining = False

    def run(self) -> None:
        try:
            while True:
                self._refill_nonblocking()
                if self._finalize_ready():
                    continue
                if self._finish_shutdown_if_drained():
                    break
                if self._wait_for_ready_inflight():
                    continue
                if self._receive_idle_request():
                    break
        finally:
            self.worker_server.profiler.close()

    def _refill_nonblocking(self) -> None:
        while not self.draining and len(self.pending_results) < self.worker_server.pipeline_depth:
            request = self._recv_nonblocking()
            if request is None:
                return
            if self._capture_shutdown(request):
                return
            self.pending_results.append(self._dispatch(request))

    def _finalize_ready(self) -> bool:
        if not self.pending_results:
            return False
        for index, item in enumerate(self.pending_results):
            if self.worker_server.result_ready(item[1]):
                ready = self.pending_results[index]
                del self.pending_results[index]
                self.worker_server.respond_pending(ready[1])
                return True
        return False

    def _wait_for_ready_inflight(self) -> bool:
        if not self.pending_results:
            return False
        while self.pending_results:
            self._refill_nonblocking()
            if self._finalize_ready():
                return True
            time.sleep(0.0005)
        return True

    def _finish_shutdown_if_drained(self) -> bool:
        if not self.draining or self.pending_results:
            return False
        self.worker_server.respond_pending(self.shutdown_response or {"kind": "ok"})
        return True

    def _receive_idle_request(self) -> bool:
        request = self._recv_blocking()
        if self._capture_shutdown(request):
            self.worker_server.respond_pending(self.shutdown_response or {"kind": "ok"})
            return True
        self.pending_results.append(self._dispatch(request))
        return False

    def _capture_shutdown(self, request: dict[str, Any]) -> bool:
        if request.get("kind") != "shutdown":
            return False
        self.shutdown_response = self._shutdown_response(request)
        self.draining = True
        return True

    def _recv_nonblocking(self) -> dict[str, Any] | None:
        try_recv = getattr(self.ipc_endpoint, "try_recv", None)
        if not callable(try_recv):
            return None
        started_ns = self.worker_server.metrics.now_ns()
        request = try_recv()
        self.worker_server.metrics.record_pipeline(
            "recv",
            self.worker_server.metrics.now_ns() - started_ns,
        )
        return cast(dict[str, Any] | None, request)

    def _recv_blocking(self) -> dict[str, Any]:
        started_ns = self.worker_server.metrics.now_ns()
        request = self.ipc_endpoint.recv()
        self.worker_server.metrics.record_pipeline(
            "idle",
            self.worker_server.metrics.now_ns() - started_ns,
        )
        return request

    def _dispatch(
        self,
        request: dict[str, Any],
    ) -> tuple[int | None, PendingResult]:
        call_id = request.get("call_id")
        started_ns = self.worker_server.metrics.now_ns()
        pending = self.worker_server.pending(
            request,
            allow_deferred=self.worker_server.allow_deferred_results,
        )
        self.worker_server.metrics.record_pipeline(
            "dispatch",
            self.worker_server.metrics.now_ns() - started_ns,
        )
        if call_id is not None:
            pending.response["call_id"] = call_id
        return call_id, pending

    @staticmethod
    def _shutdown_response(
        request: dict[str, Any],
    ) -> dict[str, Any]:
        response: dict[str, Any] = {"kind": "ok"}
        call_id = request.get("call_id")
        if call_id is not None:
            response["call_id"] = call_id
        return response
