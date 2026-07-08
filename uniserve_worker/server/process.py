"""Worker process and transport scheduling."""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Protocol, cast

from .execution_pipeline import PendingResult

__all__ = [
    "WorkerProcess",
    "WorkerTransport",
]


class WorkerTransport(Protocol):
    def recv(self) -> dict[str, Any]: ...
    def try_recv(self) -> dict[str, Any] | None: ...
    def respond(self, resp: dict[str, Any]) -> None: ...


class WorkerProcess:
    """Owns pipelined receive/dispatch/finalize scheduling."""

    def __init__(self, runtime: Any, transport: WorkerTransport) -> None:
        self.runtime = runtime
        self.transport = transport
        self.inflight: deque[tuple[int | None, PendingResult]] = deque()
        self.shutdown_resp: dict[str, Any] | None = None
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
            self.runtime.profiler.close()

    def _refill_nonblocking(self) -> None:
        while not self.draining and len(self.inflight) < self.runtime.pipeline_depth:
            req = self._recv_nonblocking()
            if req is None:
                return
            if self._capture_shutdown(req):
                return
            self.inflight.append(self._dispatch(req))

    def _finalize_ready(self) -> bool:
        if not self.inflight:
            return False
        for idx, item in enumerate(self.inflight):
            if self.runtime.result_ready(item[1]):
                ready = self.inflight[idx]
                del self.inflight[idx]
                self.runtime.respond_pending(ready[1])
                return True
        return False

    def _wait_for_ready_inflight(self) -> bool:
        if not self.inflight:
            return False
        while self.inflight:
            self._refill_nonblocking()
            if self._finalize_ready():
                return True
            time.sleep(0.0005)
        return True

    def _finish_shutdown_if_drained(self) -> bool:
        if not self.draining or self.inflight:
            return False
        self.runtime.respond_pending(self.shutdown_resp or {"kind": "ok"})
        return True

    def _receive_idle_request(self) -> bool:
        req = self._recv_blocking()
        if self._capture_shutdown(req):
            self.runtime.respond_pending(self.shutdown_resp or {"kind": "ok"})
            return True
        self.inflight.append(self._dispatch(req))
        return False

    def _capture_shutdown(self, req: dict[str, Any]) -> bool:
        if req.get("kind") != "shutdown":
            return False
        self.shutdown_resp = self._shutdown_response(req)
        self.draining = True
        return True

    def _recv_nonblocking(self) -> dict[str, Any] | None:
        try_recv = getattr(self.transport, "try_recv", None)
        if not callable(try_recv):
            return None
        t0 = self.runtime.metrics.now_ns()
        req = try_recv()
        self.runtime.metrics.record_pipeline("recv", self.runtime.metrics.now_ns() - t0)
        return cast(dict[str, Any] | None, req)

    def _recv_blocking(self) -> dict[str, Any]:
        t0 = self.runtime.metrics.now_ns()
        req = self.transport.recv()
        self.runtime.metrics.record_pipeline("idle", self.runtime.metrics.now_ns() - t0)
        return req

    def _dispatch(self, req: dict[str, Any]) -> tuple[int | None, PendingResult]:
        call_id = req.get("call_id")
        t0 = self.runtime.metrics.now_ns()
        pending = self.runtime.pending(req, allow_deferred=self.runtime._defer_text_results)
        self.runtime.metrics.record_pipeline("dispatch", self.runtime.metrics.now_ns() - t0)
        if call_id is not None:
            pending.response["call_id"] = call_id
        return call_id, pending

    @staticmethod
    def _shutdown_response(req: dict[str, Any]) -> dict[str, Any]:
        resp: dict[str, Any] = {"kind": "ok"}
        call_id = req.get("call_id")
        if call_id is not None:
            resp["call_id"] = call_id
        return resp
