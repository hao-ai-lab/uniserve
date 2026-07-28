"""Blocking request loop for one canonical worker process."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any, Protocol

from ..capabilities import RequestKind

if TYPE_CHECKING:
    from .app import WorkerServer

__all__ = ["WorkerIpcTransport", "WorkerServeLoop"]


class WorkerIpcTransport(Protocol):
    def recv(self) -> dict[str, Any]: ...

    def try_recv(self) -> dict[str, Any] | None: ...

    def respond(self, response: dict[str, Any]) -> None: ...


class WorkerServeLoop:
    """Launch a bounded request pipeline and finalize responses when ready."""

    def __init__(self, worker_server: WorkerServer, ipc_endpoint: WorkerIpcTransport) -> None:
        self.worker_server = worker_server
        self.ipc_endpoint = ipc_endpoint
        self.inflight: deque[tuple[dict[str, Any], dict[str, Any]]] = deque()
        self.shutdown: tuple[dict[str, Any], dict[str, Any]] | None = None

    def run(self) -> None:
        try:
            while True:
                self._refill()
                if self._respond_ready():
                    continue
                if self.shutdown is not None and not self.inflight:
                    self.worker_server.respond(self.shutdown[1])
                    return
                if self.inflight:
                    time.sleep(0.00005)
                    continue
                request = self._receive()
                response = self.worker_server.handle(request)
                if request.get("kind") == RequestKind.SHUTDOWN.value:
                    self.worker_server.respond(response)
                    return
                self.inflight.append((request, response))
        finally:
            self.worker_server.profiler.close()
            close = getattr(self.worker_server.worker, "close", None)
            if callable(close):
                close()

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
            self.inflight.append((request, response))

    def _respond_ready(self) -> bool:
        from .app import _response_ready

        for index, (_request, response) in enumerate(self.inflight):
            if _response_ready(response):
                del self.inflight[index]
                self.worker_server.respond(response)
                return True
        return False

    def _receive(self) -> dict[str, Any]:
        started = self.worker_server.metrics.now_ns()
        request = self.ipc_endpoint.recv()
        self.worker_server.metrics.record_pipeline(
            "receive", self.worker_server.metrics.now_ns() - started
        )
        return request
