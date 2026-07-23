"""Blocking request loop for one canonical worker process."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ..capabilities import RequestKind

if TYPE_CHECKING:
    from .app import WorkerServer

__all__ = ["WorkerIpcTransport", "WorkerServeLoop"]


class WorkerIpcTransport(Protocol):
    def recv(self) -> dict[str, Any]: ...

    def respond(self, response: dict[str, Any]) -> None: ...


class WorkerServeLoop:
    """Receive, execute, and respond until a typed shutdown request commits."""

    def __init__(self, worker_server: WorkerServer, ipc_endpoint: WorkerIpcTransport) -> None:
        self.worker_server = worker_server
        self.ipc_endpoint = ipc_endpoint

    def run(self) -> None:
        try:
            while True:
                started = self.worker_server.metrics.now_ns()
                request = self.ipc_endpoint.recv()
                self.worker_server.metrics.record_pipeline(
                    "receive", self.worker_server.metrics.now_ns() - started
                )
                response = self.worker_server.handle(request)
                self.worker_server.respond(response)
                if request.get("kind") == RequestKind.SHUTDOWN.value:
                    return
        finally:
            self.worker_server.profiler.close()
            close = getattr(self.worker_server.worker, "close", None)
            if callable(close):
                close()
