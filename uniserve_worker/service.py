"""Synchronous IPC admission and response delivery for one Worker."""

from __future__ import annotations

import gc
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from uniserve.profiling import profile_range
from uniserve_worker.errors import WorkerError, classify
from uniserve_worker.execution.executor import Submission
from uniserve_worker.profiling import record_failure, worker_range_name
from uniserve_worker.protocol import messages
from uniserve_worker.protocol.batch import Batch
from uniserve_worker.protocol.worker_info import RequestKind, ResponseKind

if TYPE_CHECKING:
    from uniserve_worker.bootstrap.launch import WorkerIpcEndpoint
    from uniserve_worker.worker import Worker


@dataclass(slots=True)
class PendingResponse:
    """An IPC correlation envelope and its optional execution handle."""

    response: dict[str, Any]
    submission: Submission | None = None


class Service:
    """Borrow an endpoint and drive the same Executor as direct callers."""

    def __init__(self, worker: Worker, endpoint: WorkerIpcEndpoint) -> None:
        self.worker = worker
        self.endpoint = endpoint
        self._pending: deque[PendingResponse] = deque()
        self._closing = False
        self._shutdown_response: dict[str, Any] | None = None

    def run(self) -> None:
        """Serve until Close drains accepted work; leave resources to Worker."""
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            while True:
                advanced = self.worker.executor.advance()
                if self._send_ready() or advanced:
                    continue
                if self._closing and not self._pending:
                    if not self.worker.executor.inflight:
                        if self._shutdown_response is not None:
                            self.endpoint.respond(self._shutdown_response)
                        return
                if (
                    not self._closing
                    and len(self._pending) < self.worker.info.queue_depth
                ):
                    request = self.endpoint.try_recv()
                    if request is not None:
                        self._accept(request)
                        continue
                if self._pending or self.worker.executor.inflight:
                    # Consumer acknowledgments are shared words with no wake.
                    # Sweep promptly only while a publication awaits one.
                    awaiting = any(
                        transport.awaiting_acknowledgment()
                        for transport in self.worker.transports.values()
                    )
                    self.endpoint.wait_incoming(
                        1_000 if awaiting else 60_000_000
                    )
                    continue
                self._accept(self.endpoint.recv())
        finally:
            if gc_was_enabled:
                gc.enable()

    def _error(
        self, request: dict[str, Any], error: BaseException
    ) -> dict[str, Any]:
        classified = (
            error
            if isinstance(error, WorkerError)
            else classify(error, context=str(request.get("kind", "unknown")))
        )
        record_failure(
            request.get("kind"),
            classified,
            unexpected=not isinstance(error, WorkerError),
        )
        return messages.error_response(classified, request)

    def _accept(self, request: dict[str, Any]) -> None:
        try:
            kind = messages.request_kind(request)
            if kind is RequestKind.CLOSE:
                self._closing = True
                self._shutdown_response = messages.with_message_id(
                    messages.response(ResponseKind.OK), request
                )
                return
            if kind is RequestKind.INFO:
                response = messages.response(
                    ResponseKind.INFO, info=self.worker.info.to_mapping()
                )
                submission = None
            else:
                raw_batch = messages.required(request, "batch", kind)
                with profile_range("uniserve.worker.batch_decode"):
                    batch = (
                        raw_batch
                        if isinstance(raw_batch, Batch)
                        else Batch.from_mapping(raw_batch)
                    )
                submission = self.worker.submit(batch)
                response = messages.response(ResponseKind.RESULT)
            self._pending.append(
                PendingResponse(
                    messages.with_message_id(response, request), submission
                )
            )
        except BaseException as error:
            self._pending.append(PendingResponse(self._error(request, error)))

    def _send_ready(self) -> bool:
        # Independent completions must not wait behind another request's CPU
        # task or physical retirement. The correlation envelope identifies each.
        for pending in tuple(self._pending):
            response = pending.response
            submission = pending.submission
            if submission is not None:
                try:
                    result = self.worker.poll(submission)
                    if result is None:
                        continue
                    response["result"] = result
                except BaseException as error:
                    response = self._error(response, error)
            self._pending.remove(pending)
            with profile_range(
                worker_range_name(
                    "finalize_response",
                    batch_id=None
                    if submission is None
                    else submission.batch_id,
                    rank=self.worker.info.endpoint.rank,
                )
            ):
                response = messages.finalize_response(response)
            with profile_range("uniserve.worker.respond"):
                self.endpoint.respond(response)
            if response.get("fatal"):
                self._closing = True
            return True
        return False
