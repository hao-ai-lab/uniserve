"""Synchronous IPC admission and response delivery for one Worker.

`Service` is the worker process's serving loop. It borrows the
`WorkerIpcEndpoint` bound through `Worker.bind`, decodes engine requests,
submits batches to the same `Executor` that direct Python callers reach
through `Worker.submit`, and sends each response once `Worker.poll` yields its
result. Responses carry the request's ``message_id`` and are sent as their
results become ready, so their order can differ from request order. The loop
runs on the caller's thread. While work is outstanding it waits in
``wait_incoming``; host lanes, transports, KV imports and CUDA stream
callbacks end that wait through the endpoint wake callbacks that `Worker.bind`
registers with `Worker.set_completion_wake`.
"""

from __future__ import annotations

import gc
import logging
import time
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

# How long, in microseconds, an expert-parallel rank without a forward of its
# own waits for a request or completion before it agrees on the next expert
# step again. It bounds how long a step another rank starts waits for this
# rank while all ranks are idle.
_EXPERT_STEP_POLL_US = 200
# How long, in seconds, an expert-parallel rank with open requests waits after
# sending a result for its engine's next batch before it joins other ranks'
# steps instead: the p99 of the engine's result-to-next-batch latency measured
# on DEP4 DiffusionGemma generation was 5.7-7.6 ms per rank (p50 1.0-1.2 ms).
_OWN_STEP_WAIT_S = 0.008

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingResponse:
    """An IPC correlation envelope and its optional execution handle.

    ``kind`` is the kind field of the request the response answers, as
    received, which a failure log names. ``submission`` is None for responses
    that are complete when accepted: INFO replies and admission errors.
    """

    response: dict[str, Any]
    kind: object
    submission: Submission | None = None


class Service:
    """Borrow an endpoint and drive the same Executor as direct callers."""

    def __init__(self, worker: Worker, endpoint: WorkerIpcEndpoint) -> None:
        self.worker = worker
        self.endpoint = endpoint
        # Accepted requests whose responses are not yet sent. Its length is
        # bounded by `WorkerInfo.queue_depth`, and every in-flight executor
        # batch has an entry here, so `Executor.submit` never sees a full
        # admission queue from this loop.
        self._pending: deque[PendingResponse] = deque()
        # Set by Close or by a fatal response: stop receiving, drain pending
        # responses and in-flight batches, then return.
        self._closing = False
        # The Close acknowledgment, sent only after the drain completes.
        self._shutdown_response: dict[str, Any] | None = None
        # When this rank last sent a result, and whether its wait for its own
        # next expert step since then has already run out.
        self._last_result = time.monotonic()
        self._own_step_expired = False

    def run(self) -> None:
        """Serve until Close drains accepted work; leave resources to Worker.

        Each iteration first advances the executor and sends at most one ready
        response, then receives only when there was no progress. Close is
        acknowledged after every accepted response has been sent and the
        executor holds no in-flight batch. A fatal response also stops
        admission and drains; the loop then returns with an acknowledgment
        only if Close was already received.
        """
        # A full cyclic collection traverses every resident model, CUDA graph
        # and attention-wrapper object. Request state is ownership-structured
        # and reference-counted, so collection stays off the serving path; the
        # caller's GC mode is restored on return.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            while True:
                advanced = self.worker.executor.advance()
                if self._send_ready() or advanced:
                    continue

                if self._closing and not self._pending:
                    if not self.worker.executor.inflight:
                        # An expert-parallel rank leaves only once every rank
                        # of its group is leaving, so none waits on it.
                        runner = self.worker.runner
                        if runner.experts is not None:
                            runner.join_expert_step(leaving=True)
                            if not runner.experts.released:
                                continue
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

                # An expert-parallel rank takes part in every step another
                # rank of its expert group starts while it has no forward of
                # its own to launch, as vLLM's data-parallel engines run a
                # dummy batch while any engine has work
                # (``v1/engine/core.py:1870-1927``); it therefore never
                # blocks for long. A rank whose own next batch is coming
                # waits for it instead (``_awaits_own_step``).
                experts = self.worker.runner.experts is not None
                if experts and self._awaits_own_step():
                    self.endpoint.wait_incoming(_EXPERT_STEP_POLL_US)
                    continue
                if experts and self.worker.runner.join_expert_step():
                    continue

                if self._pending or self.worker.executor.inflight:
                    # Work is outstanding: wait for a request or a completion
                    # wake (timeouts are in microseconds). Consumer
                    # acknowledgments are shared words with no wake, so sweep
                    # on a short period only while a publication awaits one.
                    awaiting = any(
                        transport.awaiting_acknowledgment()
                        for transport in self.worker.transports.values()
                    )
                    self.endpoint.wait_incoming(
                        _EXPERT_STEP_POLL_US
                        if experts
                        else 1_000
                        if awaiting
                        else 60_000_000
                    )
                    continue

                if experts:
                    # Idle: poll for a request or another rank's step.
                    self.endpoint.wait_incoming(_EXPERT_STEP_POLL_US)
                    continue

                # Fully idle: block until the next request.
                self._accept(self.endpoint.recv())
        finally:
            if gc_was_enabled:
                gc.enable()

    def _error(
        self,
        kind: object,
        correlation: dict[str, Any],
        error: BaseException,
    ) -> dict[str, Any]:
        """Log a failed ``kind`` request and encode it as an error response.

        ``correlation`` is the request or its pending response envelope;
        either supplies the ``message_id`` that correlates the error. Errors
        that are not already `WorkerError` are classified and logged as
        unexpected.
        """
        classified = (
            error
            if isinstance(error, WorkerError)
            else classify(
                error, context="unknown" if kind is None else str(kind)
            )
        )
        record_failure(
            kind,
            classified,
            unexpected=not isinstance(error, WorkerError),
        )
        return messages.error_response(classified, correlation)

    def _awaits_own_step(self) -> bool:
        """Whether this expert-parallel rank waits for its own next batch.

        A rank with open requests takes part in every expert step with a
        forward of its own instead of joining another rank's step with none,
        as a vLLM data-parallel engine with unfinished requests always runs
        its own scheduled step and only an engine without them runs the
        dummy batch (``DPEngineCoreProc.run_busy_loop``,
        ``v1/engine/core.py:1870-1927``). Ranks with work then step together
        rather than taking turns, each of the others joining at every step.

        While one of its batches is in flight or awaiting its response, the
        rank's engine sends its next batch after that result, so it waits;
        after its last result it waits at most ``_OWN_STEP_WAIT_S`` and then
        joins, logging the expiry. The wait cannot deadlock: every batch in
        flight was launched after its step's agreement on every rank, so it
        completes and its result goes out without any further agreement, and
        the engine's next batch depends on that result alone.
        """
        if not self.worker.requests.has_open_requests():
            return False
        if self._pending or self.worker.executor.inflight:
            return True
        if self._own_step_expired:
            return False
        waited = time.monotonic() - self._last_result
        if waited < _OWN_STEP_WAIT_S:
            return True
        self._own_step_expired = True
        logger.info(
            "expert step: no batch arrived within %.1f ms of this rank's last "
            "result while it holds open requests; joining the other ranks' "
            "steps until it does",
            waited * 1e3,
        )
        return False

    def _accept(self, request: dict[str, Any]) -> None:
        """Admit one request, queueing its response envelope.

        Close is not queued: it stops admission and its acknowledgment waits
        for the drain in `run`. Any failure while decoding or submitting is
        queued as an error response instead of propagating.
        """
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
                # The IPC extension delivers a constructed `Batch`; a plain
                # mapping is decoded and validated here.
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
                    messages.with_message_id(response, request),
                    request.get("kind"),
                    submission,
                )
            )
        except BaseException as error:
            self._pending.append(
                PendingResponse(
                    self._error(request.get("kind"), request, error),
                    request.get("kind"),
                )
            )

    def _send_ready(self) -> bool:
        """Send the first ready pending response, if any.

        Returns whether a response was sent. Polling consumes a batch result
        or its error; an error replaces the result response under the same
        ``message_id``. A response marked fatal starts the shutdown drain.
        """
        # Independent completions must not wait behind another request's CPU
        # task or physical retirement, so every pending entry is polled rather
        # than only the oldest. The correlation envelope identifies each.
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
                    # The error answers the request, so it is logged under
                    # the request's kind rather than the envelope's.
                    response = self._error(pending.kind, response, error)

            self._pending.remove(pending)
            self._last_result = time.monotonic()
            self._own_step_expired = False
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
