"""Admission, execution and retirement shared by direct and IPC callers.

``Executor`` owns the lifetime of every batch a ``Worker`` accepts while
borrowing the Worker's storage owners (``TensorStore``, ``KVCacheManager``,
``LatentPool``, ``RequestPool``). Direct callers reach it through
``Worker.submit``, ``Worker.advance`` and ``Worker.poll``; the IPC loop in
``uniserve_worker.service.Service`` drives the same methods.

``submit`` admits a batch: its id must exceed every earlier one, the
admission queue must have room, and the buffers its ``Free`` commands name
are revoked at once. ``_can_start_batch`` then decides whether it may start
while earlier batches are in flight; one that may not waits in
``_queued_batches`` until ``advance`` retries it. Starting applies the
batch's commands and validates it (``_prepare_execution``), submits its
physical input reads (``advance_inputs``), and launches it through
``step.execute_batch`` once those inputs are ready. Input reads take the
rank's read tickets; while too few are free, preparation waits for one to
return and resumes where it stopped. Launch also begins
retiring its ``Finish`` and ``Free`` commands (``_retire_commands``). Once
every pending output is ready the batch materializes its outputs, and
``_advance_retirement`` completes the retirement behind device events.
``poll`` consumes the result and closes the batch.

Input-readiness callbacks may run on the thread that completes a
dependency. They only enqueue the batch on ``_preparation_ready`` and call
the Worker's completion wake when one is registered
(``_preparation_completed``); the batch is executed later by ``advance`` on
the worker thread.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from queue import SimpleQueue
from typing import TYPE_CHECKING

import torch

from uniserve.runtime.resources import close_resources
from uniserve_worker.errors import (
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
    resource_error,
)
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.media import begin_noise
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.execution.prepare import (
    capture_predicates,
    prepare_batch,
    prepare_inputs,
    validate_batch,
)
from uniserve_worker.execution.step import execute_batch
from uniserve_worker.protocol.batch import Batch, Finish, Free, Start
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.output import BatchOutput
from uniserve_worker.transport.exports import (
    forget_exports,
    release_exports,
    retiring_exports,
)
from uniserve_worker.transport.pool import ReadBackpressureError

if TYPE_CHECKING:
    from uniserve_worker.worker import Worker


@dataclass(frozen=True, slots=True, eq=False)
class Submission:
    """Identity handle consumed once by the Worker that accepted the batch.

    It exposes no mutable execution state. A handle from another Worker, a
    reconstructed handle, or an already consumed handle cannot be polled.
    """

    batch_id: int


def _input_producers(
    batch: Batch,
) -> set[tuple[RequestKey, CallId]]:
    """Return the ``(request, producer call)`` of every input the batch reads.

    Covers tensor inputs, predicates and KV inputs. Those producers' outputs
    must stay visible until this batch has acquired its inputs.
    """
    sources = {
        (reference.request_key, reference.producer_call_id)
        for call in batch.calls
        for reference in (
            *call.tensor_inputs(),
            *(() if call.predicate is None else (call.predicate,)),
        )
    }
    sources.update(
        (call.kv_input.owner, call.kv_input.producer_call_id)
        for call in batch.calls
        if call.kv_input is not None
    )
    return sources


class Executor:
    """Borrow Worker resources while owning accepted batch lifetimes."""

    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.queue_depth = int(worker.info.queue_depth)
        self._last_batch_id = -1
        self._last_collective_seq = -1
        # Accepted batches by id. Insertion order is submission order, which
        # ``_admit_batch`` makes strictly increasing by id.
        self.inflight: dict[int, BatchState] = {}
        self._handles: dict[int, Submission] = {}
        self._preparation_ready: SimpleQueue[BatchState] = SimpleQueue()
        self._executing_batches: deque[BatchState] = deque()
        self._queued_batches: deque[BatchState] = deque()
        # Whether a component this rank owns has a communicator spanning
        # several ranks, or the worker samples across a multi-rank sampling
        # group; ``_can_start_batch`` then runs one batch at a time.
        self._collective_component = any(
            group.size > 1
            for binding in worker.runner.bindings.values()
            if binding.owns
            for group in binding.communicators
        ) or (
            worker.sampling_group is not None and worker.sampling_group.size > 1
        )

    def _require_open(self) -> None:
        self.worker._require_open()

    def close(self) -> None:
        """Release all accepted batches before their backing owners close."""
        try:
            close_resources(
                *(
                    partial(self._close_batch, state)
                    for state in self.inflight.values()
                )
            )
        finally:
            self.inflight.clear()
            self._handles.clear()
            self._executing_batches.clear()
            self._queued_batches.clear()
            while not self._preparation_ready.empty():
                self._preparation_ready.get_nowait()

    def _can_start_batch(self, batch: Batch) -> bool:
        """Whether ``batch`` may start given the earlier in-flight batches.

        Preserves request dependencies and distributed invocation order.
        """
        request_ids = {
            item.request_key.request_id
            for item in (*batch.calls, *batch.commands)
        }
        # ``inflight`` iterates in increasing id and holds this batch, so the
        # scan visits every earlier batch and stops at this one.
        for previous in self.inflight.values():
            if previous.batch_id >= batch.batch_id:
                break
            # A rank of a collective component must not enter a later batch's
            # collectives while a peer is still in an earlier one, so such a
            # rank keeps one batch in flight: it starts a batch only after
            # every earlier one has left ``inflight``, which happens when its
            # result is polled.
            if self._collective_component:
                return False
            # With several ranks, an unlaunched earlier batch blocks every
            # later one; on a single rank it blocks only those sharing one of
            # its requests.
            if not (previous.launched or previous.complete) and (
                self.worker.worker_config.world_size > 1
                or request_ids.intersection(previous.request_ids)
            ):
                return False
        # A single-rank batch waiting for a physical allocation must not block
        # independent requests. Product dependencies still require publication.
        return not self._awaits_local_product(batch)

    def _awaits_local_product(self, batch: Batch) -> bool:
        """Whether an in-flight batch has yet to write a product this reads.

        Preparation resolves a product this rank produced from its own store,
        so a batch naming one cannot be prepared before the batch producing it
        has committed. On a rank that enters collectives, ``_can_start_batch``
        has already required every earlier batch to leave ``inflight``, so no
        earlier producer is found; a rank whose components each sit on one rank
        overlaps preparation with execution, and only the batches that read an
        unwritten product wait.
        """
        return any(
            producer is not None and not producer.launched
            for _request, call_id in _input_producers(batch)
            for producer in (self.inflight.get(call_id.batch_id),)
        )

    def _start_execution(self, batch: BatchState) -> None:
        """Start a batch now or queue it behind earlier work.

        A batch ``_can_start_batch`` refuses joins ``_queued_batches``.
        Otherwise it advances at once: a launched batch joins
        ``_executing_batches``, and one still waiting for its inputs registers
        ``_preparation_completed`` for their readiness. Failures are recorded
        on the batch through ``_fail_run`` rather than raised.
        """
        if not self._can_start_batch(batch.batch):
            self._queued_batches.append(batch)
            return
        try:
            if self._advance_execution(batch):
                if not batch.complete:
                    self._executing_batches.append(batch)
            else:
                self._await_preparation(batch)
        except BaseException as error:
            self._fail_run(batch, error)

    def _await_preparation(self, batch: BatchState) -> None:
        """Wake the owner once the batch can advance its preparation.

        A batch waiting for read tickets was registered for their return
        by ``advance_inputs``; any other waits for its physical dependencies.
        """
        if not batch.awaiting_reads:
            batch.on_dependencies_ready(
                partial(self._preparation_completed, batch)
            )

    def _advance_execution(self, batch: BatchState) -> bool:
        """Prepare and launch a batch as far as its readiness allows.

        Returns False while the batch cannot start or its inputs are not
        ready, and True once it has launched, completed or failed; failures
        are recorded through ``_fail_run``. Runs on the worker thread, never
        from a readiness callback.
        """
        if batch.complete or batch.launched:
            return True
        try:
            if not batch.prepared:
                if not self._can_start_batch(batch.batch):
                    return False
                self._prepare_execution(batch)
            self.advance_inputs(batch)
            if not batch.inputs_ready():
                return False
            self._execute_prepared(batch)
        except BaseException as error:
            self._fail_run(batch, error)
        return True

    def _advance_batch(self, batch: BatchState) -> None:
        """Materialize a launched batch's outputs and advance its retirement.

        Does nothing before launch or after completion. Failures are recorded
        through ``_fail_run``.
        """
        if batch.complete or not batch.launched:
            return
        try:
            self._advance_launched(batch)
        except BaseException as error:
            self._fail_run(batch, error, context="completion materialization")

    def _advance_launched(self, batch: BatchState) -> None:
        """Materialize and retire one launched batch."""
        # CPU work is submitted by the Worker, never by a readiness query.
        for output in batch.outputs:
            if isinstance(output, PendingOutput) and output.value is None:
                for task in output.host.tasks:
                    task.submit_if_ready()

        # Outputs are materialized into wire values, and their results
        # applied to the ``RequestPool``, only once every pending output of
        # the batch is ready.
        if not batch.materialized:
            outputs = tuple(batch.outputs)
            if any(value is None for value in outputs):
                raise RuntimeError("launched batch is missing a call output")
            if all(
                not isinstance(value, PendingOutput) or value.ready()
                for value in outputs
            ):
                pending = tuple(
                    value
                    for value in outputs
                    if isinstance(value, PendingOutput)
                )
                values = tuple(
                    value.materialize()
                    if isinstance(value, PendingOutput)
                    else value
                    for value in outputs
                )
                for output in pending:
                    self.worker.requests.apply_result(output.request_result())
                batch.outputs[:] = values
                batch.materialized = True

        if batch.materialized:
            batch.complete = self._advance_retirement(batch)

    def _fail_run(
        self,
        batch: BatchState,
        error: BaseException,
        *,
        context: str = "execute",
    ) -> None:
        """Record a classified failure, close the batch and mark it complete.

        A batch that is already complete is left unchanged. The recorded
        error reports the batch's call kind as its route unless the raiser set
        one; a lifecycle-only batch has none. A cleanup failure is attached as
        a note to the recorded error.
        """
        if batch.complete:
            return
        # ``classify`` returns a ``WorkerError`` itself and fills only its
        # unset fields, so ``context`` applies to other exceptions alone.
        batch.error = classify(error, context=context, route=batch.route)
        try:
            self._close_batch(batch)
        except BaseException as cleanup_error:
            batch.error.add_note(f"batch cleanup failed: {cleanup_error!r}")
        batch.complete = True

    def _preparation_completed(self, batch: BatchState) -> None:
        """Enqueue readiness before waking the IPC loop that consumes it.

        May run on whichever thread completes the batch's last dependency, so
        it only enqueues and wakes.
        """
        self._preparation_ready.put(batch)
        if self.worker._completion_wake is not None:
            self.worker._completion_wake()

    def _advance_executing_batches(self) -> bool:
        """Advance one preparation-ready batch and every executing one.

        Returns whether any batch made progress.
        """
        advanced = False
        # One readiness notification per pass; a True result makes
        # ``Service.run`` call ``advance`` again at once.
        if not self._preparation_ready.empty():
            batch = self._preparation_ready.get_nowait()
            if not batch.complete:
                launched = self._advance_execution(batch)
                if not batch.complete:
                    if launched:
                        self._executing_batches.append(batch)
                    else:
                        self._await_preparation(batch)
            advanced = True

        # Query every launched batch: one pending host read or retirement
        # must not hide an independent completion behind it.
        for _ in range(len(self._executing_batches)):
            batch = self._executing_batches.popleft()
            before = batch.materialized
            self._advance_batch(batch)
            advanced |= batch.complete or batch.materialized != before
            if not batch.complete:
                self._executing_batches.append(batch)
        return advanced

    def submit(
        self, batch: Batch, *, propagate_errors: bool = False
    ) -> Submission:
        """Accept one batch and submit available work.

        Retains the batch's asynchronous state. Call advance to progress
        pending inputs, CPU work, and retirement, then poll to consume the
        batch's result. A batch identity remains owned until its result is
        consumed or the Worker closes. IDs strictly increase, including after
        completion; a repeated or lower id raises ``invalid_descriptor``. A
        full admission queue raises ResourceError without consuming the id;
        consume a result before retrying. Launch preserves request
        dependencies and distributed invocation order while independent local
        work can proceed. With ``propagate_errors``, a failure recorded during
        submission is raised after the batch is released.
        """
        self._require_open()
        self._admit_batch(batch)

        state = BatchState(batch, propagate_errors=propagate_errors)
        self.inflight[state.batch_id] = state

        try:
            self._start_execution(state)
            self._advance_batch(state)
        except BaseException:
            self.inflight.pop(state.batch_id, None)
            self._close_batch(state)
            raise

        if state.error is not None and propagate_errors:
            error = state.error
            self.inflight.pop(state.batch_id, None)
            self._close_batch(state)
            raise error
        submission = Submission(state.batch_id)
        self._handles[state.batch_id] = submission
        return submission

    def _admit_batch(self, batch: Batch) -> None:
        """Claim identity and capacity, then revoke freed products."""
        if batch.batch_id <= self._last_batch_id:
            raise invalid_descriptor(
                f"batch id {batch.batch_id} must exceed previously "
                f"submitted id {self._last_batch_id}"
            )
        if len(self.inflight) >= self.queue_depth:
            raise resource_error("worker admission queue is full")
        self._last_batch_id = batch.batch_id
        # Revocation cannot wait behind an earlier allocation-dependent call.
        # Existing readers keep their leases until their actual last access.
        freed = tuple(
            command.buffer
            for command in batch.commands
            if isinstance(command, Free)
        )
        if freed:
            self.release_buffers(freed)

    def advance(self) -> bool:
        """Progress dependencies, computation and physical retirement.

        Returns whether any batch made progress.
        """
        self._require_open()
        self.worker.device_events.reap()
        for transport in self.worker.transports.values():
            transport.reap()
        advanced = self._advance_executing_batches()
        for _ in range(len(self._queued_batches)):
            state = self._queued_batches.popleft()
            if self._can_start_batch(state.batch):
                self._start_execution(state)
                advanced = True
            else:
                self._queued_batches.append(state)
        return advanced

    def poll(self, submission: Submission) -> BatchOutput | None:
        """Consume the batch's result without launching computation.

        Returns None while the batch is not ready. Otherwise the batch is
        released and its output returned, or its recorded error raised. A
        handle this Worker does not own raises ``invalid_descriptor``.
        """
        self._require_open()
        state = self.inflight.get(submission.batch_id)
        if (
            state is None
            or self._handles.get(submission.batch_id) is not submission
        ):
            raise invalid_descriptor(
                "poll names a batch no longer owned by this Worker"
            )
        if not state.ready():
            return None

        if state.error is not None:
            error = state.take_error()
            del self.inflight[state.batch_id]
            del self._handles[state.batch_id]
            self._close_batch(state)
            raise error

        output = state.take_output()
        del self.inflight[state.batch_id]
        del self._handles[state.batch_id]
        self._close_batch(state)
        return output

    def _execute_batch(self, state: BatchState) -> None:
        """Launch a prepared batch and release what its launch consumed.

        Afterwards the batch is marked launched and its command retirement
        begins (``_retire_commands``).
        """
        batch = state.batch
        # Cooperative ranks launch computation in increasing
        # ``collective_seq``; preparation and host completion may overlap.
        if self.worker.worker_config.world_size > 1 and batch.calls:
            if batch.collective_seq <= self._last_collective_seq:
                # Peers of an out-of-order batch are already inside the
                # collective this rank would have joined, so failing only this
                # batch would leave them waiting for a participant that never
                # arrives. The rank cannot serve further collective work.
                raise WorkerError(
                    code=WorkerErrorCode.INVARIANT_VIOLATION,
                    message=(
                        f"collective sequence does not advance: batch "
                        f"{batch.batch_id} carries {batch.collective_seq} "
                        f"after {self._last_collective_seq}"
                    ),
                    fatal=True,
                )
            self._last_collective_seq = batch.collective_seq

        # A batch with calls is one numerical step of the worker: the profiler's
        # capture window counts these steps, and the step name becomes the NVTX
        # range that Nsight shows for the batch. Lifecycle-only batches carry no
        # computation and are not counted.
        step: AbstractContextManager[None]
        if self.worker.profiler is not None and batch.calls:
            first = batch.calls[0]
            step = self.worker.profiler.step(
                f"batch:{first.kind}:{first.component}"
            )
        else:
            step = nullcontext()

        with step:
            execute_batch(
                state,
                propagate_errors=state.propagate_errors,
                kv_cache=self.worker.kv_cache,
                host_tasks=self.worker.host_tasks,
                tensor_store=self.worker.tensor_store,
                worker_info=self.worker.info,
                latent_pool=self.worker.latent_pool,
                media_mux=self.worker.media_mux,
                output_pool=self.worker.output_pool,
                publication_transports=self.worker.publication_transports,
                request_tables=self.worker.block_tables,
                request_pool=self.worker.requests,
                model_runner=self.worker.runner,
                decode_state=self.worker.decode_state,
                sampling_group=self.worker.sampling_group,
                tokenizer=self.worker.tokenizer,
                transfer_backends=self.worker.transports,
                config=self.worker.worker_config,
            )

        # Predecessor outputs this batch reads are revoked only now that its
        # inputs are acquired; ``_prepare_execution`` revoked the unread ones.
        consumed = _input_producers(batch)
        self._release_predecessors(
            tuple(
                (call.request_key, predecessor)
                for call in batch.calls
                if (predecessor := state.predecessor(call)) is not None
                and predecessor.batch_id > 0
                and (call.request_key, predecessor) in consumed
            )
        )

        # A predicate from the call's predecessor was revoked with that call's
        # outputs above; any other predicate is revoked by its buffer id.
        self.worker.tensor_store.release_buffers(
            tuple(
                predicate.buffer_id
                for call in batch.calls
                if (predicate := call.predicate) is not None
                and predicate.producer_call_id != state.predecessor(call)
            )
        )
        state.launched = True
        self._retire_commands(state)

    def _prepare_execution(self, state: BatchState) -> None:
        """Apply the batch's commands, validate it and prepare its inputs.

        Raises ``invalid_descriptor`` for call kinds this worker cannot
        execute, and propagates validation and preparation failures.
        """
        batch = state.batch
        unsupported = {
            call.kind.value
            for call in batch.calls
            if not self.worker.supports_computation(call.kind)
        }
        if unsupported:
            raise invalid_descriptor(
                "execution batch contains call kinds unsupported by "
                f"this worker: {sorted(unsupported)!r}"
            )
        # A request's first call on this rank arrives with the command that
        # establishes its lineage, so the lineage is installed before naming
        # what each call follows. The calls of a request this rank has yet to
        # admit would otherwise follow nothing.
        starts = tuple(
            command for command in batch.commands if isinstance(command, Start)
        )
        for start in starts:
            slots = self.worker.requests.apply_commands((start,))
            if slots and self.worker.decode_state is not None:
                self.worker.decode_state.reset(slots)
            if slots and start.request.diffusion is not None:
                # A video request's seeded noise is drawn while this batch and
                # the ones before latent preparation run on the device.
                begin_noise(
                    self.worker.runner,
                    self.worker.requests.get(
                        start.request.request_key.request_id
                    ),
                    self.worker.requests,
                )

        state.predecessors = self.worker.requests.predecessors(batch.calls)
        validate_batch(
            batch,
            worker_info=self.worker.info,
            model_runner=self.worker.runner,
            config=self.worker.worker_config,
            predecessors=state.predecessors,
        )

        for command in batch.commands:
            if isinstance(command, Start):
                continue
            slots = self.worker.requests.apply_commands((command,))
            if slots and self.worker.decode_state is not None:
                self.worker.decode_state.reset(slots)

        # Predecessor outputs no call of this batch reads are revoked before
        # input acquisition; ``_execute_batch`` revokes the rest after launch.
        # Batch id zero is the admission root, which has no outputs.
        consumed = _input_producers(batch)
        self._release_predecessors(
            tuple(
                (call.request_key, predecessor)
                for call in batch.calls
                if (predecessor := state.predecessor(call)) is not None
                and predecessor.batch_id > 0
                and (call.request_key, predecessor) not in consumed
            )
        )
        self._release_commands(batch)
        prepare_batch(
            state,
            kv_cache=self.worker.kv_cache,
            latent_pool=self.worker.latent_pool,
            request_tables=self.worker.block_tables,
            request_pool=self.worker.requests,
        )
        state.prepared = True
        self.advance_inputs(state)

    def advance_inputs(self, state: BatchState) -> None:
        """Submit ready physical reads and predicate copies.

        Does not launch a model. Input reads are submitted after every
        storage dependency is done when the batch imports products or KV;
        later calls only capture predicates. While too few read tickets are
        free for the next import, the batch is marked ``awaiting_reads`` and
        woken when one returns, and the next call resumes at that import.
        """
        if state.inputs_closed:
            return
        if state.inputs_submitted:
            capture_predicates(state, self.worker.tensor_store)
            return
        # A destination remains unavailable until the previous physical reader
        # retires. Dependencies with no import still gate model execution.
        if (state.input_products or state.kv_inputs) and not all(
            dependency.done() for dependency in state.storage_dependencies
        ):
            return
        if state.input_products or state.kv_inputs:
            for dependency in state.storage_dependencies:
                dependency.result()

        state.awaiting_reads = False
        try:
            prepare_inputs(
                state,
                kv_cache=self.worker.kv_cache,
                tensor_store=self.worker.tensor_store,
                latent_pool=self.worker.latent_pool,
                output_pool=self.worker.output_pool,
                request_tables=self.worker.block_tables,
                request_pool=self.worker.requests,
                model_runner=self.worker.runner,
                transfer_backends=self.worker.transports,
                config=self.worker.worker_config,
            )
        except ReadBackpressureError as error:
            # Read tickets return as reads retire, independently of this
            # batch, so the batch resumes once one does.
            state.awaiting_reads = True
            error.capacity.notify_reads_returned(
                partial(self._preparation_completed, state),
                after=error.returns,
            )
            return
        state.inputs_submitted = True
        capture_predicates(state, self.worker.tensor_store)

    def _retire_commands(self, state: BatchState) -> None:
        """Begin retiring the batch's ``Finish`` and ``Free`` commands.

        Releases the closed requests' tensor-store products, cancels their
        pending latent and KV imports, and revokes the exports of closed
        requests and freed buffers, keeping each ``Finish``'s retained
        buffers. What ``_advance_retirement`` must still wait for is recorded
        on ``state``. A batch without such commands is marked retired at once.
        """
        batch = state.batch
        closed = frozenset(
            command.request_key
            for command in batch.commands
            if isinstance(command, Finish)
        )
        # Only an epoch resident on this rank has request state to retire.
        local_closed = frozenset(
            key
            for key in closed
            if (row := self.worker.requests.peek(key.request_id)) is not None
            and row.request_key == key
        )
        freed = frozenset(
            command.buffer
            for command in batch.commands
            if isinstance(command, Free)
        )

        if not closed and not freed:
            state.retirement_cleaned = True
            return

        retained = (
            frozenset(
                buffer
                for command in batch.commands
                if isinstance(command, Finish)
                for buffer in command.retained_buffers
            )
            - freed
        )
        self.worker.tensor_store.release_requests(closed, retained=retained)

        if self.worker.latent_pool is not None:
            self.worker.latent_pool.cancel_imports(tuple(closed))

        stores = tuple(
            store
            for store in (
                self.worker.tensor_store,
                self.worker.kv_cache,
                self.worker.latent_pool,
            )
            if store is not None
        )
        selected = tuple(
            buffer
            for store in stores
            for buffer in retiring_exports(
                store.exports, buffers=freed, requests=closed, retained=retained
            )
        )
        # Revoking a publication ends new grants. Its physical retirement is
        # the storage owner's to observe: a write is not reclaimed while any
        # publication it retained is live, so retirement needs no second wait
        # on the same futures.
        for store in stores:
            release_exports(store.exports, selected)

        if self.worker.latent_pool is not None:
            self.worker.latent_pool.release_buffers(selected)
        if self.worker.kv_cache is not None:
            self.worker.kv_cache.imports.cancel_requests(
                closed, retained=retained
            )
            self.worker.kv_cache.release_buffers(selected)

        state.retirement_requests = closed
        state.retirement_local_requests = local_closed
        state.retirement_buffers = freed
        state.retained_buffers = retained
        state.retirement_exports = selected
        # Finish includes request-state writes issued after output capture.
        state.retirement_events = (
            self._record_retirement_events() if closed else ()
        )

    def _record_retirement_events(self) -> tuple[torch.cuda.Event, ...]:
        """Record one tracked event per CUDA device in the buffer pool.

        When the Worker has a completion wake bound, each event schedules it
        to run once the device work recorded before the event finishes.
        """
        events = []
        for device in self.worker.buffer_pool.devices:
            if device.type != "cuda":
                continue
            event = self.worker.device_events.acquire(device)
            self.worker.device_events.retain(event, device)
            self.worker.device_events.record(event, device)
            self.worker.device_events.schedule_completion_wake(device, event)
            events.append(event)
        return tuple(events)

    def _advance_retirement(self, state: BatchState) -> bool:
        """Advance a launched batch's command retirement.

        Waits for the release work's events, then for every store to report
        the closed requests and freed buffers ready, then forgets their
        exports and retires this rank's closed request epochs. Retiring
        submits slot-reset writes, so fresh events are recorded and a later
        call returns True once they complete.

        Returns:
            Whether retirement has finished.
        """
        self.worker.device_events.reap()
        # A device product is held until its consumers acknowledge it. They do
        # so by writing into the chunk they read, which arrives with no local
        # notification, so the producing rank looks for it here.
        for transport in self.worker.transports.values():
            transport.reap()
        if not all(event.query() for event in state.retirement_events):
            return False

        for event in state.retirement_events:
            self.worker.device_events.release(event)
        state.retirement_events = ()

        if state.retirement_cleaned:
            return True

        closed = state.retirement_requests
        freed = state.retirement_buffers
        retained = state.retained_buffers

        if any(
            not self.worker.requests.retirement_ready(key)
            for key in state.retirement_local_requests
        ):
            return False
        if not self.worker.tensor_store.retirement_ready(
            buffers=freed, requests=closed, retained=retained
        ):
            return False
        if (
            self.worker.latent_pool is not None
            and not self.worker.latent_pool.retirement_ready(tuple(closed))
        ):
            return False
        if (
            self.worker.kv_cache is not None
            and not self.worker.kv_cache.retirement_ready(
                buffers=freed, requests=closed, retained=retained
            )
        ):
            return False
        for store in (
            self.worker.tensor_store,
            self.worker.kv_cache,
            self.worker.latent_pool,
        ):
            if store is not None:
                forget_exports(store.exports, state.retirement_exports)
        for key in state.retirement_local_requests:
            self.retire_request(key, retained=retained)

        # Slot reset itself submits writes; their completion permits
        # address reuse.
        state.retirement_events = (
            self._record_retirement_events() if closed else ()
        )
        state.retirement_cleaned = True
        return not state.retirement_events

    def _close_batch(self, state: BatchState) -> None:
        """Release an owned batch whose result is consumed or abandoned.

        Calls whose outputs were never materialized are cancelled in the
        ``RequestPool``, which closes their requests. Outstanding retirement
        events are released once complete, and ``BatchState.close`` abandons
        inputs and outputs while physical readers keep their own leases.
        """
        pending = tuple(
            output
            for output in state.outputs
            if isinstance(output, PendingOutput)
        )
        self.worker.requests.cancel_calls(
            tuple(output.call for output in pending)
        )
        if state.retirement_events:
            self.worker.device_events.defer_release(
                state.retirement_events, state
            )
            state.retirement_events = ()
        state.close(
            self.worker.tensor_store,
            self.worker.latent_pool,
            self.worker.kv_cache,
        )

    def _execute_prepared(self, state: BatchState) -> None:
        """Launch a batch whose inputs are ready, then close its inputs.

        Raises ``RuntimeError`` when the inputs were already consumed, or
        when they are not ready, in which case they stay open. Otherwise
        ``BatchState.close_inputs`` runs after ``_execute_batch`` whether or
        not it raises.
        """
        self._require_open()
        if state.inputs_closed:
            raise RuntimeError("batch inputs have already been consumed")

        self.advance_inputs(state)
        if not state.inputs_ready():
            raise RuntimeError("batch was observed before dependency readiness")

        try:
            for dependency in state.storage_dependencies:
                dependency.result()
            self._execute_batch(state)
        except BaseException as error:
            try:
                state.close_inputs(
                    self.worker.tensor_store,
                    self.worker.latent_pool,
                    self.worker.kv_cache,
                )
            except BaseException as cleanup_error:
                error.add_note(f"batch input cleanup failed: {cleanup_error!r}")
            raise
        state.close_inputs(
            self.worker.tensor_store,
            self.worker.latent_pool,
            self.worker.kv_cache,
        )

    def release_buffers(self, buffers: Sequence[BufferId]) -> None:
        """Revoke new reads of ``buffers`` in every store.

        Existing readers keep their leases until their last access.
        """
        self.worker.tensor_store.release_buffers(buffers)
        if self.worker.kv_cache is not None:
            self.worker.kv_cache.release_buffers(buffers)
        if self.worker.latent_pool is not None:
            self.worker.latent_pool.release_buffers(buffers)

    def _release_predecessors(
        self, predecessors: tuple[tuple[RequestKey, CallId], ...]
    ) -> None:
        """Revoke predecessor call outputs in the tensor store and KV cache.

        Callers pass only predecessors whose declared consumers in the batch
        have acquired them or do not read them.
        """
        self.worker.tensor_store.release_calls(predecessors)
        if self.worker.kv_cache is not None:
            released = self.worker.kv_cache.release_calls(predecessors)
            self.worker.kv_cache.release_buffers(released)

    def _release_commands(self, batch: Batch) -> None:
        """Revoke what the batch's ``Free`` and ``Finish`` commands release.

        Runs during preparation, before the batch can wait for the storage
        its ``Free`` and ``Finish`` commands make reusable. Each ``Finish``
        keeps its retained buffers, and the finished requests' KV imports are
        cancelled.
        """
        freed = {
            command.buffer
            for command in batch.commands
            if isinstance(command, Free)
        }
        closed = {
            command.request_key: frozenset(command.retained_buffers) - freed
            for command in batch.commands
            if isinstance(command, Finish)
        }

        closing = tuple(
            buffer
            for request_key, retained in closed.items()
            for store in (
                self.worker.tensor_store,
                self.worker.kv_cache,
                self.worker.latent_pool,
            )
            if store is not None
            for buffer in retiring_exports(
                store.exports,
                requests=frozenset((request_key,)),
                retained=retained | freed,
            )
        )
        self.release_buffers((*freed, *closing))

        if self.worker.kv_cache is not None:
            for request_key, retained in closed.items():
                self.worker.kv_cache.imports.cancel_requests(
                    frozenset((request_key,)), retained=retained
                )

    def _release_request(
        self, request_id: int, retained: frozenset[BufferId]
    ) -> None:
        """Release a drained request's storage on this rank.

        Buffers in ``retained`` are kept; every other export and tensor-store
        product the request owns is released with its KV imports and cache
        state, decode state, block tables, prefix slots, media mux state and
        latent slot.
        """
        request = self.worker.requests.peek(request_id)
        if request is not None:
            if self.worker.kv_cache is not None:
                self.worker.kv_cache.imports.cancel_requests(
                    frozenset((request.request_key,)), retained=retained
                )
            if self.worker.decode_state is not None:
                self.worker.decode_state.reset((request.request_pool_idx,))
            if self.worker.block_tables is not None:
                self.worker.block_tables.release((request.request_pool_idx,))

        if self.worker.kv_cache is not None:
            self.worker.kv_cache.drop(request_id)
        if request is not None and self.worker.block_tables is not None:
            self.worker.block_tables.release_prefixes(request.request_key)

        for store in (
            self.worker.tensor_store,
            self.worker.kv_cache,
            self.worker.latent_pool,
        ):
            if store is not None:
                selected = tuple(
                    buffer
                    for buffer in store.exports
                    if int(buffer.owner.request_id) == request_id
                    and buffer not in retained
                )
                store.release_buffers(selected)

        if request is not None:
            self.worker.tensor_store.release_requests(
                (request.request_key,), retained=retained
            )
        if self.worker.media_mux is not None:
            self.worker.media_mux.drop(request_id)
        if request is not None and self.worker.latent_pool is not None:
            self.worker.latent_pool.release_slots((request.request_pool_idx,))

    def drop_request(self, request_id: int) -> None:
        """Release a drained request and remove its admission from the pool."""
        request_id = int(request_id)
        self._release_request(request_id, frozenset())
        self.worker.requests.drop(request_id)

    def retire_request(
        self,
        request_key: RequestKey,
        *,
        retained: frozenset[BufferId] = frozenset(),
    ) -> None:
        """Retire exactly ``request_key``'s epoch on this rank.

        Does nothing when that epoch is not resident or is already retired.
        The caller must have waited for its readers to drain, as
        ``_advance_retirement`` does through the stores' ``retirement_ready``
        checks. Buffers in ``retained`` are kept.
        """
        request = self.worker.requests.peek(request_key.request_id)
        if (
            request is None
            or request.request_key != request_key
            or request.retired
        ):
            return
        self._release_request(request_key.request_id, retained)
        self.worker.requests.retire(request_key.request_id)
