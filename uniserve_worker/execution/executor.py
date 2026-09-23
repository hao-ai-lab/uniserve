"""Admission, execution and retirement shared by direct and IPC callers."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from contextlib import nullcontext
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
    """Identify producers requiring extended visibility.

    Their values must remain visible until input acquisition.
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
        self.inflight: dict[int, BatchState] = {}
        self._handles: dict[int, Submission] = {}
        self._preparation_ready: SimpleQueue[BatchState] = SimpleQueue()
        self._executing_batches: deque[BatchState] = deque()
        self._queued_batches: deque[BatchState] = deque()
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
        """Preserve request dependencies and distributed invocation order."""
        request_ids = {
            item.request_key.request_id
            for item in (*batch.calls, *batch.commands)
        }
        for previous in self.inflight.values():
            if previous.batch_id >= batch.batch_id:
                break
            if self._collective_component:
                return False
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
        has committed. A rank that enters collectives holds one batch in
        flight and never reaches this; a rank whose components each sit on one
        rank overlaps preparation with execution, and only the batches that
        read an unwritten product wait.
        """
        return any(
            producer is not None and not producer.launched
            for _request, call_id in _input_producers(batch)
            for producer in (self.inflight.get(call_id.batch_id),)
        )

    def _start_execution(self, batch: BatchState) -> None:
        """Submit physical inputs for the batch.

        Directly launches the batch when the inputs are ready.
        """
        if not self._can_start_batch(batch.batch):
            self._queued_batches.append(batch)
            return
        try:
            if self._advance_execution(batch):
                if not batch.complete:
                    self._executing_batches.append(batch)
            else:
                batch.on_dependencies_ready(
                    partial(self._preparation_completed, batch)
                )
        except BaseException as error:
            self._fail_run(batch, error)

    def _advance_execution(self, batch: BatchState) -> bool:
        """Execute prepared inputs on the worker thread.

        Never executes from a notification callback.
        """
        if batch.complete or batch.launched:
            return True
        try:
            if not batch.inputs_submitted:
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
        """Materialize the batch's outputs once every pending one is ready.

        Also advances physical command retirement.
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
        """Record a classified failure and close the batch.

        Also wakes its waiting responses.
        """
        if batch.complete:
            return
        batch.error = (
            error
            if isinstance(error, WorkerError)
            else classify(error, context=context)
        )
        try:
            self._close_batch(batch)
        except BaseException as cleanup_error:
            batch.error.add_note(f"batch cleanup failed: {cleanup_error!r}")
        batch.complete = True

    def _preparation_completed(self, batch: BatchState) -> None:
        """Enqueue readiness before waking the IPC loop that consumes it."""
        self._preparation_ready.put(batch)
        if self.worker._completion_wake is not None:
            self.worker._completion_wake()

    def _advance_executing_batches(self) -> bool:
        """Launch preparation-ready work.

        Also advances executing runs in launch order.
        """
        advanced = False
        if not self._preparation_ready.empty():
            batch = self._preparation_ready.get_nowait()
            if not batch.complete:
                launched = self._advance_execution(batch)
                if not batch.complete:
                    if launched:
                        self._executing_batches.append(batch)
                    else:
                        batch.on_dependencies_ready(
                            partial(self._preparation_completed, batch)
                        )
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
        completion. A full admission queue raises ResourceError; consume a
        result before retrying. Launch preserves request dependencies and
        distributed invocation order while independent local work can proceed.
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
        """Progress dependencies, computation and physical retirement."""
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
        """Consume the batch's result without launching computation."""
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
        """Execute prepared numerical work.

        Retains the work's physical retirement facts.
        """
        batch = state.batch
        # Cooperative ranks launch computation in the same order. Preparation
        # and host completion may overlap; neither retains old batch identities.
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
        """Validate the batch and apply its commands.

        Also prepares the batch's physical inputs.
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
        for command in starts:
            slots = self.worker.requests.apply_commands((command,))
            if slots and self.worker.decode_state is not None:
                self.worker.decode_state.reset(slots)
            if slots and command.request.diffusion is not None:
                # A video request's seeded noise is drawn while this batch and
                # the ones before latent preparation run on the device.
                begin_noise(
                    self.worker.runner,
                    self.worker.requests.get(
                        command.request.request_key.request_id
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
        self.advance_inputs(state)

    def advance_inputs(self, state: BatchState) -> None:
        """Submit ready physical reads and predicate copies.

        Does not launch a model.
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
        state.inputs_submitted = True
        capture_predicates(state, self.worker.tensor_store)

    def _retire_commands(self, state: BatchState) -> None:
        """Submit release work for the batch's commands.

        Retains the events and futures required by its acknowledgement.
        """
        batch = state.batch
        closed = frozenset(
            command.request_key
            for command in batch.commands
            if isinstance(command, Finish)
        )
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
        """Record tracked completion events.

        One event per CUDA device in the buffer pool.
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
        """Reset retired request slots.

        Reset happens only after every physical reader has finished.
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
        """Cancel unresolved acceptance.

        Real CPU and GPU readers retain storage meanwhile.
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
        """Directly execute physical inputs once.

        Releases their preparation leases.
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
        """Revoke new reads immediately.

        Storage owners retain existing readers.
        """
        self.worker.tensor_store.release_buffers(buffers)
        if self.worker.kv_cache is not None:
            self.worker.kv_cache.release_buffers(buffers)
        if self.worker.latent_pool is not None:
            self.worker.latent_pool.release_buffers(buffers)

    def _release_predecessors(
        self, predecessors: tuple[tuple[RequestKey, CallId], ...]
    ) -> None:
        """Revoke predecessor outputs.

        Revocation happens after every declared consumer has acquired them.
        """
        self.worker.tensor_store.release_calls(predecessors)
        if self.worker.kv_cache is not None:
            released = self.worker.kv_cache.release_calls(predecessors)
            self.worker.kv_cache.release_buffers(released)

    def _release_commands(self, batch: Batch) -> None:
        """Apply Free/Finish visibility.

        Visibility applies before work can wait for their reusable storage.
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
        """Reset a drained request's storage.

        Independently owned products are preserved.
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
        """Retire the exact epoch after readers drain.

        Independent products are retained.
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
