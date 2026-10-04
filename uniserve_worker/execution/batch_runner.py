"""Numerical preparation and resource retirement for the native executor."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import TYPE_CHECKING

from uniserve_worker._uniserve_ipc import CUDAEvent, Submission
from uniserve_worker.errors import (
    invalid_descriptor,
)
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.image import reserve_images
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
from uniserve_worker.transport.exports import (
    forget_exports,
    release_exports,
    retiring_exports,
)
from uniserve_worker.transport.pool import ReadBackpressureError

if TYPE_CHECKING:
    from uniserve_worker.worker import Worker


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


class BatchRunner:
    """Prepare numerical work while Rust owns batch scheduling."""

    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.collective = any(
            group.size > 1
            for binding in worker.runner.bindings.values()
            if binding.owns
            for group in binding.communicators
        ) or (
            worker.sampling_group is not None and worker.sampling_group.size > 1
        )

    def admit(self, state: BatchState) -> None:
        # Revocation cannot wait behind an earlier allocation-dependent call.
        # Existing readers keep their leases until their actual last access.
        freed = tuple(
            command.buffer
            for command in state.batch.commands
            if isinstance(command, Free)
        )
        if freed:
            self.release_buffers(freed)

    def notify_ready(self, submission: Submission) -> None:
        submission.notify_ready()
        if self.worker._completion_wake is not None:
            self.worker._completion_wake()

    def await_inputs(self, state: BatchState, submission: Submission) -> None:
        if not state.awaiting_reads:
            state.on_dependencies_ready(partial(self.notify_ready, submission))

    def prepare_inputs(self, state: BatchState, submission: Submission) -> bool:
        self.advance_inputs(state, submission)
        return state.inputs_ready()

    def reap(self) -> None:
        self.worker.device_events.reap()
        for transport in self.worker.transports.values():
            transport.reap()

    def poll(self, batch: BatchState) -> tuple[bool, bool]:
        """Materialize and retire one launched batch."""
        before = batch.materialized

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
            return batch.materialized != before, self._advance_retirement(batch)
        return False, False

    def _execute_batch(self, state: BatchState) -> None:
        """Launch a prepared batch and release what its launch consumed.

        The native executor begins command retirement after this returns.
        """
        batch = state.batch
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
        # inputs are acquired; ``prepare`` revoked the unread ones.
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

    def prepare(self, state: BatchState) -> None:
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
        # Admitted slots reset together once the Starts are applied, so a
        # batch admitting many requests costs one set of device writes; a
        # failing Start still leaves every slot admitted before it reset.
        admitted: list[int] = []
        try:
            for start in starts:
                slots = self.worker.requests.apply_commands((start,))
                admitted.extend(slots)
                if slots and start.request.diffusion is not None:
                    # A video request's seeded noise is drawn while this batch
                    # and the ones before latent preparation run on the
                    # device.
                    begin_noise(
                        self.worker.runner,
                        self.worker.requests.get(
                            start.request.request_key.request_id
                        ),
                        self.worker.requests,
                    )
        finally:
            if admitted and self.worker.decode_state is not None:
                self.worker.decode_state.reset(admitted)
            if admitted and self.worker.canvas_slots is not None:
                self.worker.canvas_slots.reset(admitted)

        state.predecessors = self.worker.requests.predecessors(batch.calls)
        validate_batch(
            batch,
            worker_info=self.worker.info,
            model_runner=self.worker.runner,
            config=self.worker.worker_config,
            predecessors=state.predecessors,
        )

        # Slots other commands return reset together, once, after them.
        returned: dict[int, None] = {}
        try:
            for command in batch.commands:
                if isinstance(command, Start):
                    continue
                returned.update(
                    dict.fromkeys(
                        self.worker.requests.apply_commands((command,))
                    )
                )
        finally:
            if returned and self.worker.decode_state is not None:
                self.worker.decode_state.reset(tuple(returned))

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
        # Inline input images are prepared on the host lane while this thread
        # launches other batches; `inputs_ready` waits for them.
        reserve_images(
            state,
            host_tasks=self.worker.host_tasks,
            request_pool=self.worker.requests,
            model_runner=self.worker.runner,
        )

    def advance_inputs(self, state: BatchState, submission: Submission) -> None:
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
                partial(self.notify_ready, submission),
                after=error.returns,
            )
            return
        state.inputs_submitted = True
        capture_predicates(state, self.worker.tensor_store)

    def begin_retirement(self, state: BatchState) -> None:
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

    def _record_retirement_events(self) -> tuple[CUDAEvent, ...]:
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
        submits slot-reset writes; retirement finishes once they are issued,
        without waiting for them to run.

        The batch's result acknowledges its ``Finish`` and ``Free`` commands,
        and the engine reuses a released request row or cache unit only in a
        batch it builds after reading that result. The reset writes are
        issued on the device's current stream before the result is sent, and
        every later batch issues its device work on that stream, or on a
        batch stream that first waits for it (``prepare.reserve_outputs``),
        so each reuse runs after the resets. Each rank orders its own stream
        this way, so the same holds with several data- or expert-parallel
        ranks. Waiting for the resets to complete would instead hold the
        result behind every batch launched after this one, whose work
        precedes the resets on the stream.

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
        # The slot resets are stream-ordered before any reuse of the retired
        # rows (see the docstring), so no completion fence is recorded.
        self.retire_requests(
            tuple(state.retirement_local_requests), retained=retained
        )
        state.retirement_cleaned = True
        return True

    def close(self, state: BatchState) -> None:
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

    def execute(self, state: BatchState) -> None:
        """Launch a batch whose inputs are ready, then close its inputs.

        Raises ``RuntimeError`` when the inputs were already consumed, or
        when they are not ready, in which case they stay open. Otherwise
        ``BatchState.close_inputs`` runs after ``_execute_batch`` whether or
        not it raises.
        """
        if state.inputs_closed:
            raise RuntimeError("batch inputs have already been consumed")

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

    def _release_requests(
        self, request_ids: Sequence[int], retained: frozenset[BufferId]
    ) -> None:
        """Release drained requests' storage on this rank.

        Buffers in ``retained`` are kept; every other export and tensor-store
        product the requests own is released with their KV imports and cache
        state, decode state, canvas state, block tables, prefix slots, media
        mux state and latent slots. The slot resets of all the requests are
        issued together, so a burst of finishing requests costs one set of
        device writes rather than one per request.
        """
        ids = tuple(int(request_id) for request_id in request_ids)
        requests = tuple(
            request
            for request_id in ids
            if (request := self.worker.requests.peek(request_id)) is not None
        )
        keys = tuple(request.request_key for request in requests)
        slots = tuple(request.request_pool_idx for request in requests)

        if requests:
            if self.worker.kv_cache is not None:
                self.worker.kv_cache.imports.cancel_requests(
                    frozenset(keys), retained=retained
                )
            if self.worker.decode_state is not None:
                self.worker.decode_state.reset(slots)
            if self.worker.canvas_slots is not None:
                self.worker.canvas_slots.reset(slots)
            if self.worker.block_tables is not None:
                self.worker.block_tables.release(slots)

        if self.worker.kv_cache is not None:
            for request_id in ids:
                self.worker.kv_cache.drop(request_id)
        if self.worker.block_tables is not None:
            for key in keys:
                self.worker.block_tables.release_prefixes(key)

        owners = frozenset(ids)
        for store in (
            self.worker.tensor_store,
            self.worker.kv_cache,
            self.worker.latent_pool,
        ):
            if store is not None:
                selected = tuple(
                    buffer
                    for buffer in store.exports
                    if int(buffer.owner.request_id) in owners
                    and buffer not in retained
                )
                store.release_buffers(selected)

        if requests:
            self.worker.tensor_store.release_requests(keys, retained=retained)
        if self.worker.media_mux is not None:
            for request_id in ids:
                self.worker.media_mux.drop(request_id)
        if requests and self.worker.latent_pool is not None:
            self.worker.latent_pool.release_slots(slots)

    def drop_request(self, request_id: int) -> None:
        """Release a drained request and remove its admission from the pool."""
        request_id = int(request_id)
        self._release_requests((request_id,), frozenset())
        self.worker.requests.drop(request_id)

    def retire_requests(
        self,
        request_keys: Sequence[RequestKey],
        *,
        retained: frozenset[BufferId] = frozenset(),
    ) -> None:
        """Retire exactly the epochs ``request_keys`` name on this rank.

        Epochs that are not resident or are already retired are skipped.
        The caller must have waited for their readers to drain, as
        ``_advance_retirement`` does through the stores' ``retirement_ready``
        checks. Buffers in ``retained`` are kept.
        """
        live = []
        for key in request_keys:
            request = self.worker.requests.peek(key.request_id)
            if (
                request is not None
                and request.request_key == key
                and not request.retired
            ):
                live.append(key.request_id)
        if not live:
            return

        # A key named twice retires once.
        live = list(dict.fromkeys(live))
        self._release_requests(live, retained)
        for request_id in live:
            diffusion = self.worker.requests.get(request_id).diffusion
            if diffusion is not None:
                diffusion.close()
            self.worker.requests.retire(request_id)
