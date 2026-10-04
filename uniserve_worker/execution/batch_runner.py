"""Numerical preparation and output decoding for the native executor."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

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
from uniserve_worker.transport.exports import retiring_exports

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

    def prepare(self, state: BatchState) -> bool:
        """Apply the batch's commands, validate it and prepare its inputs.

        Returns whether the prepared batch imports tensor or KV products.
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
        # launches other batches; `BatchInputs.ready` waits for them.
        reserve_images(
            state,
            host_tasks=self.worker.host_tasks,
            request_pool=self.worker.requests,
            model_runner=self.worker.runner,
        )
        return bool(state.input_products or state.kv_inputs)

    def prepare_inputs(self, state: BatchState) -> None:
        """Bind tensor views and submit copies into reserved destinations."""
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

    def capture_predicates(self, state: BatchState) -> None:
        capture_predicates(state, self.worker.tensor_store)

    def close(self, state: BatchState) -> None:
        """Release an owned batch whose result is consumed or abandoned.

        Calls whose outputs were never materialized are cancelled in the
        ``RequestPool``, which closes their requests. ``BatchState.close``
        abandons inputs and outputs while physical readers keep their leases.
        """
        pending = tuple(
            output
            for output in state.outputs
            if isinstance(output, PendingOutput)
        )
        self.worker.requests.cancel_calls(
            tuple(output.call for output in pending)
        )
        state.close(
            self.worker.tensor_store,
            self.worker.latent_pool,
            self.worker.kv_cache,
        )

    def execute(self, state: BatchState) -> None:
        """Launch a batch whose inputs are ready, then close its inputs.

        Raises ``RuntimeError`` when the inputs were already consumed, or
        when they are not ready, in which case they stay open. Otherwise
        ``BatchInputs.close`` runs after ``_execute_batch`` whether or
        not it raises.
        """
        if state.inputs.closed:
            raise RuntimeError("batch inputs have already been consumed")

        if not state.inputs.ready():
            raise RuntimeError("batch was observed before dependency readiness")

        try:
            state.inputs.require_storage()
            self._execute_batch(state)
        except BaseException as error:
            try:
                state.inputs.close(
                    self.worker.tensor_store,
                    self.worker.latent_pool,
                    None
                    if self.worker.kv_cache is None
                    else self.worker.kv_cache.imports,
                )
            except BaseException as cleanup_error:
                error.add_note(f"batch input cleanup failed: {cleanup_error!r}")
            raise
        state.inputs.close(
            self.worker.tensor_store,
            self.worker.latent_pool,
            None
            if self.worker.kv_cache is None
            else self.worker.kv_cache.imports,
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
