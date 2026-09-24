"""Commit or discard one executed batch at its publication boundary.

``uniserve_worker.execution.step.execute_batch`` calls ``commit_batch`` with
the outcomes ``schedule.dispatch_batch`` returns, and ``discard_batch`` when
dispatch or the validation half of the commit fails;
``prepare.reserve_outputs`` also discards the batch when its reservation
fails after the batch's pending outputs are bound.

``commit_batch`` runs in two halves. The first fences device reads,
publishes resolved completion predicates, validates every staged tensor
write, latent update, KV publication and export against its owning store,
and checks each completion's products and size bounds; nothing it does
makes a product, latent generation, KV publication or request progress
visible. The second sets ``BatchState.published`` and applies those
changes to the ``TensorStore``, ``LatentPool``, ``KVCacheManager``,
``DecodeState`` and ``RequestPool``. Setting ``published`` is the point of
no return: ``discard_batch`` refuses a published batch, and
``execute_batch`` classifies any later failure as fatal.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import torch

from uniserve.tensors import concatenate_views
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.output import PendingOutput, logprob_entries
from uniserve_worker.execution.transfer import _release_locators
from uniserve_worker.profiling import _forward_stats, record_component
from uniserve_worker.protocol.batch import TensorPublication
from uniserve_worker.protocol.call import Call, CallStatus, MediaCall
from uniserve_worker.sampling.result import sample_columns
from uniserve_worker.transport.exports import validate_exports

if TYPE_CHECKING:
    from uniserve_worker.config.execution.execution import WorkerConfig
    from uniserve_worker.execution.request import RequestPool
    from uniserve_worker.media.mux import MediaMux
    from uniserve_worker.protocol.worker_info import WorkerInfo
    from uniserve_worker.storage.decode_state import DecodeState
    from uniserve_worker.storage.kv_cache import KVCacheManager
    from uniserve_worker.storage.latent_pool import LatentPool
    from uniserve_worker.storage.tensor_store import TensorStore
    from uniserve_worker.transport.interface import Transport


logger = logging.getLogger(__name__)


def commit_batch(
    batch_id: int,
    outcomes: tuple[PendingOutput, ...],
    started: int,
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    request_pool: RequestPool,
    decode_state: DecodeState | None,
    config: WorkerConfig,
) -> None:
    """Publish validated resources, execution progress and batch outputs.

    ``outcomes`` must be the batch's own bound ``PendingOutput`` records, in
    call order. Every validation failure raises before ``state.published``
    is set, leaving the batch discardable. On success each output is
    recorded through ``BatchState.record_outputs`` and its calls are
    registered as pending in the ``RequestPool``.
    """
    with state.scope():
        commit_started = time.perf_counter_ns()
        calls = state.batch.calls

        # All device reads must finish and every staged resource must validate
        # before completion storage becomes immutable or any publication becomes
        # visible. Predicates publish first because ``validate_writes`` rejects
        # a write that is neither producer-recorded nor deferred.
        _finish_device_reads(tensor_store=tensor_store, state=state)
        _publish_predicates(tensor_store=tensor_store, state=state)
        writes = tuple(
            write
            for request in state.pending_outputs()
            for write in request.writes
        )
        tensor_store.validate_writes(writes)
        if latent_pool is None:
            if any(
                output.latent.update.params is not None for output in outcomes
            ):
                raise RuntimeError("latent publication has no physical pool")
        else:
            latent_pool.validate_updates(
                tuple(output.latent.update for output in outcomes)
            )
        state.output_buffer.seal()

        # Prepare the execution result without mutating resident state.
        records: list[PendingOutput] = []
        report_products: list[TensorPublication] = []
        for row, (call, request, outcome) in enumerate(
            zip(
                calls,
                state.pending_outputs(),
                outcomes,
                strict=True,
            )
        ):
            _validate_completion_products(call, outcome.products)

            if outcome.kv_output is not None:
                if outcome.kv_output.source != call.kv_output:
                    raise invalid_descriptor(
                        "KV publication differs from its declared output"
                    )
                if (
                    sum(tensor.nbytes for tensor in outcome.kv_output.tensors)
                    > call.bounds.max_transfer_bytes
                ):
                    raise invalid_descriptor(
                        "KV publication exceeds its transfer-byte bound"
                    )
                # Called only for its check, which raises when the descriptor
                # exceeds ``MAX_TRANSFER_HANDLE_BYTES`` or names an unknown
                # transport.
                outcome.kv_output.encoded_size_bound()

            # The bound covers score values and prompt-position counts; framing
            # is owned by the single IPC result message, not by stored products.
            # Each entry is one 12-byte ``TokenLogprob`` of the IPC schema and
            # each reported row adds 4 bytes, the same costs the engine's
            # ``logprob_result_bytes`` uses to size ``max_completion_bytes``.
            logprob_bytes = (
                0
                if outcome.token.logprob_range is None
                else 4
                + 12 * logprob_entries(outcome, outcome.token.logprob_range)
            ) + sum(
                4 + 12 * logprob_entries(outcome, span)
                for span in outcome.token.prompt_logprob_ranges
            )
            if logprob_bytes > call.bounds.max_completion_bytes:
                raise invalid_descriptor(
                    "logprob result exceeds its registered completion capacity"
                )

            reports_output = config.rank == worker_info.output_rank(
                call.component
            )
            report_products.extend(outcome.products)
            pending = request
            # Dispatch must return the records bound to this batch, not copies.
            if outcome is not pending:
                raise RuntimeError("call completion lost its prepared output")
            pending._reports_output = reports_output

            # All ranks retain their score ranges until the output buffer
            # retires; materialization emits scores only on the designated
            # output rank.
            records.append(pending)

        record_component(
            state.component_us,
            "commit_lane",
            commit_started,
        )

        # Prepare cross-resource commit records first so no publication is
        # visible until every participating owner has accepted its state
        # transition.
        execution_us = (time.perf_counter_ns() - state.started_ns) // 1000
        stats = _forward_stats(
            state.forward_stats,
            state.component_us,
        )

        cache_publications = kv_cache
        publications = tuple(
            request.cache_publication
            for request in state.pending_outputs()
            if request.cache_publication is not None
        )
        installations = tuple(
            request.cache_installation
            for request in state.pending_outputs()
            if request.cache_installation is not None
        )
        if cache_publications is None:
            if publications or installations:
                raise RuntimeError(
                    "cache publication has no backing KV resources"
                )
        else:
            cache_publications.validate_publications(
                publications, installations
            )

        tensor_exports = {
            buffer: locations
            for request in state.pending_outputs()
            for buffer, locations in request.tensor_exports.items()
        }
        cache_exports = {
            buffer: locations
            for request in state.pending_outputs()
            for buffer, locations in request.cache_exports.items()
        }
        latent_exports = {
            buffer: locations
            for request in state.pending_outputs()
            for buffer, locations in request.latent.exports.items()
        }

        request_pool.validate_pending(tuple(record.call for record in records))
        for owner, exports in (
            (tensor_store, tensor_exports),
            (kv_cache, cache_exports),
            (latent_pool, latent_exports),
        ):
            if owner is None:
                if exports:
                    raise RuntimeError(
                        "transport export has no backing storage"
                    )
            else:
                validate_exports(owner.exports, exports)

        # From this point the batch cannot be discarded: apply
        # resource commits, then reserve the request publication that gates
        # successor readiness. ``step.execute_batch`` classifies a failure
        # below as fatal.
        state.published = True
        tensor_store.commit_writes(writes)

        if latent_pool is not None:
            latent_pool.apply_updates(
                tuple(output.latent.update for output in outcomes)
            )
        if cache_publications is not None:
            cache_publications.apply_publications(publications, installations)

        tensor_store.exports.update(tensor_exports)
        if kv_cache is not None:
            kv_cache.exports.update(cache_exports)
        if latent_pool is not None:
            latent_pool.exports.update(latent_exports)

        _commit_runtime_states(decode_state=decode_state, state=state)

        for request in state.pending_outputs():
            request.release_execution_references()

        request_pool.add_pending(tuple(record.call for record in records))
        state.record_outputs(
            tuple(records),
            products=tuple(report_products),
            execution_us=execution_us,
            stats=stats,
        )


def _commit_runtime_states(
    *,
    state: BatchState,
    decode_state: DecodeState | None,
) -> None:
    """Apply the batch's token-state updates to ``DecodeState``.

    The updates are the ``runtime_*`` fields and sampled rows that execution
    left on each pending output. Without a ``DecodeState``, a sampled row,
    runtime prompt logits or a runtime cache length raises ``RuntimeError``.
    """
    requests = state.pending_outputs()
    states = decode_state
    if states is None:
        if any(
            request.token.sampled is not None
            or request.token.runtime_prompt_logits is not None
            or request.token.runtime_cache_length is not None
            for request in requests
        ):
            raise RuntimeError(
                "runtime state publication has no backing storage"
            )
        return

    # Install lengths before advancing tokens. Decode rows share one update;
    # prefill/verification retain their explicit logical and RNG coordinates.
    for request in requests:
        if request.token.runtime_cache_length is not None:
            states.set_cache_length(
                int(request.request.request_pool_idx),
                request.token.runtime_cache_length,
            )

    decode = tuple(
        request
        for request in requests
        if request.token.sampled is not None
        and request.token.runtime_decode_increment
    )
    if decode:
        samples = tuple(
            request.token.sampled
            for request in decode
            if request.token.sampled is not None
        )
        if any(sample.request_pool_index is None for sample in samples):
            raise RuntimeError("decode samples have no device request slots")
        tokens, continuation, valid, active = sample_columns(
            samples, ("tokens", "continuation", "valid", "active")
        )
        states.apply_tokens(
            tuple(int(request.request.request_pool_idx) for request in decode),
            device_slots=concatenate_views(
                tuple(
                    cast(torch.Tensor, sample.request_pool_index)
                    for sample in samples
                )
            ),
            tokens=tokens,
            predicates=continuation,
            penalty_bases=tuple(
                request.token.runtime_penalty_base for request in decode
            ),
            valid=valid,
            active=active,
        )

    for request in requests:
        sampled = request.token.sampled
        if sampled is not None and not request.token.runtime_decode_increment:
            states.apply_tokens(
                (int(request.request.request_pool_idx),),
                tokens=sampled.tokens,
                predicates=sampled.continuation,
                logical_position=request.token.runtime_logical_position,
                sampling_position=request.token.runtime_sampling_position,
                penalty_bases=(request.token.runtime_penalty_base,),
                valid=sampled.valid,
                active=sampled.active,
            )

    for request in requests:
        if request.token.runtime_prompt_logits is not None:
            states.set_prompt_logits(
                int(request.request.request_pool_idx),
                request.token.runtime_prompt_logits,
            )


def discard_batch(
    error: BaseException | None = None,
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    transfer_backends: Mapping[str, Transport],
) -> None:
    """Release every provisional resource of an unpublished batch.

    Fences outstanding device reads, abandons the output buffer and pending
    outputs, drops the media mux state of latent-preparation calls, returns
    unpublished tensor writes, KV output buffers, imported latent slots and
    latent product buffers, and releases exported transport locators. Raises
    ``RuntimeError`` when ``state.published`` is already set.
    """
    with state.scope():
        if state.published:
            raise RuntimeError("published batch state cannot be discarded")
        _finish_device_reads(tensor_store=tensor_store, state=state)

        # Cancellation uses the same producer fence as successful CPU work.
        state.output_buffer.seal()

        for pending in state.pending_outputs():
            pending.abandon()

        if media_mux is not None:
            for call in state.batch.calls:
                if call.kind is MediaCall.LATENT_PREPARATION:
                    media_mux.drop(int(call.request_key.request_id))

        state.output_buffer.abandon()

        tensor_store.abandon_writes(
            tuple(
                write
                for request in state.pending_outputs()
                for write in request.writes
            )
        )

        if kv_cache is not None:
            kv_cache.release_buffers(
                call.kv_output
                for call in state.batch.calls
                if call.kv_output is not None
            )

        imported_slots = tuple(
            int(request.request.request_pool_idx)
            for request in state.pending_outputs()
            if request.latent.imported
        )
        if latent_pool is not None and imported_slots:
            latent_pool.release_slots(imported_slots)

        if latent_pool is not None:
            latent_pool.release_buffers(
                tuple(
                    product.buffer_id
                    for call in state.batch.calls
                    for product in call.tensor_outputs()
                )
            )

        _release_locators(
            tuple(
                locator
                for request in state.pending_outputs()
                for locator in request.exported_locators
            ),
            transfer_backends=transfer_backends,
        )

        for request in state.pending_outputs():
            request.release_execution_references()


def _validate_completion_products(
    call: Call,
    products: tuple[TensorPublication, ...],
) -> None:
    """Require each completion product to be declared by its call.

    Raises ``invalid_descriptor`` for an undeclared product, and for a
    product whose locators exceed ``MAX_TRANSFER_HANDLE_BYTES`` or name an
    unknown transport.
    """
    declared = {output: output for output in call.tensor_outputs()}
    for product in products:
        reference = declared.get(product.product)
        if reference is None:
            raise invalid_descriptor(
                "completion carries a product not declared by its call"
            )
        product.encoded_size_bound()


def _publish_predicates(
    *, state: BatchState, tensor_store: TensorStore
) -> None:
    """Publish true into the completion outputs of calls that ran.

    Predicated calls already published false through
    ``prepare._publish_predicated_outputs``, and a call whose execution wrote
    its own completion output is skipped.
    """
    writes = tuple(
        request.completion_write
        for request in state.pending_outputs()
        if request.status is not CallStatus.PREDICATED
        and request.completion_write is not None
        and not request.completion_write.producer_recorded
    )
    if not writes:
        return

    views = tensor_store.producer_write_views(writes)
    first = views[0]

    # Resolved predicates publish as 1, one scalar per producer view.
    tensor_store.publish_writes(
        writes,
        torch.ones(
            (len(writes),),
            dtype=first.dtype,
            device=first.device,
        ),
    )


def _finish_device_reads(
    *, state: BatchState, tensor_store: TensorStore
) -> None:
    """End the batch's device read leases behind their consumers' fences.

    Clears ``device_reads`` and ``feature_reads`` on every pending output.
    """
    reads = tuple(
        read
        for request in state.pending_outputs()
        for read in request.device_reads
    )
    if reads:
        # A source may belong to another request. Its reader's completion is
        # ordered by the consuming call's output, never by source identity.
        # The writes are passed only when there is one per read, the
        # precondition of the pairwise fence path of
        # ``TensorStore.complete_reads``; with none, every CUDA read is fenced
        # by an event recorded on its device's current stream.
        after_writes = tuple(
            request.producer_write
            for request in state.pending_outputs()
            for _read in request.device_reads
            if request.producer_write is not None
        )
        tensor_store.complete_reads(
            reads,
            after_writes=after_writes
            if len(after_writes) == len(reads)
            else (),
        )

    feature_reads = tuple(
        read
        for request in state.pending_outputs()
        for read in request.feature_reads
    )
    if feature_reads:
        tensor_store.complete_reads(feature_reads)

    for request in state.pending_outputs():
        request.device_reads.clear()
        request.feature_reads.clear()
