"""Commit completed operations and retire the resources of failed logical.

groups.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import torch

from uniserve.tensors import concatenate_views
from uniserve_worker.execution.output import (
    PendingOutput,
)
from uniserve_worker.execution.transfer import _release_locators
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.profiling import _forward_stats, record_component
from uniserve_worker.protocol.batch import TensorPublication
from uniserve_worker.protocol.operation import (
    OpStatus,
    PipelineStage,
    ScheduledRequest,
)
from uniserve_worker.transfer.exports import validate_exports

from .batch_state import BatchState
from .output import logprob_entries
from .sampling import sample_columns

if TYPE_CHECKING:
    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.media.mux import MediaMux
    from uniserve_worker.runtime.cache_manager import CacheManager
    from uniserve_worker.runtime.decode_state import DecodeState
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.request import RequestPool
    from uniserve_worker.runtime.tensor_store import TensorStore
    from uniserve_worker.transfer.tickets import Transport


logger = logging.getLogger(__name__)


def _commit_group(
    batch_id: int,
    completion_group: int,
    outcomes: tuple[PendingOutput, ...],
    started: int,
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    request_pool: RequestPool,
    decode_state: DecodeState | None,
    config: WorkerConfig,
) -> None:
    """Atomically publish validated completion group resources.

    execution progress, and output records.
    """
    with state.group_scope(completion_group):
        commit_started = time.perf_counter_ns()
        operations = state.group_operations(completion_group)

        # All device reads must finish and every staged resource must validate
        # before completion storage becomes immutable or any publication becomes
        # visible.
        _finish_device_reads(
            completion_group, tensor_store=tensor_store, state=state
        )
        _publish_predicates(
            completion_group, tensor_store=tensor_store, state=state
        )
        writes = tuple(
            write
            for request in state.pending_outputs(completion_group)
            for write in request.writes
        )
        tensor_store.validate_writes(writes)
        if latent_pool is None:
            if any(output.latent_params is not None for output in outcomes):
                raise RuntimeError("latent publication has no physical pool")
        else:
            latent_pool.validate_updates(outcomes)
        state.group_buffers[completion_group].seal()

        # Prepare the execution result without mutating resident state.
        records: list[PendingOutput] = []
        report_products: list[TensorPublication] = []
        for row, (operation, request, outcome) in enumerate(
            zip(
                operations,
                state.pending_outputs(completion_group),
                outcomes,
                strict=True,
            )
        ):
            _validate_completion_products(operation, outcome.products)

            if outcome.kv_output is not None:
                if outcome.kv_output.source != operation.kv_output:
                    raise invalid_descriptor(
                        "KV publication differs from its declared output"
                    )
                if (
                    sum(tensor.nbytes for tensor in outcome.kv_output.tensors)
                    > operation.bounds.max_transfer_bytes
                ):
                    raise invalid_descriptor(
                        "KV publication exceeds its transfer-byte bound"
                    )
                outcome.kv_output.encoded_size_bound()

            # The bound covers score values and prompt-position counts; framing
            # is owned by the single IPC result message, not by stored products.
            logprob_bytes = (
                0
                if outcome.logprob_range is None
                else 4 + 12 * logprob_entries(outcome, outcome.logprob_range)
            ) + sum(
                4 + 12 * logprob_entries(outcome, span)
                for span in outcome.prompt_logprob_ranges
            )
            if logprob_bytes > operation.bounds.max_completion_bytes:
                raise invalid_descriptor(
                    "logprob result exceeds its registered completion capacity"
                )

            reports_output = config.rank == worker_info.output_rank(
                operation.entry
            )
            report_products.extend(outcome.products)
            pending = request
            if outcome is not pending:
                raise RuntimeError(
                    "operation completion lost its prepared output"
                )
            pending._reports_output = reports_output

            # All ranks retain their score ranges until the output buffer
            # retires; materialization emits scores only on the designated
            # output rank.
            records.append(pending)

        record_component(
            state.group_component_us[completion_group],
            "commit_lane",
            commit_started,
        )

        # Prepare cross-resource commit records first so no publication is
        # visible until every participating owner has accepted its state
        # transition.
        execution_us = (
            time.perf_counter_ns() - state.group_started_ns[completion_group]
        ) // 1000
        stats = _forward_stats(
            state.group_forward_stats[completion_group],
            state.group_component_us[completion_group],
        )

        cache_publications = kv_cache
        publications = tuple(
            request.cache_publication
            for request in state.pending_outputs(completion_group)
            if request.cache_publication is not None
        )
        installations = tuple(
            request.cache_installation
            for request in state.pending_outputs(completion_group)
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
            for request in state.pending_outputs(completion_group)
            for buffer, locations in request.tensor_exports.items()
        }
        cache_exports = {
            buffer: locations
            for request in state.pending_outputs(completion_group)
            for buffer, locations in request.cache_exports.items()
        }
        latent_exports = {
            buffer: locations
            for request in state.pending_outputs(completion_group)
            for buffer, locations in request.latent_exports.items()
        }

        request_pool.validate_pending(records)
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

        # From this point the completion group cannot be discarded: apply
        # resource commits, then reserve the request publication that gates
        # successor readiness.
        state.group_published[completion_group] = True
        tensor_store.commit_writes(writes)

        if latent_pool is not None:
            latent_pool.apply_updates(outcomes)
        if cache_publications is not None:
            cache_publications.commit_publications(publications, installations)

        tensor_store.exports.update(tensor_exports)
        if kv_cache is not None:
            kv_cache.exports.update(cache_exports)
        if latent_pool is not None:
            latent_pool.exports.update(latent_exports)

        _commit_runtime_states(
            completion_group, decode_state=decode_state, state=state
        )

        for request in state.pending_outputs(completion_group):
            request.release_execution_references()

        request_pool.add_pending(records)
        state.record_outputs(
            completion_group,
            tuple(records),
            products=tuple(report_products),
            visible=True,
            execution_us=execution_us,
            stats=stats,
        )


def _commit_runtime_states(
    completion_group: int,
    *,
    state: BatchState,
    decode_state: DecodeState | None,
) -> None:
    """Commit numerical updates held by the same pending outputs as host.

    results.
    """
    requests = state.pending_outputs(completion_group)
    states = decode_state
    if states is None:
        if any(
            request.sampled is not None
            or request.runtime_prompt_logits is not None
            or request.runtime_cache_length is not None
            for request in requests
        ):
            raise RuntimeError(
                "runtime state publication has no backing storage"
            )
        return

    # Install lengths before advancing tokens. Decode rows share one update;
    # prefill/verification retain their explicit logical and RNG coordinates.
    for request in requests:
        if request.runtime_cache_length is not None:
            states.set_cache_length(
                int(request.request.request_pool_idx),
                request.runtime_cache_length,
            )

    decode = tuple(
        request
        for request in requests
        if request.sampled is not None and request.runtime_decode_increment
    )
    if decode:
        samples = tuple(
            request.sampled for request in decode if request.sampled is not None
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
                request.runtime_penalty_base for request in decode
            ),
            valid=valid,
            active=active,
        )

    for request in requests:
        sampled = request.sampled
        if sampled is not None and not request.runtime_decode_increment:
            states.apply_tokens(
                (int(request.request.request_pool_idx),),
                tokens=sampled.tokens,
                predicates=sampled.continuation,
                logical_position=request.runtime_logical_position,
                sampling_position=request.runtime_sampling_position,
                penalty_bases=(request.runtime_penalty_base,),
                valid=sampled.valid,
                active=sampled.active,
            )

    for request in requests:
        if request.runtime_prompt_logits is not None:
            states.set_prompt_logits(
                int(request.request.request_pool_idx),
                request.runtime_prompt_logits,
            )


def _discard_group(
    completion_group: int,
    error: BaseException | None = None,
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    transfer_backends: Mapping[str, Transport],
) -> None:
    """Release all provisional completion group resources that have not crossed.

    publication visibility.
    """
    with state.group_scope(completion_group):
        if state.group_published[completion_group]:
            raise RuntimeError(
                "published completion group state cannot be discarded"
            )
        _finish_device_reads(
            completion_group, tensor_store=tensor_store, state=state
        )

        # Cancellation uses the same producer fence as successful CPU work.
        state.group_buffers[completion_group].seal()

        for pending in state.pending_outputs(completion_group):
            pending.abandon()

        if media_mux is not None:
            for operation in state.group_operations(completion_group):
                if operation.kind is PipelineStage.LATENT_PREPARATION:
                    media_mux.drop(int(operation.request_key.request_id))

        state.group_buffers[completion_group].abandon()

        tensor_store.abandon_writes(
            tuple(
                write
                for request in state.pending_outputs(completion_group)
                for write in request.writes
            )
        )

        if kv_cache is not None:
            kv_cache.release_buffers(
                operation.kv_output
                for operation in state.group_operations(completion_group)
                if operation.kv_output is not None
            )

        imported_slots = tuple(
            int(request.request.request_pool_idx)
            for request in state.pending_outputs(completion_group)
            if request.latent_imported
        )
        if latent_pool is not None and imported_slots:
            latent_pool.release_slots(imported_slots)

        if latent_pool is not None:
            latent_pool.release_buffers(
                tuple(
                    product.buffer_id
                    for operation in state.group_operations(completion_group)
                    for product in operation.tensor_outputs()
                )
            )

        _release_locators(
            tuple(
                locator
                for request in state.pending_outputs(completion_group)
                for locator in request.exported_locators
            ),
            transfer_backends=transfer_backends,
        )

        for request in state.pending_outputs(completion_group):
            request.release_execution_references()


def _validate_completion_products(
    operation: ScheduledRequest,
    products: tuple[TensorPublication, ...],
) -> None:
    """Validate completion payloads against every product declared by the.

    operation.
    """
    declared = {output: output for output in operation.tensor_outputs()}
    for product in products:
        reference = declared.get(product.product)
        if reference is None:
            raise invalid_descriptor(
                "completion carries a product not declared by its operation"
            )
        product.encoded_size_bound()


def _publish_predicates(
    completion_group: int, *, state: BatchState, tensor_store: TensorStore
) -> None:
    """Publish predicate outputs after their producing operations have.

    resolved.
    """
    writes = tuple(
        request.completion_write
        for request in state.pending_outputs(completion_group)
        if request.status is not OpStatus.PREDICATED
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
    completion_group: int, *, state: BatchState, tensor_store: TensorStore
) -> None:
    """Complete actual consumer reads using that operation's producer fence."""
    reads = tuple(
        read
        for request in state.pending_outputs(completion_group)
        for read in request.device_reads
    )
    if reads:
        # A source may belong to another request. Its reader's completion is
        # ordered by the consuming operation's output, never by source identity.
        after_writes = tuple(
            request.producer_write
            for request in state.pending_outputs(completion_group)
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
        for request in state.pending_outputs(completion_group)
        for read in request.feature_reads
    )
    if feature_reads:
        tensor_store.complete_reads(feature_reads)

    for request in state.pending_outputs(completion_group):
        request.device_reads.clear()
        request.feature_reads.clear()
