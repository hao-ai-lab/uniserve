"""Commit completed operations and retire the resources of failed logical groups."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from uniserve_worker.execution import operations as operation_geometry
from uniserve_worker.execution.batch import (
    Checkpoint,
    CompletionState,
    DeviceSelected,
    FixedCheckpoint,
    LaneResult,
    OpCode,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    RegistrationAck,
    StorageClass,
    TransferHandle,
)
from uniserve_worker.execution.output import (
    ImagePayload,
    LogprobPayload,
    OutputRecord,
    PendingOutput,
)
from uniserve_worker.execution.rows import DecodeRuntimePublication, LaneState, Outcome
from uniserve_worker.execution.sample import copy_runtime_scalar as _copy_runtime_scalar
from uniserve_worker.execution.transfer import _release_locators
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.profiling import _forward_stats, record_component
from uniserve_worker.runtime.device_products import DeviceProductWrite
from uniserve_worker.runtime.request import RequestRuntime, SpeculativeCommit

if TYPE_CHECKING:
    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.video import VideoMuxCoordinator
    from uniserve_worker.runtime.cache_pool import CachePool
    from uniserve_worker.runtime.cache_publications import CachePublications
    from uniserve_worker.runtime.device_products import DeviceProducts
    from uniserve_worker.runtime.encoder_cache import EncoderCache
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.request import RequestPool
    from uniserve_worker.runtime.runtime_states import RuntimeStates
    from uniserve_worker.transfer.publications import TransferPublications
    from uniserve_worker.transfer.tickets import Transport


logger = logging.getLogger(__name__)

SAMPLING_COMPLETION_FIELDS = 4
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1
_GENERATION_WORK_VARIANTS = frozenset(
    {
        OpCode.DIFFUSION_PREPARE,
        OpCode.DIFFUSION_STEP,
        OpCode.DIFFUSION_DECODE,
        OpCode.MEDIA_APPEND,
        OpCode.DIFFUSION_FINALIZE,
    }
)


def _commit_lane(
    run_id: int,
    scope: LaneState,
    outcomes: tuple[Outcome, ...],
    started: int,
    *,
    cache_registry: CachePublications | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    request_pool: RequestPool,
    runtime_states: RuntimeStates | None,
    transfer_publications: TransferPublications,
    config: WorkerConfig,
) -> LaneResult:
    """Atomically publish validated lane resources, request versions, and output records."""

    commit_started = time.perf_counter_ns()
    lane = scope.lane
    operations = lane.operations

    # All device reads must finish and every staged resource must validate before
    # completion storage becomes immutable or any publication becomes visible.
    _finish_device_reads(scope, device_products=device_products, encoder_cache=encoder_cache)
    _publish_predicates(scope, device_products=device_products)
    device_products.validate_writes(tuple(scope.device_writes))
    encoder_cache.validate_writes(tuple(scope.encoder_writes))
    if latent_pool is None:
        if scope.latent_publications or scope.latent_releases:
            raise RuntimeError("latent publication has no physical pool")
    else:
        latent_pool.validate_commit(
            scope.latent_publications,
            scope.latent_releases,
        )
    scope.completion.seal()
    # Build the complete next-version description without mutating resident state.
    records: list[PendingOutput] = []
    selected_versions: dict[int, Checkpoint] = {}
    pending_completions: dict[int, CompletionState] = {}
    speculative_commits: dict[int, SpeculativeCommit] = {}
    report_products: list[ProductPayload] = []
    resolved_runtime: dict[int, RequestRuntime] = {}
    layout = scope.layout
    if layout is None or layout.operations != operations:
        raise RuntimeError("lane commit lost its aligned candidate layout")
    for row, (operation, request, outcome) in enumerate(
        zip(
            operations,
            layout.requests,
            outcomes,
            strict=True,
        )
    ):
        _validate_completion_products(operation, outcome.products)
        report_products.extend(
            product
            for product in outcome.products
            if config.rank == worker_info.output_rank(operation.entry)
            or isinstance(product.payload, TransferHandle)
        )
        pending = PendingOutput(
            (
                request.request.pending_operations.get(int(operation.parent.op_id))
                if operation.parent is not None
                and isinstance(operation.parent.point, DeviceSelected)
                else None
            ),
            scope.completion,
            row,
            partial(_finalize_predicated_runtime, operation, request_pool=request_pool),
            status=outcome.status,
            selected_point=outcome.selected_point,
            completion_tasks=(
                *outcome.completion_tasks,
                *(
                    cast(LogprobPayload, product.payload)
                    for product in outcome.products
                    if isinstance(product.payload, LogprobPayload)
                ),
            ),
        )
        record = pending.bind_record(
            OutputRecord(
                request_key=operation.request_key,
                op_id=operation.op_id,
                kind=operation.kind,
                completion_slot_generation=scope.completion.generation,
                status=outcome.status,
                selected_point=outcome.selected_point,
                logical_lengths=outcome.logical_lengths,
                token_span=outcome.token_span,
                committed_tokens=outcome.committed_tokens,
                sampling=outcome.sampling,
                finish_flags=outcome.finish_flags,
                product_generations=outcome.product_generations,
                error_code=None,
                next_cursor=outcome.next_cursor,
                done=outcome.done,
            )
        )
        records.append(record)
        if operation.advances_state:
            pending_completions[operation.request_key.request_id] = pending
            if outcome.status is OpStatus.PREDICATED:
                selected = request.request.resolve_version(operation.parent)
                if selected is None:
                    raise RuntimeError("predicated operation lost its selected parent")
                selected_versions[operation.request_key.request_id] = selected
            else:
                selected_versions[operation.request_key.request_id] = Checkpoint(
                    op_id=operation.op_id,
                    point=FixedCheckpoint(outcome.selected_point),
                )
        elif operation.parent is not None:
            selected = request.request.resolve_version(operation.parent)
            if selected is None:
                raise RuntimeError("non-state operation lost its resolved parent")
            selected_versions[operation.request_key.request_id] = selected
        resolved_runtime[operation.request_key.request_id] = RequestRuntime(
            logical_position=request.logical_position,
            rng_counter=request.rng_counter,
            latent_product=request.latent_product,
            flow_step=request.flow_step,
            kv_visible_len=outcome.logical_lengths.kv_visible_len,
            kv_computed_len=outcome.logical_lengths.kv_computed_len,
        )
        selection = outcome.selection
        if selection is not None:
            speculative_commits[operation.request_key.request_id] = SpeculativeCommit(
                draft_tokens=selection.draft_tokens,
                terminal_prefix=selection.terminal_prefix,
                base_logical_position=selection.base_logical_position,
                base_rng_counter=selection.base_rng_counter,
                base_kv_visible=selection.base_kv_visible,
                initialized_kv=selection.initialized_kv,
            )
    record_component(scope, "commit_lane", commit_started)

    # Prepare cross-resource commit records first so no publication is visible
    # until every participating owner has accepted its state transition.
    lane_report = LaneResult(
        lane_id=lane.lane_id,
        completions=tuple(records),
        products=tuple(report_products),
        registration=RegistrationAck(visible=True),
        worker_exec_us=(time.perf_counter_ns() - scope.started_ns) // 1000,
        forward_stats=_forward_stats(scope.observations, scope.component_us),
    )
    cache_publications = cache_registry
    if cache_publications is None:
        if scope.cache_publications or scope.cache_installations:
            raise RuntimeError("cache publication has no backing KV resources")
        cache_commit = None
    else:
        cache_commit = cache_publications.prepare_commit(
            scope.cache_publications,
            scope.cache_installations,
        )
    request_publication = request_pool.prepare_publication(
        run_id=run_id,
        operations=operations,
        candidates=scope.request_candidates,
        selected_versions=selected_versions,
        runtimes=resolved_runtime,
        completions=pending_completions,
        speculative=speculative_commits,
    )
    transfer_publications.validate(scope.stage_publications)
    # From this point the lane cannot be discarded: apply resource commits, then
    # reserve the request publication that gates successor readiness.
    scope.publication_started = True
    device_products.commit_writes(tuple(scope.device_writes))
    encoder_cache.commit_writes(tuple(scope.encoder_writes))
    if latent_pool is not None:
        latent_pool.apply_commit(
            scope.latent_publications,
            scope.latent_releases,
        )
    if cache_publications is not None:
        assert cache_commit is not None
        cache_publications.apply_commit(cache_commit)
    transfer_publications.commit(scope.stage_publications)
    _commit_runtime_states(scope, runtime_states=runtime_states)
    request_publication.reserve()
    return replace(lane_report, publication=request_publication)


def _commit_runtime_states(scope: LaneState, *, runtime_states: RuntimeStates | None) -> None:
    """Publish committed token, predicate, position, cache-length, and penalty state to device rows."""

    states = runtime_states
    if states is None:
        if (
            scope.runtime_publications
            or scope.prompt_logits_publications
            or scope.runtime_cache_lengths
        ):
            raise RuntimeError("runtime state publication has no backing storage")
        return

    # Cache lengths may advance without token publication, so apply their
    # scalar updates before the row-level decode state transitions.
    for slot, length in scope.runtime_cache_lengths.items():
        _copy_runtime_scalar(states.valid_cache_lengths[slot : slot + 1], length)
    for publication in scope.runtime_publications:
        if isinstance(publication, DecodeRuntimePublication):
            # Batched decode uses the fused device-state kernel, then updates
            # request-owned penalty counts only for valid active selections.
            states.publish_decode(
                publication.slots,
                device_indices=publication.device_slots,
                tokens=publication.tokens,
                predicates=publication.predicates,
                selected_points=publication.selected_points,
            )
            for index, penalty_base in enumerate(publication.penalty_bases):
                if penalty_base is None:
                    continue
                weight = (
                    publication.valid[index : index + 1] & publication.active[index : index + 1]
                ).to(dtype=penalty_base.dtype)
                penalty_base.scatter_add_(
                    0,
                    publication.tokens[index : index + 1].to(dtype=torch.int64),
                    weight,
                )
            continue

        # Non-batched publications update the same fields explicitly while
        # stripping the continuation tag from future input tokens.
        slot = publication.slot
        future_token = states.future_input_tokens[slot, :1]
        future_token.copy_(publication.token.reshape(-1)[:1])
        future_token.bitwise_and_(TOKEN_VALUE_MASK)
        states.predicates[slot : slot + 1].copy_(
            publication.predicate.reshape(-1)[:1].to(dtype=torch.bool)
        )
        states.selected_points[slot : slot + 1].copy_(
            publication.selected_point.reshape(-1)[:1].to(dtype=torch.int32)
        )
        _copy_runtime_scalar(
            states.logical_lengths[slot : slot + 1],
            publication.logical_position,
        )
        _copy_runtime_scalar(
            states.sampling_positions[slot : slot + 1],
            publication.sampling_position,
        )
        penalty_base = publication.penalty_base
        if penalty_base is not None:
            weight = (publication.valid.reshape(-1)[:1] & publication.active.reshape(-1)[:1]).to(
                dtype=penalty_base.dtype
            )
            penalty_base.scatter_add_(
                0,
                future_token.to(dtype=torch.int64),
                weight,
            )

    # Prompt logits have request-row lifetime and become visible only after all
    # scalar transition fields for the lane are committed.
    for prompt_publication in scope.prompt_logits_publications:
        states.prompt_logits[prompt_publication.slot].copy_(
            prompt_publication.logits.to(dtype=states.prompt_logits.dtype)
        )


def _discard_lane(
    scope: LaneState,
    error: BaseException | None = None,
    *,
    cache_pool: CachePool | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    latent_pool: LatentPool | None,
    media_mux: VideoMuxCoordinator | None,
    transfer_backends: Mapping[str, Transport],
) -> None:
    """Release all provisional lane resources that have not crossed publication visibility."""

    _finish_device_reads(scope, device_products=device_products, encoder_cache=encoder_cache)
    for job in scope.completion_jobs:
        job.abandon()
    for reservation in scope.cpu_tasks.values():
        reservation.abandon()
    for lease in scope.media_output_leases.values():
        lease.defer_until_ready(scope.completion.completion_future())
    if scope.publication_started:
        raise RuntimeError("published lane state cannot be discarded")
    if media_mux is not None:
        for operation in scope.lane.operations:
            if operation.kind is OpCode.DIFFUSION_PREPARE:
                media_mux.drop(int(operation.request_key.request_id))
    scope.completion.abandon()
    device_products.abandon_writes(tuple(scope.device_writes))
    encoder_cache.abandon_writes(tuple(scope.encoder_writes))
    if cache_pool is not None:
        cache_pool.release_buffers(
            product.buffer_id
            for operation in scope.lane.operations
            for product in operation.outputs
        )
    if latent_pool is not None and scope.latent_import_slots:
        latent_pool.release_slots(tuple(scope.latent_import_slots))
    if latent_pool is not None:
        latent_pool.release_buffers(
            tuple(
                product.buffer_id
                for operation in scope.lane.operations
                for product in operation.outputs
            )
        )
    _release_locators(scope.published, transfer_backends=transfer_backends)


def _finalize_predicated_runtime(
    operation: Operation, *, request_pool: RequestPool
) -> tuple[Checkpoint | None, RequestRuntime]:
    """Resolve state-dependent skips; independent computation has no selected state."""

    if operation.parent is None:
        return None, RequestRuntime()
    selected, resolved = request_pool.resolve_predicated(
        operation.request_key.request_id,
        operation.op_id,
        operation.parent,
    )
    return selected, resolved


def _validate_completion_products(
    operation: Operation,
    products: tuple[ProductPayload, ...],
) -> None:
    """Validate completion payloads against every product declared by the operation."""

    declared = {output: output for output in operation.outputs}
    for product in products:
        reference = declared.get(product.product)
        if reference is None:
            raise invalid_descriptor("completion carries a product not declared by its operation")
        payload_bound = (
            product.payload.encoded_size_bound()
            if isinstance(product.payload, TransferHandle)
            else product.payload.max_encoded_bytes()
            if isinstance(
                product.payload,
                (
                    ImagePayload,
                    LogprobPayload,
                ),
            )
            else len(product.payload)
        )
        transferred = isinstance(product.payload, TransferHandle)
        if transferred and reference.storage_class in {
            StorageClass.HOST_STAGING,
            StorageClass.PINNED_OUTPUT,
        }:
            raise invalid_descriptor("host-visible output cannot carry a transfer entry")
        if not transferred and payload_bound > int(reference.max_bytes):
            raise invalid_descriptor("completion product exceeds its registered product byte bound")
        if reference.storage_class in (
            StorageClass.HOST_STAGING,
            StorageClass.PINNED_OUTPUT,
        ) and payload_bound > int(operation.bounds.max_completion_bytes):
            raise invalid_descriptor("completion product exceeds its registered byte bound")


def _publish_predicates(scope: LaneState, *, device_products: DeviceProducts) -> None:
    """Publish predicate outputs after their producing operations have resolved."""

    producers = {
        operation_geometry.operation_identity(operation) for operation in scope.lane.operations
    }
    transitions = {id(write) for write in scope.transition_writes.values()}
    propagated = {
        id(write) for writes in scope.propagated_predicate_writes.values() for write in writes
    }
    writes = tuple(
        write
        for write in scope.device_writes
        if write.reference.kind is ProductKind.COMPLETION
        and not write.producer_recorded
        and operation_geometry.product_identity(write.reference) in producers
        and id(write) not in transitions
        and id(write) not in propagated
    )
    if not writes:
        return
    batch = device_products.producer_scalar_batch(writes)
    if batch is not None:
        batch.tensor.fill_(1)
        device_products.publish_scalar_batch(batch)
        return
    views = device_products.producer_write_views(writes)
    first = views[0]
    device_products.publish_writes(
        writes,
        torch.ones(
            (len(writes),),
            dtype=first.dtype,
            device=first.device,
        ),
    )


def _finish_device_reads(
    scope: LaneState, *, device_products: DeviceProducts, encoder_cache: EncoderCache
) -> None:
    """Record or cancel every device-product read acquired by the lane."""

    reads = tuple(scope.device_reads)
    if reads:
        after_writes: list[DeviceProductWrite] = []
        for read in reads:
            write = scope.operation_writes.get(
                (read.reference.request_key, int(read.consumer_op_id))
            )
            if write is None:
                after_writes.clear()
                break
            after_writes.append(write)
        device_products.record_readers(
            reads,
            after_writes=tuple(after_writes),
        )
    scope.device_reads.clear()
    encoder_reads = tuple(scope.encoder_reads)
    if encoder_reads:
        encoder_cache.record_readers(encoder_reads)
    scope.encoder_reads.clear()
