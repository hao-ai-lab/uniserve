"""Prepare numerical input and output resources for admitted batches.

The native executor and its `BatchRunner` drive a batch through
these stages, in order:

1. The native executor applies request commands, resolves KV input descriptions
   and collects storage completions in `BatchInputs` before this batch writes.
2. The native executor reserves tensor, latent and KV imports once storage
   dependencies are done. `BatchInputs` retains accepted inputs through
   consumption or failure. `prepare_predicates` binds completion-valued
   predicates; later advances call `capture_predicates` as sources become ready.
3. `reserve_outputs` runs under the native executor once inputs
   are ready. It creates the batch's `PendingOutput` records and completion
   buffer, then reserves host tasks, cache tables, latent buffers, and device
   output writes, and publishes the transferred inputs into their stores.

The native executor retires partially prepared resources through their
owners when a stage fails.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import torch

from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.errors import (
    invalid_descriptor,
    unsupported_setup,
)
from uniserve_worker.execution import calls as calls
from uniserve_worker.execution.host_media import encoded_unit_positions
from uniserve_worker.profiling import record_component
from uniserve_worker.protocol.batch import Batch
from uniserve_worker.protocol.call import (
    Call,
    CallStatus,
    MediaCall,
    TransferMode,
)
from uniserve_worker.protocol.identity import BufferId, CallId, CallIdentity
from uniserve_worker.protocol.tensor import DType, ShapeBound, TensorRef
from uniserve_worker.sampling.result import SAMPLING_COMPLETION_FIELDS
from uniserve_worker.storage.output import OutputBuffer
from uniserve_worker.storage.tensor_store import TensorRead

if TYPE_CHECKING:
    from uniserve_worker.config.execution.execution import WorkerConfig
    from uniserve_worker.execution.host import HostLane, HostTask
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.execution.request import RequestPool
    from uniserve_worker.protocol.worker_info import WorkerInfo
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.kv_cache import KVCacheManager
    from uniserve_worker.storage.latent_pool import LatentPool
    from uniserve_worker.storage.output import OutputPool
    from uniserve_worker.storage.tensor_store import TensorStore


logger = logging.getLogger(__name__)


def prepare_predicates(
    state: BatchState,
    *,
    tensor_store: TensorStore,
    output_pool: OutputPool,
    model_runner: ModelExecutor,
) -> None:
    """Capture completion-valued (U8) predicates into one completion buffer.

    Each such call owns one row of the buffer. Local sources are
    consumed and captured now; transferred sources (those with a prepared
    batch-owned tensor import) are recorded in `state.predicate_transfers`
    for `capture_predicates`. The buffer is sealed once every row is
    captured, and `BatchState.predicate_values` reads it after the copies
    complete. I64 relay predicates are not captured here; `_consume_predicates`
    hands them to execution as device tensors. Nor are the predicates of
    calls gated on the device (`calls.device_gated`).
    """
    # Predicate rows occupy one compact completion buffer regardless of whether
    # their source is already local or will arrive through a prepared transfer.
    scheduled = tuple(
        call
        for call in state.batch.calls
        if call.predicate is not None
        and call.predicate.dtype is DType.U8
        and not calls.device_gated(call)
    )
    if not scheduled:
        return

    buffer = output_pool.acquire(
        len(scheduled),
        token_capacity=len(scheduled),
        devices=tuple(model_runner.call_devices(call)[0] for call in scheduled),
    )
    captures: list[tuple[CallIdentity, tuple[int, int], int]] = []
    pending: list[tuple[CallIdentity, BufferId, int]] = []
    recorded: list[TensorRead] = []
    try:
        # Local sources are consumed in device batches and captured directly;
        # transferred sources retain their target row for later completion.
        grouped: dict[torch.device, list[Call]] = defaultdict(list)
        rows = {
            calls.call_identity(call): row for row, call in enumerate(scheduled)
        }
        for call in scheduled:
            source = cast(TensorRef, call.predicate).buffer_id
            if state.inputs.tensor(source) is None:
                grouped[model_runner.call_devices(call)[0]].append(call)
            else:
                pending.append(
                    (
                        calls.call_identity(call),
                        source,
                        rows[calls.call_identity(call)],
                    )
                )

        for device, device_calls in grouped.items():
            reads = tensor_store.consume_batch(
                tuple(
                    (
                        cast(TensorRef, call.predicate),
                        call.call_id,
                        device,
                    )
                    for call in device_calls
                ),
                device=device,
            )
            recorded.extend(reads)
            for call, read in zip(device_calls, reads, strict=True):
                identity = calls.call_identity(call)
                captures.append(
                    (identity, buffer.capture(read.tensor), rows[identity])
                )
            tensor_store.complete_reads(reads, device=device)

        # Sealing forbids further captures, so seal now only when no
        # transferred row remains for `capture_predicates`.
        if not pending:
            buffer.seal()
    except BaseException:
        # Every acquired read must receive a reader event even when preparation
        # fails before all device groups are captured; `complete_reads` skips
        # reads already completed above.
        unrecorded = tuple(recorded)
        if unrecorded:
            tensor_store.complete_reads(unrecorded)
        buffer.abandon()
        raise
    state.inputs.predicate = buffer
    state.predicate_entries = captures
    state.predicate_transfers = tuple(pending)


def capture_predicates(state: BatchState, tensor_store: TensorStore) -> None:
    """Submit transferred predicate copies on the worker execution thread.

    Does nothing when there is no predicate buffer, it is already sealed, or
    any transferred predicate source is not yet ready; the executor calls it
    again on later advances. On failure the predicate buffer is abandoned.
    """
    buffer = state.inputs.predicate
    if buffer is None or buffer.sealed:
        return
    if not all(
        state.inputs.input_ready(source)
        for _, source, _ in state.predicate_transfers
    ):
        return

    try:
        for identity, source, row in state.predicate_transfers:
            read = cast(TensorRead, state.inputs.tensor(source))
            tensor_store.wait_import(read)
            state.predicate_entries.append(
                (identity, buffer.capture(read.tensor), row)
            )
        buffer.seal()
    except BaseException:
        buffer.abandon()
        raise


def reserve_outputs(
    batch: Batch,
    predicate_values: Mapping[CallIdentity, bool],
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    host_tasks: HostLane,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    output_pool: OutputPool,
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelExecutor,
    config: WorkerConfig,
) -> None:
    """Bind one batch's outputs and execution resources before launch.

    `predicate_values` holds the resolved value of each completion-predicated
    call. A call whose predicate is false is marked `CallStatus.PREDICATED`:
    it keeps its aligned output row and binds its completion and transition
    scalar outputs, which receive false sentinels, but reserves no host
    tasks, cache tables, latent staging, inputs, or other outputs.

    Resources are bound in dependency order: the completion buffer and
    `PendingOutput` records, host-lane slots, cache block tables, latent
    staging, device output writes, then transferred inputs and predicates.
    If creating the records fails, the local completion buffer is abandoned.
    Once records are bound, the native executor owns failure cleanup.
    """
    required_predicates = {
        calls.call_identity(call)
        for call in batch.calls
        if call.predicate is not None
        and call.predicate.dtype is DType.U8
        and not calls.device_gated(call)
    }
    if required_predicates != set(predicate_values):
        raise invalid_descriptor(
            "completion-predicated calls require exact prepared "
            "predicate values"
        )

    scheduled = state.batch.calls

    # Predicated rows remain in aligned output/state tables but do not reserve
    # execution-only inputs, CPU tasks, or model resources.
    predicated = frozenset(
        identity
        for call in scheduled
        if (identity := calls.call_identity(call)) in predicate_values
        and not predicate_values[identity]
    )
    active_calls = (
        scheduled
        if not predicated
        else tuple(
            call
            for call in scheduled
            if calls.call_identity(call) not in predicated
        )
    )

    if active_calls:
        stream = model_runner.call_stream(active_calls[0])
        if stream is not None:
            # A standalone capability's batch runs on the stream
            # `ModelExecutor.call_stream` selects for its first active call.
            # Request slots are initialized on the device's current stream,
            # so the batch stream waits for it; independent components publish
            # their own producer/consumer fences.
            stream.wait_stream(torch.cuda.current_stream(stream.device))
            state.stream = stream

    with state.scope():
        started = time.perf_counter_ns()

        completion: OutputBuffer | None = None
        try:
            # The completion buffer and the records bound to it are owned
            # together: a failure before `state.bind_outputs` abandons any
            # acquired buffer here, and a later one is handled by
            # the native executor.
            completion = output_pool.acquire(
                len(scheduled),
                token_capacity=_completion_words(scheduled),
                devices=tuple(
                    dict.fromkeys(
                        device
                        for call in scheduled
                        for device in model_runner.call_devices(call)
                    )
                ),
            )
            state.bind_outputs(
                request_pool,
                completion,
                started,
                {identity[0].request_id for identity in predicated},
            )
        except BaseException:
            if completion is not None:
                completion.abandon()
            raise

        # Bind physical state in dependency order before publishing
        # transferred inputs.
        _reserve_host_tasks(
            active_calls,
            host_tasks=host_tasks,
            worker_info=worker_info,
            config=config,
            state=state,
        )
        if active_calls:
            identities = {calls.call_identity(call) for call in active_calls}
            has_forward = any(
                calls.call_identity(batch.calls[index]) in identities
                for index in batch.forward_call_indices
            )

            if kv_cache is None or request_tables is None:
                if has_forward:
                    raise unsupported_setup(
                        "KV-free execution received cache forward rows"
                    )
            else:
                state.bind_cache(
                    kv_cache._manager,
                    request_tables._tables,
                    request_tables._copy_tables,
                    kv_cache.cache.recycle_units,
                )
            state.bind_latents(
                latent_pool,
                model_runner.image_builder,
                model_runner.media_builder,
            )
        _reserve_outputs(
            scheduled,
            tensor_store=tensor_store,
            model_runner=model_runner,
            state=state,
        )

        state.complete_inputs(
            tensor_store,
            latent_pool,
            None if kv_cache is None else kv_cache._manager,
        )
        _consume_predicates(
            active_calls,
            tensor_store=tensor_store,
            model_runner=model_runner,
            state=state,
        )
        _publish_predicated_outputs(
            scheduled,
            tensor_store=tensor_store,
            state=state,
        )
        record_component(state.component_us, "open_lane", started)


def _completion_words(scheduled: tuple[Call, ...]) -> int:
    """Compute completion-word capacity for one batch."""
    # Capacity is counted in `OutputBuffer` elements (int64), matching the
    # `OutputPool.max_words` bound set in `uniserve_worker.worker`: one
    # sampling column of `SAMPLING_COMPLETION_FIELDS` elements per call, plus
    # each call's `max_completion_bytes` rounded up at 4 bytes per element.
    return max(
        1,
        SAMPLING_COMPLETION_FIELDS * len(scheduled)
        + sum(
            (int(call.bounds.max_completion_bytes) + 3) // 4
            for call in scheduled
        ),
    )


def _reserve_host_tasks(
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    host_tasks: HostLane,
    worker_info: WorkerInfo,
    config: WorkerConfig,
) -> None:
    """Reserve bounded host-lane slots for the calls that run host work.

    A media unit encode reserves one slot per unit this rank takes from the
    round; audio encoding, muxing and image decoding reserve one each. A
    non-distributed component's encode and mux work belongs to its
    export owner (`WorkerInfo.output_rank`), so other ranks reserve
    nothing for it; image decoding reserves on every rank that runs it.

    The slots are stored in the call's `PendingOutput.host_tasks`. A failure
    abandons the current call's reservations; those of earlier calls stay
    with their records for the native executor to release.
    """
    for call in scheduled:
        if call.kind not in {
            MediaCall.IMAGE_DECODING,
            MediaCall.MEDIA_READING,
            MediaCall.VIDEO_ENCODING,
            MediaCall.AUDIO_ENCODING,
            MediaCall.MUXING,
        }:
            continue

        component = next(
            (
                binding
                for binding in worker_info.components
                if binding.name == call.component
            ),
            None,
        )
        distributed = (
            component is not None and component.config.distribution is not None
        )
        if (
            call.kind is not MediaCall.IMAGE_DECODING
            and not distributed
            and config.rank != worker_info.output_rank(call.component)
        ):
            continue

        pending = state.pending_output(call.request_key.request_id)
        if pending.host_tasks:
            raise invalid_descriptor(
                "materialization repeats its CPU task identity"
            )

        count = 1
        if call.kind is MediaCall.VIDEO_ENCODING and component is not None:
            count = len(
                encoded_unit_positions(
                    call,
                    state=state,
                    component=component.config,
                    rank=config.rank,
                )
            )
        reservations: list[HostTask] = []
        try:
            for _ in range(count):
                reservations.append(host_tasks.reserve())
        except BaseException:
            for reservation in reservations:
                reservation.abandon()
            raise
        pending.set_host_tasks(reservations)


def _reserve_outputs(
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelExecutor,
) -> None:
    """Bind each declared device value to its concrete bounded owner.

    Transfer, model, and image outputs form one group, encoder features are
    reserved separately through `TensorStore.reserve_features`, and the
    token, completion, and transition scalars are grouped by device, dtype,
    and shape bound. `TensorStore.bind_output_groups` binds all groups or
    none. Every bound write is appended to its record's `writes`, and the
    token, transition, completion, and producer writes are also classified
    onto the record.
    """
    regions = {}
    shapes = {}
    # Scalar outputs are grouped by device, dtype, and shape bound. A group
    # binds to scheduler buffer allocations or, without one, to request-relay
    # slots, and `TensorStore.bind_outputs` rejects a group mixing the two.
    scalar_groups: dict[
        tuple[torch.device, DType, ShapeBound],
        list[tuple[TensorRef, torch.device | str]],
    ] = {}
    persistent_bindings: list[tuple[TensorRef, torch.device | str]] = []
    encoder_bindings: list[tuple[TensorRef, torch.device | str]] = []
    by_identity = {calls.call_identity(call): call for call in scheduled}

    for call in scheduled:
        # Outputs are bound on the call's output device.
        device = model_runner.call_devices(call)[2]
        request = state.pending_output(call.request_key.request_id)
        predicated = request.status is CallStatus.PREDICATED

        if not predicated:
            decode = next(
                (
                    params
                    for params in state.batch.decode_ranges
                    if params.call_id == call.call_id
                    and params.request_key == call.request_key
                ),
                None,
            )

            for output in call.outputs:
                if call.kind is TransferMode.TENSOR:
                    # Transfers publish the delivered input's representation;
                    # their destination component does not execute model
                    # mathematics.
                    persistent_bindings.append((output, device))
                    continue
                layout = model_runner.output_layout(
                    call.component,
                    output.output_index,
                    request.request.admission.diffusion,
                    decode,
                    len(request.request.admission.prompt_token_ids),
                    request.request.admission.video,
                )
                # None: this rank does not publish the output.
                if layout is None:
                    continue

                # A region is recorded only when this rank publishes part of
                # the logical shape.
                shapes[output] = layout.shape
                if layout.local_slice != tuple(
                    slice(0, extent) for extent in layout.shape
                ):
                    regions[output] = layout.local_slice
                persistent_bindings.append((output, device))

            if call.image_output is not None:
                persistent_bindings.append((call.image_output, device))
            if call.encoder_output is not None:
                encoder_bindings.append((call.encoder_output, device))

        # A skipped computation propagates false predicates but publishes no
        # sampled token, feature, image, or latent state.
        for scalar in (
            call.token_output,
            call.completion_output,
            call.transition_output,
        ):
            if scalar is None or (predicated and scalar == call.token_output):
                continue
            scalar_groups.setdefault(
                (device, scalar.dtype, scalar.shape_bound), []
            ).append((scalar, device))

    groups = tuple(tuple(group) for group in scalar_groups.values())
    if persistent_bindings:
        groups = (*groups, tuple(persistent_bindings))

    request_slots = {
        request.request.request_key: int(request.request.request_pool_idx)
        for request in state.pending_outputs()
    }
    allocations = {
        params.buffer: params for params in state.batch.buffer_allocations
    }
    bound_groups = tensor_store.bind_output_groups(
        groups,
        regions=regions,
        shapes=shapes,
        request_slots=request_slots,
        buffer_allocations=allocations,
    )

    # Record the bound writes before `reserve_features` can fail, so
    # The native executor abandons them with the rest of the batch.
    for binding in bound_groups:
        for write in binding:
            request = state.pending_output(
                write.reference.request_key.request_id
            )
            request.writes.append(write)

    features = tensor_store.reserve_features(
        tuple(encoder_bindings), buffer_allocations=allocations
    )
    for write in features:
        request = state.pending_output(write.reference.request_key.request_id)
        request.writes.append(write)

    # Classify each bound write against its producer's declared output roles.
    # `producer_write` is the call's first write bound here other than its
    # transition output; `_finish_device_reads` in
    # `uniserve_worker.execution.commit` passes it to
    # `TensorStore.complete_reads` as a fence candidate for the call's device
    # reads.
    for write in (write for binding in bound_groups for write in binding):
        identity = (
            write.reference.request_key,
            write.reference.producer_call_id,
        )
        producer = by_identity.get(identity)
        if producer is None:
            raise RuntimeError(
                "device output binding has no computation in the execution "
                "batch"
            )
        request = state.pending_output(producer.request_key.request_id)
        if write.reference == producer.token_output:
            request.token_write = write
        elif write.reference == producer.transition_output:
            request.transition_write = write
        if write.reference == producer.completion_output:
            request.completion_write = write
        if (
            write.reference != producer.transition_output
            and request.producer_write is None
        ):
            request.producer_write = write


def _consume_predicates(
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelExecutor,
) -> None:
    """Hand each active call its predicate tensor and register the reader.

    Reads are consumed from the `TensorStore` in one batch per compute
    device. Each read is appended to the record's `device_reads`, which
    `uniserve_worker.execution.commit` completes, and the tensor is stored in
    `PendingOutput.predicate` with a flag marking an I64 relay tag.
    """
    grouped: dict[
        torch.device,
        list[
            tuple[
                Call,
                tuple[TensorRef, CallId, torch.device | str | None],
            ]
        ],
    ] = {}
    for call in scheduled:
        predicate = call.predicate
        if predicate is None:
            continue
        device = model_runner.call_devices(call)[0]
        grouped.setdefault(device, []).append(
            (
                call,
                (
                    predicate,
                    call.call_id,
                    device,
                ),
            )
        )

    for device, entries in grouped.items():
        reads = tensor_store.consume_batch(
            tuple(request for _call, request in entries),
            device=device,
        )
        for (call, _request), read in zip(entries, reads, strict=True):
            predicate = cast(TensorRef, call.predicate)
            # I64 predicates are relay tags carrying a predecessor's decision;
            # U8 predicates are plain completion booleans.
            tagged = predicate.dtype is DType.I64
            request = state.pending_output(call.request_key.request_id)
            request.device_reads.append(read)
            request.predicate = (read.tensor, tagged)


def _publish_predicated_outputs(
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    tensor_store: TensorStore,
) -> None:
    """Publish false into the scalar outputs of predicated calls.

    The completion and transition outputs of a call that does not run carry
    the inactive decision to their consumers.
    """
    for call in scheduled:
        request = state.pending_output(call.request_key.request_id)
        if request.status is CallStatus.PREDICATED:
            for write in (request.completion_write, request.transition_write):
                if write is not None:
                    tensor_store.write_scalar(write, False)
