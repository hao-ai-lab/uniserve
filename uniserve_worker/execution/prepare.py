"""Validate physical runs and reserve their input, request.

and output resources.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from uniserve.media import image as media_image
from uniserve.model import VideoPostprocessor
from uniserve_worker.execution import calls as calls
from uniserve_worker.execution.batch_state import BatchState
from uniserve_worker.execution.commit import _discard_group
from uniserve_worker.execution.output import OutputBuffer, PendingOutput
from uniserve_worker.execution.rows import CallIdentity
from uniserve_worker.execution.sampling import SAMPLING_COMPLETION_FIELDS
from uniserve_worker.execution.video import require_media_output_ring
from uniserve_worker.execution.video import (
    validate_batch as validate_video_batch,
)
from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    unsupported_call,
    unsupported_setup,
)
from uniserve_worker.profiling import record_component
from uniserve_worker.protocol.batch import (
    Batch,
    LatentParams,
    TensorPublication,
)
from uniserve_worker.protocol.call import (
    Call,
    CallStatus,
    MediaCall,
    TransferMode,
)
from uniserve_worker.protocol.identity import BufferId, CallId
from uniserve_worker.protocol.tensor import DType, ShapeBound, TensorRef
from uniserve_worker.protocol.transfer import (
    DeviceProductTransferValue,
    EncoderTransferValue,
    LatentTransferValue,
)
from uniserve_worker.runtime.latent_pool import LatentImport
from uniserve_worker.runtime.tensor_store import (
    FeatureMetadata,
    ImageMetadata,
    TensorRead,
)

if TYPE_CHECKING:
    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.execution.output import OutputPool
    from uniserve_worker.media.buffers import MediaBuffers
    from uniserve_worker.media.mux import MediaMux
    from uniserve_worker.runtime.block_tables import BlockTables
    from uniserve_worker.runtime.cache_manager import CacheManager
    from uniserve_worker.runtime.host_lane import HostLane
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.request import RequestPool
    from uniserve_worker.runtime.tensor_store import TensorStore
    from uniserve_worker.transfer.tickets import Transport


logger = logging.getLogger(__name__)


def prepare_batch(
    prepared: BatchState,
    *,
    kv_cache: CacheManager | None,
    latent_pool: LatentPool | None,
    request_tables: BlockTables | None,
    request_pool: RequestPool,
) -> None:
    """Materialize KV publications and record storage write dependencies for.

    the batch.

    Runs after admission and release controls have been applied. KV install
    calls receive a cache publication for their source, and every latent,
    cache-page, and KV write records the future that must complete before its
    target storage is written.
    """
    batch = prepared.batch
    storage_dependencies: list[Future[None]] = []

    pool = latent_pool
    if pool is not None:
        admissions = {
            admission.request_key: admission.request_pool_idx
            for admission in batch.admissions
        }
        scheduled = {
            (call.request_key, call.call_id): call for call in batch.calls
        }
        for latent_params in batch.latent_params:
            call = scheduled[(latent_params.request_key, latent_params.call_id)]
            if call.kind not in {
                MediaCall.LATENT_PREPARATION,
                MediaCall.DENOISING,
            }:
                continue

            request = request_pool.peek(latent_params.request_key.request_id)
            request_slot = (
                admissions.get(latent_params.request_key)
                if request is None
                else request.request_pool_idx
            )
            if request_slot is None:
                raise invalid_descriptor(
                    "latent write has no admitted request slot"
                )
            storage_dependencies.extend(
                pool.write_dependencies(request_slot, latent_params.page_table)
            )

    # Install calls may reference sources without a scheduler-supplied
    # publication; the cache materializes one for each missing source.
    entries = list(batch.input_products)
    kv_entries = list(batch.kv_inputs)
    supplied = {publication.source for publication in kv_entries}
    for call in batch.calls:
        if call.kind is not TransferMode.KV_INSTALL:
            continue
        source = call.kv_input
        if source is None:
            raise invalid_descriptor(
                "KV installation requires a source publication"
            )
        if source not in supplied:
            if kv_cache is None:
                raise invalid_descriptor(
                    "KV installation requires cache publication storage"
                )
            kv_entries.append(kv_cache.publication(source))
            supplied.add(source)

    cache = kv_cache
    tables = request_tables
    if cache is not None and tables is not None and cache.has_pending_accesses:
        request_slots = {
            admission.request_key: admission.request_pool_idx
            for admission in batch.admissions
        }
        kv_inputs = {
            publication.source: publication for publication in kv_entries
        }
        assigned = {
            (table.request_pool_idx, table.group_id): table.page_ids
            for table in batch.block_tables
        }

        def pages_for(slot: int, group: int) -> tuple[int, ...]:
            pages = assigned.get((slot, group))
            return tables.pages(slot, group) if pages is None else pages

        # New pages are written from the start; ongoing rows append after the
        # tokens already visible in their sequence.
        for allocation in batch.new_cache_pages:
            storage_dependencies.extend(
                cache.write_dependencies(
                    allocation.page_ids,
                    group=allocation.group_id,
                    start=0,
                    length=len(allocation.page_ids) * cache.info.block_size,
                )
            )

        for row, write_kv in enumerate(batch.write_kv):
            if not write_kv:
                continue
            storage_dependencies.extend(
                cache.write_dependencies(
                    pages_for(batch.request_pool_indices[row], 0),
                    group=0,
                    start=batch.seq_lens[row] - batch.query_lens[row],
                    length=batch.query_lens[row],
                )
            )

        # An installation overwrites the published extent of its source into
        # the destination request's page table.
        for call in batch.calls:
            if call.kind is not TransferMode.KV_INSTALL:
                continue
            request = request_pool.peek(call.request_key.request_id)
            slot = (
                request_slots.get(call.request_key)
                if request is None
                else request.request_pool_idx
            )
            if slot is None:
                raise invalid_descriptor(
                    "KV installation has no admitted request slot"
                )
            source = call.kv_input
            if source is None:
                raise invalid_descriptor(
                    "KV installation requires a source publication"
                )
            kv_publication = kv_inputs.get(source)
            if kv_publication is not None:
                storage_dependencies.extend(
                    cache.write_dependencies(
                        pages_for(slot, kv_publication.group_id),
                        group=kv_publication.group_id,
                        start=kv_publication.base_extent,
                        length=kv_publication.published_extent
                        - kv_publication.base_extent,
                    )
                )

    prepared.storage_dependencies = tuple(storage_dependencies)
    prepared.input_products = tuple(entries)
    prepared.kv_inputs = tuple(kv_entries)


def prepare_inputs(
    state: BatchState,
    *,
    kv_cache: CacheManager | None,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    output_pool: OutputPool,
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    transfer_backends: Mapping[str, Transport],
    config: WorkerConfig,
) -> None:
    """Reserve transfer destinations and submit reads after their storage is.

    available.

    Each cross-call input descriptor is validated against its declared product
    before a destination is reserved; on failure, every input reserved so far
    is released before the error propagates.
    """
    from . import transfer

    batch = state.batch
    entries, kv_entries = state.input_products, state.kv_inputs
    transports = transfer_backends
    if (entries or kv_entries) and not transports:
        raise unsupported_setup(
            "cross-call input requires a configured transport"
        )

    try:
        for entry in entries:
            assert transports

            # One transferred product must land on exactly one consumer device.
            devices = {
                model_runner.call_devices(call)[0]
                for call in batch.calls
                if entry.product in call.tensor_inputs()
                or entry.product == call.predicate
            }
            if len(devices) != 1:
                raise invalid_descriptor(
                    "transferred product requires one consumer device per batch"
                )
            device = next(iter(devices))

            value = entry.value
            if isinstance(value, EncoderTransferValue):
                main = value.tensor
                if (
                    not isinstance(value.payload_kind, str)
                    or value.payload_kind
                    not in {"vision_feature", "latent_feature"}
                    or min(value.height, value.width) < 1
                    or not transfer.tensor_matches_product(main, entry.product)
                ):
                    raise invalid_descriptor(
                        "encoder transfer metadata exceeds its product bounds"
                    )
                if not any(
                    entry.product
                    == (
                        call.vision_input
                        if value.payload_kind == "vision_feature"
                        else call.latent_feature_input
                    )
                    for call in batch.calls
                ):
                    raise invalid_descriptor(
                        "encoder transfer entry disagrees with its product "
                        "identity"
                    )

            elif isinstance(value, DeviceProductTransferValue):
                main = value.tensor
                if min(value.height, value.width) < 0:
                    raise invalid_descriptor(
                        "device-product image dimensions must be non-negative"
                    )
                if (value.height == 0) != (value.width == 0):
                    raise invalid_descriptor(
                        "device-product image dimensions are incomplete"
                    )
                if value.value_range not in {"", "signed_unit", "unit"}:
                    raise invalid_descriptor(
                        "device-product value range is invalid"
                    )
                if value.height == 0 and value.value_range:
                    raise invalid_descriptor(
                        "non-image device product carries an image range"
                    )
                if not any(
                    entry.product
                    in (
                        *call.inputs,
                        call.token_input,
                        call.image_input,
                        call.predicate,
                    )
                    for call in batch.calls
                ) or not transfer.tensor_matches_product(main, entry.product):
                    raise invalid_descriptor(
                        "device-product transfer metadata exceeds its product "
                        "bounds"
                    )

            elif isinstance(value, LatentTransferValue):
                main = value.tensor
                pool = latent_pool

                # A latent payload must exactly match the pool's element
                # layout: [latent_units, latent_width] at the pool dtype.
                expected_dtype = (
                    ""
                    if pool is None
                    else str(pool.dtype).removeprefix("torch.")
                )
                expected_nbytes = (
                    0
                    if pool is None
                    else value.latent_units
                    * int(pool.latent_width)
                    * int(pool.storage.element_size())
                )
                if (
                    not any(
                        entry.product == call.latent_input
                        for call in batch.calls
                    )
                    or pool is None
                    or min(value.height, value.width, value.latent_units) < 1
                    or tuple(main.shape)
                    != (value.latent_units, int(pool.latent_width))
                    or main.dtype != expected_dtype
                    or main.nbytes != expected_nbytes
                    or main.nbytes > entry.product.max_bytes
                    or math.prod(main.shape)
                    > entry.product.shape_bound.max_elements
                ):
                    raise invalid_descriptor(
                        "latent transfer metadata exceeds its product bounds"
                    )

            else:
                raise invalid_descriptor(
                    "cross-call transfer entry has an unknown kind"
                )

            parameters = {
                params.buffer: params for params in batch.buffer_allocations
            }
            if isinstance(
                value, (EncoderTransferValue, DeviceProductTransferValue)
            ):
                request_slots = {
                    admission.request_key: int(admission.request_pool_idx)
                    for admission in batch.admissions
                }
                resident = request_pool.peek(
                    entry.product.request_key.request_id
                )
                if (
                    resident is not None
                    and resident.request_key == entry.product.request_key
                ):
                    request_slots[resident.request_key] = int(
                        resident.request_pool_idx
                    )
                imported = tensor_store.import_tensor(
                    entry.product,
                    value.tensor,
                    device=device,
                    request_slots=request_slots,
                    buffer_allocations=parameters,
                    bindings={
                        (location.source, location.backend): transports[
                            location.backend
                        ]
                        for location in value.tensor.locations
                        if location.backend in transports
                    },
                    metadata=(
                        FeatureMetadata(height=value.height, width=value.width)
                        if isinstance(value, EncoderTransferValue)
                        else None
                        if value.height == 0
                        else ImageMetadata(
                            height=value.height,
                            width=value.width,
                            value_range=None
                            if not value.value_range
                            else (
                                (-1.0, 1.0)
                                if value.value_range == "signed_unit"
                                else (0.0, 1.0)
                            ),
                        )
                    ),
                )
                assert imported.imported is not None
                state.tensor_reads[entry.product.buffer_id] = imported
                continue

            elif isinstance(value, LatentTransferValue):
                consumers = tuple(
                    call
                    for call in batch.calls
                    if entry.product in call.tensor_inputs()
                )
                if len(consumers) != 1:
                    raise invalid_descriptor(
                        "latent transfer must have one consumer"
                    )
                consumer = consumers[0]

                params = next(
                    (
                        params
                        for params in batch.latent_params
                        if (params.request_key, params.call_id)
                        == (consumer.request_key, consumer.call_id)
                    ),
                    None,
                )
                if params is None or (
                    value.latent_units,
                    value.height,
                    value.width,
                    value.step,
                ) != (
                    params.latent_units,
                    params.height,
                    params.width,
                    params.start_step,
                ):
                    raise invalid_descriptor(
                        "latent transfer disagrees with its scheduler params"
                    )

                resident = request_pool.peek(
                    entry.product.request_key.request_id
                )
                admission = next(
                    (
                        row
                        for row in batch.admissions
                        if row.request_key == entry.product.request_key
                    ),
                    None,
                )
                if (
                    resident is not None
                    and resident.request_key == entry.product.request_key
                ):
                    slot = int(resident.request_pool_idx)
                elif admission is not None:
                    slot = int(admission.request_pool_idx)
                else:
                    raise invalid_descriptor(
                        "latent transfer has no request slot"
                    )

                # Transfer metadata validation above established the physical
                # pool.
                assert latent_pool is not None
                pool = latent_pool
                binding = pool.reserve_import(
                    entry.product,
                    request_pool_idx=slot,
                    page_table=params.page_table,
                    latent_units=value.latent_units,
                )

                # The actual reservation retains tickets before fetch can fail.
                state.latent_imports[entry.product.buffer_id] = binding
                from ..transfer.layout import fetch_tensor

                fetch_tensor(
                    value.tensor,
                    binding.spans,
                    bindings={
                        (location.source, location.backend): transports[
                            location.backend
                        ]
                        for location in value.tensor.locations
                        if location.backend in transports
                    },
                    retain=partial(pool.retain_transfer, binding),
                )

        for kv_transfer in kv_entries:
            consumers = tuple(
                call
                for call in batch.calls
                if call.kv_input == kv_transfer.source
            )
            if (
                len(consumers) != 1
                or consumers[0].kind is not TransferMode.KV_INSTALL
            ):
                raise invalid_descriptor(
                    "KV input requires one installation consumer"
                )

            publications = kv_cache
            cache = kv_cache
            tables = request_tables
            if publications is None or cache is None or tables is None:
                raise invalid_descriptor(
                    "KV input requires physical cache storage"
                )

            resident = request_pool.peek(kv_transfer.source.owner.request_id)
            admission = next(
                (
                    row
                    for row in batch.admissions
                    if row.request_key == kv_transfer.source.owner
                ),
                None,
            )
            if (
                resident is not None
                and resident.request_key == kv_transfer.source.owner
            ):
                slot = int(resident.request_pool_idx)
            elif admission is not None:
                slot = int(admission.request_pool_idx)
            else:
                raise invalid_descriptor(
                    "KV transfer has no admitted request slot"
                )

            # Prefer scheduler-supplied block tables; otherwise fall back to
            # the request's currently installed pages and allocated length.
            table = next(
                (
                    table
                    for table in batch.block_tables
                    if (table.request_pool_idx, table.group_id)
                    == (slot, kv_transfer.group_id)
                ),
                None,
            )
            pages = (
                tables.pages(slot, kv_transfer.group_id)
                if table is None
                else table.page_ids
            )
            allocated = (
                tables.allocated_length(slot)
                if table is None
                else table.allocated_tokens
            )
            initialized = tuple(
                page
                for allocation in batch.new_cache_pages
                if (allocation.request_pool_idx, allocation.group_id)
                == (slot, kv_transfer.group_id)
                for page in allocation.page_ids
            )
            write = publications.prepare_install(
                kv_transfer.source,
                kv_transfer,
                request_pool_idx=slot,
                group_id=kv_transfer.group_id,
                page_ids=pages,
                allocated_length=allocated,
                initialized_pages=initialized,
                transports=transports,
            )
            state.cache_imports[kv_transfer.source] = write

        _prepare_predicates(
            state,
            tensor_store=tensor_store,
            output_pool=output_pool,
            model_runner=model_runner,
        )
    except BaseException as error:
        # Release every input reserved above; a cleanup failure annotates the
        # original error rather than masking it.
        try:
            state.close_inputs(tensor_store, latent_pool, kv_cache)
        except BaseException as cleanup_error:
            error.add_note(f"batch input cleanup failed: {cleanup_error!r}")
        raise


def _prepare_predicates(
    state: BatchState,
    *,
    tensor_store: TensorStore,
    output_pool: OutputPool,
    model_runner: ModelRunner,
) -> None:
    """Capture completion-valued predicates from local products or prepared.

    transfers.
    """
    # Predicate rows occupy one compact completion buffer regardless of whether
    # their source is already local or will arrive through a prepared transfer.
    scheduled = tuple(
        call
        for call in state.batch.calls
        if call.predicate is not None and call.predicate.dtype is DType.U8
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
            if source not in state.tensor_reads:
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

        # No pending transfer can mutate the buffer once it is sealed.
        sealed = not pending
        if sealed:
            buffer.seal()
    except BaseException:
        # Every acquired read must receive a reader event even when preparation
        # fails before all device groups are captured.
        unrecorded = tuple(recorded)
        if unrecorded:
            tensor_store.complete_reads(unrecorded)
        buffer.abandon()
        raise
    state.predicate_buffer = buffer
    state.predicate_entries = captures
    state.predicate_transfers = tuple(pending)
    state.predicates_sealed = sealed


def capture_predicates(state: BatchState, tensor_store: TensorStore) -> None:
    """Submit transferred predicate copies on the worker execution thread."""
    buffer = state.predicate_buffer
    if buffer is None or state.predicates_sealed:
        return
    if not all(
        state.input_ready(source) for _, source, _ in state.predicate_transfers
    ):
        return

    try:
        for identity, source, row in state.predicate_transfers:
            read = state.tensor_reads[source]
            tensor_store.wait_import(read)
            state.predicate_entries.append(
                (identity, buffer.capture(read.tensor), row)
            )
        buffer.seal()
    except BaseException:
        buffer.abandon()
        raise

    state.predicates_sealed = True


def _open_group(
    batch: Batch,
    completion_group: int,
    predicate_values: Mapping[CallIdentity, bool],
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    host_tasks: HostLane,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    media_buffers: MediaBuffers | None,
    output_pool: OutputPool,
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    transfer_backends: Mapping[str, Transport],
    config: WorkerConfig,
) -> int:
    """Stage one completion group's speculative state, resources, inputs.

    and completion storage.
    """
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
            # Request slots are initialized on the control stream. Independent
            # components publish their own producer/consumer fences.
            stream.wait_stream(torch.cuda.current_stream(stream.device))
            state.group_streams[completion_group] = stream

    with state.group_scope(completion_group):
        # Restrict input payloads to identities declared by this completion
        # group.
        started = time.perf_counter_ns()
        declared_inputs = {
            reference.buffer_id
            for call in scheduled
            for reference in call.tensor_inputs()
        }
        declared_inputs.update(
            call.kv_input for call in scheduled if call.kv_input is not None
        )
        declared_inputs.update(
            call.predicate.buffer_id
            for call in scheduled
            if call.predicate is not None
        )
        input_products = tuple(
            payload
            for payload in batch.input_products
            if payload.product.buffer_id in declared_inputs
        )

        completion: OutputBuffer | None = None
        try:
            # Candidate drafts and completion slots form a speculative ownership
            # unit: either all later completion group resources bind
            # successfully or both are discarded.
            request_pool_indices = tuple(
                int(
                    request_pool.get(
                        call.request_key.request_id
                    ).request_pool_idx
                )
                for call in scheduled
            )
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
            candidates = request_pool.create_outputs(
                scheduled, request_pool_indices, completion
            )
        except BaseException:
            if completion is not None:
                completion.abandon()
            raise

        assert completion is not None
        for request in candidates:
            if calls.call_identity(request.call) in predicated:
                request.status = CallStatus.PREDICATED
        state.bind_outputs(completion_group, candidates, completion, started)

        try:
            # Bind physical state in dependency order before decoding
            # transferred inputs.
            _reserve_host_tasks(
                active_calls,
                completion_group,
                host_tasks=host_tasks,
                worker_info=worker_info,
                media_buffers=media_buffers,
                postprocessor=model_runner.video_postprocessor,
                config=config,
                state=state,
            )
            if active_calls:
                identities = {calls.call_identity(op) for op in active_calls}
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
                    _bind_cache_tables(
                        active_calls,
                        completion_group,
                        kv_cache=kv_cache,
                        request_tables=request_tables,
                        state=state,
                    )
                _bind_latent_inputs(
                    active_calls,
                    completion_group,
                    latent_pool=latent_pool,
                    model_runner=model_runner,
                    state=state,
                )
            _reserve_outputs(
                scheduled,
                completion_group,
                tensor_store=tensor_store,
                model_runner=model_runner,
                state=state,
            )

            # Only live calls consume inputs; predicated outputs are
            # published directly into their aligned completion rows.
            active_inputs = {
                reference
                for call in active_calls
                for reference in call.tensor_inputs()
            }
            active_inputs.update(
                call.predicate
                for call in active_calls
                if call.predicate is not None
            )
            _stage_input_products(
                tuple(
                    payload
                    for payload in input_products
                    if payload.product in active_inputs
                ),
                completion_group,
                kv_cache=kv_cache,
                tensor_store=tensor_store,
                latent_pool=latent_pool,
                model_runner=model_runner,
                state=state,
            )
            _consume_predicates(
                active_calls,
                completion_group,
                tensor_store=tensor_store,
                model_runner=model_runner,
                state=state,
            )
            _publish_predicated_outputs(
                scheduled,
                completion_group,
                tensor_store=tensor_store,
                state=state,
            )
            state.group_registered[completion_group] = True
            record_component(
                state.group_component_us[completion_group], "open_lane", started
            )
            return completion_group
        except BaseException:
            _discard_group(
                completion_group,
                kv_cache=kv_cache,
                tensor_store=tensor_store,
                latent_pool=latent_pool,
                media_mux=media_mux,
                transfer_backends=transfer_backends,
                state=state,
            )
            raise


def _completion_words(scheduled: tuple[Call, ...]) -> int:
    """Compute fixed completion-word capacity for all calls in a.

    completion group.
    """
    # Completion storage is addressed in 4-byte words; payload byte budgets
    # round up to whole words.
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
    completion_group: int,
    *,
    state: BatchState,
    host_tasks: HostLane,
    worker_info: WorkerInfo,
    media_buffers: MediaBuffers | None,
    postprocessor: VideoPostprocessor | None,
    config: WorkerConfig,
) -> None:
    """Reserve bounded CPU slots for active calls that schedule host-side.

    work.
    """
    for call in scheduled:
        if call.kind not in {
            MediaCall.IMAGE_DECODING,
            MediaCall.VIDEO_ENCODING,
            MediaCall.AUDIO_ENCODING,
            MediaCall.MUXING,
        }:
            continue

        # A distributed component's host work belongs to whichever of its ranks
        # received the call, because each holds its own media units. Any
        # other component materializes on its single publication owner.
        distributed = any(
            component.name == call.component
            and component.config.distribution is not None
            for component in worker_info.components
        )
        if (
            postprocessor is not None
            and not distributed
            and config.rank != worker_info.output_rank(call.component)
        ):
            continue

        pending = state.pending_output(
            completion_group, call.request_key.request_id
        )
        if pending.completion_tasks:
            raise invalid_descriptor(
                "materialization repeats its CPU task identity"
            )

        reservation = host_tasks.reserve()
        try:
            if call.kind in {
                MediaCall.VIDEO_ENCODING,
                MediaCall.AUDIO_ENCODING,
            }:
                pending.media_lease = require_media_output_ring(
                    media_buffers
                ).reserve(
                    "video"
                    if call.kind is MediaCall.VIDEO_ENCODING
                    else "audio"
                )
        except BaseException:
            reservation.abandon()
            raise

        pending.completion_tasks = (reservation,)


def validate_batch(
    batch: Batch,
    *,
    worker_info: WorkerInfo,
    model_runner: ModelRunner,
    config: WorkerConfig,
    predecessors: Mapping[CallId, CallId | None],
) -> None:
    """Validate batch identity, completion group resources, routing.

    and call support before staging.
    """
    if any(
        params.offset + params.bytes > worker_info.buffer_pool_bytes
        for params in batch.buffer_allocations
    ):
        raise invalid_descriptor(
            "batch buffer params exceeds the worker buffer pool"
        )

    if worker_info.components:
        for call in batch.calls:
            entry = next(
                (
                    entry
                    for entry in worker_info.components
                    if entry.name == call.component
                ),
                None,
            )
            if entry is None or config.rank not in entry.config.ranks:
                raise invalid_descriptor(
                    f"call targets component {call.component!r} outside "
                    "this rank"
                )

    if len(batch.calls) > config.max_batch_calls:
        raise invalid_descriptor(
            "execution batch exceeds the worker_config call limit"
        )

    for call in batch.calls:
        variant = call.kind
        if variant not in worker_info.supported_ops:
            raise unsupported_call(variant.value, call.request_key.request_id)

    if any(
        index > config.max_request_pool_size
        for index in (
            *(table.request_pool_idx for table in batch.block_tables),
            *batch.request_pool_indices,
        )
    ):
        raise invalid_descriptor(
            "execution batch exceeds request-slot capacity"
        )

    validate_video_batch(
        batch,
        postprocessor=model_runner.video_postprocessor,
        predecessors=predecessors,
    )


def _reserve_outputs(
    scheduled: tuple[Call, ...],
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelRunner,
) -> None:
    """Bind each declared device value to its concrete bounded owner."""
    regions = {}
    shapes = {}
    # Scalar outputs sharing device, dtype, and shape bound are co-allocated
    # so one bounded allocation serves every compatible write.
    scalar_groups: dict[
        tuple[torch.device, DType, ShapeBound],
        list[tuple[TensorRef, torch.device | str]],
    ] = {}
    persistent_bindings: list[tuple[TensorRef, torch.device | str]] = []
    encoder_bindings: list[tuple[TensorRef, torch.device | str]] = []
    by_identity = {calls.call_identity(call): call for call in scheduled}

    for call in scheduled:
        device = model_runner.call_devices(call)[2]
        request = state.pending_output(
            completion_group, call.request_key.request_id
        )
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
                )
                if layout is None:
                    continue
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
        for request in state.pending_outputs(completion_group)
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

    # Retain each successful reservation before the next store call can fail.
    for binding in bound_groups:
        for write in binding:
            request = state.pending_output(
                completion_group, write.reference.request_key.request_id
            )
            request.writes.append(write)

    features = tensor_store.reserve_features(
        tuple(encoder_bindings), buffer_allocations=allocations
    )
    for write in features:
        request = state.pending_output(
            completion_group, write.reference.request_key.request_id
        )
        request.writes.append(write)

    # Classify each bound write against its producer's declared output roles.
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
        request = state.pending_output(
            completion_group, producer.request_key.request_id
        )
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
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelRunner,
) -> None:
    """Resolve call predicates from local device products and register.

    their readers.
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
            request = state.pending_output(
                completion_group, call.request_key.request_id
            )
            request.device_reads.append(read)
            request.predicate = (read.tensor, tagged)


def _publish_predicated_outputs(
    scheduled: tuple[Call, ...],
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
) -> None:
    """Publish inactive sentinel values for products of predicated.

    calls.
    """
    for call in scheduled:
        request = state.pending_output(
            completion_group, call.request_key.request_id
        )
        if request.status is CallStatus.PREDICATED:
            for write in (request.completion_write, request.transition_write):
                if write is not None:
                    tensor_store.publish_scalar_write(write, False)


def _bind_latent_inputs(
    scheduled: tuple[Call, ...],
    completion_group: int,
    *,
    state: BatchState,
    latent_pool: LatentPool | None,
    model_runner: ModelRunner,
) -> None:
    """Validate trajectory parameters and bind rank-local latent staging.

    views.
    """
    identities = {calls.call_identity(op) for op in scheduled}
    parameters = tuple(
        params
        for params in state.batch.latent_params
        if (params.request_key, params.call_id) in identities
    )
    if not parameters:
        return

    pool = latent_pool
    requests = {
        calls.call_identity(call): (
            call,
            state.pending_output(completion_group, call.request_key.request_id),
        )
        for call in scheduled
    }

    if pool is None:
        # Fixed request tensors own the trajectory directly. Solver progress
        # remains explicit, without a second paged-storage reservation.
        for params in parameters:
            identity = params.request_key, params.call_id
            selected = requests.get(identity)
            if selected is None:
                raise invalid_descriptor(
                    "latent params names an call outside its completion group"
                )
            call, request = selected
            if params.page_table or params.latent_units:
                raise invalid_descriptor(
                    "paged latent params require a resident latent pool"
                )

            # Preparation runs once at step zero; each denoise call advances
            # exactly one step from the committed solver state.
            if call.kind is MediaCall.LATENT_PREPARATION:
                valid = (
                    int(params.start_step) == 0 and int(params.step_count) == 0
                )
            elif call.kind is MediaCall.DENOISING:
                valid = (
                    int(params.start_step)
                    == int(calls.require_progress(request).flow_step)
                    and int(params.step_count) == 1
                )
            else:
                valid = (
                    int(params.start_step)
                    == int(calls.require_progress(request).flow_step)
                    and int(params.step_count) == 0
                )
            if not valid:
                raise invalid_descriptor(
                    "pool-free latent params disagrees with resident "
                    "generation state"
                )
        return

    # Pooled models bind each call to validated image dimensions and page
    # ownership.
    rows: list[tuple[CallIdentity, LatentParams, int]] = []
    for params in parameters:
        identity = (params.request_key, params.call_id)
        selected = requests.get(identity)
        if selected is None:
            raise invalid_descriptor(
                "latent params names an call outside its completion group"
            )
        call, request = selected
        slot = int(request.request.request_pool_idx)

        image = request.request.image
        if image is None:
            raise invalid_descriptor(
                "latent params has no admitted image dimensions"
            )
        flow = model_runner.image_builder
        if flow is None:
            raise invalid_descriptor(
                "image trajectory has no numerical input builder"
            )
        expected_units = int(
            flow.denoiser.latent_shape(
                "image",
                media_image.Config(int(params.height), int(params.width)),
            )[0]
        )
        if (
            int(params.height) != int(image.height)
            or int(params.width) != int(image.width)
            or int(params.latent_units) != expected_units
        ):
            raise invalid_descriptor(
                "latent params disagrees with admitted model dimensions"
            )

        # A transferred trajectory commits at the step it was published at;
        # a resident trajectory commits at its recorded solver step.
        transferred = next(
            (
                publication.value
                for publication in state.input_products
                if publication.product in call.tensor_inputs()
                and isinstance(publication.value, LatentTransferValue)
            ),
            None,
        )
        committed_step = (
            int(calls.require_progress(request).flow_step)
            if transferred is None
            else transferred.step
        )

        if call.kind is MediaCall.LATENT_PREPARATION:
            if int(params.start_step) != 0 or int(params.step_count) != 0:
                raise invalid_descriptor(
                    "media preparation params carries denoise steps"
                )
        elif call.kind is MediaCall.DENOISING:
            if (
                int(params.start_step) != committed_step
                or int(params.step_count) < 1
                or int(params.start_step) + int(params.step_count)
                > int(image.steps)
                or (
                    int(call.bounds.max_tokens) > 0
                    and int(params.step_count) > int(call.bounds.max_tokens)
                )
            ):
                raise invalid_descriptor(
                    "media denoise params exceeds its committed schedule"
                )
        elif (
            int(params.start_step) != committed_step
            or int(params.step_count) != 0
        ):
            raise invalid_descriptor(
                "latent reader params disagrees with committed step state"
            )

        rows.append((identity, params, slot))
    # Stage every page table together so overlapping physical ownership is
    # rejected before any call receives a writable tensor view.
    staged = pool.stage(
        tuple(params.page_table for _identity, params, _slot in rows),
        tuple(int(params.latent_units) for _identity, params, _slot in rows),
        occupied=tuple(
            output.latent_staging
            for output in state.outputs
            if isinstance(output, PendingOutput)
            and output.latent_staging is not None
        ),
    )

    for (identity, params, slot), value in zip(rows, staged, strict=True):
        request = state.pending_output(completion_group, identity[0].request_id)
        request.input_latent_params = params
        request.latent_staging = value


def _bind_cache_tables(
    scheduled: tuple[Call, ...],
    completion_group: int,
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    request_tables: BlockTables | None,
) -> None:
    """Install scheduler tables and retain row-aligned forward coordinates."""
    cache = kv_cache
    page_tables = request_tables
    if cache is None or page_tables is None:
        raise invalid_descriptor("cache tables require physical KV storage")

    started = time.perf_counter_ns()
    inputs = state.batch
    identities = {calls.call_identity(op) for op in scheduled}

    # Tables are needed for every slot this group reads or writes: its own
    # requests plus any alternative-prefix rows of its forward calls.
    slots = {
        state.pending_output(
            completion_group, call.request_key.request_id
        ).request.request_pool_idx
        for call in scheduled
    }
    slots.update(
        {
            inputs.request_pool_indices[row]
            for row, index in enumerate(inputs.forward_call_indices)
            if calls.call_identity(inputs.calls[index]) in identities
        }
    )

    tables = []
    for table in inputs.block_tables:
        if table.request_pool_idx not in slots:
            continue
        pages = cache.validate_pages(table.page_ids, group=table.group_id)
        if int(table.allocated_tokens) > len(pages) * cache.info.block_size:
            raise invalid_descriptor(
                "block-table allocation exceeds physical capacity"
            )
        tables.append(
            (
                int(table.request_pool_idx),
                int(table.group_id),
                pages,
                int(table.allocated_tokens),
            )
        )
    page_tables.install(tuple(tables))

    for allocation in inputs.new_cache_pages:
        if allocation.request_pool_idx not in slots:
            continue
        pages = cache.validate_pages(
            allocation.page_ids,
            group=allocation.group_id,
        )
        installed = page_tables.pages(
            allocation.request_pool_idx, allocation.group_id
        )
        if not set(pages).issubset(installed):
            raise invalid_descriptor(
                "new cache pages are outside the installed block table"
            )

        # Pages a KV installation will fill are initialized on arrival; every
        # other new page starts zeroed so stale cache content is never read.
        initialized = {
            page
            for write in state.cache_imports.values()
            if (write.request_pool_idx, write.group_id)
            == (allocation.request_pool_idx, allocation.group_id)
            for page in write.initialized_pages
        }
        cache.zero_pages(
            allocation.group_id,
            tuple(page for page in pages if page not in initialized),
        )

    # Indices refer to the original completion group columns, including when
    # inactive call kinds were filtered from the cache-registration view.
    rows_by_call: dict[CallIdentity, list[int]] = defaultdict(list)
    for row, index in enumerate(inputs.forward_call_indices):
        identity = calls.call_identity(inputs.calls[index])
        rows_by_call[identity].append(row)

    for call in scheduled:
        request = state.pending_output(
            completion_group, call.request_key.request_id
        )
        main_slot = int(request.request.request_pool_idx)
        call_rows = rows_by_call.get(calls.call_identity(call), [])
        state.group_forward_indices[completion_group][
            calls.call_identity(call)
        ] = tuple(call_rows)

        main_descriptor = next(
            (
                row
                for row in call_rows
                if inputs.request_pool_indices[row] == main_slot
            ),
            None,
        )
        if main_descriptor is not None:
            visible = int(request.progress.kv_visible_len)
            declared = (
                inputs.seq_lens[main_descriptor]
                - inputs.query_lens[main_descriptor]
            )
            relayed = (
                call.predicate is not None and call.predicate.dtype is DType.I64
            )
            # A queued relay carries a capacity bound computed before its
            # predecessor's predicate was known. Actual KV length and validity
            # come from the device row; an inactive descendant must still drain.
            if declared < visible or (not relayed and declared != visible):
                raise invalid_descriptor(
                    "forward row sequence length disagrees with execution state"
                )

        for descriptor in call_rows:
            slot = inputs.request_pool_indices[descriptor]
            pages = page_tables.pages(slot, 0)
            if slot != main_slot and (
                inputs.seq_lens[descriptor] - inputs.query_lens[descriptor]
            ) > page_tables.allocated_length(slot):
                raise invalid_descriptor(
                    "forward row exceeds alternative-prefix capacity"
                )
            if slot != main_slot:
                page_tables.retain_prefix(call.request_key, slot)

            # Rows that write KV this step own only their committed prefix;
            # read-only rows also cover the tokens they query.
            cache.retain_execution(
                call.request_key,
                pages,
                group=0,
                length=inputs.seq_lens[descriptor]
                - (
                    0
                    if inputs.write_kv[descriptor]
                    else inputs.query_lens[descriptor]
                ),
                completion=state.group_buffers[
                    completion_group
                ].completion_future(),
            )

    record_component(
        state.group_component_us[completion_group], "bc_tables", started
    )


def _stage_input_products(
    input_products: Sequence[TensorPublication],
    completion_group: int,
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    model_runner: ModelRunner,
) -> None:
    """Publish query-ready transferred values into their owning runtime.

    stores.
    """
    # KV imports consumed by this group must be complete and conflict-free
    # before any transferred product is published.
    cache_inputs = {call.kv_input for call in state.batch.calls}
    for buffer, write in state.cache_imports.items():
        if buffer not in cache_inputs:
            continue
        if not write.completion.done():
            raise invalid_descriptor(
                "KV input has no query-ready physical import"
            )
        if kv_cache is None:
            raise invalid_descriptor(
                "KV input requires cache publication storage"
            )
        existing = kv_cache.resident(buffer)
        if existing is not None and existing != write.publication:
            raise invalid_descriptor(
                "staged KV publication conflicts with its buffer identity"
            )

    for entry in input_products:
        product = entry.product
        # Transfer metadata determines which runtime owns the imported value;
        # each branch validates identity and shape before publication.
        if not state.input_ready(product.buffer_id):
            raise invalid_descriptor(
                "cross-call input has no query-ready prepared transfer"
            )

        value = entry.value
        if isinstance(value, LatentTransferValue):
            consumers = tuple(
                call
                for call in state.batch.calls
                if product in call.tensor_inputs()
            )
            if len(consumers) != 1:
                raise invalid_descriptor(
                    "latent transfer must have one completion group consumer"
                )

            row = state.pending_output(
                completion_group, consumers[0].request_key.request_id
            )
            params = row.input_latent_params
            staging = row.latent_staging
            if params is None or staging is None:
                raise invalid_descriptor(
                    "trajectory call has no staged latent inputs"
                )
            if (
                value.latent_units != int(params.latent_units)
                or value.height != int(params.height)
                or value.width != int(params.width)
                or value.step != int(params.start_step)
            ):
                raise invalid_descriptor(
                    "latent transfer disagrees with its scheduler params"
                )

            request = state.pending_output(
                completion_group, product.request_key.request_id
            )
            if int(calls.require_progress(request).flow_step) != 0:
                raise invalid_descriptor(
                    "latent transfer destination already owns a trajectory"
                )

            binding = state.latent_imports.get(product.buffer_id)
            if not isinstance(binding, LatentImport):
                raise RuntimeError(
                    "latent transfer lost its reserved destination"
                )
            if (binding.request_pool_idx, binding.page_table) != (
                row.request.request_pool_idx,
                tuple(params.page_table),
            ):
                raise invalid_descriptor(
                    "latent import reservation changed before execution"
                )

            # The reserved import becomes a live trajectory at the transferred
            # solver step, replacing the destination's empty progress.
            assert latent_pool is not None
            latent_pool.adopt_import(
                binding,
                generation=product.generation,
                step=value.step,
                height=value.height,
                width=value.width,
            )
            row.latent_imported = True
            request.progress = replace(
                calls.require_progress(request), flow_step=value.step
            )
            continue

        consumers = tuple(
            call
            for call in state.batch.calls
            if product in call.tensor_inputs() or call.predicate == product
        )
        if not consumers:
            raise invalid_descriptor(
                "transferred product has no completion group consumer"
            )

        devices = {model_runner.call_devices(call)[0] for call in consumers}
        if len(devices) != 1:
            raise invalid_descriptor(
                "transferred product spans multiple consumer devices"
            )
        if not isinstance(
            value, (DeviceProductTransferValue, EncoderTransferValue)
        ):
            raise RuntimeError("prepared transfer has an unknown descriptor")

        imported = state.tensor_reads.get(product.buffer_id)
        if imported is None:
            raise RuntimeError("tensor transfer lost its reserved destination")
        tensor_store.complete_import(imported)
