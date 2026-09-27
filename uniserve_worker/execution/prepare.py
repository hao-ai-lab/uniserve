"""Validate execution batches and reserve their input and output resources.

The executor (`uniserve_worker.execution.executor`) drives a batch through
these stages, in order:

1. `validate_batch` checks batch-level bounds, routing, and call support
   after the batch's `Start` commands and before its other commands are
   applied.
2. `prepare_batch` runs after the batch's commands are applied. It adds cache
   publications for KV installations and collects the futures
   (`BatchState.storage_dependencies`) that must resolve before this batch's
   writes.
3. `prepare_inputs` runs once those dependencies are done when the batch has
   transferred inputs, and immediately otherwise. It validates each
   cross-call transfer against its declared product, reserves its
   destination, starts the physical fetch, and captures completion-valued
   predicates. Each later advance of the batch calls `capture_predicates`
   until the transferred predicate sources are ready.
4. `reserve_outputs` runs from `uniserve_worker.execution.step` once inputs
   are ready. It creates the batch's `PendingOutput` records and completion
   buffer, then reserves host tasks, cache tables, latent staging, and device
   output writes, and publishes the transferred inputs into their stores.

`prepare_inputs` and `reserve_outputs` release what they reserved when they
fail partway.
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
from uniserve_worker.errors import (
    invalid_descriptor,
    unsupported_call,
    unsupported_setup,
)
from uniserve_worker.execution import calls as calls
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.commit import discard_batch
from uniserve_worker.execution.host_media import (
    BORROWED_INPUT_CALLS,
    encoded_unit_positions,
)
from uniserve_worker.execution.media import (
    validate_batch as validate_video_batch,
)
from uniserve_worker.execution.output import PendingOutput, create_outputs
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
from uniserve_worker.protocol.identity import BufferId, CallId, CallIdentity
from uniserve_worker.protocol.tensor import DType, ShapeBound, TensorRef
from uniserve_worker.protocol.transfer import (
    DeviceProductTransferValue,
    EncoderTransferValue,
    LatentTransferValue,
    PosixShmTransfer,
)
from uniserve_worker.sampling.result import SAMPLING_COMPLETION_FIELDS
from uniserve_worker.storage.block_tables import GroupTable
from uniserve_worker.storage.latent_pool import LatentImport
from uniserve_worker.storage.output import OutputBuffer
from uniserve_worker.storage.tensor_store import (
    FeatureMetadata,
    ImageMetadata,
    TensorRead,
)

if TYPE_CHECKING:
    from uniserve_worker.config.execution.execution import WorkerConfig
    from uniserve_worker.execution.host import HostLane, HostTask
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.execution.request import RequestPool
    from uniserve_worker.media.mux import MediaMux
    from uniserve_worker.protocol.worker_info import WorkerInfo
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.kv_cache import KVCacheManager
    from uniserve_worker.storage.latent_pool import LatentPool
    from uniserve_worker.storage.output import OutputPool
    from uniserve_worker.storage.tensor_store import TensorStore
    from uniserve_worker.transport.interface import Transport


logger = logging.getLogger(__name__)


def prepare_batch(
    prepared: BatchState,
    *,
    kv_cache: KVCacheManager | None,
    latent_pool: LatentPool | None,
    request_tables: BlockTables | None,
    request_pool: RequestPool,
) -> None:
    """Materialize KV publications and record the batch's write dependencies.

    Runs after admission and release controls have been applied. KV install
    calls receive a cache publication for their source, and every latent,
    cache-page, and KV write records the future that must complete before its
    target storage is written.

    Sets `storage_dependencies`, `input_products`, and `kv_inputs` on
    `prepared`; reserves nothing. Malformed batches raise
    `invalid_descriptor` errors, for example a latent write without a
    request slot, a KV installation whose source is neither supplied by the
    batch nor resident in the cache, or, when the cache has pending
    accesses, a KV installation without a request slot.
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
            # Only preparation and denoising write trajectory pages.
            # `LatentPool.write_dependencies` returns the retirements of the
            # publications that hold those pages in the bank the write
            # targets.
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
    # publication; `KVCacheManager.publication` then supplies the source's
    # resident publication and rejects a source that is not resident.
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

    # Without pending cache accesses there is nothing a write could wait for,
    # so the unit scan is skipped.
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

        # New units are covered in full, since a unit leaving one group may
        # hold any tokens of another. A forward row that writes KV covers its
        # `query_lens` tokens after the `seq_lens - query_lens` tokens
        # already in its sequence, in every group.
        for allocation in batch.new_cache_units:
            storage_dependencies.extend(
                cache.write_dependencies(
                    cache.unit_spans(cache.validate_units(allocation.unit_ids))
                )
            )

        for row, write_kv in enumerate(batch.write_kv):
            if not write_kv:
                continue
            start = batch.seq_lens[row] - batch.query_lens[row]
            for table in _slot_tables(
                batch, tables, batch.request_pool_indices[row]
            ):
                storage_dependencies.extend(
                    cache.write_dependencies(
                        table.spans(start, batch.query_lens[row])
                    )
                )

        # An installation overwrites the published extent of its source into
        # the destination request's unit tables.
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
                # Each group's import writes at most the suffix after the
                # base that its destination table still holds.
                end = kv_publication.published_extent
                for table in _slot_tables(batch, tables, slot):
                    start = max(
                        kv_publication.base_extent,
                        table.start_page * table.shape.page_tokens,
                    )
                    if start < end:
                        storage_dependencies.extend(
                            cache.write_dependencies(
                                table.spans(start, end - start)
                            )
                        )

    prepared.storage_dependencies = tuple(storage_dependencies)
    prepared.input_products = tuple(entries)
    prepared.kv_inputs = tuple(kv_entries)


def _slot_tables(
    batch: Batch, tables: BlockTables, slot: int
) -> tuple[GroupTable, ...]:
    """Return a slot's table of every cache group for this batch.

    A block table supplied by the batch takes precedence over the table an
    earlier batch installed.

    Raises:
        WorkerError: ``invalid_descriptor`` when a group has neither.
    """
    supplied = {
        table.group_id: table
        for table in batch.block_tables
        if table.request_pool_idx == slot
    }
    result = []
    for group, shape in enumerate(tables.groups):
        table = supplied.get(group)
        result.append(
            tables.table(slot, group)
            if table is None
            else GroupTable(
                shape,
                int(table.start_page),
                tuple(int(unit) for unit in table.unit_ids),
                int(table.allocated_tokens),
            )
        )
    return tuple(result)


def prepare_inputs(
    state: BatchState,
    *,
    kv_cache: KVCacheManager | None,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    output_pool: OutputPool,
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelExecutor,
    transfer_backends: Mapping[str, Transport],
    config: WorkerConfig,
) -> None:
    """Reserve transfer destinations, start their fetches, and stage predicates.

    The executor calls this once per batch after `prepare_batch`; when the
    batch has transferred inputs, the executor first waits for the batch's
    storage dependencies. Each cross-call input descriptor is validated
    against its declared product before a destination is reserved; on
    failure, every input reserved so far is released before the error
    propagates.

    Inputs read in place are recorded in `state.borrowed_inputs` instead of
    imported: a video encode's input held in a shared-memory segment on this
    node, and every transferred input of a mux call. Device and encoder
    products are imported through `TensorStore.import_tensor` into
    `state.tensor_reads`, latent products into reserved `LatentPool` pages
    (`state.latent_imports`), and KV publications through
    `KVCacheManager.prepare_install` (`state.cache_imports`).
    """
    from uniserve_worker.execution import transfer

    batch = state.batch
    entries, kv_entries = state.input_products, state.kv_inputs
    transports = transfer_backends
    if (entries or kv_entries) and not transports:
        raise unsupported_setup(
            "cross-call input requires a configured transport"
        )

    # A video encode borrows a local decoder's shared-storage segment in place.
    # A decoder on another host publishes through the rank channel instead;
    # that value must follow the ordinary import path so execution can stage a
    # local codec input. The choice is made from the publication's physical
    # locations (a shared-memory location on this node), so the same path
    # applies to every placement.
    borrowed_candidates = {
        product.buffer_id
        for call in batch.calls
        if call.kind in BORROWED_INPUT_CALLS
        for product in call.inputs
    }
    shm = transports.get("shm")
    borrowed = {
        entry.product.buffer_id
        for entry in entries
        if entry.product.buffer_id in borrowed_candidates
        and isinstance(
            entry.value, (DeviceProductTransferValue, EncoderTransferValue)
        )
        and any(
            isinstance(location.transport, PosixShmTransfer)
            and shm is not None
            and location.source.node == shm.source.node
            for location in entry.value.tensor.locations
        )
    }
    state.borrowed_inputs.update(borrowed)
    try:
        for entry in entries:
            assert transports
            if entry.product.buffer_id in borrowed:
                continue

            # One transferred product must land on exactly one consumer device:
            # the compute device (`call_devices(...)[0]`) of its consumers.
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

            # Encoded rows publish initialized prefixes within their reserved
            # capacity. The mux reads those host locations directly; importing
            # the full logical tensor would require uninitialized padding.
            if any(
                call.kind is MediaCall.MUXING and entry.product in call.inputs
                for call in batch.calls
            ):
                state.borrowed_inputs.add(entry.product.buffer_id)
                continue

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
                        and (
                            location.backend != "shm"
                            or location.source.node
                            == transports[location.backend].source.node
                        )
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

                # Record the reservation before fetching so that `close_inputs`
                # abandons it if `fetch_tensor` raises; each started copy is
                # retained on the import through `retain_transfer`.
                state.latent_imports[entry.product.buffer_id] = binding
                from uniserve_worker.transport.fetch import fetch_tensor

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

            # New units of the destination tables are zeroed by the import
            # itself before its copy; `_bind_cache_tables` skips them.
            initialized = tuple(
                unit
                for allocation in batch.new_cache_units
                if allocation.request_pool_idx == slot
                for unit in allocation.unit_ids
            )
            write = publications.prepare_install(
                kv_transfer,
                request_pool_idx=slot,
                tables=_slot_tables(batch, tables, slot),
                initialized_units=initialized,
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
    model_runner: ModelExecutor,
) -> None:
    """Capture completion-valued (U8) predicates into one completion buffer.

    Each such call owns one row of the buffer. Local sources are
    consumed and captured now; transferred sources (those with a prepared
    `state.tensor_reads` import) are recorded in `state.predicate_transfers`
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

        # Sealing forbids further captures, so seal now only when no
        # transferred row remains for `capture_predicates`.
        sealed = not pending
        if sealed:
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
    state.predicate_buffer = buffer
    state.predicate_entries = captures
    state.predicate_transfers = tuple(pending)
    state.predicates_sealed = sealed


def capture_predicates(state: BatchState, tensor_store: TensorStore) -> None:
    """Submit transferred predicate copies on the worker execution thread.

    Does nothing when there is no predicate buffer, it is already sealed, or
    any transferred predicate source is not yet ready; the executor calls it
    again on later advances. On failure the predicate buffer is abandoned.
    """
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
    media_mux: MediaMux | None,
    output_pool: OutputPool,
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelExecutor,
    transfer_backends: Mapping[str, Transport],
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
    Sets `state.registered` on success. If creating the records fails after
    the completion buffer is acquired, the buffer is abandoned; a failure
    after the records are bound to `state` discards the batch through
    `discard_batch`. The error propagates in both cases.
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
            # A standalone capability's batch runs on the stream
            # `ModelExecutor.call_stream` selects for its first active call.
            # Request slots are initialized on the device's current stream,
            # so the batch stream waits for it; independent components publish
            # their own producer/consumer fences.
            stream.wait_stream(torch.cuda.current_stream(stream.device))
            state.stream = stream

    with state.scope():
        started = time.perf_counter_ns()

        # Restrict input payloads to identities declared by this batch.
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
            # The completion buffer and the records bound to it are owned
            # together: a failure before `state.bind_outputs` abandons any
            # acquired buffer here, and a later one is handled by
            # `discard_batch`.
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
            candidates = create_outputs(
                request_pool, scheduled, request_pool_indices, completion
            )
        except BaseException:
            if completion is not None:
                completion.abandon()
            raise

        assert completion is not None
        for request in candidates:
            if calls.call_identity(request.call) in predicated:
                request.status = CallStatus.PREDICATED
        state.bind_outputs(candidates, completion, started)

        try:
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
                identities = {
                    calls.call_identity(call) for call in active_calls
                }
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
                        kv_cache=kv_cache,
                        request_tables=request_tables,
                        state=state,
                    )
                _bind_latent_inputs(
                    active_calls,
                    latent_pool=latent_pool,
                    model_runner=model_runner,
                    state=state,
                )
            _reserve_outputs(
                scheduled,
                tensor_store=tensor_store,
                model_runner=model_runner,
                state=state,
            )

            # Only live calls consume inputs; predicated calls instead publish
            # false into their completion and transition outputs
            # (`_publish_predicated_outputs`).
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
                kv_cache=kv_cache,
                tensor_store=tensor_store,
                latent_pool=latent_pool,
                model_runner=model_runner,
                state=state,
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
            state.registered = True
            record_component(state.component_us, "open_lane", started)
            return None
        except BaseException:
            discard_batch(
                kv_cache=kv_cache,
                tensor_store=tensor_store,
                latent_pool=latent_pool,
                media_mux=media_mux,
                transfer_backends=transfer_backends,
                state=state,
            )
            raise


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
    publication owner (`WorkerInfo.output_rank`), so other ranks reserve
    nothing for it; image decoding reserves on every rank that runs it.

    The slots are stored in the call's `PendingOutput.host.tasks`. A failure
    abandons the current call's reservations; those of earlier calls stay
    with their records for `discard_batch` to release.
    """
    for call in scheduled:
        if call.kind not in {
            MediaCall.IMAGE_DECODING,
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
        if pending.host.tasks:
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
        pending.host.tasks = tuple(reservations)


def validate_batch(
    batch: Batch,
    *,
    worker_info: WorkerInfo,
    model_runner: ModelExecutor,
    config: WorkerConfig,
    predecessors: Mapping[CallId, CallId | None],
) -> None:
    """Validate batch resources, routing, and call support before staging.

    Runs before the batch's `Finish` and `Free` commands are applied; `Start`
    commands are applied first so that `predecessors` covers requests the
    batch admits. Checks that buffer allocations fit the worker buffer pool,
    that every call targets a component placed on this rank (when the worker
    declares components) and a supported call kind, the batch call limit,
    the upper bound of request-slot indices, and the video preparation
    ordering checked by `uniserve_worker.execution.media.validate_batch`.
    Violations raise `invalid_descriptor` or `unsupported_call` errors.
    """
    invalid_buffer = next(
        (
            params
            for params in batch.buffer_allocations
            if params.offset + params.bytes > worker_info.buffer_pool_bytes
        ),
        None,
    )
    if invalid_buffer is not None:
        raise invalid_descriptor(
            "batch buffer params exceeds the worker buffer pool: "
            f"offset={invalid_buffer.offset}, bytes={invalid_buffer.bytes}, "
            f"capacity={worker_info.buffer_pool_bytes}, "
            f"buffer={invalid_buffer.buffer}"
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
        if variant not in worker_info.supported_calls:
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
    # `discard_batch` abandons them with the rest of the batch.
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
                    tensor_store.publish_scalar_write(write, False)


def _bind_latent_inputs(
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    latent_pool: LatentPool | None,
    model_runner: ModelExecutor,
) -> None:
    """Validate trajectory parameters and bind rank-local latent staging.

    Only the params of the given (active) calls are considered. With an
    image builder, the params are checked against the admitted image
    dimensions and the committed solver step, and every row receives a
    `LatentPool.stage` view in `latent.staging`. Without one, the params are
    validated by `_validate_sample_params` and only recorded. Raises
    `invalid_descriptor` errors on any disagreement.
    """
    identities = {calls.call_identity(call) for call in scheduled}
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
            state.pending_output(call.request_key.request_id),
        )
        for call in scheduled
    }

    if pool is None:
        raise invalid_descriptor("latent params require a resident latent pool")

    # Image trajectories bind validated image dimensions and page ownership
    # and borrow the pool's step staging. A standalone denoiser's runner
    # stages its own steps, so its calls keep only their parameters.
    rows: list[tuple[CallIdentity, LatentParams, int]] = []
    for params in parameters:
        identity = (params.request_key, params.call_id)
        selected = requests.get(identity)
        if selected is None:
            raise invalid_descriptor(
                "latent params names a call outside its batch"
            )
        call, request = selected
        slot = int(request.request.request_pool_idx)

        if model_runner.image_builder is None:
            _validate_sample_params(call, request, params, model_runner)
            request.latent.input_params = params
            continue

        image = request.request.image
        if image is None:
            raise invalid_descriptor(
                "latent params has no admitted image dimensions"
            )
        flow = model_runner.image_builder
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
    if not rows:
        return

    # Stage every page table together so overlapping physical ownership is
    # rejected before any call receives a writable tensor view.
    staged = pool.stage(
        tuple(params.page_table for _identity, params, _slot in rows),
        tuple(int(params.latent_units) for _identity, params, _slot in rows),
        occupied=tuple(
            output.latent.staging
            for output in state.outputs
            if isinstance(output, PendingOutput)
            and output.latent.staging is not None
        ),
    )

    for (identity, params, slot), value in zip(rows, staged, strict=True):
        request = state.pending_output(identity[0].request_id)
        request.latent.input_params = params
        request.latent.staging = value


def _validate_sample_params(
    call: Call,
    request: PendingOutput,
    params: LatentParams,
    model_runner: ModelExecutor,
) -> None:
    """Validate a standalone denoiser's trajectory parameters.

    The request's pages are the run its slot owns, and each call covers the
    fixed step interval of its kind: preparation opens the trajectory at step
    zero and each denoising call advances one step from the committed one.
    """
    builder = model_runner.media_builder
    if builder is None:
        raise invalid_descriptor("latent params has no denoiser to advance")
    slot = int(request.request.request_pool_idx)
    if (
        tuple(params.page_table) != builder.slot_pages(slot)
        or int(params.latent_units) != builder.sample_pages.units
    ):
        raise invalid_descriptor(
            "latent params does not name the pages of its request slot"
        )

    step = int(calls.require_progress(request).flow_step)
    if call.kind is MediaCall.LATENT_PREPARATION:
        valid = int(params.start_step) == 0 and int(params.step_count) == 0
    elif call.kind is MediaCall.DENOISING:
        valid = int(params.start_step) == step and int(params.step_count) == 1
    else:
        valid = int(params.start_step) == step and int(params.step_count) == 0
    if not valid:
        raise invalid_descriptor(
            "latent params disagrees with the request's committed step"
        )


def _bind_cache_tables(
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    request_tables: BlockTables | None,
) -> None:
    """Install scheduler tables and retain row-aligned forward coordinates.

    For every slot the active calls read or write, including the
    alternative-prefix slots of their forward rows, the batch's block tables
    are validated and installed in `BlockTables`, and new cache units that no
    KV import initializes are zeroed. Each call's forward rows are then
    recorded in `state.forward_indices`, the main row's prefix is checked
    against the call's projected visible KV length, and the units each row
    accesses in every group are retained in the `KVCacheManager` until the
    batch's output buffer completes.
    """
    cache = kv_cache
    page_tables = request_tables
    if cache is None or page_tables is None:
        raise invalid_descriptor("cache tables require physical KV storage")

    started = time.perf_counter_ns()
    inputs = state.batch
    identities = {calls.call_identity(call) for call in scheduled}

    # Tables are needed for every slot this batch reads or writes: its own
    # requests plus any alternative-prefix rows of its forward calls.
    slots = {
        state.pending_output(
            call.request_key.request_id
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

    # `BlockTables.install` checks each table's page shape and allocated
    # extent against its group.
    tables = []
    for table in inputs.block_tables:
        if table.request_pool_idx not in slots:
            continue
        tables.append(
            (
                int(table.request_pool_idx),
                cache.validate_group(table.group_id),
                int(table.start_page),
                cache.validate_units(table.unit_ids),
                int(table.allocated_tokens),
            )
        )
    page_tables.install(tuple(tables))

    for allocation in inputs.new_cache_units:
        if allocation.request_pool_idx not in slots:
            continue
        units = cache.validate_units(allocation.unit_ids)
        installed = page_tables.table(
            allocation.request_pool_idx,
            cache.validate_group(allocation.group_id),
        )
        if not set(units).issubset(installed.units):
            raise invalid_descriptor(
                "new cache units are outside the installed block table"
            )

        # A KV import zeroes the new units it covers before copying into them
        # (`CacheImport.initialized_units`); every other new unit is zeroed
        # here so stale cache content is never read.
        initialized = {
            unit
            for write in state.cache_imports.values()
            if write.request_pool_idx == allocation.request_pool_idx
            for unit in write.initialized_units
        }
        cache.zero_units(
            tuple(unit for unit in units if unit not in initialized)
        )

    # Forward rows index `inputs.calls`, the full batch including predicated
    # calls, so rows are mapped by call identity rather than by position in
    # `scheduled`.
    rows_by_call: dict[CallIdentity, list[int]] = defaultdict(list)
    for row, index in enumerate(inputs.forward_call_indices):
        identity = calls.call_identity(inputs.calls[index])
        rows_by_call[identity].append(row)

    for call in scheduled:
        request = state.pending_output(call.request_key.request_id)
        main_slot = int(request.request.request_pool_idx)
        call_rows = rows_by_call.get(calls.call_identity(call), [])
        state.forward_indices[calls.call_identity(call)] = tuple(call_rows)

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
            if slot != main_slot and (
                inputs.seq_lens[descriptor] - inputs.query_lens[descriptor]
            ) > page_tables.allocated_length(slot):
                raise invalid_descriptor(
                    "forward row exceeds alternative-prefix capacity"
                )
            if slot != main_slot:
                page_tables.retain_prefix(call.request_key, slot)

            # Rows that write KV retain their prefix plus the query tokens
            # they write (`seq_lens`); read-only rows retain only the
            # `seq_lens - query_lens` prefix. Each group's retention starts
            # at its table's first held page and lasts until the batch's
            # completion future resolves.
            length = inputs.seq_lens[descriptor] - (
                0
                if inputs.write_kv[descriptor]
                else inputs.query_lens[descriptor]
            )
            for group in range(len(page_tables.groups)):
                cache.retain_execution(
                    call.request_key,
                    page_tables.table(slot, group),
                    length=length,
                    completion=state.output_buffer.completion_future(),
                )

    record_component(state.component_us, "bc_tables", started)


def _stage_input_products(
    input_products: Sequence[TensorPublication],
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    model_runner: ModelExecutor,
) -> None:
    """Publish query-ready transferred values into their owning stores.

    Every KV import the batch consumes must be complete and agree with any
    resident publication of its buffer, and every other non-borrowed input
    must be query-ready; violations raise `invalid_descriptor`. A latent
    import is adopted by the `LatentPool` as a live trajectory at its
    transferred step, and the destination's projected `flow_step` moves to
    that step. A device or encoder import is completed in the `TensorStore`.
    Borrowed inputs are skipped.
    """
    # KV imports consumed by this batch must be complete and conflict-free
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
        if product.buffer_id in state.borrowed_inputs:
            continue
        if not state.input_ready(product.buffer_id):
            raise invalid_descriptor(
                "cross-call input has no query-ready prepared transfer"
            )

        # Transfer metadata determines which runtime owns the imported value;
        # each branch validates identity and shape before publication.
        value = entry.value
        if isinstance(value, LatentTransferValue):
            consumers = tuple(
                call
                for call in state.batch.calls
                if product in call.tensor_inputs()
            )
            if len(consumers) != 1:
                raise invalid_descriptor(
                    "latent transfer must have one batch consumer"
                )

            row = state.pending_output(consumers[0].request_key.request_id)
            params = row.latent.input_params
            staging = row.latent.staging
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

            request = state.pending_output(product.request_key.request_id)
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
            row.latent.imported = True
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
                "transferred product has no batch consumer"
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
