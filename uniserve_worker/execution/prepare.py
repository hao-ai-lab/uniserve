"""Validate physical runs and reserve their input, request, and output resources."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from uniserve_worker.execution import operations as operation_geometry
from uniserve_worker.execution.batch import (
    AttentionRegime,
    Computation,
    ComputationId,
    DeviceProductTransferValue,
    DType,
    EncoderTransferValue,
    ForwardMode,
    KvTransfer,
    LatentParams,
    LatentTransferValue,
    PipelineStage,
    Run,
    RunLane,
    ScheduledRequest,
    ShapeBound,
    TensorPublication,
    TensorRef,
    TensorTransfer,
    TransferMode,
)
from uniserve_worker.execution.commit import _discard_lane
from uniserve_worker.execution.operations import _completion_devices, _operation_device
from uniserve_worker.execution.output import OutputBuffer, TokenCapture
from uniserve_worker.execution.retirement import _apply_release_controls
from uniserve_worker.execution.rows import (
    LaneLayout,
    LaneState,
    LatentExecution,
    OperationIdentity,
    PreparedExecution,
    PreparedPredicateBatch,
    PreparedTransferInput,
)
from uniserve_worker.execution.video import require_media_output_ring
from uniserve_worker.execution.video import validate_batch as validate_video_batch
from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    unsupported_operation,
    unsupported_setup,
)
from uniserve_worker.models.video import VideoModel
from uniserve_worker.profiling import record_component
from uniserve_worker.runtime.cache_transfer import CacheWrite
from uniserve_worker.runtime.device_products import (
    DeviceProductImport,
    DeviceProductMetadata,
    DeviceProductRead,
    DeviceProductWrite,
    ImageRange,
)
from uniserve_worker.runtime.encoder_cache import EncoderMetadata, EncoderWrite
from uniserve_worker.runtime.latent_pool import LatentWrite, require_latent_pool
from uniserve_worker.transfer.tickets import TransferTicket

if TYPE_CHECKING:
    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.execution.output import OutputPool
    from uniserve_worker.execution.video import VideoMuxCoordinator, VideoOutputRing
    from uniserve_worker.models.runtime import ExecutionModel
    from uniserve_worker.runtime.cache_pool import CachePool
    from uniserve_worker.runtime.cache_publications import CachePublications
    from uniserve_worker.runtime.cpu import CpuPool
    from uniserve_worker.runtime.device_products import DeviceProducts
    from uniserve_worker.runtime.encoder_cache import EncoderCache
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.req_to_token_pool import ReqToTokenPool
    from uniserve_worker.runtime.request import RequestPool
    from uniserve_worker.runtime.runtime_states import RuntimeStates
    from uniserve_worker.transfer.publications import TransferPublications
    from uniserve_worker.transfer.tickets import Transport


logger = logging.getLogger(__name__)

SAMPLING_COMPLETION_FIELDS = 4
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1


def plan_run(
    batch: Run,
    *,
    worker_info: WorkerInfo,
    model_runner: ModelRunner,
) -> Run:
    """Derive worker-local execution lanes from a flat physical run."""

    if batch.lanes or not batch.operations:
        return batch
    if any(
        params.offset + params.bytes > worker_info.buffer_pool_bytes
        for params in batch.buffer_allocations
    ):
        raise invalid_descriptor("run buffer params exceeds the worker buffer pool")
    grouped: dict[tuple[Computation, str], list[tuple[int, ScheduledRequest]]] = {}
    for index, operation in enumerate(batch.operations):
        grouped.setdefault((operation.kind, operation.entry), []).append((index, operation))
    lanes: list[RunLane] = []
    for lane_id, ((_kind, _entry), members) in enumerate(grouped.items(), start=1):
        global_to_local = {
            global_index: local_index
            for local_index, (global_index, _operation) in enumerate(members)
        }
        member_operations = tuple(operation for _index, operation in members)
        identities = {(operation.request_key, operation.op_id) for operation in member_operations}
        rows = tuple(
            index
            for index, operation in enumerate(batch.forward_operation_indices)
            if operation in global_to_local
        )
        request_slots = {batch.request_pool_indices[index] for index in rows}
        attention = (
            AttentionRegime.CAUSAL
            if all(
                operation.kind in {ForwardMode.PREFILL, ForwardMode.DECODE, ForwardMode.VERIFY}
                for operation in member_operations
            )
            else AttentionRegime.HYBRID
            if any(operation.kind is PipelineStage.DENOISING for operation in member_operations)
            else AttentionRegime.NONE
        )
        lanes.append(
            RunLane(
                lane_id=lane_id,
                launch_id=lane_id,
                collective_seq=batch.collective_seq,
                route=0,
                attention=attention,
                shape_class=0,
                operations=member_operations,
                block_tables=tuple(
                    table
                    for table in batch.block_tables
                    if int(table.request_pool_idx) in request_slots
                ),
                new_cache_pages=tuple(
                    allocation
                    for allocation in batch.new_cache_pages
                    if int(allocation.request_pool_idx) in request_slots
                ),
                forward_operation_indices=tuple(
                    global_to_local[batch.forward_operation_indices[index]] for index in rows
                ),
                request_pool_indices=tuple(batch.request_pool_indices[index] for index in rows),
                seq_lens=tuple(batch.seq_lens[index] for index in rows),
                query_lens=tuple(batch.query_lens[index] for index in rows),
                write_kv=tuple(batch.write_kv[index] for index in rows),
                latent_params=tuple(
                    params
                    for params in batch.latent_params
                    if (params.request_key, params.op_id) in identities
                ),
                decode_ranges=tuple(
                    params
                    for params in batch.decode_ranges
                    if (params.request_key, params.op_id) in identities
                ),
                buffer_allocations=tuple(
                    params
                    for params in batch.buffer_allocations
                    if any(
                        product.buffer_id == params.buffer
                        for operation in member_operations
                        for product in (
                            *operation.tensor_inputs(),
                            *operation.tensor_outputs(),
                            *((operation.predicate,) if operation.predicate is not None else ()),
                        )
                    )
                ),
            )
        )
    return replace(batch, lanes=model_runner.plan_launches(tuple(lanes)))


def prepare_batch(
    batch: Run,
    *,
    cache_pool: CachePool | None,
    cache_registry: CachePublications | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    execution_model: ExecutionModel,
    output_pool: OutputPool,
    request_tables: ReqToTokenPool | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    runtime_states: RuntimeStates | None,
    transfer_publications: TransferPublications,
    transfer_backends: Mapping[str, Transport],
    config: WorkerConfig,
) -> PreparedExecution:
    """Submit bounded transfer and predicate observations without waiting."""

    _apply_batch_controls(
        batch,
        cache_pool=cache_pool,
        cache_registry=cache_registry,
        device_products=device_products,
        encoder_cache=encoder_cache,
        worker_info=worker_info,
        latent_pool=latent_pool,
        execution_model=execution_model,
        request_pool=request_pool,
        model_runner=model_runner,
        runtime_states=runtime_states,
        transfer_publications=transfer_publications,
        config=config,
    )
    storage_dependencies: list[Future[None]] = []
    pool = latent_pool
    if pool is not None:
        admissions = {
            admission.request_key: admission.request_pool_idx for admission in batch.admissions
        }
        operations = {
            (operation.request_key, operation.op_id): operation for operation in batch.operations
        }
        for latent_params in batch.latent_params:
            operation = operations[(latent_params.request_key, latent_params.op_id)]
            if operation.kind not in {PipelineStage.LATENT_PREPARATION, PipelineStage.DENOISING}:
                continue
            request = request_pool.peek(latent_params.request_key.request_id)
            request_slot = (
                admissions.get(latent_params.request_key)
                if request is None
                else request.request_pool_idx
            )
            if request_slot is None:
                raise invalid_descriptor("latent write has no admitted request slot")
            storage_dependencies.extend(
                pool.write_dependencies(request_slot, latent_params.page_table)
            )

    entries = list(batch.input_products)
    kv_entries = list(batch.kv_inputs)
    supplied = {publication.source for publication in kv_entries}
    for operation in batch.operations:
        if operation.kind is not TransferMode.KV_INSTALL:
            continue
        source = operation.kv_input
        if source is None:
            raise invalid_descriptor("KV installation requires a source publication")
        if source not in supplied:
            if cache_registry is None:
                raise invalid_descriptor("KV installation requires cache publication storage")
            kv_entries.append(cache_registry.publication(source))
            supplied.add(source)
    cache = cache_pool
    tables = request_tables
    if cache is not None and tables is not None and cache.has_pending_accesses:
        request_slots = {
            admission.request_key: admission.request_pool_idx for admission in batch.admissions
        }
        kv_inputs = {publication.source: publication for publication in kv_entries}
        for lane in batch.lanes:
            assigned = {
                (table.request_pool_idx, table.group_id): table.page_ids
                for table in lane.block_tables
            }

            def pages_for(slot: int, group: int) -> tuple[int, ...]:
                pages = assigned.get((slot, group))
                return tables.pages(slot, group) if pages is None else pages

            for allocation in lane.new_cache_pages:
                storage_dependencies.extend(
                    cache.write_dependencies(
                        allocation.page_ids,
                        group=allocation.group_id,
                        start=0,
                        length=len(allocation.page_ids) * cache.block_size,
                    )
                )
            for row, write_kv in enumerate(lane.write_kv):
                if not write_kv:
                    continue
                storage_dependencies.extend(
                    cache.write_dependencies(
                        pages_for(lane.request_pool_indices[row], 0),
                        group=0,
                        start=lane.seq_lens[row] - lane.query_lens[row],
                        length=lane.query_lens[row],
                    )
                )
            for operation in lane.operations:
                if operation.kind is not TransferMode.KV_INSTALL:
                    continue
                request = request_pool.peek(operation.request_key.request_id)
                slot = (
                    request_slots.get(operation.request_key)
                    if request is None
                    else request.request_pool_idx
                )
                if slot is None:
                    raise invalid_descriptor("KV installation has no admitted request slot")
                source = operation.kv_input
                if source is None:
                    raise invalid_descriptor("KV installation requires a source publication")
                kv_publication = kv_inputs.get(source)
                if kv_publication is not None:
                    storage_dependencies.extend(
                        cache.write_dependencies(
                            pages_for(slot, kv_publication.group_id),
                            group=kv_publication.group_id,
                            start=kv_publication.base_extent,
                            length=kv_publication.published_extent - kv_publication.base_extent,
                        )
                    )
    prepared = PreparedExecution(
        batch=batch,
        transfers=(),
        storage_dependencies=tuple(storage_dependencies),
    )
    if entries or kv_entries:
        # Destination addresses may still belong to an earlier physical reader.
        # Its retirement wakes the execution thread, which submits these reads.
        prepared._prepare_inputs = partial(
            _prepare_inputs,
            batch,
            tuple(entries),
            tuple(kv_entries),
            cache_pool=cache_pool,
            cache_registry=cache_registry,
            device_products=device_products,
            encoder_cache=encoder_cache,
            latent_pool=latent_pool,
            output_pool=output_pool,
            request_tables=request_tables,
            request_pool=request_pool,
            model_runner=model_runner,
            transfer_backends=transfer_backends,
            config=config,
        )
        prepared.advance()
    else:
        prepared.transfers, prepared.predicates = _prepare_inputs(
            batch,
            tuple(entries),
            tuple(kv_entries),
            cache_pool=cache_pool,
            cache_registry=cache_registry,
            device_products=device_products,
            encoder_cache=encoder_cache,
            latent_pool=latent_pool,
            output_pool=output_pool,
            request_tables=request_tables,
            request_pool=request_pool,
            model_runner=model_runner,
            transfer_backends=transfer_backends,
            config=config,
        )
    return prepared


def _prepare_inputs(
    batch: Run,
    entries: tuple[TensorPublication, ...],
    kv_entries: tuple[KvTransfer, ...],
    *,
    cache_pool: CachePool | None,
    cache_registry: CachePublications | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    latent_pool: LatentPool | None,
    output_pool: OutputPool,
    request_tables: ReqToTokenPool | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    transfer_backends: Mapping[str, Transport],
    config: WorkerConfig,
) -> tuple[tuple[PreparedTransferInput, ...], PreparedPredicateBatch | None]:
    """Reserve transfer destinations and submit reads after their storage is available."""

    from . import transfer

    transports = transfer_backends
    if (entries or kv_entries) and not transports:
        raise unsupported_setup("cross-stage input requires a configured transport")
    transfers: list[PreparedTransferInput] = []
    try:
        for entry in entries:
            assert transports
            devices = {
                model_runner.operation_device(operation)
                for operation in batch.operations
                if entry.product in operation.tensor_inputs()
                or entry.product == operation.predicate
            }
            if len(devices) != 1:
                raise invalid_descriptor("transferred product requires one consumer device per run")
            device = next(iter(devices))
            value = entry.value
            tensors: tuple[TensorTransfer, ...]
            if isinstance(value, EncoderTransferValue):
                main = value.tensor
                tensors = (main,)
                if (
                    not isinstance(value.payload_kind, str)
                    or value.payload_kind not in {"vision_feature", "latent_feature"}
                    or min(value.height, value.width) < 1
                    or not transfer.tensor_matches_product(main, entry.product)
                ):
                    raise invalid_descriptor("encoder transfer metadata exceeds its product bounds")
                if not any(
                    entry.product
                    == (
                        operation.vision_input
                        if value.payload_kind == "vision_feature"
                        else operation.latent_feature_input
                    )
                    for operation in batch.operations
                ):
                    raise invalid_descriptor(
                        "encoder transfer entry disagrees with its product identity"
                    )
            elif isinstance(value, DeviceProductTransferValue):
                main = value.tensor
                tensors = (main,)
                if min(value.height, value.width) < 0:
                    raise invalid_descriptor("device-product image geometry must be non-negative")
                if (value.height == 0) != (value.width == 0):
                    raise invalid_descriptor("device-product image geometry is incomplete")
                if value.value_range not in {"", *(member.value for member in ImageRange)}:
                    raise invalid_descriptor("device-product value range is invalid")
                if value.height == 0 and value.value_range:
                    raise invalid_descriptor("non-image device product carries an image range")
                if not any(
                    entry.product
                    in (
                        *operation.inputs,
                        operation.token_input,
                        operation.image_input,
                        operation.predicate,
                    )
                    for operation in batch.operations
                ) or not transfer.tensor_matches_product(main, entry.product):
                    raise invalid_descriptor(
                        "device-product transfer metadata exceeds its product bounds"
                    )
            elif isinstance(value, LatentTransferValue):
                main = value.tensor
                tensors = (main,)
                pool = latent_pool
                expected_dtype = "" if pool is None else str(pool.dtype).removeprefix("torch.")
                expected_nbytes = (
                    0
                    if pool is None
                    else value.latent_units
                    * int(pool.latent_width)
                    * int(pool.storage.element_size())
                )
                if (
                    not any(
                        entry.product == operation.latent_input for operation in batch.operations
                    )
                    or pool is None
                    or min(value.height, value.width, value.latent_units) < 1
                    or tuple(main.shape) != (value.latent_units, int(pool.latent_width))
                    or main.dtype != expected_dtype
                    or main.nbytes != expected_nbytes
                    or main.nbytes > entry.product.max_bytes
                    or math.prod(main.shape) > entry.product.shape_bound.max_elements
                ):
                    raise invalid_descriptor("latent transfer metadata exceeds its product bounds")
            else:
                raise invalid_descriptor("cross-stage transfer entry has an unknown kind")
            binding: DeviceProductWrite | EncoderWrite | LatentWrite | None = None
            discard: Callable[[], None] | None = None
            target: torch.Tensor | tuple[torch.Tensor, ...] | None = None
            parameters = {params.buffer: params for params in batch.buffer_allocations}
            if isinstance(value, EncoderTransferValue):
                binding = encoder_cache.bind_outputs(
                    ((entry.product, device),),
                    buffer_allocations=parameters,
                )[0]
                discard = partial(encoder_cache.abandon_writes, (binding,))
                target = binding.buffer_binding.tensor
            elif isinstance(value, DeviceProductTransferValue):
                request_slots = {
                    admission.request_key: int(admission.request_pool_idx)
                    for admission in batch.admissions
                }
                resident = request_pool.peek(entry.product.request_key.request_id)
                if resident is not None and resident.request_key == entry.product.request_key:
                    request_slots[resident.request_key] = int(resident.request_pool_idx)
                imported = device_products.import_tensor(
                    entry.product,
                    value.tensor,
                    device=device,
                    request_slots=request_slots,
                    buffer_allocations=parameters,
                    bindings={
                        (location.source, location.backend): transports[location.backend]
                        for location in value.tensor.locations
                        if location.backend in transports
                    },
                    metadata=(
                        None
                        if value.height == 0
                        else DeviceProductMetadata(
                            height=value.height,
                            width=value.width,
                            value_range=None
                            if not value.value_range
                            else ImageRange(value.value_range),
                        )
                    ),
                )
                transfers.append(
                    PreparedTransferInput(
                        buffer=entry.product.buffer_id,
                        value=value,
                        tickets=imported.tickets,
                        buffers=(imported.tensor,),
                        destination=imported,
                        _discard_destination=imported.close,
                    )
                )
                continue
            elif isinstance(value, LatentTransferValue):
                consumers = tuple(
                    operation
                    for operation in batch.operations
                    if entry.product in operation.tensor_inputs()
                )
                if len(consumers) != 1:
                    raise invalid_descriptor("latent transfer must have one consumer")
                consumer = consumers[0]
                params = next(
                    (
                        params
                        for params in batch.latent_params
                        if (params.request_key, params.op_id)
                        == (consumer.request_key, consumer.op_id)
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
                    raise invalid_descriptor("latent transfer disagrees with its scheduler params")
                resident = request_pool.peek(entry.product.request_key.request_id)
                admission = next(
                    (
                        row
                        for row in batch.admissions
                        if row.request_key == entry.product.request_key
                    ),
                    None,
                )
                if resident is not None and resident.request_key == entry.product.request_key:
                    slot = int(resident.request_pool_idx)
                elif admission is not None:
                    slot = int(admission.request_pool_idx)
                else:
                    raise invalid_descriptor("latent transfer has no request slot")
                pool = require_latent_pool(latent_pool)
                binding = pool.reserve_import(
                    entry.product,
                    request_pool_idx=slot,
                    page_table=params.page_table,
                    latent_units=value.latent_units,
                )
                discard = partial(pool.abandon_import, binding)
                target = binding.spans
            from ..transfer.layout import fetch_tensor

            tickets: list[TransferTicket] = []
            buffers: list[torch.Tensor | tuple[torch.Tensor, ...]] = []

            def retain(ticket: TransferTicket) -> None:
                if isinstance(binding, EncoderWrite):
                    encoder_cache.retain_transfer(binding, ticket)
                elif isinstance(binding, DeviceProductWrite):
                    device_products.retain_transfer(binding, ticket)
                elif isinstance(binding, LatentWrite):
                    require_latent_pool(latent_pool).retain_transfer(binding, ticket)

            try:
                for tensor in tensors:
                    destination: torch.Tensor | tuple[torch.Tensor, ...]
                    assert target is not None
                    if isinstance(target, torch.Tensor):
                        destination = target.reshape(-1)[: math.prod(tensor.shape)].reshape(
                            tensor.shape
                        )
                    else:
                        destination = target
                    buffers.append(destination)
                    tickets.extend(
                        fetch_tensor(
                            tensor,
                            destination,
                            bindings={
                                (location.source, location.backend): transports[location.backend]
                                for location in tensor.locations
                                if location.backend in transports
                            },
                            retain=retain,
                        )
                    )
                transfers.append(
                    PreparedTransferInput(
                        buffer=entry.product.buffer_id,
                        value=value,
                        tickets=tuple(tickets),
                        buffers=tuple(buffers),
                        destination=binding,
                        _discard_destination=discard,
                    )
                )
            except BaseException:
                for ticket in tickets:
                    ticket.cancel()
                if discard is not None:
                    discard()
                raise
        for kv_transfer in kv_entries:
            consumers = tuple(
                operation
                for operation in batch.operations
                if operation.kv_input == kv_transfer.source
            )
            if len(consumers) != 1 or consumers[0].kind is not TransferMode.KV_INSTALL:
                raise invalid_descriptor("KV input requires one installation consumer")
            publications = cache_registry
            cache = cache_pool
            tables = request_tables
            if publications is None or cache is None or tables is None:
                raise invalid_descriptor("KV input requires physical cache storage")
            resident = request_pool.peek(kv_transfer.source.owner.request_id)
            admission = next(
                (row for row in batch.admissions if row.request_key == kv_transfer.source.owner),
                None,
            )
            if resident is not None and resident.request_key == kv_transfer.source.owner:
                slot = int(resident.request_pool_idx)
            elif admission is not None:
                slot = int(admission.request_pool_idx)
            else:
                raise invalid_descriptor("KV transfer has no admitted request slot")
            table = next(
                (
                    table
                    for lane in batch.lanes
                    for table in lane.block_tables
                    if (table.request_pool_idx, table.group_id) == (slot, kv_transfer.group_id)
                ),
                None,
            )
            pages = tables.pages(slot, kv_transfer.group_id) if table is None else table.page_ids
            allocated = tables.allocated_length(slot) if table is None else table.allocated_tokens
            initialized = tuple(
                page
                for lane in batch.lanes
                for allocation in lane.new_cache_pages
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
            transfers.append(
                PreparedTransferInput(
                    buffer=kv_transfer.source,
                    value=kv_transfer,
                    tickets=(),
                    buffers=(),
                    destination=write,
                    _discard_destination=partial(cache.imports.abandon, write),
                )
            )
        predicates = _prepare_predicates(
            batch,
            transfers=tuple(transfers),
            device_products=device_products,
            output_pool=output_pool,
            config=config,
        )
    except BaseException:
        for prepared_transfer in transfers:
            prepared_transfer.close()
        raise
    return tuple(transfers), predicates


def _prepare_predicates(
    batch: Run,
    *,
    transfers: tuple[PreparedTransferInput, ...],
    device_products: DeviceProducts,
    output_pool: OutputPool,
    config: WorkerConfig,
) -> PreparedPredicateBatch | None:
    """Capture completion-valued predicates from local products or prepared transfers."""

    # Predicate rows occupy one compact completion buffer regardless of whether
    # their source is already local or will arrive through a prepared transfer.
    operations = tuple(
        operation
        for operation in batch.operations
        if operation.predicate is not None and operation.predicate.dtype is DType.U8
    )
    if not operations:
        return None
    transferred = {transfer.buffer: transfer for transfer in transfers}
    buffer = output_pool.acquire(
        len(operations),
        token_capacity=len(operations),
        devices=tuple(_operation_device(operation, config=config) for operation in operations),
    )
    captures: list[tuple[OperationIdentity, TokenCapture, int]] = []
    pending: list[tuple[OperationIdentity, PreparedTransferInput, int]] = []
    recorded: list[DeviceProductRead] = []
    try:
        # Local sources are consumed in device batches and captured directly;
        # transferred sources retain their target row for later completion.
        grouped: dict[torch.device, list[ScheduledRequest]] = defaultdict(list)
        rows = {
            operation_geometry.operation_identity(operation): row
            for row, operation in enumerate(operations)
        }
        for operation in operations:
            transfer = transferred.get(cast(TensorRef, operation.predicate).buffer_id)
            if transfer is None:
                grouped[_operation_device(operation, config=config)].append(operation)
            else:
                pending.append(
                    (
                        operation_geometry.operation_identity(operation),
                        transfer,
                        rows[operation_geometry.operation_identity(operation)],
                    )
                )
        for device, device_operations in grouped.items():
            reads = device_products.consume_batch(
                tuple(
                    (
                        cast(TensorRef, operation.predicate),
                        operation.op_id,
                        device,
                    )
                    for operation in device_operations
                ),
                device=device,
            )
            recorded.extend(reads)
            for operation, read in zip(device_operations, reads, strict=True):
                identity = operation_geometry.operation_identity(operation)
                captures.append((identity, buffer.capture(read.tensor), rows[identity]))
            device_products.record_readers(reads, device=device)

        # No pending transfer can mutate the buffer once it is sealed.
        sealed = not pending
        if sealed:
            buffer.seal()
    except BaseException:
        # Every acquired read must receive a reader event even when preparation
        # fails before all device groups are captured.
        unrecorded = tuple(recorded)
        if unrecorded:
            device_products.record_readers(unrecorded)
        buffer.abandon()
        raise
    return PreparedPredicateBatch(
        buffer=buffer,
        entries=captures,
        transferred=tuple(pending),
        sealed=sealed,
    )


def _apply_batch_controls(
    batch: Run,
    *,
    cache_pool: CachePool | None,
    cache_registry: CachePublications | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    execution_model: ExecutionModel,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    runtime_states: RuntimeStates | None,
    transfer_publications: TransferPublications,
    config: WorkerConfig,
) -> None:
    """Validate and apply ordered controls before reserving asynchronous execution.

    Free can retire a published page bank needed by this run. Applying it before
    the storage wait prevents a dependency on the run's own unprocessed release.
    """

    _validate_batch(
        batch,
        worker_info=worker_info,
        execution_model=execution_model,
        model_runner=model_runner,
        config=config,
    )
    for command in batch.commands:
        started_slots = request_pool.apply_commands((command,))
        if started_slots and runtime_states is not None:
            runtime_states.reset(started_slots)
    _apply_release_controls(
        batch,
        before_execution=True,
        cache_pool=cache_pool,
        cache_registry=cache_registry,
        device_products=device_products,
        encoder_cache=encoder_cache,
        latent_pool=latent_pool,
        transfer_publications=transfer_publications,
    )


def _open_lane(
    batch: Run,
    lane: RunLane,
    prepared: tuple[PreparedTransferInput, ...],
    predicate_values: Mapping[OperationIdentity, bool],
    graph_eligible: bool,
    *,
    cache_pool: CachePool | None,
    cache_registry: CachePublications | None,
    cpu_tasks: CpuPool,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: VideoMuxCoordinator | None,
    media_output_ring: VideoOutputRing | None,
    execution_model: ExecutionModel,
    output_pool: OutputPool,
    request_tables: ReqToTokenPool | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    transfer_backends: Mapping[str, Transport],
    config: WorkerConfig,
) -> LaneState:
    """Stage one lane's speculative state, resources, inputs, and completion storage."""

    operations = lane.operations

    # Predicated rows remain in aligned output/state tables but do not reserve
    # execution-only inputs, CPU tasks, or model resources.
    predicated = frozenset(
        identity
        for operation in operations
        if (identity := operation_geometry.operation_identity(operation)) in predicate_values
        and not predicate_values[identity]
    )
    active_operations = (
        operations
        if not predicated
        else tuple(
            operation
            for operation in operations
            if operation_geometry.operation_identity(operation) not in predicated
        )
    )
    # Restrict input payloads to identities declared by this lane.
    started = time.perf_counter_ns()
    declared_inputs = {
        reference.buffer_id for operation in operations for reference in operation.tensor_inputs()
    }
    declared_inputs.update(
        operation.kv_input for operation in operations if operation.kv_input is not None
    )
    declared_inputs.update(
        operation.predicate.buffer_id for operation in operations if operation.predicate is not None
    )
    input_products = tuple(
        payload for payload in batch.input_products if payload.product.buffer_id in declared_inputs
    )
    completion: OutputBuffer | None = None
    try:
        # Candidate drafts and completion slots form a speculative ownership unit:
        # either all later lane resources bind successfully or both are discarded.
        request_pool_indices = tuple(
            int(request_pool.get(operation.request_key.request_id).request_pool_idx)
            for operation in operations
        )
        candidates = request_pool.stage_lane(operations, request_pool_indices)
        for operation, request in zip(operations, candidates, strict=True):
            request.install_runtime(request.predecessor.runtime)
        completion = output_pool.acquire(
            len(operations),
            token_capacity=_lane_completion_words(operations),
            devices=_completion_devices(operations, config=config),
        )
    except BaseException:
        if completion is not None:
            completion.abandon()
        raise
    assert completion is not None
    scope = LaneState(
        lane=lane,
        started_ns=started,
        graph_eligible=graph_eligible,
        request_candidates=candidates,
        request_rows={request.request.request_id: request for request in candidates},
        completion=completion,
        prepared_transfers={
            transfer.buffer: transfer for transfer in prepared if transfer.buffer in declared_inputs
        },
        predicated_operations=predicated,
    )
    try:
        # Bind physical state in dependency order before decoding transferred inputs.
        _reserve_cpu_tasks(
            active_operations,
            scope,
            cpu_tasks=cpu_tasks,
            worker_info=worker_info,
            media_output_ring=media_output_ring,
            execution_model=execution_model,
            config=config,
        )
        active_lane = _active_lane(lane, active_operations)
        if active_lane is not None:
            if cache_pool is None or request_tables is None:
                if (
                    active_lane.block_tables
                    or active_lane.new_cache_pages
                    or active_lane.forward_operation_indices
                ):
                    raise unsupported_setup(
                        "KV-free execution received cache tables or packed forward rows"
                    )
            else:
                _bind_cache_tables(
                    active_lane, scope, cache_pool=cache_pool, request_tables=request_tables
                )
        # Preserve operation/request row alignment for forward packing and commit.
        scope.layout = LaneLayout(
            operations=operations,
            requests=candidates,
            # Physical row columns bound queued work. A verifier can publish a
            # shorter accepted prefix before this lane becomes executable.
            seq_lens=tuple(
                int(request.predecessor.runtime.kv_visible_len) for request in candidates
            ),
            identities=tuple(
                operation_geometry.operation_identity(operation) for operation in operations
            ),
        )
        if active_lane is not None:
            _bind_latent_rows(
                active_lane, scope, latent_pool=latent_pool, model_runner=model_runner
            )
        _reserve_outputs(
            operations,
            scope,
            device_products=device_products,
            encoder_cache=encoder_cache,
            execution_model=execution_model,
            config=config,
        )

        # Only live operations consume inputs; predicated outputs are published
        # directly into their aligned completion rows.
        active_inputs = {
            reference for operation in active_operations for reference in operation.tensor_inputs()
        }
        active_inputs.update(
            operation.predicate
            for operation in active_operations
            if operation.predicate is not None
        )
        _stage_input_products(
            tuple(payload for payload in input_products if payload.product in active_inputs),
            scope,
            cache_registry=cache_registry,
            encoder_cache=encoder_cache,
            latent_pool=latent_pool,
            config=config,
        )
        _consume_predicates(
            active_operations, scope, device_products=device_products, config=config
        )
        _publish_predicated_outputs(operations, scope, device_products=device_products)
        scope.registration_visible = True
        record_component(scope, "open_lane", started)
        return scope
    except BaseException:
        _discard_lane(
            scope,
            cache_pool=cache_pool,
            device_products=device_products,
            encoder_cache=encoder_cache,
            latent_pool=latent_pool,
            media_mux=media_mux,
            transfer_backends=transfer_backends,
        )
        raise


def _active_lane(
    lane: RunLane,
    operations: tuple[ScheduledRequest, ...],
) -> RunLane | None:
    """Rebuild lane-indexed rows and parameters after predicated operations are removed."""

    if not operations:
        return None
    if operations is lane.operations:
        return lane
    identities = {operation_geometry.operation_identity(operation) for operation in operations}
    old_to_new = {
        index: selected
        for selected, (index, operation) in enumerate(
            (
                item
                for item in enumerate(lane.operations)
                if operation_geometry.operation_identity(item[1]) in identities
            )
        )
    }
    return replace(
        lane,
        operations=operations,
        forward_operation_indices=tuple(
            old_to_new[operation]
            for row, operation in enumerate(lane.forward_operation_indices)
            if operation in old_to_new
        ),
        request_pool_indices=tuple(
            lane.request_pool_indices[row]
            for row, operation in enumerate(lane.forward_operation_indices)
            if operation in old_to_new
        ),
        seq_lens=tuple(
            lane.seq_lens[row]
            for row, operation in enumerate(lane.forward_operation_indices)
            if operation in old_to_new
        ),
        query_lens=tuple(
            lane.query_lens[row]
            for row, operation in enumerate(lane.forward_operation_indices)
            if operation in old_to_new
        ),
        write_kv=tuple(
            lane.write_kv[row]
            for row, operation in enumerate(lane.forward_operation_indices)
            if operation in old_to_new
        ),
        latent_params=tuple(
            params
            for params in lane.latent_params
            if (params.request_key, params.op_id) in identities
        ),
    )


def _lane_completion_words(operations: tuple[ScheduledRequest, ...]) -> int:
    """Compute fixed completion-word capacity for all operations in a lane."""

    return max(
        1,
        SAMPLING_COMPLETION_FIELDS * len(operations)
        + sum((int(operation.bounds.max_completion_bytes) + 3) // 4 for operation in operations),
    )


def _reserve_cpu_tasks(
    operations: tuple[ScheduledRequest, ...],
    scope: LaneState,
    *,
    cpu_tasks: CpuPool,
    worker_info: WorkerInfo,
    media_output_ring: VideoOutputRing | None,
    execution_model: ExecutionModel,
    config: WorkerConfig,
) -> None:
    """Reserve bounded CPU slots for active operations that schedule host-side work."""

    video_model = isinstance(execution_model, VideoModel)
    for operation in operations:
        if operation.kind not in {
            PipelineStage.IMAGE_DECODING,
            PipelineStage.VIDEO_ENCODING,
            PipelineStage.AUDIO_ENCODING,
            PipelineStage.MUXING,
        }:
            continue
        if video_model and config.rank != worker_info.output_rank(operation.entry):
            continue
        identity = operation_geometry.operation_identity(operation)
        if identity in scope.cpu_tasks:
            raise invalid_descriptor("materialization repeats its CPU task identity")
        reservation = cpu_tasks.reserve()
        try:
            if operation.kind in {PipelineStage.VIDEO_ENCODING, PipelineStage.AUDIO_ENCODING}:
                scope.media_output_leases[identity] = require_media_output_ring(
                    media_output_ring
                ).reserve("video" if operation.kind is PipelineStage.VIDEO_ENCODING else "audio")
        except BaseException:
            reservation.abandon()
            raise
        scope.cpu_tasks[identity] = reservation


def _validate_batch(
    batch: Run,
    *,
    worker_info: WorkerInfo,
    execution_model: ExecutionModel,
    model_runner: ModelRunner,
    config: WorkerConfig,
) -> None:
    """Validate run identity, lane resources, routing, and operation support before staging."""

    if len(batch.operations) > config.max_batch_operations:
        raise invalid_descriptor("execution batch exceeds the worker_config operation limit")
    for operation in batch.operations:
        variant = operation.kind
        if variant not in worker_info.supported_ops:
            raise unsupported_operation(variant.value, operation.request_key.request_id)
    if any(
        index > config.max_request_pool_size
        for lane in batch.lanes
        for index in (
            *(table.request_pool_idx for table in lane.block_tables),
            *lane.request_pool_indices,
        )
    ):
        raise invalid_descriptor("execution batch exceeds request-slot capacity")
    model_runner.validate_launches(batch.lanes)
    validate_video_batch(batch, execution_model=execution_model)


def _reserve_outputs(
    operations: tuple[ScheduledRequest, ...],
    scope: LaneState,
    *,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    execution_model: ExecutionModel,
    config: WorkerConfig,
) -> None:
    """Bind each declared device value to its concrete bounded owner."""

    regions = {}
    shapes = {}
    scalar_groups: dict[
        tuple[torch.device, DType, ShapeBound], list[tuple[TensorRef, torch.device | str]]
    ] = {}
    persistent_bindings: list[tuple[TensorRef, torch.device | str]] = []
    encoder_bindings: list[tuple[TensorRef, torch.device | str]] = []
    by_identity = {
        operation_geometry.operation_identity(operation): operation for operation in operations
    }
    for operation in operations:
        device = _operation_device(operation, config=config)
        request = operation_geometry.request_row(scope, operation.request_key.request_id)
        predicated = operation_geometry.operation_identity(operation) in scope.predicated_operations
        if not predicated:
            decode = next(
                (
                    params
                    for params in scope.lane.decode_ranges
                    if params.op_id == operation.op_id
                    and params.request_key == operation.request_key
                ),
                None,
            )
            for output in operation.outputs:
                layout = execution_model.output_layout(
                    operation.entry,
                    output.output_index,
                    request.request.admission.diffusion,
                    decode,
                    len(request.request.admission.prompt_token_ids),
                )
                if layout is None:
                    continue
                if layout.shape is not None:
                    shapes[output] = layout.shape
                if layout.region is not None:
                    regions[output] = layout.region
                persistent_bindings.append((output, device))
            if operation.image_output is not None:
                persistent_bindings.append((operation.image_output, device))
            if operation.encoder_output is not None:
                encoder_bindings.append((operation.encoder_output, device))
        # A skipped computation propagates false predicates but publishes no
        # sampled token, feature, image, or latent state.
        for scalar in (
            operation.token_output,
            operation.completion_output,
            operation.transition_output,
        ):
            if scalar is None or (predicated and scalar == operation.token_output):
                continue
            scalar_groups.setdefault((device, scalar.dtype, scalar.shape_bound), []).append(
                (scalar, device)
            )
    groups = tuple(tuple(group) for group in scalar_groups.values())
    if persistent_bindings:
        groups = (*groups, tuple(persistent_bindings))
    request_slots = {
        request.request.request_key: int(request.request.request_pool_idx)
        for request in scope.request_candidates
    }
    allocations = {params.buffer: params for params in scope.lane.buffer_allocations}
    bound_groups = device_products.bind_output_groups(
        groups,
        regions=regions,
        shapes=shapes,
        request_slots=request_slots,
        buffer_allocations=allocations,
    )
    scope.device_writes.extend(write for binding in bound_groups for write in binding.writes)
    scope.encoder_writes.extend(
        encoder_cache.bind_outputs(tuple(encoder_bindings), buffer_allocations=allocations)
    )
    for write in scope.device_writes:
        identity = operation_geometry.product_identity(write.reference)
        producer = by_identity.get(identity)
        if producer is None:
            raise RuntimeError("device output binding has no computation in the execution batch")
        if write.reference == producer.token_output:
            scope.token_writes[identity] = write
            scope.operation_writes.setdefault(identity, write)
        elif write.reference == producer.transition_output:
            scope.transition_writes[identity] = write
        else:
            scope.operation_writes.setdefault(identity, write)
        if identity in scope.predicated_operations and write.reference in (
            producer.completion_output,
            producer.transition_output,
        ):
            scope.propagated_predicate_writes.setdefault(identity, ())
            scope.propagated_predicate_writes[identity] = (
                *scope.propagated_predicate_writes[identity],
                write,
            )


def _consume_predicates(
    operations: tuple[ScheduledRequest, ...],
    scope: LaneState,
    *,
    device_products: DeviceProducts,
    config: WorkerConfig,
) -> None:
    """Resolve operation predicates from local device products and register their readers."""

    grouped: dict[
        torch.device,
        list[
            tuple[
                ScheduledRequest,
                tuple[TensorRef, ComputationId, torch.device | str | None],
            ]
        ],
    ] = {}
    for operation in operations:
        predicate = operation.predicate
        if predicate is None:
            continue
        device = _operation_device(operation, config=config)
        grouped.setdefault(device, []).append(
            (
                operation,
                (
                    predicate,
                    operation.op_id,
                    device,
                ),
            )
        )
    for device, entries in grouped.items():
        reads = device_products.consume_batch(
            tuple(request for _operation, request in entries),
            device=device,
        )
        scope.device_reads.extend(reads)
        for (operation, _request), read in zip(entries, reads, strict=True):
            predicate = cast(TensorRef, operation.predicate)
            tagged = predicate.dtype is DType.I64
            scope.predicate_values[operation_geometry.operation_identity(operation)] = (
                read.tensor,
                tagged,
            )


def _publish_predicated_outputs(
    operations: tuple[ScheduledRequest, ...], scope: LaneState, *, device_products: DeviceProducts
) -> None:
    """Publish inactive sentinel values for products of predicated operations."""

    declared = {operation_geometry.operation_identity(operation) for operation in operations}
    for identity, writes in scope.propagated_predicate_writes.items():
        if identity not in declared:
            raise RuntimeError("predicated output has no operation in its lane")
        for write in writes:
            device_products.publish_scalar_write(write, False)


def _bind_latent_rows(
    lane: RunLane, scope: LaneState, *, latent_pool: LatentPool | None, model_runner: ModelRunner
) -> None:
    """Validate trajectory parameters and bind rank-local latent staging views."""

    if not lane.latent_params:
        return
    pool = latent_pool
    operations = {
        operation_geometry.operation_identity(operation): (
            operation,
            operation_geometry.request_row(scope, operation.request_key.request_id),
        )
        for operation in lane.operations
    }
    if pool is None:
        # Fixed request tensors own the trajectory directly. Solver progress
        # remains explicit, without a second paged-storage reservation.
        for params in lane.latent_params:
            identity = params.request_key, params.op_id
            selected = operations.get(identity)
            if selected is None:
                raise invalid_descriptor("latent params names an operation outside its lane")
            operation, request = selected
            if params.page_table or params.latent_units:
                raise invalid_descriptor("paged latent params require a resident latent pool")
            if operation.kind is PipelineStage.LATENT_PREPARATION:
                valid = int(params.start_step) == 0 and int(params.step_count) == 0
            elif operation.kind is PipelineStage.DENOISING:
                valid = (
                    int(params.start_step) == int(request.flow_step) and int(params.step_count) == 1
                )
            else:
                valid = (
                    int(params.start_step) == int(request.flow_step) and int(params.step_count) == 0
                )
            if not valid:
                raise invalid_descriptor(
                    "pool-free latent params disagrees with resident generation state"
                )
        return
    # Pooled models bind each operation to validated image geometry and page ownership.
    rows: list[tuple[OperationIdentity, LatentParams, int]] = []
    for params in lane.latent_params:
        identity = (params.request_key, params.op_id)
        selected = operations.get(identity)
        if selected is None:
            raise invalid_descriptor("latent params names an operation outside its lane")
        operation, request = selected
        slot = int(request.request.request_pool_idx)
        image = request.request.image
        if image is None:
            raise invalid_descriptor("latent params has no admitted image geometry")
        flow = model_runner.generation()
        expected_units = int(flow.image_tokens(int(params.height), int(params.width)))
        if (
            int(params.height) != int(image.height)
            or int(params.width) != int(image.width)
            or int(params.latent_units) != expected_units
        ):
            raise invalid_descriptor("latent params disagrees with admitted model geometry")
        transferred = next(
            (
                prepared.value
                for reference in operation.tensor_inputs()
                if (prepared := scope.prepared_transfers.get(reference.buffer_id)) is not None
                and isinstance(prepared.value, LatentTransferValue)
            ),
            None,
        )
        committed_step = int(request.flow_step) if transferred is None else transferred.step
        if operation.kind is PipelineStage.LATENT_PREPARATION:
            if int(params.start_step) != 0 or int(params.step_count) != 0:
                raise invalid_descriptor("media preparation params carries denoise steps")
        elif operation.kind is PipelineStage.DENOISING:
            if (
                int(params.start_step) != committed_step
                or int(params.step_count) < 1
                or int(params.start_step) + int(params.step_count) > int(image.steps)
                or (
                    int(operation.bounds.max_tokens) > 0
                    and int(params.step_count) > int(operation.bounds.max_tokens)
                )
            ):
                raise invalid_descriptor("media denoise params exceeds its committed schedule")
        elif int(params.start_step) != committed_step or int(params.step_count) != 0:
            raise invalid_descriptor("latent reader params disagrees with committed step state")
        rows.append((identity, params, slot))
    # Stage every page table together so overlapping physical ownership is
    # rejected before any operation receives a writable tensor view.
    staged = pool.stage(
        tuple(params.page_table for _identity, params, _slot in rows),
        tuple(int(params.latent_units) for _identity, params, _slot in rows),
    )
    scope.latent_rows = {
        identity: LatentExecution(
            params=params,
            request_pool_idx=slot,
            staging=value,
        )
        for (identity, params, slot), value in zip(rows, staged, strict=True)
    }


def _bind_cache_tables(
    lane: RunLane,
    scope: LaneState,
    *,
    cache_pool: CachePool | None,
    request_tables: ReqToTokenPool | None,
) -> None:
    """Install scheduler tables and retain row-aligned forward coordinates."""

    cache = cache_pool
    page_tables = request_tables
    if cache is None or page_tables is None:
        raise invalid_descriptor("cache tables require physical KV storage")
    started = time.perf_counter_ns()
    tables = []
    for table in lane.block_tables:
        pages = cache.validate_pages(table.page_ids, group=table.group_id)
        if int(table.allocated_tokens) > len(pages) * cache.block_size:
            raise invalid_descriptor("block-table allocation exceeds physical capacity")
        tables.append(
            (
                int(table.request_pool_idx),
                int(table.group_id),
                pages,
                int(table.allocated_tokens),
            )
        )
    page_tables.install(tuple(tables))
    for allocation in lane.new_cache_pages:
        pages = cache.validate_pages(
            allocation.page_ids,
            group=allocation.group_id,
        )
        installed = page_tables.pages(allocation.request_pool_idx, allocation.group_id)
        if not set(pages).issubset(installed):
            raise invalid_descriptor("new cache pages are outside the installed block table")
        initialized = {
            page
            for transfer in scope.prepared_transfers.values()
            if isinstance((write := transfer.destination), CacheWrite)
            and (write.request_pool_idx, write.group_id)
            == (allocation.request_pool_idx, allocation.group_id)
            for page in write.initialized_pages
        }
        cache.zero_pages(
            allocation.group_id, tuple(page for page in pages if page not in initialized)
        )

    # Indices refer to the original lane columns, including when inactive
    # computations were filtered from the cache-registration view.
    inputs = scope.lane
    rows_by_operation: dict[OperationIdentity, list[int]] = defaultdict(list)
    for row, index in enumerate(inputs.forward_operation_indices):
        identity = operation_geometry.operation_identity(inputs.operations[index])
        rows_by_operation[identity].append(row)

    for operation in lane.operations:
        request = operation_geometry.request_row(scope, operation.request_key.request_id)
        main_slot = int(request.request.request_pool_idx)
        parent_runtime = request.predecessor.runtime
        operation_rows = rows_by_operation.get(operation_geometry.operation_identity(operation), [])
        scope.forward_indices[operation_geometry.operation_identity(operation)] = tuple(
            operation_rows
        )
        main_descriptor = next(
            (row for row in operation_rows if inputs.request_pool_indices[row] == main_slot),
            None,
        )
        if main_descriptor is not None:
            visible = int(parent_runtime.kv_visible_len)
            declared = inputs.seq_lens[main_descriptor] - inputs.query_lens[main_descriptor]
            relayed = operation.predicate is not None and operation.predicate.dtype is DType.I64
            # A queued relay carries a capacity bound computed before its
            # predecessor's predicate was known. Actual KV length and validity
            # come from the device row; an inactive descendant must still drain.
            if declared < visible or (not relayed and declared != visible):
                raise invalid_descriptor(
                    "forward row sequence length disagrees with execution state"
                )
        for descriptor in operation_rows:
            slot = inputs.request_pool_indices[descriptor]
            pages = page_tables.pages(slot, 0)
            if slot != main_slot and (
                inputs.seq_lens[descriptor] - inputs.query_lens[descriptor]
            ) > page_tables.allocated_length(slot):
                raise invalid_descriptor("forward row exceeds alternative-prefix capacity")
            if slot != main_slot:
                page_tables.retain_prefix(operation.request_key, slot)
            cache.retain_execution(
                operation.request_key,
                pages,
                group=0,
                length=inputs.seq_lens[descriptor]
                - (0 if inputs.write_kv[descriptor] else inputs.query_lens[descriptor]),
                completion=scope.completion.completion_future(),
            )
    record_component(scope, "bc_tables", started)


def _stage_input_products(
    input_products: Sequence[TensorPublication],
    scope: LaneState,
    *,
    cache_registry: CachePublications | None,
    encoder_cache: EncoderCache,
    latent_pool: LatentPool | None,
    config: WorkerConfig,
) -> None:
    """Publish query-ready transferred values into their owning runtime stores."""

    for buffer, prepared_kv in scope.prepared_transfers.items():
        if not isinstance(prepared_kv.value, KvTransfer):
            continue
        if not prepared_kv.ready():
            raise invalid_descriptor("KV input has no query-ready physical import")
        if cache_registry is None:
            raise invalid_descriptor("KV input requires cache publication storage")
        existing = cache_registry.resident(buffer)
        if existing is not None and existing != prepared_kv.value:
            raise invalid_descriptor("staged KV publication conflicts with its buffer identity")
        scope.cache_publication_inputs[buffer] = prepared_kv.value
    for entry in input_products:
        product = entry.product
        # Transfer metadata determines which runtime owns the imported value;
        # each branch validates identity and geometry before publication.
        transfer = scope.prepared_transfers.get(product.buffer_id)
        if transfer is None or not transfer.ready():
            raise invalid_descriptor("cross-stage input has no query-ready prepared transfer")
        value = transfer.value
        if isinstance(value, LatentTransferValue):
            consumers = tuple(
                operation
                for operation in scope.lane.operations
                if product in operation.tensor_inputs()
            )
            if len(consumers) != 1:
                raise invalid_descriptor("latent transfer must have one lane consumer")
            row = operation_geometry.latent_row(consumers[0], scope)
            if (
                value.latent_units != int(row.params.latent_units)
                or value.height != int(row.params.height)
                or value.width != int(row.params.width)
                or value.step != int(row.params.start_step)
            ):
                raise invalid_descriptor("latent transfer disagrees with its scheduler params")
            request = operation_geometry.request_row(scope, product.request_key.request_id)
            if request.latent_product is not None or int(request.flow_step) != 0:
                raise invalid_descriptor("latent transfer destination already owns a trajectory")
            binding = transfer.destination
            if not isinstance(binding, LatentWrite):
                raise RuntimeError("latent transfer lost its reserved destination")
            if (binding.request_pool_idx, binding.page_table) != (
                row.request_pool_idx,
                tuple(row.params.page_table),
            ):
                raise invalid_descriptor("latent import reservation changed before execution")
            require_latent_pool(latent_pool).adopt_import(
                binding,
                generation=product.generation,
                step=value.step,
                height=value.height,
                width=value.width,
            )
            transfer.adopt_destination()
            scope.latent_import_slots.append(row.request_pool_idx)
            request.latent_product = product
            request.flow_step = value.step
            continue
        tensors = transfer.tensors()
        if len(tensors) != 1:
            raise invalid_descriptor("product transfer produced an invalid tensor set")
        consumers = tuple(
            operation
            for operation in scope.lane.operations
            if product in operation.tensor_inputs() or operation.predicate == product
        )
        if not consumers:
            raise invalid_descriptor("transferred product has no lane consumer")
        devices = {_operation_device(operation, config=config) for operation in consumers}
        if len(devices) != 1:
            raise invalid_descriptor("transferred product spans multiple consumer devices")
        if isinstance(value, DeviceProductTransferValue):
            binding = transfer.destination
            if not isinstance(binding, DeviceProductImport):
                raise RuntimeError("device-product transfer lost its reserved destination")
            if not transfer.destination_adopted:
                binding.commit()
                transfer.adopt_destination()
            continue
        if not isinstance(value, EncoderTransferValue):
            raise RuntimeError("prepared transfer has an unknown descriptor")
        encoder_binding = transfer.destination
        if not isinstance(encoder_binding, EncoderWrite):
            raise RuntimeError("encoder transfer lost its reserved destination")
        if not transfer.destination_adopted:
            encoder_cache.publish(
                encoder_binding,
                tensors[0],
                EncoderMetadata(height=value.height, width=value.width),
            )
            encoder_cache.commit_writes((encoder_binding,))
            transfer.adopt_destination()
        continue
