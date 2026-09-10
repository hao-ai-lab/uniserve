"""Candidate preparation and resource-specific publication for execution batches."""

from __future__ import annotations

import logging
import math
import time
import traceback
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from uniserve_worker.execution.batch import (
    ArResult,
    AttentionRegime,
    BufferId,
    Checkpoint,
    CompletionState,
    DeviceProductTransferValue,
    DeviceSelected,
    DiffusionResult,
    Domain,
    DType,
    EncoderResult,
    EncoderTransferValue,
    ErrorCode,
    Finish,
    FinishFlags,
    FixedCheckpoint,
    Free,
    KvTransferValue,
    LaneResult,
    LatentParams,
    LatentTransferValue,
    Locator,
    LogicalLengths,
    ModelOutput,
    OpCode,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    RegistrationAck,
    RequestKey,
    Retire,
    Run,
    RunLane,
    RunResult,
    ShapeBound,
    StorageClass,
    TensorTransfer,
    TimingCounters,
    TokenSpan,
    TransferHandle,
    TransferResult,
    WorkerForwardStats,
    decode_sampling_state_bytes,
    decode_token_product_bytes,
)
from uniserve_worker.execution.cuda_graph import MixedCapture
from uniserve_worker.execution.forward_batch import (
    ModelPhase,
)
from uniserve_worker.execution.output import (
    ImagePayload,
    LogprobPayload,
    OutputBuffer,
    OutputRecord,
    PendingOutput,
    TokenCapture,
)
from uniserve_worker.execution.trace import (
    ExecutionPhase,
    OperationTrace,
)
from uniserve_worker.execution.video import (
    run_action as run_video_action,
)
from uniserve_worker.execution.video import (
    validate_batch as validate_video_batch,
)
from uniserve_worker.foundation.errors import (
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
    should_capture_trace,
    unsupported_operation,
    unsupported_setup,
)
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.models.generation import (
    GenerationPipeline,
)
from uniserve_worker.models.inputs import ImageProcessor
from uniserve_worker.models.runtime import (
    ExecutionModel,
)
from uniserve_worker.models.video import VideoModel
from uniserve_worker.profiling import profile_range
from uniserve_worker.runtime.cache_transfer import CacheWrite
from uniserve_worker.runtime.device_products import (
    DeviceProductImport,
    DeviceProductMetadata,
    DeviceProductRead,
    DeviceProductWrite,
    ImageRange,
)
from uniserve_worker.runtime.encoder_cache import (
    EncoderMetadata,
    EncoderWrite,
)
from uniserve_worker.runtime.latent_pool import (
    LatentPool,
    LatentWrite,
)
from uniserve_worker.runtime.request import (
    RequestRuntime,
    SpeculativeCommit,
)
from uniserve_worker.transfer.tickets import TransferTicket

from .attention import columns as _attention_columns
from .attention import dense_columns as _dense_attention_columns
from .model_runner import ForwardResult, ModelRunner, RunObservation, RunPath
from .rows import (
    DecodeRuntimePublication,
    ForwardRow,
    LaneLayout,
    LaneState,
    LatentExecution,
    OperationIdentity,
    OperationState,
    Outcome,
    PreparedExecution,
    PreparedPredicateBatch,
    PreparedTransferInput,
    SampleWork,
    dependencies_ready,
)
from .sample import (
    copy_runtime_scalar as _copy_runtime_scalar,
)
from .sample import (
    sample as _sample_task_batch,
)

if TYPE_CHECKING:
    from ..worker.worker import Worker


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
_MIN_MIXED_SERVICE_SPEEDUP = 1.03


def _operation_identity(operation: Operation) -> OperationIdentity:
    """Return the request key and operation identifier for one operation."""

    return operation.request_key, int(operation.op_id)


def _reference_operation_identity(reference: ProductRef) -> OperationIdentity:
    """Return the producer request key and operation identifier for a product reference."""

    return reference.request_key, int(reference.producer_op_id)


def _unique_scopes(scopes: Sequence[LaneState]) -> tuple[LaneState, ...]:
    """Deduplicate lane scopes by object identity while preserving order."""

    unique: list[LaneState] = []
    seen: set[int] = set()
    for scope in scopes:
        identity = id(scope)
        if identity not in seen:
            seen.add(identity)
            unique.append(scope)
    return tuple(unique)


def _completion_error_code(code: WorkerErrorCode) -> ErrorCode:
    """Map internal failure classes to their completion-wire error codes."""

    if code == WorkerErrorCode.RESOURCE_ERROR:
        return ErrorCode.RESOURCE_EXHAUSTED
    if code == WorkerErrorCode.COMPUTE_ERROR:
        return ErrorCode.COMPUTE_ERROR
    if code in {WorkerErrorCode.INVARIANT_VIOLATION, WorkerErrorCode.FATAL_WORKER_FAILURE}:
        return ErrorCode.INTERNAL
    return ErrorCode.INVALID_OPERATION


def plan_run(runtime: Worker, batch: Run) -> Run:
    """Derive worker-local execution lanes from a flat physical run."""

    if batch.lanes or not batch.operations:
        return batch
    if any(
        params.offset + params.bytes > runtime.info.buffer_pool_bytes
        for params in batch.buffer_allocations
    ):
        raise invalid_descriptor("run buffer params exceeds the worker buffer pool")
    grouped: dict[tuple[Domain, str], list[tuple[int, Operation]]] = {}
    for index, operation in enumerate(batch.operations):
        grouped.setdefault((operation.domain, operation.entry), []).append((index, operation))
    lanes: list[RunLane] = []
    for lane_id, ((domain, _entry), members) in enumerate(grouped.items(), start=1):
        global_to_local = {
            global_index: local_index
            for local_index, (global_index, _operation) in enumerate(members)
        }
        member_operations = tuple(operation for _index, operation in members)
        identities = {
            (operation.request_key, int(operation.op_id)) for operation in member_operations
        }
        rows = tuple(
            replace(row, operation_index=global_to_local[int(row.operation_index)])
            for row in batch.forward_rows
            if int(row.operation_index) in global_to_local
        )
        request_slots = {int(row.request_pool_index) for row in rows}
        attention = (
            AttentionRegime.CAUSAL
            if all(
                operation.kind in {OpCode.AR_EXTEND, OpCode.AR_DECODE, OpCode.AR_VERIFY}
                for operation in member_operations
            )
            else AttentionRegime.HYBRID
            if any(operation.kind is OpCode.DIFFUSION_STEP for operation in member_operations)
            else AttentionRegime.NONE
        )
        lanes.append(
            RunLane(
                lane_id=lane_id,
                launch_id=lane_id,
                collective_seq=batch.collective_seq,
                domain=domain,
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
                forward_rows=rows,
                latent_params=tuple(
                    params
                    for params in batch.latent_params
                    if (params.request_key, int(params.op_id)) in identities
                ),
                decode_ranges=tuple(
                    params
                    for params in batch.decode_ranges
                    if (params.request_key, int(params.op_id)) in identities
                ),
                buffer_allocations=tuple(
                    params
                    for params in batch.buffer_allocations
                    if any(
                        product.buffer_id == params.buffer
                        for operation in member_operations
                        for product in (
                            *operation.inputs,
                            *operation.outputs,
                            *((operation.predicate,) if operation.predicate is not None else ()),
                        )
                    )
                ),
            )
        )
    decode = next((lane for lane in lanes if lane.domain is Domain.DECODE), None)
    flow = next((lane for lane in lanes if lane.domain is Domain.FLOW), None)
    if (
        decode is not None
        and flow is not None
        and runtime.model.tensorized_mixed
        and {operation.kind for lane in (decode, flow) for operation in lane.operations}
        == {OpCode.AR_DECODE, OpCode.DIFFUSION_STEP}
        and runtime.runner.allows_mixed(_mixed_bucket(runtime, (decode, flow)))
    ):
        launch_id = min(decode.launch_id, flow.launch_id)
        lanes = [
            replace(lane, launch_id=launch_id) if lane is decode or lane is flow else lane
            for lane in lanes
        ]
    return replace(batch, lanes=tuple(lanes))


def prepare_batch(runtime: Worker, batch: Run) -> PreparedExecution:
    """Submit bounded transfer and predicate observations without waiting."""

    _apply_batch_controls(runtime, batch)
    storage_dependencies: list[Future[None]] = []
    pool = runtime.latent_pool
    if pool is not None:
        admissions = {
            admission.request_key: admission.request_pool_idx for admission in batch.admissions
        }
        operations = {
            (operation.request_key, operation.op_id): operation for operation in batch.operations
        }
        for latent_params in batch.latent_params:
            operation = operations[(latent_params.request_key, latent_params.op_id)]
            if operation.kind not in {OpCode.DIFFUSION_PREPARE, OpCode.DIFFUSION_STEP}:
                continue
            request = runtime.requests.peek(latent_params.request_key.request_id)
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

    entries = [
        payload for payload in batch.input_products if isinstance(payload.payload, TransferHandle)
    ]
    supplied = {entry.product for entry in entries}
    for operation in batch.operations:
        if operation.kind is not OpCode.TRANSFER_KV_INSTALL:
            continue
        for product in operation.inputs:
            if product.kind is ProductKind.KV and product not in supplied:
                publications = runtime.cache_publications
                if publications is None:
                    raise invalid_descriptor("KV installation requires cache publication storage")
                publication = publications.publication(product)
                entries.append(ProductPayload(product, TransferHandle(publication)))
                supplied.add(product)
    cache = runtime.cache_pool
    tables = runtime.req_to_token_pool
    if cache is not None and tables is not None and cache.has_pending_accesses:
        request_slots = {
            admission.request_key: admission.request_pool_idx for admission in batch.admissions
        }
        kv_inputs = {
            entry.product: entry.payload.value
            for entry in entries
            if isinstance(entry.payload, TransferHandle)
            and isinstance(entry.payload.value, KvTransferValue)
        }
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
            for row in lane.forward_rows:
                if not row.write_kv:
                    continue
                storage_dependencies.extend(
                    cache.write_dependencies(
                        pages_for(row.request_pool_index, 0),
                        group=0,
                        start=row.seq_len,
                        length=row.query_len,
                    )
                )
            for operation in lane.operations:
                if operation.kind is not OpCode.TRANSFER_KV_INSTALL:
                    continue
                request = runtime.requests.peek(operation.request_key.request_id)
                slot = (
                    request_slots.get(operation.request_key)
                    if request is None
                    else request.request_pool_idx
                )
                if slot is None:
                    raise invalid_descriptor("KV installation has no admitted request slot")
                for reference in operation.inputs:
                    kv_publication = kv_inputs.get(reference)
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
    if entries:
        # Destination addresses may still belong to an earlier physical reader.
        # Its retirement wakes the execution thread, which submits these reads.
        prepared._prepare_inputs = partial(_prepare_inputs, runtime, batch, tuple(entries))
        prepared.advance()
    else:
        prepared.transfers, prepared.predicates = _prepare_inputs(runtime, batch, tuple(entries))
    return prepared


def _prepare_inputs(
    runtime: Worker,
    batch: Run,
    entries: tuple[ProductPayload, ...],
) -> tuple[tuple[PreparedTransferInput, ...], PreparedPredicateBatch | None]:
    """Reserve transfer destinations and submit reads after their storage is available."""

    from . import transfer

    transports = runtime.transports
    if entries and not transports:
        raise unsupported_setup("cross-stage input requires a configured transport")
    transfers: list[PreparedTransferInput] = []
    try:
        for entry in entries:
            assert transports
            assert isinstance(entry.payload, TransferHandle)
            devices = {
                runtime.operation_device(operation)
                for operation in batch.operations
                if entry.product in operation.inputs or entry.product == operation.predicate
            }
            if len(devices) != 1:
                raise invalid_descriptor("transferred product requires one consumer device per run")
            device = next(iter(devices))
            value = entry.payload.value
            tensors: tuple[TensorTransfer, ...]
            if isinstance(value, EncoderTransferValue):
                main = value.tensor
                tensors = (main,)
                if (
                    not isinstance(value.payload_kind, str)
                    or value.payload_kind
                    not in {ProductKind.VISION_FEATURE.value, ProductKind.LATENT_FEATURE.value}
                    or min(value.height, value.width, value.generation) < 1
                    or value.generation != entry.product.generation
                    or not transfer.tensor_matches_product(main, entry.product)
                ):
                    raise invalid_descriptor("encoder transfer metadata exceeds its product bounds")
                if (
                    entry.product.kind.value != value.payload_kind
                    or entry.product.storage_class is not StorageClass.LATENT_ARENA
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
                if (
                    value.generation != entry.product.generation
                    or not transfer.requires_device_product_binding(entry.product)
                    or not transfer.tensor_matches_product(main, entry.product)
                ):
                    raise invalid_descriptor(
                        "device-product transfer metadata exceeds its product bounds"
                    )
            elif isinstance(value, LatentTransferValue):
                main = value.tensor
                tensors = (main,)
                pool = runtime.latent_pool
                expected_dtype = "" if pool is None else str(pool.dtype).removeprefix("torch.")
                expected_nbytes = (
                    0
                    if pool is None
                    else value.latent_units
                    * int(pool.latent_width)
                    * int(pool.storage.element_size())
                )
                if (
                    entry.product.kind is not ProductKind.LATENT
                    or entry.product.storage_class is not StorageClass.LATENT_ARENA
                    or pool is None
                    or min(value.height, value.width, value.latent_units, value.generation) < 1
                    or value.generation != entry.product.generation
                    or tuple(main.shape) != (value.latent_units, int(pool.latent_width))
                    or main.dtype != expected_dtype
                    or main.nbytes != expected_nbytes
                    or main.nbytes > entry.product.max_bytes
                    or math.prod(main.shape) > entry.product.shape_bound.max_elements
                ):
                    raise invalid_descriptor("latent transfer metadata exceeds its product bounds")
            elif isinstance(value, KvTransferValue):
                if (
                    entry.product.kind is not ProductKind.KV
                    or entry.product.storage_class is not StorageClass.PAGED_KV
                    or value.generation != entry.product.generation
                ):
                    raise invalid_descriptor("KV transfer entry names a non-KV product")
                consumers = tuple(
                    operation for operation in batch.operations if entry.product in operation.inputs
                )
                if len(consumers) != 1 or consumers[0].kind is not OpCode.TRANSFER_KV_INSTALL:
                    raise invalid_descriptor("KV input requires one installation consumer")
                publications = runtime.cache_publications
                cache = runtime.cache_pool
                tables = runtime.req_to_token_pool
                if publications is None or cache is None or tables is None:
                    raise invalid_descriptor("KV input requires physical cache storage")
                resident = runtime.requests.peek(entry.product.request_key.request_id)
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
                    raise invalid_descriptor("KV transfer has no admitted request slot")
                table = next(
                    (
                        table
                        for lane in batch.lanes
                        for table in lane.block_tables
                        if (table.request_pool_idx, table.group_id) == (slot, value.group_id)
                    ),
                    None,
                )
                pages = tables.pages(slot, value.group_id) if table is None else table.page_ids
                allocated = (
                    tables.allocated_length(slot) if table is None else table.allocated_tokens
                )
                initialized = tuple(
                    page
                    for lane in batch.lanes
                    for allocation in lane.new_cache_pages
                    if (allocation.request_pool_idx, allocation.group_id) == (slot, value.group_id)
                    for page in allocation.page_ids
                )
                write = publications.prepare_install(
                    entry.product,
                    value,
                    request_pool_idx=slot,
                    group_id=value.group_id,
                    page_ids=pages,
                    allocated_length=allocated,
                    initialized_pages=initialized,
                    transports=transports,
                )
                transfers.append(
                    PreparedTransferInput(
                        product=entry.product,
                        value=value,
                        tickets=(),
                        buffers=(),
                        destination=write,
                        _discard_destination=partial(cache.imports.abandon, write),
                    )
                )
                continue
            else:
                raise invalid_descriptor("cross-stage transfer entry has an unknown kind")
            binding: DeviceProductWrite | EncoderWrite | LatentWrite | None = None
            discard: Callable[[], None] | None = None
            target: torch.Tensor | tuple[torch.Tensor, ...] | None = None
            parameters = {params.buffer: params for params in batch.buffer_allocations}
            if isinstance(value, EncoderTransferValue):
                binding = runtime.encoder_cache.bind_outputs(
                    ((entry.product, device),),
                    buffer_allocations=parameters,
                )[0]
                discard = partial(runtime.encoder_cache.abandon_writes, (binding,))
                target = binding.buffer_binding.tensor
            elif isinstance(value, DeviceProductTransferValue):
                request_slots = {
                    admission.request_key: int(admission.request_pool_idx)
                    for admission in batch.admissions
                }
                resident = runtime.requests.peek(entry.product.request_key.request_id)
                if resident is not None and resident.request_key == entry.product.request_key:
                    request_slots[resident.request_key] = int(resident.request_pool_idx)
                imported = runtime.device_products.import_tensor(
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
                        product=entry.product,
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
                    operation for operation in batch.operations if entry.product in operation.inputs
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
                resident = runtime.requests.peek(entry.product.request_key.request_id)
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
                pool = _latent_pool(runtime)
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
                    runtime.encoder_cache.retain_transfer(binding, ticket)
                elif isinstance(binding, DeviceProductWrite):
                    runtime.device_products.retain_transfer(binding, ticket)
                elif isinstance(binding, LatentWrite):
                    _latent_pool(runtime).retain_transfer(binding, ticket)

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
                        product=entry.product,
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
        predicates = _prepare_predicates(
            runtime,
            batch,
            transfers=tuple(transfers),
        )
    except BaseException:
        for prepared_transfer in transfers:
            prepared_transfer.close()
        raise
    return tuple(transfers), predicates


def _prepare_predicates(
    runtime: Worker,
    batch: Run,
    *,
    transfers: tuple[PreparedTransferInput, ...],
) -> PreparedPredicateBatch | None:
    """Capture completion-valued predicates from local products or prepared transfers."""

    # Predicate rows occupy one compact completion buffer regardless of whether
    # their source is already local or will arrive through a prepared transfer.
    operations = tuple(
        operation
        for operation in batch.operations
        if operation.predicate is not None and operation.predicate.kind is ProductKind.COMPLETION
    )
    if not operations:
        return None
    transferred = {transfer.product: transfer for transfer in transfers}
    buffer = runtime.output_pool.acquire(
        len(operations),
        token_capacity=len(operations),
        devices=tuple(_operation_device(runtime, operation) for operation in operations),
    )
    captures: list[tuple[OperationIdentity, TokenCapture, int]] = []
    pending: list[tuple[OperationIdentity, PreparedTransferInput, int]] = []
    recorded: list[DeviceProductRead] = []
    try:
        # Local sources are consumed in device batches and captured directly;
        # transferred sources retain their target row for later completion.
        grouped: dict[torch.device, list[Operation]] = defaultdict(list)
        rows = {_operation_identity(operation): row for row, operation in enumerate(operations)}
        for operation in operations:
            transfer = transferred.get(cast(ProductRef, operation.predicate))
            if transfer is None:
                grouped[_operation_device(runtime, operation)].append(operation)
            else:
                pending.append(
                    (
                        _operation_identity(operation),
                        transfer,
                        rows[_operation_identity(operation)],
                    )
                )
        for device, device_operations in grouped.items():
            reads = runtime.device_products.consume_batch(
                tuple(
                    (
                        cast(ProductRef, operation.predicate),
                        int(operation.op_id),
                        device,
                    )
                    for operation in device_operations
                ),
                device=device,
            )
            recorded.extend(reads)
            for operation, read in zip(device_operations, reads, strict=True):
                identity = _operation_identity(operation)
                captures.append((identity, buffer.capture(read.tensor), rows[identity]))
            runtime.device_products.record_readers(reads, device=device)

        # No pending transfer can mutate the buffer once it is sealed.
        sealed = not pending
        if sealed:
            buffer.seal()
    except BaseException:
        # Every acquired read must receive a reader event even when preparation
        # fails before all device groups are captured.
        unrecorded = tuple(read for read in recorded if not read._recorded)
        if unrecorded:
            runtime.device_products.record_readers(unrecorded)
        buffer.abandon()
        raise
    return PreparedPredicateBatch(
        buffer=buffer,
        entries=captures,
        transferred=tuple(pending),
        sealed=sealed,
    )


def execute_prepared(runtime: Worker, prepared: PreparedExecution) -> RunResult:
    """Execute a fully staged batch through the bound single-use preparation callback."""

    if not prepared.ready():
        raise RuntimeError("prepared execution was observed before transfer readiness")
    return _execute(
        runtime,
        prepared.batch,
        prepared=prepared.transfers,
        predicate_values=prepared.predicate_values(),
        propagate_errors=False,
        graph_eligible=True,
        controls_applied=True,
    )


def complete_startup(runtime: Worker) -> None:
    """Retire pre-admission collective identities before serving traffic."""

    runtime.runner.complete_startup()
    if runtime.requests.request_ids():
        raise RuntimeError("startup completed with resident requests")
    runtime._collective_history.clear()


def execute_batch(
    runtime: Worker,
    batch: Run,
    *,
    prepared: tuple[PreparedTransferInput, ...] = (),
) -> RunResult:
    """Execute one canonical batch after server-side duplicate registration."""

    return _execute(
        runtime,
        batch,
        prepared=prepared,
        predicate_values={},
        propagate_errors=False,
        graph_eligible=True,
    )


def execute_startup(
    runtime: Worker,
    batch: Run,
    *,
    catalog_graphs: bool = True,
) -> RunResult:
    """Execute pre-admission work with direct errors and normal storage retirement."""

    report = _execute(
        runtime,
        batch,
        prepared=(),
        predicate_values={},
        propagate_errors=True,
        graph_eligible=bool(catalog_graphs),
    )
    return runtime._retire_commands(batch, report)


def _apply_batch_controls(runtime: Worker, batch: Run) -> None:
    """Validate and apply ordered controls before reserving asynchronous execution.

    Free can retire a published page bank needed by this run. Applying it before
    the storage wait prevents a dependency on the run's own unprocessed release.
    """

    operations = _trace_envelopes(batch.operations)
    validation_started = time.perf_counter_ns()
    try:
        _validate_batch(runtime, batch)
    except BaseException as error:
        runtime.trace.emit(
            ExecutionPhase.INPUT_VALIDATION,
            operations,
            duration_us=(time.perf_counter_ns() - validation_started) // 1000,
            error=error,
        )
        raise
    runtime.trace.emit(
        ExecutionPhase.INPUT_VALIDATION,
        operations,
        duration_us=(time.perf_counter_ns() - validation_started) // 1000,
    )
    for command in batch.commands:
        started_slots = runtime.requests.apply_commands((command,))
        if started_slots and runtime.runtime_states is not None:
            runtime.runtime_states.reset(started_slots)
    _apply_release_controls(runtime, batch, before_execution=True)


def _execute(
    runtime: Worker,
    batch: Run,
    *,
    prepared: tuple[PreparedTransferInput, ...],
    predicate_values: Mapping[OperationIdentity, bool],
    propagate_errors: bool,
    graph_eligible: bool,
    controls_applied: bool = False,
) -> RunResult:
    """Execute a prepared lane batch for startup or admitted traffic."""

    started = time.perf_counter_ns()
    if not controls_applied:
        _apply_batch_controls(runtime, batch)
    # Collective sequence names computation order. Input preparation can arrive
    # out of that order while an independent run's physical reads are pending.
    validate_collective_sequence(
        runtime.worker_config.world_size, runtime._collective_history, batch
    )
    required_predicates = {
        _operation_identity(operation)
        for operation in batch.operations
        if operation.predicate is not None and operation.predicate.kind is ProductKind.COMPLETION
    }
    if required_predicates != set(predicate_values):
        raise invalid_descriptor(
            "completion-predicated operations require exact prepared predicate values"
        )
    if not batch.operations:
        return RunResult(
            batch_id=batch.batch_id,
            run_id=batch.run_id,
            lanes=(),
        )
    reports: dict[int, LaneResult] = {}
    groups: dict[int, list[RunLane]] = {}
    for lane in batch.lanes:
        groups.setdefault(lane.launch_id, []).append(lane)

    for lanes in groups.values():
        scopes: list[LaneState] = []
        for lane in lanes:
            try:
                scopes.append(
                    _open_lane(
                        runtime,
                        batch,
                        lane,
                        prepared,
                        predicate_values,
                        graph_eligible,
                    )
                )
            except BaseException as error:
                classified = _classify_lane_failure(
                    runtime,
                    lane,
                    error,
                    phase="lane registration",
                )
                if propagate_errors or classified.fatal:
                    for scope in scopes:
                        _discard_lane(runtime, scope, classified)
                    raise classified
                reports[lane.lane_id] = _registration_error_lane(
                    runtime,
                    batch.run_id,
                    lane,
                    classified,
                    started,
                )

        if not scopes:
            continue
        try:
            outcomes, execution_errors = _execute_lane_group(
                runtime,
                tuple(scopes),
                qualify_mixed=propagate_errors,
            )
        except BaseException as error:
            classified = _classify_lane_failure(
                runtime,
                lanes[0],
                error,
                phase="lane execution",
            )
            for scope in scopes:
                _discard_lane(runtime, scope, classified)
            if propagate_errors or classified.fatal:
                raise classified
            for scope in scopes:
                reports[scope.lane.lane_id] = _error_lane(
                    runtime,
                    batch.run_id,
                    scope,
                    classified,
                    started,
                )
            continue

        if propagate_errors and execution_errors:
            first_lane = next(lane for lane in lanes if lane.lane_id in execution_errors)
            classified = _classify_lane_failure(
                runtime,
                first_lane,
                execution_errors[first_lane.lane_id],
                phase="lane execution",
            )
            for scope in scopes:
                _discard_lane(runtime, scope, classified)
            raise classified

        for scope in scopes:
            lane_error = execution_errors.get(scope.lane.lane_id)
            if lane_error is not None:
                classified = _classify_lane_failure(
                    runtime,
                    scope.lane,
                    lane_error,
                    phase="lane execution",
                )
                _discard_lane(runtime, scope, classified)
                if propagate_errors or classified.fatal:
                    raise classified
                reports[scope.lane.lane_id] = _error_lane(
                    runtime,
                    batch.run_id,
                    scope,
                    classified,
                    started,
                )
                continue
            lane_outcomes = outcomes[scope.lane.lane_id]
            try:
                reports[scope.lane.lane_id] = _commit_lane(
                    runtime,
                    batch.run_id,
                    scope,
                    lane_outcomes,
                    started,
                )
            except BaseException as error:
                if scope.publication_started:
                    classified = _published_lane_failure(
                        runtime,
                        scope.lane,
                        error,
                    )
                else:
                    classified = _classify_lane_failure(
                        runtime,
                        scope.lane,
                        error,
                        phase="lane commit",
                    )
                    _discard_lane(runtime, scope, classified)
                if propagate_errors or classified.fatal:
                    raise classified
                reports[scope.lane.lane_id] = _error_lane(
                    runtime,
                    batch.run_id,
                    scope,
                    classified,
                    started,
                )

    _apply_release_controls(runtime, batch, before_execution=False)
    report = RunResult(
        batch_id=batch.batch_id,
        run_id=batch.run_id,
        lanes=tuple(reports[lane.lane_id] for lane in batch.lanes),
    )
    runtime.trace.emit(
        ExecutionPhase.COMMIT,
        _trace_envelopes(batch.operations),
        duration_us=(time.perf_counter_ns() - started) // 1000,
    )
    return report


def _classify_lane_failure(
    runtime: Worker,
    lane: RunLane,
    error: BaseException,
    *,
    phase: str,
) -> WorkerError:
    """Classify a pre-publication lane failure with complete operation and route context."""

    operations = tuple(
        (
            int(operation.request_key.authority_id),
            int(operation.request_key.request_id),
            int(operation.request_key.epoch),
            int(operation.op_id),
        )
        for operation in lane.operations
    )
    sole = lane.operations[0] if len(lane.operations) == 1 else None
    classified = classify(
        error,
        context=phase,
        phase=phase,
        operations=operations,
        req_id=None if sole is None else int(sole.request_key.request_id),
        op_id=None if sole is None else int(sole.op_id),
        op_kind=None if sole is None else sole.kind.value,
        route=str(lane.route),
    )
    _log_lane_failure(runtime, lane, classified, cause=error)
    return classified


def _published_lane_failure(
    runtime: Worker,
    lane: RunLane,
    error: BaseException,
) -> WorkerError:
    """Classify a post-visibility publication failure as a fatal invariant violation."""

    operations = tuple(
        (
            int(operation.request_key.authority_id),
            int(operation.request_key.request_id),
            int(operation.request_key.epoch),
            int(operation.op_id),
        )
        for operation in lane.operations
    )
    classified = WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=f"lane publication failed after visibility began: {error}",
        fatal=True,
        phase="lane publication",
        route=str(lane.route),
        operations=operations,
    )
    _log_lane_failure(runtime, lane, classified, cause=error)
    return classified


def _log_lane_failure(
    runtime: Worker,
    lane: RunLane,
    error: WorkerError,
    *,
    cause: BaseException | None = None,
) -> None:
    """Log a classified lane failure with traceback only for diagnostic error classes."""

    capture_trace = should_capture_trace(error.code)
    log = logger.error if capture_trace else logger.warning
    log(
        "lane failed: %s [code=%s lane_id=%s route=%s operations=%s]",
        error.message,
        error.code,
        lane.lane_id,
        lane.route,
        error.operations,
        exc_info=(type(cause), cause, cause.__traceback__)
        if capture_trace and cause is not None
        else None,
    )
    # A queued log record may outlive the worker. Keep the traceback locations
    # and exception chain, but do not let diagnostic frames retain borrowed
    # staging tensors after their CUDA lane has been closed.
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        traceback.clear_frames(cause.__traceback__)
        cause = cause.__cause__ or cause.__context__


def _open_lane(
    runtime: Worker,
    batch: Run,
    lane: RunLane,
    prepared: tuple[PreparedTransferInput, ...],
    predicate_values: Mapping[OperationIdentity, bool],
    graph_eligible: bool,
) -> LaneState:
    """Stage one lane's speculative state, resources, inputs, and completion storage."""

    operations = lane.operations

    # Predicated rows remain in aligned output/state tables but do not reserve
    # execution-only inputs, CPU tasks, or model resources.
    predicated = frozenset(
        identity
        for operation in operations
        if (identity := _operation_identity(operation)) in predicate_values
        and not predicate_values[identity]
    )
    active_operations = (
        operations
        if not predicated
        else tuple(
            operation
            for operation in operations
            if _operation_identity(operation) not in predicated
        )
    )
    # Restrict input payloads to identities declared by this lane.
    traced = _trace_envelopes(operations)
    started = time.perf_counter_ns()
    declared_inputs = {reference for operation in operations for reference in operation.inputs}
    declared_inputs.update(
        operation.predicate for operation in operations if operation.predicate is not None
    )
    input_products = tuple(
        payload for payload in batch.input_products if payload.product in declared_inputs
    )
    completion: OutputBuffer | None = None
    try:
        # Candidate drafts and completion slots form a speculative ownership unit:
        # either all later lane resources bind successfully or both are discarded.
        request_pool_indices = tuple(
            int(runtime.requests.get(operation.request_key.request_id).request_pool_idx)
            for operation in operations
        )
        candidates = runtime.requests.stage_lane(operations, request_pool_indices)
        for operation, request in zip(operations, candidates, strict=True):
            request.install_runtime(request.request.parent_runtime(operation.parent))
        completion = runtime.output_pool.acquire(
            len(operations),
            token_capacity=_lane_completion_words(runtime, operations),
            devices=_completion_devices(runtime, operations),
        )
    except BaseException as error:
        if completion is not None:
            completion.abandon()
        runtime.trace.emit(
            ExecutionPhase.CANDIDATE_STAGE,
            traced,
            duration_us=(time.perf_counter_ns() - started) // 1000,
            error=error,
        )
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
            transfer.product: transfer
            for transfer in prepared
            if transfer.product in declared_inputs
        },
        predicated_operations=predicated,
    )
    runtime.trace.emit(
        ExecutionPhase.CANDIDATE_STAGE,
        traced,
        duration_us=(time.perf_counter_ns() - started) // 1000,
    )
    try:
        # Bind physical state in dependency order before decoding transferred inputs.
        _reserve_cpu_tasks(runtime, active_operations, scope)
        active_lane = _active_lane(runtime, lane, active_operations)
        if active_lane is not None:
            if runtime.cache_pool is None or runtime.req_to_token_pool is None:
                if (
                    active_lane.block_tables
                    or active_lane.new_cache_pages
                    or active_lane.forward_rows
                ):
                    raise unsupported_setup(
                        "KV-free execution received cache tables or packed forward rows"
                    )
            else:
                _bind_cache_tables(runtime, active_lane, scope)
        # Preserve operation/request row alignment for forward packing and commit.
        scope.layout = LaneLayout(
            operations=operations,
            requests=candidates,
            seq_lens=tuple(
                int(request.request.parent_runtime(operation.parent).kv_visible_len)
                for operation, request in zip(operations, candidates, strict=True)
            ),
            weights=(
                _weights(
                    runtime,
                ),
            )
            * len(candidates),
            identities=tuple(_operation_identity(operation) for operation in operations),
        )
        if active_lane is not None:
            _bind_latent_rows(runtime, active_lane, scope)
        _reserve_outputs(runtime, operations, scope)

        # Only live operations consume inputs; predicated outputs are published
        # directly into their aligned completion rows.
        active_inputs = {
            reference for operation in active_operations for reference in operation.inputs
        }
        active_inputs.update(
            operation.predicate
            for operation in active_operations
            if operation.predicate is not None
        )
        _stage_input_products(
            runtime,
            tuple(payload for payload in input_products if payload.product in active_inputs),
            scope,
        )
        _consume_predicates(runtime, active_operations, scope)
        _publish_predicated_outputs(runtime, operations, scope)
        scope.registration_visible = True
        _record_component(scope, "open_lane", started)
        return scope
    except BaseException:
        _discard_lane(runtime, scope)
        raise


def _active_lane(
    runtime: Worker,
    lane: RunLane,
    operations: tuple[Operation, ...],
) -> RunLane | None:
    """Rebuild lane-indexed rows and parameters after predicated operations are removed."""

    if not operations:
        return None
    if operations is lane.operations:
        return lane
    identities = {_operation_identity(operation) for operation in operations}
    old_to_new = {
        index: selected
        for selected, (index, operation) in enumerate(
            (
                item
                for item in enumerate(lane.operations)
                if _operation_identity(item[1]) in identities
            )
        )
    }
    return replace(
        lane,
        operations=operations,
        forward_rows=tuple(
            replace(row, operation_index=old_to_new[row.operation_index])
            for row in lane.forward_rows
            if row.operation_index in old_to_new
        ),
        latent_params=tuple(
            params
            for params in lane.latent_params
            if (params.request_key, int(params.op_id)) in identities
        ),
    )


def _lane_completion_words(runtime: Worker, operations: tuple[Operation, ...]) -> int:
    """Compute fixed completion-word capacity for all operations in a lane."""

    return max(
        1,
        SAMPLING_COMPLETION_FIELDS * len(operations)
        + sum((int(operation.bounds.max_completion_bytes) + 3) // 4 for operation in operations),
    )


def _execute_lane_group(
    runtime: Worker,
    scopes: tuple[LaneState, ...],
    *,
    qualify_mixed: bool,
) -> tuple[dict[int, tuple[Outcome, ...]], dict[int, BaseException]]:
    """Execute active operations across lanes and align outcomes with original lane order."""

    from . import token

    for scope in scopes:
        active = tuple(
            operation
            for operation in scope.lane.operations
            if _operation_identity(operation) not in scope.predicated_operations
        )
        for device in _completion_devices(runtime, active):
            scope.completion.begin_device(device)
    grouped: list[list[Outcome | None]] = [[None] * len(scope.lane.operations) for scope in scopes]
    group_active = tuple(
        operation
        for scope in scopes
        for operation in scope.lane.operations
        if _operation_identity(operation) not in scope.predicated_operations
    )
    homogeneous_decode = bool(group_active) and all(
        operation.kind is OpCode.AR_DECODE for operation in group_active
    )
    states: list[OperationState] = []
    locations: dict[int, tuple[int, int]] = {}
    for scope_index, scope in enumerate(scopes):
        active = tuple(
            operation
            for operation in scope.lane.operations
            if _operation_identity(operation) not in scope.predicated_operations
        )
        if homogeneous_decode:
            decoded = token.decode_batch(runtime, active, scope)
            decoded_by_identity = dict(
                zip(
                    (_operation_identity(operation) for operation in active),
                    decoded,
                    strict=True,
                )
            )
        else:
            decoded_by_identity = {}
        for operation_index, operation in enumerate(scope.lane.operations):
            if _operation_identity(operation) in scope.predicated_operations:
                grouped[scope_index][operation_index] = _predicated_outcome(
                    runtime,
                    operation,
                    scope,
                )
                continue
            identity = _operation_identity(operation)
            if identity in decoded_by_identity:
                grouped[scope_index][operation_index] = decoded_by_identity[identity]
                continue
            state = OperationState(operation=operation, lane=scope)
            locations[id(state)] = (scope_index, operation_index)
            states.append(state)
    errors = _run_ready_set(runtime, states, qualify_mixed=qualify_mixed)
    for state in states:
        scope_index, operation_index = locations[id(state)]
        if state.outcome is not None:
            grouped[scope_index][operation_index] = state.outcome
    outcomes: dict[int, tuple[Outcome, ...]] = {}
    for scope, lane_outcomes in zip(scopes, grouped, strict=True):
        lane_id = scope.lane.lane_id
        if lane_id in errors:
            continue
        if any(outcome is None for outcome in lane_outcomes):
            raise RuntimeError("successful lane did not resolve every operation")
        outcomes[lane_id] = tuple(cast(Outcome, outcome) for outcome in lane_outcomes)
    return outcomes, errors


def _run_ready_set(
    runtime: Worker,
    states: list[OperationState],
    *,
    qualify_mixed: bool,
) -> dict[int, BaseException]:
    """Advance dependency-ready operations through forward, sampling, integration, and actions."""

    from . import encode, flow, token, transfer

    # Products define the in-lane dependency graph; failures suppress only the
    # affected lane while independent lanes continue through the ready set.
    producers = {output: state for state in states for output in state.operation.outputs}
    errors: dict[int, BaseException] = {}

    def live(state: OperationState) -> bool:
        """Select unresolved operations whose lane has not recorded a failure."""

        return state.outcome is None and state.lane.lane.lane_id not in errors

    while any(live(state) for state in states):
        # Pack all ready neural work first. Flow prefix preparation forms an
        # exclusive wave because it may change the rows available to peers.
        forward: list[tuple[OperationState, object]] = []
        ready = tuple(
            state for state in states if live(state) and dependencies_ready(state, producers)
        )
        flow_ready = tuple(
            state for state in ready if state.operation.kind is OpCode.DIFFUSION_STEP
        )
        flow_ready_ids = {id(state) for state in flow_ready}
        for state in flow_ready:
            try:
                rows = _pack_state_forward(runtime, state)
            except BaseException as error:
                errors[state.lane.lane.lane_id] = error
                continue
            forward.extend((state, row) for row in rows)
        preparing_flow_prefix = any(
            live(state) and state.phase == "prefix_pending" for state in flow_ready
        )
        if not preparing_flow_prefix:
            for state in ready:
                if id(state) in flow_ready_ids or not live(state):
                    continue
                try:
                    rows = _pack_state_forward(runtime, state)
                except BaseException as error:
                    errors[state.lane.lane.lane_id] = error
                    continue
                forward.extend((state, row) for row in rows)
        if forward:
            outputs = _run_laneed_wave(
                runtime,
                tuple((cast(ForwardRow, row), state.lane) for state, row in forward),
                qualify_mixed=qualify_mixed,
            )
            aligned: dict[int, list[torch.Tensor]] = defaultdict(list)
            ordered: list[OperationState] = []
            for (state, _row), output in zip(forward, outputs, strict=True):
                if id(state) not in aligned:
                    ordered.append(state)
                aligned[id(state)].append(output)
            for state in ordered:
                try:
                    _consume_state_forward(runtime, state, tuple(aligned[id(state)]))
                except BaseException as error:
                    errors[state.lane.lane.lane_id] = error
            continue

        # Sampling runs after its logits dependencies land and publishes token
        # products that can unlock later operations in the same lane.
        samples: dict[int, list[tuple[OperationState, SampleWork]]] = defaultdict(list)
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            sample = token.pack_sample(state)
            if sample is not None:
                samples[state.lane.lane.lane_id].append((state, cast(SampleWork, sample)))
        if samples:
            for lane_id, candidates in samples.items():
                state = candidates[0][0]
                try:
                    values = _sample_task_batch(
                        tuple(sample for _state, sample in candidates),
                        state.lane.completion,
                        device_products=runtime.device_products,
                        device_reads=tuple(state.lane.device_reads),
                        selection_broadcast=runtime.broadcast_tp_selection,
                    )
                    for (candidate, _sample), value in zip(candidates, values, strict=True):
                        token.consume_sample(runtime, candidate, value)
                except BaseException as error:
                    errors[lane_id] = error
            continue

        # Integration consumes velocity outputs without another model call.
        # Other host/device actions run only when no forward or sample is ready.
        progressed = False
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            try:
                progressed = flow.integrate(runtime, state) or progressed
            except BaseException as error:
                errors[state.lane.lane.lane_id] = error
        if progressed:
            continue
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            try:
                progressed = encode.run_action(runtime, state) or progressed
                progressed = transfer.run_action(runtime, state) or progressed
                progressed = run_video_action(runtime, state) or progressed
            except BaseException as error:
                errors[state.lane.lane.lane_id] = error
        if progressed:
            continue

        # Reaching a live fixed point indicates a dependency or state-machine
        # invariant violation rather than ordinary asynchronous waiting.
        if not any(live(state) for state in states):
            break
        blocked = tuple(_operation_identity(state.operation) for state in states if live(state))
        raise RuntimeError(f"execution ready set made no progress: {blocked!r}")
    return errors


def _pack_state_forward(
    runtime: Worker,
    state: OperationState,
) -> tuple[object, ...]:
    """Dispatch an operation state to its token, flow, or encoder forward packer."""

    from . import encode, flow, token

    operation = state.operation
    if operation.kind.token_mode is not None:
        return token.pack_forward(runtime, state)
    if operation.kind is OpCode.DIFFUSION_STEP and runtime.latent_pool is not None:
        return flow.pack_forward(runtime, state)
    if operation.kind.encode_mode is not None or (
        operation.kind is OpCode.DIFFUSION_FINALIZE and runtime.latent_pool is not None
    ):
        return encode.pack_forward(runtime, state)
    return ()


def _consume_state_forward(
    runtime: Worker,
    state: OperationState,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    """Dispatch aligned model outputs to the operation family's consumer."""

    from . import encode, flow, token

    operation = state.operation
    if operation.kind.token_mode is not None:
        token.consume_forward(runtime, state, outputs)
    elif operation.kind is OpCode.DIFFUSION_STEP and runtime.latent_pool is not None:
        flow.consume_forward(runtime, state, outputs)
    elif operation.kind.encode_mode is not None or (
        operation.kind is OpCode.DIFFUSION_FINALIZE and runtime.latent_pool is not None
    ):
        encode.consume_forward(runtime, state, outputs)
    else:
        raise RuntimeError("model output has no operation consumer")


def _commit_lane(
    runtime: Worker,
    run_id: int,
    scope: LaneState,
    outcomes: tuple[Outcome, ...],
    started: int,
) -> LaneResult:
    """Atomically publish validated lane resources, request versions, and output records."""

    commit_started = time.perf_counter_ns()
    lane = scope.lane
    operations = lane.operations

    # All device reads must finish and every staged resource must validate before
    # completion storage becomes immutable or any publication becomes visible.
    _finish_device_reads(runtime, scope)
    _publish_predicates(runtime, scope)
    runtime.device_products.validate_writes(tuple(scope.device_writes))
    runtime.encoder_cache.validate_writes(tuple(scope.encoder_writes))
    if runtime.latent_pool is None:
        if scope.latent_publications or scope.latent_releases:
            raise RuntimeError("latent publication has no physical pool")
    else:
        runtime.latent_pool.validate_commit(
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
        _validate_completion_products(runtime, operation, outcome.products)
        report_products.extend(
            product
            for product in outcome.products
            if runtime.worker_config.rank == runtime.output_rank(operation.entry)
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
            partial(_finalize_predicated_runtime, runtime, operation),
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
    _record_component(scope, "commit_lane", commit_started)

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
    cache_publications = runtime.cache_publications
    if cache_publications is None:
        if scope.cache_publications or scope.cache_installations:
            raise RuntimeError("cache publication has no backing KV resources")
        cache_commit = None
    else:
        cache_commit = cache_publications.prepare_commit(
            scope.cache_publications,
            scope.cache_installations,
        )
    request_publication = runtime.requests.prepare_publication(
        run_id=run_id,
        operations=operations,
        candidates=scope.request_candidates,
        selected_versions=selected_versions,
        runtimes=resolved_runtime,
        completions=pending_completions,
        speculative=speculative_commits,
    )
    for identity, locators in scope.stage_publications.items():
        existing = runtime._transport_publications.get(identity)
        if existing is not None and existing != locators:
            raise RuntimeError("committed transport publication identity was reused")
    # From this point the lane cannot be discarded: apply resource commits, then
    # reserve the request publication that gates successor readiness.
    scope.publication_started = True
    runtime.device_products.commit_writes(tuple(scope.device_writes))
    runtime.encoder_cache.commit_writes(tuple(scope.encoder_writes))
    if runtime.latent_pool is not None:
        runtime.latent_pool.apply_commit(
            scope.latent_publications,
            scope.latent_releases,
        )
    if cache_publications is not None:
        assert cache_commit is not None
        cache_publications.apply_commit(cache_commit)
    for publication_identity, locators in scope.stage_publications.items():
        runtime._transport_publications[publication_identity] = locators
    _commit_runtime_states(runtime, scope)
    request_publication.reserve()
    return replace(lane_report, publication=request_publication)


def _commit_runtime_states(runtime: Worker, scope: LaneState) -> None:
    """Publish committed token, predicate, position, cache-length, and penalty state to device rows."""

    states = runtime.runtime_states
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
    runtime: Worker,
    scope: LaneState,
    error: BaseException | None = None,
) -> None:
    """Release all provisional lane resources that have not crossed publication visibility."""

    _finish_device_reads(runtime, scope)
    for reservation in scope.cpu_tasks.values():
        reservation.abandon()
    for lease in scope.media_output_leases.values():
        lease.release()
    if scope.publication_started:
        raise RuntimeError("published lane state cannot be discarded")
    if runtime.media_mux is not None:
        for operation in scope.lane.operations:
            if operation.kind is OpCode.DIFFUSION_PREPARE:
                runtime.media_mux.drop(int(operation.request_key.request_id))
    scope.completion.abandon()
    runtime.device_products.abandon_writes(tuple(scope.device_writes))
    runtime.encoder_cache.abandon_writes(tuple(scope.encoder_writes))
    if runtime.cache_pool is not None:
        runtime.cache_pool.release_buffers(
            product.buffer_id
            for operation in scope.lane.operations
            for product in operation.outputs
        )
    if runtime.latent_pool is not None and scope.latent_import_slots:
        runtime.latent_pool.release_slots(tuple(scope.latent_import_slots))
    if runtime.latent_pool is not None:
        runtime.latent_pool.release_buffers(
            tuple(
                product.buffer_id
                for operation in scope.lane.operations
                for product in operation.outputs
            )
        )
    _release_locators(runtime, scope.published)
    runtime.trace.emit(
        ExecutionPhase.CANDIDATE_DISCARD,
        _trace_envelopes(scope.lane.operations),
        error=error,
    )


def _reserve_cpu_tasks(
    runtime: Worker,
    operations: tuple[Operation, ...],
    scope: LaneState,
) -> None:
    """Reserve bounded CPU slots for active operations that schedule host-side work."""

    video_model = isinstance(runtime.model, VideoModel)
    for operation in operations:
        if operation.kind is not OpCode.DIFFUSION_FINALIZE and not (
            video_model and operation.kind is OpCode.MEDIA_APPEND
        ):
            continue
        if video_model and runtime.worker_config.rank != runtime.output_rank(operation.entry):
            continue
        identity = _operation_identity(operation)
        if identity in scope.cpu_tasks:
            raise invalid_descriptor("materialization repeats its CPU task identity")
        reservation = runtime.cpu_tasks.reserve()
        try:
            if video_model and operation.kind is OpCode.MEDIA_APPEND:
                params = next(
                    (
                        params
                        for params in scope.lane.decode_ranges
                        if params.request_key == operation.request_key
                        and int(params.op_id) == int(operation.op_id)
                    ),
                    None,
                )
                if params is None:
                    raise invalid_descriptor("video decode operation has no exact decode params")
                scope.media_output_leases[identity] = runtime.require_media_output_ring().reserve(
                    params.track.value
                )
        except BaseException:
            reservation.abandon()
            raise
        scope.cpu_tasks[identity] = reservation


def _registration_error_lane(
    runtime: Worker,
    run_id: int,
    lane: RunLane,
    error: WorkerError,
    started: int,
) -> LaneResult:
    """Build aligned error outputs without committing candidate request state."""

    generation = 1
    report = _build_error_lane(
        runtime,
        lane,
        generation,
        False,
        error,
        started,
        WorkerForwardStats(),
    )
    return report


def _error_lane(
    runtime: Worker,
    run_id: int,
    scope: LaneState,
    error: WorkerError,
    started: int,
) -> LaneResult:
    """Build a failed lane report and release its unpublished resources."""

    report = _build_error_lane(
        runtime,
        scope.lane,
        scope.completion.generation,
        scope.registration_visible,
        error,
        scope.started_ns,
        _forward_stats(scope.observations, scope.component_us),
    )
    return report


def _build_error_lane(
    runtime: Worker,
    lane: RunLane,
    generation: int,
    registration_visible: bool,
    error: WorkerError,
    started: int,
    forward_stats: WorkerForwardStats,
) -> LaneResult:
    """Materialize one error completion per lane operation without mutating request state."""

    completion_code = _completion_error_code(error.code)
    records: list[ModelOutput] = []
    for operation in lane.operations:
        # Resolve only enough parent state to preserve the scheduler-visible
        # checkpoint and logical lengths in the failed completion.
        request = runtime.requests.peek(operation.request_key.request_id)
        selected_parent = (
            operation.parent
            if operation.parent is not None and operation.parent.is_fixed()
            else None
            if request is None
            else request.resolve_version(operation.parent)
        )
        point = None if selected_parent is None else selected_parent.point
        selected_point = point.point_index if isinstance(point, FixedCheckpoint) else 0
        if request is None or operation.parent is None:
            lengths = LogicalLengths()
        else:
            parent = request.parent_runtime(operation.parent)
            lengths = LogicalLengths(
                token_len=request.logical_position,
                kv_visible_len=parent.kv_visible_len,
                kv_computed_len=parent.kv_computed_len,
                latent_len=request.flow_step,
            )
        payload_type = (
            ArResult
            if operation.kind in {OpCode.AR_EXTEND, OpCode.AR_DECODE, OpCode.AR_VERIFY}
            else EncoderResult
            if operation.kind in {OpCode.ENCODER_VISION, OpCode.ENCODER_LATENT, OpCode.ENCODER_TEXT}
            else DiffusionResult
            if operation.kind
            in {
                OpCode.DIFFUSION_PREPARE,
                OpCode.DIFFUSION_STEP,
                OpCode.DIFFUSION_DECODE,
                OpCode.MEDIA_APPEND,
                OpCode.DIFFUSION_FINALIZE,
            }
            else TransferResult
        )

        # All result families share an empty token span. Diffusion additionally
        # carries its cursor fields so the wire payload remains schema-complete.
        payload_args = (
            lengths,
            TokenSpan(base=lengths.token_len, len=0),
            (),
            FinishFlags(),
            None,
        )
        payload = (
            DiffusionResult(*payload_args, next_cursor=0, done=False)
            if payload_type is DiffusionResult
            else payload_type(*payload_args)
        )
        placeholder = ModelOutput(
            request_key=operation.request_key,
            op_id=operation.op_id,
            completion_slot_generation=max(1, generation),
            status=OpStatus.ERROR,
            selected_point=selected_point,
            product_generations=(),
            error_code=completion_code,
            timing_counters=TimingCounters(),
            payload=payload,
        )
        records.append(placeholder)
    return LaneResult(
        lane_id=lane.lane_id,
        completions=tuple(records),
        registration=RegistrationAck(visible=registration_visible),
        worker_exec_us=(time.perf_counter_ns() - started) // 1000,
        forward_stats=forward_stats,
    )


def _finalize_predicated_runtime(
    runtime: Worker,
    operation: Operation,
) -> tuple[Checkpoint | None, RequestRuntime]:
    """Resolve state-dependent skips; independent computation has no selected state."""

    if operation.parent is None:
        return None, RequestRuntime()
    selected, resolved = runtime.requests.resolve_predicated(
        operation.request_key.request_id,
        operation.op_id,
        operation.parent,
    )
    return selected, resolved


def _validate_batch(runtime: Worker, batch: Run) -> None:
    """Validate run identity, lane resources, routing, and operation support before staging."""

    if len(batch.operations) > runtime.worker_config.max_batch_operations:
        raise invalid_descriptor("execution batch exceeds the worker_config operation limit")
    for operation in batch.operations:
        variant = operation.kind
        if variant not in runtime._effective_work_variants:
            raise unsupported_operation(variant.value, operation.request_key.request_id)
    if any(
        index > runtime.worker_config.max_request_pool_size
        for lane in batch.lanes
        for index in (
            *(table.request_pool_idx for table in lane.block_tables),
            *(row.request_pool_index for row in lane.forward_rows),
        )
    ):
        raise invalid_descriptor("execution batch exceeds request-slot capacity")
    groups: dict[int, list[RunLane]] = defaultdict(list)
    for lane in batch.lanes:
        groups[lane.launch_id].append(lane)
    for lanes in groups.values():
        if len(lanes) < 2:
            continue
        variants = {operation.kind for lane in lanes for operation in lane.operations}
        if not runtime.model.tensorized_mixed or variants != {
            OpCode.AR_DECODE,
            OpCode.DIFFUSION_STEP,
        }:
            raise invalid_descriptor(
                "tensorized mixed submission exceeds the supported mixed buckets"
            )
        bucket = _mixed_bucket(runtime, tuple(lanes))
        if not runtime.runner.allows_mixed(bucket):
            raise invalid_descriptor("tensorized mixed submission has no exact qualified bucket")
    validate_video_batch(runtime, batch)


def validate_collective_sequence(
    process_world_size: int,
    history: OrderedDict[int, object],
    batch: Run,
) -> None:
    """Reject divergent or non-advancing collective identities across all worker roots."""

    if not batch.operations:
        return
    collective_seq = int(batch.collective_seq)
    collective_identity = (collective_seq, batch.operations)
    existing = history.get(collective_seq)
    if existing is not None:
        if existing != collective_identity:
            raise invalid_descriptor("collective sequence was reused with different work")
        return
    if process_world_size > 1 and history and collective_seq <= next(reversed(history)):
        raise invalid_descriptor("collective sequence does not advance")
    history[collective_seq] = collective_identity
    while len(history) > 4096:
        history.popitem(last=False)


def _completion_devices(runtime: Worker, operations: tuple[Operation, ...]) -> tuple[str, ...]:
    """List distinct devices that may contribute asynchronous completion fields."""

    worker_config = runtime.worker_config
    generation_device = worker_config.generation_device
    device = worker_config.device
    selected: list[str] = []
    for operation in operations:
        target = (
            generation_device
            if generation_device is not None and operation.kind in _GENERATION_WORK_VARIANTS
            else device
        )
        if target not in selected:
            selected.append(target)
    return tuple(selected)


def _mixed_bucket(
    runtime: Worker,
    lanes: tuple[RunLane, ...],
) -> MixedCapture:
    """Resolve a shared captured-graph bucket for a compatible mixed lane group."""

    decode_rows = sum(
        operation.kind is OpCode.AR_DECODE for lane in lanes for operation in lane.operations
    )
    flow_operations = tuple(
        operation
        for lane in lanes
        for operation in lane.operations
        if operation.kind is OpCode.DIFFUSION_STEP
    )
    latent_params = {
        (params.request_key, int(params.op_id)): params
        for lane in lanes
        for params in lane.latent_params
    }
    branch_counts: dict[tuple[RequestKey, int], int] = defaultdict(int)
    generation = runtime.model.generation
    if flow_operations and generation is None:
        raise invalid_descriptor("tensorized mixed flow has no generation runtime")
    for lane in lanes:
        for index, operation in enumerate(lane.operations):
            params = latent_params.get((operation.request_key, int(operation.op_id)))
            query_len = (
                None
                if params is None or generation is None
                else generation.physical_tokens(int(params.height), int(params.width))
            )
            branch_counts[(operation.request_key, int(operation.op_id))] = min(
                0 if generation is None else int(generation.max_cfg_branches),
                sum(
                    row.operation_index == index
                    and (query_len is None or int(row.query_len) == int(query_len))
                    for row in lane.forward_rows
                ),
            )
    geometries = {
        (
            int(latent_params[(operation.request_key, int(operation.op_id))].height),
            int(latent_params[(operation.request_key, int(operation.op_id))].width),
            branch_counts[(operation.request_key, int(operation.op_id))],
        )
        for operation in flow_operations
        if (operation.request_key, int(operation.op_id)) in latent_params
    }
    if len(geometries) != 1 or len(latent_params) != len(flow_operations):
        raise invalid_descriptor("tensorized mixed flow rows disagree on physical geometry")
    height, width, cfg_branches = next(iter(geometries))
    return MixedCapture(
        decode_rows=decode_rows,
        flow_rows=len(flow_operations),
        height=height,
        width=width,
        cfg_branches=cfg_branches,
    )


def _reserve_outputs(
    runtime: Worker,
    operations: tuple[Operation, ...],
    scope: LaneState,
) -> None:
    """Bind each declared device value to its concrete bounded owner."""

    from . import transfer

    regions = {}
    shapes = {}
    scalar_groups: dict[
        tuple[torch.device, ProductKind, DType, ShapeBound],
        list[tuple[ProductRef, torch.device | str]],
    ] = {}
    general_bindings: list[tuple[ProductRef, torch.device | str]] = []
    persistent_bindings: list[tuple[ProductRef, torch.device | str]] = []
    encoder_bindings: list[tuple[ProductRef, torch.device | str]] = []
    for operation in operations:
        device = _operation_device(runtime, operation)
        request = runtime.request_row(scope, operation.request_key.request_id)
        media = request.request.admission.diffusion
        decode = next(
            (
                params
                for params in scope.lane.decode_ranges
                if params.op_id == operation.op_id and params.request_key == operation.request_key
            ),
            None,
        )
        for output in operation.outputs:
            if (
                _operation_identity(operation) in scope.predicated_operations
                and output.kind is not ProductKind.COMPLETION
            ):
                continue
            if output.kind is ProductKind.TENSOR:
                layout = runtime.model.output_layout(
                    operation.entry,
                    output.output_index,
                    None if media is None else media.geometry,
                    decode,
                )
                if layout is None:
                    continue
                if layout.shape is not None:
                    shapes[output] = layout.shape
                if layout.region is not None:
                    regions[output] = layout.region
            if output.kind in {
                ProductKind.VISION_FEATURE,
                ProductKind.LATENT_FEATURE,
            }:
                encoder_bindings.append((output, device))
                continue
            if transfer.requires_device_product_binding(output):
                binding = (output, device)
                if output.uses_persistent_buffer():
                    persistent_bindings.append(binding)
                elif output.shape_bound.max_elements == 1:
                    scalar_groups.setdefault(
                        (device, output.kind, output.dtype, output.shape_bound),
                        [],
                    ).append(binding)
                else:
                    general_bindings.append(binding)
    groups = tuple(tuple(group) for group in scalar_groups.values())
    if general_bindings:
        groups = (*groups, tuple(general_bindings))
    if persistent_bindings:
        groups = (*groups, tuple(persistent_bindings))
    request_slots = {
        request.request.request_key: int(request.request.request_pool_idx)
        for request in scope.request_candidates
    }
    bound_groups = runtime.device_products.bind_output_groups(
        groups,
        regions=regions,
        shapes=shapes,
        request_slots=request_slots,
        buffer_allocations={params.buffer: params for params in scope.lane.buffer_allocations},
    )
    scope.device_writes.extend(write for binding in bound_groups for write in binding.writes)
    scope.encoder_writes.extend(
        runtime.encoder_cache.bind_outputs(
            tuple(encoder_bindings),
            buffer_allocations={params.buffer: params for params in scope.lane.buffer_allocations},
        )
    )
    operation_identities = {_operation_identity(operation) for operation in operations}
    token_operation_identities = {
        _operation_identity(operation)
        for operation in operations
        if operation.kind.token_mode is not None
    }
    for write in scope.device_writes:
        operation_identity = _reference_operation_identity(write.reference)
        if operation_identity not in operation_identities:
            raise RuntimeError("device output binding has no operation in the execution batch")
        if write.reference.kind is ProductKind.TOKEN:
            scope.token_writes[operation_identity] = write
            scope.operation_writes.setdefault(operation_identity, write)
        elif write.reference.kind is ProductKind.SELECTED_POINT:
            scope.selected_point_writes[operation_identity] = write
        elif (
            write.reference.kind is ProductKind.COMPLETION
            and operation_identity in token_operation_identities
            and int(write.reference.output_index) == 3
        ):
            scope.transition_writes[operation_identity] = write
        else:
            scope.operation_writes.setdefault(operation_identity, write)
        if (
            operation_identity in scope.predicated_operations
            and write.reference.kind is ProductKind.COMPLETION
        ):
            scope.propagated_predicate_writes.setdefault(operation_identity, ())
            scope.propagated_predicate_writes[operation_identity] = (
                *scope.propagated_predicate_writes[operation_identity],
                write,
            )


def _operation_device(runtime: Worker, operation: Operation) -> torch.device:
    """Resolve the execution device for an operation's model phase."""

    return (
        runtime._generation_device
        if operation.kind
        in {
            OpCode.DIFFUSION_PREPARE,
            OpCode.DIFFUSION_STEP,
            OpCode.DIFFUSION_FINALIZE,
        }
        else runtime._device
    )


def _validate_completion_products(
    runtime: Worker,
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


def _consume_predicates(
    runtime: Worker,
    operations: tuple[Operation, ...],
    scope: LaneState,
) -> None:
    """Resolve operation predicates from local device products and register their readers."""

    grouped: dict[
        torch.device,
        list[
            tuple[
                Operation,
                tuple[ProductRef, int, torch.device | str | None],
            ]
        ],
    ] = {}
    for operation in operations:
        predicate = operation.predicate
        if predicate is None:
            continue
        device = _operation_device(runtime, operation)
        grouped.setdefault(device, []).append(
            (
                operation,
                (
                    predicate,
                    int(operation.op_id),
                    device,
                ),
            )
        )
    for device, entries in grouped.items():
        reads = runtime.device_products.consume_batch(
            tuple(request for _operation, request in entries),
            device=device,
        )
        scope.device_reads.extend(reads)
        for (operation, _request), read in zip(entries, reads, strict=True):
            predicate = cast(ProductRef, operation.predicate)
            tagged = predicate.kind is ProductKind.TOKEN and predicate.dtype is DType.U32
            scope.predicate_values[_operation_identity(operation)] = (read.tensor, tagged)


def _publish_predicates(
    runtime: Worker,
    scope: LaneState,
) -> None:
    """Publish predicate outputs after their producing operations have resolved."""

    producers = {_operation_identity(operation) for operation in scope.lane.operations}
    transitions = {id(write) for write in scope.transition_writes.values()}
    propagated = {
        id(write) for writes in scope.propagated_predicate_writes.values() for write in writes
    }
    writes = tuple(
        write
        for write in scope.device_writes
        if write.reference.kind is ProductKind.COMPLETION
        and not write.producer_recorded
        and _reference_operation_identity(write.reference) in producers
        and id(write) not in transitions
        and id(write) not in propagated
    )
    if not writes:
        return
    batch = runtime.device_products.producer_scalar_batch(writes)
    if batch is not None:
        batch.tensor.fill_(1)
        runtime.device_products.publish_scalar_batch(batch)
        return
    views = runtime.device_products.producer_write_views(writes)
    first = views[0]
    runtime.device_products.publish_writes(
        writes,
        torch.ones(
            (len(writes),),
            dtype=first.dtype,
            device=first.device,
        ),
    )


def _publish_predicated_outputs(
    runtime: Worker,
    operations: tuple[Operation, ...],
    scope: LaneState,
) -> None:
    """Publish inactive sentinel values for products of predicated operations."""

    declared = {_operation_identity(operation) for operation in operations}
    for identity, writes in scope.propagated_predicate_writes.items():
        if identity not in declared:
            raise RuntimeError("predicated output has no operation in its lane")
        for write in writes:
            runtime.device_products.publish_scalar_write(write, False)


def _finish_device_reads(
    runtime: Worker,
    scope: LaneState,
) -> None:
    """Record or cancel every device-product read acquired by the lane."""

    reads = tuple(read for read in scope.device_reads if not read._recorded)
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
        runtime.device_products.record_readers(
            reads,
            after_writes=tuple(after_writes),
        )
    scope.device_reads.clear()
    encoder_reads = tuple(read for read in scope.encoder_reads if not read._recorded)
    if encoder_reads:
        runtime.encoder_cache.record_readers(encoder_reads)
    scope.encoder_reads.clear()


def _apply_release_controls(runtime: Worker, batch: Run, *, before_execution: bool) -> None:
    """Apply lifecycle releases in the phase required by their ownership contract."""

    consumed = {
        (reference.request_key, int(reference.producer_op_id))
        for operation in batch.operations
        for reference in (
            *operation.inputs,
            *(() if operation.predicate is None else (operation.predicate,)),
        )
    }
    releases = tuple(
        (operation.request_key, operation.parent.op_id)
        for operation in batch.operations
        if operation.parent is not None
        and operation.parent.op_id > 0
        and (
            ((operation.request_key, int(operation.parent.op_id)) not in consumed)
            == before_execution
        )
    )
    runtime.device_products.release_operations(releases)
    runtime.encoder_cache.release_operations(releases)
    if runtime.cache_publications is not None:
        released = runtime.cache_publications.release_operations(releases)
        if runtime.cache_pool is not None:
            runtime.cache_pool.release_buffers(released)
        for buffer in released:
            # Keep the registration until Free/Finish can observe its physical
            # retirement. Semantic release only revokes acquisition by new readers.
            _release_locators(runtime, runtime._transport_publications.get(buffer, ()))
    if before_execution:
        freed = {command.buffer for command in batch.commands if isinstance(command, Free)}
        closed = {
            command.request_key: frozenset(command.retained_buffers) - freed
            for command in batch.commands
            if isinstance(command, (Finish, Retire))
        }
        closing_publications = (
            tuple(
                buffer
                for buffer in runtime._transport_publications
                if buffer not in freed
                and buffer.owner in closed
                and buffer not in closed[buffer.owner]
            )
            if closed
            else ()
        )
        buffers = (*freed, *closing_publications)
        runtime.release_buffers(buffers)
        if runtime.cache_pool is not None:
            for request_key, retained in closed.items():
                runtime.cache_pool.imports.cancel_requests(
                    frozenset((request_key,)), retained=retained
                )
    if not before_execution:
        consumed_predicates = tuple(
            predicate.buffer_id
            for operation in batch.operations
            if (predicate := operation.predicate) is not None
            and (operation.parent is None or predicate.producer_op_id != operation.parent.op_id)
        )
        runtime.device_products.release_buffers(consumed_predicates)


def drop_request(
    runtime: Worker, request_id: int, *, retained: frozenset[BufferId] = frozenset()
) -> None:
    """Release request state and publications whose allocation ownership ends with it."""

    request = runtime.requests.peek(int(request_id))
    if request is not None:
        if runtime.cache_pool is not None:
            runtime.cache_pool.imports.cancel_requests(
                frozenset((request.request_key,)), retained=retained
            )
        if runtime.runtime_states is not None:
            runtime.runtime_states.release((int(request.request_pool_idx),))
        if runtime.req_to_token_pool is not None:
            runtime.req_to_token_pool.release((int(request.request_pool_idx),))
    if runtime.cache_publications is not None:
        runtime.cache_publications.drop(request_id)
    if request is not None and runtime.req_to_token_pool is not None:
        runtime.req_to_token_pool.release(
            tuple(runtime._flow_prefix_slots.pop(request.request_key, ()))
        )
    if not runtime.transports:
        return
    selected = tuple(
        identity
        for identity in runtime._transport_publications
        if int(identity.owner.request_id) == int(request_id) and identity not in retained
    )
    for identity in selected:
        _release_locators(runtime, runtime._transport_publications.pop(identity))
    if runtime.cache_pool is not None:
        runtime.cache_pool.release_buffers(selected)


def _bind_latent_rows(
    runtime: Worker,
    lane: RunLane,
    scope: LaneState,
) -> None:
    """Validate trajectory parameters and bind rank-local latent staging views."""

    if not lane.latent_params:
        return
    pool = runtime.latent_pool
    operations = {
        _operation_identity(operation): (
            operation,
            runtime.request_row(scope, operation.request_key.request_id),
        )
        for operation in lane.operations
    }
    if pool is None:
        # Fixed request tensors own the trajectory directly. Solver progress
        # remains explicit, without a second paged-storage reservation.
        for params in lane.latent_params:
            identity = params.request_key, int(params.op_id)
            selected = operations.get(identity)
            if selected is None:
                raise invalid_descriptor("latent params names an operation outside its lane")
            operation, request = selected
            if params.page_table or params.latent_units:
                raise invalid_descriptor("paged latent params require a resident latent pool")
            if operation.kind is OpCode.DIFFUSION_PREPARE:
                valid = int(params.start_step) == 0 and int(params.step_count) == 0
            elif operation.kind is OpCode.DIFFUSION_STEP:
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
        identity = (params.request_key, int(params.op_id))
        selected = operations.get(identity)
        if selected is None:
            raise invalid_descriptor("latent params names an operation outside its lane")
        operation, request = selected
        slot = int(request.request.request_pool_idx)
        image = request.request.image
        if image is None:
            raise invalid_descriptor("latent params has no admitted image geometry")
        flow = _generation(
            runtime,
        )
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
                for reference in operation.inputs
                if (prepared := scope.prepared_transfers.get(reference)) is not None
                and isinstance(prepared.value, LatentTransferValue)
            ),
            None,
        )
        committed_step = int(request.flow_step) if transferred is None else transferred.step
        if operation.kind is OpCode.DIFFUSION_PREPARE:
            if int(params.start_step) != 0 or int(params.step_count) != 0:
                raise invalid_descriptor("media preparation params carries denoise steps")
        elif operation.kind is OpCode.DIFFUSION_STEP:
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
    runtime: Worker,
    lane: RunLane,
    scope: LaneState,
) -> None:
    """Install scheduler tables and retain row-aligned forward coordinates."""

    cache = runtime.cache_pool
    page_tables = runtime.req_to_token_pool
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

    rows_by_operation: dict[int, list] = defaultdict(list)
    for row in lane.forward_rows:
        rows_by_operation[int(row.operation_index)].append(row)

    for operation_index, operation in enumerate(lane.operations):
        request = runtime.request_row(scope, operation.request_key.request_id)
        main_slot = int(request.request.request_pool_idx)
        parent_runtime = request.request.parent_runtime(operation.parent)
        operation_rows = rows_by_operation.get(operation_index, [])
        scope.forward_rows[_operation_identity(operation)] = tuple(operation_rows)
        main_descriptor = next(
            (row for row in operation_rows if int(row.request_pool_index) == main_slot),
            None,
        )
        if main_descriptor is not None and int(main_descriptor.seq_len) != int(
            parent_runtime.kv_visible_len
        ):
            raise invalid_descriptor("forward row sequence length disagrees with its parent")
        for descriptor in operation_rows:
            slot = int(descriptor.request_pool_index)
            pages = page_tables.pages(slot, 0)
            if slot != main_slot and int(descriptor.seq_len) > page_tables.allocated_length(slot):
                raise invalid_descriptor("forward row exceeds alternative-prefix capacity")
            if slot != main_slot:
                runtime._flow_prefix_slots.setdefault(operation.request_key, set()).add(slot)
            cache.retain_execution(
                operation.request_key,
                pages,
                group=0,
                length=int(descriptor.seq_len)
                + (int(descriptor.query_len) if descriptor.write_kv else 0),
                completion=scope.completion.completion_future(),
            )
    _record_component(scope, "bc_tables", started)


def _stage_input_products(
    runtime: Worker,
    input_products: Sequence[ProductPayload],
    scope: LaneState,
) -> None:
    """Decode ephemeral host inputs and publish transferred physical values."""

    for entry in input_products:
        product = entry.product
        if isinstance(entry.payload, TransferHandle):
            # Transfer metadata determines which runtime owns the imported value;
            # each branch validates identity and geometry before publication.
            transfer = scope.prepared_transfers.get(product)
            if transfer is None or not transfer.ready():
                raise invalid_descriptor("cross-stage input has no query-ready prepared transfer")
            value = transfer.value
            if isinstance(value, KvTransferValue):
                snapshot = value
                publications = runtime.cache_publications
                if publications is None:
                    raise invalid_descriptor("KV input requires cache publication storage")
                existing = publications.resident(product)
                if existing is not None and existing != snapshot:
                    raise invalid_descriptor(
                        "staged KV publication conflicts with its product identity"
                    )
                staged = scope.cache_publication_inputs.get(product)
                if staged is not None and staged != snapshot:
                    raise invalid_descriptor(
                        "batch repeats a KV product with conflicting publication data"
                    )
                scope.cache_publication_inputs[product] = snapshot
                continue
            if isinstance(value, LatentTransferValue):
                consumers = tuple(
                    operation for operation in scope.lane.operations if product in operation.inputs
                )
                if len(consumers) != 1:
                    raise invalid_descriptor("latent transfer must have one lane consumer")
                row = runtime.latent_row(consumers[0], scope)
                if (
                    value.latent_units != int(row.params.latent_units)
                    or value.height != int(row.params.height)
                    or value.width != int(row.params.width)
                    or value.step != int(row.params.start_step)
                    or value.generation != int(product.generation)
                ):
                    raise invalid_descriptor("latent transfer disagrees with its scheduler params")
                request = runtime.request_row(scope, product.request_key.request_id)
                if request.latent_product is not None or int(request.flow_step) != 0:
                    raise invalid_descriptor(
                        "latent transfer destination already owns a trajectory"
                    )
                binding = transfer.destination
                if not isinstance(binding, LatentWrite):
                    raise RuntimeError("latent transfer lost its reserved destination")
                if (binding.request_pool_idx, binding.page_table) != (
                    row.request_pool_idx,
                    tuple(row.params.page_table),
                ):
                    raise invalid_descriptor("latent import reservation changed before execution")
                _latent_pool(runtime).adopt_import(
                    binding,
                    generation=value.generation,
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
                if product in operation.inputs or operation.predicate == product
            )
            if not consumers:
                raise invalid_descriptor("transferred product has no lane consumer")
            devices = {_operation_device(runtime, operation) for operation in consumers}
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
                runtime.encoder_cache.publish(
                    encoder_binding,
                    tensors[0],
                    EncoderMetadata(height=value.height, width=value.width),
                )
                runtime.encoder_cache.commit_writes((encoder_binding,))
                transfer.adopt_destination()
            continue
        # Inline payloads remain host-owned until their consuming operation stages them.
        if product.kind is ProductKind.SAMPLING_STATE:
            scope.sampling_states[_reference_operation_identity(product)] = (
                decode_sampling_state_bytes(entry.payload)
            )
            continue
        if product.kind is ProductKind.TOKEN:
            scope.input_tokens[product] = decode_token_product_bytes(entry.payload)
            continue
        if product.kind is not ProductKind.ARTIFACT:
            raise invalid_descriptor("host-staging payload has no concrete product owner")
        if product.storage_class is not StorageClass.HOST_STAGING or not entry.payload:
            raise invalid_descriptor("source image payload has invalid storage metadata")
        scope.input_images[product] = entry.payload.decode("utf-8")


def _predicated_outcome(
    runtime: Worker,
    operation: Operation,
    scope: LaneState,
) -> Outcome:
    """Construct an inactive outcome while preserving declared product generations."""

    request = runtime.request_row(scope, operation.request_key.request_id)
    lengths = runtime.logical_lengths(operation, request, None)
    selected = request.request.resolve_version(operation.parent)
    if operation.parent is not None and (
        selected is None or not isinstance(selected.point, FixedCheckpoint)
    ):
        raise invalid_descriptor("predicated operation parent has no selected fixed checkpoint")
    selected_point = (
        0 if selected is None else int(cast(FixedCheckpoint, selected.point).point_index)
    )
    return Outcome(
        status=OpStatus.PREDICATED,
        selected_point=selected_point,
        logical_lengths=lengths,
        token_span=TokenSpan(base=int(lengths.token_len), len=0),
        finish_flags=FinishFlags(),
        product_generations=(),
    )


def _run_laneed_wave(
    runtime: Worker,
    tasks: tuple[tuple[ForwardRow, LaneState], ...],
    *,
    qualify_mixed: bool,
) -> tuple[torch.Tensor, ...]:
    """Group compatible forward rows, execute each group, and restore task order."""

    # Launch identity, physical lane, and shape key jointly define rows that may
    # share one model invocation without changing scheduler ordering.
    grouped: dict[
        tuple[object, ...],
        list[tuple[int, ForwardRow, LaneState]],
    ] = defaultdict(list)
    for index, (task, scope) in enumerate(tasks):
        grouped[
            (
                scope.lane.launch_id,
                _lane_identity(
                    runtime.runner, _phase_device(runtime, task.phase), scope.lane.domain
                ),
                *_group_key(runtime, task),
            )
        ].append((index, task, scope))

    result: list[torch.Tensor | None] = [None] * len(tasks)
    output_events: list[tuple[torch.device, torch.cuda.Event]] = []
    for group in grouped.values():
        kinds = frozenset(task.kind for _index, task, _scope in group)
        if (
            len(kinds) > 1
            and not _model(
                runtime,
            ).tensorized_mixed
        ):
            raise invalid_descriptor("tensorized mixed submission is outside the model limits")
        indexes = tuple(index for index, _task, _scope in group)
        group_tasks = tuple(task for _index, task, _scope in group)
        group_scopes = tuple(scope for _index, _task, scope in group)
        target = _phase_device(runtime, group_tasks[0].phase)
        for scope in _unique_scopes(group_scopes):
            scope.completion.register_device(target)
        if qualify_mixed and len(kinds) > 1:
            # Measure whether tensorizing these rows improves service time.
            # Numerical conformance belongs to independent model/operator
            # tests, not a comparison against another batch shape at startup.
            output, observation, mixed_us = _run_startup_forward(
                runtime,
                group_tasks,
                group_scopes[0],
                target,
            )
            mixed_output = tuple(value.clone() for value in output)
            homogeneous: dict[
                str,
                list[tuple[ForwardRow, LaneState]],
            ] = defaultdict(list)
            for _index, task, scope in group:
                homogeneous[task.kind].append((task, scope))
            homogeneous_us: list[int] = []
            for members in homogeneous.values():
                _reference, _reference_observation, reference_us = _run_startup_forward(
                    runtime,
                    tuple(task for task, _scope in members),
                    members[0][1],
                    target,
                    force_eager=observation.path is RunPath.EAGER,
                )
                homogeneous_us.append(reference_us)
            service_paths = {RunPath.EAGER, RunPath.GRAPH_REPLAY}
            if observation.path in service_paths:
                serial_us = sum(homogeneous_us)
                bucket = _mixed_bucket(
                    runtime, tuple(scope.lane for scope in _unique_scopes(group_scopes))
                )
                speedup = serial_us / max(1, mixed_us)
                qualified = runtime.runner.qualify_mixed(
                    bucket, target.type != "cuda" or speedup >= _MIN_MIXED_SERVICE_SPEEDUP
                )
                logger.info(
                    "evaluated mixed execution bucket=%r mixed_us=%d homogeneous_us=%r "
                    "serial_over_mixed=%.3f service_eligible=%s",
                    bucket,
                    mixed_us,
                    tuple(homogeneous_us),
                    speedup,
                    qualified,
                )
            output = mixed_output
            output_event = None
        else:
            forward_result = _run_forward_group(runtime, group_tasks, group_scopes[0])
            output = forward_result.materialize_values()
            observation = forward_result.observation
            output_event = forward_result.output_event
        group_scopes[0].observations.append(observation)
        if output_event is not None:
            output_events.append((target, output_event))
        # Scatter each grouped result back to the caller's task ordering.
        for index, value in zip(indexes, output, strict=True):
            result[index] = value
    for device, event in output_events:
        torch.cuda.current_stream(device).wait_event(event)
    return tuple(cast(torch.Tensor, value) for value in result)


def _run_startup_forward(
    runtime: Worker,
    tasks: tuple[ForwardRow, ...],
    scope: LaneState,
    target: torch.device,
    *,
    force_eager: bool = False,
) -> tuple[tuple[torch.Tensor, ...], RunObservation, int]:
    """Execute startup forward rows eagerly or through graph qualification without publication."""

    with profile_range("uniserve.startup.mixed_service_measurement"):
        if target.type != "cuda":
            started = time.perf_counter_ns()
            result = _run_forward_group(runtime, tasks, scope, force_eager=force_eager)
            elapsed_us = max(1, (time.perf_counter_ns() - started) // 1000)
        else:
            stream = torch.cuda.current_stream(target)
            start = torch.cuda.Event(blocking=False, enable_timing=True)
            end = torch.cuda.Event(blocking=False, enable_timing=True)
            start.record(stream)
            result = _run_forward_group(runtime, tasks, scope, force_eager=force_eager)
            if result.output_event is not None:
                stream.wait_event(result.output_event)
            end.record(stream)
            end.synchronize()
            elapsed_us = max(1, round(float(start.elapsed_time(end)) * 1000.0))
        return result.materialize_values(), result.observation, elapsed_us


def _run_observed_forward_group(
    runtime: Worker,
    tasks: tuple[ForwardRow, ...],
    scope: LaneState,
) -> ForwardResult:
    """Run a forward group while recording timing and operation trace metadata."""

    result = _run_forward_group(runtime, tasks, scope)
    scope.observations.append(result.observation)
    return result


def _group_key(runtime: Worker, task: ForwardRow) -> tuple[object, ...]:
    """Build the route, mode, geometry, and weight identity used to batch forward rows."""

    phase = (
        "textual"
        if task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE}
        and _model(
            runtime,
        ).tensorized_mixed
        else task.phase.value
    )
    return (
        phase,
        str(_phase_device(runtime, task.phase)),
        task.weights.version,
        (
            ()
            if _model(
                runtime,
            ).tensorized_mixed
            and not runtime.runner.uses_lanes
            else _task_shape(runtime, task)
        ),
    )


def _lane_identity(runner: ModelRunner, device: torch.device, domain: Domain) -> int:
    """Return the physical runner, device, and domain identity used for lane grouping."""

    for lane in runner.execution_lanes:
        if lane.device == device and domain in lane.domains:
            return id(lane)
    raise invalid_descriptor(f"execution has no {domain.value!r} lane for {device}")


def _run_forward_group(
    runtime: Worker,
    tasks: tuple[ForwardRow, ...],
    scope: LaneState,
    *,
    force_eager: bool = False,
) -> ForwardResult:
    """Execute one compatible forward-row group and register model output products."""

    target = _phase_device(runtime, tasks[0].phase)
    scope.completion.register_device(target)
    attention = (
        _attention_columns(runtime, tasks)
        if all(task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE} for task in tasks)
        else _dense_attention_columns(len(tasks), tuple(task.query_tokens for task in tasks))
    )
    result = runtime.runner.run(
        tasks,
        device=target,
        attention=attention,
        graph_shape=_group_graph_shape(runtime, tasks),
        graph_eligible=(
            scope.graph_eligible
            and not force_eager
            and all(task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE} for task in tasks)
        ),
        domain=scope.lane.domain,
    )
    if int(result.request_pool_indices.numel()) != len(tasks):
        raise RuntimeError("model runner returned without aligned request slots")
    for index, task in enumerate(tasks):
        task.request_pool_index = result.request_pool_indices[index : index + 1]
    return result


def _weights(runtime: Worker) -> WeightSet:
    """Return the runtime's installed live-weight registry."""

    return runtime.weights


def _model(runtime: Worker) -> ExecutionModel:
    """Return the execution model currently bound to the runtime."""

    return runtime.model


def _generation(runtime: Worker) -> GenerationPipeline:
    """Require and return the model's diffusion-generation pipeline."""

    value = _model(
        runtime,
    ).generation
    if not isinstance(value, GenerationPipeline):
        raise invalid_descriptor("operation requires model generation behavior")
    return value


def _latent_pool(runtime: Worker) -> LatentPool:
    """Require and return runtime-owned latent trajectory storage."""

    if runtime.latent_pool is None:
        raise unsupported_setup("operation requires a physical latent pool")
    return runtime.latent_pool


def _image_processor(runtime: Worker) -> ImageProcessor:
    """Require and return the model's image preprocessing contract."""

    value = _model(
        runtime,
    ).image_processor
    if not isinstance(value, ImageProcessor):
        raise invalid_descriptor("operation requires model image processing")
    return value


def _phase_device(runtime: Worker, phase: ModelPhase) -> torch.device:
    """Resolve the model device responsible for an execution phase."""

    worker_config = runtime.worker_config
    if phase in {ModelPhase.ENCODE_LATENT, ModelPhase.DECODE_LATENT}:
        return torch.device(worker_config.generation_device or worker_config.device)
    return torch.device(worker_config.device)


def _phase_topology(runtime: Worker, phase: ModelPhase) -> tuple[str, ...]:
    """Resolve the distributed mesh axes used by an execution phase."""

    if phase in {ModelPhase.TEXT, ModelPhase.DENOISE}:
        return _model(
            runtime,
        ).text_topology
    return ("tp",)


def _task_shape(runtime: Worker, task: ForwardRow) -> tuple[int, ...]:
    """Build the graph-relevant shape signature for one forward row."""

    if task.encode_pixels is not None:
        return tuple(int(value) for value in task.encode_pixels.shape)
    if task.latent is not None:
        return task.image_height, task.image_width
    return ()


def _group_graph_shape(runtime: Worker, tasks: tuple[ForwardRow, ...]) -> tuple[object, ...]:
    """Require one shared graph-shape signature across grouped forward rows."""

    return (
        len(tasks),
        sum(task.query_tokens for task in tasks),
        tuple(task.query_tokens for task in tasks),
        tuple((task.image_height, task.image_width) for task in tasks if task.latent is not None),
    )


def _release_locators(runtime: Worker, locators: Iterable[Locator]) -> None:
    """Release transfer locators through the runtime transport owner."""

    if not runtime.transports:
        return
    for locator in locators:
        runtime.transports[locator.backend].release(locator)


def _trace_envelopes(
    operations: Sequence[Operation],
) -> tuple[OperationTrace, ...]:
    """Encode operation identities and kinds for execution-trace records."""

    return tuple(
        OperationTrace(
            authority_id=int(operation.request_key.authority_id),
            request_id=int(operation.request_key.request_id),
            epoch=int(operation.request_key.epoch),
            op_id=int(operation.op_id),
            version=int(point.point_index) if isinstance(point, FixedCheckpoint) else 0,
        )
        for operation in operations
        for point in (None if operation.parent is None else operation.parent.point,)
    )


def _fixed_parent(operation: Operation) -> FixedCheckpoint:
    """Return the fixed parent point a depth-one operation commits over."""

    point = operation.state_parent.point
    if not isinstance(point, FixedCheckpoint):
        raise invalid_descriptor("operation names a device parent; depth one commits fixed")
    return point


def _record_component(scope: LaneState, name: str, started_ns: int) -> None:
    """Accumulate elapsed microseconds for one lane execution component."""

    elapsed_us = max(0, (time.perf_counter_ns() - int(started_ns)) // 1000)
    scope.component_us[name] = scope.component_us.get(name, 0) + elapsed_us


def _forward_stats(
    observations: Sequence[RunObservation],
    component_us: Mapping[str, int] | None = None,
) -> WorkerForwardStats:
    """Aggregate forward observations into stable per-component and total timing statistics."""

    route_counts: dict[str, int] = {}
    route_rows: dict[str, int] = {}
    route_us: dict[str, int] = {}
    path_counts: dict[str, int] = {}
    captures = 0
    replays = 0
    fallbacks = 0
    graph_unpadded_tokens = 0
    graph_padded_tokens = 0
    for observation in observations:
        route_counts[observation.route] = route_counts.get(observation.route, 0) + 1
        route_rows[observation.route] = route_rows.get(observation.route, 0) + int(
            observation.row_count
        )
        route_us[observation.route] = route_us.get(observation.route, 0) + int(
            observation.duration_us
        )
        path_counts[observation.path.value] = path_counts.get(observation.path.value, 0) + 1
        captures += observation.path is RunPath.GRAPH_CAPTURE
        replays += observation.path is RunPath.GRAPH_REPLAY
        fallbacks += observation.path is RunPath.GRAPH_FALLBACK
        graph_unpadded_tokens += int(observation.graph_unpadded_tokens)
        graph_padded_tokens += int(observation.graph_padded_tokens)
    components: dict[str, int] = {}
    if observations:
        components["forward"] = sum(route_us.values())
    for name, value in (component_us or {}).items():
        components[str(name)] = components.get(str(name), 0) + max(0, int(value))
    return WorkerForwardStats(
        mode_counts=route_counts,
        mode_tokens=route_rows,
        mode_us=route_us,
        component_us=components,
        cuda_graph_captures=int(captures),
        cuda_graph_replays=int(replays),
        cuda_graph_misses=int(fallbacks),
        cuda_graph_fallbacks=int(fallbacks),
        cuda_graph_unpadded_tokens=graph_unpadded_tokens,
        cuda_graph_padded_tokens=graph_padded_tokens,
        cuda_graph_runtime_mode_counts=path_counts,
    )


__all__ = [
    "PreparedExecution",
    "complete_startup",
    "drop_request",
    "execute_batch",
    "execute_prepared",
    "execute_startup",
    "prepare_batch",
]
