"""Bounded immutable device values with generation-safe stream lifetimes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import RLock
from typing import Final, cast

import torch

from uniserve.runtime.device import canonical_device
from uniserve.tensors import ImageRange, TensorRegion
from uniserve_worker.protocol.batch import ComputationId

from ..foundation.errors import WorkerError, WorkerErrorCode, invalid_descriptor, resource_error
from ..protocol.batch import (
    BufferAllocation,
    BufferId,
    DType,
    RequestKey,
    StaticDim,
    TensorRef,
    TensorTransfer,
    WorkerEndpoint,
)
from ..transfer.exports import ExportLocations, release_exports
from ..transfer.layout import fetch_tensor
from ..transfer.tickets import TransferTicket, Transport
from .buffer_pool import BufferBinding, BufferPool
from .device_events import EventPool

_MAX_GENERATION: Final[int] = (1 << 32) - 1
_DEVICE_DTYPES: Final[dict[DType, torch.dtype]] = {
    DType.U8: torch.uint8,
    DType.I32: torch.int32,
    DType.I16: torch.int16,
    DType.I64: torch.long,
    DType.F16: torch.float16,
    DType.BF16: torch.bfloat16,
    DType.F32: torch.float32,
}
_DEVICE_TORCH_DTYPES: Final[tuple[torch.dtype, ...]] = tuple(dict.fromkeys(_DEVICE_DTYPES.values()))
_DTYPE_STORAGE: Final[dict[DType, tuple[str, int]]] = {
    dtype: (
        str(torch_dtype).removeprefix("torch."),
        int(torch.empty((), dtype=torch_dtype).element_size()),
    )
    for dtype, torch_dtype in _DEVICE_DTYPES.items()
}
_TORCH_DTYPE_BYTES: Final[dict[torch.dtype, int]] = {
    torch_dtype: int(torch.empty((), dtype=torch_dtype).element_size())
    for torch_dtype in _DEVICE_TORCH_DTYPES
}


def device_product_storage(dtype: DType) -> tuple[str, int]:
    """Return the concrete tensor storage used for one product dtype."""

    return _DTYPE_STORAGE[DType(dtype)]


def device_product_capacity_bytes(
    slot_capacity: int,
    device_count: int,
    *,
    max_value_bytes: int,
) -> int:
    """Return the fixed backing bound for one ``TensorStore`` owner."""

    slots = int(slot_capacity)
    devices = int(device_count)
    value_bytes = int(max_value_bytes)
    if min(slots, devices, value_bytes) < 1:
        raise ValueError("device-product geometry must be positive")
    scalar_bytes = slots * devices * sum(dict(_DTYPE_STORAGE.values()).values())
    return scalar_bytes + slots * devices * value_bytes


def _invariant(message: str) -> WorkerError:
    """Construct a classified invariant error for device-product misuse."""

    return WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


_ReferenceKey = tuple[int, int, int, ComputationId, int]
_OperationKey = tuple[RequestKey, ComputationId]
_SlotStorageKey = tuple[str, tuple[int, ...], torch.dtype]


def _reference_key(reference: TensorRef) -> _ReferenceKey:
    """Build the logical identity whose live record validates the full generation."""

    key = reference.request_key
    return (
        int(key.engine_id),
        int(key.request_id),
        int(key.request_epoch),
        reference.producer_op_id,
        int(reference.output_index),
    )


def _device_dtype(dtype: DType) -> torch.dtype:
    """Resolve a product dtype descriptor to its torch dtype."""

    return _DEVICE_DTYPES[dtype]


def _device_shape(reference: TensorRef) -> tuple[int, ...]:
    """Resolve a product's bounded tensor dimensions to a concrete shape."""

    dims = tuple(
        dim.extent if isinstance(dim, StaticDim) else dim.bound
        for dim in reference.shape_bound.dims
    )
    return dims or (1,)


def _event_ready(event: torch.cuda.Event | None) -> bool:
    """Return whether an optional CUDA event is query-ready."""

    return event is None or bool(event.query())


@dataclass(slots=True)
class RelaySlot:
    """Tracks ownership, generation, storage geometry, and relay binding for one device-product slot."""

    index: int
    device_name: str
    generation: int = 0
    owner: int | None = None
    tensor: torch.Tensor | None = None
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    relay_lane: tuple[str, int, RequestKey, ComputationId, int] | None = None


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Spatial geometry and numerical value range of an immutable image tensor."""

    height: int = 0
    width: int = 0
    value_range: ImageRange | None = None

    def __post_init__(self) -> None:
        """Require complete, nonnegative image geometry."""

        if self.height < 0 or self.width < 0:
            raise ValueError("device-product image geometry must be non-negative")
        if (self.height == 0) != (self.width == 0):
            raise ValueError("device-product image geometry must be complete")


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Spatial geometry of an immutable floating-point encoder feature."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if min(self.height, self.width) < 1:
            raise ValueError("encoder feature geometry must be positive")


@dataclass(slots=True)
class TensorRecord:
    """One table-issued physical binding retained through producer submission."""

    reference: TensorRef
    tensor: torch.Tensor
    device_name: str
    shape: tuple[int, ...]
    physical_generation: int
    binding_id: int
    buffer_binding: BufferBinding | None = None
    relay_slot: RelaySlot | None = None
    feature: bool = False
    retired: bool = False
    # Publication exposes the logical product; release may precede retirement.
    committed: bool = False
    region: TensorRegion | None = None
    logical_shape: tuple[int, ...] | None = None
    transfers: tuple[TransferTicket, ...] = ()
    publications: tuple[Future[None], ...] = ()
    readers: int = 0
    producer_event: torch.cuda.Event | None = None
    producer_stream: int | None = None
    producer_recorded: bool = False
    reader_events: torch.cuda.Event | list[torch.cuda.Event] | None = None
    released: bool = False
    _indexed: bool = False
    actual_extent: int = 0
    actual_shape: tuple[int, ...] = ()
    metadata: ImageMetadata | FeatureMetadata | None = None


@dataclass(slots=True)
class TensorRead:
    """One generation-validated device read retained until its stream snapshots it."""

    tensor: torch.Tensor
    consumer_op_id: ComputationId | None
    _write: TensorRecord = field(repr=False, compare=False)
    region: TensorRegion | None = None
    metadata: ImageMetadata | FeatureMetadata | None = None
    imported: TensorImport | None = field(default=None, repr=False, compare=False)
    _recorded: bool = field(default=False, repr=False, compare=False)


@dataclass(slots=True)
class TensorImport:
    """One shared fill of missing immutable regions, retained by consumer reads."""

    write: TensorRecord
    tensor: torch.Tensor
    tickets: tuple[TransferTicket, ...]
    metadata: ImageMetadata | FeatureMetadata | None
    users: int = 0
    committed: bool = False


class TensorStore:
    """Bounded physical slots for generation-tagged device products.

    Registration, lookup, stream waits, reader recording, release, and
    reclamation all validate the exact logical and physical generation here.
    Reclamation only queries events; it never synchronizes a device stream.
    """

    def __init__(
        self,
        *,
        capacity: int = 0,
        byte_capacity: int | None = None,
        entry_capacity: int = 0,
        max_entry_bytes: int = 1,
        devices: tuple[torch.device | str, ...] = (),
        request_capacity: int = 0,
        relay_depth: int = 0,
        buffer_pool: BufferPool,
        event_pool: EventPool | None = None,
    ) -> None:
        """Initialize bounded product registries, relay arenas, and event ownership."""

        # Validate independent slot-byte capacity and the coupled request-relay
        # geometry before creating any registries.
        self.capacity = int(capacity)
        self.entry_capacity = int(entry_capacity)
        self.max_entry_bytes = int(max_entry_bytes)
        self.devices = tuple(canonical_device(device) for device in devices)
        if self.capacity < 0 or self.entry_capacity < 0 or self.max_entry_bytes < 1:
            raise ValueError("tensor store capacities are invalid")
        self.byte_capacity = int(
            buffer_pool.byte_capacity if byte_capacity is None else byte_capacity
        )
        if self.byte_capacity < 1:
            raise ValueError("device-product byte capacity must be positive")
        self.request_capacity = int(request_capacity)
        self.relay_depth = int(relay_depth)
        self.buffer_pool = buffer_pool
        if (self.request_capacity == 0) != (self.relay_depth == 0):
            raise ValueError("request-relay geometry must be complete")
        if self.request_capacity < 0 or self.relay_depth < 0:
            raise ValueError("request-relay geometry must not be negative")

        # Storage pools are partitioned by device and tensor geometry; relay
        # arenas reserve stable request/lane addresses for graph capture.
        self._allocated_bytes = 0
        self._relay_arenas: dict[tuple[str, torch.dtype, int], torch.Tensor] = {}
        # Group fields by physical lane so allocation and retirement inspect only
        # this request's owners, independently of other admitted requests.
        self._relay_slots: dict[tuple[str, int, int], dict[tuple[torch.dtype, int], RelaySlot]] = {}
        self._relay_operation_lanes: dict[tuple[str, int, RequestKey, ComputationId], int] = {}

        # Both reserved and committed products retain their logical identity
        # until physical retirement. Only committed records are consumable.
        # Physical handles are validated independently of logical publication.
        self.event_pool = EventPool() if event_pool is None else event_pool
        self._products: dict[_ReferenceKey, TensorRecord] = {}
        self._writes: dict[int, TensorRecord] = {}
        self._imports: dict[_ReferenceKey, TensorImport] = {}
        self._operation_writes: dict[
            _OperationKey,
            TensorRecord | list[TensorRecord],
        ] = {}
        self.exports: dict[BufferId, ExportLocations] = {}
        self.export_releases: dict[BufferId, tuple[Future[None], ...]] = {}
        self._next_binding_id = 1
        self._lock = RLock()

    def resident_bytes(self, device: torch.device | str) -> int:
        """Return retained backing bytes on a device, counting shared arenas once."""

        name = str(torch.device(device))
        with self._lock:
            tensors = [
                tensor
                for (owner, _dtype, _width), tensor in self._relay_arenas.items()
                if owner == name
            ]
            tensors.extend(
                entry.tensor for entry in self._writes.values() if entry.device_name == name
            )
            storages = {
                tensor.untyped_storage().data_ptr(): tensor.untyped_storage() for tensor in tensors
            }
            return sum(storage.nbytes() for storage in storages.values())

    def close(self) -> None:
        """Release all resident product slots, events, relay storage, and persistent bindings."""

        with self._lock:
            self.exports.clear()
            self.export_releases.clear()
            self._imports.clear()
            self._products.clear()
            self._writes.clear()
            self._operation_writes.clear()
            self._relay_arenas.clear()
            self._relay_slots.clear()
            self._relay_operation_lanes.clear()
            self._allocated_bytes = 0

    @staticmethod
    def _tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        """Compute bytes required for a tensor shape and dtype."""

        return int(math.prod(shape)) * _TORCH_DTYPE_BYTES[dtype]

    def _require_byte_capacity_locked(self, projected: int) -> None:
        """Reject a projected persistent allocation above the configured byte capacity."""

        if projected > self.byte_capacity:
            raise resource_error(
                f"device-product byte capacity is exhausted ({projected}>{self.byte_capacity})"
            )

    def bind_outputs(
        self,
        bindings: tuple[tuple[TensorRef, torch.device | str], ...],
        *,
        request_slots: Mapping[RequestKey, int] | None = None,
        buffer_allocations: Mapping[BufferId, BufferAllocation] | None = None,
        regions: Mapping[TensorRef, TensorRegion] | None = None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None = None,
    ) -> tuple[TensorRecord, ...]:
        """Atomically bind outputs and retain their direct scalar range."""

        device_bindings = bindings
        if not device_bindings:
            return ()

        # Validate logical extents before acquiring any physical storage.
        for reference, _device in device_bindings:
            region = None if regions is None else regions.get(reference)
            logical_shape = (
                _device_shape(reference)
                if shapes is None
                else shapes.get(reference, _device_shape(reference))
            )
            if not reference.shape_bound.contains_shape(logical_shape):
                raise invalid_descriptor("tensor binding shape disagrees with its logical bounds")
            if region is not None and not region.within(logical_shape):
                raise invalid_descriptor("tensor binding region disagrees with its logical bounds")
        # Allocation records select physical ownership. Tensor identity carries
        # no semantic role or duplicate storage-class tag.
        persistent = tuple(
            buffer_allocations is not None and reference.buffer_id in buffer_allocations
            for reference, _device in device_bindings
        )
        if any(persistent):
            if not all(persistent):
                raise invalid_descriptor("persistent-buffer bindings cannot share a generic group")
            assert buffer_allocations is not None
            return self._bind_persistent_outputs(
                device_bindings, buffer_allocations, regions, shapes
            )
        if request_slots is not None:
            return self._bind_relay_outputs(device_bindings, request_slots)
        raise invalid_descriptor("tensor output requires a buffer allocation or request relay slot")

    def reserve_features(
        self,
        bindings: tuple[tuple[TensorRef, torch.device | str], ...],
        *,
        buffer_allocations: Mapping[BufferId, BufferAllocation],
        regions: Mapping[TensorRef, TensorRegion] | None = None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None = None,
    ) -> tuple[TensorRecord, ...]:
        """Reserve features against their independent global entry and byte bounds."""

        return self._bind_persistent_outputs(
            bindings, buffer_allocations, regions, shapes, feature=True
        )

    def _bind_persistent_outputs(
        self,
        bindings: tuple[tuple[TensorRef, torch.device | str], ...],
        allocations: Mapping[BufferId, BufferAllocation],
        regions: Mapping[TensorRef, TensorRegion] | None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None,
        *,
        feature: bool = False,
    ) -> tuple[TensorRecord, ...]:
        """Reserve scheduler placements without a second persistent-slot allocator."""

        keys = tuple(_reference_key(reference) for reference, _device in bindings)
        if len(set(keys)) != len(keys):
            raise invalid_descriptor("tensor registration repeats an output identity")
        with self._lock:
            self._reclaim_ready_locked()
            records = self._writes.values()
            counts: dict[str, int] = {}
            for entry in records:
                if not entry.feature and entry.relay_slot is None:
                    counts[entry.device_name] = counts.get(entry.device_name, 0) + 1
            if (
                feature
                and sum(entry.feature for entry in records) + len(bindings) > self.entry_capacity
            ):
                raise resource_error("encoder cache has no query-ready entry capacity")
            for (reference, raw_device), key in zip(bindings, keys, strict=True):
                device = canonical_device(raw_device)
                if key in self._products:
                    raise invalid_descriptor("persistent output is already registered")
                if reference.buffer_id not in allocations:
                    raise invalid_descriptor("persistent output has no buffer allocation")
                if feature:
                    if reference.dtype not in {DType.F16, DType.BF16, DType.F32}:
                        raise invalid_descriptor("encoder feature dtype is unsupported")
                    if reference.max_bytes > self.max_entry_bytes:
                        raise resource_error(
                            "encoder feature exceeds the fixed entry byte capacity"
                        )
                    if device not in self.devices:
                        raise invalid_descriptor("encoder feature names an undeclared device")
                else:
                    name = str(device)
                    counts[name] = counts.get(name, 0) + 1
                    if counts[name] > self.capacity:
                        raise resource_error(
                            f"device-product arena for {name} has no query-ready free generation"
                        )
            writes: list[TensorRecord] = []
            try:
                for (reference, raw_device), key in zip(bindings, keys, strict=True):
                    device = canonical_device(raw_device)
                    dtype = _device_dtype(reference.dtype)
                    logical_shape = (
                        _device_shape(reference)
                        if shapes is None
                        else shapes.get(reference, _device_shape(reference))
                    )
                    region = None if regions is None else regions.get(reference)
                    if region == TensorRegion((0,) * len(logical_shape), logical_shape):
                        region = None
                    shape = logical_shape if region is None else region.shape
                    allocation = allocations[reference.buffer_id]
                    full_storage = region is not None and allocation.bytes >= self._tensor_bytes(
                        logical_shape, dtype
                    )
                    binding = self.buffer_pool.bind(
                        reference,
                        allocation,
                        device=device,
                        dtype=dtype,
                        shape=logical_shape if full_storage else shape,
                    )
                    tensor = (
                        binding.tensor[region.slices()]
                        if full_storage and region is not None
                        else binding.tensor
                    )
                    write = TensorRecord(
                        reference=reference,
                        tensor=tensor,
                        device_name=str(device),
                        shape=shape,
                        physical_generation=binding.binding_id,
                        binding_id=self._next_binding_id,
                        buffer_binding=binding,
                        feature=feature,
                        region=region,
                        logical_shape=logical_shape,
                    )
                    self._next_binding_id += 1
                    self._writes[write.binding_id] = write
                    self._products[key] = write
                    writes.append(write)
            except BaseException:
                for write in reversed(writes):
                    self._writes.pop(write.binding_id)
                    self._products.pop(_reference_key(write.reference))
                    self._release_storage_locked(write)
                raise
            return tuple(writes)

    def _bind_relay_outputs(
        self,
        bindings: tuple[tuple[TensorRef, torch.device | str], ...],
        request_slots: Mapping[RequestKey, int],
    ) -> tuple[TensorRecord, ...]:
        """Bind product references to stable request-and-lane relay slots for graph-safe output."""

        if self.request_capacity < 1 or self.relay_depth < 1:
            raise resource_error("worker has no request-relay arena")
        fields: dict[tuple[str, int, RequestKey, ComputationId, torch.dtype], int] = {}
        requested_rows = []
        for reference, raw_device in bindings:
            device = canonical_device(raw_device)
            request_slot = int(request_slots.get(reference.request_key, 0))
            dtype = _device_dtype(reference.dtype)
            field_key = (
                str(device),
                request_slot,
                reference.request_key,
                reference.producer_op_id,
                dtype,
            )
            field = fields.get(field_key, 0)
            fields[field_key] = field + 1
            requested_rows.append((reference, device, request_slot, dtype, field))
        requested = tuple(requested_rows)
        if any(
            slot < 1 or slot > self.request_capacity or math.prod(_device_shape(reference)) != 1
            for reference, _device, slot, _dtype, _field in requested
        ):
            raise invalid_descriptor("request-relay output has invalid slot or scalar geometry")
        keys = tuple(
            _reference_key(reference) for reference, _device, _slot, _dtype, _field in requested
        )
        if len(set(keys)) != len(keys):
            raise invalid_descriptor("request-relay registration repeats an output identity")
        with self._lock:
            self._reclaim_ready_locked()
            for (reference, _device, _slot, _dtype, _field), key in zip(
                requested, keys, strict=True
            ):
                if int(reference.generation) < 1:
                    raise invalid_descriptor(
                        "request-relay registration requires a positive logical generation"
                    )
                existing = self._products.get(key)
                if existing is not None:
                    if not existing.committed:
                        raise invalid_descriptor("request-relay output already has a candidate")
                    if existing.reference != reference:
                        raise invalid_descriptor("stale request-relay logical generation")
                    raise invalid_descriptor("request-relay output is already registered")

            operation_lanes: dict[tuple[str, int, RequestKey, ComputationId], int] = {}
            for reference, device, request_slot, _dtype, _field in requested:
                operation = (
                    str(device),
                    request_slot,
                    reference.request_key,
                    reference.producer_op_id,
                )
                lane = self._relay_operation_lanes.get(operation)
                if lane is None:
                    lane = operation_lanes.get(operation)
                if lane is None:
                    lane = next(
                        (
                            candidate
                            for candidate in range(self.relay_depth)
                            if self._relay_lane_free_locked(str(device), request_slot, candidate)
                        ),
                        None,
                    )
                    if lane is None:
                        raise resource_error("request-relay unresolved window is exhausted")
                operation_lanes[operation] = lane

            writes: list[TensorRecord] = []
            installed_operations: set[tuple[str, int, RequestKey, ComputationId]] = set()
            try:
                for (reference, device, request_slot, dtype, field), key in zip(
                    requested, keys, strict=True
                ):
                    operation = (
                        str(device),
                        request_slot,
                        reference.request_key,
                        reference.producer_op_id,
                    )
                    lane = operation_lanes[operation]
                    slot = self._relay_slot_locked(
                        device,
                        request_slot,
                        lane,
                        dtype,
                        field,
                        operation,
                    )
                    if slot.owner is not None:
                        raise _invariant("request-relay lane was assigned more than once")
                    generation = slot.generation + 1
                    slot.generation = 1 if generation > _MAX_GENERATION else generation
                    write = TensorRecord(
                        reference=reference,
                        tensor=cast(torch.Tensor, slot.tensor),
                        device_name=slot.device_name,
                        shape=(1,),
                        relay_slot=slot,
                        physical_generation=slot.generation,
                        binding_id=self._next_binding_id,
                    )
                    self._next_binding_id += 1
                    slot.owner = write.binding_id
                    self._writes[write.binding_id] = write
                    self._products[key] = write
                    self._relay_operation_lanes[operation] = lane
                    installed_operations.add(operation)
                    writes.append(write)
            except BaseException:
                for write in reversed(writes):
                    self._writes.pop(write.binding_id)
                    self._products.pop(_reference_key(write.reference))
                    self._release_storage_locked(write)
                for operation in installed_operations:
                    self._release_relay_operation_locked(operation)
                raise
            return tuple(writes)

    def _relay_lane_free_locked(
        self,
        device_name: str,
        request_slot: int,
        lane: int,
    ) -> bool:
        """Return whether a request relay lane has no bound operation."""

        fields = self._relay_slots.get((device_name, request_slot, lane))
        return fields is None or all(slot.owner is None for slot in fields.values())

    def _relay_slot_locked(
        self,
        device: torch.device,
        request_slot: int,
        lane: int,
        dtype: torch.dtype,
        field: int,
        operation: tuple[str, int, RequestKey, ComputationId],
    ) -> RelaySlot:
        """Resolve or create one stable scalar relay slot inside its geometry-specific arena."""

        device_name = str(device)
        lane_key = (device_name, request_slot, lane)
        fields = self._relay_slots.setdefault(lane_key, {})
        key = (dtype, int(field))
        slot = fields.get(key)
        if slot is None:
            arena_key = (device_name, dtype, int(field))
            arena = self._relay_arenas.get(arena_key)
            if arena is None:
                elements = (self.request_capacity + 1) * self.relay_depth
                projected = self._allocated_bytes + elements * _TORCH_DTYPE_BYTES[dtype]
                self._require_byte_capacity_locked(projected)
                arena = torch.empty((elements,), dtype=dtype, device=device)
                self._relay_arenas[arena_key] = arena
                self._allocated_bytes = projected
            index = request_slot * self.relay_depth + lane
            slot = RelaySlot(
                index=index,
                device_name=device_name,
                tensor=arena[index : index + 1],
                shape=(1,),
                dtype=dtype,
            )
            fields[key] = slot
        if slot.relay_lane is not None and slot.relay_lane[:4] != operation:
            raise _invariant("request-relay slot retained a conflicting operation identity")
        slot.relay_lane = (*operation, lane)
        return slot

    def _release_relay_operation_locked(
        self,
        operation: tuple[str, int, RequestKey, ComputationId],
    ) -> None:
        """Release the stable relay-lane association for one completed operation."""

        lane = self._relay_operation_lanes.get(operation)
        if lane is None:
            return
        if self._relay_lane_free_locked(operation[0], operation[1], lane):
            self._relay_operation_lanes.pop(operation, None)

    def bind_output_groups(
        self,
        groups: tuple[
            tuple[tuple[TensorRef, torch.device | str], ...],
            ...,
        ],
        *,
        request_slots: Mapping[RequestKey, int] | None = None,
        buffer_allocations: Mapping[BufferId, BufferAllocation] | None = None,
        regions: Mapping[TensorRef, TensorRegion] | None = None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None = None,
    ) -> tuple[tuple[TensorRecord, ...], ...]:
        """Atomically bind output groups while preserving direct producer ranges."""

        bindings: list[tuple[TensorRecord, ...]] = []
        with self._lock:
            try:
                for group in groups:
                    if group:
                        bindings.append(
                            self.bind_outputs(
                                group,
                                request_slots=request_slots,
                                buffer_allocations=buffer_allocations,
                                regions=regions,
                                shapes=shapes,
                            )
                        )
            except BaseException:
                self.abandon_writes(tuple(write for binding in bindings for write in binding))
                raise
        return tuple(bindings)

    def producer_write_views(
        self,
        writes: tuple[TensorRecord, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return unpublished tensors from table-issued physical bindings."""

        if not writes:
            return ()
        with self._lock:
            entries = tuple(self._require_write_locked(write) for write in writes)
            if any(entry.producer_recorded for entry in entries):
                raise _invariant("device product was published more than once")
            tensors = tuple(entry.tensor for entry in entries)
            if any(tensor is None for tensor in tensors):
                raise _invariant("device product has no physical tensor")
            return tuple(tensor for tensor in tensors if tensor is not None)

    def publish_write(
        self,
        write: TensorRecord,
        value: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None = None,
        metadata: ImageMetadata | FeatureMetadata | None = None,
    ) -> torch.Tensor:
        """Commit a tensor into a reserved product slot with producer-stream synchronization."""

        with self._lock:
            entry = self._require_write_locked(write)
            if entry.feature:
                if not isinstance(metadata, FeatureMetadata):
                    raise invalid_descriptor("encoder feature requires spatial metadata")
                value = value.to(dtype=entry.tensor.dtype)
            elif isinstance(metadata, FeatureMetadata):
                raise invalid_descriptor("feature metadata requires feature admission")
            result = self._publish_locked(entry, value, producer_event=producer_event)
            entry.metadata = metadata
            return result

    def _publish_locked(
        self,
        entry: TensorRecord,
        value: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None,
    ) -> torch.Tensor:
        """Copy a value into a reserved slot and transfer readiness-event ownership."""

        if entry.producer_recorded:
            raise _invariant("device product was published more than once")
        target = entry.tensor
        if target is None:
            raise _invariant("device product has no physical tensor")
        if value.dtype != target.dtype:
            raise invalid_descriptor("tensor publication changes its declared dtype")
        if entry.region is not None:
            shape_matches = tuple(value.shape) == entry.region.shape
        else:
            shape_matches = entry.reference.shape_bound.contains_shape(tuple(value.shape))
        if not shape_matches:
            raise invalid_descriptor("tensor publication changes its declared shape")
        flat = value.detach().reshape(-1)
        if flat.numel() > target.numel():
            raise _invariant("device product exceeds its registered shape bound")
        view = (
            target
            if entry.region is not None
            else target.reshape(-1)[: flat.numel()].reshape(value.shape)
        )
        source = value.detach()
        if (
            view.data_ptr() != source.data_ptr()
            or view.dtype != source.dtype
            or view.stride() != source.stride()
        ):
            view.copy_(source.to(dtype=target.dtype), non_blocking=value.device.type == "cuda")
        if target.device.type == "cuda":
            entry.producer_event, entry.producer_stream = self._producer_event_locked(
                target.device,
                producer_event,
            )
            self.event_pool.retain(entry.producer_event, target.device)
        entry.actual_extent = int(flat.numel())
        entry.actual_shape = tuple(int(size) for size in value.shape)
        entry.producer_recorded = True
        return view.reshape(value.shape)

    def publish_writes(
        self,
        writes: tuple[TensorRecord, ...],
        values: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Publish table-issued scalar bindings with one completion event."""

        if not writes:
            return ()
        with self._lock:
            entries = tuple(self._require_write_locked(write) for write in writes)
            return self._publish_batch_locked(
                entries,
                values,
                producer_event=producer_event,
            )

    def _publish_batch_locked(
        self,
        entries: tuple[TensorRecord, ...],
        values: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None,
    ) -> tuple[torch.Tensor, ...]:
        """Publish aligned tensor values atomically across a reserved write batch."""

        flat = values.detach().reshape(-1)
        if int(flat.numel()) != len(entries):
            raise invalid_descriptor(
                "batched device-product publication requires one scalar per output"
            )
        if any(entry.producer_recorded for entry in entries):
            raise _invariant("device product was published more than once")
        targets = tuple(entry.tensor for entry in entries)
        if any(target is None or int(target.numel()) != 1 for target in targets):
            raise invalid_descriptor(
                "batched device-product publication requires scalar output bounds"
            )
        tensors = tuple(target for target in targets if target is not None)
        first = tensors[0]
        if any(tensor.device != first.device or tensor.dtype != first.dtype for tensor in tensors):
            raise invalid_descriptor(
                "batched device-product publication spans incompatible storage"
            )

        source = flat.to(dtype=first.dtype)
        # Keep the scalar batch in one native copy operation. CUDA can scatter
        # these independent destinations together; other device combinations
        # retain PyTorch's ordinary copy and non-blocking semantics.
        torch._foreach_copy_(
            tensors,
            source.reshape(-1, 1).unbind(0),
            non_blocking=source.device.type == "cuda",
        )

        event: torch.cuda.Event | None = None
        if first.device.type == "cuda":
            event, stream_id = self._producer_event_locked(
                first.device,
                producer_event,
            )
            self.event_pool.retain(event, first.device, len(entries))
        else:
            stream_id = None
        for entry in entries:
            entry.producer_event = event
            entry.producer_stream = stream_id
            entry.actual_extent = 1
            entry.actual_shape = (1,)
            entry.producer_recorded = True
        return tensors

    def publish_scalar_write(
        self,
        write: TensorRecord,
        value: bool | int,
        *,
        producer_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        """Commit one boolean or integer into a reserved scalar product slot."""

        with self._lock:
            entry = self._require_write_locked(write)
            return self._publish_scalar_locked(
                entry,
                value,
                producer_event=producer_event,
            )

    def _publish_scalar_locked(
        self,
        entry: TensorRecord,
        value: bool | int,
        *,
        producer_event: torch.cuda.Event | None,
    ) -> torch.Tensor:
        """Store one host scalar in its reserved device slot and mark the write visible."""

        if entry.producer_recorded:
            raise _invariant("device product was published more than once")
        tensor = entry.tensor
        if tensor is None:
            raise _invariant("device product has no physical tensor")
        tensor.reshape(-1)[:1].fill_(int(value))
        if tensor.device.type == "cuda":
            entry.producer_event, entry.producer_stream = self._producer_event_locked(
                tensor.device,
                producer_event,
            )
            self.event_pool.retain(entry.producer_event, tensor.device)
        entry.actual_extent = 1
        entry.actual_shape = (1,)
        entry.producer_recorded = True
        return tensor.reshape(-1)[:1]

    def consume(
        self,
        reference: TensorRef,
        *,
        consumer_op_id: ComputationId,
        device: torch.device | str | None = None,
    ) -> TensorRead:
        """Acquire a generation-safe read of a published product on the consumer device."""

        return self.consume_batch(((reference, consumer_op_id, device),))[0]

    def consume_batch(
        self,
        requests: tuple[
            tuple[TensorRef, ComputationId, torch.device | str | None],
            ...,
        ],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[TensorRead, ...]:
        """Resolve exact generations and enqueue each producer event once per stream."""

        if not requests:
            return ()
        shared_target = None if device is None else canonical_device(device)
        if shared_target is not None:
            target_name = str(shared_target)
            with self._lock:
                shared_resolved: list[tuple[TensorRecord, torch.Tensor, ComputationId]] = []
                for reference, consumer_op_id, requested_device in requests:
                    entry = self._require_locked(reference)
                    if entry.released:
                        raise invalid_descriptor(
                            "device product was consumed after logical release"
                        )
                    if not entry.producer_recorded:
                        raise invalid_descriptor(
                            "device product was consumed before producer publication"
                        )
                    storage = entry.tensor
                    if storage is None:
                        raise _invariant("device product has no physical tensor")
                    if (
                        requested_device is not None
                        and canonical_device(requested_device) != shared_target
                    ):
                        raise invalid_descriptor(
                            "device-product batch names conflicting consumer devices"
                        )
                    if entry.device_name != target_name:
                        raise invalid_descriptor("device product consumer names a different device")
                    tensor = (
                        storage
                        if entry.actual_shape == entry.shape
                        else storage.reshape(-1)[: entry.actual_extent].reshape(entry.actual_shape)
                    )
                    shared_resolved.append((entry, tensor, consumer_op_id))

                if shared_target.type == "cuda":
                    first_event = shared_resolved[0][0].producer_event
                    if first_event is None:
                        raise _invariant("CUDA device product has no producer event")
                    if all(
                        entry.producer_event is first_event
                        for entry, _tensor, _op in shared_resolved
                    ):
                        stream = torch.cuda.current_stream(shared_target)
                        stream_id = int(stream.cuda_stream)
                        if any(
                            entry.producer_stream != stream_id
                            for entry, _tensor, _op in shared_resolved
                        ):
                            stream.wait_event(first_event)
                    else:
                        shared_waited: set[int] = set()
                        stream = torch.cuda.current_stream(shared_target)
                        for entry, _tensor, _consumer_op_id in shared_resolved:
                            event = entry.producer_event
                            if event is None:
                                raise _invariant("CUDA device product has no producer event")
                            identity = id(event)
                            if identity in shared_waited:
                                continue
                            stream.wait_event(event)
                            shared_waited.add(identity)

                for entry, _tensor, _consumer_op_id in shared_resolved:
                    entry.readers += 1
                return tuple(
                    TensorRead(
                        tensor=tensor,
                        consumer_op_id=consumer_op_id,
                        _write=entry,
                        region=entry.region,
                        metadata=entry.metadata,
                    )
                    for entry, tensor, consumer_op_id in shared_resolved
                )
        assert shared_target is None
        with self._lock:
            resolved: list[tuple[TensorRecord, torch.Tensor, torch.device, ComputationId]] = []
            for reference, consumer_op_id, requested_device in requests:
                entry = self._require_locked(reference)
                if entry.released:
                    raise invalid_descriptor("device product was consumed after logical release")
                if not entry.producer_recorded:
                    raise invalid_descriptor(
                        "device product was consumed before producer publication"
                    )
                storage = entry.tensor
                if storage is None:
                    raise _invariant("device product has no physical tensor")
                tensor = (
                    storage
                    if entry.actual_shape == entry.shape
                    else storage.reshape(-1)[: entry.actual_extent].reshape(entry.actual_shape)
                )
                target = (
                    storage.device
                    if requested_device is None
                    else canonical_device(requested_device)
                )
                if target != storage.device:
                    raise invalid_descriptor("device product consumer names a different device")
                resolved.append((entry, tensor, target, consumer_op_id))

            first_entry, _tensor, first_target, _consumer_op_id = resolved[0]
            if first_target.type == "cuda":
                first_event = first_entry.producer_event
                if first_event is None:
                    raise _invariant("CUDA device product has no producer event")
                if all(
                    target == first_target and entry.producer_event is first_event
                    for entry, _tensor, target, _consumer_op_id in resolved
                ):
                    stream = torch.cuda.current_stream(first_target)
                    if any(
                        entry.producer_stream != int(stream.cuda_stream)
                        for entry, _tensor, _target, _consumer_op_id in resolved
                    ):
                        stream.wait_event(first_event)
                else:
                    waited: set[tuple[str, int]] = set()
                    for entry, _tensor, target, _consumer_op_id in resolved:
                        if target.type != "cuda":
                            continue
                        event = entry.producer_event
                        if event is None:
                            raise _invariant("CUDA device product has no producer event")
                        event_identity = (str(target), id(event))
                        if event_identity in waited:
                            continue
                        torch.cuda.current_stream(target).wait_event(event)
                        waited.add(event_identity)

            reads = []
            for entry, tensor, _target, consumer_op_id in resolved:
                entry.readers += 1
                reads.append(
                    TensorRead(
                        tensor=tensor,
                        consumer_op_id=consumer_op_id,
                        _write=entry,
                        region=entry.region,
                        metadata=entry.metadata,
                    )
                )
            return tuple(reads)

    def complete_reads(
        self,
        reads: tuple[TensorRead, ...],
        *,
        device: torch.device | str | None = None,
        after_writes: tuple[TensorRecord, ...] = (),
    ) -> None:
        """End read leases only after their consumer completion fences are installed."""

        with self._lock:
            pending = tuple(read for read in reads if not read._recorded)
            if not pending:
                return
            self._record_reader_fences(pending, device=device, after_writes=after_writes)
            for read in pending:
                entry = self._require_read_locked(read)
                entry.readers -= 1
                read._recorded = True
                imported = read.imported
                if imported is not None:
                    imported.users -= 1
                    if imported.users == 0:
                        self._imports.pop(_reference_key(entry.reference))
                        for ticket in imported.tickets:
                            if not imported.committed:
                                ticket.cancel()
                            ticket.close()
                        if not entry.producer_recorded:
                            self.abandon_writes((entry,))
                    read.imported = None
                if entry.released:
                    self._wake_retirement_locked(entry)

    def _record_reader_fences(
        self,
        reads: tuple[TensorRead, ...],
        *,
        device: torch.device | str | None = None,
        after_writes: tuple[TensorRecord, ...] = (),
    ) -> None:
        """Fence reads with a later output write or one event per consuming stream."""

        if not reads:
            return
        declared_target = None if device is None else canonical_device(device)
        with self._lock:
            if len(after_writes) == len(reads):
                target = reads[0].tensor.device
                if declared_target is not None and declared_target != target:
                    raise _invariant("device-product reader completed on a different device")
                stream_id = (
                    int(torch.cuda.current_stream(target).cuda_stream)
                    if target.type == "cuda"
                    else None
                )
                aligned: list[tuple[TensorRecord, torch.cuda.Event | None]] = []
                for read, write in zip(reads, after_writes, strict=True):
                    entry = self._require_read_locked(read)
                    completion = self._require_write_locked(write)
                    event = completion.producer_event
                    if (
                        read.tensor.device != target
                        or completion.reference.producer_op_id != read.consumer_op_id
                        or (
                            target.type == "cuda"
                            and (
                                event is None
                                or completion.tensor is None
                                or completion.tensor.device != target
                                or completion.producer_stream != stream_id
                            )
                        )
                    ):
                        break
                    aligned.append((entry, event))
                else:
                    if target.type != "cuda":
                        return
                    first_event = aligned[0][1]
                    if all(
                        event is first_event and entry.reader_events is None
                        for entry, event in aligned
                    ):
                        retained_event = cast(torch.cuda.Event, first_event)
                        for entry, _event in aligned:
                            entry.reader_events = retained_event
                        self.event_pool.retain(
                            retained_event,
                            target,
                            len(aligned),
                        )
                        return
                    for entry, event in aligned:
                        self._append_reader_event_locked(
                            entry,
                            cast(torch.cuda.Event, event),
                            target,
                        )
                    return
            write_fences: dict[ComputationId, TensorRecord] = {}
            for write in after_writes:
                entry = self._require_write_locked(write)
                if not entry.producer_recorded:
                    continue
                write_fences.setdefault(entry.reference.producer_op_id, entry)
            if declared_target is not None:
                target = declared_target
                entries = []
                for read in reads:
                    entry = self._require_read_locked(read)
                    if target != read.tensor.device:
                        raise _invariant("device-product reader completed on a different device")
                    entries.append((read, entry))
                if target.type == "cuda":
                    stream = torch.cuda.current_stream(target)
                    stream_id = int(stream.cuda_stream)
                    pending: list[TensorRecord] = []
                    for read, entry in entries:
                        completion_fence = (
                            None
                            if read.consumer_op_id is None
                            else write_fences.get(read.consumer_op_id)
                        )
                        if (
                            completion_fence is not None
                            and completion_fence.producer_event is not None
                            and completion_fence.tensor is not None
                            and completion_fence.tensor.device == target
                            and completion_fence.producer_stream == stream_id
                        ):
                            event = completion_fence.producer_event
                            self._append_reader_event_locked(entry, event, target)
                        else:
                            pending.append(entry)
                    if pending:
                        event, _stream_id = self._record_event_locked(target)
                        for entry in pending:
                            self._append_reader_event_locked(entry, event, target)
                return

            grouped: dict[
                str,
                tuple[torch.device, list[tuple[TensorRead, TensorRecord]]],
            ] = {}
            for read in reads:
                entry = self._require_read_locked(read)
                target = read.tensor.device
                device_name = str(target)
                group = grouped.get(device_name)
                if group is None:
                    grouped[device_name] = (target, [(read, entry)])
                else:
                    group[1].append((read, entry))
            for target, entries in grouped.values():
                if target.type != "cuda":
                    continue
                stream = torch.cuda.current_stream(target)
                stream_id = int(stream.cuda_stream)
                pending = []
                for read, entry in entries:
                    completion_fence = (
                        None
                        if read.consumer_op_id is None
                        else write_fences.get(read.consumer_op_id)
                    )
                    if (
                        completion_fence is not None
                        and completion_fence.producer_event is not None
                        and completion_fence.tensor is not None
                        and completion_fence.tensor.device == target
                        and completion_fence.producer_stream == stream_id
                    ):
                        event = completion_fence.producer_event
                        self._append_reader_event_locked(entry, event, target)
                    else:
                        pending.append(entry)
                if pending:
                    event, _stream_id = self._record_event_locked(target)
                    for entry in pending:
                        self._append_reader_event_locked(entry, event, target)

    def release_operations(
        self,
        releases: Iterable[tuple[RequestKey, ComputationId]],
    ) -> None:
        """Release all product ownership associated with completed operation identities."""

        with self._lock:
            for request_key, raw_op_id in releases:
                op_id = raw_op_id
                operation_key = (request_key, op_id)
                operation_writes = self._operation_writes.pop(operation_key, None)
                entries = list(
                    operation_writes
                    if isinstance(operation_writes, list)
                    else (() if operation_writes is None else (operation_writes,))
                )
                direct_key = (
                    int(request_key.engine_id),
                    int(request_key.request_id),
                    int(request_key.request_epoch),
                    op_id,
                    0,
                )
                direct = self._products.get(direct_key)
                if (
                    direct is not None
                    and direct.committed
                    and not direct._indexed
                    and direct.reference.request_key == request_key
                    and all(entry is not direct for entry in entries)
                ):
                    entries.append(direct)
                for entry in entries:
                    entry._indexed = False
                    if entry.released:
                        continue
                    entry.released = True

    def release_buffers(self, buffers: Iterable[BufferId]) -> None:
        """Revoke exact buffer identities and preserve all active physical leases."""

        selected = set(buffers)
        release_exports(self.exports, self.export_releases, selected)
        if not selected:
            return
        with self._lock:
            for entry in tuple(self._writes.values()):
                if entry.reference.buffer_id in selected:
                    self._release_entry_locked(entry)
            self._reclaim_ready_locked()

    def release_requests(
        self, requests: Iterable[RequestKey], *, retained: frozenset[BufferId] = frozenset()
    ) -> None:
        """Revoke request-owned products while preserving transferred allocation ownership."""

        selected = set(requests)
        if not selected:
            return
        with self._lock:
            for entry in tuple(self._writes.values()):
                if entry.reference.request_key in selected:
                    if entry.reference.buffer_id in retained:
                        if entry.buffer_binding is None:
                            raise invalid_descriptor("finish cannot retain request-slot storage")
                    else:
                        self._release_entry_locked(entry)
            self._reclaim_ready_locked()

    def retirement_ready(
        self,
        *,
        buffers: frozenset[BufferId],
        requests: frozenset[RequestKey],
        retained: frozenset[BufferId] = frozenset(),
    ) -> bool:
        """Confirm selected allocations have returned after every physical reader."""

        with self._lock:
            for entry in tuple(self._writes.values()):
                if entry.reference.buffer_id in buffers or (
                    entry.reference.request_key in requests
                    and entry.reference.buffer_id not in retained
                ):
                    for transfer in entry.transfers:
                        transfer.retirement_ready()
                    for publication in entry.publications:
                        if publication.done():
                            publication.result()
            self._reclaim_ready_locked()
            return not any(
                entry.reference.buffer_id in buffers
                or (
                    entry.reference.request_key in requests
                    and entry.reference.buffer_id not in retained
                )
                for entry in self._writes.values()
            )

    def _release_entry_locked(self, entry: TensorRecord) -> None:
        self._detach_write_locked(entry)
        entry.released = True
        self._wake_retirement_locked(entry)
        if not entry.producer_recorded:
            for transfer in entry.transfers:
                transfer.cancel()

    def _wake_retirement_locked(self, entry: TensorRecord) -> None:
        readers = entry.reader_events
        events = (
            entry.producer_event,
            *(readers if isinstance(readers, list) else (() if readers is None else (readers,))),
        )
        for event in events:
            if event is not None and not event.query():
                self.event_pool.schedule_completion_wake(entry.device_name, event)

    def retain_publication(self, write: TensorRecord, retirement: Future[None]) -> None:
        """Keep a published pool range immutable until its transport registration retires."""

        with self._lock:
            entry = self._require_write_locked(write)
            retained = tuple(
                publication
                for publication in entry.publications
                if not publication.done() or publication.exception() is not None
            )
            entry.publications = (*retained, retirement)

    def retain_transfer(self, write: TensorRecord, ticket: TransferTicket) -> None:
        """Guard a reserved destination until its backend finishes physical access."""

        with self._lock:
            entry = self._require_write_locked(write)
            if entry.producer_recorded or entry.released:
                raise _invariant("transfer destination already has a producer")
            entry.transfers = (*entry.transfers, ticket)

        def reclaim() -> None:
            with self._lock:
                self._reclaim_ready_locked()

        ticket.add_retirement_callback(reclaim)

    def abandon_writes(self, writes: tuple[TensorRecord, ...]) -> None:
        """Return unpublished reserved writes without creating resident products."""

        with self._lock:
            for write in writes:
                try:
                    entry = self._require_write_locked(write)
                except WorkerError:
                    continue
                entry.released = True
            self._reclaim_ready_locked()

    def import_tensor(
        self,
        reference: TensorRef,
        tensor: TensorTransfer,
        *,
        device: torch.device | str,
        bindings: Mapping[tuple[WorkerEndpoint, str], Transport],
        request_slots: Mapping[RequestKey, int],
        buffer_allocations: Mapping[BufferId, BufferAllocation],
        metadata: ImageMetadata | FeatureMetadata | None = None,
    ) -> TensorRead:
        """Borrow resident coverage and fetch only missing immutable regions.

        A shard can be expanded in place only when its scheduler reservation
        contains the full logical tensor. Compact shard reservations remain
        valid producer storage but cannot serve a full local consumer.
        """

        target_device = canonical_device(device)
        key = _reference_key(reference)
        with self._lock:
            pending = self._imports.get(key)
            if pending is not None:
                if (
                    pending.write.reference != reference
                    or pending.tensor.device != target_device
                    or tuple(pending.tensor.shape) != tensor.shape
                    or pending.metadata != metadata
                ):
                    raise invalid_descriptor(
                        "product import conflicts with pending materialization"
                    )
                pending.users += 1
                pending.write.readers += 1
                return TensorRead(
                    pending.tensor,
                    None,
                    pending.write,
                    metadata=pending.metadata,
                    imported=pending,
                )
            existing = self._products.get(key)
            if existing is not None and not existing.committed:
                existing = None
            missing: tuple[TensorRegion, ...]
            full = TensorRegion((0,) * len(tensor.shape), tensor.shape)
            if existing is not None:
                write = self._require_locked(reference)
                if write.released or not write.producer_recorded:
                    raise invalid_descriptor("product import requires a live published generation")
                if write.device_name != str(target_device) or write.metadata != metadata:
                    raise invalid_descriptor("product import conflicts with resident ownership")
                storage = write.tensor
                assert storage is not None
                if write.region is None:
                    if write.actual_shape != tensor.shape:
                        raise invalid_descriptor("product import changes resident tensor geometry")
                    destination = (
                        storage
                        if tuple(storage.shape) == tensor.shape
                        else storage.reshape(-1)[: write.actual_extent].reshape(tensor.shape)
                    )
                    missing = ()
                else:
                    persistent = write.buffer_binding
                    if persistent is None or tuple(persistent.tensor.shape) != tensor.shape:
                        raise invalid_descriptor(
                            "product import exceeds its reserved logical storage"
                        )
                    destination = persistent.tensor
                    if any(not ticket.retired() for ticket in write.transfers):
                        raise resource_error("product storage has pending physical reads")
                    missing = full.subtract(write.region)
            else:
                if isinstance(metadata, FeatureMetadata):
                    write = self._bind_persistent_outputs(
                        ((reference, target_device),),
                        buffer_allocations,
                        None,
                        {reference: tensor.shape},
                        feature=True,
                    )[0]
                else:
                    write = self.bind_outputs(
                        ((reference, target_device),),
                        request_slots=request_slots,
                        buffer_allocations=buffer_allocations,
                        shapes={reference: tensor.shape},
                    )[0]
                destination = self.producer_write_views((write,))[0]
                destination = destination.reshape(-1)[: math.prod(tensor.shape)].reshape(
                    tensor.shape
                )
                missing = (full,)
            if str(destination.dtype).removeprefix("torch.") != tensor.dtype:
                if existing is None:
                    self.abandon_writes((write,))
                raise invalid_descriptor("product import changes resident tensor dtype")
            # This lease protects both existing readers and a not-yet-published
            # destination through submission, cancellation and final adoption.
            write.readers += 1
            tickets: list[TransferTicket] = []

            def retain(ticket: TransferTicket) -> None:
                write.transfers = (*write.transfers, ticket)
                tickets.append(ticket)

                def reclaim() -> None:
                    with self._lock:
                        self._reclaim_ready_locked()

                ticket.add_retirement_callback(reclaim)

            try:
                for region in missing:
                    fetch_tensor(
                        tensor,
                        destination[region.slices()],
                        bindings=bindings,
                        region=region,
                        retain=retain,
                    )
            except BaseException:
                for ticket in tickets:
                    ticket.cancel()
                    ticket.close()
                write.readers -= 1
                if existing is None:
                    self.abandon_writes((write,))
                raise
            materialization = TensorImport(
                write,
                destination,
                tuple(tickets),
                metadata,
                users=1,
                committed=existing is not None and write.region is None,
            )
            self._imports[key] = materialization
            return TensorRead(
                destination,
                None,
                write,
                metadata=metadata,
                imported=materialization,
            )

    def wait_import(self, read: TensorRead) -> None:
        """Order a prepared read after its existing producer and physical transfers."""

        if read._recorded or read.imported is None:
            raise _invariant("closed or ordinary tensor read is not an import")
        for ticket in read.imported.tickets:
            ticket.result()
        event = read._write.producer_event
        if event is not None:
            torch.cuda.current_stream(read.tensor.device).wait_event(event)

    def complete_import(self, read: TensorRead) -> None:
        """Adopt complete coverage after ordering access on the consuming stream."""

        self.wait_import(read)
        value = read.imported
        assert value is not None
        with self._lock:
            write = self._require_write_locked(value.write)
            for ticket in value.tickets:
                ticket.result()
            if value.committed:
                return
            if not write.producer_recorded:
                self.publish_write(write, value.tensor, metadata=value.metadata)
                self.commit_writes((write,))
            else:
                # A new fence covers the original shard and every fetched hole.
                # Existing read leases retain their original view and region.
                if write.producer_event is not None:
                    stream = torch.cuda.current_stream(value.tensor.device)
                    stream.wait_event(write.producer_event)
                    previous = write.producer_event
                    write.producer_event, write.producer_stream = self._record_event_locked(
                        value.tensor.device
                    )
                    self.event_pool.retain(write.producer_event, value.tensor.device)
                    self.event_pool.defer_release((previous,), write)
                write.tensor = value.tensor
                write.shape = tuple(value.tensor.shape)
                write.region = None
                write.actual_shape = tuple(value.tensor.shape)
                write.actual_extent = int(value.tensor.numel())
            value.committed = True

    def validate_writes(self, writes: tuple[TensorRecord, ...]) -> None:
        """Verify that a write batch still refers to active unpublished reservations."""

        with self._lock:
            for write in writes:
                entry = self._require_write_locked(write)
                if entry.committed:
                    raise _invariant("device-product candidate is not live")
                if not entry.producer_recorded:
                    raise _invariant("completion packing found an unpublished device product")

    def commit_writes(self, writes: tuple[TensorRecord, ...]) -> None:
        """Make a validated set of produced candidate generations addressable."""

        if not writes:
            return
        with self._lock:
            entries = tuple(self._require_write_locked(write) for write in writes)
            keys = tuple(_reference_key(entry.reference) for entry in entries)
            if len(set(keys)) != len(keys):
                raise _invariant("device-product publication repeats a product identity")
            for key, entry in zip(keys, entries, strict=True):
                if entry.committed:
                    raise _invariant("device-product candidate is not live")
                if not entry.producer_recorded:
                    raise _invariant("device-product candidate has no producer readiness")
                if self._products.get(key) is not entry:
                    raise _invariant("device-product publication lost its reserved identity")
            for entry in entries:
                entry.committed = True
                operation_key = (
                    entry.reference.request_key,
                    entry.reference.producer_op_id,
                )
                operation_writes = self._operation_writes.get(operation_key)
                if operation_writes is None:
                    self._operation_writes[operation_key] = entry
                elif isinstance(operation_writes, list):
                    operation_writes.append(entry)
                else:
                    self._operation_writes[operation_key] = [operation_writes, entry]
                entry._indexed = True

    def _require_locked(self, reference: TensorRef) -> TensorRecord:
        entry = self._products.get(_reference_key(reference))
        if entry is None or not entry.committed:
            raise invalid_descriptor("unknown device-product reference")
        if entry.reference != reference:
            raise invalid_descriptor("stale device-product logical generation")
        return self._require_write_locked(entry)

    def _require_write_locked(self, write: TensorRecord) -> TensorRecord:
        if write.retired or self._writes.get(write.binding_id) is not write:
            raise _invariant("stale device-product physical generation")
        slot = write.relay_slot
        if slot is not None and (
            slot.owner != write.binding_id or slot.generation != write.physical_generation
        ):
            raise _invariant("stale request-relay physical generation")
        return write

    def _require_read_locked(self, read: TensorRead) -> TensorRecord:
        return self._require_write_locked(read._write)

    def _release_storage_locked(self, entry: TensorRecord) -> None:
        """Return a persistent span or relay version only after its readers retire."""

        if entry.retired:
            raise _invariant("tensor storage was retired more than once")
        slot = entry.relay_slot
        if slot is not None:
            slot.owner = None
            association = slot.relay_lane
            if association is not None:
                operation = association[:4]
                self._release_relay_operation_locked(operation)
                if operation not in self._relay_operation_lanes:
                    fields = self._relay_slots[(association[0], association[1], association[4])]
                    for candidate in fields.values():
                        if (
                            candidate.relay_lane is not None
                            and candidate.relay_lane[:4] == operation
                        ):
                            candidate.relay_lane = None
        elif entry.buffer_binding is not None:
            self.buffer_pool.release(entry.buffer_binding)
        entry.retired = True

    def _detach_write_locked(self, entry: TensorRecord) -> None:
        """Remove a logical write and release its physical slot when no aliases remain."""

        if not entry._indexed:
            return
        reference = entry.reference
        operation_key = (
            reference.request_key,
            reference.producer_op_id,
        )
        operation_writes = self._operation_writes.get(operation_key)
        if operation_writes is entry:
            self._operation_writes.pop(operation_key, None)
        elif isinstance(operation_writes, list):
            remaining = [candidate for candidate in operation_writes if candidate is not entry]
            if not remaining:
                self._operation_writes.pop(operation_key, None)
            elif len(remaining) == 1:
                self._operation_writes[operation_key] = remaining[0]
            elif len(remaining) != len(operation_writes):
                self._operation_writes[operation_key] = remaining
        entry._indexed = False

    def _producer_event_locked(
        self,
        device: torch.device,
        event: torch.cuda.Event | None,
    ) -> tuple[torch.cuda.Event, int]:
        """Validate or record the producer event that guards one device publication."""

        if event is None:
            return self._record_event_locked(device)
        return event, self.event_pool.declare_stream(event, device)

    def _record_event_locked(self, device: torch.device) -> tuple[torch.cuda.Event, int]:
        """Acquire and record a reusable event on the current stream."""

        event = self.event_pool.acquire(device)
        return event, self.event_pool.record(event, device)

    def _append_reader_event_locked(
        self,
        entry: TensorRecord,
        event: torch.cuda.Event,
        device: torch.device,
    ) -> None:
        """Attach one reader event to a write while deduplicating shared event ownership."""

        current = entry.reader_events
        if current is event:
            return
        if isinstance(current, list):
            if any(candidate is event for candidate in current):
                return
            current.append(event)
        elif current is None:
            entry.reader_events = event
        else:
            entry.reader_events = [current, event]
        self.event_pool.retain(event, device)

    def _reclaim_ready_locked(self) -> int:
        """Reclaim released writes whose producer and reader events are query-ready."""

        reclaimed = 0
        readiness: dict[int, bool] = {}
        released_events: dict[int, tuple[torch.cuda.Event, int]] = {}

        def release_event(event: torch.cuda.Event) -> None:
            """Accumulate one pooled-event reference for release after reclamation."""

            identity = id(event)
            current = released_events.get(identity)
            if current is None:
                released_events[identity] = (event, 1)
            elif current[0] is not event:
                raise _invariant("device event identity changed during reclamation")
            else:
                released_events[identity] = (event, current[1] + 1)

        def ready(event: torch.cuda.Event | None) -> bool:
            """Query each CUDA event at most once during this reclamation pass."""

            if event is None:
                return True
            identity = id(event)
            result = readiness.get(identity)
            if result is None:
                result = _event_ready(event)
                readiness[identity] = result
            return result

        def readers_ready(entry: TensorRecord) -> bool:
            """Require every consumer event associated with a product generation to complete."""

            events = entry.reader_events
            if events is None:
                return True
            if isinstance(events, list):
                return all(ready(event) for event in events)
            return ready(events)

        for binding_id, entry in tuple(self._writes.items()):
            if (
                entry.released
                and entry.readers == 0
                and all(transfer.retired() for transfer in entry.transfers)
                and all(
                    publication.done() and publication.exception() is None
                    for publication in entry.publications
                )
                and ready(entry.producer_event)
                and readers_ready(entry)
            ):
                key = _reference_key(entry.reference)
                if self._products.get(key) is not entry:
                    raise _invariant("device-product reclamation lost its reserved identity")
                self._require_write_locked(entry)
                self._writes.pop(binding_id)
                self._products.pop(key)
                self._detach_write_locked(entry)
                self._release_storage_locked(entry)
                if entry.producer_event is not None:
                    release_event(entry.producer_event)
                reader_events = entry.reader_events
                if isinstance(reader_events, list):
                    for event in reader_events:
                        release_event(event)
                elif reader_events is not None:
                    release_event(reader_events)
                reclaimed += 1
        for event, count in released_events.values():
            self.event_pool.release(event, count)
        return reclaimed


__all__ = [
    "TensorRead",
    "TensorStore",
    "ImageMetadata",
    "TensorRecord",
    "FeatureMetadata",
    "ImageRange",
    "device_product_capacity_bytes",
    "device_product_storage",
]
