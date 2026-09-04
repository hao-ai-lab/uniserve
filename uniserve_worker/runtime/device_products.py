"""Bounded immutable device values with generation-safe stream lifetimes."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from typing import Final, cast

import torch

from ..execution.batch import (
    BufferId,
    BufferPlacement,
    DType,
    ProductKind,
    ProductRef,
    RequestKey,
    StaticDim,
    StorageClass,
)
from ..foundation.errors import WorkerError, WorkerErrorCode, invalid_descriptor, resource_error
from .device import canonical_device
from .device_events import DeviceEventPool
from .persistent_buffers import PersistentBufferBinding, PersistentBuffers

_MAX_GENERATION: Final[int] = (1 << 32) - 1
_DEVICE_DTYPES: Final[dict[DType, torch.dtype]] = {
    DType.U8: torch.uint8,
    DType.U16: torch.int32,
    DType.U32: torch.long,
    DType.I32: torch.int32,
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
_DEVICE_PRODUCT_KINDS: Final[frozenset[ProductKind]] = frozenset(
    {
        ProductKind.TOKEN,
        ProductKind.ARTIFACT,
        ProductKind.COMPLETION,
        ProductKind.SELECTED_POINT,
    }
)
_REQUEST_RELAY_KINDS: Final[frozenset[ProductKind]] = frozenset(
    {ProductKind.TOKEN, ProductKind.COMPLETION, ProductKind.SELECTED_POINT}
)


def device_product_storage(dtype: DType) -> tuple[str, int]:
    """Return the concrete tensor storage used for one product dtype."""

    return _DTYPE_STORAGE[DType(dtype)]


def device_product_capacity_bytes(
    slot_capacity: int,
    device_count: int,
    *,
    selected_points_per_operation: int,
    max_value_bytes: int,
) -> int:
    """Return the fixed backing bound for one ``DeviceProducts`` owner."""

    slots = int(slot_capacity)
    devices = int(device_count)
    points = int(selected_points_per_operation)
    value_bytes = int(max_value_bytes)
    if min(slots, devices, points, value_bytes) < 1:
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


_ReferenceKey = tuple[int, int, int, int, int]
_OperationKey = tuple[RequestKey, int]
_SlotStorageKey = tuple[str, tuple[int, ...], torch.dtype]


def _reference_key(reference: ProductRef) -> _ReferenceKey:
    """Build the generation-tagged lookup key for a product reference."""

    key = reference.request_key
    return (
        int(key.authority_id),
        int(key.request_id),
        int(key.epoch),
        int(reference.producer_op_id),
        int(reference.output_index),
    )


def _device_dtype(dtype: DType) -> torch.dtype:
    """Resolve a product dtype descriptor to its torch dtype."""

    return _DEVICE_DTYPES[dtype]


def _device_shape(reference: ProductRef) -> tuple[int, ...]:
    """Resolve a product's bounded tensor dimensions to a concrete shape."""

    dims = tuple(
        dim.extent if isinstance(dim, StaticDim) else dim.bound
        for dim in reference.shape_bound.dims
    )
    return dims or (1,)


def _event_ready(event: torch.cuda.Event | None) -> bool:
    """Return whether an optional CUDA event is query-ready."""

    return event is None or bool(event.query())


def _resolved_device(device: torch.device | str) -> torch.device:
    """Resolve a concrete torch device with an explicit CUDA index."""

    if isinstance(device, torch.device) and (device.type != "cuda" or device.index is not None):
        return device
    return canonical_device(device)


def _validate_owner(reference: ProductRef) -> None:
    """Validate that a product reference declares complete producer ownership."""

    if reference.kind not in _DEVICE_PRODUCT_KINDS:
        raise invalid_descriptor("product kind does not belong to DeviceProducts")
    if reference.kind is ProductKind.ARTIFACT:
        valid = reference.storage_class is StorageClass.LATENT_ARENA
    elif reference.storage_class is StorageClass.REQUEST_RELAY:
        valid = reference.kind in _REQUEST_RELAY_KINDS
    else:
        valid = reference.storage_class is StorageClass.DEVICE_TENSOR
    if not valid:
        raise invalid_descriptor("device product has an incompatible storage class")


@dataclass(slots=True)
class _DeviceSlot:
    """Tracks ownership, generation, storage geometry, and relay binding for one device-product slot."""

    index: int
    device_name: str
    generation: int = 0
    owner: int | None = None
    tensor: torch.Tensor | None = None
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    relay_lane: tuple[str, int, RequestKey, int, int] | None = None
    persistent_buffer: PersistentBufferBinding | None = None


class ImageRange(StrEnum):
    """Defines whether image values use signed-unit or unit numeric range."""

    SIGNED_UNIT = "signed_unit"
    UNIT = "unit"


@dataclass(frozen=True, slots=True)
class DeviceProductMetadata:
    """Describes a device product’s semantic kind, media geometry, value range, and tensor geometry."""

    height: int = 0
    width: int = 0
    value_range: ImageRange | None = None

    def __post_init__(self) -> None:
        """Validate product kind, tensor geometry, media metadata, and value range."""

        if self.height < 0 or self.width < 0:
            raise ValueError("device-product image geometry must be non-negative")
        if (self.height == 0) != (self.width == 0):
            raise ValueError("device-product image geometry must be complete")


@dataclass(slots=True)
class DeviceProductWrite:
    """One table-issued physical binding retained through producer submission."""

    reference: ProductRef
    slot: _DeviceSlot
    physical_generation: int
    binding_id: int
    producer_event: torch.cuda.Event | None = None
    producer_stream: int | None = None
    producer_recorded: bool = False
    reader_events: torch.cuda.Event | list[torch.cuda.Event] | None = None
    logical_references: int = 1
    released: bool = False
    _indexed: bool = False
    actual_extent: int = 0
    actual_shape: tuple[int, ...] = ()
    metadata: DeviceProductMetadata | None = None
    _scalar_batch: DeviceProductScalarBatch | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(slots=True)
class DeviceProductRead:
    """One generation-validated device read retained until its stream snapshots it."""

    tensor: torch.Tensor
    consumer_op_id: int
    _write: DeviceProductWrite = field(repr=False, compare=False)
    _recorded: bool = field(default=False, repr=False, compare=False)

    @property
    def reference(self) -> ProductRef:
        """Expose the immutable logical product identity guarded by this read lease."""

        return self._write.reference

    @property
    def physical_generation(self) -> int:
        """Expose the slot generation captured when this read lease was acquired."""

        return self._write.physical_generation

    @property
    def metadata(self) -> DeviceProductMetadata | None:
        """Expose producer metadata published with the product, if present."""

        return self._write.metadata


@dataclass(slots=True)
class DeviceProductScalarBatch:
    """One contiguous scalar binding validated for direct producer output."""

    writes: tuple[DeviceProductWrite, ...]
    tensor: torch.Tensor
    _table_token: object = field(repr=False, compare=False)
    _linked: bool = field(default=False, repr=False, compare=False)
    _valid: bool = field(default=True, repr=False, compare=False)
    _published: bool = field(default=False, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DeviceProductBindingBatch:
    """Atomic output bindings with an optional direct scalar producer range."""

    writes: tuple[DeviceProductWrite, ...]
    scalar: DeviceProductScalarBatch | None = None


class DeviceProducts:
    """Bounded physical slots for generation-tagged device products.

    Registration, lookup, stream waits, reader recording, release, and
    reclamation all validate the exact logical and physical generation here.
    Reclamation only queries events; it never synchronizes a device stream.
    """

    def __init__(
        self,
        *,
        capacity: int,
        byte_capacity: int,
        request_capacity: int = 0,
        relay_depth: int = 0,
        persistent_buffers: PersistentBuffers,
        event_pool: DeviceEventPool | None = None,
    ) -> None:
        """Initialize bounded product registries, relay arenas, and event ownership."""

        # Validate independent slot-byte capacity and the coupled request-relay
        # geometry before creating any registries.
        self.capacity = max(1, int(capacity))
        self.byte_capacity = int(byte_capacity)
        if self.byte_capacity < 1:
            raise ValueError("device-product byte capacity must be positive")
        self.request_capacity = int(request_capacity)
        self.relay_depth = int(relay_depth)
        self.persistent_buffers = persistent_buffers
        if (self.request_capacity == 0) != (self.relay_depth == 0):
            raise ValueError("request-relay geometry must be complete")
        if self.request_capacity < 0 or self.relay_depth < 0:
            raise ValueError("request-relay geometry must not be negative")

        # Storage pools are partitioned by device and tensor geometry; relay
        # arenas reserve stable request/lane addresses for graph capture.
        self._allocated_bytes = 0
        self._slots: dict[str, list[_DeviceSlot]] = {}
        self._occupied_slots: dict[str, int] = {}
        self._free_slots: dict[str, set[int]] = {}
        self._free_slot_queues: dict[str, deque[int]] = {}
        self._compatible_free_slots: dict[_SlotStorageKey, deque[int]] = {}
        self._scalar_arenas: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._relay_arenas: dict[
            tuple[str, ProductKind, torch.dtype, int], torch.Tensor
        ] = {}
        self._relay_slots: dict[
            tuple[str, int, int, ProductKind, torch.dtype, int], _DeviceSlot
        ] = {}
        self._relay_operation_lanes: dict[
            tuple[str, int, RequestKey, int], int
        ] = {}

        # Logical references point at generation-tagged physical writes. The
        # shared event pool owns readiness events until every reader releases.
        self.event_pool = DeviceEventPool() if event_pool is None else event_pool
        self._entries: dict[_ReferenceKey, DeviceProductWrite] = {}
        self._candidates: dict[int, DeviceProductWrite] = {}
        self._operation_writes: dict[
            _OperationKey,
            DeviceProductWrite | list[DeviceProductWrite],
        ] = {}
        self._next_binding_id = 1
        self._binding_token = object()
        self._lock = RLock()

    def close(self) -> None:
        """Release all resident product slots, events, relay storage, and persistent bindings."""

        with self._lock:
            self._entries.clear()
            self._candidates.clear()
            self._operation_writes.clear()
            self._scalar_arenas.clear()
            self._relay_arenas.clear()
            self._relay_slots.clear()
            self._relay_operation_lanes.clear()
            self._slots.clear()
            self._occupied_slots.clear()
            self._free_slots.clear()
            self._free_slot_queues.clear()
            self._compatible_free_slots.clear()
            self._allocated_bytes = 0

    def warmup_scattered_publication(
        self,
        devices: Iterable[torch.device | str],
    ) -> None:
        """Materialize the CUDA kernels used by scattered scalar publication."""

        resolved = tuple(dict.fromkeys(_resolved_device(device) for device in devices))
        for device in resolved:
            if device.type != "cuda":
                continue
            retained: list[torch.Tensor] = []
            with torch.cuda.device(device):
                for dtype in _DEVICE_TORCH_DTYPES:
                    source = torch.empty(2, dtype=dtype, device=device)
                    targets = (
                        torch.empty(1, dtype=dtype, device=device),
                        torch.empty(1, dtype=dtype, device=device),
                    )
                    torch._foreach_copy_(
                        targets,
                        (source[0:1], source[1:2]),
                        non_blocking=True,
                    )
                    retained.extend((source, *targets))
                torch.cuda.current_stream(device).synchronize()

    @staticmethod
    def _tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        """Compute bytes required for a tensor shape and dtype."""

        return int(math.prod(shape)) * _TORCH_DTYPE_BYTES[dtype]

    @staticmethod
    def _slot_bytes(slot: _DeviceSlot) -> int:
        """Return persistent bytes allocated by one physical product slot."""

        if (
            slot.persistent_buffer is not None
            or slot.tensor is None
            or slot.shape is None
            or math.prod(slot.shape) == 1
        ):
            return 0
        return int(slot.tensor.numel()) * int(slot.tensor.element_size())

    def _require_byte_capacity_locked(self, projected: int) -> None:
        """Reject a projected persistent allocation above the configured byte capacity."""

        if projected > self.byte_capacity:
            raise resource_error(
                f"device-product byte capacity is exhausted ({projected}>{self.byte_capacity})"
            )

    def bind_outputs(
        self,
        bindings: tuple[tuple[ProductRef, torch.device | str], ...],
        *,
        request_slots: Mapping[RequestKey, int] | None = None,
        buffer_placements: Mapping[BufferId, BufferPlacement] | None = None,
    ) -> tuple[DeviceProductWrite, ...]:
        """Reserve compatible device slots for operation outputs as one atomic binding batch."""

        return self.bind_output_batch(
            bindings,
            request_slots=request_slots,
            buffer_placements=buffer_placements,
        ).writes

    def bind_output_batch(
        self,
        bindings: tuple[tuple[ProductRef, torch.device | str], ...],
        *,
        request_slots: Mapping[RequestKey, int] | None = None,
        buffer_placements: Mapping[BufferId, BufferPlacement] | None = None,
    ) -> DeviceProductBindingBatch:
        """Atomically bind outputs and retain their direct scalar range."""

        device_bindings = bindings
        if not device_bindings:
            return DeviceProductBindingBatch(())

        # Storage classes select disjoint ownership arenas. Mixing classes in a
        # group would make rollback and publication semantics ambiguous.
        for reference, _device in device_bindings:
            _validate_owner(reference)
        relay = tuple(
            reference.storage_class is StorageClass.REQUEST_RELAY
            for reference, _device in device_bindings
        )
        if any(relay):
            if not all(relay):
                raise invalid_descriptor("request-relay bindings cannot share a generic group")
            if request_slots is None:
                raise invalid_descriptor("request-relay binding has no stable request slot")
            return self._bind_relay_outputs(device_bindings, request_slots)
        persistent = tuple(
            reference.uses_persistent_buffer() for reference, _device in device_bindings
        )
        if any(persistent):
            if not all(persistent):
                raise invalid_descriptor(
                    "persistent-buffer bindings cannot share a generic group"
                )
            if buffer_placements is None:
                raise invalid_descriptor("persistent output has no buffer placement")
            return self._bind_persistent_outputs(
                device_bindings,
                buffer_placements,
            )
        first_reference, first_raw_device = device_bindings[0]
        first_device_object = _resolved_device(first_raw_device)
        first_shape = _device_shape(first_reference)
        first_dtype = _device_dtype(first_reference.dtype)

        # Homogeneous scalar groups can bind a contiguous arena slice, allowing
        # producers to write and publish the entire group without packing.
        homogeneous = all(
            _resolved_device(device) == first_device_object
            and reference.shape_bound == first_reference.shape_bound
            and reference.dtype is first_reference.dtype
            for reference, device in device_bindings
        )
        if homogeneous and math.prod(first_shape) == 1:
            return self._bind_homogeneous_outputs(
                device_bindings,
                first_device_object,
                first_shape,
                first_dtype,
            )
        requested = tuple(
            (
                reference,
                _resolved_device(device),
                _device_shape(reference),
                _device_dtype(reference.dtype),
            )
            for reference, device in device_bindings
        )
        keys = [_reference_key(reference) for reference, _device, _shape, _dtype in requested]
        if len(set(keys)) != len(keys):
            raise invalid_descriptor("device-product registration repeats an output identity")
        with self._lock:
            # Validate every identity before reserving storage so descriptor
            # failures leave the bounded physical table unchanged.
            candidate_keys = {
                _reference_key(candidate.reference) for candidate in self._candidates.values()
            }
            for (reference, _device, _shape, _dtype), key in zip(requested, keys, strict=True):
                if int(reference.generation) < 1:
                    raise invalid_descriptor(
                        "device-product registration requires a positive logical generation"
                    )
                existing = self._entries.get(key)
                if existing is not None:
                    if existing.reference != reference:
                        raise invalid_descriptor("stale device-product logical generation")
                    raise invalid_descriptor("device-product output is already registered")
                if key in candidate_keys:
                    raise invalid_descriptor("device-product output already has a candidate")
            planned_slots: list[_DeviceSlot | None] = [None] * len(requested)
            reclaimed = False
            by_device: dict[
                str,
                list[tuple[int, ProductRef, torch.device, tuple[int, ...], torch.dtype]],
            ] = {}
            for index, (reference, device, shape, dtype) in enumerate(requested):
                by_device.setdefault(str(device), []).append(
                    (index, reference, device, shape, dtype)
                )
            try:
                for device_name, group in by_device.items():
                    ordered = sorted(
                        group,
                        key=lambda item: (
                            math.prod(item[3]) != 1,
                            str(item[4]),
                            item[3],
                            item[0],
                        ),
                    )
                    available, reclaimed = self._plan_compatible_slots_locked(
                        device_name,
                        ordered,
                        reclaimed=reclaimed,
                    )
                    for (
                        request_index,
                        _reference,
                        _device,
                        _shape,
                        _dtype,
                    ), slot in zip(ordered, available, strict=True):
                        planned_slots[request_index] = slot
            except BaseException:
                self._restore_planned_slots_locked(
                    slot for slot in planned_slots if slot is not None
                )
                raise
            writes: list[DeviceProductWrite] = []
            try:
                for request_index, (
                    (reference, device, shape, dtype),
                    key,
                ) in enumerate(zip(requested, keys, strict=True)):
                    planned_slot = planned_slots[request_index]
                    if planned_slot is None:
                        raise _invariant("device-product registration lost its physical-slot plan")
                    slot = self._acquire_slot_locked(
                        device,
                        shape,
                        dtype,
                        planned_slot,
                    )
                    write = DeviceProductWrite(
                        reference=reference,
                        slot=slot,
                        physical_generation=slot.generation,
                        binding_id=self._next_binding_id,
                    )
                    self._next_binding_id += 1
                    slot.owner = write.binding_id
                    self._occupied_slots[slot.device_name] = (
                        self._occupied_slots.get(slot.device_name, 0) + 1
                    )
                    self._candidates[write.binding_id] = write
                    writes.append(write)
            except BaseException:
                # Binding is transactional: newly owned and merely planned
                # slots both return to their prior free-list state.
                for entry in reversed(writes):
                    self._candidates.pop(entry.binding_id, None)
                    self._return_slot_locked(entry.slot)
                self._restore_planned_slots_locked(
                    slot for slot in planned_slots[len(writes) :] if slot is not None
                )
                raise
            return DeviceProductBindingBatch(tuple(writes))

    def _bind_persistent_outputs(
        self,
        bindings: tuple[tuple[ProductRef, torch.device | str], ...],
        placements: Mapping[BufferId, BufferPlacement],
    ) -> DeviceProductBindingBatch:
        """Bind logical outputs to caller-placed persistent buffers transactionally."""

        requested = tuple(
            (
                reference,
                _resolved_device(device),
                _device_shape(reference),
                _device_dtype(reference.dtype),
            )
            for reference, device in bindings
        )
        keys = tuple(
            _reference_key(reference)
            for reference, _device, _shape, _dtype in requested
        )
        if len(set(keys)) != len(keys):
            raise invalid_descriptor(
                "persistent-buffer registration repeats an output identity"
            )
        with self._lock:
            # Completed candidates release their slots before capacity planning.
            self._reclaim_ready_locked()
            candidate_keys = {
                _reference_key(candidate.reference)
                for candidate in self._candidates.values()
            }
            for (reference, _device, _shape, _dtype), key in zip(
                requested, keys, strict=True
            ):
                if key in self._entries or key in candidate_keys:
                    raise invalid_descriptor("persistent output is already registered")
                if reference.buffer_id not in placements:
                    raise invalid_descriptor("persistent output has no buffer placement")
            slots: list[_DeviceSlot] = []
            reclaimed = False
            try:
                for _reference, device, _shape, _dtype in requested:
                    device_name = str(device)
                    available, reclaimed = self._plan_slots_locked(
                        device_name,
                        1,
                        reclaimed=reclaimed,
                    )
                    slots.extend(available)
            except BaseException:
                self._restore_planned_slots_locked(slots)
                raise
            writes: list[DeviceProductWrite] = []
            try:
                # PersistentBuffers owns each tensor allocation; the product
                # slot carries its generation and unpublished write lease.
                for (reference, device, shape, dtype), slot in zip(
                    requested, slots, strict=True
                ):
                    binding = self.persistent_buffers.bind(
                        reference,
                        placements[reference.buffer_id],
                        device=device,
                        dtype=dtype,
                        shape=shape,
                    )
                    generation = slot.generation + 1
                    slot.generation = 1 if generation > _MAX_GENERATION else generation
                    slot.tensor = binding.tensor
                    slot.shape = shape
                    slot.dtype = dtype
                    slot.persistent_buffer = binding
                    write = DeviceProductWrite(
                        reference=reference,
                        slot=slot,
                        physical_generation=slot.generation,
                        binding_id=self._next_binding_id,
                    )
                    self._next_binding_id += 1
                    slot.owner = write.binding_id
                    self._occupied_slots[slot.device_name] = (
                        self._occupied_slots.get(slot.device_name, 0) + 1
                    )
                    self._candidates[write.binding_id] = write
                    writes.append(write)
            except BaseException:
                # No candidate survives unless every binding succeeds.
                for write in reversed(writes):
                    self._candidates.pop(write.binding_id, None)
                    self._return_slot_locked(write.slot)
                self._restore_planned_slots_locked(slots[len(writes) :])
                raise
            return DeviceProductBindingBatch(tuple(writes))

    def _bind_relay_outputs(
        self,
        bindings: tuple[tuple[ProductRef, torch.device | str], ...],
        request_slots: Mapping[RequestKey, int],
    ) -> DeviceProductBindingBatch:
        """Bind product references to stable request-and-lane relay slots for graph-safe output."""

        if self.request_capacity < 1 or self.relay_depth < 1:
            raise resource_error("worker has no request-relay arena")
        fields: dict[tuple[str, int, RequestKey, int, ProductKind, torch.dtype], int] = {}
        requested_rows = []
        for reference, raw_device in bindings:
            device = _resolved_device(raw_device)
            request_slot = int(request_slots.get(reference.request_key, 0))
            dtype = _device_dtype(reference.dtype)
            field_key = (
                str(device),
                request_slot,
                reference.request_key,
                int(reference.producer_op_id),
                reference.kind,
                dtype,
            )
            field = fields.get(field_key, 0)
            fields[field_key] = field + 1
            requested_rows.append((reference, device, request_slot, dtype, field))
        requested = tuple(requested_rows)
        if any(
            slot < 1
            or slot > self.request_capacity
            or math.prod(_device_shape(reference)) != 1
            for reference, _device, slot, _dtype, _field in requested
        ):
            raise invalid_descriptor("request-relay output has invalid slot or scalar geometry")
        keys = tuple(
            _reference_key(reference)
            for reference, _device, _slot, _dtype, _field in requested
        )
        if len(set(keys)) != len(keys):
            raise invalid_descriptor("request-relay registration repeats an output identity")
        with self._lock:
            self._reclaim_ready_locked()
            candidate_keys = {
                _reference_key(candidate.reference) for candidate in self._candidates.values()
            }
            for (reference, _device, _slot, _dtype, _field), key in zip(
                requested, keys, strict=True
            ):
                if int(reference.generation) < 1:
                    raise invalid_descriptor(
                        "request-relay registration requires a positive logical generation"
                    )
                existing = self._entries.get(key)
                if existing is not None:
                    if existing.reference != reference:
                        raise invalid_descriptor("stale request-relay logical generation")
                    raise invalid_descriptor("request-relay output is already registered")
                if key in candidate_keys:
                    raise invalid_descriptor("request-relay output already has a candidate")

            operation_lanes: dict[tuple[str, int, RequestKey, int], int] = {}
            for reference, device, request_slot, _dtype, _field in requested:
                operation = (
                    str(device),
                    request_slot,
                    reference.request_key,
                    int(reference.producer_op_id),
                )
                lane = self._relay_operation_lanes.get(operation)
                if lane is None:
                    lane = operation_lanes.get(operation)
                if lane is None:
                    lane = next(
                        (
                            candidate
                            for candidate in range(self.relay_depth)
                            if self._relay_lane_free_locked(
                                str(device), request_slot, candidate
                            )
                        ),
                        None,
                    )
                    if lane is None:
                        raise resource_error(
                            "request-relay unresolved window is exhausted"
                        )
                operation_lanes[operation] = lane

            writes: list[DeviceProductWrite] = []
            installed_operations: set[tuple[str, int, RequestKey, int]] = set()
            try:
                for reference, device, request_slot, dtype, field in requested:
                    operation = (
                        str(device),
                        request_slot,
                        reference.request_key,
                        int(reference.producer_op_id),
                    )
                    lane = operation_lanes[operation]
                    slot = self._relay_slot_locked(
                        device,
                        request_slot,
                        lane,
                        reference.kind,
                        dtype,
                        field,
                        operation,
                    )
                    if slot.owner is not None:
                        raise _invariant("request-relay lane was assigned more than once")
                    generation = slot.generation + 1
                    slot.generation = 1 if generation > _MAX_GENERATION else generation
                    write = DeviceProductWrite(
                        reference=reference,
                        slot=slot,
                        physical_generation=slot.generation,
                        binding_id=self._next_binding_id,
                    )
                    self._next_binding_id += 1
                    slot.owner = write.binding_id
                    self._candidates[write.binding_id] = write
                    self._relay_operation_lanes[operation] = lane
                    installed_operations.add(operation)
                    writes.append(write)
            except BaseException:
                for write in reversed(writes):
                    self._candidates.pop(write.binding_id, None)
                    self._return_slot_locked(write.slot)
                for operation in installed_operations:
                    self._release_relay_operation_locked(operation)
                raise
            return DeviceProductBindingBatch(tuple(writes))

    def _relay_lane_free_locked(
        self,
        device_name: str,
        request_slot: int,
        lane: int,
    ) -> bool:
        """Return whether a request relay lane has no bound operation."""

        return not any(
            slot.owner is not None
            for (name, row, candidate, _kind, _dtype, _field), slot in self._relay_slots.items()
            if name == device_name and row == request_slot and candidate == lane
        )

    def _relay_slot_locked(
        self,
        device: torch.device,
        request_slot: int,
        lane: int,
        kind: ProductKind,
        dtype: torch.dtype,
        field: int,
        operation: tuple[str, int, RequestKey, int],
    ) -> _DeviceSlot:
        """Resolve or create one stable scalar relay slot inside its geometry-specific arena."""

        device_name = str(device)
        key = (device_name, request_slot, lane, kind, dtype, int(field))
        slot = self._relay_slots.get(key)
        if slot is None:
            arena_key = (device_name, kind, dtype, int(field))
            arena = self._relay_arenas.get(arena_key)
            if arena is None:
                elements = (self.request_capacity + 1) * self.relay_depth
                projected = self._allocated_bytes + elements * _TORCH_DTYPE_BYTES[dtype]
                self._require_byte_capacity_locked(projected)
                arena = torch.empty((elements,), dtype=dtype, device=device)
                self._relay_arenas[arena_key] = arena
                self._allocated_bytes = projected
            index = request_slot * self.relay_depth + lane
            slot = _DeviceSlot(
                index=index,
                device_name=device_name,
                tensor=arena[index : index + 1],
                shape=(1,),
                dtype=dtype,
            )
            self._relay_slots[key] = slot
        if slot.relay_lane is not None and slot.relay_lane[:4] != operation:
            raise _invariant("request-relay slot retained a conflicting operation identity")
        slot.relay_lane = (*operation, lane)
        return slot

    def _release_relay_operation_locked(
        self,
        operation: tuple[str, int, RequestKey, int],
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
            tuple[tuple[ProductRef, torch.device | str], ...],
            ...,
        ],
        *,
        request_slots: Mapping[RequestKey, int] | None = None,
        buffer_placements: Mapping[BufferId, BufferPlacement] | None = None,
    ) -> tuple[DeviceProductBindingBatch, ...]:
        """Atomically bind output groups while preserving direct producer ranges."""

        bindings: list[DeviceProductBindingBatch] = []
        with self._lock:
            try:
                for group in groups:
                    if group:
                        bindings.append(
                            self.bind_output_batch(
                                group,
                                request_slots=request_slots,
                                buffer_placements=buffer_placements,
                            )
                        )
            except BaseException:
                self.abandon_writes(
                    tuple(write for binding in bindings for write in binding.writes)
                )
                raise
        return tuple(bindings)

    def _plan_compatible_slots_locked(
        self,
        device_name: str,
        requested: list[tuple[int, ProductRef, torch.device, tuple[int, ...], torch.dtype]],
        *,
        reclaimed: bool,
    ) -> tuple[list[_DeviceSlot], bool]:
        """Reserve geometry-compatible free slots for a heterogeneous binding request."""

        reclaimed = self._ensure_free_slots_locked(
            device_name,
            len(requested),
            reclaimed=reclaimed,
        )
        selected: list[_DeviceSlot] = []
        for _index, _reference, _device, shape, dtype in requested:
            candidate = self._take_compatible_slot_locked(device_name, shape, dtype)
            if candidate is None:
                candidate = self._take_free_slot_locked(device_name)
            selected.append(candidate)
        return selected, reclaimed

    def _bind_homogeneous_outputs(
        self,
        bindings: tuple[tuple[ProductRef, torch.device | str], ...],
        device: torch.device,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> DeviceProductBindingBatch:
        """Reserve same-device scalar outputs and link adjacent slots as one arena view."""

        with self._lock:
            candidate_keys = {
                _reference_key(candidate.reference) for candidate in self._candidates.values()
            }
            keys: list[_ReferenceKey] = []
            seen: set[_ReferenceKey] = set()
            for reference, _device in bindings:
                key = _reference_key(reference)
                if key in seen:
                    raise invalid_descriptor(
                        "device-product registration repeats an output identity"
                    )
                seen.add(key)
                keys.append(key)
                if int(reference.generation) < 1:
                    raise invalid_descriptor(
                        "device-product registration requires a positive logical generation"
                    )
                existing = self._entries.get(key)
                if existing is not None:
                    if existing.reference != reference:
                        raise invalid_descriptor("stale device-product logical generation")
                    raise invalid_descriptor("device-product output is already registered")
                if key in candidate_keys:
                    raise invalid_descriptor("device-product output already has a candidate")
            slots, _reclaimed = self._plan_slots_locked(
                str(device),
                len(bindings),
                reclaimed=False,
            )
            try:
                # Establish identical tensor geometry and advance each slot's
                # physical generation before assigning candidate ownership.
                self._prepare_homogeneous_slots_locked(device, shape, dtype, slots)
            except BaseException:
                self._restore_planned_slots_locked(slots)
                raise
            writes: list[DeviceProductWrite] = []
            owned_slots: list[_DeviceSlot] = []
            device_name = str(device)
            occupied_before = self._occupied_slots.get(device_name, 0)
            self._occupied_slots[device_name] = occupied_before + len(bindings)
            try:
                for (reference, _device), key, slot in zip(
                    bindings,
                    keys,
                    slots,
                    strict=True,
                ):
                    write = DeviceProductWrite(
                        reference=reference,
                        slot=slot,
                        physical_generation=slot.generation,
                        binding_id=self._next_binding_id,
                    )
                    self._next_binding_id += 1
                    owned_slots.append(slot)
                    slot.owner = write.binding_id
                    self._candidates[write.binding_id] = write
                    writes.append(write)
                bound_writes = tuple(writes)
                first_slot = bound_writes[0].slot
                start = first_slot.index

                # Only adjacent physical slots can expose a shared tensor view.
                contiguous = all(
                    write.slot.index == start + offset for offset, write in enumerate(bound_writes)
                )
                scalar: DeviceProductScalarBatch | None = None
                if contiguous:
                    arena = self._scalar_arenas.get((first_slot.device_name, dtype))
                    if arena is None:
                        raise _invariant("device-product scalar binding lost its physical arena")
                    scalar = DeviceProductScalarBatch(
                        writes=bound_writes,
                        tensor=arena[start : start + len(bound_writes)],
                        _table_token=self._binding_token,
                        _linked=True,
                    )
                    for write in bound_writes:
                        write._scalar_batch = scalar
                return DeviceProductBindingBatch(bound_writes, scalar)
            except BaseException:
                # Candidate ownership and occupancy accounting roll back as a unit.
                for entry in reversed(writes):
                    self._candidates.pop(entry.binding_id, None)
                for slot in owned_slots:
                    slot.owner = None
                self._occupied_slots[device_name] = occupied_before
                self._restore_planned_slots_locked(slots)
                raise

    def producer_write_views(
        self,
        writes: tuple[DeviceProductWrite, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return unpublished tensors from table-issued physical bindings."""

        if not writes:
            return ()
        with self._lock:
            entries = tuple(self._require_write_locked(write) for write in writes)
            if any(entry.producer_recorded for entry in entries):
                raise _invariant("device product was published more than once")
            tensors = tuple(entry.slot.tensor for entry in entries)
            if any(tensor is None for tensor in tensors):
                raise _invariant("device product has no physical tensor")
            return tuple(tensor for tensor in tensors if tensor is not None)

    def producer_scalar_batch(
        self,
        writes: tuple[DeviceProductWrite, ...],
    ) -> DeviceProductScalarBatch | None:
        """Bind a contiguous scalar range for one direct device producer."""

        if not writes:
            return None
        with self._lock:
            linked = writes[0]._scalar_batch
            if (
                linked is not None
                and len(linked.writes) == len(writes)
                and all(
                    linked_write is write
                    for linked_write, write in zip(
                        linked.writes,
                        writes,
                        strict=True,
                    )
                )
                and linked._table_token is self._binding_token
            ):
                self._require_live_scalar_batch_locked(linked)
                return linked
            return self._scalar_batch_locked(writes)

    def publish_write(
        self,
        write: DeviceProductWrite,
        value: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None = None,
        metadata: DeviceProductMetadata | None = None,
    ) -> torch.Tensor:
        """Commit a tensor into a reserved product slot with producer-stream synchronization."""

        with self._lock:
            entry = self._require_write_locked(write)
            result = self._publish_locked(entry, value, producer_event=producer_event)
            entry.metadata = metadata
            return result

    def _publish_locked(
        self,
        entry: DeviceProductWrite,
        value: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None,
    ) -> torch.Tensor:
        """Copy a value into a reserved slot and transfer readiness-event ownership."""

        if entry.producer_recorded:
            raise _invariant("device product was published more than once")
        target = entry.slot.tensor
        if target is None:
            raise _invariant("device product has no physical tensor")
        flat = value.detach().reshape(-1)
        if flat.numel() > target.numel():
            raise _invariant("device product exceeds its registered shape bound")
        view = target.reshape(-1)[: flat.numel()]
        view.copy_(flat.to(dtype=target.dtype), non_blocking=value.device.type == "cuda")
        if target.device.type == "cuda":
            entry.producer_event, entry.producer_stream = self._producer_event_locked(
                target.device,
                producer_event,
            )
            self._retain_event_locked(entry.producer_event, target.device)
        entry.actual_extent = int(flat.numel())
        entry.actual_shape = tuple(int(size) for size in value.shape)
        entry.producer_recorded = True
        if entry._scalar_batch is not None:
            entry._scalar_batch._published = True
        return view.reshape(value.shape)

    def publish_writes(
        self,
        writes: tuple[DeviceProductWrite, ...],
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

    def publish_scalar_batch(
        self,
        batch: DeviceProductScalarBatch,
        *,
        after_reads: tuple[DeviceProductRead, ...] = (),
        producer_event: torch.cuda.Event | None = None,
    ) -> None:
        """Publish a direct scalar batch after its device producer is enqueued."""

        with self._lock:
            event = self._publish_scalar_batch_locked(
                batch,
                producer_event=producer_event,
            )
            self._record_readers_after_write_locked(
                after_reads,
                batch.writes,
                event=event,
                device=batch.tensor.device,
            )

    def publish_scalar_group(
        self,
        batches: tuple[DeviceProductScalarBatch, ...],
        *,
        after_reads: tuple[DeviceProductRead, ...] = (),
    ) -> None:
        """Publish prefilled scalar output batches behind one producer fence."""

        if not batches:
            raise invalid_descriptor("scalar publication group is empty")
        with self._lock:
            device = batches[0].tensor.device
            for batch in batches:
                self._require_live_scalar_batch_locked(batch)
                if batch._published:
                    raise _invariant("device product was published more than once")
                if batch.tensor.device != device:
                    raise invalid_descriptor("scalar publication group spans incompatible devices")
            writes = tuple(write for batch in batches for write in batch.writes)
            if len({id(write) for write in writes}) != len(writes):
                raise invalid_descriptor("scalar publication group repeats an output binding")

            event: torch.cuda.Event | None = None
            if device.type == "cuda":
                event, _stream_id = self._record_event_locked(device)
            for batch in batches:
                self._publish_scalar_batch_locked(batch, producer_event=event)
            self._record_readers_after_write_locked(
                after_reads,
                writes,
                event=event,
                device=device,
            )

    def _publish_scalar_batch_locked(
        self,
        batch: DeviceProductScalarBatch,
        *,
        producer_event: torch.cuda.Event | None,
    ) -> torch.cuda.Event | None:
        """Publish a contiguous scalar batch into prebound relay or ordinary product slots."""

        self._require_live_scalar_batch_locked(batch)
        if batch._published:
            raise _invariant("device product was published more than once")
        if not batch._linked:
            entries = tuple(self._require_write_locked(write) for write in batch.writes)
            if any(entry.producer_recorded for entry in entries):
                raise _invariant("device product was published more than once")
        event: torch.cuda.Event | None = None
        if batch.tensor.device.type == "cuda":
            event, stream_id = self._producer_event_locked(
                batch.tensor.device,
                producer_event,
            )
            self._retain_event_locked(
                event,
                batch.tensor.device,
                len(batch.writes),
            )
        else:
            stream_id = None
        for entry in batch.writes:
            entry.producer_event = event
            entry.producer_stream = stream_id
            entry.actual_extent = 1
            entry.actual_shape = (1,)
            entry.producer_recorded = True
        batch._published = True
        return event

    def _publish_batch_locked(
        self,
        entries: tuple[DeviceProductWrite, ...],
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
        targets = tuple(entry.slot.tensor for entry in entries)
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
        slots = tuple(entry.slot.index for entry in entries)
        arena = (
            None
            if entries[0].slot.relay_lane is not None
            else self._scalar_arenas.get((str(first.device), first.dtype))
        )
        contiguous = slots == tuple(range(slots[0], slots[0] + len(slots)))
        if arena is not None and contiguous:
            destination = arena.narrow(0, slots[0], len(slots))
            aliases_destination = (
                source.device == destination.device
                and source.dtype == destination.dtype
                and source.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr()
                and int(source.storage_offset()) == int(destination.storage_offset())
            )
            if not aliases_destination:
                destination.copy_(
                    source,
                    non_blocking=source.device.type == "cuda",
                )
        elif arena is not None:
            aliases_arena = (
                source.device == arena.device
                and source.dtype == arena.dtype
                and source.untyped_storage().data_ptr() == arena.untyped_storage().data_ptr()
            )
            if aliases_arena:
                source = source.clone()
            torch._foreach_copy_(
                tensors,
                tuple(source[index : index + 1] for index in range(len(tensors))),
                non_blocking=source.device.type == "cuda",
            )
        else:
            for index, tensor in enumerate(tensors):
                source_view = source[index : index + 1]
                aliases_destination = (
                    source_view.device == tensor.device
                    and source_view.dtype == tensor.dtype
                    and source_view.untyped_storage().data_ptr()
                    == tensor.untyped_storage().data_ptr()
                    and int(source_view.storage_offset()) == int(tensor.storage_offset())
                )
                if not aliases_destination:
                    tensor.copy_(
                        source_view,
                        non_blocking=source.device.type == "cuda",
                    )

        event: torch.cuda.Event | None = None
        if first.device.type == "cuda":
            event, stream_id = self._producer_event_locked(
                first.device,
                producer_event,
            )
            self._retain_event_locked(event, first.device, len(entries))
        else:
            stream_id = None
        for entry in entries:
            entry.producer_event = event
            entry.producer_stream = stream_id
            entry.actual_extent = 1
            entry.actual_shape = (1,)
            entry.producer_recorded = True
        linked = entries[0]._scalar_batch
        if linked is not None:
            linked._published = True
        return tensors

    def publish_scalar_write(
        self,
        write: DeviceProductWrite,
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
        entry: DeviceProductWrite,
        value: bool | int,
        *,
        producer_event: torch.cuda.Event | None,
    ) -> torch.Tensor:
        """Store one host scalar in its reserved device slot and mark the write visible."""

        if entry.producer_recorded:
            raise _invariant("device product was published more than once")
        tensor = entry.slot.tensor
        if tensor is None:
            raise _invariant("device product has no physical tensor")
        tensor.reshape(-1)[:1].fill_(int(value))
        if tensor.device.type == "cuda":
            entry.producer_event, entry.producer_stream = self._producer_event_locked(
                tensor.device,
                producer_event,
            )
            self._retain_event_locked(entry.producer_event, tensor.device)
        entry.actual_extent = 1
        entry.actual_shape = (1,)
        entry.producer_recorded = True
        if entry._scalar_batch is not None:
            entry._scalar_batch._published = True
        return tensor.reshape(-1)[:1]

    def consume(
        self,
        reference: ProductRef,
        *,
        consumer_op_id: int,
        device: torch.device | str | None = None,
    ) -> DeviceProductRead:
        """Acquire a generation-safe read of a published product on the consumer device."""

        return self.consume_batch(((reference, consumer_op_id, device),))[0]

    def consume_candidate(
        self,
        write: DeviceProductWrite,
        *,
        consumer_op_id: int,
        device: torch.device | str | None = None,
    ) -> DeviceProductRead:
        """Read one unpublished candidate inside its producing lane."""

        with self._lock:
            entry = self._require_write_locked(write)
            if self._candidates.get(entry.binding_id) is not entry:
                raise _invariant("device-product candidate is not live")
            if not entry.producer_recorded:
                raise invalid_descriptor("device product was consumed before producer readiness")
            storage = entry.slot.tensor
            if storage is None:
                raise _invariant("device product has no physical tensor")
            target = storage.device if device is None else _resolved_device(device)
            if target != storage.device:
                raise invalid_descriptor("device product consumer names a different device")
            if target.type == "cuda":
                event = entry.producer_event
                if event is None:
                    raise _invariant("CUDA device product has no producer event")
                stream = torch.cuda.current_stream(target)
                if entry.producer_stream != int(stream.cuda_stream):
                    stream.wait_event(event)
            tensor = (
                storage
                if entry.actual_shape == entry.slot.shape
                else storage.reshape(-1)[: entry.actual_extent].reshape(entry.actual_shape)
            )
            return DeviceProductRead(
                tensor=tensor,
                consumer_op_id=int(consumer_op_id),
                _write=entry,
            )

    def consume_batch(
        self,
        requests: tuple[
            tuple[ProductRef, int, torch.device | str | None],
            ...,
        ],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[DeviceProductRead, ...]:
        """Resolve exact generations and enqueue each producer event once per stream."""

        if not requests:
            return ()
        shared_target = None if device is None else _resolved_device(device)
        if shared_target is not None:
            target_name = str(shared_target)
            with self._lock:
                shared_resolved: list[tuple[DeviceProductWrite, torch.Tensor, int]] = []
                for reference, consumer_op_id, requested_device in requests:
                    entry = self._require_locked(reference)
                    if entry.released or entry.logical_references < 1:
                        raise invalid_descriptor(
                            "device product was consumed after logical release"
                        )
                    if not entry.producer_recorded:
                        raise invalid_descriptor(
                            "device product was consumed before producer publication"
                        )
                    storage = entry.slot.tensor
                    if storage is None:
                        raise _invariant("device product has no physical tensor")
                    if (
                        requested_device is not None
                        and _resolved_device(requested_device) != shared_target
                    ):
                        raise invalid_descriptor(
                            "device-product batch names conflicting consumer devices"
                        )
                    if entry.slot.device_name != target_name:
                        raise invalid_descriptor("device product consumer names a different device")
                    tensor = (
                        storage
                        if entry.actual_shape == entry.slot.shape
                        else storage.reshape(-1)[: entry.actual_extent].reshape(entry.actual_shape)
                    )
                    shared_resolved.append((entry, tensor, int(consumer_op_id)))

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

                return tuple(
                    DeviceProductRead(
                        tensor=tensor,
                        consumer_op_id=consumer_op_id,
                        _write=entry,
                    )
                    for entry, tensor, consumer_op_id in shared_resolved
                )
        assert shared_target is None
        with self._lock:
            resolved: list[tuple[DeviceProductWrite, torch.Tensor, torch.device, int]] = []
            for reference, consumer_op_id, requested_device in requests:
                entry = self._require_locked(reference)
                if entry.released or entry.logical_references < 1:
                    raise invalid_descriptor("device product was consumed after logical release")
                if not entry.producer_recorded:
                    raise invalid_descriptor(
                        "device product was consumed before producer publication"
                    )
                storage = entry.slot.tensor
                if storage is None:
                    raise _invariant("device product has no physical tensor")
                tensor = (
                    storage
                    if entry.actual_shape == entry.slot.shape
                    else storage.reshape(-1)[: entry.actual_extent].reshape(entry.actual_shape)
                )
                target = (
                    storage.device
                    if requested_device is None
                    else _resolved_device(requested_device)
                )
                if target != storage.device:
                    raise invalid_descriptor("device product consumer names a different device")
                resolved.append((entry, tensor, target, int(consumer_op_id)))

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
                reads.append(
                    DeviceProductRead(
                        tensor=tensor,
                        consumer_op_id=consumer_op_id,
                        _write=entry,
                    )
                )
            return tuple(reads)

    def record_readers(
        self,
        reads: tuple[DeviceProductRead, ...],
        *,
        device: torch.device | str | None = None,
        after_writes: tuple[DeviceProductWrite, ...] = (),
    ) -> None:
        """Fence reads with a later output write or one event per consuming stream."""

        if not reads:
            return
        declared_target = None if device is None else _resolved_device(device)
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
                aligned: list[tuple[DeviceProductWrite, torch.cuda.Event | None]] = []
                for read, write in zip(reads, after_writes, strict=True):
                    entry = self._require_read_locked(read)
                    completion = self._require_write_locked(write)
                    event = completion.producer_event
                    if (
                        read.tensor.device != target
                        or int(completion.reference.producer_op_id) != int(read.consumer_op_id)
                        or (
                            target.type == "cuda"
                            and (
                                event is None
                                or completion.slot.tensor is None
                                or completion.slot.tensor.device != target
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
                        self._retain_event_locked(
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
            write_fences: dict[int, DeviceProductWrite] = {}
            for write in after_writes:
                entry = self._require_write_locked(write)
                if not entry.producer_recorded:
                    continue
                write_fences.setdefault(int(entry.reference.producer_op_id), entry)
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
                    pending: list[DeviceProductWrite] = []
                    for read, entry in entries:
                        completion_fence = write_fences.get(int(read.consumer_op_id))
                        if (
                            completion_fence is not None
                            and completion_fence.producer_event is not None
                            and completion_fence.slot.tensor is not None
                            and completion_fence.slot.tensor.device == target
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
                tuple[torch.device, list[tuple[DeviceProductRead, DeviceProductWrite]]],
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
                    completion_fence = write_fences.get(int(read.consumer_op_id))
                    if (
                        completion_fence is not None
                        and completion_fence.producer_event is not None
                        and completion_fence.slot.tensor is not None
                        and completion_fence.slot.tensor.device == target
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
        releases: Iterable[tuple[RequestKey, int]],
    ) -> None:
        """Release all product ownership associated with completed operation identities."""

        with self._lock:
            for request_key, raw_op_id in releases:
                op_id = int(raw_op_id)
                operation_key = (request_key, op_id)
                operation_writes = self._operation_writes.pop(operation_key, None)
                entries = list(
                    operation_writes
                    if isinstance(operation_writes, list)
                    else (() if operation_writes is None else (operation_writes,))
                )
                direct_key = (
                    int(request_key.authority_id),
                    int(request_key.request_id),
                    int(request_key.epoch),
                    op_id,
                    0,
                )
                direct = self._entries.get(direct_key)
                if (
                    direct is not None
                    and not direct._indexed
                    and direct.reference.request_key == request_key
                    and all(entry is not direct for entry in entries)
                ):
                    entries.append(direct)
                for entry in entries:
                    entry._indexed = False
                    if entry.released:
                        continue
                    entry.logical_references = 0
                    entry.released = True

    def release_generations(self, generations: Iterable[int]) -> None:
        """Release resident products addressed by scheduler-visible generations."""

        requested = {int(generation) for generation in generations}
        if not requested:
            return
        with self._lock:
            for entry in self._entries.values():
                if int(entry.reference.generation) not in requested:
                    continue
                self._detach_write_locked(entry)
                entry.logical_references = 0
                entry.released = True
            self._reclaim_ready_locked()

    def drop_request(
        self,
        request_id: int,
        *,
        retained_generations: frozenset[int] = frozenset(),
    ) -> None:
        """Release a request’s products except explicitly retained generations."""

        with self._lock:
            target_request_id = int(request_id)
            for entry in self._entries.values():
                if int(entry.reference.request_key.request_id) != target_request_id:
                    continue
                if int(entry.reference.generation) in retained_generations:
                    continue
                self._detach_write_locked(entry)
                entry.logical_references = 0
                entry.released = True
            self._reclaim_ready_locked()

    def abandon_writes(self, writes: tuple[DeviceProductWrite, ...]) -> None:
        """Return unpublished reserved writes without creating resident products."""

        with self._lock:
            for write in writes:
                try:
                    entry = self._require_write_locked(write)
                except WorkerError:
                    continue
                entry.logical_references = 0
                entry.released = True
            self._reclaim_ready_locked()

    def physical_generation(self, reference: ProductRef) -> int:
        """Resolve a resident logical product to its current physical slot generation."""

        with self._lock:
            return self._require_locked(reference).physical_generation

    def validate_writes(self, writes: tuple[DeviceProductWrite, ...]) -> None:
        """Verify that a write batch still refers to active unpublished reservations."""

        with self._lock:
            for write in writes:
                entry = self._require_write_locked(write)
                if self._candidates.get(entry.binding_id) is not entry:
                    raise _invariant("device-product candidate is not live")
                if not entry.producer_recorded:
                    raise _invariant("completion packing found an unpublished device product")

    def commit_writes(self, writes: tuple[DeviceProductWrite, ...]) -> None:
        """Make a validated set of produced candidate generations addressable."""

        if not writes:
            return
        with self._lock:
            entries = tuple(self._require_write_locked(write) for write in writes)
            keys = tuple(_reference_key(entry.reference) for entry in entries)
            if len(set(keys)) != len(keys):
                raise _invariant("device-product publication repeats a product identity")
            for key, entry in zip(keys, entries, strict=True):
                if self._candidates.get(entry.binding_id) is not entry:
                    raise _invariant("device-product candidate is not live")
                if not entry.producer_recorded:
                    raise _invariant("device-product candidate has no producer readiness")
                if key in self._entries:
                    raise _invariant("device-product publication identity is already resident")
            for key, entry in zip(keys, entries, strict=True):
                self._entries[key] = entry
                operation_key = (
                    entry.reference.request_key,
                    int(entry.reference.producer_op_id),
                )
                operation_writes = self._operation_writes.get(operation_key)
                if operation_writes is None:
                    self._operation_writes[operation_key] = entry
                elif isinstance(operation_writes, list):
                    operation_writes.append(entry)
                else:
                    self._operation_writes[operation_key] = [operation_writes, entry]
                entry._indexed = True
                self._candidates.pop(entry.binding_id)

    def _require_locked(self, reference: ProductRef) -> DeviceProductWrite:
        """Resolve a live generation-tagged product write or raise a descriptor error."""

        key = _reference_key(reference)
        entry = self._entries.get(key)
        if entry is None:
            raise invalid_descriptor("unknown device-product reference")
        if entry.reference != reference:
            raise invalid_descriptor("stale device-product logical generation")
        if entry.slot.owner != entry.binding_id:
            raise _invariant("device-product entry lost physical-slot ownership")
        if entry.slot.generation != entry.physical_generation:
            raise _invariant("device-product entry lost its physical generation")
        return entry

    def _require_write_locked(self, write: DeviceProductWrite) -> DeviceProductWrite:
        """Validate write-handle authenticity and return its live registry entry."""

        entry = write
        if entry.slot.owner != entry.binding_id:
            raise _invariant("stale device-product physical generation")
        if entry.slot.generation != entry.physical_generation:
            raise _invariant("stale device-product physical generation")
        return entry

    def _require_read_locked(self, read: DeviceProductRead) -> DeviceProductWrite:
        """Validate read-handle authenticity and return its live registry entry."""

        entry = read._write
        if entry.slot.owner != entry.binding_id:
            raise _invariant("stale device-product physical generation")
        if entry.slot.generation != entry.physical_generation:
            raise _invariant("stale device-product physical generation")
        return entry

    def _require_live_scalar_batch_locked(
        self,
        batch: DeviceProductScalarBatch,
    ) -> None:
        """Validate that a scalar batch still owns each contiguous write binding."""

        if batch._table_token is not self._binding_token:
            raise _invariant("device-product scalar binding belongs to a different table")
        if not batch._valid:
            raise _invariant("stale device-product physical generation")
        if batch._linked:
            return
        for write in batch.writes:
            self._require_write_locked(write)

    def _scalar_batch_locked(
        self,
        writes: tuple[DeviceProductWrite, ...],
    ) -> DeviceProductScalarBatch | None:
        """Recognize writes backed by one contiguous scalar arena and derive their slice."""

        entries = tuple(self._require_write_locked(write) for write in writes)
        if any(entry.producer_recorded for entry in entries):
            raise _invariant("device product was published more than once")
        first_slot = entries[0].slot
        if first_slot.relay_lane is not None:
            return None
        first = first_slot.tensor
        if first is None:
            raise _invariant("device product has no physical tensor")
        if first_slot.shape is None or math.prod(first_slot.shape) != 1:
            raise invalid_descriptor(
                "batched device-product publication requires scalar output bounds"
            )
        if first_slot.dtype is None:
            raise _invariant("device product has no physical dtype")
        start = first_slot.index
        for offset, entry in enumerate(entries):
            slot = entry.slot
            tensor = slot.tensor
            if tensor is None:
                raise _invariant("device product has no physical tensor")
            if slot.shape is None or math.prod(slot.shape) != 1:
                raise invalid_descriptor(
                    "batched device-product publication requires scalar output bounds"
                )
            if slot.device_name != first_slot.device_name or slot.dtype != first_slot.dtype:
                raise invalid_descriptor(
                    "batched device-product publication spans incompatible storage"
                )
            if slot.index != start + offset:
                return None
        arena = self._scalar_arenas.get((first_slot.device_name, first_slot.dtype))
        if arena is None:
            raise _invariant("device-product scalar binding lost its physical arena")
        return DeviceProductScalarBatch(
            writes=writes,
            tensor=arena[start : start + len(entries)],
            _table_token=self._binding_token,
        )

    def _plan_slots_locked(
        self,
        device_name: str,
        count: int,
        *,
        reclaimed: bool,
    ) -> tuple[list[_DeviceSlot], bool]:
        """Reserve a requested count of free slots from one device queue."""

        reclaimed = self._ensure_free_slots_locked(
            device_name,
            count,
            reclaimed=reclaimed,
        )
        return [self._take_free_slot_locked(device_name) for _ in range(count)], reclaimed

    def _ensure_free_slots_locked(
        self,
        device_name: str,
        count: int,
        *,
        reclaimed: bool,
    ) -> bool:
        """Ensure enough free physical slots, reclaiming ready generations at most once."""

        slots = self._slots.get(device_name)
        if slots is None:
            slots = [
                _DeviceSlot(index=index, device_name=device_name) for index in range(self.capacity)
            ]
            self._slots[device_name] = slots
            self._free_slots[device_name] = set(range(self.capacity))
            self._free_slot_queues[device_name] = deque(range(self.capacity))
        free = self._free_slots[device_name]
        if len(free) < count and not reclaimed:
            self._reclaim_ready_locked()
            reclaimed = True
        if len(free) < count:
            raise resource_error(
                f"device-product arena for {device_name} has no query-ready free generation"
            )
        return reclaimed

    def _take_free_slot_locked(self, device_name: str) -> _DeviceSlot:
        """Remove and return the next free slot for one device."""

        free = self._free_slots[device_name]
        queue = self._free_slot_queues[device_name]
        while queue:
            index = queue.popleft()
            if index in free:
                free.remove(index)
                return self._slots[device_name][index]
        if not free:
            raise _invariant("device-product free-slot index lost its capacity accounting")
        queue.extend(sorted(free))
        index = queue.popleft()
        free.remove(index)
        return self._slots[device_name][index]

    def _take_compatible_slot_locked(
        self,
        device_name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> _DeviceSlot | None:
        """Remove a free slot matching one device, shape, and dtype."""

        free = self._free_slots[device_name]
        queue = self._compatible_free_slots.get((device_name, shape, dtype))
        if queue is None:
            return None
        slots = self._slots[device_name]
        while queue:
            index = queue.popleft()
            slot = slots[index]
            if index in free and slot.shape == shape and slot.dtype == dtype:
                free.remove(index)
                return slot
        return None

    def _restore_planned_slots_locked(self, slots: Iterable[_DeviceSlot]) -> None:
        """Return uncommitted planned slots to their free queues after binding failure."""

        for slot in slots:
            if slot.owner is not None:
                continue
            free = self._free_slots[slot.device_name]
            if slot.index in free:
                continue
            free.add(slot.index)
            self._free_slot_queues[slot.device_name].append(slot.index)
            if slot.shape is not None and slot.dtype is not None:
                self._compatible_free_slots.setdefault(
                    (slot.device_name, slot.shape, slot.dtype),
                    deque(),
                ).append(slot.index)

    def _acquire_slot_locked(
        self,
        device: torch.device,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        slot: _DeviceSlot,
    ) -> _DeviceSlot:
        """Attach compatible storage to a reserved slot and account its persistent bytes."""

        name = str(device)
        if slot.device_name != name or slot.owner is not None:
            raise _invariant("device-product allocation lost its planned physical slot")
        generation = slot.generation + 1
        slot.generation = 1 if generation > _MAX_GENERATION else generation
        replacing = slot.tensor is None or slot.shape != shape or slot.dtype != dtype
        projected = self._allocated_bytes
        scalar = math.prod(shape) == 1
        arena_key = (name, dtype)
        arena = self._scalar_arenas.get(arena_key) if scalar else None
        if replacing:
            projected -= self._slot_bytes(slot)
            if scalar:
                if arena is None:
                    projected += self.capacity * _TORCH_DTYPE_BYTES[dtype]
            else:
                projected += self._tensor_bytes(shape, dtype)
        self._require_byte_capacity_locked(projected)
        try:
            if scalar:
                if arena is None:
                    arena = torch.empty((self.capacity,), dtype=dtype, device=device)
                    self._scalar_arenas[arena_key] = arena
                if replacing:
                    slot.tensor = arena[slot.index : slot.index + 1].reshape(shape)
                    slot.shape = shape
                    slot.dtype = dtype
            elif replacing:
                slot.tensor = torch.empty(shape, dtype=dtype, device=device)
                slot.shape = shape
                slot.dtype = dtype
        except BaseException:
            self._return_slot_locked(slot)
            raise
        self._allocated_bytes = projected
        return slot

    def _prepare_homogeneous_slots_locked(
        self,
        device: torch.device,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        slots: list[_DeviceSlot],
    ) -> None:
        """Allocate one contiguous arena for compatible empty slots and bind their views."""

        device_name = str(device)
        scalar = math.prod(shape) == 1
        arena_key = (device_name, dtype)
        arena = self._scalar_arenas.get(arena_key) if scalar else None
        projected = self._allocated_bytes
        replacements = [
            slot
            for slot in slots
            if slot.tensor is None or slot.shape != shape or slot.dtype != dtype
        ]
        replacement_indices = {slot.index for slot in replacements}
        for slot in replacements:
            projected -= self._slot_bytes(slot)
        if scalar:
            if arena is None:
                projected += self.capacity * _TORCH_DTYPE_BYTES[dtype]
        else:
            projected += len(replacements) * self._tensor_bytes(shape, dtype)
        self._require_byte_capacity_locked(projected)
        pending_arena: torch.Tensor | None = None
        pending_tensors: dict[int, torch.Tensor] = {}
        try:
            if scalar and arena is None:
                pending_arena = torch.empty((self.capacity,), dtype=dtype, device=device)
                arena = pending_arena
            if not scalar:
                pending_tensors = {
                    slot.index: torch.empty(shape, dtype=dtype, device=device)
                    for slot in replacements
                }
        except BaseException:
            self._restore_planned_slots_locked(slots)
            raise
        if pending_arena is not None:
            self._scalar_arenas[arena_key] = pending_arena
        for slot in slots:
            if slot.device_name != device_name or slot.owner is not None:
                raise _invariant("device-product allocation lost its planned physical slot")
            generation = slot.generation + 1
            slot.generation = 1 if generation > _MAX_GENERATION else generation
            if scalar:
                if slot.index in replacement_indices:
                    slot.tensor = cast(torch.Tensor, arena)[slot.index : slot.index + 1].reshape(
                        shape
                    )
                    slot.shape = shape
                    slot.dtype = dtype
            elif slot.index in replacement_indices:
                slot.tensor = pending_tensors[slot.index]
                slot.shape = shape
                slot.dtype = dtype
        self._allocated_bytes = projected

    def _return_slot_locked(self, slot: _DeviceSlot) -> None:
        """Return a detached physical slot to geometry-indexed free queues."""

        relay_lane = slot.relay_lane
        if relay_lane is not None:
            slot.owner = None
            operation = relay_lane[:4]
            self._release_relay_operation_locked(operation)
            if operation not in self._relay_operation_lanes:
                for candidate in self._relay_slots.values():
                    if candidate.relay_lane is not None and candidate.relay_lane[:4] == operation:
                        candidate.relay_lane = None
            return
        if slot.owner is not None:
            occupied = self._occupied_slots.get(slot.device_name, 0)
            if occupied < 1:
                raise _invariant("device-product occupancy underflow")
            self._occupied_slots[slot.device_name] = occupied - 1
        slot.owner = None
        if slot.persistent_buffer is not None:
            self.persistent_buffers.release(slot.persistent_buffer)
            slot.persistent_buffer = None
            slot.tensor = None
            slot.shape = None
            slot.dtype = None
        self._restore_planned_slots_locked((slot,))

    def _detach_write_locked(self, entry: DeviceProductWrite) -> None:
        """Remove a logical write and release its physical slot when no aliases remain."""

        if not entry._indexed:
            return
        reference = entry.reference
        operation_key = (
            reference.request_key,
            int(reference.producer_op_id),
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

    def _retain_event_locked(
        self,
        event: torch.cuda.Event,
        device: torch.device,
        count: int = 1,
    ) -> None:
        """Retain shared ownership references to an active device event."""

        self.event_pool.retain(event, device, count)

    def _release_event_locked(
        self,
        event: torch.cuda.Event,
        count: int = 1,
    ) -> None:
        """Release shared ownership references from an active device event."""

        self.event_pool.release(event, count)

    def _append_reader_event_locked(
        self,
        entry: DeviceProductWrite,
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
        self._retain_event_locked(event, device)

    def _record_readers_after_write_locked(
        self,
        reads: tuple[DeviceProductRead, ...],
        writes: tuple[DeviceProductWrite, ...],
        *,
        event: torch.cuda.Event | None,
        device: torch.device,
    ) -> None:
        """Attach a shared reader-completion event to newly published writes."""

        if not reads:
            return
        entries: list[DeviceProductWrite] = []
        aligned_reads: list[DeviceProductRead] = []
        if len(reads) == len(writes) and all(
            int(read.consumer_op_id) == int(write.reference.producer_op_id)
            for read, write in zip(reads, writes, strict=True)
        ):
            for read in reads:
                if read._recorded:
                    continue
                entry = self._require_read_locked(read)
                if entry.slot.device_name != str(device):
                    raise _invariant("device-product reader completed on a different device")
                entries.append(entry)
                aligned_reads.append(read)
        else:
            consumer_ops = {int(write.reference.producer_op_id) for write in writes}
            for read in reads:
                if read._recorded or int(read.consumer_op_id) not in consumer_ops:
                    continue
                entry = self._require_read_locked(read)
                if entry.slot.device_name != str(device):
                    raise _invariant("device-product reader completed on a different device")
                entries.append(entry)
                aligned_reads.append(read)
        if not entries:
            return
        if device.type == "cuda":
            if event is None:
                raise _invariant("CUDA device-product write has no producer event")
            if all(entry.reader_events is None for entry in entries):
                for entry in entries:
                    entry.reader_events = event
                self._retain_event_locked(event, device, len(entries))
            else:
                for entry in entries:
                    self._append_reader_event_locked(entry, event, device)
        for read in aligned_reads:
            read._recorded = True

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

        def readers_ready(entry: DeviceProductWrite) -> bool:
            """Require every consumer event associated with a product generation to complete."""

            events = entry.reader_events
            if events is None:
                return True
            if isinstance(events, list):
                return all(ready(event) for event in events)
            return ready(events)

        for key, entry in tuple(self._entries.items()):
            if (
                entry.released
                and entry.logical_references == 0
                and ready(entry.producer_event)
                and readers_ready(entry)
            ):
                if self._entries.get(key) is not entry:
                    raise _invariant("device-product reclamation lost its indexed generation")
                if entry.slot.owner != entry.binding_id:
                    raise _invariant("device-product reclamation found a stale physical generation")
                if entry.slot.generation != entry.physical_generation:
                    raise _invariant("device-product reclamation found a stale physical generation")
                self._entries.pop(key)
                self._detach_write_locked(entry)
                if entry._scalar_batch is not None:
                    entry._scalar_batch._valid = False
                self._return_slot_locked(entry.slot)
                if entry.producer_event is not None:
                    release_event(entry.producer_event)
                reader_events = entry.reader_events
                if isinstance(reader_events, list):
                    for event in reader_events:
                        release_event(event)
                elif reader_events is not None:
                    release_event(reader_events)
                reclaimed += 1
        for binding_id, entry in tuple(self._candidates.items()):
            if (
                entry.released
                and entry.logical_references == 0
                and ready(entry.producer_event)
                and readers_ready(entry)
            ):
                if self._candidates.get(binding_id) is not entry:
                    raise _invariant("device-product reclamation lost its candidate generation")
                self._candidates.pop(binding_id)
                if entry._scalar_batch is not None:
                    entry._scalar_batch._valid = False
                self._return_slot_locked(entry.slot)
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
            self._release_event_locked(event, count)
        return reclaimed


__all__ = [
    "DeviceProductRead",
    "DeviceProducts",
    "DeviceProductScalarBatch",
    "DeviceProductBindingBatch",
    "DeviceProductMetadata",
    "DeviceProductWrite",
    "ImageRange",
    "device_product_capacity_bytes",
    "device_product_storage",
]
