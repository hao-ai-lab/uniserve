"""Transactional storage and event-safe device-product lifetimes."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from itertools import chain
from threading import RLock
from typing import Final, cast

import torch

from ..batch import (
    DType,
    ProductRef,
    RequestKey,
    StaticDim,
    StorageClass,
    TokenMode,
)
from ..foundation.errors import ErrorCode, WorkerError, invalid_descriptor, resource_error
from .device_events import DeviceEventPool
from .host_staging import canonical_device

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


def _invariant(message: str) -> WorkerError:
    return WorkerError(
        code=ErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


_ReferenceKey = tuple[int, int, int, int, int]
_OperationKey = tuple[RequestKey, int]


def _reference_key(reference: ProductRef) -> _ReferenceKey:
    key = reference.request_key
    return (
        int(key.authority_id),
        int(key.session_id),
        int(key.epoch),
        int(reference.producer_op_id),
        int(reference.output_index),
    )


def _device_dtype(dtype: DType) -> torch.dtype:
    return _DEVICE_DTYPES[dtype]


def _device_shape(reference: ProductRef) -> tuple[int, ...]:
    dims = tuple(
        dim.extent if isinstance(dim, StaticDim) else dim.bound
        for dim in reference.shape_bound.dims
    )
    return dims or (1,)


def _event_ready(event: torch.cuda.Event | None) -> bool:
    return event is None or bool(event.query())


def _resolved_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device) and (device.type != "cuda" or device.index is not None):
        return device
    return canonical_device(device)


@dataclass(slots=True)
class _DeviceSlot:
    index: int
    device_name: str
    generation: int = 0
    owner: int | None = None
    tensor: torch.Tensor | None = None
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None


@dataclass(slots=True)
class DeviceProductWrite:
    """One table-issued physical binding retained through producer submission."""

    reference: ProductRef
    producer_plan_digest: str
    slot: _DeviceSlot
    physical_generation: int
    binding_id: int
    producer_event: torch.cuda.Event | None = None
    producer_stream: int | None = None
    producer_recorded: bool = False
    reader_events: torch.cuda.Event | list[torch.cuda.Event] | None = None
    logical_references: int = 1
    released: bool = False
    _indexed: bool = True
    actual_extent: int = 0
    actual_shape: tuple[int, ...] = ()
    inflight_read_batches: int = 0
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
        return self._write.reference

    @property
    def physical_slot(self) -> int:
        return self._write.slot.index

    @property
    def physical_generation(self) -> int:
        return self._write.physical_generation


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


@dataclass(slots=True)
class DeviceProductContinuationBatch:
    """One aligned device-parent read and scalar-output write transaction."""

    inputs: tuple[torch.Tensor, ...]
    writes: tuple[DeviceProductWrite, ...]
    scalar: DeviceProductScalarBatch | None
    _parents: tuple[DeviceProductWrite, ...] = field(repr=False, compare=False)
    _consumer_op_ids: tuple[int, ...] = field(repr=False, compare=False)
    _device: torch.device = field(repr=False, compare=False)
    _table_token: object = field(repr=False, compare=False)
    _published: bool = field(default=False, repr=False, compare=False)
    _readers_recorded: bool = field(default=False, repr=False, compare=False)

    @property
    def published(self) -> bool:
        return self._published


class DeviceProductTable:
    """Bounded physical slots for generation-tagged device products.

    Registration, lookup, stream waits, reader recording, release, and
    reclamation all validate the exact logical and physical generation here.
    Reclamation only queries events; it never synchronizes a device stream.
    """

    def __init__(
        self,
        *,
        capacity: int,
        event_pool: DeviceEventPool | None = None,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self._slots: dict[str, list[_DeviceSlot]] = {}
        self._occupied_slots: dict[str, int] = {}
        self._allocation_cursor: dict[str, int] = {}
        self._scalar_arenas: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.event_pool = DeviceEventPool() if event_pool is None else event_pool
        self._entries: dict[_ReferenceKey, DeviceProductWrite] = {}
        self._operation_writes: dict[
            _OperationKey,
            DeviceProductWrite | list[DeviceProductWrite],
        ] = {}
        self._next_binding_id = 1
        self._binding_token = object()
        self._lock = RLock()

    def bind_outputs(
        self,
        bindings: tuple[tuple[ProductRef, str, torch.device | str], ...],
    ) -> tuple[DeviceProductWrite, ...]:
        return self.bind_output_batch(bindings).writes

    def bind_output_batch(
        self,
        bindings: tuple[tuple[ProductRef, str, torch.device | str], ...],
    ) -> DeviceProductBindingBatch:
        """Atomically bind outputs and retain their direct scalar range."""

        device_bindings = tuple(
            binding
            for binding in bindings
            if binding[0].storage_class in {StorageClass.DEVICE_TENSOR, StorageClass.LATENT_ARENA}
        )
        if not device_bindings:
            return DeviceProductBindingBatch(())
        first_reference, _first_digest, first_raw_device = device_bindings[0]
        first_device_object = _resolved_device(first_raw_device)
        first_shape = _device_shape(first_reference)
        first_dtype = _device_dtype(first_reference.dtype)
        homogeneous = all(
            _resolved_device(device) == first_device_object
            and reference.shape_bound == first_reference.shape_bound
            and reference.dtype is first_reference.dtype
            for reference, _plan_digest, device in device_bindings
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
                plan_digest,
                _resolved_device(device),
                _device_shape(reference),
                _device_dtype(reference.dtype),
            )
            for reference, plan_digest, device in device_bindings
        )
        keys = [
            _reference_key(reference) for reference, _digest, _device, _shape, _dtype in requested
        ]
        if len(set(keys)) != len(keys):
            raise invalid_descriptor("device-product registration repeats an output identity")
        with self._lock:
            for (reference, _digest, _device, _shape, _dtype), key in zip(
                requested, keys, strict=True
            ):
                if int(reference.generation) < 1:
                    raise invalid_descriptor(
                        "device-product registration requires a positive logical generation"
                    )
                existing = self._entries.get(key)
                if existing is not None:
                    if existing.reference != reference:
                        raise invalid_descriptor("stale device-product logical generation")
                    raise invalid_descriptor("device-product output is already registered")
            planned_slots: list[_DeviceSlot | None] = [None] * len(requested)
            reclaimed = False
            by_device: dict[
                str,
                list[tuple[int, ProductRef, torch.device, tuple[int, ...], torch.dtype]],
            ] = {}
            for index, (reference, _digest, device, shape, dtype) in enumerate(requested):
                by_device.setdefault(str(device), []).append(
                    (index, reference, device, shape, dtype)
                )
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
                available, reclaimed = self._plan_slots_locked(
                    device_name,
                    len(ordered),
                    reclaimed=reclaimed,
                )
                available = self._prefer_compatible_slots_locked(
                    device_name,
                    ordered,
                    available,
                )
                for (
                    request_index,
                    _reference,
                    _device,
                    _shape,
                    _dtype,
                ), slot in zip(ordered, available, strict=True):
                    planned_slots[request_index] = slot
            registered: list[_ReferenceKey] = []
            writes: list[DeviceProductWrite] = []
            try:
                for request_index, (
                    (reference, plan_digest, device, shape, dtype),
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
                        producer_plan_digest=str(plan_digest),
                        slot=slot,
                        physical_generation=slot.generation,
                        binding_id=self._next_binding_id,
                    )
                    self._next_binding_id += 1
                    slot.owner = write.binding_id
                    self._occupied_slots[slot.device_name] = (
                        self._occupied_slots.get(slot.device_name, 0) + 1
                    )
                    self._entries[key] = write
                    operation_key = (
                        reference.request_key,
                        int(reference.producer_op_id),
                    )
                    operation_writes = self._operation_writes.get(operation_key)
                    if operation_writes is None:
                        self._operation_writes[operation_key] = write
                    elif isinstance(operation_writes, list):
                        operation_writes.append(write)
                    else:
                        self._operation_writes[operation_key] = [
                            operation_writes,
                            write,
                        ]
                    registered.append(key)
                    writes.append(write)
            except BaseException:
                for key in reversed(registered):
                    entry = self._entries.pop(key)
                    self._detach_write_locked(entry)
                    self._return_slot_locked(entry.slot)
                raise
            return DeviceProductBindingBatch(tuple(writes))

    def _prefer_compatible_slots_locked(
        self,
        device_name: str,
        requested: list[tuple[int, ProductRef, torch.device, tuple[int, ...], torch.dtype]],
        planned: list[_DeviceSlot],
    ) -> list[_DeviceSlot]:
        slots = self._slots[device_name]
        selected: list[_DeviceSlot] = []
        selected_ids: set[int] = set()
        for _index, _reference, _device, shape, dtype in requested:
            compatible = next(
                (
                    slot
                    for slot in slots
                    if slot.owner is None
                    and slot.index not in selected_ids
                    and slot.shape == shape
                    and slot.dtype == dtype
                ),
                None,
            )
            candidate = compatible or next(
                (slot for slot in planned if slot.index not in selected_ids),
                None,
            )
            if candidate is None:
                candidate = next(
                    slot for slot in slots if slot.owner is None and slot.index not in selected_ids
                )
            selected.append(candidate)
            selected_ids.add(candidate.index)
        return selected

    def _bind_homogeneous_outputs(
        self,
        bindings: tuple[tuple[ProductRef, str, torch.device | str], ...],
        device: torch.device,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        index_operations: bool = True,
    ) -> DeviceProductBindingBatch:
        with self._lock:
            keys: list[_ReferenceKey] = []
            seen: set[_ReferenceKey] = set()
            for reference, _digest, _device in bindings:
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
            slots, _reclaimed = self._plan_slots_locked(
                str(device),
                len(bindings),
                reclaimed=False,
            )
            self._prepare_homogeneous_slots_locked(device, shape, dtype, slots)
            registered: list[_ReferenceKey] = []
            writes: list[DeviceProductWrite] = []
            owned_slots: list[_DeviceSlot] = []
            device_name = str(device)
            occupied_before = self._occupied_slots.get(device_name, 0)
            self._occupied_slots[device_name] = occupied_before + len(bindings)
            try:
                for (reference, plan_digest, _device), key, slot in zip(
                    bindings,
                    keys,
                    slots,
                    strict=True,
                ):
                    write = DeviceProductWrite(
                        reference=reference,
                        producer_plan_digest=str(plan_digest),
                        slot=slot,
                        physical_generation=slot.generation,
                        binding_id=self._next_binding_id,
                        _indexed=index_operations,
                    )
                    self._next_binding_id += 1
                    owned_slots.append(slot)
                    slot.owner = write.binding_id
                    self._entries[key] = write
                    registered.append(key)
                    if index_operations:
                        operation_key = (
                            reference.request_key,
                            int(reference.producer_op_id),
                        )
                        operation_writes = self._operation_writes.get(operation_key)
                        if operation_writes is None:
                            self._operation_writes[operation_key] = write
                        elif isinstance(operation_writes, list):
                            operation_writes.append(write)
                        else:
                            self._operation_writes[operation_key] = [
                                operation_writes,
                                write,
                            ]
                    writes.append(write)
                bound_writes = tuple(writes)
                first_slot = bound_writes[0].slot
                start = first_slot.index
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
                for key in reversed(registered):
                    entry = self._entries.pop(key)
                    self._detach_write_locked(entry)
                for slot in owned_slots:
                    slot.owner = None
                self._occupied_slots[device_name] = occupied_before
                raise

    def bind_scalar_continuation(
        self,
        *,
        outputs: tuple[tuple[ProductRef, str], ...],
        parents: tuple[tuple[ProductRef, int, str], ...],
        device: torch.device | str,
    ) -> DeviceProductContinuationBatch:
        """Pin aligned device parents and atomically bind scalar descendants."""

        if not outputs or len(outputs) != len(parents):
            raise invalid_descriptor(
                "device continuation requires aligned non-empty parents and outputs"
            )
        target = _resolved_device(device)
        target_name = str(target)
        with self._lock:
            parent_entries: list[DeviceProductWrite] = []
            input_tensors: list[torch.Tensor] = []
            consumer_op_ids: list[int] = []
            for reference, consumer_op_id, producer_plan_digest in parents:
                entry = self._require_locked(reference)
                if entry.released or entry.logical_references < 1:
                    raise invalid_descriptor("device product was consumed after logical release")
                if not entry.producer_recorded:
                    raise invalid_descriptor(
                        "device product was consumed before producer publication"
                    )
                if entry.producer_plan_digest != producer_plan_digest:
                    raise invalid_descriptor(
                        "device parent plan digest does not match its producer"
                    )
                storage = entry.slot.tensor
                if storage is None:
                    raise _invariant("device product has no physical tensor")
                if entry.slot.device_name != target_name:
                    raise invalid_descriptor("device product consumer names a different device")
                tensor = (
                    storage
                    if entry.actual_shape == entry.slot.shape
                    else storage.reshape(-1)[: entry.actual_extent].reshape(entry.actual_shape)
                )
                parent_entries.append(entry)
                input_tensors.append(tensor)
                consumer_op_ids.append(int(consumer_op_id))

            if target.type == "cuda":
                stream = torch.cuda.current_stream(target)
                stream_id = int(stream.cuda_stream)
                first_event = parent_entries[0].producer_event
                if first_event is None:
                    raise _invariant("CUDA device product has no producer event")
                if all(entry.producer_event is first_event for entry in parent_entries):
                    if any(entry.producer_stream != stream_id for entry in parent_entries):
                        stream.wait_event(first_event)
                else:
                    waited: set[int] = set()
                    for entry in parent_entries:
                        event = entry.producer_event
                        if event is None:
                            raise _invariant("CUDA device product has no producer event")
                        identity = id(event)
                        if identity in waited:
                            continue
                        stream.wait_event(event)
                        waited.add(identity)

            first_output = outputs[0][0]
            output_dtype = _device_dtype(first_output.dtype)
            if any(
                reference.storage_class
                not in {
                    StorageClass.DEVICE_TENSOR,
                    StorageClass.LATENT_ARENA,
                }
                or int(reference.output_index) != 0
                or _device_shape(reference) != (1,)
                or _device_dtype(reference.dtype) != output_dtype
                for reference, _plan_digest in outputs
            ):
                raise invalid_descriptor(
                    "device continuation outputs must use compatible scalar storage"
                )
            binding: DeviceProductBindingBatch | None = None
            try:
                binding = self._bind_homogeneous_outputs(
                    tuple((reference, plan_digest, target) for reference, plan_digest in outputs),
                    target,
                    (1,),
                    output_dtype,
                    index_operations=False,
                )
                if len(binding.writes) != len(outputs):
                    raise _invariant("device continuation lost an output registration")
                if any(
                    int(write.reference.producer_op_id) != consumer_op_id
                    for write, consumer_op_id in zip(
                        binding.writes,
                        consumer_op_ids,
                        strict=True,
                    )
                ):
                    raise invalid_descriptor(
                        "device continuation output does not match its consumer operation"
                    )
                continuation = DeviceProductContinuationBatch(
                    inputs=tuple(input_tensors),
                    writes=binding.writes,
                    scalar=binding.scalar,
                    _parents=tuple(parent_entries),
                    _consumer_op_ids=tuple(consumer_op_ids),
                    _device=target,
                    _table_token=self._binding_token,
                )
            except BaseException:
                if binding is not None:
                    self.abandon_writes(binding.writes)
                raise
            for entry in parent_entries:
                entry.inflight_read_batches += 1
            return continuation

    def producer_view(self, reference: ProductRef) -> torch.Tensor:
        """Return a bound unpublished tensor for the registered producer."""

        return self.producer_views((reference,))[0]

    def producer_views(
        self,
        references: tuple[ProductRef, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return bound unpublished tensors under one generation check batch."""

        if not references:
            return ()
        with self._lock:
            entries = tuple(self._require_locked(reference) for reference in references)
            if any(entry.producer_recorded for entry in entries):
                raise _invariant("device product was published more than once")
            tensors = tuple(entry.slot.tensor for entry in entries)
            if any(tensor is None for tensor in tensors):
                raise _invariant("device product has no physical tensor")
            return tuple(tensor for tensor in tensors if tensor is not None)

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
                and linked.writes is writes
                and linked._table_token is self._binding_token
            ):
                self._require_live_scalar_batch_locked(linked)
                return linked
            return self._scalar_batch_locked(writes)

    def publish(
        self,
        reference: ProductRef,
        value: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        with self._lock:
            entry = self._require_locked(reference)
            return self._publish_locked(entry, value, producer_event=producer_event)

    def publish_write(
        self,
        write: DeviceProductWrite,
        value: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        with self._lock:
            entry = self._require_write_locked(write)
            return self._publish_locked(entry, value, producer_event=producer_event)

    def _publish_locked(
        self,
        entry: DeviceProductWrite,
        value: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None,
    ) -> torch.Tensor:
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

    def publish_batch(
        self,
        references: tuple[ProductRef, ...],
        values: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Publish aligned scalar products with one stream-completion event."""

        if not references:
            return ()
        with self._lock:
            entries = tuple(self._require_locked(reference) for reference in references)
            return self._publish_batch_locked(
                entries,
                values,
                producer_event=producer_event,
            )

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

    def publish_continuation(
        self,
        continuation: DeviceProductContinuationBatch,
        values: torch.Tensor | None = None,
        *,
        after_reads: tuple[DeviceProductRead, ...] = (),
        producer_event: torch.cuda.Event | None = None,
    ) -> None:
        """Publish scalar descendants and fence their pinned device parents."""

        with self._lock:
            self._require_continuation_locked(continuation)
            if continuation._published:
                raise _invariant("device continuation was published more than once")
            scalar = continuation.scalar
            if scalar is not None:
                if values is not None:
                    source = values.detach().reshape(-1)
                    if int(source.numel()) != len(continuation.writes):
                        raise invalid_descriptor(
                            "device continuation requires one scalar per output"
                        )
                    aliases_destination = (
                        source.device == scalar.tensor.device
                        and source.dtype == scalar.tensor.dtype
                        and source.untyped_storage().data_ptr()
                        == scalar.tensor.untyped_storage().data_ptr()
                        and int(source.storage_offset()) == int(scalar.tensor.storage_offset())
                    )
                    if not aliases_destination:
                        scalar.tensor.copy_(
                            source.to(dtype=scalar.tensor.dtype),
                            non_blocking=source.device.type == "cuda",
                        )
                event = self._publish_scalar_batch_locked(
                    scalar,
                    producer_event=producer_event,
                )
            else:
                if values is None:
                    raise invalid_descriptor(
                        "non-contiguous device continuation requires scalar values"
                    )
                self._publish_batch_locked(
                    continuation.writes,
                    values,
                    producer_event=producer_event,
                )
                event = continuation.writes[0].producer_event
            continuation._published = True
            self._record_continuation_readers_locked(
                continuation,
                event=event,
            )
            self._record_readers_after_write_locked(
                after_reads,
                continuation.writes,
                event=event,
                device=continuation._device,
            )

    def finish_continuation(
        self,
        continuation: DeviceProductContinuationBatch,
    ) -> None:
        """Fence pinned parents when continuation execution exits before publish."""

        with self._lock:
            self._require_continuation_locked(continuation)
            if continuation._readers_recorded:
                return
            event: torch.cuda.Event | None = None
            if continuation._device.type == "cuda":
                event, _stream_id = self._record_event_locked(continuation._device)
            self._record_continuation_readers_locked(
                continuation,
                event=event,
            )

    def validate_continuation(
        self,
        continuation: DeviceProductContinuationBatch,
    ) -> None:
        with self._lock:
            self._require_continuation_locked(continuation)
            if not continuation._published:
                raise _invariant("completion packing found an unpublished device continuation")
            if not continuation._readers_recorded:
                raise _invariant("device continuation did not fence its parent products")

    def _publish_scalar_batch_locked(
        self,
        batch: DeviceProductScalarBatch,
        *,
        producer_event: torch.cuda.Event | None,
    ) -> torch.cuda.Event | None:
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
        arena = self._scalar_arenas.get((str(first.device), first.dtype))
        if arena is not None and slots == tuple(range(slots[0], slots[0] + len(slots))):
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

    def publish_scalar(
        self,
        reference: ProductRef,
        value: bool | int,
        *,
        producer_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        with self._lock:
            entry = self._require_locked(reference)
            return self._publish_scalar_locked(
                entry,
                value,
                producer_event=producer_event,
            )

    def publish_scalar_write(
        self,
        write: DeviceProductWrite,
        value: bool | int,
        *,
        producer_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
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
        producer_plan_digest: str | None = None,
        device: torch.device | str | None = None,
    ) -> DeviceProductRead:
        return self.consume_batch(((reference, consumer_op_id, producer_plan_digest, device),))[0]

    def consume_batch(
        self,
        requests: tuple[
            tuple[ProductRef, int, str | None, torch.device | str | None],
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
                for reference, consumer_op_id, producer_plan_digest, requested_device in requests:
                    entry = self._require_locked(reference)
                    if entry.released or entry.logical_references < 1:
                        raise invalid_descriptor(
                            "device product was consumed after logical release"
                        )
                    if not entry.producer_recorded:
                        raise invalid_descriptor(
                            "device product was consumed before producer publication"
                        )
                    if (
                        producer_plan_digest is not None
                        and entry.producer_plan_digest != producer_plan_digest
                    ):
                        raise invalid_descriptor(
                            "device parent plan digest does not match its producer"
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
            for reference, consumer_op_id, producer_plan_digest, requested_device in requests:
                entry = self._require_locked(reference)
                if entry.released or entry.logical_references < 1:
                    raise invalid_descriptor("device product was consumed after logical release")
                if not entry.producer_recorded:
                    raise invalid_descriptor(
                        "device product was consumed before producer publication"
                    )
                if (
                    producer_plan_digest is not None
                    and entry.producer_plan_digest != producer_plan_digest
                ):
                    raise invalid_descriptor(
                        "device parent plan digest does not match its producer"
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

    def record_reader(
        self,
        read: DeviceProductRead,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.record_readers((read,), device=device)

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

    def release_operation(self, request_key: RequestKey, op_id: int) -> None:
        self.release_operations(((request_key, op_id),))

    def release_operations(
        self,
        releases: Iterable[tuple[RequestKey, int]],
    ) -> None:
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
                    int(request_key.session_id),
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

    def drop_session(
        self,
        session_id: int,
        *,
        retained_generations: frozenset[int] = frozenset(),
    ) -> None:
        with self._lock:
            target_session_id = int(session_id)
            for entry in self._entries.values():
                if int(entry.reference.request_key.session_id) != target_session_id:
                    continue
                if int(entry.reference.generation) in retained_generations:
                    continue
                self._detach_write_locked(entry)
                entry.logical_references = 0
                entry.released = True

    def abandon_outputs(self, references: tuple[ProductRef, ...]) -> None:
        with self._lock:
            for reference in references:
                key = _reference_key(reference)
                entry = self._entries.get(key)
                if entry is None or entry.reference != reference:
                    continue
                entry.logical_references = 0
                entry.released = True
            self._reclaim_ready_locked()

    def abandon_writes(self, writes: tuple[DeviceProductWrite, ...]) -> None:
        with self._lock:
            for write in writes:
                try:
                    entry = self._require_write_locked(write)
                except WorkerError:
                    continue
                entry.logical_references = 0
                entry.released = True
            self._reclaim_ready_locked()

    def reclaim_ready(self) -> int:
        with self._lock:
            return self._reclaim_ready_locked()

    def physical_generation(self, reference: ProductRef) -> int:
        with self._lock:
            return self._require_locked(reference).physical_generation

    def validate_completion(self, references: tuple[ProductRef, ...]) -> None:
        with self._lock:
            for reference in references:
                if reference.storage_class is not StorageClass.DEVICE_TENSOR:
                    continue
                entry = self._require_locked(reference)
                if not entry.producer_recorded:
                    raise _invariant("completion packing found an unpublished device product")

    def validate_writes(self, writes: tuple[DeviceProductWrite, ...]) -> None:
        with self._lock:
            for write in writes:
                entry = self._require_write_locked(write)
                if not entry.producer_recorded:
                    raise _invariant("completion packing found an unpublished device product")

    def _require_locked(self, reference: ProductRef) -> DeviceProductWrite:
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
        entry = write
        if entry.slot.owner != entry.binding_id:
            raise _invariant("stale device-product physical generation")
        if entry.slot.generation != entry.physical_generation:
            raise _invariant("stale device-product physical generation")
        return entry

    def _require_read_locked(self, read: DeviceProductRead) -> DeviceProductWrite:
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
        if batch._table_token is not self._binding_token:
            raise _invariant("device-product scalar binding belongs to a different table")
        if not batch._valid:
            raise _invariant("stale device-product physical generation")
        if batch._linked:
            return
        for write in batch.writes:
            self._require_write_locked(write)

    def _require_continuation_locked(
        self,
        continuation: DeviceProductContinuationBatch,
    ) -> None:
        if continuation._table_token is not self._binding_token:
            raise _invariant("device continuation belongs to a different product table")
        scalar = continuation.scalar
        if scalar is not None:
            self._require_live_scalar_batch_locked(scalar)
            return
        for write in continuation.writes:
            self._require_write_locked(write)

    def _record_continuation_readers_locked(
        self,
        continuation: DeviceProductContinuationBatch,
        *,
        event: torch.cuda.Event | None,
    ) -> None:
        if continuation._readers_recorded:
            return
        parents = continuation._parents
        if continuation._device.type == "cuda":
            if event is None:
                raise _invariant("CUDA device continuation has no reader completion event")
            if all(parent.reader_events is None for parent in parents):
                for parent in parents:
                    parent.reader_events = event
                self._retain_event_locked(
                    event,
                    continuation._device,
                    len(parents),
                )
            else:
                for parent in parents:
                    self._append_reader_event_locked(
                        parent,
                        event,
                        continuation._device,
                    )
        for parent in parents:
            if parent.inflight_read_batches < 1:
                raise _invariant("device continuation parent pin underflow")
            parent.inflight_read_batches -= 1
        continuation._readers_recorded = True

    def _scalar_batch_locked(
        self,
        writes: tuple[DeviceProductWrite, ...],
    ) -> DeviceProductScalarBatch | None:
        entries = tuple(self._require_write_locked(write) for write in writes)
        if any(entry.producer_recorded for entry in entries):
            raise _invariant("device product was published more than once")
        first_slot = entries[0].slot
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
        slots = self._slots.get(device_name)
        if slots is None:
            slots = [
                _DeviceSlot(index=index, device_name=device_name) for index in range(self.capacity)
            ]
            self._slots[device_name] = slots
        occupied = self._occupied_slots.get(device_name, 0)
        if self.capacity - occupied < count and not reclaimed:
            self._reclaim_ready_locked()
            reclaimed = True
            occupied = self._occupied_slots.get(device_name, 0)
        if self.capacity - occupied < count:
            raise resource_error(
                f"device-product arena for {device_name} has no query-ready free generation"
            )

        cursor = self._allocation_cursor.get(device_name, 0)
        last_start = self.capacity - count
        starts = (
            chain(
                range(cursor, last_start + 1),
                range(0, min(cursor, last_start + 1)),
            )
            if cursor <= last_start
            else range(0, last_start + 1)
        )
        run_start = next(
            (
                start
                for start in starts
                if all(slots[index].owner is None for index in range(start, start + count))
            ),
            None,
        )
        available = (
            slots[run_start : run_start + count]
            if run_start is not None
            else [slot for slot in slots if slot.owner is None][:count]
        )
        if run_start is not None:
            self._allocation_cursor[device_name] = (run_start + count) % self.capacity
        return available, reclaimed

    def _acquire_slot_locked(
        self,
        device: torch.device,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        slot: _DeviceSlot,
    ) -> _DeviceSlot:
        name = str(device)
        if slot.device_name != name or slot.owner is not None:
            raise _invariant("device-product allocation lost its planned physical slot")
        generation = slot.generation + 1
        slot.generation = 1 if generation > _MAX_GENERATION else generation
        try:
            if math.prod(shape) == 1:
                arena_key = (name, dtype)
                arena = self._scalar_arenas.get(arena_key)
                if arena is None:
                    arena = torch.empty((self.capacity,), dtype=dtype, device=device)
                    self._scalar_arenas[arena_key] = arena
                if slot.tensor is None or slot.shape != shape or slot.dtype != dtype:
                    slot.tensor = arena[slot.index : slot.index + 1].reshape(shape)
                    slot.shape = shape
                    slot.dtype = dtype
            elif slot.tensor is None or slot.shape != shape or slot.dtype != dtype:
                slot.tensor = torch.empty(shape, dtype=dtype, device=device)
                slot.shape = shape
                slot.dtype = dtype
        except BaseException:
            self._return_slot_locked(slot)
            raise
        return slot

    def _prepare_homogeneous_slots_locked(
        self,
        device: torch.device,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        slots: list[_DeviceSlot],
    ) -> None:
        device_name = str(device)
        scalar = math.prod(shape) == 1
        arena: torch.Tensor | None = None
        if scalar:
            arena_key = (device_name, dtype)
            arena = self._scalar_arenas.get(arena_key)
            if arena is None:
                arena = torch.empty((self.capacity,), dtype=dtype, device=device)
                self._scalar_arenas[arena_key] = arena
        for slot in slots:
            if slot.device_name != device_name or slot.owner is not None:
                raise _invariant("device-product allocation lost its planned physical slot")
            generation = slot.generation + 1
            slot.generation = 1 if generation > _MAX_GENERATION else generation
            if scalar:
                if slot.tensor is None or slot.shape != shape or slot.dtype != dtype:
                    slot.tensor = cast(torch.Tensor, arena)[slot.index : slot.index + 1].reshape(
                        shape
                    )
                    slot.shape = shape
                    slot.dtype = dtype
            elif slot.tensor is None or slot.shape != shape or slot.dtype != dtype:
                slot.tensor = torch.empty(shape, dtype=dtype, device=device)
                slot.shape = shape
                slot.dtype = dtype

    def _return_slot_locked(self, slot: _DeviceSlot) -> None:
        if slot.owner is not None:
            occupied = self._occupied_slots.get(slot.device_name, 0)
            if occupied < 1:
                raise _invariant("device-product occupancy underflow")
            self._occupied_slots[slot.device_name] = occupied - 1
        slot.owner = None

    def _detach_write_locked(self, entry: DeviceProductWrite) -> None:
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
        if event is None:
            return self._record_event_locked(device)
        return event, self.event_pool.declare_stream(event, device)

    def _record_event_locked(self, device: torch.device) -> tuple[torch.cuda.Event, int]:
        event = self.event_pool.acquire(device)
        return event, self.event_pool.record(event, device)

    def _retain_event_locked(
        self,
        event: torch.cuda.Event,
        device: torch.device,
        count: int = 1,
    ) -> None:
        self.event_pool.retain(event, device, count)

    def _release_event_locked(
        self,
        event: torch.cuda.Event,
        count: int = 1,
    ) -> None:
        self.event_pool.release(event, count)

    def _append_reader_event_locked(
        self,
        entry: DeviceProductWrite,
        event: torch.cuda.Event,
        device: torch.device,
    ) -> None:
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
        reclaimed = 0
        readiness: dict[int, bool] = {}
        released_events: dict[int, tuple[torch.cuda.Event, int]] = {}

        def release_event(event: torch.cuda.Event) -> None:
            identity = id(event)
            current = released_events.get(identity)
            if current is None:
                released_events[identity] = (event, 1)
            elif current[0] is not event:
                raise _invariant("device event identity changed during reclamation")
            else:
                released_events[identity] = (event, current[1] + 1)

        def ready(event: torch.cuda.Event | None) -> bool:
            if event is None:
                return True
            identity = id(event)
            result = readiness.get(identity)
            if result is None:
                result = _event_ready(event)
                readiness[identity] = result
            return result

        def readers_ready(entry: DeviceProductWrite) -> bool:
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
                and entry.inflight_read_batches == 0
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
        for event, count in released_events.values():
            self._release_event_locked(event, count)
        return reclaimed


@dataclass(frozen=True, slots=True)
class VisionFeatureProduct:
    features: torch.Tensor
    grid: torch.Tensor | None
    height: int
    width: int
    source_base64: str | None


@dataclass(frozen=True, slots=True)
class LatentFeatureProduct:
    latent: torch.Tensor
    height: int
    width: int
    source_base64: str | None


@dataclass(frozen=True, slots=True)
class LogitsProduct:
    logits: torch.Tensor
    source_mode: TokenMode
    draft_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.source_mode not in {
            TokenMode.EXTEND,
            TokenMode.DECODE,
            TokenMode.VERIFY,
        }:
            raise ValueError("logits product source mode is invalid")
        if self.source_mode is not TokenMode.VERIFY and self.draft_token_ids:
            raise ValueError("only verify logits may carry draft token ids")


class ImageRange(StrEnum):
    SIGNED_UNIT = "signed_unit"
    UNIT = "unit"


@dataclass(frozen=True, slots=True)
class ImageTensorProduct:
    image: torch.Tensor
    height: int
    width: int
    value_range: ImageRange


@dataclass(frozen=True, slots=True)
class EncodedImageProduct:
    base64: str

    def __post_init__(self) -> None:
        if not self.base64:
            raise ValueError("encoded image product must not be empty")


@dataclass(frozen=True, slots=True)
class FrameCollectionProduct:
    frames: tuple[EncodedImageProduct, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("frame collection must contain at least one frame")


ProductPayload = (
    VisionFeatureProduct
    | LatentFeatureProduct
    | LogitsProduct
    | ImageTensorProduct
    | EncodedImageProduct
    | FrameCollectionProduct
)


@dataclass(frozen=True, slots=True)
class ProductRecord:
    handle: int
    session_id: int
    payload: ProductPayload
    locator: str = ""

    def __post_init__(self) -> None:
        if self.handle < 1:
            raise ValueError("product handle must be positive")
        if not isinstance(
            self.payload,
            (
                VisionFeatureProduct,
                LatentFeatureProduct,
                LogitsProduct,
                ImageTensorProduct,
                EncodedImageProduct,
                FrameCollectionProduct,
            ),
        ):
            raise TypeError("product record payload is not a closed product variant")
        dimensions = (
            (self.payload.height, self.payload.width)
            if isinstance(
                self.payload,
                (VisionFeatureProduct, LatentFeatureProduct, ImageTensorProduct),
            )
            else None
        )
        if dimensions is not None and min(dimensions) < 1:
            raise ValueError("product image geometry must be positive")


class ProductStore:
    """Own session products and scheduler-managed encoder-cache products.

    Encoder feature handles belong to the scheduler's cross-session encoder
    cache once published. They remain resident across session cleanup and are
    reclaimed only through ``release``. Every other product follows its
    originating session's lifetime.
    """

    def __init__(
        self,
        *,
        encoder_cache_budget: int = 0,
        device_product_capacity: int = 1,
    ) -> None:
        self.encoder_cache_budget = int(encoder_cache_budget)
        self.device_events = DeviceEventPool()
        self.device_products = DeviceProductTable(
            capacity=device_product_capacity,
            event_pool=self.device_events,
        )
        self._records: dict[int, ProductRecord] = {}
        self._session_handles: dict[int, set[int]] = {}
        self._revisions: dict[int, int] = {}
        self._next_revision = 1
        self._lock = RLock()

    def get(self, handle: int) -> ProductRecord | None:
        with self._lock:
            return self._records.get(int(handle))

    def require(self, handle: int) -> ProductRecord:
        value = self.get(handle)
        if value is None:
            raise KeyError(f"unknown product handle {handle}")
        return value

    def encoder_output_count(self) -> int:
        """Return committed encoder products in the scheduler's handle unit."""

        with self._lock:
            return sum(
                isinstance(record.payload, (VisionFeatureProduct, LatentFeatureProduct))
                for record in self._records.values()
            )

    def release(self, handles: tuple[int, ...]) -> None:
        self.device_products.release_generations(handles)
        with self._lock:
            for raw in handles:
                handle = int(raw)
                record = self._records.pop(handle, None)
                if record is not None:
                    self._session_handles.get(record.session_id, set()).discard(handle)
                    self._revisions[handle] = self._revision()

    def session_records(self, session_id: int) -> tuple[ProductRecord, ...]:
        with self._lock:
            return tuple(
                self._records[handle]
                for handle in self._session_handles.get(int(session_id), set())
                if handle in self._records
            )

    def rewrite_locators(self, session_ids: set[int], replacements: dict[str, str]) -> None:
        requested = {int(value) for value in session_ids}
        if not replacements:
            return
        with self._lock:
            for handle, record in tuple(self._records.items()):
                replacement = replacements.get(record.locator)
                if record.session_id in requested and replacement is not None:
                    self._records[handle] = replace(record, locator=replacement)

    def drop(self, session_id: int) -> None:
        with self._lock:
            handles = self._session_handles.pop(int(session_id), set())
            retained = frozenset(
                handle
                for handle in handles
                if (record := self._records.get(handle)) is not None
                and isinstance(
                    record.payload,
                    (VisionFeatureProduct, LatentFeatureProduct),
                )
            )
            for handle in handles:
                record = self._records.get(handle)
                if record is not None and isinstance(
                    record.payload,
                    (VisionFeatureProduct, LatentFeatureProduct),
                ):
                    continue
                self._records.pop(handle, None)
                self._revisions[handle] = self._revision()
        self.device_products.drop_session(
            session_id,
            retained_generations=retained,
        )

    def snapshot_records(self, session_ids: set[int]) -> tuple[ProductRecord, ...]:
        requested = {int(value) for value in session_ids}
        with self._lock:
            return tuple(
                replace(record, payload=_snapshot_payload(record.payload))
                for record in self._records.values()
                if record.session_id in requested
            )

    def restore_records(
        self,
        session_ids: set[int],
        records: tuple[ProductRecord, ...],
    ) -> None:
        requested = {int(value) for value in session_ids}
        staged = {int(record.handle): record for record in records}
        if len(staged) != len(records):
            raise ValueError("product snapshot repeats a handle")
        if any(record.session_id not in requested for record in staged.values()):
            raise ValueError("product snapshot contains an undeclared session")
        with self._lock:
            projected = {
                handle: record
                for handle, record in self._records.items()
                if record.session_id not in requested
            }
            if set(projected) & set(staged):
                raise ValueError("product snapshot handle conflicts with another session")
            projected.update(staged)
            used = sum(
                isinstance(record.payload, (VisionFeatureProduct, LatentFeatureProduct))
                for record in projected.values()
            )
            if used > self.encoder_cache_budget:
                raise ValueError(
                    "product snapshot exceeds encoder-output capacity "
                    f"({used}>{self.encoder_cache_budget})"
                )
            replaced = [
                handle for handle, record in self._records.items() if record.session_id in requested
            ]
            self._records = projected
            self._session_handles = {}
            for handle, record in projected.items():
                self._session_handles.setdefault(record.session_id, set()).add(handle)
            for handle in (*replaced, *staged):
                self._revisions[handle] = self._revision()

    def begin_step(self, request_ids: set[int]) -> ProductTxn:
        return ProductTxn(self, frozenset(int(value) for value in request_ids))

    def _revision(self) -> int:
        value = self._next_revision
        self._next_revision += 1
        return value


class ProductTxn:
    def __init__(self, store: ProductStore, session_ids: frozenset[int]) -> None:
        self._store = store
        self._session_ids = session_ids
        self._staged: dict[int, ProductRecord] = {}
        self._bases: dict[int, int] = {}
        self._prior: dict[int, ProductRecord | None] = {}
        self._published: dict[int, int] = {}
        self._lock_held = False
        self._closed = False

    def view(self) -> ProductView:
        self._require_open()
        return ProductView(self)

    def stage(self, record: ProductRecord) -> None:
        self._require_open()
        if record.session_id not in self._session_ids:
            raise ValueError("product belongs to a session outside this step")
        if record.handle not in self._bases:
            with self._store._lock:
                self._bases[record.handle] = self._store._revisions.get(record.handle, 0)
        self._staged[record.handle] = record

    def read(self, handle: int) -> ProductRecord | None:
        self._require_open()
        if int(handle) in self._staged:
            return self._staged[int(handle)]
        return self._store.get(int(handle))

    def prepare(self) -> None:
        self._require_open()
        with self._store._lock:
            self._validate()

    def publish(self) -> None:
        self._require_open()
        self._store._lock.acquire()
        self._lock_held = True
        try:
            self._validate()
            for handle, record in self._staged.items():
                self._prior[handle] = self._store._records.get(handle)
                self._store._records[handle] = record
                self._store._session_handles.setdefault(record.session_id, set()).add(handle)
                revision = self._store._revision()
                self._store._revisions[handle] = revision
                self._published[handle] = revision
        except BaseException:
            self._lock_held = False
            self._store._lock.release()
            raise

    def rollback(self) -> None:
        if self._closed:
            return
        try:
            if self._published:
                with self._store._lock:
                    for handle, revision in self._published.items():
                        if self._store._revisions.get(handle) != revision:
                            raise RuntimeError("published product changed before rollback")
                        record = self._store._records.get(handle)
                        if record is not None:
                            self._store._session_handles.get(record.session_id, set()).discard(
                                handle
                            )
                        prior = self._prior[handle]
                        if prior is None:
                            self._store._records.pop(handle, None)
                        else:
                            self._store._records[handle] = prior
                            self._store._session_handles.setdefault(prior.session_id, set()).add(
                                handle
                            )
                        self._store._revisions[handle] = self._store._revision()
        finally:
            self._release()

    def finalize(self) -> None:
        self._require_open()
        self._release()

    def _validate(self) -> None:
        for handle, revision in self._bases.items():
            if self._store._revisions.get(handle, 0) != revision:
                raise RuntimeError("product changed during step execution")
        projected = dict(self._store._records)
        projected.update(self._staged)
        used = sum(
            isinstance(record.payload, (VisionFeatureProduct, LatentFeatureProduct))
            for record in projected.values()
        )
        if used > self._store.encoder_cache_budget:
            raise RuntimeError(
                "encoder-output residency exceeds capacity "
                f"({used}>{self._store.encoder_cache_budget})"
            )

    def _release(self) -> None:
        if self._lock_held:
            self._lock_held = False
            self._store._lock.release()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("product transaction is closed")


class ProductView:
    def __init__(self, transaction: ProductTxn) -> None:
        self._transaction = transaction

    def put(self, record: ProductRecord) -> None:
        self._transaction.stage(record)

    def get(self, handle: int) -> ProductRecord | None:
        return self._transaction.read(handle)

    def require(self, handle: int) -> ProductRecord:
        record = self.get(handle)
        if record is None:
            raise KeyError(f"unknown product handle {handle}")
        return record


def _snapshot_payload(payload: ProductPayload) -> ProductPayload:
    if isinstance(payload, VisionFeatureProduct):
        return replace(
            payload,
            features=payload.features.detach().cpu().contiguous(),
            grid=(None if payload.grid is None else payload.grid.detach().cpu().contiguous()),
        )
    if isinstance(payload, LatentFeatureProduct):
        return replace(payload, latent=payload.latent.detach().cpu().contiguous())
    if isinstance(payload, LogitsProduct):
        return replace(payload, logits=payload.logits.detach().cpu().contiguous())
    if isinstance(payload, ImageTensorProduct):
        return replace(payload, image=payload.image.detach().cpu().contiguous())
    return payload


__all__ = [
    "DeviceProductRead",
    "DeviceProductTable",
    "EncodedImageProduct",
    "FrameCollectionProduct",
    "ImageRange",
    "ImageTensorProduct",
    "LatentFeatureProduct",
    "LogitsProduct",
    "ProductRecord",
    "ProductPayload",
    "ProductStore",
    "ProductTxn",
    "ProductView",
    "VisionFeatureProduct",
]
