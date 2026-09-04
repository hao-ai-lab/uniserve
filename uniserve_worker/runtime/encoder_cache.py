"""Bounded immutable encoder features keyed by exact semantic products."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Final

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
_DTYPES: Final[dict[DType, torch.dtype]] = {
    DType.F16: torch.float16,
    DType.BF16: torch.bfloat16,
    DType.F32: torch.float32,
}
_ReferenceKey = tuple[int, int, int, int, int]
_OperationKey = tuple[RequestKey, int]


def _invariant(message: str) -> WorkerError:
    """Construct a classified invariant error for encoder-cache misuse."""

    return WorkerError(code=WorkerErrorCode.INVARIANT_VIOLATION, message=message, fatal=True)


def _reference_key(reference: ProductRef) -> _ReferenceKey:
    """Build the generation-tagged lookup key for an encoder feature."""

    key = reference.request_key
    return (
        int(key.authority_id),
        int(key.request_id),
        int(key.epoch),
        int(reference.producer_op_id),
        int(reference.output_index),
    )


def _shape(reference: ProductRef) -> tuple[int, ...]:
    """Resolve an encoder feature's bounded dimensions to a concrete shape."""

    dims = tuple(
        dim.extent if isinstance(dim, StaticDim) else dim.bound
        for dim in reference.shape_bound.dims
    )
    return dims or (1,)


@dataclass(frozen=True, slots=True)
class EncoderMetadata:
    """Describes an encoder feature’s generation, media geometry, payload kind, and tensor geometry."""

    height: int
    width: int

    def __post_init__(self) -> None:
        """Validate feature generation, payload kind, media geometry, and tensor shape."""

        if min(self.height, self.width) < 1:
            raise ValueError("encoder feature geometry must be positive")


@dataclass(slots=True)
class _EncoderSlot:
    """Tracks the device, generation, and owner of one encoder-cache slot."""

    index: int
    device: torch.device
    generation: int = 0
    owner: int = 0


@dataclass(slots=True)
class EncoderWrite:
    """Owns a writable encoder-cache slot and its producer/reader synchronization events."""

    reference: ProductRef
    slot: _EncoderSlot
    physical_generation: int
    binding_id: int
    buffer_binding: PersistentBufferBinding
    producer_event: torch.cuda.Event | None = None
    producer_stream: int | None = None
    reader_events: list[torch.cuda.Event] = field(default_factory=list)
    tensor: torch.Tensor | None = None
    metadata: EncoderMetadata | None = None
    published: bool = False
    released: bool = False


@dataclass(slots=True)
class EncoderRead:
    """Retains a published encoder feature until its consumer stream is recorded."""

    tensor: torch.Tensor
    metadata: EncoderMetadata
    consumer_op_id: int
    _write: EncoderWrite = field(repr=False, compare=False)
    _recorded: bool = field(default=False, repr=False, compare=False)

    @property
    def reference(self) -> ProductRef:
        """Expose the immutable logical encoder-product identity guarded by this read lease."""

        return self._write.reference


class EncoderCache:
    """Own immutable encoded features behind fixed entry and byte capacities."""

    def __init__(
        self,
        *,
        entry_capacity: int,
        max_entry_bytes: int,
        devices: tuple[torch.device | str, ...],
        persistent_buffers: PersistentBuffers,
        event_pool: DeviceEventPool | None = None,
    ) -> None:
        """Configure bounded immutable entries across the declared execution devices."""

        self.entry_capacity = int(entry_capacity)
        self.max_entry_bytes = int(max_entry_bytes)
        if self.entry_capacity < 0 or self.max_entry_bytes < 1:
            raise ValueError("encoder-cache geometry is invalid")
        normalized: list[torch.device] = []
        for raw in devices:
            device = canonical_device(raw)
            if device not in normalized:
                normalized.append(device)
        if not normalized:
            normalized.append(torch.device("cpu"))
        self.devices = tuple(normalized)
        self.byte_capacity = persistent_buffers.byte_capacity
        self.persistent_buffers = persistent_buffers
        self.event_pool = DeviceEventPool() if event_pool is None else event_pool
        self._slots = {
            str(device): tuple(
                _EncoderSlot(index=index, device=device) for index in range(self.entry_capacity)
            )
            for device in self.devices
        }
        self._free = {str(device): deque(range(self.entry_capacity)) for device in self.devices}
        self._entries: dict[_ReferenceKey, EncoderWrite] = {}
        self._candidates: dict[int, EncoderWrite] = {}
        self._operations: dict[_OperationKey, list[EncoderWrite]] = {}
        self._next_binding_id = 1
        self._lock = RLock()

    @property
    def resident_entries(self) -> int:
        """Count published encoder tensors that still retain cache ownership."""

        with self._lock:
            return sum(not entry.released for entry in self._entries.values())

    @property
    def resident_bytes(self) -> int:
        """Return bytes held by currently resident encoder tensors."""

        with self._lock:
            return sum(
                0
                if entry.tensor is None or entry.released
                else entry.tensor.numel() * entry.tensor.element_size()
                for entry in self._entries.values()
            )

    def bind_outputs(
        self,
        bindings: tuple[tuple[ProductRef, torch.device | str], ...],
        *,
        buffer_placements: Mapping[BufferId, BufferPlacement],
    ) -> tuple[EncoderWrite, ...]:
        """Reserve shape-compatible encoder slots for operation outputs as one atomic batch."""

        if not bindings:
            return ()
        with self._lock:
            self._reclaim_ready_locked()
            if len(self._entries) + len(self._candidates) + len(bindings) > self.entry_capacity:
                raise resource_error("encoder cache has no query-ready entry capacity")
            keys = tuple(_reference_key(reference) for reference, _device in bindings)
            if len(set(keys)) != len(keys):
                raise invalid_descriptor("encoder cache registration repeats a product identity")
            validated: list[tuple[ProductRef, torch.device, BufferPlacement]] = []
            requested_by_device: dict[str, int] = {}
            for (reference, raw_device), key in zip(bindings, keys, strict=True):
                if reference.kind not in {
                    ProductKind.VISION_FEATURE,
                    ProductKind.LATENT_FEATURE,
                }:
                    raise invalid_descriptor("encoder cache received a non-feature product")
                if reference.storage_class is not StorageClass.LATENT_ARENA:
                    raise invalid_descriptor("encoder feature has an incompatible storage class")
                dtype = _DTYPES.get(reference.dtype)
                if dtype is None:
                    raise invalid_descriptor("encoder feature dtype is unsupported")
                if int(reference.max_bytes) > self.max_entry_bytes:
                    raise resource_error("encoder feature exceeds the fixed entry byte capacity")
                placement = buffer_placements.get(reference.buffer_id)
                if placement is None:
                    raise invalid_descriptor("encoder feature has no buffer placement")
                existing = self._entries.get(key)
                if existing is not None:
                    if existing.reference != reference:
                        raise invalid_descriptor("stale encoder feature generation")
                    raise invalid_descriptor("encoder feature is already registered")
                if any(
                    _reference_key(candidate.reference) == key
                    for candidate in self._candidates.values()
                ):
                    raise invalid_descriptor("encoder feature already has a candidate")
                device = canonical_device(raw_device)
                free = self._free.get(str(device))
                if free is None:
                    raise invalid_descriptor("encoder feature names an undeclared device")
                requested_by_device[str(device)] = requested_by_device.get(str(device), 0) + 1
                if requested_by_device[str(device)] > len(free):
                    raise resource_error("encoder cache has no query-ready device slot")
                validated.append((reference, device, placement))
            prepared: list[
                tuple[ProductRef, torch.device, BufferPlacement, _EncoderSlot]
            ] = []
            try:
                for reference, device, placement in validated:
                    slot = self._slots[str(device)][self._free[str(device)].popleft()]
                    prepared.append((reference, device, placement, slot))
            except BaseException:
                for _reference, device, _placement, slot in reversed(prepared):
                    self._free[str(device)].appendleft(slot.index)
                raise
            writes: list[EncoderWrite] = []
            try:
                for (reference, device, placement, slot), key in zip(
                    prepared, keys, strict=True
                ):
                    generation = slot.generation + 1
                    slot.generation = 1 if generation > _MAX_GENERATION else generation
                    binding_id = self._next_binding_id
                    self._next_binding_id += 1
                    buffer_binding = self.persistent_buffers.bind(
                        reference,
                        placement,
                        device=device,
                        dtype=_DTYPES[reference.dtype],
                        shape=_shape(reference),
                    )
                    slot.owner = binding_id
                    write = EncoderWrite(
                        reference=reference,
                        slot=slot,
                        physical_generation=slot.generation,
                        binding_id=binding_id,
                        buffer_binding=buffer_binding,
                    )
                    self._candidates[write.binding_id] = write
                    writes.append(write)
            except BaseException:
                for write in writes:
                    self._candidates.pop(write.binding_id, None)
                    self.persistent_buffers.release(write.buffer_binding)
                    write.slot.owner = 0
                    self._free[str(write.slot.device)].appendleft(write.slot.index)
                for _reference, device, _placement, slot in reversed(
                    prepared[len(writes) :]
                ):
                    self._free[str(device)].appendleft(slot.index)
                raise
            return tuple(writes)

    def publish(
        self,
        write: EncoderWrite,
        value: torch.Tensor,
        metadata: EncoderMetadata,
    ) -> torch.Tensor:
        """Commit an immutable encoder tensor and metadata into its reserved slot."""

        with self._lock:
            entry = self._require_write_locked(write)
            if entry.published:
                raise _invariant("encoder feature was published more than once")
            dtype = _DTYPES[entry.reference.dtype]
            shape = _shape(entry.reference)
            target = entry.buffer_binding.tensor
            if target.dtype != dtype or tuple(target.shape) != shape:
                raise _invariant("encoder feature buffer has incompatible physical geometry")
            flat = value.detach().reshape(-1)
            if int(flat.numel()) > int(target.numel()):
                raise _invariant("encoder feature exceeds its registered shape bound")
            resident = target.reshape(-1)[: int(flat.numel())].reshape(value.shape)
            resident.copy_(
                flat.reshape(value.shape).to(dtype=dtype), non_blocking=value.device.type == "cuda"
            )
            if resident.device.type == "cuda":
                event = self.event_pool.acquire(resident.device)
                entry.producer_stream = self.event_pool.record(event, resident.device)
                self.event_pool.retain(event, resident.device)
                entry.producer_event = event
            entry.tensor = resident
            entry.metadata = metadata
            entry.published = True
            return resident

    def consume(
        self,
        reference: ProductRef,
        *,
        consumer_op_id: int,
        device: torch.device | str | None = None,
    ) -> EncoderRead:
        """Acquire a generation-safe encoder feature read on the consumer device."""

        with self._lock:
            entry = self._require_locked(reference)
            if entry.released:
                raise invalid_descriptor("encoder feature was consumed after release")
            if not entry.published or entry.tensor is None or entry.metadata is None:
                raise invalid_descriptor("encoder feature was consumed before publication")
            target = entry.tensor.device if device is None else canonical_device(device)
            if target != entry.tensor.device:
                raise invalid_descriptor("encoder feature consumer names a different device")
            if target.type == "cuda":
                event = entry.producer_event
                if event is None:
                    raise _invariant("CUDA encoder feature has no producer event")
                stream = torch.cuda.current_stream(target)
                if int(stream.cuda_stream) != entry.producer_stream:
                    stream.wait_event(event)
            return EncoderRead(
                tensor=entry.tensor,
                metadata=entry.metadata,
                consumer_op_id=int(consumer_op_id),
                _write=entry,
            )

    def consume_candidate(
        self,
        write: EncoderWrite,
        *,
        consumer_op_id: int,
        device: torch.device | str | None = None,
    ) -> EncoderRead:
        """Read one unpublished feature inside its consuming lane."""

        with self._lock:
            entry = self._require_write_locked(write)
            if self._candidates.get(entry.binding_id) is not entry:
                raise _invariant("encoder feature candidate is not live")
            if not entry.published or entry.tensor is None or entry.metadata is None:
                raise invalid_descriptor("encoder feature was consumed before producer readiness")
            target = entry.tensor.device if device is None else canonical_device(device)
            if target != entry.tensor.device:
                raise invalid_descriptor("encoder feature consumer names a different device")
            if target.type == "cuda":
                event = entry.producer_event
                if event is None:
                    raise _invariant("CUDA encoder feature has no producer event")
                stream = torch.cuda.current_stream(target)
                if int(stream.cuda_stream) != entry.producer_stream:
                    stream.wait_event(event)
            return EncoderRead(
                tensor=entry.tensor,
                metadata=entry.metadata,
                consumer_op_id=int(consumer_op_id),
                _write=entry,
            )

    def record_readers(self, reads: tuple[EncoderRead, ...]) -> None:
        """Record consumer streams and release their retained encoder reads."""

        if not reads:
            return
        with self._lock:
            grouped: dict[str, tuple[torch.device, list[tuple[EncoderRead, EncoderWrite]]]] = {}
            for read in reads:
                if read._recorded:
                    continue
                entry = self._require_read_locked(read)
                target = read.tensor.device
                grouped.setdefault(str(target), (target, []))[1].append((read, entry))
            for target, entries in grouped.values():
                if target.type == "cuda":
                    event = self.event_pool.acquire(target)
                    self.event_pool.record(event, target)
                    self.event_pool.retain(event, target, len(entries))
                    for _read, entry in entries:
                        entry.reader_events.append(event)
                for read, _entry in entries:
                    read._recorded = True

    def validate_writes(self, writes: tuple[EncoderWrite, ...]) -> None:
        """Verify that encoder writes still refer to active unpublished reservations."""

        with self._lock:
            for write in writes:
                entry = self._require_write_locked(write)
                if self._candidates.get(entry.binding_id) is not entry:
                    raise _invariant("encoder feature candidate is not live")
                if not entry.published:
                    raise _invariant("completion found an unpublished encoder feature")

    def commit_writes(self, writes: tuple[EncoderWrite, ...]) -> None:
        """Make validated candidate feature generations addressable."""

        if not writes:
            return
        with self._lock:
            entries = tuple(self._require_write_locked(write) for write in writes)
            keys = tuple(_reference_key(entry.reference) for entry in entries)
            if len(set(keys)) != len(keys):
                raise _invariant("encoder publication repeats a product identity")
            for key, entry in zip(keys, entries, strict=True):
                if self._candidates.get(entry.binding_id) is not entry:
                    raise _invariant("encoder feature candidate is not live")
                if not entry.published:
                    raise _invariant("encoder feature candidate has no producer readiness")
                if key in self._entries:
                    raise _invariant("encoder publication identity is already resident")
            for key, entry in zip(keys, entries, strict=True):
                self._entries[key] = entry
                self._operations.setdefault(
                    (entry.reference.request_key, int(entry.reference.producer_op_id)), []
                ).append(entry)
                self._candidates.pop(entry.binding_id)

    def release_generations(self, generations: Iterable[int]) -> None:
        """Release resident encoder products belonging to selected generations."""

        selected = {int(value) for value in generations}
        if not selected:
            return
        with self._lock:
            for entry in self._entries.values():
                if int(entry.reference.generation) in selected:
                    entry.released = True
                    self._detach_locked(entry)
            self._reclaim_ready_locked()

    def drop_request(self, request_id: int) -> None:
        """Release every resident encoder feature owned by one request identifier."""

        target = int(request_id)
        with self._lock:
            for entry in self._entries.values():
                if int(entry.reference.request_key.request_id) == target:
                    entry.released = True
                    self._detach_locked(entry)
            self._reclaim_ready_locked()

    def release_operations(self, releases: Iterable[tuple[RequestKey, int]]) -> None:
        """Release encoder products associated with completed operation identities."""

        with self._lock:
            for request_key, raw_op_id in releases:
                for entry in self._operations.pop((request_key, int(raw_op_id)), ()):
                    entry.released = True
            self._reclaim_ready_locked()

    def abandon_writes(self, writes: tuple[EncoderWrite, ...]) -> None:
        """Return unpublished encoder reservations to the free pool."""

        with self._lock:
            for write in writes:
                try:
                    entry = self._require_write_locked(write)
                except WorkerError:
                    continue
                entry.released = True
                self._detach_locked(entry)
            self._reclaim_ready_locked()

    def close(self) -> None:
        """Release all encoder slots, resident tensors, and synchronization events."""

        with self._lock:
            self._entries.clear()
            self._candidates.clear()
            self._operations.clear()
            self._free.clear()
            self._slots.clear()

    def _require_locked(self, reference: ProductRef) -> EncoderWrite:
        """Resolve a live generation-tagged encoder entry."""

        entry = self._entries.get(_reference_key(reference))
        if entry is None:
            raise invalid_descriptor("unknown encoder feature")
        if entry.reference != reference:
            raise invalid_descriptor("stale encoder feature generation")
        return self._require_write_locked(entry)

    @staticmethod
    def _require_write_locked(write: EncoderWrite) -> EncoderWrite:
        """Validate an encoder write handle and return its live entry."""

        if (
            write.slot.owner != write.binding_id
            or write.slot.generation != write.physical_generation
        ):
            raise _invariant("stale encoder-cache physical generation")
        return write

    @staticmethod
    def _require_read_locked(read: EncoderRead) -> EncoderWrite:
        """Validate an encoder read handle and return its live write entry."""

        return EncoderCache._require_write_locked(read._write)

    def _detach_locked(self, entry: EncoderWrite) -> None:
        """Remove one encoder entry and release its physical storage and events."""

        key = (entry.reference.request_key, int(entry.reference.producer_op_id))
        entries = self._operations.get(key)
        if entries is None:
            return
        remaining = [candidate for candidate in entries if candidate is not entry]
        if remaining:
            self._operations[key] = remaining
        else:
            self._operations.pop(key, None)

    def _reclaim_ready_locked(self) -> int:
        """Reclaim released feature entries after producer and reader events become ready."""

        reclaimed = 0
        event_releases: dict[int, tuple[torch.cuda.Event, int]] = {}

        def ready(event: torch.cuda.Event | None) -> bool:
            """Treat absent synchronization as ready and query recorded CUDA events."""

            return event is None or bool(event.query())

        def release(event: torch.cuda.Event) -> None:
            """Accumulate pooled-event references for release after cache reclamation."""

            existing = event_releases.get(id(event))
            event_releases[id(event)] = (event, 1 if existing is None else existing[1] + 1)

        for key, entry in tuple(self._entries.items()):
            if (
                not entry.released
                or not ready(entry.producer_event)
                or not all(ready(event) for event in entry.reader_events)
            ):
                continue
            self._entries.pop(key)
            self._detach_locked(entry)
            if entry.producer_event is not None:
                release(entry.producer_event)
            for event in entry.reader_events:
                release(event)
            entry.slot.owner = 0
            self.persistent_buffers.release(entry.buffer_binding)
            self._free[str(entry.slot.device)].append(entry.slot.index)
            reclaimed += 1
        for binding_id, entry in tuple(self._candidates.items()):
            if (
                not entry.released
                or not ready(entry.producer_event)
                or not all(ready(event) for event in entry.reader_events)
            ):
                continue
            self._candidates.pop(binding_id)
            if entry.producer_event is not None:
                release(entry.producer_event)
            for event in entry.reader_events:
                release(event)
            entry.slot.owner = 0
            self.persistent_buffers.release(entry.buffer_binding)
            self._free[str(entry.slot.device)].append(entry.slot.index)
            reclaimed += 1
        for event, count in event_releases.values():
            self.event_pool.release(event, count)
        return reclaimed


__all__ = [
    "EncoderCache",
    "EncoderMetadata",
    "EncoderRead",
    "EncoderWrite",
]
