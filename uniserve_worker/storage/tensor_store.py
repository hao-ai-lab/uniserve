"""Bounded immutable device values with generation-safe stream lifetimes.

`TensorStore` is the worker's catalog of device products: tensors that one
call writes and later calls, other ranks, or host work read. Execution code
in ``uniserve_worker.execution`` reserves outputs while preparing a batch,
publishes values after the producing call runs, and commits call outputs in
`commit_batch`. `Executor` releases products by producer call once the
consuming batch has acquired its reads, by buffer when buffers are freed, and
by request when requests close.

Each product has a logical identity (engine, request, epoch, producer call,
output index) plus a logical generation carried by its `TensorRef`, and one
live `TensorRecord` that binds that identity to physical storage. Storage
comes from one of three places:

- Persistent products borrow `BufferPool` ranges through the scheduler's
  `BufferAllocation`; this store bounds their count per device.
- Encoder features also borrow `BufferPool` ranges and are bounded by a fixed
  per-entry byte limit. How many stay resident is the scheduler's encoder
  cache policy: it retains up to ``WorkerInfo.encoder_cache_entries``
  features past their requests, places each new feature in its buffer pool
  before it evicts a retained one, and frees evicted features by buffer.
- Request-relay products are single scalars in flat arenas owned here, one
  element per (request slot, lane, dtype, field). A slot's physical
  generation changes on every rebinding so stale handles fail validation.

A record moves through reserve, producer publication, commit (only committed
records are consumable by reference), logical release, and physical
retirement. Retirement never synchronizes a device stream: a released record
is reclaimed only once it has no read leases, its transfer tickets have
retired, its transport publications have finished successfully, and its
producer and reader CUDA events query as complete.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import RLock
from typing import Final, cast

import torch

from uniserve import _slices
from uniserve.runtime import EventPool
from uniserve.runtime.device import canonical_device
from uniserve_worker.errors import (
    WorkerError,
    WorkerErrorCode,
    invalid_descriptor,
    resource_error,
)
from uniserve_worker.protocol.batch import BufferAllocation
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.tensor import DType, StaticDim, TensorRef
from uniserve_worker.protocol.transfer import TensorTransfer, WorkerEndpoint
from uniserve_worker.storage.buffer_pool import BufferBinding, BufferPool
from uniserve_worker.transport.exports import ExportLocations, release_exports
from uniserve_worker.transport.fetch import fetch_tensor
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.ticket import TransferTicket

# Relay slot physical generations are 32-bit tags that wrap back to one; zero
# is the initial value of a slot that has never been bound.
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
_DEVICE_TORCH_DTYPES: Final[tuple[torch.dtype, ...]] = tuple(
    dict.fromkeys(_DEVICE_DTYPES.values())
)
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
    """Return the concrete tensor storage used for one product dtype.

    Returns:
        The torch dtype name without its ``torch.`` prefix (for example
        ``"bfloat16"``) and the element size in bytes.
    """
    return _DTYPE_STORAGE[DType(dtype)]


def device_product_capacity_bytes(
    slot_capacity: int,
    device_count: int,
    *,
    max_value_bytes: int,
) -> int:
    """Return the fixed backing bound for one ``TensorStore`` owner.

    The bound is ``slot_capacity * device_count * (scalars + max_value_bytes)``
    where ``scalars`` sums the element size of every supported product dtype.
    Request-relay arena bytes are not included;
    ``uniserve_worker.bootstrap.capacity`` adds them separately.

    Raises:
        ValueError: If any dimension is less than one.
    """
    slots = int(slot_capacity)
    devices = int(device_count)
    value_bytes = int(max_value_bytes)
    if min(slots, devices, value_bytes) < 1:
        raise ValueError("device-product dimensions must be positive")
    scalar_bytes = slots * devices * sum(dict(_DTYPE_STORAGE.values()).values())
    return scalar_bytes + slots * devices * value_bytes


def _invariant(message: str) -> WorkerError:
    """Construct a classified invariant error for device-product misuse."""
    return WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


# Logical product identity:
# (engine, request, epoch, producer call, output index).
# The logical generation is not part of the key; `_require_locked` compares
# the full `TensorRef` of the live record to reject stale generations.
_ReferenceKey = tuple[int, int, int, CallId, int]
_CallKey = tuple[RequestKey, CallId]
_SlotStorageKey = tuple[str, tuple[int, ...], torch.dtype]


def _reference_key(reference: TensorRef) -> _ReferenceKey:
    """Build the logical identity of a product, excluding its generation."""
    key = reference.request_key
    return (
        int(key.engine_id),
        int(key.request_id),
        int(key.request_epoch),
        reference.producer_call_id,
        int(reference.output_index),
    )


def _device_dtype(dtype: DType) -> torch.dtype:
    """Resolve a product dtype descriptor to its torch dtype."""
    return _DEVICE_DTYPES[dtype]


def _device_shape(reference: TensorRef) -> tuple[int, ...]:
    """Resolve a product's bounded tensor dimensions to a concrete shape.

    Static dimensions use their extent and dynamic dimensions their upper
    bound. A zero-dimensional product is stored as shape ``(1,)``.
    """
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
    """One scalar element of a request-relay arena and its current binding.

    Slots are created lazily by `TensorStore._relay_slot_locked` and persist,
    at a fixed arena address, until `TensorStore.close`.

    Attributes:
        index: Flat element index ``request_slot * relay_depth + lane`` in
            the arena shared by every slot of the same device, dtype, and
            field.
        device_name: Canonical device string of the arena.
        generation: Physical generation, bumped on every binding and wrapped
            at ``_MAX_GENERATION``; a `TensorRecord` whose
            ``physical_generation`` differs is stale.
        owner: ``binding_id`` of the record currently bound to this slot, or
            ``None`` while free.
        tensor: One-element view into the arena.
        shape: Always ``(1,)``.
        dtype: Element dtype of the arena.
        relay_lane: ``(device, request slot, request key, call, lane)`` of the
            call associated with this slot. It can outlive ``owner`` and is
            cleared once the call no longer holds its lane.
    """

    index: int
    device_name: str
    generation: int = 0
    owner: int | None = None
    tensor: torch.Tensor | None = None
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    relay_lane: tuple[str, int, RequestKey, CallId, int] | None = None


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Spatial dimensions and numerical value range of an immutable image.

    Zero height and width mean the dimensions are undeclared. ``value_range``
    is the numerical interval of the pixel values, such as ``(-1.0, 1.0)``.
    """

    height: int = 0
    width: int = 0
    value_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Require complete, nonnegative image dimensions."""
        if self.height < 0 or self.width < 0:
            raise ValueError(
                "device-product image dimensions must be non-negative"
            )
        if (self.height == 0) != (self.width == 0):
            raise ValueError("device-product image dimensions must be complete")


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Spatial dimensions of an immutable floating-point encoder feature."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if min(self.height, self.width) < 1:
            raise ValueError("encoder feature dimensions must be positive")


@dataclass(slots=True)
class TensorRecord:
    """One store-issued binding of a logical product to physical storage.

    Records are issued by `TensorStore` and act as handles. Store methods that
    accept a record revalidate it with ``_require_write_locked``, so a record
    that has been retired, or whose relay slot was rebound, raises an
    invariant error instead of touching reused storage; `abandon_writes`
    skips such records instead.

    ``binding_id`` is unique within the store and keys its record table.
    ``physical_generation`` is the `BufferBinding.binding_id` for
    persistent storage or the `RelaySlot.generation` for relay storage.
    ``tensor`` has ``shape``. For a shard, ``shape`` is that of ``region``
    within ``logical_shape`` until `TensorStore.complete_import` expands the
    record to the full tensor.
    """

    reference: TensorRef
    tensor: torch.Tensor
    device_name: str
    shape: tuple[int, ...]
    physical_generation: int
    binding_id: int
    buffer_binding: BufferBinding | None = None
    relay_slot: RelaySlot | None = None
    feature: bool = False

    # Lifecycle: reserved, published by its producer (`producer_recorded`),
    # committed for logical consumption, logically released, then physically
    # retired. Release may precede retirement.
    retired: bool = False
    # Only committed records resolve through `TensorStore._require_locked`,
    # which every by-reference read uses.
    committed: bool = False

    region: tuple[slice, ...] | None = None
    logical_shape: tuple[int, ...] | None = None
    # Transfer tickets that write into this storage and transport
    # publications that expose it; the record is not reclaimed until every
    # ticket has retired and every publication has finished successfully.
    transfers: tuple[TransferTicket, ...] = ()
    publications: tuple[Future[None], ...] = ()

    # Stream-safety state: reader leases plus producer/reader events order
    # reclamation without ever synchronizing a device stream.
    readers: int = 0
    producer_event: torch.cuda.Event | None = None
    producer_stream: int | None = None
    producer_recorded: bool = False
    # A deferred write is produced by host work after its call is committed;
    # it is published and committed when that work completes.
    deferred: bool = False
    # One event or a deduplicated list; each attached event holds one
    # `EventPool` reference for this record until reclamation.
    reader_events: torch.cuda.Event | list[torch.cuda.Event] | None = None
    released: bool = False
    # Whether the record is listed in the store's per-call release index.
    _indexed: bool = False

    # Extent actually published by the producer, bounded by the reserved shape.
    actual_extent: int = 0
    actual_shape: tuple[int, ...] = ()
    metadata: ImageMetadata | FeatureMetadata | None = None


@dataclass(slots=True)
class TensorRead:
    """One generation-validated device read lease on a `TensorRecord`.

    The lease keeps the record's storage from being reclaimed until
    `TensorStore.complete_reads` fences it on the consuming stream. ``tensor``
    is the published extent, or the import destination for an imported read.
    ``consumer_call_id`` is ``None`` for imports.
    """

    tensor: torch.Tensor
    consumer_call_id: CallId | None
    _write: TensorRecord = field(repr=False, compare=False)
    region: tuple[slice, ...] | None = None
    metadata: ImageMetadata | FeatureMetadata | None = None
    imported: TensorImport | None = field(
        default=None, repr=False, compare=False
    )
    _recorded: bool = field(default=False, repr=False, compare=False)


@dataclass(slots=True)
class TensorImport:
    """One shared fill of a product's missing regions from transfer sources.

    Concurrent imports of the same product share one instance; ``users``
    counts the `TensorRead` leases holding it, and the last completed read
    drops it. ``committed`` becomes true once the destination holds complete
    coverage that the store has adopted, or at creation when the product was
    already fully resident.
    """

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

    Reclamation is opportunistic: it runs when outputs are bound, when
    buffers or requests are released, from transfer-ticket retirement
    callbacks, and from `retirement_ready` and `abandon_writes`. One
    reentrant lock guards all tables; the retirement callbacks also take it.

    Three bounds are independent: ``capacity`` limits resident (not yet
    reclaimed) generic persistent products per device, ``max_entry_bytes``
    limits one encoder feature, and ``byte_capacity`` limits relay-arena
    bytes, the only storage this store allocates itself. Encoder features
    have no count bound here: the scheduler's buffer-pool placements, which
    `BufferPool.bind` validates, bound their storage.
    """

    def __init__(
        self,
        *,
        capacity: int = 0,
        byte_capacity: int | None = None,
        max_entry_bytes: int = 1,
        devices: tuple[torch.device | str, ...] = (),
        request_capacity: int = 0,
        relay_depth: int = 0,
        buffer_pool: BufferPool,
        event_pool: EventPool | None = None,
    ) -> None:
        """Initialize empty product registries and validate every bound.

        Relay arenas are allocated lazily, on first use of each device, dtype,
        and field.

        Args:
            capacity: Resident generic persistent products per device.
            byte_capacity: Bound on relay-arena bytes; defaults to the
                `BufferPool` byte capacity.
            max_entry_bytes: Largest ``max_bytes`` of one encoder feature.
            devices: Devices on which encoder features may be reserved.
            request_capacity: Request slots; relay request slots are
                one-based, ``1..request_capacity``.
            relay_depth: Relay lanes per request slot, each held by one call
                at a time. Zero together with a zero ``request_capacity``
                disables request relays.
            buffer_pool: Borrowed owner of persistent product storage.
            event_pool: Borrowed owner of CUDA events; a private pool is
                created when omitted.

        Raises:
            ValueError: If a capacity or request-relay dimension is negative,
                ``max_entry_bytes`` or the byte capacity is less than one, or
                exactly one of the request-relay dimensions is zero.
        """
        # Validate the independent capacities and the coupled request-relay
        # dimensions before creating any registries.
        self.capacity = int(capacity)
        self.max_entry_bytes = int(max_entry_bytes)
        self.devices = tuple(canonical_device(device) for device in devices)
        if self.capacity < 0 or self.max_entry_bytes < 1:
            raise ValueError("tensor store capacities are invalid")

        self.byte_capacity = int(
            buffer_pool.byte_capacity
            if byte_capacity is None
            else byte_capacity
        )
        if self.byte_capacity < 1:
            raise ValueError("device-product byte capacity must be positive")

        self.request_capacity = int(request_capacity)
        self.relay_depth = int(relay_depth)
        self.buffer_pool = buffer_pool
        if (self.request_capacity == 0) != (self.relay_depth == 0):
            raise ValueError("request-relay dimensions must be complete")
        if self.request_capacity < 0 or self.relay_depth < 0:
            raise ValueError("request-relay dimensions must not be negative")

        # One flat relay arena per (device, dtype, field index), holding one
        # element per (request slot, lane). Arenas are never reallocated, so
        # each relay slot keeps a fixed address until `close`.
        # `_allocated_bytes` counts only these arenas.
        self._allocated_bytes = 0
        self._relay_arenas: dict[
            tuple[str, torch.dtype, int], torch.Tensor
        ] = {}
        # Relay slots keyed by lane (device, request slot, lane), then by
        # (dtype, field index), so lane occupancy checks inspect only this
        # request slot's fields. `_relay_call_lanes` records the lane each
        # call holds until every field of that lane is unowned.
        self._relay_slots: dict[
            tuple[str, int, int], dict[tuple[torch.dtype, int], RelaySlot]
        ] = {}
        self._relay_call_lanes: dict[
            tuple[str, int, RequestKey, CallId], int
        ] = {}

        # Both reserved and committed products retain their logical identity
        # in `_products` until physical retirement. Only committed records are
        # consumable. `_writes` holds the same records by `binding_id`;
        # `_require_write_locked` validates physical handles against it
        # independently of logical publication.
        # `exports` maps each buffer this store's products were published
        # under to its transport registrations (see
        # `uniserve_worker.transport.exports`); execution code fills it.
        self.event_pool = EventPool() if event_pool is None else event_pool
        self._products: dict[_ReferenceKey, TensorRecord] = {}
        self._writes: dict[int, TensorRecord] = {}
        self._imports: dict[_ReferenceKey, TensorImport] = {}
        self._call_writes: dict[
            _CallKey,
            TensorRecord | list[TensorRecord],
        ] = {}
        self.exports: dict[BufferId, ExportLocations] = {}
        self._next_binding_id = 1
        self._lock = RLock()

    def resident_bytes(self, device: torch.device | str) -> int:
        """Return the device bytes of the backing this store allocates.

        Only request-relay arenas count: they are the storage this store
        allocates itself, which ``byte_capacity`` bounds and the arena's
        ``device_product_bytes`` sizes, and callers subtract the result from
        that bound to find what is still to be allocated. Persistent products
        and encoder features view `BufferPool` storage, whose whole grant the
        worker layout counts separately, so they contribute nothing. Relay
        records view these arenas. CUDA arenas from
        ``uniserve_kernels.peer_storage.empty`` report their page-rounded
        size.
        """
        name = str(torch.device(device))
        with self._lock:
            # Count each physical allocation once, keyed by its storage
            # pointer.
            storages = {
                tensor.untyped_storage().data_ptr(): tensor.untyped_storage()
                for (
                    owner,
                    _dtype,
                    _field,
                ), tensor in self._relay_arenas.items()
                if owner == name
            }
            return sum(storage.nbytes() for storage in storages.values())

    def close(self) -> None:
        """Drop every record, import, export, and relay arena this store holds.

        This does not return bindings to `BufferPool` or event references to
        `EventPool`; the worker closes both owners after this store.
        """
        with self._lock:
            self.exports.clear()
            self._imports.clear()
            self._products.clear()
            self._writes.clear()
            self._call_writes.clear()
            self._relay_arenas.clear()
            self._relay_slots.clear()
            self._relay_call_lanes.clear()
            self._allocated_bytes = 0

    @staticmethod
    def _tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        """Compute bytes required for a tensor shape and dtype."""
        return int(math.prod(shape)) * _TORCH_DTYPE_BYTES[dtype]

    def _require_byte_capacity_locked(self, projected: int) -> None:
        """Reject a projected relay-arena total above the byte capacity."""
        if projected > self.byte_capacity:
            raise resource_error(
                f"device-product byte capacity is exhausted "
                f"({projected}>{self.byte_capacity})"
            )

    def bind_outputs(
        self,
        bindings: tuple[tuple[TensorRef, torch.device | str], ...],
        *,
        request_slots: Mapping[RequestKey, int] | None = None,
        buffer_allocations: Mapping[BufferId, BufferAllocation] | None = None,
        regions: Mapping[TensorRef, tuple[slice, ...]] | None = None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None = None,
    ) -> tuple[TensorRecord, ...]:
        """Reserve storage for one group of outputs, all or none.

        A group binds either entirely to persistent storage, when every
        output's ``buffer_id`` has an entry in ``buffer_allocations``, or
        entirely to request-relay scalars addressed by ``request_slots``.
        Reserved records are neither published nor committed.

        Args:
            bindings: Each output reference and the device that produces it.
            request_slots: One-based request slot of each request, used
                by relay outputs.
            buffer_allocations: Scheduler placements of persistent outputs.
            regions: Shard region of an output within its logical shape.
                Validated for every output; only persistent outputs use it.
            shapes: Logical shape overriding the shape bound of an output.
                Validated for every output; only persistent outputs use it.

        Returns:
            One record per binding, in order.

        Raises:
            WorkerError: If shapes, regions, placements, or identities are
                invalid, or a capacity is exhausted. Records reserved by a
                failed call are rolled back.
        """
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
                raise invalid_descriptor(
                    "tensor binding shape disagrees with its logical bounds"
                )
            if region is not None and not _slices.within(region, logical_shape):
                raise invalid_descriptor(
                    "tensor binding region disagrees with its logical bounds"
                )

        # Allocation records select physical ownership. Tensor identity carries
        # no semantic role or duplicate storage-class tag.
        persistent = tuple(
            buffer_allocations is not None
            and reference.buffer_id in buffer_allocations
            for reference, _device in device_bindings
        )
        if any(persistent):
            # The whole group goes to one binder, so persistent and
            # request-relay outputs cannot share a group.
            if not all(persistent):
                raise invalid_descriptor(
                    "persistent-buffer bindings cannot share a generic group"
                )
            assert buffer_allocations is not None
            return self._bind_persistent_outputs(
                device_bindings, buffer_allocations, regions, shapes
            )

        if request_slots is not None:
            return self._bind_relay_outputs(device_bindings, request_slots)
        raise invalid_descriptor(
            "tensor output requires a buffer allocation or request relay slot"
        )

    def reserve_features(
        self,
        bindings: tuple[tuple[TensorRef, torch.device | str], ...],
        *,
        buffer_allocations: Mapping[BufferId, BufferAllocation],
        regions: Mapping[TensorRef, tuple[slice, ...]] | None = None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None = None,
    ) -> tuple[TensorRecord, ...]:
        """Reserve encoder features in persistent storage.

        Features do not count against the per-device product capacity:
        each is bounded by ``max_entry_bytes`` and by its buffer allocation,
        whose placement `BufferPool.bind` validates, so the scheduler's
        encoder cache may keep its full budget of retained features resident
        while the next one is encoded. Features must use a floating-point
        dtype and target a device the store was constructed with. The
        arguments mean the same as in `bind_outputs`, but every output needs
        a buffer allocation and shapes and regions are not checked against
        the shape bounds here.
        """
        return self._bind_persistent_outputs(
            bindings, buffer_allocations, regions, shapes, feature=True
        )

    def _bind_persistent_outputs(
        self,
        bindings: tuple[tuple[TensorRef, torch.device | str], ...],
        allocations: Mapping[BufferId, BufferAllocation],
        regions: Mapping[TensorRef, tuple[slice, ...]] | None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None,
        *,
        feature: bool = False,
    ) -> tuple[TensorRecord, ...]:
        """Bind outputs to the `BufferPool` ranges of their allocations.

        This store rejects repeated or registered identities and missing
        allocations and enforces the generic products' per-device count bound
        (and, for features, the dtype, byte, and device limits);
        `BufferPool.bind` validates and places each allocation, first-fit
        when the pool is compact. On any failure every binding made by this
        call is released and no record remains.
        """
        keys = tuple(
            _reference_key(reference) for reference, _device in bindings
        )
        if len(set(keys)) != len(keys):
            raise invalid_descriptor(
                "tensor registration repeats an output identity"
            )

        with self._lock:
            self._reclaim_ready_locked()

            # Generic products share the per-device slot bound; features and
            # relay lanes are bounded independently and do not consume slots.
            counts: dict[str, int] = {}
            for entry in self._writes.values():
                if not entry.feature and entry.relay_slot is None:
                    counts[entry.device_name] = (
                        counts.get(entry.device_name, 0) + 1
                    )

            # Validate every output and count it against its bound before
            # binding any placement.
            for (reference, raw_device), key in zip(
                bindings, keys, strict=True
            ):
                device = canonical_device(raw_device)
                if key in self._products:
                    raise invalid_descriptor(
                        "persistent output is already registered"
                    )
                if reference.buffer_id not in allocations:
                    raise invalid_descriptor(
                        "persistent output has no buffer allocation"
                    )
                if feature:
                    if reference.dtype not in {
                        DType.F16,
                        DType.BF16,
                        DType.F32,
                    }:
                        raise invalid_descriptor(
                            "encoder feature dtype is unsupported"
                        )
                    if reference.max_bytes > self.max_entry_bytes:
                        raise resource_error(
                            "encoder feature exceeds the fixed entry byte "
                            "capacity: "
                            f"requested={reference.max_bytes}, "
                            f"capacity={self.max_entry_bytes}"
                        )
                    if device not in self.devices:
                        raise invalid_descriptor(
                            "encoder feature names an undeclared device"
                        )
                else:
                    name = str(device)
                    counts[name] = counts.get(name, 0) + 1
                    if counts[name] > self.capacity:
                        raise resource_error(
                            f"device-product arena for {name} has no "
                            "query-ready free generation"
                        )

            writes: list[TensorRecord] = []
            try:
                for (reference, raw_device), key in zip(
                    bindings, keys, strict=True
                ):
                    device = canonical_device(raw_device)
                    dtype = _device_dtype(reference.dtype)
                    logical_shape = (
                        _device_shape(reference)
                        if shapes is None
                        else shapes.get(reference, _device_shape(reference))
                    )
                    region = None if regions is None else regions.get(reference)
                    # A region spanning the full logical shape is no
                    # region at all.
                    if region == tuple(
                        slice(start, start + extent)
                        for start, extent in zip(
                            (0,) * len(logical_shape),
                            logical_shape,
                            strict=True,
                        )
                    ):
                        region = None
                    shape = (
                        logical_shape
                        if region is None
                        else _slices.shape(region)
                    )
                    allocation = allocations[reference.buffer_id]
                    # When the allocation covers the full logical tensor, bind
                    # the whole storage so a later import can expand this shard
                    # in place; otherwise bind only the shard region.
                    full_storage = (
                        region is not None
                        and allocation.bytes
                        >= self._tensor_bytes(logical_shape, dtype)
                    )
                    binding = self.buffer_pool.bind(
                        reference,
                        allocation,
                        device=device,
                        dtype=dtype,
                        shape=logical_shape if full_storage else shape,
                    )
                    tensor = (
                        binding.tensor[region]
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
                # Roll back in reverse so a partial batch leaves no resident
                # record or borrowed storage behind.
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
        """Bind scalar outputs to request-relay slots.

        All outputs of one call on one device and request slot share a lane:
        the lane the call already holds, or the first lane of that request
        slot with no bound field. Within the lane, outputs of the same dtype
        take consecutive field indices from zero in binding order. Each
        binding bumps its slot's physical generation.

        Raises:
            WorkerError: If request relays are disabled; an output is not a
                single element, has no request slot or one outside
                ``1..request_capacity``, has a non-positive logical
                generation, or repeats or reuses a registered identity; the
                request slot has no free lane; the arena byte bound would be
                exceeded; or relay bookkeeping is inconsistent. Records
                reserved by a failed call are rolled back.
        """
        if self.request_capacity < 1 or self.relay_depth < 1:
            raise resource_error("worker has no request-relay arena")

        # Number each output within its (device, request slot, call,
        # dtype) field group before touching shared relay state.
        fields: dict[tuple[str, int, RequestKey, CallId, torch.dtype], int] = {}
        requested_rows = []
        for reference, raw_device in bindings:
            device = canonical_device(raw_device)
            request_slot = int(request_slots.get(reference.request_key, 0))
            dtype = _device_dtype(reference.dtype)
            field_key = (
                str(device),
                request_slot,
                reference.request_key,
                reference.producer_call_id,
                dtype,
            )
            field = fields.get(field_key, 0)
            fields[field_key] = field + 1
            requested_rows.append(
                (reference, device, request_slot, dtype, field)
            )
        requested = tuple(requested_rows)

        # Relay lanes carry one scalar element per field; larger products use
        # persistent buffer storage instead.
        if any(
            slot < 1
            or slot > self.request_capacity
            or math.prod(_device_shape(reference)) != 1
            for reference, _device, slot, _dtype, _field in requested
        ):
            raise invalid_descriptor(
                "request-relay output has an invalid slot or scalar shape"
            )

        keys = tuple(
            _reference_key(reference)
            for reference, _device, _slot, _dtype, _field in requested
        )
        if len(set(keys)) != len(keys):
            raise invalid_descriptor(
                "request-relay registration repeats an output identity"
            )

        with self._lock:
            self._reclaim_ready_locked()
            for (reference, _device, _slot, _dtype, _field), key in zip(
                requested, keys, strict=True
            ):
                if int(reference.generation) < 1:
                    raise invalid_descriptor(
                        "request-relay registration requires a positive "
                        "logical generation"
                    )
                existing = self._products.get(key)
                if existing is not None:
                    if not existing.committed:
                        raise invalid_descriptor(
                            "request-relay output already has a candidate"
                        )
                    if existing.reference != reference:
                        raise invalid_descriptor(
                            "stale request-relay logical generation"
                        )
                    raise invalid_descriptor(
                        "request-relay output is already registered"
                    )

            # Keep one call on one lane: reuse its established lane, or
            # take the first lane whose fields all have no bound owner.
            call_lanes: dict[tuple[str, int, RequestKey, CallId], int] = {}
            for reference, device, request_slot, _dtype, _field in requested:
                call = (
                    str(device),
                    request_slot,
                    reference.request_key,
                    reference.producer_call_id,
                )
                lane = self._relay_call_lanes.get(call)
                if lane is None:
                    lane = call_lanes.get(call)
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
                call_lanes[call] = lane

            writes: list[TensorRecord] = []
            installed_calls: set[tuple[str, int, RequestKey, CallId]] = set()
            try:
                for (reference, device, request_slot, dtype, field), key in zip(
                    requested, keys, strict=True
                ):
                    call = (
                        str(device),
                        request_slot,
                        reference.request_key,
                        reference.producer_call_id,
                    )
                    lane = call_lanes[call]
                    slot = self._relay_slot_locked(
                        device,
                        request_slot,
                        lane,
                        dtype,
                        field,
                        call,
                    )
                    if slot.owner is not None:
                        raise _invariant(
                            "request-relay lane was assigned more than once"
                        )
                    # Bump the slot's physical generation so handles held by
                    # its previous owner fail validation.
                    generation = slot.generation + 1
                    slot.generation = (
                        1 if generation > _MAX_GENERATION else generation
                    )
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
                    self._relay_call_lanes[call] = lane
                    installed_calls.add(call)
                    writes.append(write)
            except BaseException:
                for write in reversed(writes):
                    self._writes.pop(write.binding_id)
                    self._products.pop(_reference_key(write.reference))
                    self._release_storage_locked(write)
                for call in installed_calls:
                    self._release_relay_call_locked(call)
                raise
            return tuple(writes)

    def _relay_lane_free_locked(
        self,
        device_name: str,
        request_slot: int,
        lane: int,
    ) -> bool:
        """Return whether a request relay lane has no bound call."""
        fields = self._relay_slots.get((device_name, request_slot, lane))
        return fields is None or all(
            slot.owner is None for slot in fields.values()
        )

    def _relay_slot_locked(
        self,
        device: torch.device,
        request_slot: int,
        lane: int,
        dtype: torch.dtype,
        field: int,
        call: tuple[str, int, RequestKey, CallId],
    ) -> RelaySlot:
        """Resolve or create one scalar relay slot and associate it with a call.

        Creating the first slot of a (device, dtype, field) allocates that
        arena, charged against ``byte_capacity``. The caller must hold the
        store lock and still has to check the slot's ``owner``.

        Raises:
            WorkerError: If the arena would exceed ``byte_capacity``, or the
                slot is still associated with a different call.
        """
        device_name = str(device)
        lane_key = (device_name, request_slot, lane)
        fields = self._relay_slots.setdefault(lane_key, {})
        key = (dtype, int(field))
        slot = fields.get(key)
        if slot is None:
            arena_key = (device_name, dtype, int(field))
            arena = self._relay_arenas.get(arena_key)
            if arena is None:
                # Every (request_slot, lane) pair of this dtype and field shares
                # one flat arena. Request slots are one-based and index rows
                # directly, so row zero is never bound.
                elements = (self.request_capacity + 1) * self.relay_depth
                projected = (
                    self._allocated_bytes + elements * _TORCH_DTYPE_BYTES[dtype]
                )
                self._require_byte_capacity_locked(projected)
                # CUDA arenas use exportable peer storage, like the
                # `BufferPool` arenas.
                if device.type == "cuda":
                    from uniserve_kernels.peer_storage import empty

                    arena = empty((elements,), dtype=dtype, device=device)
                else:
                    arena = torch.empty((elements,), dtype=dtype, device=device)
                self._relay_arenas[arena_key] = arena
                self._allocated_bytes = projected

            # Flat [request_slot, lane] position of this scalar view.
            index = request_slot * self.relay_depth + lane
            slot = RelaySlot(
                index=index,
                device_name=device_name,
                tensor=arena[index : index + 1],
                shape=(1,),
                dtype=dtype,
            )
            fields[key] = slot

        if slot.relay_lane is not None and slot.relay_lane[:4] != call:
            raise _invariant(
                "request-relay slot retained a conflicting call identity"
            )
        slot.relay_lane = (*call, lane)
        return slot

    def _release_relay_call_locked(
        self,
        call: tuple[str, int, RequestKey, CallId],
    ) -> None:
        """Drop a call's lane association once no field of that lane is owned.

        The association is kept while any field of the lane still has an
        owner, so later outputs of the same call keep using the same lane.
        """
        lane = self._relay_call_lanes.get(call)
        if lane is None:
            return
        if self._relay_lane_free_locked(call[0], call[1], lane):
            self._relay_call_lanes.pop(call, None)

    def bind_output_groups(
        self,
        groups: tuple[
            tuple[tuple[TensorRef, torch.device | str], ...],
            ...,
        ],
        *,
        request_slots: Mapping[RequestKey, int] | None = None,
        buffer_allocations: Mapping[BufferId, BufferAllocation] | None = None,
        regions: Mapping[TensorRef, tuple[slice, ...]] | None = None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None = None,
    ) -> tuple[tuple[TensorRecord, ...], ...]:
        """Bind several output groups, all or none.

        Each non-empty group goes through `bind_outputs` with the shared
        arguments; empty groups are skipped and produce no entry in the
        result. If any group fails, the records of earlier groups are
        abandoned before the error propagates.
        """
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
                self.abandon_writes(
                    tuple(write for binding in bindings for write in binding)
                )
                raise
        return tuple(bindings)

    def producer_write_views(
        self,
        writes: tuple[TensorRecord, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return the reserved storage views that producers write into.

        Raises:
            WorkerError: If a record is stale or already published.
        """
        if not writes:
            return ()
        with self._lock:
            entries = tuple(
                self._require_write_locked(write) for write in writes
            )
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
        """Publish a value into a reserved record.

        Encoder features require `FeatureMetadata` and are cast to their
        storage dtype; any other record rejects `FeatureMetadata`. The record
        becomes consumable only after `commit_writes`.

        Returns:
            The storage view holding the value, shaped like ``value``.

        Raises:
            WorkerError: If the record is stale or already published, the
                metadata does not match the record kind, or the value's dtype
                or shape disagrees with the reservation.
        """
        with self._lock:
            entry = self._require_write_locked(write)
            if entry.feature:
                if not isinstance(metadata, FeatureMetadata):
                    raise invalid_descriptor(
                        "encoder feature requires spatial metadata"
                    )
                value = value.to(dtype=entry.tensor.dtype)
            elif isinstance(metadata, FeatureMetadata):
                raise invalid_descriptor(
                    "feature metadata requires feature admission"
                )
            result = self._publish_locked(
                entry, value, producer_event=producer_event
            )
            entry.metadata = metadata
            return result

    def _publish_locked(
        self,
        entry: TensorRecord,
        value: torch.Tensor,
        *,
        producer_event: torch.cuda.Event | None,
    ) -> torch.Tensor:
        """Copy a value into a reserved record and record its producer event.

        A shard record requires exactly its region's shape; any other record
        accepts a shape within its logical bound and stores it as a prefix of
        the flattened storage. On CUDA the copy is enqueued on the current
        stream. Without a supplied ``producer_event``, a pooled event is
        recorded after the copy; a supplied event is only bound to the
        current stream, so ordering it after the copy is the caller's
        obligation. The record holds one `EventPool` reference to the event.
        """
        if entry.producer_recorded:
            raise _invariant("device product was published more than once")
        target = entry.tensor
        if target is None:
            raise _invariant("device product has no physical tensor")
        if value.dtype != target.dtype:
            raise invalid_descriptor(
                "tensor publication changes its declared dtype"
            )
        if entry.region is not None:
            shape_matches = tuple(value.shape) == _slices.shape(entry.region)
        else:
            shape_matches = entry.reference.shape_bound.contains_shape(
                tuple(value.shape)
            )
        if not shape_matches:
            raise invalid_descriptor(
                "tensor publication changes its declared shape"
            )

        flat = value.detach().reshape(-1)
        if flat.numel() > target.numel():
            raise _invariant(
                "device product exceeds its registered shape bound"
            )
        view = (
            target
            if entry.region is not None
            else target.reshape(-1)[: flat.numel()].reshape(value.shape)
        )

        # Skip the copy when the value already aliases the target exactly
        # (same storage, dtype, and strides).
        source = value.detach()
        if (
            view.data_ptr() != source.data_ptr()
            or view.dtype != source.dtype
            or view.stride() != source.stride()
        ):
            view.copy_(
                source.to(dtype=target.dtype),
                non_blocking=value.device.type == "cuda",
            )

        # Record or adopt the event that orders consumers after the producer.
        if target.device.type == "cuda":
            entry.producer_event, entry.producer_stream = (
                self._producer_event_locked(
                    target.device,
                    producer_event,
                )
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
        """Publish one scalar per record from a single values tensor.

        ``values`` flattens to exactly one element per record, in order. All
        records must be single-element and share one device and dtype; one
        producer event covers the whole batch. Validation happens before any
        copy, so a rejected batch publishes nothing.

        Returns:
            The single-element storage view of each record.
        """
        if not writes:
            return ()
        with self._lock:
            entries = tuple(
                self._require_write_locked(write) for write in writes
            )
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
        """Scatter one scalar per record and share one producer event."""
        flat = values.detach().reshape(-1)
        if int(flat.numel()) != len(entries):
            raise invalid_descriptor(
                "batched device-product publication requires one scalar "
                "per output"
            )
        if any(entry.producer_recorded for entry in entries):
            raise _invariant("device product was published more than once")

        targets = tuple(entry.tensor for entry in entries)
        if any(
            target is None or int(target.numel()) != 1 for target in targets
        ):
            raise invalid_descriptor(
                "batched device-product publication requires scalar "
                "output bounds"
            )
        tensors = tuple(target for target in targets if target is not None)
        first = tensors[0]
        if any(
            tensor.device != first.device or tensor.dtype != first.dtype
            for tensor in tensors
        ):
            raise invalid_descriptor(
                "batched device-product publication spans incompatible storage"
            )

        source = flat.to(dtype=first.dtype)
        # Keep the scalar batch in one native copy call. CUDA can scatter
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
            # One shared event guards the batch; each entry releases one of
            # these references when it is reclaimed.
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
        """Publish one host boolean or integer into a reserved record.

        The value fills only the record's first element, and the record
        reports a published shape of ``(1,)``. The record becomes consumable
        only after `commit_writes`.

        Returns:
            The single-element storage view.
        """
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
        """Fill a record's first element and record its producer event."""
        if entry.producer_recorded:
            raise _invariant("device product was published more than once")
        tensor = entry.tensor
        if tensor is None:
            raise _invariant("device product has no physical tensor")

        tensor.reshape(-1)[:1].fill_(int(value))
        if tensor.device.type == "cuda":
            entry.producer_event, entry.producer_stream = (
                self._producer_event_locked(
                    tensor.device,
                    producer_event,
                )
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
        consumer_call_id: CallId,
        device: torch.device | str | None = None,
    ) -> TensorRead:
        """Acquire a read lease on one committed product.

        See `consume_batch` for the contract.
        """
        return self.consume_batch(((reference, consumer_call_id, device),))[0]

    def consume_batch(
        self,
        requests: tuple[
            tuple[TensorRef, CallId, torch.device | str | None],
            ...,
        ],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[TensorRead, ...]:
        """Acquire read leases on committed products for consuming calls.

        Each reference must name a committed, published, unreleased product
        with its exact logical generation. On CUDA the current stream of the
        consumer device waits on each distinct producer event once; the wait
        is skipped when one event covers the batch and every product was
        produced on the consumer's current stream. No data moves: the
        consumer device must be the product's device.

        Every returned read holds a lease that blocks reclamation until the
        caller passes it to `complete_reads`.

        Args:
            requests: Each reference, its consuming call, and an optional
                consumer device. With ``device`` set, a per-request device
                must be absent or equal to it.
            device: Shared consumer device for the whole batch.

        Returns:
            One read per request, in order, viewing the published extent.

        Raises:
            WorkerError: If a reference is unknown, uncommitted, stale,
                released, or unpublished, or names a different device. No
                lease is taken when any request fails validation.
        """
        if not requests:
            return ()
        shared_target = None if device is None else canonical_device(device)

        # Shared-device batch: every product must live on `device`.
        if shared_target is not None:
            target_name = str(shared_target)
            with self._lock:
                shared_resolved: list[
                    tuple[TensorRecord, torch.Tensor, CallId]
                ] = []
                for reference, consumer_call_id, requested_device in requests:
                    entry = self._require_locked(reference)
                    if entry.released:
                        raise invalid_descriptor(
                            "device product was consumed after logical release"
                        )
                    if not entry.producer_recorded:
                        raise invalid_descriptor(
                            "device product was consumed before producer "
                            "publication"
                        )
                    storage = entry.tensor
                    if storage is None:
                        raise _invariant(
                            "device product has no physical tensor"
                        )
                    if (
                        requested_device is not None
                        and canonical_device(requested_device) != shared_target
                    ):
                        raise invalid_descriptor(
                            "device-product batch names conflicting "
                            "consumer devices"
                        )
                    if entry.device_name != target_name:
                        raise invalid_descriptor(
                            "device product consumer names a different device"
                        )
                    # A producer may publish less than the reserved shape;
                    # the read views only the published prefix.
                    tensor = (
                        storage
                        if entry.actual_shape == entry.shape
                        else storage.reshape(-1)[: entry.actual_extent].reshape(
                            entry.actual_shape
                        )
                    )
                    shared_resolved.append((entry, tensor, consumer_call_id))

                if shared_target.type == "cuda":
                    first_event = shared_resolved[0][0].producer_event
                    if first_event is None:
                        raise _invariant(
                            "CUDA device product has no producer event"
                        )
                    if all(
                        entry.producer_event is first_event
                        for entry, _tensor, _op in shared_resolved
                    ):
                        # One producer event covers the batch: a single wait
                        # suffices, and only when the consumer runs on a
                        # different stream than the producer.
                        stream = torch.cuda.current_stream(shared_target)
                        stream_id = int(stream.cuda_stream)
                        if any(
                            entry.producer_stream != stream_id
                            for entry, _tensor, _op in shared_resolved
                        ):
                            stream.wait_event(first_event)
                    else:
                        # Distinct producer events: wait on each unique
                        # one once.
                        shared_waited: set[int] = set()
                        stream = torch.cuda.current_stream(shared_target)
                        for (
                            entry,
                            _tensor,
                            _consumer_call_id,
                        ) in shared_resolved:
                            event = entry.producer_event
                            if event is None:
                                raise _invariant(
                                    "CUDA device product has no producer event"
                                )
                            identity = id(event)
                            if identity in shared_waited:
                                continue
                            stream.wait_event(event)
                            shared_waited.add(identity)

                for entry, _tensor, _consumer_call_id in shared_resolved:
                    entry.readers += 1
                return tuple(
                    TensorRead(
                        tensor=tensor,
                        consumer_call_id=consumer_call_id,
                        _write=entry,
                        region=entry.region,
                        metadata=entry.metadata,
                    )
                    for entry, tensor, consumer_call_id in shared_resolved
                )

        # Per-request batch: each read defaults to its product's own device.
        assert shared_target is None
        with self._lock:
            resolved: list[
                tuple[TensorRecord, torch.Tensor, torch.device, CallId]
            ] = []
            for reference, consumer_call_id, requested_device in requests:
                entry = self._require_locked(reference)
                if entry.released:
                    raise invalid_descriptor(
                        "device product was consumed after logical release"
                    )
                if not entry.producer_recorded:
                    raise invalid_descriptor(
                        "device product was consumed before producer "
                        "publication"
                    )
                storage = entry.tensor
                if storage is None:
                    raise _invariant("device product has no physical tensor")
                tensor = (
                    storage
                    if entry.actual_shape == entry.shape
                    else storage.reshape(-1)[: entry.actual_extent].reshape(
                        entry.actual_shape
                    )
                )
                target = (
                    storage.device
                    if requested_device is None
                    else canonical_device(requested_device)
                )
                if target != storage.device:
                    raise invalid_descriptor(
                        "device product consumer names a different device"
                    )
                resolved.append((entry, tensor, target, consumer_call_id))

            first_entry, _tensor, first_target, _consumer_call_id = resolved[0]
            if first_target.type == "cuda":
                first_event = first_entry.producer_event
                if first_event is None:
                    raise _invariant(
                        "CUDA device product has no producer event"
                    )
                if all(
                    target == first_target
                    and entry.producer_event is first_event
                    for entry, _tensor, target, _consumer_call_id in resolved
                ):
                    # One device and one producer event: wait once, unless the
                    # consumer already runs on the producer stream.
                    stream = torch.cuda.current_stream(first_target)
                    if any(
                        entry.producer_stream != int(stream.cuda_stream)
                        for entry, _tensor, _target, _consumer in resolved
                    ):
                        stream.wait_event(first_event)
                else:
                    # Mixed devices or events: wait each (device, event) once.
                    waited: set[tuple[str, int]] = set()
                    for entry, _tensor, target, _consumer_call_id in resolved:
                        if target.type != "cuda":
                            continue
                        event = entry.producer_event
                        if event is None:
                            raise _invariant(
                                "CUDA device product has no producer event"
                            )
                        event_identity = (str(target), id(event))
                        if event_identity in waited:
                            continue
                        torch.cuda.current_stream(target).wait_event(event)
                        waited.add(event_identity)

            reads = []
            for entry, tensor, _target, consumer_call_id in resolved:
                entry.readers += 1
                reads.append(
                    TensorRead(
                        tensor=tensor,
                        consumer_call_id=consumer_call_id,
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
        """Fence reads on their consuming streams, then end their leases.

        Call this after the consumer's work on each read has been enqueued on
        the current stream. Each CUDA read gets a reader event that the
        record keeps until reclamation: the producer event of the consuming
        call's output in ``after_writes`` when that output was produced on
        the current stream, otherwise an event recorded now. Reads already
        completed are skipped.

        Completing the last read of an import also drops the shared import:
        its tickets are cancelled if the import was never adopted, every
        ticket is closed, and a still-unpublished destination is abandoned.

        Args:
            reads: Leases returned by `consume_batch` or `import_tensor`.
            device: Device every read must be on, when known.
            after_writes: Outputs of the consuming calls. When it has one
                entry per read and every pair shares the read's device and
                consumer call (and, on CUDA, the current stream), each read
                is fenced by its pair; otherwise published outputs are
                matched to reads by consumer call.
        """
        with self._lock:
            pending = tuple(read for read in reads if not read._recorded)
            if not pending:
                return
            self._record_reader_fences(
                pending, device=device, after_writes=after_writes
            )
            for read in pending:
                entry = self._require_read_locked(read)
                entry.readers -= 1
                read._recorded = True

                imported = read.imported
                if imported is not None:
                    imported.users -= 1
                    if imported.users == 0:
                        # The last user drops the shared materialization,
                        # cancels its tickets unless it was adopted, and
                        # closes them.
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
        """Attach a reader event to each read's record on its consuming stream.

        Reads on non-CUDA devices need no fence. See `complete_reads` for the
        fence selection.
        """
        if not reads:
            return
        declared_target = None if device is None else canonical_device(device)
        with self._lock:
            # Pairwise fast path: when every read pairs with the write its own
            # consumer call produced on the same stream, that write's
            # producer event already fences the read. Any mismatch breaks out
            # to the per-stream fence path below.
            if len(after_writes) == len(reads):
                target = reads[0].tensor.device
                if declared_target is not None and declared_target != target:
                    raise _invariant(
                        "device-product reader completed on a different device"
                    )
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
                        or completion.reference.producer_call_id
                        != read.consumer_call_id
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
                        # One shared fence with no prior reader events: attach
                        # it once and hold one pooled reference per entry.
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

            # Fallback: prefer a producer event recorded by the read's own
            # consumer call when it matches the current stream; record a
            # fresh event for anything else.
            write_fences: dict[CallId, TensorRecord] = {}
            for write in after_writes:
                entry = self._require_write_locked(write)
                if not entry.producer_recorded:
                    continue
                write_fences.setdefault(entry.reference.producer_call_id, entry)
            if declared_target is not None:
                target = declared_target
                entries = []
                for read in reads:
                    entry = self._require_read_locked(read)
                    if target != read.tensor.device:
                        raise _invariant(
                            "device-product reader completed on a "
                            "different device"
                        )
                    entries.append((read, entry))

                if target.type == "cuda":
                    stream = torch.cuda.current_stream(target)
                    stream_id = int(stream.cuda_stream)
                    pending: list[TensorRecord] = []
                    for read, entry in entries:
                        completion_fence = (
                            None
                            if read.consumer_call_id is None
                            else write_fences.get(read.consumer_call_id)
                        )
                        if (
                            completion_fence is not None
                            and completion_fence.producer_event is not None
                            and completion_fence.tensor is not None
                            and completion_fence.tensor.device == target
                            and completion_fence.producer_stream == stream_id
                        ):
                            event = completion_fence.producer_event
                            self._append_reader_event_locked(
                                entry, event, target
                            )
                        else:
                            pending.append(entry)

                    # Entries without a matching write fence share one event
                    # recorded on the current stream.
                    if pending:
                        event, _stream_id = self._record_event_locked(target)
                        for entry in pending:
                            self._append_reader_event_locked(
                                entry, event, target
                            )
                return

            # No declared device: group reads by their tensor's device and
            # fence each device independently.
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
                        if read.consumer_call_id is None
                        else write_fences.get(read.consumer_call_id)
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

                # Entries without a matching write fence share one event
                # recorded on this device's current stream.
                if pending:
                    event, _stream_id = self._record_event_locked(target)
                    for entry in pending:
                        self._append_reader_event_locked(entry, event, target)

    def release_calls(
        self,
        releases: Iterable[tuple[RequestKey, CallId]],
    ) -> None:
        """Logically release every committed product of the given calls.

        Released products reject new reads. Their storage is reclaimed by a
        later reclamation pass once their leases, tickets, publications, and
        events have retired; this method does not run one.
        """
        with self._lock:
            for request_key, raw_call_id in releases:
                call_id = raw_call_id
                call_key = (request_key, call_id)
                call_writes = self._call_writes.pop(call_key, None)
                entries = list(
                    call_writes
                    if isinstance(call_writes, list)
                    else (() if call_writes is None else (call_writes,))
                )

                # Also release the call's output-zero product when it is
                # committed but absent from the call index.
                direct_key = (
                    int(request_key.engine_id),
                    int(request_key.request_id),
                    int(request_key.request_epoch),
                    call_id,
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
        """Revoke exports and logically release records of the given buffers.

        Existing read leases, tickets, and publications keep their storage
        until they retire; when any buffer is given, a reclamation pass runs
        before returning.
        """
        selected = set(buffers)
        release_exports(self.exports, selected)
        if not selected:
            return
        with self._lock:
            for entry in tuple(self._writes.values()):
                if entry.reference.buffer_id in selected:
                    self._release_entry_locked(entry)
            self._reclaim_ready_locked()

    def release_requests(
        self,
        requests: Iterable[RequestKey],
        *,
        retained: frozenset[BufferId] = frozenset(),
    ) -> None:
        """Logically release every record of the given requests.

        Records whose buffer is in ``retained`` are left live; only persistent
        records can be retained. When any request is given, a reclamation
        pass runs before returning.

        Raises:
            WorkerError: If ``retained`` names the buffer of a relay record.
        """
        selected = set(requests)
        if not selected:
            return
        with self._lock:
            for entry in tuple(self._writes.values()):
                if entry.reference.request_key in selected:
                    if entry.reference.buffer_id in retained:
                        if entry.buffer_binding is None:
                            raise invalid_descriptor(
                                "finish cannot retain request-slot storage"
                            )
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
        """Return whether the selected records have all been reclaimed.

        Selects records of ``buffers`` and records of ``requests`` whose
        buffer is not in ``retained``. Runs a reclamation pass first.

        Raises:
            WorkerError: If a selected record's transfer ticket cannot report
                physical completion.
            Exception: The failure of a finished transport publication of a
                selected record.
        """
        with self._lock:
            for entry in tuple(self._writes.values()):
                if entry.reference.buffer_id in buffers or (
                    entry.reference.request_key in requests
                    and entry.reference.buffer_id not in retained
                ):
                    # Raise for a transfer whose physical completion is
                    # unknown and for a failed publication; a failed
                    # publication otherwise blocks reclamation indefinitely.
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
        """Logically release one write and arrange its retirement wake."""
        self._detach_write_locked(entry)
        entry.released = True
        self._wake_retirement_locked(entry)
        # Transfers still filling a never-published destination are
        # cancelled; each ticket keeps the storage until it retires.
        if not entry.producer_recorded:
            for transfer in entry.transfers:
                transfer.cancel()

    def _wake_retirement_locked(self, entry: TensorRecord) -> None:
        """Schedule a completion wake on each incomplete event of an entry.

        When a completion wake is installed (`Worker.set_completion_wake`),
        the `EventPool` wake rouses the worker's result polling once the event
        completes, so a later pass can reclaim the entry.
        """
        readers = entry.reader_events
        events = (
            entry.producer_event,
            *(
                readers
                if isinstance(readers, list)
                else (() if readers is None else (readers,))
            ),
        )
        for event in events:
            if event is not None and not event.query():
                self.event_pool.schedule_completion_wake(
                    entry.device_name, event
                )

    def retain_publication(
        self, write: TensorRecord, retirement: Future[None]
    ) -> None:
        """Block reclamation of a record until a transport publication retires.

        ``retirement`` completes when the transport registration exposing
        the record's storage has retired. Publications that finished
        successfully are pruned from the record here.
        """
        with self._lock:
            entry = self._require_write_locked(write)
            # Keep unfinished and failed retirements: a failed publication
            # blocks reclamation so the failure surfaces to its owner.
            retained = tuple(
                publication
                for publication in entry.publications
                if not publication.done() or publication.exception() is not None
            )
            entry.publications = (*retained, retirement)

    def retain_transfer(
        self, write: TensorRecord, ticket: TransferTicket
    ) -> None:
        """Block reclamation of an unpublished record until a transfer retires.

        The record is the destination of ``ticket``. A reclamation pass runs
        when the ticket retires.

        Raises:
            WorkerError: If the record is stale, already published, or
                released.
        """
        with self._lock:
            entry = self._require_write_locked(write)
            if entry.producer_recorded or entry.released:
                raise _invariant("transfer destination already has a producer")
            entry.transfers = (*entry.transfers, ticket)

        # The ticket may retire after the write was released; re-run
        # reclamation so the storage is not stranded.
        def reclaim() -> None:
            with self._lock:
                self._reclaim_ready_locked()

        ticket.add_retirement_callback(reclaim)

    def abandon_writes(self, writes: tuple[TensorRecord, ...]) -> None:
        """Release reserved records that will not be committed.

        Records that are already stale are skipped. Each remaining record is
        marked released, and storage without pending leases, tickets,
        publications, or events is reclaimed before returning.
        """
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

        An import of a product that already has a pending import shares it.
        A committed, fully resident product needs no fetch. A committed shard
        fetches the rest of the logical tensor around itself; this requires
        that its scheduler reservation contains the full logical tensor and
        that no earlier transfer into it is still unretired. Compact shard
        reservations remain valid producer storage but cannot serve a full
        local consumer. Without a committed product, a new record is reserved
        (as an encoder feature when ``metadata`` is `FeatureMetadata`) and
        the whole tensor is fetched.

        Fetches are only submitted here. Once every ticket is ready, the
        caller either orders its consumer with `wait_import` or adopts the
        result with `complete_import`, which also waits, and ends the lease
        with `complete_reads`.

        Returns:
            A read lease on the destination, which has ``tensor.shape``.

        Raises:
            WorkerError: If the import conflicts with a pending import or
                with the resident product's generation, state, device,
                metadata, shape, or dtype; if a resident shard's storage
                cannot hold the full tensor or still has unretired
                transfers; or if reservation fails. Errors from submitting
                a fetch propagate unchanged. A failed call releases its
                lease, cancels and closes its tickets, and abandons a record
                it reserved.
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
            # Uncommitted candidates are not consumable coverage.
            if existing is not None and not existing.committed:
                existing = None

            missing: tuple[tuple[slice, ...], ...]
            full = tuple(
                slice(start, start + extent)
                for start, extent in zip(
                    (0,) * len(tensor.shape), tensor.shape, strict=True
                )
            )
            if existing is not None:
                write = self._require_locked(reference)
                if write.released or not write.producer_recorded:
                    raise invalid_descriptor(
                        "product import requires a live published generation"
                    )
                if (
                    write.device_name != str(target_device)
                    or write.metadata != metadata
                ):
                    raise invalid_descriptor(
                        "product import conflicts with resident ownership"
                    )
                storage = write.tensor
                assert storage is not None
                if write.region is None:
                    if write.actual_shape != tensor.shape:
                        raise invalid_descriptor(
                            "product import changes resident tensor shape"
                        )
                    destination = (
                        storage
                        if tuple(storage.shape) == tensor.shape
                        else storage.reshape(-1)[: write.actual_extent].reshape(
                            tensor.shape
                        )
                    )
                    missing = ()
                else:
                    persistent = write.buffer_binding
                    if (
                        persistent is None
                        or tuple(persistent.tensor.shape) != tensor.shape
                    ):
                        raise invalid_descriptor(
                            "product import exceeds its reserved logical "
                            "storage"
                        )
                    destination = persistent.tensor
                    if any(not ticket.retired() for ticket in write.transfers):
                        raise resource_error(
                            "product storage has pending physical reads"
                        )
                    missing = _slices.subtract(full, write.region)
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
                destination = destination.reshape(-1)[
                    : math.prod(tensor.shape)
                ].reshape(tensor.shape)
                missing = (full,)

            if str(destination.dtype).removeprefix("torch.") != tensor.dtype:
                if existing is None:
                    self.abandon_writes((write,))
                raise invalid_descriptor(
                    "product import changes resident tensor dtype"
                )

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

            # Fetch only the regions not already resident.
            try:
                for region in missing:
                    fetch_tensor(
                        tensor,
                        destination[region],
                        bindings=bindings,
                        region=region,
                        retain=retain,
                    )
            except BaseException:
                # Drop the lease and every reservation made for this attempt.
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
        """Make the current stream wait for an import's data.

        Calls `TransferTicket.result` on every ticket, which makes the current
        stream wait on the ticket's fence when it carries one, then waits on
        the record's producer event. The caller must first observe every
        ticket as ready;
        `TransferTicket.result` raises otherwise, and it re-raises a failed
        transfer.

        Raises:
            WorkerError: If ``read`` is completed or is not an import.
        """
        if read._recorded or read.imported is None:
            raise _invariant("closed or ordinary tensor read is not an import")
        for ticket in read.imported.tickets:
            ticket.result()
        event = read._write.producer_event
        if event is not None:
            torch.cuda.current_stream(read.tensor.device).wait_event(event)

    def complete_import(self, read: TensorRead) -> None:
        """Adopt an import's complete coverage into its record.

        Orders the current stream after the import like `wait_import`, then:
        a freshly reserved destination is published and committed; an
        expanded shard becomes a full-tensor record, and on CUDA its new
        producer event is recorded on the current stream. An import already
        adopted, including one of a fully resident product, is only waited
        on. The read lease stays open until `complete_reads`.
        """
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
                # The previous producer event's reference is released only
                # once that event completes, through `EventPool.defer_release`.
                if write.producer_event is not None:
                    stream = torch.cuda.current_stream(value.tensor.device)
                    stream.wait_event(write.producer_event)
                    previous = write.producer_event
                    write.producer_event, write.producer_stream = (
                        self._record_event_locked(value.tensor.device)
                    )
                    self.event_pool.retain(
                        write.producer_event, value.tensor.device
                    )
                    self.event_pool.defer_release((previous,), write)
                write.tensor = value.tensor
                write.shape = tuple(value.tensor.shape)
                write.region = None
                write.actual_shape = tuple(value.tensor.shape)
                write.actual_extent = int(value.tensor.numel())
            value.committed = True

    def defer_write(self, write: TensorRecord) -> None:
        """Let host work publish a reserved write after its call commits.

        The call's completion packs without it; the write is published and
        committed when the work that fills it completes.
        """
        with self._lock:
            entry = self._require_write_locked(write)
            if entry.committed or entry.producer_recorded:
                raise _invariant("a published write cannot be deferred")
            entry.deferred = True

    def validate_writes(self, writes: tuple[TensorRecord, ...]) -> None:
        """Check that a batch's records are live, uncommitted candidates.

        Each record must be published, or deferred to host work that will
        publish it later.

        Raises:
            WorkerError: If a record is stale, committed, or unpublished
                without being deferred.
        """
        with self._lock:
            for write in writes:
                entry = self._require_write_locked(write)
                if entry.committed:
                    raise _invariant("device-product candidate is not live")
                if not entry.producer_recorded and not entry.deferred:
                    raise _invariant(
                        "completion packing found an unpublished device product"
                    )

    def commit_writes(self, writes: tuple[TensorRecord, ...]) -> None:
        """Make published records consumable and index them by call.

        Deferred records that are still unpublished are skipped and commit
        later, when their host work publishes and commits them. All checks
        run before any record is committed.

        Raises:
            WorkerError: If a record is stale, already committed, unpublished,
                repeated, or no longer the registered record of its identity.
        """
        if not writes:
            return
        with self._lock:
            # A deferred write commits when the host work filling it has
            # published it, not with the call that reserved it.
            entries = tuple(
                entry
                for entry in (
                    self._require_write_locked(write) for write in writes
                )
                if not (entry.deferred and not entry.producer_recorded)
            )
            for entry in entries:
                entry.deferred = False
            if not entries:
                return
            keys = tuple(_reference_key(entry.reference) for entry in entries)
            if len(set(keys)) != len(keys):
                raise _invariant(
                    "device-product publication repeats a product identity"
                )
            for key, entry in zip(keys, entries, strict=True):
                if entry.committed:
                    raise _invariant("device-product candidate is not live")
                if not entry.producer_recorded:
                    raise _invariant(
                        "device-product candidate has no producer readiness"
                    )
                if self._products.get(key) is not entry:
                    raise _invariant(
                        "device-product publication lost its reserved identity"
                    )

            # Index each committed product under its call so it can be
            # released by call identity later.
            for entry in entries:
                entry.committed = True
                call_key = (
                    entry.reference.request_key,
                    entry.reference.producer_call_id,
                )
                call_writes = self._call_writes.get(call_key)
                if call_writes is None:
                    self._call_writes[call_key] = entry
                elif isinstance(call_writes, list):
                    call_writes.append(entry)
                else:
                    self._call_writes[call_key] = [
                        call_writes,
                        entry,
                    ]
                entry._indexed = True

    def _require_locked(self, reference: TensorRef) -> TensorRecord:
        """Resolve a committed product and reject stale logical generations."""
        entry = self._products.get(_reference_key(reference))
        if entry is None or not entry.committed:
            # The message lists the request's committed products so that a
            # product this rank never held can be told apart from one whose
            # producing call has not committed yet.
            committed = sorted(
                (key[3].batch_id, key[4])
                for key, value in self._products.items()
                if key[1] == int(reference.request_key.request_id)
                and value.committed
            )
            raise invalid_descriptor(
                f"unknown device-product reference: request "
                f"{reference.request_key.request_id} produced by "
                f"{reference.producer_call_id} output "
                f"{reference.output_index} generation "
                f"{reference.generation}; committed {committed}"
            )
        if entry.reference != reference:
            raise invalid_descriptor("stale device-product logical generation")
        return self._require_write_locked(entry)

    def _require_write_locked(self, write: TensorRecord) -> TensorRecord:
        """Reject stale physical handles, including recycled relay slots."""
        if write.retired or self._writes.get(write.binding_id) is not write:
            raise _invariant("stale device-product physical generation")
        slot = write.relay_slot
        if slot is not None and (
            slot.owner != write.binding_id
            or slot.generation != write.physical_generation
        ):
            raise _invariant("stale request-relay physical generation")
        return write

    def _require_read_locked(self, read: TensorRead) -> TensorRecord:
        """Resolve the live write behind a read lease."""
        return self._require_write_locked(read._write)

    def _release_storage_locked(self, entry: TensorRecord) -> None:
        """Return a record's persistent span or relay slot and mark it retired.

        The caller has already established that nothing can still access the
        storage. For a relay slot, the call's lane association is dropped
        once no field of the lane is owned.
        """
        if entry.retired:
            raise _invariant("tensor storage was retired more than once")

        slot = entry.relay_slot
        if slot is not None:
            slot.owner = None
            association = slot.relay_lane
            if association is not None:
                call = association[:4]
                self._release_relay_call_locked(call)
                # Once the call loses its lane, clear the association on
                # every field so the lane becomes fully reusable.
                if call not in self._relay_call_lanes:
                    fields = self._relay_slots[
                        (association[0], association[1], association[4])
                    ]
                    for candidate in fields.values():
                        if (
                            candidate.relay_lane is not None
                            and candidate.relay_lane[:4] == call
                        ):
                            candidate.relay_lane = None
        elif entry.buffer_binding is not None:
            self.buffer_pool.release(entry.buffer_binding)

        entry.retired = True

    def _detach_write_locked(self, entry: TensorRecord) -> None:
        """Remove a record from the per-call release index, if listed."""
        if not entry._indexed:
            return
        reference = entry.reference
        call_key = (
            reference.request_key,
            reference.producer_call_id,
        )
        call_writes = self._call_writes.get(call_key)
        if call_writes is entry:
            self._call_writes.pop(call_key, None)
        elif isinstance(call_writes, list):
            # The index stores a bare entry for one product and a list for
            # several; collapse back to a bare entry when one remains.
            remaining = [
                candidate for candidate in call_writes if candidate is not entry
            ]
            if not remaining:
                self._call_writes.pop(call_key, None)
            elif len(remaining) == 1:
                self._call_writes[call_key] = remaining[0]
            elif len(remaining) != len(call_writes):
                self._call_writes[call_key] = remaining
        entry._indexed = False

    def _producer_event_locked(
        self,
        device: torch.device,
        event: torch.cuda.Event | None,
    ) -> tuple[torch.cuda.Event, int]:
        """Return the producer event for a publication and its stream id.

        A given event is bound to the current stream through
        `EventPool.declare_stream` without being recorded; otherwise a pooled
        event is recorded on the current stream. The caller retains the
        returned event.
        """
        if event is None:
            return self._record_event_locked(device)
        return event, self.event_pool.declare_stream(event, device)

    def _record_event_locked(
        self, device: torch.device
    ) -> tuple[torch.cuda.Event, int]:
        """Acquire and record a reusable event on the current stream."""
        event = self.event_pool.acquire(device)
        return event, self.event_pool.record(event, device)

    def _append_reader_event_locked(
        self,
        entry: TensorRecord,
        event: torch.cuda.Event,
        device: torch.device,
    ) -> None:
        """Attach a reader event to a record unless it is already attached."""
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

        # One pooled reference per attached entry, released at reclamation.
        self.event_pool.retain(event, device)

    def _reclaim_ready_locked(self) -> int:
        """Reclaim every released record that nothing can still access.

        A record is reclaimed once it is released, has no read leases, all
        its transfer tickets have retired, all its publications have
        finished successfully, and its producer and reader events query as
        complete. Reclamation drops it from every table, returns its storage,
        and releases its event references.

        Returns:
            The number of records reclaimed.
        """
        reclaimed = 0
        readiness: dict[int, bool] = {}
        released_events: dict[int, tuple[torch.cuda.Event, int]] = {}

        def release_event(event: torch.cuda.Event) -> None:
            """Count one event reference to release after the pass."""
            identity = id(event)
            current = released_events.get(identity)
            if current is None:
                released_events[identity] = (event, 1)
            elif current[0] is not event:
                raise _invariant(
                    "device event identity changed during reclamation"
                )
            else:
                released_events[identity] = (event, current[1] + 1)

        def ready(event: torch.cuda.Event | None) -> bool:
            """Query each CUDA event at most once during this pass."""
            if event is None:
                return True
            identity = id(event)
            result = readiness.get(identity)
            if result is None:
                result = _event_ready(event)
                readiness[identity] = result
            return result

        def readers_ready(entry: TensorRecord) -> bool:
            """Return whether every reader event of a record is complete."""
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
                    raise _invariant(
                        "device-product reclamation lost its reserved identity"
                    )
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

        # Return pooled-event references after the pass, with one
        # `EventPool.release` call per distinct event.
        for event, count in released_events.values():
            self.event_pool.release(event, count)
        return reclaimed


__all__ = [
    "TensorRead",
    "TensorStore",
    "ImageMetadata",
    "TensorRecord",
    "FeatureMetadata",
    "device_product_capacity_bytes",
    "device_product_storage",
]
