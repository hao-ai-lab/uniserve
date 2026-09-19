"""Validated descriptions of tensor storage published between workers."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import TypeAlias

from ..foundation.errors import invalid_descriptor
from . import identity
from .validation import _bytes, _int, _ints, _map, _seq, _str, _uint, _uints

MAX_TRANSFER_HANDLE_BYTES = 64 * 1024

#: Bytes of a CUDA process descriptor handle, which names an allocation only
#: within the host that exported it.
DESCRIPTOR_HANDLE_BYTES = 4
#: Bytes of a CUDA fabric handle, which another host can import.
FABRIC_HANDLE_BYTES = 64


@dataclass(frozen=True, slots=True)
class WorkerEndpoint:
    """A rank incarnation and its actual host address space.

    Worker and rank names survive restarts. Incarnation identifies this
    loaded rank; address_space identifies its process, independently of
    Worker grouping. Backend publication addresses and storage generations
    remain in Locator.
    """

    worker_id: str
    rank: int
    node: str
    address_space: str
    incarnation: str

    def __post_init__(self) -> None:
        """Validate that every component of the endpoint identity is present."""
        if (
            not self.worker_id
            or self.rank < 0
            or not self.node
            or not self.address_space
            or not self.incarnation
        ):
            raise invalid_descriptor("worker endpoint identity is incomplete")

    @classmethod
    def local(cls, worker_id: str = "worker", rank: int = 0) -> WorkerEndpoint:
        """Identify a new rank in this process.

        Applies after a process fork as well.
        """
        import socket

        return cls(
            worker_id,
            rank,
            socket.gethostname(),
            _address_space,
            uuid.uuid4().hex,
        )

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "endpoint"
    ) -> WorkerEndpoint:
        """Parse a complete worker endpoint identity from the wire schema."""
        data = _map(value, where)
        return cls(
            worker_id=_str(data.get("worker_id"), f"{where}.worker_id"),
            rank=_uint(data.get("rank"), f"{where}.rank"),
            node=_str(data.get("node"), f"{where}.node"),
            address_space=_str(
                data.get("address_space"), f"{where}.address_space"
            ),
            incarnation=_str(data.get("incarnation"), f"{where}.incarnation"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the endpoint identity for IPC."""
        return {
            "worker_id": self.worker_id,
            "rank": self.rank,
            "node": self.node,
            "address_space": self.address_space,
            "incarnation": self.incarnation,
        }


_address_space: str


def _identify_address_space() -> None:
    # A process identity is independent of how many loaded Worker instances it
    # owns. Refresh after fork so inherited module state cannot identify a child
    # as its parent, including PID reuse in a long-lived process tree.
    global _address_space
    _address_space = f"{os.getpid()}:{uuid.uuid4().hex}"


_identify_address_space()
os.register_at_fork(after_in_child=_identify_address_space)


@dataclass(frozen=True, slots=True)
class LocalTransfer:
    """Identifies an in-process transfer by endpoint and registry key."""

    endpoint: str
    key: int

    def __post_init__(self) -> None:
        """Validate the process-local endpoint and registry key."""
        if not self.endpoint or self.key < 0:
            raise invalid_descriptor("local transfer handle is invalid")


@dataclass(frozen=True, slots=True)
class PosixShmTransfer:
    """Identifies shared-memory storage.

    Also identifies the endpoint that grants ready reads.
    """

    endpoint: str
    name: str

    def __post_init__(self) -> None:
        """Validate the shared-memory name and publishing endpoint."""
        if not self.endpoint or not self.name:
            raise invalid_descriptor("shared-memory transfer handle is invalid")


@dataclass(frozen=True, slots=True)
class CudaVmmTransfer:
    """Identify an immutable CUDA allocation and its reader-lease endpoint.

    Byte offsets locate ordered first-axis spans sharing the tensor strides.
    Lengths and counts encode consecutive runs of equally sized spans, keeping
    page maps compact without changing logical coverage or physical ownership.
    """

    endpoint: str
    publication_id: str
    storage_size_bytes: int
    storage_offsets_bytes: tuple[int, ...]
    span_lengths: tuple[int, ...]
    span_counts: tuple[int, ...]
    tensor_stride: tuple[int, ...]
    ready_event_handle: bytes
    # The producing rank's shareable allocation handle, of the type its device
    # was probed for. A fabric handle is importable from another host, so it
    # travels with the publication rather than through a descriptor grant that
    # only reaches this one.
    allocation_handle: bytes = b""
    # Byte offset of this publication's acknowledgment header inside the
    # exported allocation. A consumer writes its own slot's word there once its
    # reads retire, which retires the chunk without a host-local connection.
    # Negative when the publication carries no header.
    acknowledgment_offset: int = -1

    def __post_init__(self) -> None:
        """Validate native CUDA handles and the declared allocation bounds."""
        if (
            not self.endpoint
            or len(self.publication_id) != 32
            or self.storage_size_bytes < 1
            or not self.storage_offsets_bytes
            or len(self.span_counts) != len(self.span_lengths)
            or sum(self.span_counts) != len(self.storage_offsets_bytes)
            or any(count < 1 for count in self.span_counts)
            or any(
                not 0 <= offset < self.storage_size_bytes
                for offset in self.storage_offsets_bytes
            )
            or any(length < 1 for length in self.span_lengths)
            or any(stride < 0 for stride in self.tensor_stride)
            # A publication read only from this host carries the event its
            # consumers wait on. One read from another host carries no fence,
            # because none would reach there; its producer drained its stream
            # before publishing instead. Any other length is not a handle a
            # consumer could import.
            or len(self.ready_event_handle) not in (0, 64)
            # A fabric handle is 64 bytes and a process descriptor is 4; any
            # other length is not a handle this rank can import.
            or len(self.allocation_handle)
            not in (DESCRIPTOR_HANDLE_BYTES, FABRIC_HANDLE_BYTES)
        ):
            raise invalid_descriptor("CUDA VMM transfer handle is incomplete")


@dataclass(frozen=True, slots=True)
class ChannelTransfer:
    """Carry a host product's bytes on the rank channel's data path.

    Shared memory names a segment in one host's namespace, so a product whose
    consumer is on another host travels as bytes: in the producing rank's
    result, into the head's custody, and out in the consuming rank's batch.
    """

    endpoint: str
    payload: bytes

    def __post_init__(self) -> None:
        """Validate the publishing endpoint and the carried bytes."""
        # The bytes are the product; a locator without them names nothing.
        if not self.endpoint or not self.payload:
            raise invalid_descriptor("channel transfer handle is invalid")


TransferTransport: TypeAlias = (
    LocalTransfer | PosixShmTransfer | CudaVmmTransfer | ChannelTransfer
)


@dataclass(frozen=True, slots=True)
class Locator:
    """Describes a typed tensor view and its transport-specific handle.

    The handle owns the view's storage.
    """

    source: WorkerEndpoint
    transport: TransferTransport
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    offset: tuple[int, ...]
    device: str

    @property
    def backend(self) -> str:
        """Return the mechanism that owns this physical publication."""
        if isinstance(self.transport, LocalTransfer):
            return "local"
        if isinstance(self.transport, PosixShmTransfer):
            return "shm"
        if isinstance(self.transport, ChannelTransfer):
            return "channel"
        if isinstance(self.transport, CudaVmmTransfer):
            return "cuda_vmm"
        raise invalid_descriptor("tensor locator names an unknown transport")

    def __post_init__(self) -> None:
        """Validate tensor shape and handle kind.

        Ensures the handle matches its transport kind.
        """
        if (
            self.nbytes < 1
            or not self.dtype
            or not self.shape
            or any(extent < 1 for extent in self.shape)
            or not self.device
            or len(self.offset) != len(self.shape)
            or any(start < 0 for start in self.offset)
        ):
            raise invalid_descriptor(
                "transfer locator has invalid tensor bounds"
            )
        if isinstance(self.transport, CudaVmmTransfer) and (
            len(self.transport.tensor_stride) != len(self.shape)
            or sum(
                length * count
                for length, count in zip(
                    self.transport.span_lengths,
                    self.transport.span_counts,
                    strict=True,
                )
            )
            != self.shape[0]
        ):
            raise invalid_descriptor(
                "CUDA VMM physical spans do not match its shape"
            )

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "transfer locator"
    ) -> Locator:
        """Parse a tensor locator and validate it.

        Validation covers the transport handle, shape, dtype, and byte
        bounds.
        """
        data = _map(value, where)
        kind = _str(data.get("transport"), f"{where}.transport")

        if kind == "local":
            transport: TransferTransport = LocalTransfer(
                endpoint=_str(data.get("endpoint"), f"{where}.endpoint"),
                key=_uint(data.get("key"), f"{where}.key"),
            )
        elif kind == "channel":
            transport = ChannelTransfer(
                endpoint=_str(data.get("endpoint"), f"{where}.endpoint"),
                payload=_bytes(data.get("payload"), f"{where}.payload"),
            )
        elif kind == "posix_shm":
            transport = PosixShmTransfer(
                endpoint=_str(data.get("endpoint"), f"{where}.endpoint"),
                name=_str(data.get("name"), f"{where}.name"),
            )
        elif kind == "cuda_vmm":
            transport = CudaVmmTransfer(
                endpoint=_str(data.get("endpoint"), f"{where}.endpoint"),
                publication_id=_str(
                    data.get("publication_id"), f"{where}.publication_id"
                ),
                storage_size_bytes=_uint(
                    data.get("storage_size_bytes"),
                    f"{where}.storage_size_bytes",
                ),
                storage_offsets_bytes=tuple(
                    _ints(
                        data.get("storage_offsets_bytes"),
                        f"{where}.storage_offsets_bytes",
                    )
                ),
                span_lengths=tuple(
                    _ints(data.get("span_lengths"), f"{where}.span_lengths")
                ),
                span_counts=tuple(
                    _ints(data.get("span_counts"), f"{where}.span_counts")
                ),
                tensor_stride=tuple(
                    _ints(data.get("tensor_stride"), f"{where}.tensor_stride")
                ),
                ready_event_handle=_bytes(
                    data.get("ready_event_handle"),
                    f"{where}.ready_event_handle",
                ),
                allocation_handle=_bytes(
                    data.get("allocation_handle"),
                    f"{where}.allocation_handle",
                ),
                acknowledgment_offset=_int(
                    data.get("acknowledgment_offset"),
                    f"{where}.acknowledgment_offset",
                ),
            )
        else:
            raise invalid_descriptor(f"{where}.transport is invalid")

        return cls(
            source=WorkerEndpoint.from_mapping(
                data.get("source"), f"{where}.source"
            ),
            transport=transport,
            nbytes=_uint(data.get("nbytes"), f"{where}.nbytes"),
            dtype=_str(data.get("dtype"), f"{where}.dtype"),
            shape=tuple(_uints(data.get("shape"), f"{where}.shape")),
            offset=tuple(_uints(data.get("offset"), f"{where}.offset")),
            device=_str(data.get("device"), f"{where}.device"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode the tensor shape and transport-specific handle.

        Produces a wire mapping.
        """
        output: dict[str, object] = {
            "source": self.source.to_mapping(),
            "nbytes": self.nbytes,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset": list(self.offset),
            "device": self.device,
        }

        transport = self.transport
        if isinstance(transport, LocalTransfer):
            output.update(
                transport="local",
                endpoint=transport.endpoint,
                key=transport.key,
            )
        elif isinstance(transport, PosixShmTransfer):
            output.update(
                transport="posix_shm",
                endpoint=transport.endpoint,
                name=transport.name,
            )
        elif isinstance(transport, ChannelTransfer):
            output.update(
                transport="channel",
                endpoint=transport.endpoint,
                payload=transport.payload,
            )
        else:
            output.update(
                transport="cuda_vmm",
                endpoint=transport.endpoint,
                publication_id=transport.publication_id,
                storage_size_bytes=transport.storage_size_bytes,
                storage_offsets_bytes=list(transport.storage_offsets_bytes),
                span_lengths=list(transport.span_lengths),
                span_counts=list(transport.span_counts),
                tensor_stride=list(transport.tensor_stride),
                ready_event_handle=transport.ready_event_handle,
                allocation_handle=transport.allocation_handle,
                acknowledgment_offset=transport.acknowledgment_offset,
            )
        return output


@dataclass(frozen=True, slots=True)
class TensorTransfer:
    """Actual logical tensor shape and its immutable physical locations.

    Locations can be shards or equivalent replicas. They need not cover the
    whole value until a consumer binds its required region. All coordinates
    are in logical element order; a backend's native strides describe
    physical order.
    """

    shape: tuple[int, ...]
    locations: tuple[Locator, ...]

    def __post_init__(self) -> None:
        """Validate location consistency.

        Every location must be a consistent region of the logical tensor.
        """
        if (
            not self.shape
            or any(extent < 1 for extent in self.shape)
            or not self.locations
        ):
            raise invalid_descriptor(
                "tensor transfer has no shape or locations"
            )

        # The first location fixes the element byte width; every location must
        # share its dtype, stay inside the logical shape, and occupy exactly
        # shape * element_bytes bytes.
        first = self.locations[0]
        elements = math.prod(first.shape)
        if first.nbytes % elements or first.nbytes < elements:
            raise invalid_descriptor(
                "tensor transfer has an invalid element size"
            )

        element_bytes = first.nbytes // elements
        for location in self.locations:
            if (
                len(location.shape) != len(self.shape)
                or any(
                    start + extent > bound
                    for start, extent, bound in zip(
                        location.offset, location.shape, self.shape, strict=True
                    )
                )
                or location.dtype != first.dtype
                or location.nbytes != math.prod(location.shape) * element_bytes
            ):
                raise invalid_descriptor(
                    "tensor location disagrees with its logical representation"
                )

    @property
    def dtype(self) -> str:
        return self.locations[0].dtype

    @property
    def nbytes(self) -> int:
        first = self.locations[0]
        return math.prod(self.shape) * (first.nbytes // math.prod(first.shape))

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "tensor transfer"
    ) -> TensorTransfer:
        data = _map(value, where)
        return cls(
            shape=tuple(_uints(data.get("shape"), f"{where}.shape")),
            locations=tuple(
                Locator.from_mapping(item, f"{where}.locations[{index}]")
                for index, item in enumerate(
                    _seq(data.get("locations"), f"{where}.locations")
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "locations": [location.to_mapping() for location in self.locations],
        }


@dataclass(frozen=True, slots=True)
class EncoderTransferValue:
    """Describes the media shape and payload encoding.

    Also describes the transferred encoder features.
    """

    height: int
    width: int
    payload_kind: str
    tensor: TensorTransfer


@dataclass(frozen=True, slots=True)
class DeviceProductTransferValue:
    """Describes the media shape and numeric range.

    Also describes the transferred device product.
    """

    height: int
    width: int
    value_range: str
    tensor: TensorTransfer


@dataclass(frozen=True, slots=True)
class KvTransfer:
    """Describes a versioned KV extent and its page locators.

    Also describes the source-to-destination buffer relation.
    """

    tensors: tuple[TensorTransfer, ...]
    source: identity.BufferId
    destination: str
    base: identity.BufferId | None
    base_extent: int
    published_extent: int
    group_id: int
    compute_dtype: str
    page_size: int

    def __post_init__(self) -> None:
        """Validate the exact KV source.

        Also validates the installed base and represented extent.
        """
        if (
            not self.destination
            or self.base_extent < 0
            or self.published_extent < self.base_extent
        ):
            raise invalid_descriptor(
                "KV publication extent or destination is invalid"
            )
        if self.base is None and self.base_extent != 0:
            raise invalid_descriptor(
                "KV publication base identity disagrees with its extent"
            )
        if (
            self.group_id < 0
            or self.page_size < 1
            or self.compute_dtype
            not in {"float16", "bfloat16", "float32", "float64"}
        ):
            raise invalid_descriptor(
                "KV publication storage identity is invalid"
            )

        suffix = self.published_extent - self.base_extent
        if not suffix:
            if self.tensors:
                raise invalid_descriptor(
                    "empty KV suffix carries physical tensors"
                )
            return

        if len(self.tensors) not in {2, 3}:
            raise invalid_descriptor(
                "KV publication requires raw keys, values and optional scales"
            )

        # Key and value tensors are [suffix tokens, layers, heads, head dim].
        key, value = self.tensors[:2]
        if (
            len(key.shape) != 4
            or key.shape[0] != suffix
            or value.shape != key.shape
            or value.dtype != key.dtype
            or key.dtype
            not in {
                "float16",
                "bfloat16",
                "float32",
                "float64",
                "float8_e4m3fn",
            }
        ):
            raise invalid_descriptor(
                "KV publication has invalid token, layer, or head bounds"
            )

        quantized = key.dtype == "float8_e4m3fn"
        if (len(self.tensors) == 3) != quantized:
            raise invalid_descriptor(
                "KV publication scale presence disagrees with its storage"
            )

        if quantized:
            scales = self.tensors[2]
            # Pages cover the unaligned base tail plus the suffix, and scales
            # are [pages, K/V, layers, head groups] per source page.
            pages = (
                self.base_extent % self.page_size + suffix + self.page_size - 1
            ) // self.page_size
            if (
                scales.dtype != "float32"
                or len(scales.shape) != 4
                or scales.shape[:3] != (pages, 2, key.shape[1])
                or key.shape[2] % scales.shape[3]
            ):
                raise invalid_descriptor(
                    "KV publication scales disagree with its source pages"
                )

    @property
    def scale_head_size(self) -> int:
        """Heads sharing each source scale; zero denotes unquantized storage.

        Scale tensors use [page, K/V, layer, head group]. Groups partition the
        logical head axis uniformly; each locator identifies the producer's
        actual group, including replicated full-cache representations.
        """
        return (
            self.tensors[0].shape[2] // self.tensors[2].shape[3]
            if len(self.tensors) == 3
            else 0
        )

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "kv_transfer"
    ) -> KvTransfer:
        """Decode source identity and physical cache representation."""
        data = _map(value, where)
        raw_base = data.get("base")
        return cls(
            tensors=tuple(
                TensorTransfer.from_mapping(item, f"{where}.tensors[{index}]")
                for index, item in enumerate(
                    _seq(data.get("tensors"), f"{where}.tensors")
                )
            ),
            source=identity.BufferId.from_mapping(
                data.get("source"), f"{where}.source"
            ),
            destination=_str(data.get("destination"), f"{where}.destination"),
            base=(
                None
                if raw_base is None
                else identity.BufferId.from_mapping(raw_base, f"{where}.base")
            ),
            base_extent=_uint(
                data.get("base_extent", 0), f"{where}.base_extent"
            ),
            published_extent=_uint(
                data.get("published_extent", 0),
                f"{where}.published_extent",
            ),
            group_id=_uint(data.get("group_id", 0), f"{where}.group_id"),
            compute_dtype=_str(
                data.get("compute_dtype"), f"{where}.compute_dtype"
            ),
            page_size=_uint(data.get("page_size"), f"{where}.page_size"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode the cache publication without a generic product envelope."""
        return {
            "tensors": [tensor.to_mapping() for tensor in self.tensors],
            "source": self.source.to_mapping(),
            "destination": self.destination,
            "base": None if self.base is None else self.base.to_mapping(),
            "base_extent": self.base_extent,
            "published_extent": self.published_extent,
            "group_id": self.group_id,
            "compute_dtype": self.compute_dtype,
            "page_size": self.page_size,
        }

    def encoded_size_bound(self) -> int:
        """Bound all page and scale locators and publication metadata."""
        size = (
            _tensor_transfers_size(self.tensors)
            + len(self.destination.encode())
            + len(self.compute_dtype.encode())
        )
        if size > MAX_TRANSFER_HANDLE_BYTES:
            raise invalid_descriptor(
                "KV transfer exceeds its descriptor byte bound"
            )
        return size


@dataclass(frozen=True, slots=True)
class LatentTransferValue:
    """Describes transferred latent storage.

    Carries a denoising step and media shape.
    """

    height: int
    width: int
    latent_units: int
    step: int
    tensor: TensorTransfer


TransferValue: TypeAlias = (
    EncoderTransferValue | DeviceProductTransferValue | LatentTransferValue
)


def _tensor_transfers_size(tensors: tuple[TensorTransfer, ...]) -> int:
    """Estimate the encoded byte size of tensor locators for handle bounding.

    The integer constants are conservative per-record overheads (field names,
    tags, lengths), not exact wire sizes; only string and list lengths are
    measured from the values themselves.
    """
    locators = tuple(
        location for tensor in tensors for location in tensor.locations
    )
    size = 512 + sum(64 + 8 * len(tensor.shape) for tensor in tensors)

    for locator in locators:
        size += (
            256
            + len(locator.dtype.encode())
            + len(locator.device.encode())
            + 16 * len(locator.shape)
            + len(locator.source.worker_id.encode())
            + len(locator.source.node.encode())
            + len(locator.source.address_space.encode())
            + len(locator.source.incarnation.encode())
        )

        transport = locator.transport
        if isinstance(transport, LocalTransfer):
            size += len(transport.endpoint.encode()) + 16
        elif isinstance(transport, PosixShmTransfer):
            size += (
                len(transport.endpoint.encode())
                + len(transport.name.encode())
                + 16
            )
        elif isinstance(transport, ChannelTransfer):
            # The bytes are the product, not the handle: their budget is the
            # channel's byte capacity and the message caps of the rank channel
            # they travel on, so only the locator's framing counts here.
            size += len(transport.endpoint.encode()) + 24
        elif isinstance(transport, CudaVmmTransfer):
            size += (
                len(transport.endpoint.encode())
                + len(transport.publication_id.encode())
                + len(transport.ready_event_handle)
                + 8 * len(transport.tensor_stride)
                + 8 * len(transport.storage_offsets_bytes)
                + 12 * len(transport.span_lengths)
                + 64
            )
        else:
            raise invalid_descriptor(
                "tensor transfer names an unknown transport"
            )
    return size
