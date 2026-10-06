"""Validated descriptions of tensor storage published between workers.

A producing rank publishes a tensor through one of the transports in
`uniserve_worker.transport` and reports it as a `Locator`: the publishing
`WorkerEndpoint`, the view's dtype, shape, and offset within the logical
tensor, and a transport handle (`LocalTransfer`, `PosixShmTransfer`,
`CudaVmmTransfer`, or `ChannelTransfer`) that a consumer opens. A
`TensorTransfer` groups the shard and replica locators of one logical tensor;
the `TransferValue` variants add product metadata to it, and `KvTransfer`
carries a published KV extent.

`Locator` owns the native worker-ipc description shared by Rust transport
owners and read consumers. Python accesses its immutable tensor metadata and
transport fields for numerical copies. `to_mapping` and `from_mapping` serve
the Python IPC records; native consumers use the description directly.
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import TypeAlias

from uniserve_worker._uniserve_ipc import Locator as Locator
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol import identity
from uniserve_worker.protocol.validation import (
    _map,
    _seq,
    _str,
    _uint,
    _uints,
)

#: Upper bound on a transfer descriptor's estimated encoded size, in bytes. It
#: equals the worker-ipc crate's `MAX_TRANSFER_HANDLE_BYTES`, which the crate's
#: validators enforce on their own estimate of the same descriptor.
MAX_TRANSFER_HANDLE_BYTES = 64 * 1024

#: Bytes of a POSIX file-descriptor allocation handle. It names an open file of
#: the exporting process, so a consumer on the same host receives a usable
#: descriptor through `uniserve_worker.transport.descriptor_grants`, and no
#: other host can import it.
DESCRIPTOR_HANDLE_BYTES = 4
#: Bytes of a CUDA fabric handle, which another host can import.
FABRIC_HANDLE_BYTES = 64


@dataclass(frozen=True, slots=True)
class WorkerEndpoint:
    """A rank incarnation and the host address space it runs in.

    `worker_id` and `rank` name the logical position and survive a restart;
    `incarnation` is fresh for every loaded rank. `address_space` identifies
    the process, independently of how many Worker instances it hosts, and
    `node` is the host name. Export addresses belong to each
    `Locator`'s transport handle, not to the endpoint.
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
        """Identify a new rank incarnation in this process.

        The address space is the current process's, including in a child
        created by fork, because `_identify_address_space` reruns after fork.
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


@dataclass(frozen=True, slots=True)
class PosixShmTransfer:
    """Identifies a POSIX shared-memory segment by name.

    `endpoint` names the producer's transport, which refuses a release from
    another endpoint. Readiness travels in the segment header.
    """

    endpoint: str
    name: str


@dataclass(frozen=True, slots=True)
class CudaVmmTransfer:
    """Identify an immutable CUDA allocation and the spans of one view in it.

    `storage_offsets_bytes` gives, in order, the byte offset of each
    first-axis span of the view within the exported allocation, which is
    `storage_size_bytes` long. Every span shares `tensor_stride`, in elements
    with one entry per axis. `span_lengths` and `span_counts` run-length
    encode the spans' first-axis lengths: ``span_counts[i]`` consecutive spans
    each hold ``span_lengths[i]`` rows, keeping page maps compact without
    changing logical coverage. `Locator` checks that the
    expanded lengths sum to the view's first extent.

    `endpoint` names the producer's export table and, for a descriptor
    handle, its grant socket. `export_id` is 32 characters (the producer
    uses ``uuid.uuid4().hex``) and keys that grant.
    """

    endpoint: str
    export_id: str
    storage_size_bytes: int
    storage_offsets_bytes: tuple[int, ...]
    span_lengths: tuple[int, ...]
    span_counts: tuple[int, ...]
    tensor_stride: tuple[int, ...]
    ready_event_handle: bytes
    # The producing rank's shareable allocation handle, of the type its device
    # was probed for. A fabric handle is importable from another host, so it
    # travels with the export rather than through a descriptor grant that
    # only reaches this one. A descriptor travels only so a consumer can tell
    # which kind it is; the usable one comes from the grant.
    allocation_handle: bytes = b""
    # Byte offset of this export's acknowledgment header inside the
    # exported allocation. A consumer writes its own slot's word there once its
    # reads retire, which retires the chunk without a host-local connection.
    # Negative when the export carries no header, which is the case for
    # storage exported where it lies rather than copied into the device pool.
    acknowledgment_offset: int = -1


@dataclass(frozen=True, slots=True)
class ChannelTransfer:
    """Carry a host product's bytes on the rank channel's data path.

    Shared storage names a segment in one host's namespace, so a product whose
    consumer is on another host travels as bytes: in the producing rank's
    result, into the head's custody, and out in the consuming rank's batch.
    """

    endpoint: str
    payload: bytes


TransferTransport: TypeAlias = (
    LocalTransfer | PosixShmTransfer | CudaVmmTransfer | ChannelTransfer
)


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
        """Element type name, shared by every location."""
        return self.locations[0].dtype

    @property
    def nbytes(self) -> int:
        """Bytes of the whole logical tensor, not of any one location."""
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
    """Encoder features published for another stage.

    `height` and `width` are the source image's size in pixels.
    `payload_kind` is ``"vision_feature"`` or ``"latent_feature"``, the wire
    names of the worker-ipc crate's `FeatureKind`.
    """

    height: int
    width: int
    payload_kind: str
    tensor: TensorTransfer


@dataclass(frozen=True, slots=True)
class DeviceProductTransferValue:
    """A device-resident product used by another model stage.

    `height` and `width` are in pixels, or zero for a non-image tensor.
    `value_range` names the semantic numeric range of the values and may be
    empty.
    """

    height: int
    width: int
    value_range: str
    tensor: TensorTransfer


@dataclass(frozen=True, slots=True)
class KvGroupTransfer:
    """One cache group's share of a KV export.

    The tensors carry the group's tokens ``[start, exported_extent)`` of the
    enclosing `KvTransfer`: a full-attention group starts at the
    export's base extent, and a sliding-window group no earlier than
    the first token its readers need. With ``T`` those tokens:

    - keys and values are ``[T, layers, kv heads, head dim]`` over the
      group's layers, with one dtype shared by both;
    - with ``float8_e4m3fn`` storage only, a third ``float32`` scale tensor
      is ``[pages, 2, layers, head groups]``, where ``pages`` counts the
      source pages of `page_tokens` tokens the carried tokens touch (the
      first one holds ``start``), the second axis selects K or V, and the
      kv-head count is a multiple of ``head groups``.

    A group that carries no token has no tensors.
    """

    start: int
    page_tokens: int
    tensors: tuple[TensorTransfer, ...]

    def validate(self, base_extent: int, exported_extent: int) -> None:
        """Validate the carried interval and tensor layout.

        Raises:
            WorkerError: `invalid_descriptor` when the interval lies outside
                ``[base_extent, exported_extent]``, the tensors disagree
                with it, or the K/V/scale geometry is invalid.
        """
        if (
            self.page_tokens < 1
            or not base_extent <= self.start <= exported_extent
        ):
            raise invalid_descriptor("KV group transfer interval is invalid")

        carried = exported_extent - self.start
        if not carried:
            if self.tensors:
                raise invalid_descriptor(
                    "empty KV group interval carries physical tensors"
                )
            return

        if len(self.tensors) not in {2, 3}:
            raise invalid_descriptor(
                "KV export requires raw keys, values and optional scales"
            )

        # Key and value tensors are [carried tokens, layers, heads, head dim].
        key, value = self.tensors[:2]
        if (
            len(key.shape) != 4
            or key.shape[0] != carried
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
                "KV export has invalid token, layer, or head bounds"
            )

        quantized = key.dtype == "float8_e4m3fn"
        if (len(self.tensors) == 3) != quantized:
            raise invalid_descriptor(
                "KV export scale presence disagrees with its storage"
            )

        if quantized:
            scales = self.tensors[2]
            # Scale rows cover whole source pages, from the page holding
            # ``start`` through the page holding the last carried token, so
            # tokens already installed on a partially filled boundary page
            # count toward the page total.
            pages = (
                self.start % self.page_tokens + carried + self.page_tokens - 1
            ) // self.page_tokens
            if (
                scales.dtype != "float32"
                or len(scales.shape) != 4
                or scales.shape[:3] != (pages, 2, key.shape[1])
                or key.shape[2] % scales.shape[3]
            ):
                raise invalid_descriptor(
                    "KV export scales disagree with its source pages"
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
        cls, value: object, where: str = "kv_group_transfer"
    ) -> KvGroupTransfer:
        """Decode one group's carried interval and tensors."""
        data = _map(value, where)
        return cls(
            start=_uint(data.get("start"), f"{where}.start"),
            page_tokens=_uint(data.get("page_tokens"), f"{where}.page_tokens"),
            tensors=tuple(
                TensorTransfer.from_mapping(item, f"{where}.tensors[{index}]")
                for index, item in enumerate(
                    _seq(data.get("tensors"), f"{where}.tensors")
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode one group's carried interval and tensors."""
        return {
            "start": self.start,
            "page_tokens": self.page_tokens,
            "tensors": [tensor.to_mapping() for tensor in self.tensors],
        }


@dataclass(frozen=True, slots=True)
class KvTransfer:
    """A published KV extent and the physical tensors that install its suffix.

    The export is incremental: the destination already holds `base`,
    when set, up to `base_extent` tokens, and each of `groups` carries only
    its tokens from its `KvGroupTransfer.start` on. An unchanged extent
    carries no groups; otherwise there is one entry per cache group, in
    table order. `compute_dtype` is the precision used when reading
    quantized source pages. `__post_init__` does not check the descriptor
    size bound; native output commit and batch validation check it.
    The worker-ipc crate's `KvTransfer::validate` checks the same relations
    and the size bound in the crate's codec.
    """

    groups: tuple[KvGroupTransfer, ...]
    source: identity.BufferId
    destination: str
    base: identity.BufferId | None
    base_extent: int
    exported_extent: int
    compute_dtype: str

    def __post_init__(self) -> None:
        """Validate the extent, base identity, dtype, and every group."""
        if (
            not self.destination
            or self.base_extent < 0
            or self.exported_extent < self.base_extent
        ):
            raise invalid_descriptor(
                "KV export extent or destination is invalid"
            )
        if self.base is None and self.base_extent != 0:
            raise invalid_descriptor(
                "KV export base identity disagrees with its extent"
            )
        if self.compute_dtype not in {
            "float16",
            "bfloat16",
            "float32",
            "float64",
        }:
            raise invalid_descriptor("KV export storage identity is invalid")
        if bool(self.groups) != (self.exported_extent > self.base_extent):
            raise invalid_descriptor(
                "KV group presence disagrees with its incremental extent"
            )
        for group in self.groups:
            group.validate(self.base_extent, self.exported_extent)

    @property
    def tensors(self) -> tuple[TensorTransfer, ...]:
        """Every group's published tensors, in group order."""
        return tuple(
            tensor for group in self.groups for tensor in group.tensors
        )

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "kv_transfer"
    ) -> KvTransfer:
        """Decode source identity and every group's physical representation.

        Absent ``base_extent`` and ``exported_extent`` decode as zero.
        """
        data = _map(value, where)
        raw_base = data.get("base")
        return cls(
            groups=tuple(
                KvGroupTransfer.from_mapping(item, f"{where}.groups[{index}]")
                for index, item in enumerate(
                    _seq(data.get("groups"), f"{where}.groups")
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
            exported_extent=_uint(
                data.get("exported_extent", 0),
                f"{where}.exported_extent",
            ),
            compute_dtype=_str(
                data.get("compute_dtype"), f"{where}.compute_dtype"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode the cache export without a generic product envelope."""
        return {
            "groups": [group.to_mapping() for group in self.groups],
            "source": self.source.to_mapping(),
            "destination": self.destination,
            "base": None if self.base is None else self.base.to_mapping(),
            "base_extent": self.base_extent,
            "exported_extent": self.exported_extent,
            "compute_dtype": self.compute_dtype,
        }

    def encoded_size_bound(self) -> int:
        """Return the estimated descriptor size, enforcing the handle bound.

        The estimate covers every page and scale locator of every group plus
        the export metadata, and parallels the worker-ipc crate's
        `KvTransfer::encoded_size_bound`.

        Raises:
            WorkerError: `invalid_descriptor` when the estimate exceeds
                `MAX_TRANSFER_HANDLE_BYTES` or a locator names an unknown
                transport.
        """
        size = (
            _tensor_transfers_size(self.tensors)
            + 32 * len(self.groups)
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
    """A diffusion trajectory tensor published for another stage.

    `height` and `width` are the output image's size in pixels,
    `latent_units` the logical latent allocation units, and `step` the
    denoising step the tensor represents.
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
    measured from the values themselves. The estimate parallels the
    worker-ipc crate's `transfer_encoded_size`, which the crate's validators
    check against the same bound when the codec encodes or decodes a batch
    or a batch result; an estimate here below the crate's lets the worker
    commit a descriptor that the codec then rejects.
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
                + len(transport.export_id.encode())
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
