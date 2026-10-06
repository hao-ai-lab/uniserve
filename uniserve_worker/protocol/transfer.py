"""Validated descriptions of tensor storage published between workers.

A producing rank publishes a tensor through one of the transports in
`uniserve_worker.transport` and reports it as a `Locator`: the publishing
`WorkerEndpoint`, the view's dtype, shape, and offset within the logical
tensor, and a transport handle (`LocalTransfer`, `PosixShmTransfer`,
`CudaVmmTransfer`, or `ChannelTransfer`) that a consumer opens. A
`TensorTransfer` groups the shard and replica locators of one logical tensor;
the `TransferValue` variants add product metadata to it, and `KvTransfer`
carries a published KV extent.

These records mirror the worker-ipc crate's `Locator`, `TensorTransfer`, and
`KvTransfer`, whose validators the crate's codec runs whenever it encodes or
decodes a batch or a batch result, so the two sides must change together.
`Locator.to_mapping` writes, and `Locator.from_mapping` reads, a flattened
form with a plain ``transport`` string beside the handle's fields; the PyO3
extension decodes that form rather than the crate's serde representation.
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import TypeAlias

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol import identity
from uniserve_worker.protocol.validation import (
    _bytes,
    _int,
    _ints,
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

    def __post_init__(self) -> None:
        """Validate the process-local endpoint and registry key."""
        if not self.endpoint or self.key < 0:
            raise invalid_descriptor("local transfer handle is invalid")


@dataclass(frozen=True, slots=True)
class PosixShmTransfer:
    """Identifies a POSIX shared-memory segment by name.

    `endpoint` names the producer's buffer registry (`BufferRegistry.name` in
    `uniserve_worker._uniserve_ipc`), which refuses a release carrying
    another table's name. Readiness travels in the segment header.
    """

    endpoint: str
    name: str

    def __post_init__(self) -> None:
        """Validate the shared-storage name and publishing endpoint."""
        if not self.endpoint or not self.name:
            raise invalid_descriptor(
                "shared-storage transfer handle is invalid"
            )


@dataclass(frozen=True, slots=True)
class CudaVmmTransfer:
    """Identify an immutable CUDA allocation and the spans of one view in it.

    `storage_offsets_bytes` gives, in order, the byte offset of each
    first-axis span of the view within the exported allocation, which is
    `storage_size_bytes` long. Every span shares `tensor_stride`, in elements
    with one entry per axis. `span_lengths` and `span_counts` run-length
    encode the spans' first-axis lengths: ``span_counts[i]`` consecutive spans
    each hold ``span_lengths[i]`` rows, keeping page maps compact without
    changing logical coverage. `Locator.__post_init__` checks that the
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

    def __post_init__(self) -> None:
        """Validate native CUDA handles and the declared allocation bounds."""
        if (
            not self.endpoint
            or len(self.export_id) != 32
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
            # A 64-byte CUDA IPC event handle fences the export for
            # consumers on the producer's host. A producing rank with any
            # consumer on another host publishes no fence, because an event
            # handle does not reach there; it synchronizes its stream before
            # publishing instead. Any other length is not a handle a consumer
            # could import.
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

    Shared storage names a segment in one host's namespace, so a product whose
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
    """One physical shard or replica of a logical tensor.

    `shape` and `offset` place this view as a box inside the owning
    `TensorTransfer`'s logical shape, in elements with one entry per axis;
    `nbytes` is the view's size and `dtype` its element type name.
    `transport` is the handle a consumer opens and `source` the publishing
    rank. Except for a `ChannelTransfer`, whose payload is the bytes, the
    publishing rank's transport keeps the storage until the export
    retires after the engine releases it.
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
        """Return the name of the transport that owns this export.

        The names are the `WorkerInfo.transfer_backends` vocabulary and key a
        rank's transport table. A `PosixShmTransfer` is ``"shm"`` here but
        ``"posix_shm"`` in the wire mapping.
        """
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
        """Validate the view's bounds and, for CUDA VMM, its span layout.

        Placement inside the logical tensor is checked by
        `TensorTransfer.__post_init__`, which sees the logical shape.
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
        """Parse a locator from the flattened mapping `to_mapping` writes.

        The ``transport`` key selects the handle type (``local``,
        ``channel``, ``posix_shm``, or ``cuda_vmm``), and the handle's fields
        sit beside it. Each constructed record validates itself.
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
                export_id=_str(data.get("export_id"), f"{where}.export_id"),
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
        """Encode the view and its handle as one flattened wire mapping."""
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
                export_id=transport.export_id,
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
    size bound; native output commit and `Batch.validate` check it.
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
