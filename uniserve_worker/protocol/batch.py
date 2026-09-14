"""Scheduler-to-worker execution records and their validation."""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from typing import Any, TypeAlias, TypeVar, cast

from ..foundation.errors import invalid_descriptor

MAX_TRANSFER_HANDLE_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class WorkerEndpoint:
    """A rank incarnation and its actual host address space.

    Worker and rank names survive restarts. Incarnation identifies this loaded
    rank; address_space identifies its process, independently of Worker grouping.
    Backend publication addresses and storage generations remain in Locator.
    """

    worker_id: str
    rank: int
    node: str
    address_space: str
    incarnation: str

    def __post_init__(self) -> None:
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
        """Identify a new rank in this process, including after a process fork."""

        import socket

        return cls(worker_id, rank, socket.gethostname(), _address_space, uuid.uuid4().hex)

    @classmethod
    def from_mapping(cls, value: object, where: str = "endpoint") -> WorkerEndpoint:
        data = _map(value, where)
        return cls(
            worker_id=_str(data.get("worker_id"), f"{where}.worker_id"),
            rank=_uint(data.get("rank"), f"{where}.rank"),
            node=_str(data.get("node"), f"{where}.node"),
            address_space=_str(data.get("address_space"), f"{where}.address_space"),
            incarnation=_str(data.get("incarnation"), f"{where}.incarnation"),
        )

    def to_mapping(self) -> dict[str, object]:
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
    """Identifies shared-memory storage and the endpoint that grants ready reads."""

    endpoint: str
    name: str

    def __post_init__(self) -> None:
        """Validate the shared-memory name and publishing endpoint."""

        if not self.endpoint or not self.name:
            raise invalid_descriptor("shared-memory transfer handle is invalid")


@dataclass(frozen=True, slots=True)
class CudaIpcTransfer:
    """Identify an immutable CUDA allocation and its reader-lease endpoint.

    Byte offsets locate ordered first-axis spans sharing the tensor strides.
    Lengths and counts encode consecutive runs of equally sized spans, keeping
    page maps compact without changing logical coverage or physical ownership.
    """

    endpoint: str
    publication_id: str
    storage_handle: bytes
    storage_size_bytes: int
    storage_offsets_bytes: tuple[int, ...]
    span_lengths: tuple[int, ...]
    span_counts: tuple[int, ...]
    tensor_stride: tuple[int, ...]
    ready_event_handle: bytes

    def __post_init__(self) -> None:
        """Validate native CUDA handles and the declared allocation bounds."""

        if (
            not self.endpoint
            or len(self.publication_id) != 32
            or len(self.storage_handle) != 64
            or self.storage_size_bytes < 1
            or not self.storage_offsets_bytes
            or len(self.span_counts) != len(self.span_lengths)
            or sum(self.span_counts) != len(self.storage_offsets_bytes)
            or any(count < 1 for count in self.span_counts)
            or any(
                not 0 <= offset < self.storage_size_bytes for offset in self.storage_offsets_bytes
            )
            or any(length < 1 for length in self.span_lengths)
            or any(stride < 0 for stride in self.tensor_stride)
            or len(self.ready_event_handle) != 64
        ):
            raise invalid_descriptor("CUDA IPC transfer handle is incomplete")


TransferTransport: TypeAlias = LocalTransfer | PosixShmTransfer | CudaIpcTransfer


@dataclass(frozen=True, slots=True)
class Locator:
    """Describes a typed tensor view and the transport-specific handle that owns its storage."""

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
        return "cuda_ipc"

    def __post_init__(self) -> None:
        """Validate tensor geometry and ensure the handle matches its transport kind."""

        if (
            self.nbytes < 1
            or not self.dtype
            or not self.shape
            or any(extent < 1 for extent in self.shape)
            or not self.device
            or len(self.offset) != len(self.shape)
            or any(start < 0 for start in self.offset)
        ):
            raise invalid_descriptor("transfer locator has invalid tensor bounds")
        if isinstance(self.transport, CudaIpcTransfer) and (
            len(self.transport.tensor_stride) != len(self.shape)
            or sum(
                length * count
                for length, count in zip(
                    self.transport.span_lengths, self.transport.span_counts, strict=True
                )
            )
            != self.shape[0]
        ):
            raise invalid_descriptor("CUDA IPC physical spans do not match its shape")

    @classmethod
    def from_mapping(cls, value: object, where: str = "transfer locator") -> Locator:
        """Parse a tensor locator and validate its transport handle, shape, dtype, and byte bounds."""

        data = _map(value, where)
        kind = _str(data.get("transport"), f"{where}.transport")
        if kind == "local":
            transport: TransferTransport = LocalTransfer(
                endpoint=_str(data.get("endpoint"), f"{where}.endpoint"),
                key=_uint(data.get("key"), f"{where}.key"),
            )
        elif kind == "posix_shm":
            transport = PosixShmTransfer(
                endpoint=_str(data.get("endpoint"), f"{where}.endpoint"),
                name=_str(data.get("name"), f"{where}.name"),
            )
        elif kind == "cuda_ipc":
            transport = CudaIpcTransfer(
                endpoint=_str(data.get("endpoint"), f"{where}.endpoint"),
                publication_id=_str(data.get("publication_id"), f"{where}.publication_id"),
                storage_handle=_bytes(data.get("storage_handle"), f"{where}.storage_handle"),
                storage_size_bytes=_uint(
                    data.get("storage_size_bytes"), f"{where}.storage_size_bytes"
                ),
                storage_offsets_bytes=tuple(
                    _ints(data.get("storage_offsets_bytes"), f"{where}.storage_offsets_bytes")
                ),
                span_lengths=tuple(_ints(data.get("span_lengths"), f"{where}.span_lengths")),
                span_counts=tuple(_ints(data.get("span_counts"), f"{where}.span_counts")),
                tensor_stride=tuple(_ints(data.get("tensor_stride"), f"{where}.tensor_stride")),
                ready_event_handle=_bytes(
                    data.get("ready_event_handle"), f"{where}.ready_event_handle"
                ),
            )
        else:
            raise invalid_descriptor(f"{where}.transport is invalid")
        return cls(
            source=WorkerEndpoint.from_mapping(data.get("source"), f"{where}.source"),
            transport=transport,
            nbytes=_uint(data.get("nbytes"), f"{where}.nbytes"),
            dtype=_str(data.get("dtype"), f"{where}.dtype"),
            shape=tuple(_uints(data.get("shape"), f"{where}.shape")),
            offset=tuple(_uints(data.get("offset"), f"{where}.offset")),
            device=_str(data.get("device"), f"{where}.device"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode the tensor geometry and transport-specific handle as a wire mapping."""

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
            output.update(transport="local", endpoint=transport.endpoint, key=transport.key)
        elif isinstance(transport, PosixShmTransfer):
            output.update(
                transport="posix_shm",
                endpoint=transport.endpoint,
                name=transport.name,
            )
        else:
            output.update(
                transport="cuda_ipc",
                endpoint=transport.endpoint,
                publication_id=transport.publication_id,
                storage_handle=transport.storage_handle,
                storage_size_bytes=transport.storage_size_bytes,
                storage_offsets_bytes=list(transport.storage_offsets_bytes),
                span_lengths=list(transport.span_lengths),
                span_counts=list(transport.span_counts),
                tensor_stride=list(transport.tensor_stride),
                ready_event_handle=transport.ready_event_handle,
            )
        return output


@dataclass(frozen=True, slots=True)
class TensorTransfer:
    """Actual logical tensor geometry and its immutable physical locations.

    Locations can be shards or equivalent replicas. They need not cover the
    whole value until a consumer binds its required region. All coordinates are
    in logical element order; a backend's native strides describe physical order.
    """

    shape: tuple[int, ...]
    locations: tuple[Locator, ...]

    def __post_init__(self) -> None:
        if not self.shape or any(extent < 1 for extent in self.shape) or not self.locations:
            raise invalid_descriptor("tensor transfer has no geometry or locations")
        first = self.locations[0]
        elements = math.prod(first.shape)
        if first.nbytes % elements or first.nbytes < elements:
            raise invalid_descriptor("tensor transfer has an invalid element size")
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
    def from_mapping(cls, value: object, where: str = "tensor transfer") -> TensorTransfer:
        data = _map(value, where)
        return cls(
            shape=tuple(_uints(data.get("shape"), f"{where}.shape")),
            locations=tuple(
                Locator.from_mapping(item, f"{where}.locations[{index}]")
                for index, item in enumerate(_seq(data.get("locations"), f"{where}.locations"))
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "locations": [location.to_mapping() for location in self.locations],
        }


@dataclass(frozen=True, slots=True)
class EncoderTransferValue:
    """Describes the media geometry, payload encoding, and location of transferred encoder features."""

    height: int
    width: int
    payload_kind: str
    tensor: TensorTransfer


@dataclass(frozen=True, slots=True)
class DeviceProductTransferValue:
    """Describes the media geometry, numeric range, and location of a transferred device product."""

    height: int
    width: int
    value_range: str
    tensor: TensorTransfer


@dataclass(frozen=True, slots=True)
class KvTransfer:
    """Describes a versioned KV extent, its page locators, and source-to-destination buffer relation."""

    tensors: tuple[TensorTransfer, ...]
    source: BufferId
    destination: str
    base: BufferId | None
    base_extent: int
    published_extent: int
    group_id: int
    compute_dtype: str
    page_size: int

    def __post_init__(self) -> None:
        """Validate the exact KV source, installed base and represented extent."""

        if not self.destination or self.base_extent < 0 or self.published_extent < self.base_extent:
            raise invalid_descriptor("KV publication extent or destination is invalid")
        if self.base is None and self.base_extent != 0:
            raise invalid_descriptor("KV publication base identity disagrees with its extent")
        if (
            self.group_id < 0
            or self.page_size < 1
            or self.compute_dtype not in {"float16", "bfloat16", "float32", "float64"}
        ):
            raise invalid_descriptor("KV publication storage identity is invalid")
        suffix = self.published_extent - self.base_extent
        if not suffix:
            if self.tensors:
                raise invalid_descriptor("empty KV suffix carries physical tensors")
            return
        if len(self.tensors) not in {2, 3}:
            raise invalid_descriptor("KV publication requires raw keys, values and optional scales")
        key, value = self.tensors[:2]
        if (
            len(key.shape) != 4
            or key.shape[0] != suffix
            or value.shape != key.shape
            or value.dtype != key.dtype
            or key.dtype not in {"float16", "bfloat16", "float32", "float64", "float8_e4m3fn"}
        ):
            raise invalid_descriptor("KV publication has invalid raw token/layer/head geometry")
        quantized = key.dtype == "float8_e4m3fn"
        if (len(self.tensors) == 3) != quantized:
            raise invalid_descriptor("KV publication scale presence disagrees with its storage")
        if quantized:
            scales = self.tensors[2]
            pages = (
                self.base_extent % self.page_size + suffix + self.page_size - 1
            ) // self.page_size
            if (
                scales.dtype != "float32"
                or len(scales.shape) != 4
                or scales.shape[:3] != (pages, 2, key.shape[1])
                or key.shape[2] % scales.shape[3]
            ):
                raise invalid_descriptor("KV publication scales disagree with its source pages")

    @property
    def scale_head_size(self) -> int:
        """Heads sharing each source scale; zero denotes unquantized storage.

        Scale tensors use [page, K/V, layer, head group]. Groups partition the
        logical head axis uniformly; each locator identifies the producer's
        actual group, including replicated full-cache representations.
        """

        return self.tensors[0].shape[2] // self.tensors[2].shape[3] if len(self.tensors) == 3 else 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "kv_transfer") -> KvTransfer:
        """Decode source identity and physical cache representation."""

        data = _map(value, where)
        raw_base = data.get("base")
        return cls(
            tensors=tuple(
                TensorTransfer.from_mapping(item, f"{where}.tensors[{index}]")
                for index, item in enumerate(_seq(data.get("tensors"), f"{where}.tensors"))
            ),
            source=BufferId.from_mapping(data.get("source"), f"{where}.source"),
            destination=_str(data.get("destination"), f"{where}.destination"),
            base=(None if raw_base is None else BufferId.from_mapping(raw_base, f"{where}.base")),
            base_extent=_uint(data.get("base_extent", 0), f"{where}.base_extent"),
            published_extent=_uint(
                data.get("published_extent", 0),
                f"{where}.published_extent",
            ),
            group_id=_uint(data.get("group_id", 0), f"{where}.group_id"),
            compute_dtype=_str(data.get("compute_dtype"), f"{where}.compute_dtype"),
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
            raise invalid_descriptor("KV transfer exceeds its descriptor byte bound")
        return size


@dataclass(frozen=True, slots=True)
class LatentTransferValue:
    """Describes a denoising step and media geometry for transferred latent storage."""

    height: int
    width: int
    latent_units: int
    step: int
    tensor: TensorTransfer


TransferValue: TypeAlias = EncoderTransferValue | DeviceProductTransferValue | LatentTransferValue


def _tensor_transfers_size(tensors: tuple[TensorTransfer, ...]) -> int:
    locators = tuple(location for tensor in tensors for location in tensor.locations)
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
            size += len(transport.endpoint.encode()) + len(transport.name.encode()) + 16
        else:
            size += (
                len(transport.endpoint.encode())
                + len(transport.publication_id.encode())
                + len(transport.storage_handle)
                + len(transport.ready_event_handle)
                + 8 * len(transport.tensor_stride)
                + 8 * len(transport.storage_offsets_bytes)
                + 12 * len(transport.span_lengths)
                + 64
            )
    return size


class ForwardMode(StrEnum):
    """The numerical mode of an autoregressive or mixed model forward."""

    PREFILL = "prefill"
    DECODE = "decode"
    VERIFY = "verify"


class PipelineStage(StrEnum):
    """A concrete encoder, diffusion, decoder, or media-output computation."""

    VISION_ENCODING = "vision_encoding"
    LATENT_ENCODING = "latent_encoding"
    TEXT_ENCODING = "text_encoding"
    LATENT_PREPARATION = "latent_preparation"
    DENOISING = "denoising"
    IMAGE_DECODING = "image_decoding"
    VIDEO_DECODING = "video_decoding"
    AUDIO_DECODING = "audio_decoding"
    VIDEO_ENCODING = "video_encoding"
    AUDIO_ENCODING = "audio_encoding"
    MUXING = "muxing"


class TransferMode(StrEnum):
    """The storage action between a concrete producer and consumer."""

    TENSOR = "tensor"
    KV_PUBLISH = "kv_publish"
    KV_INSTALL = "kv_install"


Computation: TypeAlias = ForwardMode | PipelineStage | TransferMode

VIDEO_STAGES = (
    PipelineStage.TEXT_ENCODING,
    PipelineStage.LATENT_PREPARATION,
    PipelineStage.DENOISING,
    PipelineStage.VIDEO_DECODING,
    PipelineStage.AUDIO_DECODING,
    PipelineStage.VIDEO_ENCODING,
    PipelineStage.AUDIO_ENCODING,
    PipelineStage.MUXING,
)

COMPUTATIONS: tuple[Computation, ...] = (
    ForwardMode.PREFILL,
    ForwardMode.DECODE,
    ForwardMode.VERIFY,
    *PipelineStage,
    *TransferMode,
)


def computation(value: object, where: str) -> Computation:
    """Decode one concrete computation, excluding mixed model-batch metadata."""

    if isinstance(value, (ForwardMode, PipelineStage, TransferMode)) and value in COMPUTATIONS:
        return value
    if type(value) is str:
        member = _COMPUTATION_BY_VALUE.get(value)
        if member is not None:
            return member
    raise invalid_descriptor(f"{where} is not a supported computation")


class DType(StrEnum):
    """Defines wire-stable scalar dtypes supported by scheduler descriptors."""

    U8 = "u8"
    I32 = "i32"
    I16 = "i16"
    I64 = "i64"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"

    @property
    def element_bytes(self) -> int:
        """Width of one scalar in the logical representation."""

        return {
            DType.U8: 1,
            DType.I32: 4,
            DType.I16: 2,
            DType.I64: 8,
            DType.F16: 2,
            DType.BF16: 2,
            DType.F32: 4,
        }[self]


class OpStatus(StrEnum):
    """Classifies an operation result as successful, predicated away, or failed."""

    OK = "ok"
    PREDICATED = "predicated"
    ERROR = "error"


class ErrorCode(StrEnum):
    """Classifies bounded execution failures returned to the scheduler."""

    INVALID_OPERATION = "invalid_operation"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    COMPUTE_ERROR = "compute_error"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class DrawLayout(StrEnum):
    """Assigns deterministic RNG coordinates to sampling, speculation, and flow noise."""

    TARGET_SAMPLING = "target_sampling"
    SPECULATIVE_PROPOSAL = "speculative_proposal"
    FLOW_NOISE = "flow_noise"


_STATE_ADVANCING_WORK = frozenset(
    {
        ForwardMode.PREFILL,
        ForwardMode.DECODE,
        ForwardMode.VERIFY,
        PipelineStage.LATENT_PREPARATION,
        PipelineStage.DENOISING,
    }
)

_COMPUTATION_BY_VALUE = {member.value: member for member in COMPUTATIONS}


def native_run(
    batch_id: int,
    run_id: int,
    collective_seq: int,
    operations: tuple[ScheduledRequest, ...],
    block_tables: tuple[BlockTable, ...],
    new_cache_pages: tuple[CachePageAllocation, ...],
    forward_inputs: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[bool, ...]
    ],
    latent_params: Sequence[object],
    decode_ranges: Sequence[object],
    buffer_allocations: Sequence[object],
    commands: tuple[BatchCommand, ...],
    input_products: Sequence[object],
    kv_inputs: Sequence[object],
) -> ScheduleBatch:
    """Assemble a physical run from transport-constructed members."""

    run = object.__new__(ScheduleBatch)
    set_field = object.__setattr__
    set_field(run, "batch_id", batch_id)
    set_field(run, "run_id", run_id)
    set_field(run, "collective_seq", collective_seq)
    set_field(run, "operations", operations)
    set_field(run, "block_tables", block_tables)
    set_field(run, "new_cache_pages", new_cache_pages)
    set_field(run, "forward_operation_indices", forward_inputs[0])
    set_field(run, "request_pool_indices", forward_inputs[1])
    set_field(run, "seq_lens", forward_inputs[2])
    set_field(run, "query_lens", forward_inputs[3])
    set_field(run, "write_kv", forward_inputs[4])
    set_field(
        run,
        "latent_params",
        tuple(
            LatentParams.from_mapping(item, f"run.latent_params[{index}]")
            for index, item in enumerate(latent_params)
        ),
    )
    set_field(
        run,
        "decode_ranges",
        tuple(
            DecodeRange.from_mapping(item, f"run.decode_ranges[{index}]")
            for index, item in enumerate(decode_ranges)
        ),
    )
    set_field(
        run,
        "buffer_allocations",
        tuple(
            BufferAllocation.from_mapping(item, f"run.buffer_allocations[{index}]")
            for index, item in enumerate(buffer_allocations)
        ),
    )
    set_field(run, "commands", commands)
    set_field(
        run,
        "input_products",
        tuple(
            TensorPublication.from_mapping(item, f"run.input_products[{index}]")
            for index, item in enumerate(input_products)
        ),
    )
    set_field(run, "kv_inputs", tuple(KvTransfer.from_mapping(value) for value in kv_inputs))
    return run


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Defines sampling filters, token penalties, log-probability output, and terminal token rules."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    ignore_eos: bool = False
    seed: int | None = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: tuple[tuple[int, float], ...] = ()
    min_tokens: int = 0
    return_logprobs: bool = False
    n_logprobs: int = 0
    return_prompt_logprobs: bool = False
    n_prompt_logprobs: int = 0
    logprob_token_ids: tuple[int, ...] = ()
    bad_words_ids: tuple[tuple[int, ...], ...] = ()
    allowed_token_ids: tuple[int, ...] | None = None
    typical_p: float = 1.0
    forced_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate probability filters, penalties, token rules, and log-probability bounds."""

        for name in (
            "temperature",
            "top_p",
            "min_p",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise invalid_descriptor(f"sampling.{name} must be finite")
        if self.temperature < 0:
            raise invalid_descriptor("sampling.temperature must not be negative")
        if not 0 < self.top_p <= 1:
            raise invalid_descriptor("sampling.top_p must be in (0, 1]")
        if not 0 <= self.min_p <= 1:
            raise invalid_descriptor("sampling.min_p must be in [0, 1]")
        if not 0 < self.typical_p <= 1:
            raise invalid_descriptor("sampling.typical_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise invalid_descriptor("sampling.repetition_penalty must be positive")
        if any(not math.isfinite(float(bias)) for _, bias in self.logit_bias):
            raise invalid_descriptor("sampling.logit_bias values must be finite")
        if self.allowed_token_ids == ():
            raise invalid_descriptor("sampling.allowed_token_ids must not be empty when present")
        if any(not values for values in self.bad_words_ids):
            raise invalid_descriptor("sampling.bad_words_ids entries must not be empty")
        for name in ("top_k", "min_tokens", "n_logprobs", "n_prompt_logprobs"):
            _nonnegative(getattr(self, name), f"sampling.{name}")
        if self.seed is not None:
            _nonnegative(self.seed, "sampling.seed")

    @classmethod
    def from_mapping(cls, value: object, where: str = "sampling") -> SamplingParams:
        """Parse sampling controls while validating probabilities, penalties, token sets, and output bounds."""

        data = _map(value, where)
        return cls(
            temperature=_float(data.get("temperature", 0.0), f"{where}.temperature"),
            top_k=_uint(data.get("top_k", 0), f"{where}.top_k"),
            top_p=_float(data.get("top_p", 1.0), f"{where}.top_p"),
            ignore_eos=_bool(data.get("ignore_eos", False), f"{where}.ignore_eos"),
            seed=_optional_uint(data.get("seed"), f"{where}.seed"),
            min_p=_float(data.get("min_p", 0.0), f"{where}.min_p"),
            repetition_penalty=_float(
                data.get("repetition_penalty", 1.0), f"{where}.repetition_penalty"
            ),
            frequency_penalty=_float(
                data.get("frequency_penalty", 0.0), f"{where}.frequency_penalty"
            ),
            presence_penalty=_float(data.get("presence_penalty", 0.0), f"{where}.presence_penalty"),
            logit_bias=tuple(
                (
                    _uint(pair[0], f"{where}.logit_bias[{index}][0]"),
                    _float(pair[1], f"{where}.logit_bias[{index}][1]"),
                )
                for index, item in enumerate(
                    _seq(data.get("logit_bias", ()), f"{where}.logit_bias")
                )
                for pair in (_pair(item, f"{where}.logit_bias[{index}]"),)
            ),
            min_tokens=_uint(data.get("min_tokens", 0), f"{where}.min_tokens"),
            return_logprobs=_bool(data.get("return_logprobs", False), f"{where}.return_logprobs"),
            n_logprobs=_uint(data.get("n_logprobs", 0), f"{where}.n_logprobs"),
            return_prompt_logprobs=_bool(
                data.get("return_prompt_logprobs", False),
                f"{where}.return_prompt_logprobs",
            ),
            n_prompt_logprobs=_uint(data.get("n_prompt_logprobs", 0), f"{where}.n_prompt_logprobs"),
            logprob_token_ids=_uints(
                data.get("logprob_token_ids", ()), f"{where}.logprob_token_ids"
            ),
            bad_words_ids=tuple(
                _uints(item, f"{where}.bad_words_ids[{index}]")
                for index, item in enumerate(
                    _seq(data.get("bad_words_ids", ()), f"{where}.bad_words_ids")
                )
            ),
            allowed_token_ids=(
                None
                if data.get("allowed_token_ids") is None
                else _uints(data["allowed_token_ids"], f"{where}.allowed_token_ids")
            ),
            typical_p=_float(data.get("typical_p", 1.0), f"{where}.typical_p"),
            forced_token_ids=_uints(data.get("forced_token_ids", ()), f"{where}.forced_token_ids"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode all sampling filters, penalties, token rules, and log-probability controls."""

        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "ignore_eos": self.ignore_eos,
            "seed": self.seed,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "logit_bias": [list(value) for value in self.logit_bias],
            "min_tokens": self.min_tokens,
            "return_logprobs": self.return_logprobs,
            "n_logprobs": self.n_logprobs,
            "return_prompt_logprobs": self.return_prompt_logprobs,
            "n_prompt_logprobs": self.n_prompt_logprobs,
            "logprob_token_ids": list(self.logprob_token_ids),
            "bad_words_ids": [list(value) for value in self.bad_words_ids],
            "allowed_token_ids": (
                None if self.allowed_token_ids is None else list(self.allowed_token_ids)
            ),
            "typical_p": self.typical_p,
            "forced_token_ids": list(self.forced_token_ids),
        }


@dataclass(frozen=True, slots=True)
class ImageParams:
    """Defines source image bytes, format, dimensions, and preprocessing bounds for an encode request."""

    steps: int = 50
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.0
    cfg_renorm_type: str = "global"
    cfg_renorm_min: float = 0.0
    cfg_interval: tuple[float, float] = (0.0, 1.0)
    timestep_shift: float = 1.0
    height: int = 512
    width: int = 512
    seed: int | None = None
    negative_prompt: str = ""
    max_images: int = 1
    image_prompts: tuple[str, ...] = ()
    retain_images: bool = True

    def __post_init__(self) -> None:
        """Validate encoded image payload, dimensions, format, and preprocessing bounds."""

        if not 1 <= self.steps <= 1000:
            raise invalid_descriptor("image.steps must be in 1..=1000")
        for name in ("height", "width"):
            dimension = int(getattr(self, name))
            if not 16 <= dimension <= 4096 or dimension % 16:
                raise invalid_descriptor(f"image.{name} must be a multiple of 16 in 16..=4096")
        for name in ("cfg_text_scale", "cfg_img_scale"):
            scale = float(getattr(self, name))
            if not math.isfinite(scale) or not 0 <= scale <= 100:
                raise invalid_descriptor(f"image.{name} must be finite and in [0, 100]")
        for name, value in (
            ("cfg_renorm_min", self.cfg_renorm_min),
            ("cfg_interval[0]", self.cfg_interval[0]),
            ("cfg_interval[1]", self.cfg_interval[1]),
            ("timestep_shift", self.timestep_shift),
        ):
            if not math.isfinite(float(value)):
                raise invalid_descriptor(f"image.{name} must be finite")
        if self.cfg_interval[0] > self.cfg_interval[1]:
            raise invalid_descriptor("image.cfg_interval must be ordered")
        if not self.cfg_renorm_type.strip():
            raise invalid_descriptor("image.cfg_renorm_type must not be empty")
        if not 1 <= self.max_images <= 256:
            raise invalid_descriptor("image.max_images must be in 1..=256")
        if self.seed is not None:
            _nonnegative(self.seed, "image.seed")

    @classmethod
    def from_mapping(cls, value: object, where: str = "image") -> ImageParams:
        """Parse encoded image input and validate format, dimensions, and processing bounds."""

        data = _map(value, where)
        interval = _pair(data.get("cfg_interval", (0.0, 1.0)), f"{where}.cfg_interval")
        return cls(
            steps=_uint(data.get("steps", 50), f"{where}.steps"),
            cfg_text_scale=_float(data.get("cfg_text_scale", 4.0), f"{where}.cfg_text_scale"),
            cfg_img_scale=_float(data.get("cfg_img_scale", 1.0), f"{where}.cfg_img_scale"),
            cfg_renorm_type=_str(data.get("cfg_renorm_type", "global"), f"{where}.cfg_renorm_type"),
            cfg_renorm_min=_float(data.get("cfg_renorm_min", 0.0), f"{where}.cfg_renorm_min"),
            cfg_interval=(
                _float(interval[0], f"{where}.cfg_interval[0]"),
                _float(interval[1], f"{where}.cfg_interval[1]"),
            ),
            timestep_shift=_float(data.get("timestep_shift", 1.0), f"{where}.timestep_shift"),
            height=_uint(data.get("height", 512), f"{where}.height"),
            width=_uint(data.get("width", 512), f"{where}.width"),
            seed=_optional_uint(data.get("seed"), f"{where}.seed"),
            negative_prompt=_str(data.get("negative_prompt", ""), f"{where}.negative_prompt"),
            max_images=_uint(data.get("max_images", 1), f"{where}.max_images"),
            image_prompts=tuple(
                _str(item, f"{where}.image_prompts[{index}]")
                for index, item in enumerate(
                    _seq(data.get("image_prompts", ()), f"{where}.image_prompts")
                )
            ),
            retain_images=_bool(data.get("retain_images", True), f"{where}.retain_images"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize image-generation controls into their wire mapping."""

        return {
            "steps": self.steps,
            "cfg_text_scale": self.cfg_text_scale,
            "cfg_img_scale": self.cfg_img_scale,
            "cfg_renorm_type": self.cfg_renorm_type,
            "cfg_renorm_min": self.cfg_renorm_min,
            "cfg_interval": list(self.cfg_interval),
            "timestep_shift": self.timestep_shift,
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "max_images": self.max_images,
            "image_prompts": list(self.image_prompts),
            "retain_images": self.retain_images,
        }


@dataclass(frozen=True, slots=True, order=True)
class ComputationId:
    """Logical batch and selection ordinal, independent of physical worker packing."""

    batch_id: int
    request_index: int

    _hash_value: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0 <= self.batch_id <= 0xFFFFFFFFFFFFFFFF:
            raise invalid_descriptor("computation batch id is outside uint64")
        if not 0 <= self.request_index <= 0xFFFFFFFF:
            raise invalid_descriptor("computation request index is outside uint32")
        if self.batch_id == 0 and self.request_index != 0:
            raise invalid_descriptor("admission identity requires request index zero")

    def __hash__(self) -> int:
        """Reuse the hash of these immutable integer identity coordinates."""

        value = self._hash_value
        if value is None:
            value = hash((self.batch_id, self.request_index))
            object.__setattr__(self, "_hash_value", value)
        return value

    @classmethod
    def from_mapping(cls, value: object, where: str = "computation_id") -> ComputationId:
        if isinstance(value, cls):
            return value
        data = _map(value, where)
        return cls(
            batch_id=_uint(data.get("batch_id"), f"{where}.batch_id"),
            request_index=_uint(data.get("request_index"), f"{where}.request_index"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"batch_id": self.batch_id, "request_index": self.request_index}


@dataclass(frozen=True, slots=True)
class RequestKey:
    """Identifies one request epoch within an engine instance."""

    engine_id: int
    request_id: int
    request_epoch: int

    _hash_value: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the non-negative request identifier and epoch."""

        _nonnegative(self.engine_id, "request_key.engine_id")
        _nonnegative(self.request_id, "request_key.request_id")
        _nonnegative(self.request_epoch, "request_key.request_epoch")

    def __hash__(self) -> int:
        """Reuse the hash of these immutable integer identity coordinates."""

        value = self._hash_value
        if value is None:
            value = hash((self.engine_id, self.request_id, self.request_epoch))
            object.__setattr__(self, "_hash_value", value)
        return value

    @classmethod
    def from_mapping(cls, value: object, where: str = "request_key") -> RequestKey:
        """Parse and validate an engine instance, request id, and admission epoch."""

        key = _fast_request_key(value)
        if key is not None:
            return key
        data = _map(value, where)
        return cls(
            engine_id=_uint(data.get("engine_id"), f"{where}.engine_id"),
            request_id=_uint(data.get("request_id"), f"{where}.request_id"),
            request_epoch=_uint(data.get("request_epoch"), f"{where}.request_epoch"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the complete request-generation identity for IPC."""

        return {
            "engine_id": self.engine_id,
            "request_id": self.request_id,
            "request_epoch": self.request_epoch,
        }


@dataclass(frozen=True, slots=True)
class StaticDim:
    """Fixes one tensor dimension to an exact extent."""

    extent: int


@dataclass(frozen=True, slots=True)
class DeviceDim:
    """Bounds one tensor dimension whose live extent is selected on the device."""

    bound: int


DimBound: TypeAlias = StaticDim | DeviceDim


@dataclass(frozen=True, slots=True)
class ShapeBound:
    """Defines the maximum physical tensor shape allowed for a product."""

    dims: tuple[DimBound, ...] = ()

    def __post_init__(self) -> None:
        """Normalize dimensions and reject empty or non-positive shape bounds."""

        device_dims = sum(1 for dim in self.dims if isinstance(dim, DeviceDim))
        if device_dims > 1:
            raise invalid_descriptor("a shape bound carries more than one device-actual dimension")
        if any((dim.extent if isinstance(dim, StaticDim) else dim.bound) < 1 for dim in self.dims):
            raise invalid_descriptor("a shape bound contains a zero extent")

    @property
    def max_elements(self) -> int:
        """Multiply static extents and the maximum device-selected extent."""

        elements = 1
        for dim in self.dims:
            elements *= dim.extent if isinstance(dim, StaticDim) else dim.bound
        return elements

    def contains_shape(self, shape: tuple[int, ...]) -> bool:
        """Check tensor geometry; a single dynamic dimension denotes flat capacity."""

        if any(extent < 1 for extent in shape):
            return False
        elements = math.prod(shape)
        if not self.dims:
            return elements == 1
        if len(self.dims) == 1 and isinstance(self.dims[0], DeviceDim):
            return elements <= self.dims[0].bound
        return len(shape) == len(self.dims) and all(
            extent == bound.extent if isinstance(bound, StaticDim) else extent <= bound.bound
            for extent, bound in zip(shape, self.dims, strict=True)
        )

    @classmethod
    def from_mapping(cls, value: object, where: str = "shape_bound") -> ShapeBound:
        """Parse static and device-selected dimension bounds from the wire schema."""

        data = _map(value, where)
        dims: list[DimBound] = []
        for index, item in enumerate(_seq(data.get("dims", ()), f"{where}.dims")):
            kind, payload = _tagged(item, f"{where}.dims[{index}]")
            if kind == "static":
                dims.append(StaticDim(_uint(payload, f"{where}.dims[{index}].value")))
            elif kind == "device":
                inner = _map(payload, f"{where}.dims[{index}].value")
                dims.append(DeviceDim(_uint(inner.get("max"), f"{where}.dims[{index}].value.max")))
            else:
                raise invalid_descriptor(f"{where}.dims[{index}] has unknown variant {kind!r}")
        return cls(tuple(dims))

    def to_mapping(self) -> dict[str, object]:
        """Serialize ordered dimension bounds into tagged wire variants."""

        return {"dims": [_dim_to_mapping(dim) for dim in self.dims]}


def _dim_to_mapping(dim: DimBound) -> dict[str, object]:
    """Encode a static or symbolic dimension bound for the wire format."""

    if isinstance(dim, StaticDim):
        return {"kind": "static", "value": dim.extent}
    return {"kind": "device", "value": {"max": dim.bound}}


@dataclass(frozen=True, slots=True)
class OutputInfo:
    """Name and bounded representation of an entry result before request binding."""

    name: str
    dtype: DType
    shape_bound: ShapeBound

    def __post_init__(self) -> None:
        if not self.name:
            raise invalid_descriptor("tensor result must have a name")

    @property
    def max_bytes(self) -> int:
        """Maximum physical storage required by this Tensor result."""

        return self.shape_bound.max_elements * self.dtype.element_bytes

    @classmethod
    def from_mapping(cls, value: object, where: str = "output_info") -> OutputInfo:
        data = _map(value, where)
        return cls(
            name=_str(data.get("name"), f"{where}.name"),
            dtype=DType(_str(data.get("dtype"), f"{where}.dtype")),
            shape_bound=ShapeBound.from_mapping(data.get("shape_bound"), f"{where}.shape_bound"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "shape_bound": self.shape_bound.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class TensorRef:
    """Identifies tensor storage independently of its role in a computation."""

    request_key: RequestKey
    producer_op_id: ComputationId
    output_index: int
    generation: int
    dtype: DType
    shape_bound: ShapeBound

    _buffer_id: BufferId | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate allocation generation and bounded tensor capacity."""

        if self.generation < 1:
            raise invalid_descriptor("product reference has no logical generation")
        self.shape_bound.__post_init__()

    @property
    def buffer_id(self) -> BufferId:
        """Borrow the immutable storage identity shared by this reference's consumers."""

        identity = self._buffer_id
        if identity is None:
            identity = BufferId(
                owner=self.request_key,
                producer_op_id=self.producer_op_id,
                output_index=self.output_index,
                generation=self.generation,
            )
            object.__setattr__(self, "_buffer_id", identity)
        return identity

    @property
    def max_bytes(self) -> int:
        """Return the maximum physical bytes allowed by this product’s shape and dtype."""

        return self.shape_bound.max_elements * self.dtype.element_bytes

    @classmethod
    def from_mapping(cls, value: object, where: str = "tensor_ref") -> TensorRef:
        """Parse and validate a typed logical product and its storage bounds."""

        reference = _fast_tensor_ref(value)
        if reference is not None:
            return reference
        data = _map(value, where)
        return cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            producer_op_id=ComputationId.from_mapping(
                data.get("producer_op_id"), f"{where}.producer_op_id"
            ),
            output_index=_uint(data.get("output_index"), f"{where}.output_index"),
            generation=_uint(data.get("generation"), f"{where}.generation"),
            dtype=_enum(DType, data.get("dtype"), f"{where}.dtype"),
            shape_bound=ShapeBound.from_mapping(data.get("shape_bound"), f"{where}.shape_bound"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the complete logical product contract for IPC."""

        return {
            "request_key": self.request_key.to_mapping(),
            "producer_op_id": self.producer_op_id.to_mapping(),
            "output_index": self.output_index,
            "generation": self.generation,
            "dtype": self.dtype.value,
            "shape_bound": self.shape_bound.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class BufferId:
    """Identifies a versioned operation output buffer and its owning request."""

    owner: RequestKey
    producer_op_id: ComputationId
    output_index: int
    generation: int

    _hash_value: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the owning request, operation identifier, and version."""

        if self.generation < 1:
            raise invalid_descriptor("buffer id has no logical generation")

    def __hash__(self) -> int:
        """Reuse the hash of these immutable integer identity coordinates."""

        value = self._hash_value
        if value is None:
            value = hash((self.owner, self.producer_op_id, self.output_index, self.generation))
            object.__setattr__(self, "_hash_value", value)
        return value

    @classmethod
    def from_mapping(cls, value: object, where: str = "buffer_id") -> BufferId:
        """Parse a versioned persistent-buffer identity from the wire schema."""

        data = _map(value, where)
        return cls(
            owner=RequestKey.from_mapping(data.get("owner"), f"{where}.owner"),
            producer_op_id=ComputationId.from_mapping(
                data.get("producer_op_id"), f"{where}.producer_op_id"
            ),
            output_index=_uint(data.get("output_index"), f"{where}.output_index"),
            generation=_uint(data.get("generation"), f"{where}.generation"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize persistent-buffer ownership and generation fields for IPC."""

        return {
            "owner": self.owner.to_mapping(),
            "producer_op_id": self.producer_op_id.to_mapping(),
            "output_index": self.output_index,
            "generation": self.generation,
        }


@dataclass(frozen=True, slots=True)
class Bounds:
    """Caps tokens, pages, latent bytes, completion bytes, and transfer bytes for one operation."""

    max_tokens: int = 0
    max_kv_pages: int = 0
    max_latent_bytes: int = 0
    max_completion_bytes: int = 0
    max_transfer_bytes: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "bounds") -> Bounds:
        """Parse scheduler-enforced resource ceilings for one operation."""

        bounds = _fast_bounds(value)
        if bounds is not None:
            return bounds
        data = _map(value, where)
        return cls(
            max_tokens=_uint(data.get("max_tokens"), f"{where}.max_tokens"),
            max_kv_pages=_uint(data.get("max_kv_pages"), f"{where}.max_kv_pages"),
            max_latent_bytes=_uint(data.get("max_latent_bytes"), f"{where}.max_latent_bytes"),
            max_completion_bytes=_uint(
                data.get("max_completion_bytes"), f"{where}.max_completion_bytes"
            ),
            max_transfer_bytes=_uint(data.get("max_transfer_bytes"), f"{where}.max_transfer_bytes"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize all operation resource ceilings for IPC."""

        return {
            "max_tokens": self.max_tokens,
            "max_kv_pages": self.max_kv_pages,
            "max_latent_bytes": self.max_latent_bytes,
            "max_completion_bytes": self.max_completion_bytes,
            "max_transfer_bytes": self.max_transfer_bytes,
        }


@dataclass(frozen=True, slots=True)
class Rng:
    """Defines the seed, semantic offset, and draw layout for deterministic random values."""

    seed: int
    semantic_index_base: int
    draw_layout: DrawLayout

    @classmethod
    def from_mapping(cls, value: object, where: str = "rng") -> Rng:
        """Parse deterministic random seed, semantic offset, and draw layout."""

        rng = _fast_rng(value)
        if rng is not None:
            return rng
        data = _map(value, where)
        return cls(
            seed=_uint(data.get("seed"), f"{where}.seed"),
            semantic_index_base=_uint(
                data.get("semantic_index_base"), f"{where}.semantic_index_base"
            ),
            draw_layout=_enum(DrawLayout, data.get("draw_layout"), f"{where}.draw_layout"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize deterministic RNG coordinates for IPC."""

        return {
            "seed": self.seed,
            "semantic_index_base": self.semantic_index_base,
            "draw_layout": self.draw_layout.value,
        }


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """One computation with its request identity, dependencies, and output limits."""

    request_key: RequestKey
    op_id: ComputationId
    predecessor: ComputationId | None
    kind: Computation
    bounds: Bounds
    entry: str = "model"
    inputs: tuple[TensorRef, ...] = ()
    outputs: tuple[TensorRef, ...] = ()
    token_input: TensorRef | None = None
    token_output: TensorRef | None = None
    vision_input: TensorRef | None = None
    latent_feature_input: TensorRef | None = None
    encoder_output: TensorRef | None = None
    latent_input: TensorRef | None = None
    latent_output: TensorRef | None = None
    image_input: TensorRef | None = None
    image_output: TensorRef | None = None
    completion_output: TensorRef | None = None
    transition_output: TensorRef | None = None
    predicate: TensorRef | None = None
    rng: Rng | None = None
    sampling_state: SamplingState | None = None
    input_token_ids: tuple[int, ...] = ()
    input_image: str | None = None
    kv_input: BufferId | None = None
    kv_output: BufferId | None = None

    def tensor_inputs(self) -> tuple[TensorRef, ...]:
        """Return tensor inputs from the computation signature, excluding its predicate."""

        return (
            *self.inputs,
            *(
                value
                for value in (
                    self.token_input,
                    self.vision_input,
                    self.latent_feature_input,
                    self.latent_input,
                    self.image_input,
                )
                if value is not None
            ),
        )

    def tensor_outputs(self) -> tuple[TensorRef, ...]:
        """Return every tensor declaration owned by this computation."""

        return (
            *self.outputs,
            *(
                value
                for value in (
                    self.token_output,
                    self.completion_output,
                    self.transition_output,
                    self.encoder_output,
                    self.latent_output,
                    self.image_output,
                )
                if value is not None
            ),
        )

    def buffer_inputs(self) -> tuple[TensorRef, ...]:
        """Return inputs that require persistent destination storage."""

        return (
            *self.inputs,
            *(
                value
                for value in (self.vision_input, self.latent_feature_input, self.image_input)
                if value is not None
            ),
        )

    def buffer_outputs(self) -> tuple[TensorRef, ...]:
        """Return outputs backed by scheduler-allocated persistent buffers."""

        return (
            *self.outputs,
            *(
                value
                for value in (
                    self.encoder_output,
                    self.image_output,
                )
                if value is not None
            ),
        )

    @property
    def advances_state(self) -> bool:
        """Indicate whether this operation advances accepted request progress."""

        return self.kind in _STATE_ADVANCING_WORK

    def validate(self) -> None:
        """Enforce operation-family, predecessor, bound, dataflow, predicate, and RNG invariants."""

        if self.op_id.batch_id < 1:
            raise invalid_descriptor("operation id must be positive")
        if not isinstance(self.entry, str) or not self.entry:
            raise invalid_descriptor("operation entry must not be empty")
        if self.kind not in COMPUTATIONS:
            raise invalid_descriptor("operation requires a valid computation tag")
        if self.predecessor is None:
            if (
                self.advances_state
                or self.kind is TransferMode.KV_INSTALL
                or self.latent_input is not None
            ):
                raise invalid_descriptor("state-changing operation requires a predecessor")
        if self.predecessor is not None and not self.predecessor < self.op_id:
            raise invalid_descriptor("predecessor must precede operation")
        if len(self.input_token_ids) > self.bounds.max_tokens:
            raise invalid_descriptor("input token count exceeds the computation token bound")
        if any(token < 0 or token > 0xFFFFFFFF for token in self.input_token_ids):
            raise invalid_descriptor("input token id is outside uint32")
        if self.sampling_state is not None:
            self.sampling_state.validate()
        if self.input_image is not None and (
            not isinstance(self.input_image, str)
            or not self.input_image
            or self.kind not in {PipelineStage.VISION_ENCODING, PipelineStage.LATENT_ENCODING}
            or self.image_input is not None
        ):
            raise invalid_descriptor(
                "encoded image requires an image encoder without another image source"
            )
        output_indices: set[int] = set()
        publishes_kv = self.kind in {TransferMode.KV_PUBLISH, TransferMode.KV_INSTALL}
        if (self.kv_output is not None) != publishes_kv:
            raise invalid_descriptor(
                "KV publication or installation requires one cache output identity"
            )
        if self.kv_output is not None:
            if (
                self.kv_output.owner != self.request_key
                or self.kv_output.producer_op_id != self.op_id
            ):
                raise invalid_descriptor("KV output is not owned by its producing computation")
            output_indices.add(self.kv_output.output_index)
        if self.kv_input is not None and (
            self.kv_input.owner != self.request_key
            or self.kind
            not in {
                TransferMode.KV_INSTALL,
                PipelineStage.LATENT_PREPARATION,
                PipelineStage.DENOISING,
            }
        ):
            raise invalid_descriptor("KV input is incompatible with its computation or request")
        if self.kind is TransferMode.KV_INSTALL and self.kv_input is None:
            raise invalid_descriptor("KV installation requires a source publication")

        for product in self.tensor_outputs():
            if product.request_key != self.request_key or product.producer_op_id != self.op_id:
                raise invalid_descriptor(
                    "an output product is not owned by its producing operation"
                )
            if product.generation < 1:
                raise invalid_descriptor("an output product has no logical generation")
            if product.output_index in output_indices:
                raise invalid_descriptor("operation repeats an output index")
            output_indices.add(product.output_index)
        for tensor in (self.encoder_output, self.latent_output, self.image_output):
            if tensor is not None and tensor.max_bytes > self.bounds.max_latent_bytes:
                raise invalid_descriptor(
                    "image or trajectory output exceeds its declared byte capacity"
                )
        if self.token_input is not None and (
            self.kind is not TransferMode.TENSOR
            or self.token_input.dtype is not DType.I64
            or self.token_input.shape_bound.max_elements != 1
        ):
            raise invalid_descriptor(
                "token transfer input requires a tensor transfer of one int64 element"
            )
        for tensor in (self.token_output,):
            if tensor is not None and (
                tensor.dtype is not DType.I64 or tensor.shape_bound.max_elements != 1
            ):
                raise invalid_descriptor("device token relay requires one int64 element")
        for tensor in (self.completion_output, self.transition_output):
            if tensor is not None and (
                tensor.dtype is not DType.U8 or tensor.shape_bound.max_elements != 1
            ):
                raise invalid_descriptor("device completion requires one uint8 element")
        for product in self.tensor_inputs():
            if product.request_key != self.request_key and product not in (
                self.vision_input,
                self.latent_feature_input,
            ):
                raise invalid_descriptor("request-local tensor belongs to another request lineage")
        if self.predicate is not None:
            if self.predicate.request_key != self.request_key:
                raise invalid_descriptor("computation predicate belongs to another request lineage")
            if (
                self.predicate.dtype not in {DType.U8, DType.I64}
                or self.predicate.shape_bound.max_elements != 1
            ):
                raise invalid_descriptor(
                    "device predicate requires a boolean or packed continuation scalar"
                )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "operation",
    ) -> ScheduledRequest:
        """Parse and validate a computation, its identity, and its execution dependencies."""

        # Field decoding follows declaration order with a no-allocation fast
        # path per field. Irregular values use the validating field decoders so
        # diagnostics identify the first invalid declaration.
        data = _map(value, where)
        get = data.get
        request_key = _fast_request_key(get("request_key"))
        if request_key is None:
            request_key = RequestKey.from_mapping(get("request_key"), f"{where}.request_key")
        op_id = ComputationId.from_mapping(get("op_id"), f"{where}.op_id")
        predecessor_value = get("predecessor")
        predecessor = (
            None
            if predecessor_value is None
            else ComputationId.from_mapping(predecessor_value, f"{where}.predecessor")
        )
        entry = _str(get("entry"), f"{where}.entry")
        work = computation(get("code"), f"{where}.code")
        bounds = _fast_bounds(get("bounds"))
        if bounds is None:
            bounds = Bounds.from_mapping(get("bounds"), f"{where}.bounds")
        inputs = _fast_tensor_refs(get("inputs", ()))
        if inputs is None:
            inputs = tuple(
                TensorRef.from_mapping(item, f"{where}.inputs[{index}]")
                for index, item in enumerate(_seq(get("inputs", ()), f"{where}.inputs"))
            )
        outputs = _fast_tensor_refs(get("outputs", ()))
        if outputs is None:
            outputs = tuple(
                TensorRef.from_mapping(item, f"{where}.outputs[{index}]")
                for index, item in enumerate(_seq(get("outputs", ()), f"{where}.outputs"))
            )
        predicate_raw = get("predicate")
        if predicate_raw is None:
            predicate = None
        else:
            predicate = _fast_tensor_ref(predicate_raw)
            if predicate is None:
                predicate = TensorRef.from_mapping(predicate_raw, f"{where}.predicate")
        rng_raw = get("rng")
        if rng_raw is None:
            rng = None
        else:
            rng = _fast_rng(rng_raw)
            if rng is None:
                rng = Rng.from_mapping(rng_raw, f"{where}.rng")
        operation = cls(
            request_key=request_key,
            op_id=op_id,
            predecessor=predecessor,
            entry=entry,
            kind=work,
            bounds=bounds,
            inputs=inputs,
            outputs=outputs,
            token_input=None
            if get("token_input") is None
            else TensorRef.from_mapping(get("token_input"), f"{where}.token_input"),
            token_output=None
            if get("token_output") is None
            else TensorRef.from_mapping(get("token_output"), f"{where}.token_output"),
            vision_input=None
            if get("vision_input") is None
            else TensorRef.from_mapping(get("vision_input"), f"{where}.vision_input"),
            latent_feature_input=None
            if get("latent_feature_input") is None
            else TensorRef.from_mapping(
                get("latent_feature_input"), f"{where}.latent_feature_input"
            ),
            encoder_output=None
            if get("encoder_output") is None
            else TensorRef.from_mapping(get("encoder_output"), f"{where}.encoder_output"),
            latent_input=None
            if get("latent_input") is None
            else TensorRef.from_mapping(get("latent_input"), f"{where}.latent_input"),
            latent_output=None
            if get("latent_output") is None
            else TensorRef.from_mapping(get("latent_output"), f"{where}.latent_output"),
            image_input=None
            if get("image_input") is None
            else TensorRef.from_mapping(get("image_input"), f"{where}.image_input"),
            image_output=None
            if get("image_output") is None
            else TensorRef.from_mapping(get("image_output"), f"{where}.image_output"),
            completion_output=None
            if get("completion_output") is None
            else TensorRef.from_mapping(get("completion_output"), f"{where}.completion_output"),
            transition_output=None
            if get("transition_output") is None
            else TensorRef.from_mapping(get("transition_output"), f"{where}.transition_output"),
            predicate=predicate,
            rng=rng,
            input_token_ids=tuple(_ints(get("input_token_ids"), "input_token_ids")),
            kv_input=None
            if get("kv_input") is None
            else BufferId.from_mapping(get("kv_input"), f"{where}.kv_input"),
            kv_output=None
            if get("kv_output") is None
            else BufferId.from_mapping(get("kv_output"), f"{where}.kv_output"),
            input_image=(
                None if get("input_image") is None else _str(get("input_image"), "input_image")
            ),
            sampling_state=(
                None
                if get("sampling_state") is None
                else SamplingState.from_mapping(get("sampling_state"))
            ),
        )
        operation.validate()
        return operation

    def to_mapping(self) -> dict[str, object]:
        """Encode computation fields and their request and predecessor identities."""

        return {
            "request_key": self.request_key.to_mapping(),
            "op_id": self.op_id.to_mapping(),
            "predecessor": None if self.predecessor is None else self.predecessor.to_mapping(),
            "entry": self.entry,
            "code": self.kind.value,
            "bounds": self.bounds.to_mapping(),
            "inputs": [product.to_mapping() for product in self.inputs],
            "outputs": [product.to_mapping() for product in self.outputs],
            "token_input": None if self.token_input is None else self.token_input.to_mapping(),
            "token_output": None if self.token_output is None else self.token_output.to_mapping(),
            "vision_input": None if self.vision_input is None else self.vision_input.to_mapping(),
            "latent_feature_input": None
            if self.latent_feature_input is None
            else self.latent_feature_input.to_mapping(),
            "encoder_output": None
            if self.encoder_output is None
            else self.encoder_output.to_mapping(),
            "latent_input": None if self.latent_input is None else self.latent_input.to_mapping(),
            "latent_output": None
            if self.latent_output is None
            else self.latent_output.to_mapping(),
            "image_input": None if self.image_input is None else self.image_input.to_mapping(),
            "image_output": None if self.image_output is None else self.image_output.to_mapping(),
            "completion_output": None
            if self.completion_output is None
            else self.completion_output.to_mapping(),
            "transition_output": None
            if self.transition_output is None
            else self.transition_output.to_mapping(),
            "predicate": None if self.predicate is None else self.predicate.to_mapping(),
            "rng": None if self.rng is None else self.rng.to_mapping(),
            "input_token_ids": list(self.input_token_ids),
            "input_image": self.input_image,
            "kv_input": None if self.kv_input is None else self.kv_input.to_mapping(),
            "kv_output": None if self.kv_output is None else self.kv_output.to_mapping(),
            "sampling_state": (
                None if self.sampling_state is None else self.sampling_state.to_mapping()
            ),
        }


@dataclass(frozen=True, slots=True)
class FinishFlags:
    """Records length, stop-token, EOS, and forced termination conditions for generated output."""

    eos: bool = False
    length: bool = False
    stop: bool = False

    @classmethod
    def from_mapping(cls, value: object, where: str = "finish_flags") -> FinishFlags:
        """Parse EOS, length-limit, and stop-sequence termination flags."""

        data = _map(value, where)
        return cls(
            eos=_bool(data.get("eos", False), f"{where}.eos"),
            length=_bool(data.get("length", False), f"{where}.length"),
            stop=_bool(data.get("stop", False), f"{where}.stop"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize generation termination flags for IPC."""

        return {"eos": self.eos, "length": self.length, "stop": self.stop}


@dataclass(frozen=True, slots=True)
class TimingCounters:
    """Accumulates queue, device, copy, and host execution time in microseconds."""

    queued_us: int = 0
    device_us: int = 0
    copy_us: int = 0
    host_us: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "timing_counters") -> TimingCounters:
        """Parse nonnegative queue, device, copy, and host timings in microseconds."""

        data = _map(value, where)
        return cls(
            queued_us=_uint(data.get("queued_us", 0), f"{where}.queued_us"),
            device_us=_uint(data.get("device_us", 0), f"{where}.device_us"),
            copy_us=_uint(data.get("copy_us", 0), f"{where}.copy_us"),
            host_us=_uint(data.get("host_us", 0), f"{where}.host_us"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize execution-stage timings in microseconds for IPC."""

        return {
            "queued_us": self.queued_us,
            "device_us": self.device_us,
            "copy_us": self.copy_us,
            "host_us": self.host_us,
        }


@dataclass(frozen=True, slots=True)
class PosixShmArtifact:
    """Identifies a completed media artifact stored in POSIX shared memory."""

    name: str

    def __post_init__(self) -> None:
        """Validate the shared-memory artifact name and byte length."""

        if not self.name or "/" in self.name:
            raise invalid_descriptor("POSIX shared-memory artifact name is invalid")

    @classmethod
    def from_mapping(cls, value: object, where: str = "artifact handle") -> PosixShmArtifact:
        """Parse and validate a POSIX shared-memory media handle."""

        data = _map(value, where)
        if data.get("transport") != "posix_shm":
            raise invalid_descriptor(f"{where}.transport is invalid")
        payload = _map(data.get("value"), f"{where}.value")
        return cls(name=_str(payload.get("name"), f"{where}.value.name"))

    def to_mapping(self) -> dict[str, object]:
        """Serialize the shared-memory handle as a tagged transport value."""

        return {"transport": "posix_shm", "value": {"name": self.name}}


@dataclass(frozen=True, slots=True)
class MediaOutput:
    """Describes a produced media artifact by format, dimensions, duration, and storage reference."""

    handle: PosixShmArtifact
    bytes: int

    def __post_init__(self) -> None:
        """Validate media format, dimensions, duration, and artifact consistency."""

        if self.bytes < 1:
            raise invalid_descriptor("media output locator is invalid")

    @classmethod
    def from_mapping(cls, value: object, where: str = "media_output") -> MediaOutput:
        """Parse a validated media artifact handle and byte extent."""

        data = _map(value, where)
        return cls(
            handle=PosixShmArtifact.from_mapping(data.get("handle"), f"{where}.handle"),
            bytes=_uint(data.get("bytes"), f"{where}.bytes"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize a completed media artifact for IPC."""

        return {"handle": self.handle.to_mapping(), "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """An operation completion with accepted progress, tokens, products, and timing."""

    request_key: RequestKey
    op_id: ComputationId
    status: OpStatus
    product_generations: tuple[int, ...]
    error_code: ErrorCode | None
    timing_counters: TimingCounters
    kind: Computation
    position: int
    kv_visible_len: int
    kv_computed_len: int
    num_completed_steps: int
    committed_tokens: tuple[int, ...]
    finish_flags: FinishFlags
    media_output: MediaOutput | None = None
    kv_output: KvTransfer | None = None
    sampled_logprob: float | None = None
    top_logprobs: tuple[tuple[int, float, int], ...] = ()
    prompt_logprobs: tuple[tuple[tuple[int, float, int], ...], ...] = ()

    def validate(self) -> None:
        """Verify that status, products, errors, and timing form a coherent completion."""

        if self.op_id.batch_id < 1:
            raise invalid_descriptor("completion op id must be positive")
        if self.kv_output is not None and (
            self.status is not OpStatus.OK
            or self.kind is not TransferMode.KV_PUBLISH
            or self.kv_output.source.owner != self.request_key
            or self.kv_output.source.producer_op_id != self.op_id
        ):
            raise invalid_descriptor("KV publication does not belong to its successful completion")

        if self.kv_visible_len > self.kv_computed_len:
            raise invalid_descriptor("completion selected KV length exceeds computed length")
        if (
            min(self.position, self.kv_visible_len, self.kv_computed_len, self.num_completed_steps)
            < 0
        ):
            raise invalid_descriptor("completion execution coordinates must be non-negative")
        if self.status is OpStatus.ERROR:
            if self.error_code is None:
                raise invalid_descriptor("an error completion must carry an error code")
        elif self.error_code is not None:
            raise invalid_descriptor("a non-error completion must not carry an error code")
        if self.status is OpStatus.PREDICATED and (
            self.committed_tokens
            or self.sampled_logprob is not None
            or self.top_logprobs
            or self.prompt_logprobs
            or self.product_generations
            or self.finish_flags.eos
            or self.finish_flags.length
            or self.finish_flags.stop
        ):
            raise invalid_descriptor(
                "a predicated completion must select its parent without semantic output"
            )

    @classmethod
    def from_mapping(cls, value: object, where: str = "completion") -> RequestOutput:
        """Parse a completion and enforce its status-specific result and error contract."""

        data = _map(value, where)
        record = cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            op_id=ComputationId.from_mapping(data.get("op_id"), f"{where}.op_id"),
            status=_enum(OpStatus, data.get("status"), f"{where}.status"),
            product_generations=_uints(
                data.get("product_generations", ()), f"{where}.product_generations"
            ),
            error_code=(
                None
                if data.get("error_code") is None
                else _enum(ErrorCode, data["error_code"], f"{where}.error_code")
            ),
            timing_counters=TimingCounters.from_mapping(
                data.get("timing_counters"), f"{where}.timing_counters"
            ),
            kind=computation(data.get("code"), f"{where}.code"),
            position=_uint(data.get("position"), f"{where}.position"),
            kv_visible_len=_uint(data.get("kv_visible_len"), f"{where}.kv_visible_len"),
            kv_computed_len=_uint(data.get("kv_computed_len"), f"{where}.kv_computed_len"),
            num_completed_steps=_uint(
                data.get("num_completed_steps"), f"{where}.num_completed_steps"
            ),
            sampled_logprob=(
                None
                if data.get("sampled_logprob") is None
                else _logprob_value(data["sampled_logprob"])
            ),
            top_logprobs=_logprob_entries(data.get("top_logprobs", ())),
            prompt_logprobs=tuple(
                _logprob_entries(entries)
                for entries in _seq(data.get("prompt_logprobs", ()), "prompt_logprobs")
            ),
            committed_tokens=_uints(data.get("committed_tokens", ()), f"{where}.committed_tokens"),
            finish_flags=FinishFlags.from_mapping(
                data.get("finish_flags"), f"{where}.finish_flags"
            ),
            kv_output=None
            if data.get("kv_output") is None
            else KvTransfer.from_mapping(data["kv_output"], f"{where}.kv_output"),
            media_output=None
            if data.get("media_output") is None
            else MediaOutput.from_mapping(data["media_output"], f"{where}.media_output"),
        )
        record.validate()
        return record

    def to_mapping(self) -> dict[str, object]:
        """Encode one operation completion with accepted progress, products, timing, and error metadata."""

        flags = self.finish_flags
        key = self.request_key
        timing = self.timing_counters
        error_code = self.error_code
        return {
            "request_key": {
                "engine_id": key.engine_id,
                "request_id": key.request_id,
                "request_epoch": key.request_epoch,
            },
            "op_id": self.op_id.to_mapping(),
            "status": self.status.value,
            "code": self.kind.value,
            "position": self.position,
            "kv_visible_len": self.kv_visible_len,
            "kv_computed_len": self.kv_computed_len,
            "num_completed_steps": self.num_completed_steps,
            "committed_tokens": list(self.committed_tokens),
            "sampled_logprob": self.sampled_logprob,
            "top_logprobs": [
                {"token_id": token, "logprob": value, "rank": rank}
                for token, value, rank in self.top_logprobs
            ],
            "prompt_logprobs": [
                [
                    {"token_id": token, "logprob": value, "rank": rank}
                    for token, value, rank in entries
                ]
                for entries in self.prompt_logprobs
            ],
            "finish_flags": {"eos": flags.eos, "length": flags.length, "stop": flags.stop},
            "media_output": None if self.media_output is None else self.media_output.to_mapping(),
            "kv_output": None if self.kv_output is None else self.kv_output.to_mapping(),
            "product_generations": list(self.product_generations),
            "error_code": None if error_code is None else error_code.value,
            "timing_counters": {
                "queued_us": timing.queued_us,
                "device_us": timing.device_us,
                "copy_us": timing.copy_us,
                "host_us": timing.host_us,
            },
        }


@dataclass(frozen=True, slots=True)
class Start:
    """Registers a new request and its immutable execution parameters."""

    request: NewRequest

    @property
    def request_key(self) -> RequestKey:
        """Expose the request generation registered by this start command."""

        return self.request.request_key

    @classmethod
    def from_mapping(cls, value: object, where: str = "start command") -> Start:
        """Parse a request admission from a start-command payload."""

        data = _map(value, where)
        return cls(request=NewRequest.from_mapping(data.get("request"), f"{where}.request"))


@dataclass(frozen=True, slots=True)
class Finish:
    """Close a lineage while preserving explicitly retained persistent allocations.

    Retained buffers are owned outside the request and remain readable until Free;
    request slots, KV state, and unretained products retire after readers finish.
    """

    request_key: RequestKey
    retained_buffers: tuple[BufferId, ...] = ()

    def __post_init__(self) -> None:
        _validate_retained_buffers(self.request_key, self.retained_buffers)


def _validate_retained_buffers(request: RequestKey, retained: tuple[BufferId, ...]) -> None:
    if any(buffer.owner != request for buffer in retained):
        raise invalid_descriptor("retained buffer belongs to another request")
    if len(set(retained)) != len(retained):
        raise invalid_descriptor("request retirement repeats a retained buffer")


@dataclass(frozen=True, slots=True)
class Free:
    """Releases one scheduler-owned persistent buffer."""

    buffer: BufferId

    @property
    def request_key(self) -> RequestKey:
        """Expose the request generation that owns the freed persistent buffer."""

        return self.buffer.owner


BatchCommand: TypeAlias = Start | Finish | Free


def _command_variant_index(command: BatchCommand) -> int:
    """Return the stable wire-union variant index for a lifecycle command."""

    if isinstance(command, Start):
        return 0
    if isinstance(command, Finish):
        return 1
    return 2


def command_from_mapping(
    value: object,
    where: str = "batch command",
) -> BatchCommand:
    """Parse a tagged start, finish, or free lifecycle command."""

    kind, payload = _tagged(value, where)
    data = _map(payload, f"{where}.value")
    if kind == "start":
        return Start.from_mapping(data, f"{where}.value")
    request_key = _fast_request_key(data.get("request_key"))
    if kind != "free" and request_key is None:
        request_key = RequestKey.from_mapping(data.get("request_key"), f"{where}.value.request_key")
    if kind == "finish":
        assert request_key is not None
        command: BatchCommand = Finish(
            request_key=request_key,
            retained_buffers=tuple(
                BufferId.from_mapping(buffer, f"{where}.value.retained_buffers[{index}]")
                for index, buffer in enumerate(
                    _seq(data.get("retained_buffers"), f"{where}.value.retained_buffers")
                )
            ),
        )
    elif kind == "free":
        command = Free(buffer=BufferId.from_mapping(data.get("buffer"), f"{where}.value.buffer"))
    else:
        raise invalid_descriptor(f"{where} has unknown variant {kind!r}")
    return command


def command_to_mapping(command: BatchCommand) -> dict[str, object]:
    """Encode a lifecycle command with its stable variant tag."""

    if isinstance(command, Start):
        return {
            "kind": "start",
            "value": {"request": command.request.to_mapping()},
        }
    if isinstance(command, Finish):
        return {
            "kind": "finish",
            "value": {
                "request_key": command.request_key.to_mapping(),
                "retained_buffers": [buffer.to_mapping() for buffer in command.retained_buffers],
            },
        }
    return {"kind": "free", "value": {"buffer": command.buffer.to_mapping()}}


@dataclass(frozen=True, slots=True)
class ArRequestParams:
    """Defines prompt tokens and sampling policy for autoregressive execution.

    The parameters also bound generated tokens and carry speculative draft input.
    """

    sampling: SamplingParams = field(default_factory=SamplingParams)
    negative_token_ids: tuple[int, ...] = ()
    finish_token_ids: tuple[int, ...] = ()
    initial_position: int = 0

    def __post_init__(self) -> None:
        """Validate prompt, generation limit, draft tokens, and sampling policy."""

        _nonnegative(self.initial_position, "autoregressive initial position")
        if any(
            left >= right
            for left, right in zip(
                self.finish_token_ids,
                self.finish_token_ids[1:],
                strict=False,
            )
        ):
            raise invalid_descriptor("autoregressive finish token ids are not canonical")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "autoregressive parameters"
    ) -> ArRequestParams:
        """Parse sampling policy, token controls, and initial position for autoregressive work."""

        data = _map(value, where)
        return cls(
            sampling=SamplingParams.from_mapping(data.get("sampling", {}), f"{where}.sampling"),
            negative_token_ids=_uints(
                data.get("negative_token_ids", ()), f"{where}.negative_token_ids"
            ),
            finish_token_ids=_uints(data.get("finish_token_ids", ()), f"{where}.finish_token_ids"),
            initial_position=_uint(data.get("initial_position", 0), f"{where}.initial_position"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize autoregressive sampling and token controls for admission."""

        return {
            "sampling": self.sampling.to_mapping(),
            "negative_token_ids": list(self.negative_token_ids),
            "finish_token_ids": list(self.finish_token_ids),
            "initial_position": self.initial_position,
        }


@dataclass(frozen=True, slots=True)
class UmmRequestParams:
    """Defines multimodal request text, media inputs, and model-specific generation controls."""

    image: ImageParams = field(default_factory=ImageParams)

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "unified-multimodal parameters"
    ) -> UmmRequestParams:
        """Parse unified multimodal image-generation parameters for admission."""

        data = _map(value, where)
        return cls(image=ImageParams.from_mapping(data.get("image", {}), f"{where}.image"))

    def to_mapping(self) -> dict[str, object]:
        """Serialize unified multimodal controls for admission."""

        return {"image": self.image.to_mapping()}


@dataclass(frozen=True, slots=True)
class DiffusionSamplingParams:
    """Effective diffusion bounds and seed resolved by model preprocessing."""

    num_frames: int
    num_decode_chunks: int
    num_inference_steps: int
    seed: int

    def __post_init__(self) -> None:
        """Require positive work bounds and a nonnegative deterministic seed."""

        for name in ("num_frames", "num_decode_chunks", "num_inference_steps"):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"diffusion {name} must be positive")
        _nonnegative(self.seed, "diffusion seed")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "diffusion sampling"
    ) -> DiffusionSamplingParams:
        """Parse the core's effective diffusion parameters."""

        data = _map(value, where)
        return cls(
            num_frames=_uint(data.get("num_frames"), f"{where}.num_frames"),
            num_decode_chunks=_uint(data.get("num_decode_chunks"), f"{where}.num_decode_chunks"),
            num_inference_steps=_uint(
                data.get("num_inference_steps"), f"{where}.num_inference_steps"
            ),
            seed=_uint(data.get("seed"), f"{where}.seed"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the shared diffusion parameter definition."""

        return {
            "num_frames": self.num_frames,
            "num_decode_chunks": self.num_decode_chunks,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class NewRequest:
    """Binds one request key to its autoregressive, multimodal, or diffusion parameters."""

    request_key: RequestKey
    request_pool_idx: int
    ar: ArRequestParams | None = None
    umm: UmmRequestParams | None = None
    diffusion: DiffusionSamplingParams | None = None
    prompt_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Require a valid request slot, family parameters, and diffusion prompt tokens."""

        if self.diffusion is not None and not self.prompt_token_ids:
            raise invalid_descriptor("diffusion prompt tokens must not be empty")
        if self.request_pool_idx < 1:
            raise invalid_descriptor("request-pool index must be positive")
        if self.ar is None and self.umm is None and self.diffusion is None:
            raise invalid_descriptor("request start must declare one runtime-family parameter set")

    @classmethod
    def from_mapping(cls, value: object, where: str = "admission") -> NewRequest:
        """Parse a request slot and its optional execution-family parameter sets."""

        data = _map(value, where)
        admission = cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            request_pool_idx=_uint(data.get("request_pool_idx"), f"{where}.request_pool_idx"),
            prompt_token_ids=_uints(data.get("prompt_token_ids", ()), f"{where}.prompt_token_ids"),
            ar=(
                None
                if data.get("ar") is None
                else ArRequestParams.from_mapping(data["ar"], f"{where}.ar")
            ),
            umm=(
                None
                if data.get("umm") is None
                else UmmRequestParams.from_mapping(data["umm"], f"{where}.umm")
            ),
            diffusion=(
                None
                if data.get("diffusion") is None
                else DiffusionSamplingParams.from_mapping(data["diffusion"], f"{where}.diffusion")
            ),
        )
        return admission

    def to_mapping(self) -> dict[str, object]:
        """Serialize request identity, slot, and family parameters for admission."""

        return {
            "request_key": self.request_key.to_mapping(),
            "request_pool_idx": self.request_pool_idx,
            "prompt_token_ids": list(self.prompt_token_ids),
            "ar": None if self.ar is None else self.ar.to_mapping(),
            "umm": None if self.umm is None else self.umm.to_mapping(),
            "diffusion": None if self.diffusion is None else self.diffusion.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class BlockTable:
    """Maps a request and KV group to its ordered physical pages and allocated token extent."""

    request_pool_idx: int
    group_id: int
    page_ids: tuple[int, ...]
    allocated_tokens: int

    def __post_init__(self) -> None:
        """Validate request/group identity, ordered pages, and allocated token extent."""

        if (
            self.request_pool_idx < 1
            or self.group_id < 0
            or self.allocated_tokens < 0
            or any(page < 1 for page in self.page_ids)
            or len(set(self.page_ids)) != len(self.page_ids)
            or (not self.page_ids and self.allocated_tokens != 0)
        ):
            raise invalid_descriptor("block table is invalid")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "block table",
    ) -> BlockTable:
        """Parse an installed KV page table and its allocated token extent."""

        data = _map(value, where)

        def uint_field(name: str) -> int:
            """Decode a nonnegative integer field through the fast scalar path."""

            raw = data.get(name)
            return raw if type(raw) is int and raw >= 0 else _uint(raw, f"{where}.{name}")

        page_ids = _fast_uints(data.get("page_ids", ()))
        if page_ids is None:
            page_ids = _uints(data.get("page_ids", ()), f"{where}.page_ids")
        fields = (
            uint_field("request_pool_idx"),
            uint_field("group_id"),
            page_ids,
            uint_field("allocated_tokens"),
        )
        return cls(*fields)

    def to_mapping(self) -> dict[str, object]:
        """Serialize an installed request-and-group KV page table."""

        return {
            "request_pool_idx": self.request_pool_idx,
            "group_id": self.group_id,
            "page_ids": list(self.page_ids),
            "allocated_tokens": self.allocated_tokens,
        }


@dataclass(frozen=True, slots=True)
class CachePageAllocation:
    """Declares newly assigned physical pages for a request KV group."""

    request_pool_idx: int
    group_id: int
    page_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate newly assigned page identifiers and request/group ownership."""

        if (
            self.request_pool_idx < 1
            or self.group_id < 0
            or not self.page_ids
            or any(page < 1 for page in self.page_ids)
            or len(set(self.page_ids)) != len(self.page_ids)
        ):
            raise invalid_descriptor("cache-page allocation is invalid")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "cache-page allocation",
    ) -> CachePageAllocation:
        """Parse newly assigned physical KV pages for one request and cache group."""

        data = _map(value, where)

        def uint_field(name: str) -> int:
            """Decode a nonnegative integer field through the fast scalar path."""

            raw = data.get(name)
            return raw if type(raw) is int and raw >= 0 else _uint(raw, f"{where}.{name}")

        page_ids = _fast_uints(data.get("page_ids", ()))
        if page_ids is None:
            page_ids = _uints(data.get("page_ids", ()), f"{where}.page_ids")
        fields = (
            uint_field("request_pool_idx"),
            uint_field("group_id"),
            page_ids,
        )
        return cls(*fields)

    def to_mapping(self) -> dict[str, object]:
        """Serialize newly assigned KV pages for lane registration."""

        return {
            "request_pool_idx": self.request_pool_idx,
            "group_id": self.group_id,
            "page_ids": list(self.page_ids),
        }


def _validate_forward_inputs(
    operation_count: int,
    operation_indices: tuple[int, ...],
    request_pool_indices: tuple[int, ...],
    seq_lens: tuple[int, ...],
    query_lens: tuple[int, ...],
    write_kv: tuple[bool, ...],
) -> None:
    """Validate aligned model inputs before lane preparation indexes their columns."""

    rows = len(operation_indices)
    if any(
        len(column) != rows for column in (request_pool_indices, seq_lens, query_lens, write_kv)
    ):
        raise invalid_descriptor("forward input columns have different lengths")
    if any(index < 0 or index >= operation_count for index in operation_indices):
        raise invalid_descriptor("forward operation index is outside its batch")
    if any(slot < 1 for slot in request_pool_indices):
        raise invalid_descriptor("forward input carries the reserved request slot")
    if any(total < query for total, query in zip(seq_lens, query_lens, strict=True)) or any(
        length < 1 for length in query_lens
    ):
        raise invalid_descriptor("forward input has invalid token lengths")


@dataclass(frozen=True, slots=True)
class LatentParams:
    """Solver-step range and optional paged storage for one request trajectory."""

    request_key: RequestKey
    op_id: ComputationId
    page_table: tuple[int, ...]
    latent_units: int
    height: int
    width: int
    start_step: int
    step_count: int

    def __post_init__(self) -> None:
        """Validate latent page ownership, bank, units, width, and raster geometry."""

        if self.op_id.batch_id < 1:
            raise invalid_descriptor("latent params operation id must be positive")
        if min(self.height, self.width) < 1 or self.latent_units < 0:
            raise invalid_descriptor("latent params geometry must be positive")
        if (
            bool(self.page_table) != (self.latent_units > 0)
            or any(page < 1 for page in self.page_table)
            or len(set(self.page_table)) != len(self.page_table)
        ):
            raise invalid_descriptor(
                "latent params page table disagrees with its units, repeats a page, or carries page zero"
            )

    @classmethod
    def from_mapping(cls, value: object, where: str = "latent params") -> LatentParams:
        """Parse request-owned latent pages, raster geometry, and solver-step range."""

        data = _map(value, where)
        return cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            op_id=ComputationId.from_mapping(data.get("op_id"), f"{where}.op_id"),
            page_table=_uints(data.get("page_table", ()), f"{where}.page_table"),
            latent_units=_uint(data.get("latent_units"), f"{where}.latent_units"),
            height=_uint(data.get("height"), f"{where}.height"),
            width=_uint(data.get("width"), f"{where}.width"),
            start_step=_uint(data.get("start_step"), f"{where}.start_step"),
            step_count=_uint(data.get("step_count"), f"{where}.step_count"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize a physical latent trajectory params for lane execution."""

        return {
            "request_key": self.request_key.to_mapping(),
            "op_id": self.op_id.to_mapping(),
            "page_table": list(self.page_table),
            "latent_units": self.latent_units,
            "height": self.height,
            "width": self.width,
            "start_step": self.start_step,
            "step_count": self.step_count,
        }


class MediaTrack(StrEnum):
    """Independent media stream addressed by a bounded operation."""

    VIDEO = "video"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class DecodeRange:
    """Selects a bounded temporal range for a concrete media computation."""

    request_key: RequestKey
    op_id: ComputationId
    cursor: int
    max_units: int

    def __post_init__(self) -> None:
        """Validate latent slice bounds and destination output-ring slot."""

        if self.op_id.batch_id < 1 or self.max_units < 1:
            raise invalid_descriptor("decode params identity and unit bound must be positive")
        _nonnegative(self.cursor, "decode params cursor")

    @classmethod
    def from_mapping(cls, value: object, where: str = "decode params") -> DecodeRange:
        """Parse the reconstruction cursor and bounded unit count for one operation."""

        data = _map(value, where)
        return cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            op_id=ComputationId.from_mapping(data.get("op_id"), f"{where}.op_id"),
            cursor=_uint(data.get("cursor"), f"{where}.cursor"),
            max_units=_uint(data.get("max_units"), f"{where}.max_units"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize a media reconstruction params for lane execution."""

        return {
            "request_key": self.request_key.to_mapping(),
            "op_id": self.op_id.to_mapping(),
            "cursor": self.cursor,
            "max_units": self.max_units,
        }


@dataclass(frozen=True, slots=True)
class BufferAllocation:
    """Assigns a product to a bounded slice of scheduler-managed persistent storage."""

    buffer: BufferId
    offset: int
    bytes: int

    def __post_init__(self) -> None:
        """Validate the persistent-buffer identifier, offset, and bounded shape."""

        if self.offset < 0 or self.bytes < 1:
            raise invalid_descriptor("buffer params span is invalid")
        if self.offset + self.bytes > (1 << 64) - 1:
            raise invalid_descriptor("buffer params span overflows")

    @classmethod
    def from_mapping(cls, value: object, where: str = "buffer params") -> BufferAllocation:
        """Parse a bounded byte slice in scheduler-managed persistent storage."""

        data = _map(value, where)
        return cls(
            buffer=BufferId.from_mapping(data.get("buffer"), f"{where}.buffer"),
            offset=_uint(data.get("offset"), f"{where}.offset"),
            bytes=_uint(data.get("bytes"), f"{where}.bytes"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize persistent-buffer identity, offset, and byte extent."""

        return {
            "buffer": self.buffer.to_mapping(),
            "offset": self.offset,
            "bytes": self.bytes,
        }


def _validate_buffer_allocations(
    operations: Sequence[ScheduledRequest],
    parameters: Sequence[BufferAllocation],
    where: str,
) -> None:
    """Validate persistent-buffer parameters against producing operations and shape bounds."""

    by_id: dict[BufferId, BufferAllocation] = {}
    spans: list[tuple[int, int]] = []
    for params in parameters:
        if params.buffer in by_id:
            raise invalid_descriptor(f"{where} repeats a buffer params identity")
        by_id[params.buffer] = params
        spans.append((params.offset, params.offset + params.bytes))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise invalid_descriptor(f"{where} buffer parameters overlap")
    for operation in operations:
        for output in operation.buffer_outputs():
            output_allocation = by_id.get(output.buffer_id)
            if output_allocation is None:
                raise invalid_descriptor("persistent operation output has no buffer params")
            if output_allocation.bytes < output.max_bytes:
                raise invalid_descriptor("buffer params is smaller than its declared output")


@dataclass(frozen=True, slots=True)
class ScheduleBatch:
    """Describes one scheduler-submitted collection of operations."""

    batch_id: int
    run_id: int
    collective_seq: int = 1
    operations: tuple[ScheduledRequest, ...] = ()
    block_tables: tuple[BlockTable, ...] = ()
    new_cache_pages: tuple[CachePageAllocation, ...] = ()
    forward_operation_indices: tuple[int, ...] = ()
    request_pool_indices: tuple[int, ...] = ()
    seq_lens: tuple[int, ...] = ()
    query_lens: tuple[int, ...] = ()
    write_kv: tuple[bool, ...] = ()
    latent_params: tuple[LatentParams, ...] = ()
    decode_ranges: tuple[DecodeRange, ...] = ()
    buffer_allocations: tuple[BufferAllocation, ...] = ()
    commands: tuple[BatchCommand, ...] = ()
    input_products: tuple[TensorPublication, ...] = ()
    kv_inputs: tuple[KvTransfer, ...] = ()

    def __post_init__(self) -> None:
        """Validate run identity and lifecycle-operation consistency."""

        self.validate()

    @property
    def admissions(self) -> tuple[NewRequest, ...]:
        """Extract new-request payloads from lifecycle commands in submission order."""

        return tuple(command.request for command in self.commands if isinstance(command, Start))

    def validate(self) -> None:
        """Enforce run identity, command ordering, operation counts, and token bounds."""

        if not self.operations and not self.commands:
            raise invalid_descriptor(
                "a submission batch must carry at least one operation or command"
            )
        if self.collective_seq < 1:
            raise invalid_descriptor("run collective sequence must be positive")
        _validate_forward_inputs(
            len(self.operations),
            self.forward_operation_indices,
            self.request_pool_indices,
            self.seq_lens,
            self.query_lens,
            self.write_kv,
        )
        computation_ids = [operation.op_id for operation in self.operations]
        if any(identity.batch_id != self.batch_id for identity in computation_ids):
            raise invalid_descriptor("computation identity belongs to another logical batch")
        if len(set(computation_ids)) != len(computation_ids):
            raise invalid_descriptor("a submission batch repeats a computation identity")
        request_keys = [operation.request_key for operation in self.operations]
        if len(set(request_keys)) != len(request_keys):
            raise invalid_descriptor(
                "a submission batch carries multiple operations for one request"
            )
        admitted = [admission.request_key for admission in self.admissions]
        if len(set(admitted)) != len(admitted):
            raise invalid_descriptor("a submission batch carries a duplicate admission")
        identities: dict[tuple[RequestKey, int, BufferId | None], BatchCommand] = {}
        for command in self.commands:
            buffer = command.buffer if isinstance(command, Free) else None
            identity = (
                command.request_key,
                _command_variant_index(command),
                buffer,
            )
            existing = identities.get(identity)
            if existing is not None and existing != command:
                raise invalid_descriptor(
                    "a submission batch reuses a command identity with different content"
                )
            identities[identity] = command
        declared_inputs = {
            product
            for operation in self.operations
            for product in (*operation.tensor_inputs(), operation.predicate)
            if product is not None
        }
        supplied_inputs: set[TensorRef] = set()
        for payload in self.input_products:
            product = payload.product
            if product not in declared_inputs:
                raise invalid_descriptor(
                    "an input product payload is not declared by any operation"
                )
            if product in supplied_inputs:
                raise invalid_descriptor("a submission batch repeats an input product payload")
            supplied_inputs.add(product)
        sources: set[BufferId] = set()
        for publication in self.kv_inputs:
            publication.encoded_size_bound()
            if publication.source in sources:
                raise invalid_descriptor("run repeats a KV input")
            sources.add(publication.source)
            consumers = tuple(
                operation
                for operation in self.operations
                if operation.kv_input == publication.source
            )
            if len(consumers) != 1 or consumers[0].kind is not TransferMode.KV_INSTALL:
                raise invalid_descriptor("KV transfer requires one installation consumer")
            if (
                sum(tensor.nbytes for tensor in publication.tensors)
                > consumers[0].bounds.max_transfer_bytes
            ):
                raise invalid_descriptor("KV input exceeds its installation transfer-byte bound")
        _validate_buffer_allocations(self.operations, self.buffer_allocations, "run")

    @classmethod
    def from_mapping(cls, value: object) -> ScheduleBatch:
        """Parse a scheduler run and validate all lifecycle commands and physical inputs."""

        data = _map(value, "execute run")
        batch_id = _uint(data.get("batch_id"), "execute run.batch_id")
        run_id = _uint(data.get("run_id"), "execute run.run_id")
        operations = tuple(
            ScheduledRequest.from_mapping(item, f"execute run.operations[{index}]")
            for index, item in enumerate(_seq(data.get("operations", ()), "execute run.operations"))
        )
        commands = tuple(
            command_from_mapping(
                item,
                f"execute run.commands[{index}]",
            )
            for index, item in enumerate(_seq(data.get("commands", ()), "execute run.commands"))
        )
        input_products = tuple(
            TensorPublication.from_mapping(item, f"execute run.input_products[{index}]")
            for index, item in enumerate(
                _seq(data.get("input_products", ()), "execute run.input_products")
            )
        )
        return cls(
            batch_id=batch_id,
            run_id=run_id,
            collective_seq=_uint(data.get("collective_seq"), "execute run.collective_seq"),
            operations=operations,
            block_tables=tuple(
                BlockTable.from_mapping(item, f"execute run.block_tables[{index}]")
                for index, item in enumerate(
                    _seq(data.get("block_tables", ()), "execute run.block_tables")
                )
            ),
            new_cache_pages=tuple(
                CachePageAllocation.from_mapping(item, f"execute run.new_cache_pages[{index}]")
                for index, item in enumerate(
                    _seq(data.get("new_cache_pages", ()), "execute run.new_cache_pages")
                )
            ),
            forward_operation_indices=_uints(
                data.get("forward_operation_indices", ()),
                "forward inputs.forward_operation_indices",
            ),
            request_pool_indices=_uints(
                data.get("request_pool_indices", ()), "forward inputs.request_pool_indices"
            ),
            seq_lens=_uints(data.get("seq_lens", ()), "forward inputs.seq_lens"),
            query_lens=_uints(data.get("query_lens", ()), "forward inputs.query_lens"),
            write_kv=tuple(
                _bool(value, "forward inputs.write_kv")
                for value in _seq(data.get("write_kv", ()), "forward inputs.write_kv")
            ),
            latent_params=tuple(
                LatentParams.from_mapping(item, f"execute run.latent_params[{index}]")
                for index, item in enumerate(
                    _seq(data.get("latent_params", ()), "execute run.latent_params")
                )
            ),
            decode_ranges=tuple(
                DecodeRange.from_mapping(item, f"execute run.decode_ranges[{index}]")
                for index, item in enumerate(
                    _seq(data.get("decode_ranges", ()), "execute run.decode_ranges")
                )
            ),
            buffer_allocations=tuple(
                BufferAllocation.from_mapping(item, f"execute run.buffer_allocations[{index}]")
                for index, item in enumerate(
                    _seq(data.get("buffer_allocations", ()), "execute run.buffer_allocations")
                )
            ),
            commands=commands,
            input_products=input_products,
            kv_inputs=tuple(
                KvTransfer.from_mapping(value)
                for value in _seq(data.get("kv_inputs", ()), "run.kv_inputs")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode run identity, lifecycle commands, and physical lane descriptors."""

        return {
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "collective_seq": self.collective_seq,
            "operations": [value.to_mapping() for value in self.operations],
            "block_tables": [value.to_mapping() for value in self.block_tables],
            "new_cache_pages": [value.to_mapping() for value in self.new_cache_pages],
            "forward_operation_indices": list(self.forward_operation_indices),
            "request_pool_indices": list(self.request_pool_indices),
            "seq_lens": list(self.seq_lens),
            "query_lens": list(self.query_lens),
            "write_kv": list(self.write_kv),
            "latent_params": [value.to_mapping() for value in self.latent_params],
            "decode_ranges": [value.to_mapping() for value in self.decode_ranges],
            "buffer_allocations": [value.to_mapping() for value in self.buffer_allocations],
            "commands": [command_to_mapping(value) for value in self.commands],
            "input_products": [value.to_mapping() for value in self.input_products],
            "kv_inputs": [value.to_mapping() for value in self.kv_inputs],
        }


@dataclass(frozen=True, slots=True)
class RegistrationAck:
    """Reports whether a lane publication is visible to consumers."""

    visible: bool = False

    @classmethod
    def from_mapping(cls, value: object, where: str = "registration") -> RegistrationAck:
        """Parse whether registration effects became visible to successor runs."""

        data = _map(value, where)
        return cls(visible=_bool(data.get("visible", False), f"{where}.visible"))

    def to_mapping(self) -> dict[str, object]:
        """Serialize registration visibility for IPC."""

        return {"visible": self.visible}


@dataclass(frozen=True, slots=True)
class TensorPublication:
    """A tensor identity and the physical metadata needed by its consumer."""

    product: TensorRef
    value: TransferValue

    def encoded_size_bound(self) -> int:
        """Bound the transfer metadata bytes, including every tensor location."""

        size = _tensor_transfers_size((self.value.tensor,))
        if size > MAX_TRANSFER_HANDLE_BYTES:
            raise invalid_descriptor("transfer metadata exceeds its byte bound")
        return size

    @classmethod
    def from_mapping(cls, value: object, where: str = "tensor publication") -> TensorPublication:
        """Parse the single generation owner and its typed transfer metadata."""

        data = _map(value, where)
        product = TensorRef.from_mapping(data.get("product"), f"{where}.product")
        transfer = _map(data.get("value"), f"{where}.value")
        kind = _str(transfer.get("kind"), f"{where}.value.kind")
        payload = _map(transfer.get("value"), f"{where}.value.value")
        height = _uint(
            payload.get("height", 0) if kind == "device_product" else payload.get("height"),
            f"{where}.value.height",
        )
        width = _uint(
            payload.get("width", 0) if kind == "device_product" else payload.get("width"),
            f"{where}.value.width",
        )
        tensor = TensorTransfer.from_mapping(payload.get("tensor"), f"{where}.value.tensor")
        if kind == "encoder":
            typed: TransferValue = EncoderTransferValue(
                height=height,
                width=width,
                payload_kind=_str(payload.get("payload_kind"), f"{where}.value.payload_kind"),
                tensor=tensor,
            )
        elif kind == "device_product":
            typed = DeviceProductTransferValue(
                height=height,
                width=width,
                value_range=_str(payload.get("value_range", ""), f"{where}.value.value_range"),
                tensor=tensor,
            )
        elif kind == "latent":
            typed = LatentTransferValue(
                height=height,
                width=width,
                latent_units=_uint(payload.get("latent_units"), f"{where}.value.latent_units"),
                step=_uint(payload.get("step", 0), f"{where}.value.step"),
                tensor=tensor,
            )
        else:
            raise invalid_descriptor(f"{where}.value.kind is invalid")
        return cls(product=product, value=typed)

    def to_mapping(self) -> dict[str, object]:
        """Encode tensor identity once alongside the concrete transfer variant."""

        typed = self.value
        value: dict[str, object] = {
            "height": typed.height,
            "width": typed.width,
            "tensor": typed.tensor.to_mapping(),
        }
        if isinstance(typed, EncoderTransferValue):
            kind = "encoder"
            value["payload_kind"] = typed.payload_kind
        elif isinstance(typed, DeviceProductTransferValue):
            kind = "device_product"
            value["value_range"] = typed.value_range
        else:
            kind = "latent"
            value["latent_units"] = typed.latent_units
            value["step"] = typed.step
        return {
            "product": self.product.to_mapping(),
            "value": {"kind": kind, "value": value},
        }


@dataclass(frozen=True, slots=True)
class SamplingState:
    """Canonical branch-local token processor inputs for one operation.

    Penalty token counts are not carried here: they are a device-resident
    accepted base plus bounded deltas folded when sampling accepts tokens,
    so no host token history participates in a successor's sampling input.
    """

    allowed_token_ids: tuple[int, ...] | None = None
    suppressed_token_ids: tuple[int, ...] = ()
    finish_token_ids: tuple[int, ...] = ()
    transition_token_ids: tuple[int, ...] = ()
    force_finish: bool = False

    def validate(self) -> None:
        """Require canonical token sets and preserve an explicitly empty whitelist."""

        for ids in (
            self.allowed_token_ids,
            self.suppressed_token_ids,
            self.finish_token_ids,
            self.transition_token_ids,
        ):
            if ids is None:
                continue
            if any(token < 0 or token > 0xFFFFFFFF for token in ids):
                raise invalid_descriptor("sampling-state token id is outside uint32")
            if any(left >= right for left, right in zip(ids, ids[1:], strict=False)):
                raise invalid_descriptor("sampling-state token ids are not canonical")

    @classmethod
    def from_mapping(cls, value: object) -> SamplingState:
        """Read direct sampler inputs from a computation mapping."""

        data = _map(value, "sampling_state")
        allowed = data.get("allowed_token_ids")
        return cls(
            allowed_token_ids=(
                None if allowed is None else tuple(_ints(allowed, "allowed_token_ids"))
            ),
            suppressed_token_ids=tuple(
                _ints(data.get("suppressed_token_ids"), "suppressed_token_ids")
            ),
            finish_token_ids=tuple(_ints(data.get("finish_token_ids"), "finish_token_ids")),
            transition_token_ids=tuple(
                _ints(data.get("transition_token_ids"), "transition_token_ids")
            ),
            force_finish=_bool(data.get("force_finish", False), "force_finish"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize sampler constraints without assigning a storage identity."""

        return {
            "allowed_token_ids": None
            if self.allowed_token_ids is None
            else list(self.allowed_token_ids),
            "suppressed_token_ids": list(self.suppressed_token_ids),
            "finish_token_ids": list(self.finish_token_ids),
            "transition_token_ids": list(self.transition_token_ids),
            "force_finish": self.force_finish,
        }


@dataclass(frozen=True, slots=True)
class ForwardStats:
    """Aggregates model-path, attention, graph, relay, and speculative-decoding measurements for a run."""

    mode_counts: Mapping[str, int] = field(default_factory=dict)
    mode_tokens: Mapping[str, int] = field(default_factory=dict)
    mode_us: Mapping[str, int] = field(default_factory=dict)
    component_us: Mapping[str, int] = field(default_factory=dict)
    attention_launches: int = 0
    attention_us: int = 0
    attention_backend_counts: Mapping[str, int] = field(default_factory=dict)
    cuda_graph_captures: int = 0
    cuda_graph_replays: int = 0
    cuda_graph_misses: int = 0
    cuda_graph_fallbacks: int = 0
    cuda_graph_unpadded_tokens: int = 0
    cuda_graph_padded_tokens: int = 0
    cuda_graph_runtime_mode_counts: Mapping[str, int] = field(default_factory=dict)
    text_decode_token_relay_hits: int = 0
    text_decode_token_relay_misses: int = 0
    text_decode_position_relay_hits: int = 0
    text_decode_position_relay_misses: int = 0
    flashinfer_decode_plan_calls: int = 0
    flashinfer_decode_plan_reuses: int = 0
    flashinfer_decode_plan_rows: int = 0
    flashinfer_decode_plan_indices: int = 0
    flashinfer_decode_graph_plan_calls: int = 0
    flashinfer_decode_graph_plan_reuses: int = 0
    spec_verify_rows: int = 0
    spec_verify_draft_tokens: int = 0
    spec_verify_accepted_tokens: int = 0
    spec_verify_rejected_tokens: int = 0
    spec_verify_committed_tokens: int = 0
    spec_verify_path_counts: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def combine(cls, values: Sequence[ForwardStats]) -> ForwardStats:
        """Sum scalar and keyed counters without changing their wire definitions."""

        if not values:
            return cls()
        if len(values) == 1:
            return values[0]
        merged: dict[str, object] = {}
        for name in cls.__dataclass_fields__:
            fields = tuple(getattr(value, name) for value in values)
            if isinstance(fields[0], Mapping):
                totals: dict[str, int] = {}
                for field_value in fields:
                    for key, count in cast(Mapping[str, int], field_value).items():
                        totals[key] = totals.get(key, 0) + count
                merged[name] = totals
            else:
                merged[name] = sum(cast(tuple[int, ...], fields))
        return cls(**cast(Any, merged))

    @classmethod
    def from_mapping(cls, value: object, where: str = "worker forward stats") -> ForwardStats:
        """Parse aggregate execution counters and reject malformed mode, backend, or speculative statistics."""

        data = _map(value, where)

        def counter_map(name: str) -> dict[str, int]:
            """Parse one string-keyed map of nonnegative execution counters."""

            values = _map(data.get(name, {}), f"{where}.{name}")
            return {
                _str(key, f"{where}.{name}.key"): _uint(raw, f"{where}.{name}.{key}")
                for key, raw in values.items()
            }

        map_fields = {
            name: counter_map(name)
            for name in (
                "mode_counts",
                "mode_tokens",
                "mode_us",
                "component_us",
                "attention_backend_counts",
                "cuda_graph_runtime_mode_counts",
                "spec_verify_path_counts",
            )
        }
        scalar_fields = {
            name: _uint(data.get(name, 0), f"{where}.{name}")
            for name in (
                "attention_launches",
                "attention_us",
                "cuda_graph_captures",
                "cuda_graph_replays",
                "cuda_graph_misses",
                "cuda_graph_fallbacks",
                "cuda_graph_unpadded_tokens",
                "cuda_graph_padded_tokens",
                "text_decode_token_relay_hits",
                "text_decode_token_relay_misses",
                "text_decode_position_relay_hits",
                "text_decode_position_relay_misses",
                "flashinfer_decode_plan_calls",
                "flashinfer_decode_plan_reuses",
                "flashinfer_decode_plan_rows",
                "flashinfer_decode_plan_indices",
                "flashinfer_decode_graph_plan_calls",
                "flashinfer_decode_graph_plan_reuses",
                "spec_verify_rows",
                "spec_verify_draft_tokens",
                "spec_verify_accepted_tokens",
                "spec_verify_rejected_tokens",
                "spec_verify_committed_tokens",
            )
        }
        return cls(
            mode_counts=map_fields["mode_counts"],
            mode_tokens=map_fields["mode_tokens"],
            mode_us=map_fields["mode_us"],
            component_us=map_fields["component_us"],
            attention_launches=scalar_fields["attention_launches"],
            attention_us=scalar_fields["attention_us"],
            attention_backend_counts=map_fields["attention_backend_counts"],
            cuda_graph_captures=scalar_fields["cuda_graph_captures"],
            cuda_graph_replays=scalar_fields["cuda_graph_replays"],
            cuda_graph_misses=scalar_fields["cuda_graph_misses"],
            cuda_graph_fallbacks=scalar_fields["cuda_graph_fallbacks"],
            cuda_graph_unpadded_tokens=scalar_fields["cuda_graph_unpadded_tokens"],
            cuda_graph_padded_tokens=scalar_fields["cuda_graph_padded_tokens"],
            cuda_graph_runtime_mode_counts=map_fields["cuda_graph_runtime_mode_counts"],
            text_decode_token_relay_hits=scalar_fields["text_decode_token_relay_hits"],
            text_decode_token_relay_misses=scalar_fields["text_decode_token_relay_misses"],
            text_decode_position_relay_hits=scalar_fields["text_decode_position_relay_hits"],
            text_decode_position_relay_misses=scalar_fields["text_decode_position_relay_misses"],
            flashinfer_decode_plan_calls=scalar_fields["flashinfer_decode_plan_calls"],
            flashinfer_decode_plan_reuses=scalar_fields["flashinfer_decode_plan_reuses"],
            flashinfer_decode_plan_rows=scalar_fields["flashinfer_decode_plan_rows"],
            flashinfer_decode_plan_indices=scalar_fields["flashinfer_decode_plan_indices"],
            flashinfer_decode_graph_plan_calls=scalar_fields["flashinfer_decode_graph_plan_calls"],
            flashinfer_decode_graph_plan_reuses=scalar_fields[
                "flashinfer_decode_graph_plan_reuses"
            ],
            spec_verify_rows=scalar_fields["spec_verify_rows"],
            spec_verify_draft_tokens=scalar_fields["spec_verify_draft_tokens"],
            spec_verify_accepted_tokens=scalar_fields["spec_verify_accepted_tokens"],
            spec_verify_rejected_tokens=scalar_fields["spec_verify_rejected_tokens"],
            spec_verify_committed_tokens=scalar_fields["spec_verify_committed_tokens"],
            spec_verify_path_counts=map_fields["spec_verify_path_counts"],
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize scalar and keyed execution counters using plain wire values."""

        return {
            name: dict(value) if isinstance(value, Mapping) else value
            for name, value in ((name, getattr(self, name)) for name in self.__dataclass_fields__)
        }


@dataclass(frozen=True, slots=True)
class BatchOutput:
    """One wire-ready response fragment containing only host-owned values."""

    batch_id: int
    run_id: int
    completions: tuple[RequestOutput, ...] = ()
    products: tuple[TensorPublication, ...] = ()
    registration: RegistrationAck = field(default_factory=RegistrationAck)
    worker_exec_us: int | None = None
    forward_stats: ForwardStats | None = None
    done: bool = True

    @classmethod
    def combine(cls, fragments: Sequence[BatchOutput]) -> BatchOutput:
        """Collect response fragments from one run without changing field ordering."""

        if not fragments:
            raise ValueError("batch output requires at least one fragment")
        first = fragments[0]
        if any(
            (value.batch_id, value.run_id) != (first.batch_id, first.run_id) for value in fragments
        ):
            raise invalid_descriptor("output fragments belong to different batches")
        payloads = tuple(
            value
            for value in fragments
            if value.completions
            or value.products
            or value.worker_exec_us is not None
            or value.forward_stats is not None
        )
        durations = tuple(
            value.worker_exec_us for value in payloads if value.worker_exec_us is not None
        )
        stats = tuple(value.forward_stats for value in payloads if value.forward_stats is not None)
        return cls(
            batch_id=first.batch_id,
            run_id=first.run_id,
            completions=tuple(output for value in fragments for output in value.completions),
            products=tuple(output for value in fragments for output in value.products),
            registration=RegistrationAck(
                visible=bool(payloads) and all(value.registration.visible for value in payloads)
            ),
            worker_exec_us=max(durations) if durations else None,
            forward_stats=ForwardStats.combine(stats) if stats else None,
            done=fragments[-1].done,
        )

    @classmethod
    def from_mapping(cls, value: object, where: str = "completion report") -> BatchOutput:
        """Parse the unchanged flat response fields without execution wrappers."""

        data = _map(value, where)
        return cls(
            batch_id=_uint(data.get("batch_id"), f"{where}.batch_id"),
            run_id=_uint(data.get("run_id"), f"{where}.run_id"),
            completions=tuple(
                RequestOutput.from_mapping(item, f"{where}.completions[{index}]")
                for index, item in enumerate(
                    _seq(data.get("completions", ()), f"{where}.completions")
                )
            ),
            products=tuple(
                TensorPublication.from_mapping(item, f"{where}.products[{index}]")
                for index, item in enumerate(_seq(data.get("products", ()), f"{where}.products"))
            ),
            registration=RegistrationAck.from_mapping(
                data.get("registration", {}), f"{where}.registration"
            ),
            worker_exec_us=_optional_uint(data.get("worker_exec_us"), f"{where}.worker_exec_us"),
            forward_stats=(
                None
                if data.get("forward_stats") is None
                else ForwardStats.from_mapping(data["forward_stats"], f"{where}.forward_stats")
            ),
            done=_bool(data.get("done", True), f"{where}.done"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode final values in protocol order; pending resources cannot enter this type."""

        return {
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "completions": [value.to_mapping() for value in self.completions],
            "products": [value.to_mapping() for value in self.products],
            "registration": self.registration.to_mapping(),
            "worker_exec_us": self.worker_exec_us,
            "forward_stats": None
            if self.forward_stats is None
            else self.forward_stats.to_mapping(),
            "done": self.done,
        }


# Each `_fast_*` helper recognizes the exact built-in IPC shape without
# allocating error-location strings. A non-matching value returns ``None`` so
# the caller applies the canonical validated constructor and its precise error.

_E = TypeVar("_E", bound=StrEnum)

_DTYPE_BY_VALUE: Mapping[str, DType] = DType._value2member_map_  # type: ignore[assignment]
_DRAW_LAYOUT_BY_VALUE: Mapping[str, DrawLayout] = DrawLayout._value2member_map_  # type: ignore[assignment]


def _enum(kind: type[_E], value: object, where: str) -> _E:
    """Decode and validate one string-backed enum value for a wire field."""

    if type(value) is str:
        member = kind._value2member_map_.get(value)
        if member is not None:
            return cast(_E, member)
        raise invalid_descriptor(f"{where} has unknown value {value!r}")
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    try:
        return kind(value)
    except ValueError:
        raise invalid_descriptor(f"{where} has unknown value {value!r}") from None


def _map(value: object, where: str) -> Mapping[str, Any]:
    """Require a decoded wire field to be a mapping."""

    if type(value) is dict:
        return value
    if not isinstance(value, Mapping):
        raise invalid_descriptor(f"{where} must be a map")
    return cast(Mapping[str, Any], value)


def _seq(value: object, where: str) -> Sequence[Any]:
    """Require a decoded wire field to be a non-string sequence."""

    kind = type(value)
    if kind is list or kind is tuple:
        return cast(Sequence[Any], value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{where} must be a list")
    return value


def _pair(value: object, where: str) -> Sequence[Any]:
    """Require a decoded wire field to contain exactly two values."""

    items = _seq(value, where)
    if len(items) != 2:
        raise invalid_descriptor(f"{where} must contain two values")
    return items


def _tagged(value: object, where: str) -> tuple[str, object]:
    """Extract a non-empty variant tag and its mapping payload."""

    data = _map(value, where)
    return _str(data.get("kind"), f"{where}.kind"), data.get("value")


def _str(value: object, where: str) -> str:
    """Require a decoded wire field to contain text."""

    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    return value


def _bool(value: object, where: str) -> bool:
    """Require a decoded wire field to contain a boolean."""

    if value is True or value is False:
        return value
    raise invalid_descriptor(f"{where} must be a bool")


def _uint(value: object, where: str) -> int:
    """Decode a non-negative integer wire field."""

    if type(value) is int and value >= 0:
        return value
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"{where} must be a non-negative integer")
    return value


def _optional_uint(value: object, where: str) -> int | None:
    """Decode an optional non-negative integer wire field."""

    return None if value is None else _uint(value, where)


def _float(value: object, where: str) -> float:
    """Decode a finite floating-point wire field while rejecting booleans."""

    kind = type(value)
    if kind is not float and kind is not int:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise invalid_descriptor(f"{where} must be a number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise invalid_descriptor(f"{where} must be finite")
    return result


def _uints(value: object, where: str) -> tuple[int, ...]:
    """Decode a sequence of non-negative integer wire values."""

    items = _seq(value, where)
    for item in items:
        if not (type(item) is int and item >= 0):
            return tuple(_uint(item, f"{where}[{index}]") for index, item in enumerate(items))
    return tuple(items)


def _ints(value: object, where: str) -> tuple[int, ...]:
    """Decode a sequence of integer wire values."""

    items = _seq(value, where)
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        raise invalid_descriptor(f"{where} must contain integers")
    return tuple(int(item) for item in items)


def _bytes(value: object, where: str) -> bytes:
    """Decode a bytes-like wire payload to immutable bytes."""

    if type(value) is bytes:
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    return bytes(_uints(value, where))


def _nonnegative(value: int, where: str) -> None:
    """Validate a decoded integer is non-negative."""

    _uint(value, where)


# Each returns the decoded record for a well-formed IPC value and ``None``
# otherwise; the caller uses validating decoders in declaration order so the
# first invalid field receives a precise diagnostic.
# Construction bypasses ``__init__``/``__post_init__`` only where the fast
# path itself enforces everything those validators check.


@lru_cache(maxsize=8192)
def _interned_request_key(engine_id: int, request_id: int, request_epoch: int) -> RequestKey:
    """Reuse an immutable request key for identical engine, request, and epoch values."""

    key = object.__new__(RequestKey)
    object.__setattr__(key, "engine_id", engine_id)
    object.__setattr__(key, "request_id", request_id)
    object.__setattr__(key, "request_epoch", request_epoch)
    object.__setattr__(key, "_hash_value", None)
    return key


def _fast_request_key(value: object) -> RequestKey | None:
    """Decode a trusted compact request-key mapping."""

    if type(value) is not dict:
        return None
    engine_id = value.get("engine_id")
    request_id = value.get("request_id")
    request_epoch = value.get("request_epoch")
    if (
        type(engine_id) is int
        and engine_id >= 0
        and type(request_id) is int
        and request_id >= 0
        and type(request_epoch) is int
        and request_epoch >= 0
    ):
        return _interned_request_key(engine_id, request_id, request_epoch)
    return None


@lru_cache(maxsize=1024)
def _interned_shape_bound(
    encoded_dims: tuple[tuple[bool, int], ...],
) -> ShapeBound:
    """Reuse an immutable shape bound for an already encoded dimension tuple."""

    shape = object.__new__(ShapeBound)
    object.__setattr__(
        shape,
        "dims",
        tuple(
            DeviceDim(extent) if device_actual else StaticDim(extent)
            for device_actual, extent in encoded_dims
        ),
    )
    return shape


def _fast_shape_bound(value: object) -> ShapeBound | None:
    """Decode a trusted compact shape-bound mapping without generic schema dispatch."""

    if type(value) is not dict:
        return None
    raw_dims = value.get("dims", ())
    kind = type(raw_dims)
    if kind is not list and kind is not tuple:
        return None
    dims: list[tuple[bool, int]] = []
    device_dims = 0
    for item in raw_dims:
        if type(item) is not dict:
            return None
        tag = item.get("kind")
        payload = item.get("value")
        if tag == "static" and type(tag) is str:
            if not (type(payload) is int and payload >= 0):
                return None
            dims.append((False, payload))
        elif tag == "device" and type(tag) is str:
            if type(payload) is not dict:
                return None
            bound = payload.get("max")
            if not (type(bound) is int and bound >= 0):
                return None
            dims.append((True, bound))
            device_dims += 1
        else:
            return None
    if device_dims > 1:
        # Delegate the invalid shape to the validating decoder.
        return None
    return _interned_shape_bound(tuple(dims))


def _fast_tensor_ref(value: object) -> TensorRef | None:
    """Decode a trusted compact product reference and its optional tensor bound."""

    if type(value) is not dict:
        return None
    request_key = _fast_request_key(value.get("request_key"))
    if request_key is None:
        return None
    producer_op_id = value.get("producer_op_id")
    output_index = value.get("output_index")
    generation = value.get("generation")
    if not (
        isinstance(producer_op_id, ComputationId)
        and type(output_index) is int
        and output_index >= 0
        and type(generation) is int
        and generation > 0
    ):
        return None
    raw_dtype = value.get("dtype")
    if type(raw_dtype) is not str:
        return None
    dtype = _DTYPE_BY_VALUE.get(raw_dtype)
    if dtype is None:
        return None
    shape_bound = _fast_shape_bound(value.get("shape_bound"))
    if shape_bound is None:
        return None
    reference = object.__new__(TensorRef)
    set_field = object.__setattr__
    set_field(reference, "request_key", request_key)
    set_field(reference, "producer_op_id", producer_op_id)
    set_field(reference, "output_index", output_index)
    set_field(reference, "generation", generation)
    set_field(reference, "dtype", dtype)
    set_field(reference, "shape_bound", shape_bound)
    set_field(reference, "_buffer_id", None)
    return reference


def _fast_tensor_refs(value: object) -> tuple[TensorRef, ...] | None:
    """Decode a trusted sequence of compact product references."""

    kind = type(value)
    if kind is not list and kind is not tuple:
        return None
    items = cast(list[object] | tuple[object, ...], value)
    references: list[TensorRef] = []
    for item in items:
        reference = _fast_tensor_ref(item)
        if reference is None:
            return None
        references.append(reference)
    return tuple(references)


@lru_cache(maxsize=256)
def _interned_bounds(
    max_tokens: int,
    max_kv_pages: int,
    max_latent_bytes: int,
    max_completion_bytes: int,
    max_transfer_bytes: int,
) -> Bounds:
    """Reuse immutable execution bounds for an identical capacity tuple."""

    bounds = object.__new__(Bounds)
    set_field = object.__setattr__
    set_field(bounds, "max_tokens", max_tokens)
    set_field(bounds, "max_kv_pages", max_kv_pages)
    set_field(bounds, "max_latent_bytes", max_latent_bytes)
    set_field(bounds, "max_completion_bytes", max_completion_bytes)
    set_field(bounds, "max_transfer_bytes", max_transfer_bytes)
    return bounds


def _fast_bounds(value: object) -> Bounds | None:
    """Decode trusted optional execution bounds from their compact wire representation."""

    if type(value) is not dict:
        return None
    max_tokens = value.get("max_tokens")
    max_kv_pages = value.get("max_kv_pages")
    max_latent_bytes = value.get("max_latent_bytes")
    max_completion_bytes = value.get("max_completion_bytes")
    max_transfer_bytes = value.get("max_transfer_bytes")
    if (
        type(max_tokens) is int
        and max_tokens >= 0
        and type(max_kv_pages) is int
        and max_kv_pages >= 0
        and type(max_latent_bytes) is int
        and max_latent_bytes >= 0
        and type(max_completion_bytes) is int
        and max_completion_bytes >= 0
        and type(max_transfer_bytes) is int
        and max_transfer_bytes >= 0
    ):
        return _interned_bounds(
            max_tokens,
            max_kv_pages,
            max_latent_bytes,
            max_completion_bytes,
            max_transfer_bytes,
        )
    return None


def _fast_rng(value: object) -> Rng | None:
    """Decode trusted optional RNG coordinates from their compact wire representation."""

    if type(value) is not dict:
        return None
    seed = value.get("seed")
    semantic_index_base = value.get("semantic_index_base")
    raw_layout = value.get("draw_layout")
    if not (
        type(seed) is int
        and seed >= 0
        and type(semantic_index_base) is int
        and semantic_index_base >= 0
        and type(raw_layout) is str
    ):
        return None
    draw_layout = _DRAW_LAYOUT_BY_VALUE.get(raw_layout)
    if draw_layout is None:
        return None
    rng = object.__new__(Rng)
    object.__setattr__(rng, "seed", seed)
    object.__setattr__(rng, "semantic_index_base", semantic_index_base)
    object.__setattr__(rng, "draw_layout", draw_layout)
    return rng


def _fast_uints(value: object) -> tuple[int, ...] | None:
    """Decode a trusted integer sequence while enforcing non-negative values."""

    kind = type(value)
    if kind is not list and kind is not tuple:
        return None
    items = cast(list[object] | tuple[object, ...], value)
    for item in items:
        if not (type(item) is int and item >= 0):
            return None
    return cast(tuple[int, ...], tuple(items))


__all__ = [name for name in globals() if not name.startswith("_")]


def _logprob_entries(value: object) -> tuple[tuple[int, float, int], ...]:
    """Read ranked scores carried directly by the result record."""

    entries = []
    for item in _seq(value, "logprob entries"):
        data = _map(item, "logprob entry")
        entries.append(
            (
                _uint(data.get("token_id"), "logprob.token_id"),
                _logprob_value(data.get("logprob")),
                _uint(data.get("rank"), "logprob.rank"),
            )
        )
    return tuple(entries)


def _logprob_value(value: object) -> float:
    """Read a score, including negative infinity for zero-probability tokens."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise invalid_descriptor("logprob value must be numeric")
    return float(value)
