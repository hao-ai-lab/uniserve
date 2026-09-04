"""Scheduler-to-worker execution records and their validation."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from typing import Any, ClassVar, Protocol, TypeAlias, TypeVar, cast

from ..foundation.errors import invalid_descriptor

MAX_TRANSFER_HANDLE_BYTES = 64 * 1024


class CompletionState(Protocol):
    """Structural interface for one pending output row."""

    def ready(self) -> bool:
        """Return whether the pending row can be finalized without blocking."""

        ...

    def finalize(self) -> object:
        """Materialize the completed row into its wire-ready payload."""

        ...


class LanePublication(Protocol):
    """One-shot request-state publication owned by a lane result."""

    def finish(self, completions: tuple[ModelOutput, ...]) -> None:
        """Atomically publish successor-visible request state from finalized lane completions."""

        ...

    def cancel(self) -> None:
        """Discard the unpublished request-state transition."""

        ...

    @property
    def successors_ready(self) -> bool:
        """Return whether the request-state transition is visible to successor operations."""

        ...


class TransferKind(StrEnum):
    """Identifies encoder, device-product, KV, and latent transfer payloads on the wire."""

    ENCODER = "encoder"
    DEVICE_PRODUCT = "device_product"
    KV = "kv"
    LATENT = "latent"


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
    """Identifies shared-memory storage and its optional readiness semaphore."""

    name: str
    ready_header_bytes: int
    ready_semaphore: str | None

    def __post_init__(self) -> None:
        """Validate shared-memory and optional readiness-semaphore names."""

        if not self.name or self.ready_header_bytes < 0:
            raise invalid_descriptor("shared-memory transfer handle is invalid")
        if self.ready_header_bytes and not self.ready_semaphore:
            raise invalid_descriptor("shared-memory transfer has no readiness semaphore")


@dataclass(frozen=True, slots=True)
class CudaIpcTransfer:
    """Carries CUDA IPC memory, reference-count, and event handles for one tensor view."""

    endpoint: str
    publication_id: str
    storage_handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    tensor_offset: int
    tensor_stride: tuple[int, ...]
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool
    ready_event_handle: bytes

    def __post_init__(self) -> None:
        """Validate CUDA IPC handles, device identity, allocation bounds, and view offsets."""

        if (
            not self.endpoint
            or not self.publication_id
            or not self.storage_handle
            or self.storage_size_bytes < 1
            or self.storage_offset_bytes < 0
            or self.tensor_offset < 0
            or not self.ref_counter_handle
            or self.ref_counter_offset < 0
            or not self.event_handle
            or not self.ready_event_handle
        ):
            raise invalid_descriptor("CUDA IPC transfer handle is incomplete")
        opaque_bytes = sum(
            len(value)
            for value in (
                self.storage_handle,
                self.ref_counter_handle,
                self.event_handle,
                self.ready_event_handle,
            )
        )
        if opaque_bytes > MAX_TRANSFER_HANDLE_BYTES:
            raise invalid_descriptor("CUDA IPC transfer handles exceed their byte bound")


TransferTransport: TypeAlias = LocalTransfer | PosixShmTransfer | CudaIpcTransfer


@dataclass(frozen=True, slots=True)
class TransferLocator:
    """Describes a typed tensor view and the transport-specific handle that owns its storage."""

    transport: TransferTransport
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    device: str

    def __post_init__(self) -> None:
        """Validate tensor geometry and ensure the handle matches its transport kind."""

        if (
            self.nbytes < 1
            or not self.dtype
            or not self.shape
            or any(extent < 1 for extent in self.shape)
            or not self.device
        ):
            raise invalid_descriptor("transfer locator has invalid tensor bounds")
        if isinstance(self.transport, CudaIpcTransfer) and len(self.transport.tensor_stride) != len(
            self.shape
        ):
            raise invalid_descriptor("CUDA IPC tensor stride does not match its shape")

    @classmethod
    def from_mapping(cls, value: object, where: str = "transfer locator") -> TransferLocator:
        """Parse a tensor locator and validate its transport handle, shape, dtype, and byte bounds."""

        data = _map(value, where)
        kind = _str(data.get("transport"), f"{where}.transport")
        if kind == "local":
            transport: TransferTransport = LocalTransfer(
                endpoint=_str(data.get("endpoint"), f"{where}.endpoint"),
                key=_uint(data.get("key"), f"{where}.key"),
            )
        elif kind == "posix_shm":
            raw_semaphore = data.get("ready_semaphore")
            transport = PosixShmTransfer(
                name=_str(data.get("name"), f"{where}.name"),
                ready_header_bytes=_uint(
                    data.get("ready_header_bytes", 0), f"{where}.ready_header_bytes"
                ),
                ready_semaphore=(
                    None
                    if raw_semaphore is None
                    else _str(raw_semaphore, f"{where}.ready_semaphore")
                ),
            )
        elif kind == "cuda_ipc":
            transport = CudaIpcTransfer(
                endpoint=_str(data.get("endpoint"), f"{where}.endpoint"),
                publication_id=_str(
                    data.get("publication_id"), f"{where}.publication_id"
                ),
                storage_handle=_bytes(data.get("storage_handle"), f"{where}.storage_handle"),
                storage_size_bytes=_uint(
                    data.get("storage_size_bytes"), f"{where}.storage_size_bytes"
                ),
                storage_offset_bytes=_uint(
                    data.get("storage_offset_bytes"), f"{where}.storage_offset_bytes"
                ),
                tensor_offset=_uint(data.get("tensor_offset"), f"{where}.tensor_offset"),
                tensor_stride=tuple(
                    _ints(data.get("tensor_stride"), f"{where}.tensor_stride")
                ),
                ref_counter_handle=_bytes(
                    data.get("ref_counter_handle"), f"{where}.ref_counter_handle"
                ),
                ref_counter_offset=_uint(
                    data.get("ref_counter_offset"), f"{where}.ref_counter_offset"
                ),
                event_handle=_bytes(data.get("event_handle"), f"{where}.event_handle"),
                event_sync_required=_bool(
                    data.get("event_sync_required", False),
                    f"{where}.event_sync_required",
                ),
                ready_event_handle=_bytes(
                    data.get("ready_event_handle"), f"{where}.ready_event_handle"
                ),
            )
        else:
            raise invalid_descriptor(f"{where}.transport is invalid")
        return cls(
            transport=transport,
            nbytes=_uint(data.get("nbytes"), f"{where}.nbytes"),
            dtype=_str(data.get("dtype"), f"{where}.dtype"),
            shape=tuple(_uints(data.get("shape"), f"{where}.shape")),
            device=_str(data.get("device"), f"{where}.device"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode the tensor geometry and transport-specific handle as a wire mapping."""

        output: dict[str, object] = {
            "nbytes": self.nbytes,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "device": self.device,
        }
        transport = self.transport
        if isinstance(transport, LocalTransfer):
            output.update(transport="local", endpoint=transport.endpoint, key=transport.key)
        elif isinstance(transport, PosixShmTransfer):
            output.update(
                transport="posix_shm",
                name=transport.name,
                ready_header_bytes=transport.ready_header_bytes,
                ready_semaphore=transport.ready_semaphore,
            )
        else:
            output.update(
                transport="cuda_ipc",
                endpoint=transport.endpoint,
                publication_id=transport.publication_id,
                storage_handle=transport.storage_handle,
                storage_size_bytes=transport.storage_size_bytes,
                storage_offset_bytes=transport.storage_offset_bytes,
                tensor_offset=transport.tensor_offset,
                tensor_stride=list(transport.tensor_stride),
                ref_counter_handle=transport.ref_counter_handle,
                ref_counter_offset=transport.ref_counter_offset,
                event_handle=transport.event_handle,
                event_sync_required=transport.event_sync_required,
                ready_event_handle=transport.ready_event_handle,
            )
        return output


@dataclass(frozen=True, slots=True)
class EncoderTransferValue:
    """Describes the generation, media geometry, payload encoding, and location of transferred encoder features."""

    generation: int
    height: int
    width: int
    payload_kind: str
    locator: TransferLocator


@dataclass(frozen=True, slots=True)
class DeviceProductTransferValue:
    """Describes the generation, media geometry, numeric range, and location of a transferred device product."""

    generation: int
    height: int
    width: int
    value_range: str
    locator: TransferLocator


@dataclass(frozen=True, slots=True)
class KvTransferValue:
    """Describes a versioned KV extent, its page locators, and source-to-destination checkpoint relation."""

    generation: int
    locators: tuple[TransferLocator, ...]
    source: Checkpoint
    destination: str
    base: Checkpoint | None
    base_extent: int
    published_extent: int
    group_id: int
    scale_identity: str


@dataclass(frozen=True, slots=True)
class LatentTransferValue:
    """Describes a generation step and media geometry for transferred latent storage."""

    generation: int
    height: int
    width: int
    latent_units: int
    step: int
    locator: TransferLocator


TransferValue: TypeAlias = (
    EncoderTransferValue
    | DeviceProductTransferValue
    | KvTransferValue
    | LatentTransferValue
)


@dataclass(frozen=True, slots=True)
class TransferHandle:
    """Wraps one typed transfer descriptor for bounded wire encoding and decoding."""

    value: TransferValue

    @property
    def kind(self) -> TransferKind:
        """Return the semantic transfer kind selected by the wrapped descriptor."""

        if isinstance(self.value, EncoderTransferValue):
            return TransferKind.ENCODER
        if isinstance(self.value, DeviceProductTransferValue):
            return TransferKind.DEVICE_PRODUCT
        if isinstance(self.value, KvTransferValue):
            return TransferKind.KV
        return TransferKind.LATENT

    def encoded_size_bound(self) -> int:
        """Calculate a conservative wire-size bound for the nested transport handles."""

        typed = self.value
        locators = typed.locators if isinstance(typed, KvTransferValue) else (typed.locator,)
        size = 512
        for locator in locators:
            size += 64 + len(locator.dtype) + len(locator.device) + 8 * len(locator.shape)
            transport = locator.transport
            if isinstance(transport, LocalTransfer):
                size += len(transport.endpoint) + 16
            elif isinstance(transport, PosixShmTransfer):
                size += len(transport.name) + len(transport.ready_semaphore or "") + 16
            else:
                size += (
                    len(transport.endpoint)
                    + len(transport.publication_id)
                    + len(transport.storage_handle)
                    + len(transport.ref_counter_handle)
                    + len(transport.event_handle)
                    + len(transport.ready_event_handle)
                    + 8 * len(transport.tensor_stride)
                    + 64
                )
        if size > MAX_TRANSFER_HANDLE_BYTES:
            raise invalid_descriptor("transfer handle exceeds its byte bound")
        return size

    @classmethod
    def from_mapping(cls, value: object, where: str = "transfer handle") -> TransferHandle:
        """Parse and validate the kind-specific encoder, product, KV, or latent transfer descriptor."""

        data = _map(value, where)
        kind = _enum(TransferKind, data.get("kind"), f"{where}.kind")
        payload = _map(data.get("value"), f"{where}.value")
        generation = _uint(payload.get("generation"), f"{where}.value.generation")
        if kind is TransferKind.ENCODER:
            typed: TransferValue = EncoderTransferValue(
                generation=generation,
                height=_uint(payload.get("height"), f"{where}.value.height"),
                width=_uint(payload.get("width"), f"{where}.value.width"),
                payload_kind=_str(
                    payload.get("payload_kind"), f"{where}.value.payload_kind"
                ),
                locator=TransferLocator.from_mapping(
                    payload.get("locator"), f"{where}.value.locator"
                ),
            )
        elif kind is TransferKind.DEVICE_PRODUCT:
            typed = DeviceProductTransferValue(
                generation=generation,
                height=_uint(payload.get("height", 0), f"{where}.value.height"),
                width=_uint(payload.get("width", 0), f"{where}.value.width"),
                value_range=_str(
                    payload.get("value_range", ""), f"{where}.value.value_range"
                ),
                locator=TransferLocator.from_mapping(
                    payload.get("locator"), f"{where}.value.locator"
                ),
            )
        elif kind is TransferKind.KV:
            raw_base = payload.get("base")
            typed = KvTransferValue(
                generation=generation,
                locators=tuple(
                    TransferLocator.from_mapping(item, f"{where}.value.locators[{index}]")
                    for index, item in enumerate(
                        _seq(payload.get("locators"), f"{where}.value.locators")
                    )
                ),
                source=Checkpoint.from_mapping(payload.get("source"), f"{where}.value.source"),
                destination=_str(
                    payload.get("destination"), f"{where}.value.destination"
                ),
                base=(
                    None
                    if raw_base is None
                    else Checkpoint.from_mapping(raw_base, f"{where}.value.base")
                ),
                base_extent=_uint(
                    payload.get("base_extent", 0), f"{where}.value.base_extent"
                ),
                published_extent=_uint(
                    payload.get("published_extent", 0),
                    f"{where}.value.published_extent",
                ),
                group_id=_uint(payload.get("group_id", 0), f"{where}.value.group_id"),
                scale_identity=_str(
                    payload.get("scale_identity"), f"{where}.value.scale_identity"
                ),
            )
        else:
            typed = LatentTransferValue(
                generation=generation,
                height=_uint(payload.get("height"), f"{where}.value.height"),
                width=_uint(payload.get("width"), f"{where}.value.width"),
                latent_units=_uint(
                    payload.get("latent_units"), f"{where}.value.latent_units"
                ),
                step=_uint(payload.get("step", 0), f"{where}.value.step"),
                locator=TransferLocator.from_mapping(
                    payload.get("locator"), f"{where}.value.locator"
                ),
            )
        return cls(value=typed)

    def to_mapping(self) -> dict[str, object]:
        """Encode the wrapped kind-specific transfer descriptor for IPC."""

        typed = self.value
        if isinstance(typed, EncoderTransferValue):
            value: dict[str, object] = {
                "generation": typed.generation,
                "height": typed.height,
                "width": typed.width,
                "payload_kind": typed.payload_kind,
                "locator": typed.locator.to_mapping(),
            }
        elif isinstance(typed, DeviceProductTransferValue):
            value = {
                "generation": typed.generation,
                "height": typed.height,
                "width": typed.width,
                "value_range": typed.value_range,
                "locator": typed.locator.to_mapping(),
            }
        elif isinstance(typed, KvTransferValue):
            value = {
                "generation": typed.generation,
                "locators": [locator.to_mapping() for locator in typed.locators],
                "source": typed.source.to_mapping(),
                "destination": typed.destination,
                "base": None if typed.base is None else typed.base.to_mapping(),
                "base_extent": typed.base_extent,
                "published_extent": typed.published_extent,
                "group_id": typed.group_id,
                "scale_identity": typed.scale_identity,
            }
        else:
            value = {
                "generation": typed.generation,
                "height": typed.height,
                "width": typed.width,
                "latent_units": typed.latent_units,
                "step": typed.step,
                "locator": typed.locator.to_mapping(),
            }
        return {"kind": self.kind.value, "value": value}


class TokenMode(StrEnum):
    """Selects prompt extension, single-token decode, or speculative verification."""

    EXTEND = "extend"
    DECODE = "decode"
    VERIFY = "verify"


class EncodeMode(StrEnum):
    """Selects vision-feature or latent-feature encoding."""

    VISION = "vision"
    LATENT = "latent"


class TransferMode(StrEnum):
    """Selects product movement, KV publication, or KV installation."""

    PRODUCT = "product"
    KV_PUBLISH = "kv_publish"
    KV_INSTALL = "kv_install"


class MediaMode(StrEnum):
    """Selects diffusion preparation, denoising, or reconstruction."""

    PREPARE = "prepare"
    DENOISE = "denoise"
    RECONSTRUCT = "reconstruct"


class OpKind(StrEnum):
    """Stable logical operation kinds advertised to the control plane."""

    AR_EXTEND = "ar_extend"
    AR_DECODE = "ar_decode"
    AR_VERIFY = "ar_verify"
    ENCODER_EXECUTE = "encoder_execute"
    DIFFUSION_PREPARE = "diffusion_prepare"
    DIFFUSION_STEP = "diffusion_step"
    DIFFUSION_DECODE = "diffusion_decode"


class RunKind(StrEnum):
    """Defines every executable scheduler operation and its canonical model or transfer route."""

    AR_EXTEND = "ar_extend"
    AR_DECODE = "ar_decode"
    AR_VERIFY = "ar_verify"
    ENCODER_VISION = "encoder_vision"
    ENCODER_LATENT = "encoder_latent"
    TRANSFER_PRODUCT = "transfer_product"
    TRANSFER_KV_PUBLISH = "transfer_kv_publish"
    TRANSFER_KV_INSTALL = "transfer_kv_install"
    DIFFUSION_PREPARE = "diffusion_prepare"
    DIFFUSION_STEP = "diffusion_step"
    DIFFUSION_FINALIZE = "diffusion_finalize"
    DIFFUSION_DECODE = "diffusion_decode"

    @property
    def op_kind(self) -> OpKind | None:
        """Map an executable route to its control-plane capability family."""

        return {
            RunKind.AR_EXTEND: OpKind.AR_EXTEND,
            RunKind.AR_DECODE: OpKind.AR_DECODE,
            RunKind.AR_VERIFY: OpKind.AR_VERIFY,
            RunKind.ENCODER_VISION: OpKind.ENCODER_EXECUTE,
            RunKind.ENCODER_LATENT: OpKind.ENCODER_EXECUTE,
            RunKind.TRANSFER_PRODUCT: OpKind.ENCODER_EXECUTE,
            RunKind.TRANSFER_KV_PUBLISH: OpKind.AR_DECODE,
            RunKind.TRANSFER_KV_INSTALL: OpKind.DIFFUSION_STEP,
            RunKind.DIFFUSION_PREPARE: OpKind.DIFFUSION_PREPARE,
            RunKind.DIFFUSION_STEP: OpKind.DIFFUSION_STEP,
            RunKind.DIFFUSION_FINALIZE: OpKind.DIFFUSION_DECODE,
            RunKind.DIFFUSION_DECODE: OpKind.DIFFUSION_DECODE,
        }.get(self)

    @property
    def advances_state(self) -> bool:
        """Indicate whether successful execution creates a new request checkpoint."""

        return self in _STATE_ADVANCING_WORK

    @property
    def requires_fixed_parent(self) -> bool:
        """Indicate whether host-resolved parent state is required before execution."""

        return self is RunKind.TRANSFER_KV_PUBLISH

    @property
    def token_mode(self) -> TokenMode | None:
        """Resolve the autoregressive token mode for token routes."""

        if self is RunKind.AR_EXTEND:
            return TokenMode.EXTEND
        if self is RunKind.AR_DECODE:
            return TokenMode.DECODE
        if self is RunKind.AR_VERIFY:
            return TokenMode.VERIFY
        return None

    @property
    def encode_mode(self) -> EncodeMode | None:
        """Resolve the encoder modality for encoder routes."""

        if self is RunKind.ENCODER_VISION:
            return EncodeMode.VISION
        if self is RunKind.ENCODER_LATENT:
            return EncodeMode.LATENT
        return None

    @property
    def transfer_mode(self) -> TransferMode | None:
        """Resolve the data-movement mode for transfer routes."""

        if self is RunKind.TRANSFER_PRODUCT:
            return TransferMode.PRODUCT
        if self is RunKind.TRANSFER_KV_PUBLISH:
            return TransferMode.KV_PUBLISH
        if self is RunKind.TRANSFER_KV_INSTALL:
            return TransferMode.KV_INSTALL
        return None

    @property
    def media_mode(self) -> MediaMode | None:
        """Resolve the diffusion media phase for trajectory routes."""

        if self is RunKind.DIFFUSION_PREPARE:
            return MediaMode.PREPARE
        if self is RunKind.DIFFUSION_STEP:
            return MediaMode.DENOISE
        if self is RunKind.DIFFUSION_DECODE:
            return MediaMode.RECONSTRUCT
        return None

    @classmethod
    def token(cls, mode: TokenMode) -> RunKind:
        """Select the executable autoregressive route for a token mode."""

        return {
            TokenMode.EXTEND: cls.AR_EXTEND,
            TokenMode.DECODE: cls.AR_DECODE,
            TokenMode.VERIFY: cls.AR_VERIFY,
        }[mode]

    @classmethod
    def encode(cls, mode: EncodeMode) -> RunKind:
        """Select the executable encoder route for a modality."""

        return {
            EncodeMode.VISION: cls.ENCODER_VISION,
            EncodeMode.LATENT: cls.ENCODER_LATENT,
        }[mode]

    @classmethod
    def transfer(cls, mode: TransferMode) -> RunKind:
        """Select the executable data-movement route for a transfer mode."""

        return {
            TransferMode.PRODUCT: cls.TRANSFER_PRODUCT,
            TransferMode.KV_PUBLISH: cls.TRANSFER_KV_PUBLISH,
            TransferMode.KV_INSTALL: cls.TRANSFER_KV_INSTALL,
        }[mode]

    @classmethod
    def media(cls, mode: MediaMode) -> RunKind:
        """Select the executable diffusion route for a media phase."""

        return {
            MediaMode.PREPARE: cls.DIFFUSION_PREPARE,
            MediaMode.DENOISE: cls.DIFFUSION_STEP,
            MediaMode.RECONSTRUCT: cls.DIFFUSION_DECODE,
        }[mode]


def logical_op_kinds(kinds: Sequence[RunKind]) -> tuple[OpKind, ...]:
    """Return unique logical capabilities in canonical declaration order."""

    available = {kind.op_kind for kind in kinds if kind.op_kind is not None}
    return tuple(kind for kind in OpKind if kind in available)


class Domain(StrEnum):
    """Separates prefill, autoregressive decode, and diffusion-flow execution domains."""

    PREFILL = "prefill"
    DECODE = "decode"
    FLOW = "flow"


_DOMAIN_BY_WORK_VARIANT = {
    RunKind.AR_EXTEND: Domain.PREFILL,
    RunKind.AR_DECODE: Domain.DECODE,
    RunKind.AR_VERIFY: Domain.DECODE,
    RunKind.ENCODER_VISION: Domain.PREFILL,
    RunKind.ENCODER_LATENT: Domain.PREFILL,
    RunKind.TRANSFER_PRODUCT: Domain.PREFILL,
    RunKind.TRANSFER_KV_PUBLISH: Domain.PREFILL,
    RunKind.TRANSFER_KV_INSTALL: Domain.PREFILL,
    RunKind.DIFFUSION_PREPARE: Domain.FLOW,
    RunKind.DIFFUSION_STEP: Domain.FLOW,
    RunKind.DIFFUSION_FINALIZE: Domain.FLOW,
    RunKind.DIFFUSION_DECODE: Domain.FLOW,
}


def execution_domain(kind: RunKind) -> Domain:
    """Map an executable operation kind to its prefill, decode, or flow scheduling domain."""

    return _DOMAIN_BY_WORK_VARIANT[kind]


class AttentionRegime(StrEnum):
    """Specifies whether an operation uses no attention, causal attention, bidirectional attention, or a hybrid mask."""

    NONE = "none"
    CAUSAL = "causal"
    BIDIRECTIONAL = "bidirectional"
    HYBRID = "hybrid"


class ProductKind(StrEnum):
    """Identifies the semantic value carried by an operation input or output edge."""

    TOKEN = "token"
    LOGPROB = "logprob"
    VISION_FEATURE = "vision_feature"
    LATENT_FEATURE = "latent_feature"
    KV = "kv"
    LATENT = "latent"
    ARTIFACT = "artifact"
    COMPLETION = "completion"
    SAMPLING_STATE = "sampling_state"
    SELECTED_POINT = "selected_point"


class StorageClass(StrEnum):
    """Identifies the physical storage owner used for an execution product."""

    DEVICE_TENSOR = "device_tensor"
    REQUEST_RELAY = "request_relay"
    PAGED_KV = "paged_kv"
    LATENT_ARENA = "latent_arena"
    HOST_STAGING = "host_staging"
    PINNED_OUTPUT = "pinned_output"


class DType(StrEnum):
    """Defines wire-stable scalar dtypes supported by scheduler descriptors."""

    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    I32 = "i32"
    I64 = "i64"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"


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


class Disposition(StrEnum):
    """Controls whether a committed product is published, retained privately, or discarded."""

    PUBLISH = "publish"
    RETAIN = "retain"
    DISCARD = "discard"


class CloseReason(StrEnum):
    """Records why a request reached its terminal lifecycle state."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"
    PREEMPTED = "preempted"


_STATE_ADVANCING_WORK = frozenset(
    {
        RunKind.AR_EXTEND,
        RunKind.AR_DECODE,
        RunKind.AR_VERIFY,
        RunKind.DIFFUSION_PREPARE,
        RunKind.DIFFUSION_STEP,
        RunKind.DIFFUSION_DECODE,
    }
)

_FORWARD_MODE_BY_VALUE = {member.value: member for member in RunKind}


def native_run(
    batch_id: int,
    run_id: int,
    collective_seq: int,
    operations: tuple[Operation, ...],
    block_tables: tuple[BlockTable, ...],
    new_cache_pages: tuple[CachePageAllocation, ...],
    forward_rows: tuple[RowGeometry, ...],
    latent_placements: Sequence[object],
    decode_placements: Sequence[object],
    buffer_placements: Sequence[object],
    commands: tuple[BatchCommand, ...],
    input_products: Sequence[object],
) -> Run:
    """Assemble a physical run from transport-constructed members."""

    run = object.__new__(Run)
    set_field = object.__setattr__
    set_field(run, "batch_id", batch_id)
    set_field(run, "run_id", run_id)
    set_field(run, "collective_seq", collective_seq)
    set_field(run, "operations", operations)
    set_field(run, "block_tables", block_tables)
    set_field(run, "new_cache_pages", new_cache_pages)
    set_field(run, "forward_rows", forward_rows)
    set_field(
        run,
        "latent_placements",
        tuple(
            LatentPlacement.from_mapping(item, f"run.latent_placements[{index}]")
            for index, item in enumerate(latent_placements)
        ),
    )
    set_field(
        run,
        "decode_placements",
        tuple(
            DecodePlacement.from_mapping(item, f"run.decode_placements[{index}]")
            for index, item in enumerate(decode_placements)
        ),
    )
    set_field(
        run,
        "buffer_placements",
        tuple(
            BufferPlacement.from_mapping(item, f"run.buffer_placements[{index}]")
            for index, item in enumerate(buffer_placements)
        ),
    )
    set_field(run, "commands", commands)
    set_field(
        run,
        "input_products",
        tuple(
            ProductPayload.from_mapping(item, f"run.input_products[{index}]")
            for index, item in enumerate(input_products)
        ),
    )
    set_field(run, "lanes", ())
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


@dataclass(frozen=True, slots=True)
class RequestKey:
    """Identifies one request epoch within a scheduler authority."""

    authority_id: int
    request_id: int
    epoch: int

    def __post_init__(self) -> None:
        """Validate the non-negative request identifier and epoch."""

        _nonnegative(self.authority_id, "request_key.authority_id")
        _nonnegative(self.request_id, "request_key.request_id")
        _nonnegative(self.epoch, "request_key.epoch")

    @classmethod
    def from_mapping(cls, value: object, where: str = "request_key") -> RequestKey:
        """Parse and validate a scheduler authority, request id, and admission epoch."""

        key = _fast_request_key(value)
        if key is not None:
            return key
        data = _map(value, where)
        return cls(
            authority_id=_uint(data.get("authority_id"), f"{where}.authority_id"),
            request_id=_uint(data.get("request_id"), f"{where}.request_id"),
            epoch=_uint(data.get("epoch"), f"{where}.epoch"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the complete request-generation identity for IPC."""

        return {
            "authority_id": self.authority_id,
            "request_id": self.request_id,
            "epoch": self.epoch,
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
class PointRange:
    """Defines a half-open range of speculative or diffusion checkpoint points."""

    base_point: int = 0
    max_points: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "point_range") -> PointRange:
        """Parse a half-open checkpoint-point capacity range."""

        point_range = _fast_point_range(value)
        if point_range is not None:
            return point_range
        data = _map(value, where)
        return cls(
            base_point=_uint(data.get("base_point"), f"{where}.base_point"),
            max_points=_uint(data.get("max_points"), f"{where}.max_points"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize checkpoint base and maximum point count for IPC."""

        return {"base_point": self.base_point, "max_points": self.max_points}


@dataclass(frozen=True, slots=True)
class ProductRef:
    """Identifies a typed, versioned operation output and its bounded physical storage contract."""

    request_key: RequestKey
    producer_op_id: int
    output_index: int
    generation: int
    kind: ProductKind
    storage_class: StorageClass
    dtype: DType
    shape_bound: ShapeBound
    point_range: PointRange

    def __post_init__(self) -> None:
        """Validate product identity, version, kind, and bounded tensor contract."""

        if self.generation < 1:
            raise invalid_descriptor("product reference has no logical generation")
        self.shape_bound.__post_init__()

    @property
    def buffer_id(self) -> BufferId:
        """Derive the persistent-buffer identity carried by this product generation."""

        return BufferId(
            owner=self.request_key,
            producer_op_id=self.producer_op_id,
            output_index=self.output_index,
            generation=self.generation,
        )

    @property
    def max_bytes(self) -> int:
        """Return the maximum physical bytes allowed by this product’s shape and dtype."""

        element_bytes = {
            DType.U8: 1,
            DType.U16: 2,
            DType.U32: 4,
            DType.I32: 4,
            DType.I64: 8,
            DType.F16: 2,
            DType.BF16: 2,
            DType.F32: 4,
        }[self.dtype]
        return self.shape_bound.max_elements * element_bytes

    def uses_persistent_buffer(self) -> bool:
        """Return whether the product is assigned to scheduler-managed persistent storage."""

        if self.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}:
            return True
        return self.kind is ProductKind.ARTIFACT and self.storage_class in {
            StorageClass.DEVICE_TENSOR,
            StorageClass.LATENT_ARENA,
        }

    @classmethod
    def from_mapping(cls, value: object, where: str = "product_ref") -> ProductRef:
        """Parse and validate a typed logical product and its storage bounds."""

        reference = _fast_product_ref(value)
        if reference is not None:
            return reference
        data = _map(value, where)
        return cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            producer_op_id=_uint(data.get("producer_op_id"), f"{where}.producer_op_id"),
            output_index=_uint(data.get("output_index"), f"{where}.output_index"),
            generation=_uint(data.get("generation"), f"{where}.generation"),
            kind=_enum(ProductKind, data.get("kind"), f"{where}.kind"),
            storage_class=_enum(StorageClass, data.get("storage_class"), f"{where}.storage_class"),
            dtype=_enum(DType, data.get("dtype"), f"{where}.dtype"),
            shape_bound=ShapeBound.from_mapping(data.get("shape_bound"), f"{where}.shape_bound"),
            point_range=PointRange.from_mapping(data.get("point_range"), f"{where}.point_range"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the complete logical product contract for IPC."""

        return {
            "request_key": self.request_key.to_mapping(),
            "producer_op_id": self.producer_op_id,
            "output_index": self.output_index,
            "generation": self.generation,
            "kind": self.kind.value,
            "storage_class": self.storage_class.value,
            "dtype": self.dtype.value,
            "shape_bound": self.shape_bound.to_mapping(),
            "point_range": self.point_range.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class BufferId:
    """Identifies a versioned operation output buffer and its owning request."""

    owner: RequestKey
    producer_op_id: int
    output_index: int
    generation: int

    def __post_init__(self) -> None:
        """Validate the owning request, operation identifier, and version."""

        if self.generation < 1:
            raise invalid_descriptor("buffer id has no logical generation")

    @classmethod
    def from_mapping(cls, value: object, where: str = "buffer_id") -> BufferId:
        """Parse a versioned persistent-buffer identity from the wire schema."""

        data = _map(value, where)
        return cls(
            owner=RequestKey.from_mapping(data.get("owner"), f"{where}.owner"),
            producer_op_id=_uint(data.get("producer_op_id"), f"{where}.producer_op_id"),
            output_index=_uint(data.get("output_index"), f"{where}.output_index"),
            generation=_uint(data.get("generation"), f"{where}.generation"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize persistent-buffer ownership and generation fields for IPC."""

        return {
            "owner": self.owner.to_mapping(),
            "producer_op_id": self.producer_op_id,
            "output_index": self.output_index,
            "generation": self.generation,
        }


@dataclass(frozen=True, slots=True)
class FixedCheckpoint:
    """Selects a fixed speculative checkpoint by point index."""

    point: int

    @property
    def point_index(self) -> int:
        """Expose the fixed checkpoint point selected by the scheduler."""

        return self.point


@dataclass(frozen=True, slots=True)
class DeviceSelected:
    """Marks a checkpoint whose committed point is selected by device execution."""

    pass


CheckpointPoint: TypeAlias = FixedCheckpoint | DeviceSelected


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Identifies an operation checkpoint using either a fixed or device-selected point."""

    op_id: int
    point: CheckpointPoint

    def is_fixed(self) -> bool:
        """Indicate whether the checkpoint point is already host-resolved."""

        return isinstance(self.point, FixedCheckpoint)

    @classmethod
    def from_mapping(cls, value: object, where: str = "checkpoint") -> Checkpoint:
        """Parse an operation checkpoint with fixed or device-selected point semantics."""

        reference = _fast_version_ref(value)
        if reference is not None:
            return reference
        data = _map(value, where)
        kind, payload = _tagged(data.get("point"), f"{where}.point")
        if kind == "fixed":
            point: CheckpointPoint = FixedCheckpoint(
                _uint(payload, f"{where}.point.value")
            )
        elif kind == "device_selected":
            point = DeviceSelected()
        else:
            raise invalid_descriptor(f"{where}.point has unknown variant {kind!r}")
        return cls(op_id=_uint(data.get("op_id"), f"{where}.op_id"), point=point)

    def to_mapping(self) -> dict[str, object]:
        """Serialize the checkpoint point as its tagged wire variant."""

        point = (
            {"kind": "fixed", "value": self.point.point}
            if isinstance(self.point, FixedCheckpoint)
            else {"kind": "device_selected", "value": None}
        )
        return {"op_id": self.op_id, "point": point}


@dataclass(frozen=True, slots=True)
class Bounds:
    """Caps tokens, pages, latent bytes, completion bytes, and transfer bytes for one operation."""

    max_points: int = 0
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
            max_points=_uint(data.get("max_points"), f"{where}.max_points"),
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
            "max_points": self.max_points,
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
class _OpPayload:
    """Holds the shared bounds, dataflow edges, predicate, RNG, and control sequence for an operation family."""

    family: ClassVar[str]
    bounds: Bounds
    inputs: tuple[ProductRef, ...]
    outputs: tuple[ProductRef, ...]
    predicate: ProductRef | None
    rng: Rng | None
    control_seq: int

    def to_mapping(self) -> dict[str, object]:
        """Serialize fields shared by every operation-family payload."""

        return {
            "family": self.family,
            "bounds": self.bounds.to_mapping(),
            "inputs": [product.to_mapping() for product in self.inputs],
            "outputs": [product.to_mapping() for product in self.outputs],
            "predicate": None if self.predicate is None else self.predicate.to_mapping(),
            "rng": None if self.rng is None else self.rng.to_mapping(),
            "control_seq": self.control_seq,
        }


class ArOpPayload(_OpPayload):
    """Carries the shared payload contract for autoregressive operations."""

    family = "ar"


class EncoderOpPayload(_OpPayload):
    """Carries the shared payload contract for encoder operations."""

    family = "encoder"


class DiffusionOpPayload(_OpPayload):
    """Carries the shared payload contract for diffusion operations."""

    family = "diffusion"


class TransferOpPayload(_OpPayload):
    """Carries the shared payload contract for transfer operations."""

    family = "transfer"


OpPayload: TypeAlias = ArOpPayload | EncoderOpPayload | DiffusionOpPayload | TransferOpPayload


def _payload_class(kind: RunKind) -> type[_OpPayload]:
    """Return the family-specific payload class required by an operation kind."""

    if kind in {RunKind.AR_EXTEND, RunKind.AR_DECODE, RunKind.AR_VERIFY}:
        return ArOpPayload
    if kind in {RunKind.ENCODER_VISION, RunKind.ENCODER_LATENT}:
        return EncoderOpPayload
    if kind in {
        RunKind.DIFFUSION_PREPARE,
        RunKind.DIFFUSION_STEP,
        RunKind.DIFFUSION_DECODE,
        RunKind.DIFFUSION_FINALIZE,
    }:
        return DiffusionOpPayload
    return TransferOpPayload


@dataclass(frozen=True, slots=True)
class Operation:
    """Binds one typed operation payload to its request identity, parent checkpoint, and executable kind."""

    request_key: RequestKey
    op_id: int
    parent: Checkpoint
    kind: RunKind
    payload: OpPayload

    @classmethod
    def registered(
        cls,
        *,
        request_key: RequestKey,
        op_id: int,
        parent: Checkpoint,
        kind: RunKind,
        bounds: Bounds,
        inputs: tuple[ProductRef, ...] = (),
        outputs: tuple[ProductRef, ...] = (),
        predicate: ProductRef | None = None,
        rng: Rng | None = None,
        control_seq: int = 0,
    ) -> Operation:
        """Construct an operation from already typed scheduler fields and the payload matching its family."""

        return cls(
            request_key=request_key,
            op_id=op_id,
            parent=parent,
            kind=kind,
            payload=_payload_class(kind)(bounds, inputs, outputs, predicate, rng, control_seq),
        )

    @property
    def domain(self) -> Domain:
        """Select the execution lane domain implied by this operation's route."""

        return execution_domain(self.kind)

    @property
    def advances_state(self) -> bool:
        """Indicate whether this operation publishes a new request checkpoint."""

        return self.kind.advances_state

    @property
    def bounds(self) -> Bounds:
        """Expose scheduler-enforced resource ceilings from the typed payload."""

        return self.payload.bounds

    @property
    def inputs(self) -> tuple[ProductRef, ...]:
        """Expose logical products consumed in descriptor order."""

        return self.payload.inputs

    @property
    def outputs(self) -> tuple[ProductRef, ...]:
        """Expose logical products produced in descriptor order."""

        return self.payload.outputs

    @property
    def predicate(self) -> ProductRef | None:
        """Expose the device decision that conditionally enables this operation."""

        return self.payload.predicate

    @property
    def rng(self) -> Rng | None:
        """Expose deterministic draw coordinates required by this operation."""

        return self.payload.rng

    @property
    def control_seq(self) -> int:
        """Expose the scheduler control sequence used for ordering checks."""

        return self.payload.control_seq

    def validate(self) -> None:
        """Enforce operation-family, parent, bound, dataflow, predicate, RNG, and control-sequence invariants."""

        if self.op_id < 1:
            raise invalid_descriptor("operation id must be positive")
        if not isinstance(self.payload, _payload_class(self.kind)):
            raise invalid_descriptor("operation payload family disagrees with its kind")
        if self.kind.requires_fixed_parent and not isinstance(self.parent.point, FixedCheckpoint):
            raise invalid_descriptor("operation requires a fixed semantic parent")
        output_indices: set[int] = set()
        for product in self.outputs:
            if product.request_key != self.request_key or product.producer_op_id != self.op_id:
                raise invalid_descriptor(
                    "an output product is not owned by its producing operation"
                )
            if product.generation < 1:
                raise invalid_descriptor("an output product has no logical generation")
            if product.point_range.max_points > max(1, self.bounds.max_points):
                raise invalid_descriptor("an output product exceeds the operation point bound")
            if (
                product.storage_class is StorageClass.LATENT_ARENA
                and product.max_bytes > self.bounds.max_latent_bytes
            ):
                raise invalid_descriptor(
                    "a latent-arena output exceeds the operation latent-byte bound"
                )
            if (
                product.storage_class in (StorageClass.HOST_STAGING, StorageClass.PINNED_OUTPUT)
                and product.max_bytes > self.bounds.max_completion_bytes
            ):
                raise invalid_descriptor(
                    "a host-visible output exceeds the operation completion-byte bound"
                )
            if (
                product.storage_class is StorageClass.PAGED_KV
                and product.max_bytes > self.bounds.max_transfer_bytes
            ):
                raise invalid_descriptor(
                    "a paged-KV output exceeds the operation transfer-byte bound"
                )
            if product.output_index in output_indices:
                raise invalid_descriptor("operation repeats an output index")
            output_indices.add(product.output_index)
        if any(
            product.request_key != self.request_key
            and product.kind not in (ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE)
            for product in self.inputs
        ):
            raise invalid_descriptor(
                "a request-local input product belongs to another request lineage"
            )
        if self.predicate is not None:
            if self.predicate.request_key != self.request_key:
                raise invalid_descriptor("operation predicate belongs to another request lineage")
            continuation_token = (
                self.predicate.kind is ProductKind.TOKEN
                and self.predicate.dtype is DType.U32
                and self.predicate.shape_bound.max_elements == 1
            )
            if (
                self.predicate.generation < 1
                or self.predicate.storage_class
                not in {StorageClass.DEVICE_TENSOR, StorageClass.REQUEST_RELAY}
                or not (self.predicate.kind is ProductKind.COMPLETION or continuation_token)
            ):
                raise invalid_descriptor(
                    "operation predicate is not a generation-tagged device decision product"
                )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "operation",
    ) -> Operation:
        """Parse a typed operation and validate its identity, parent checkpoint, and family payload."""

        # Field decoding follows declaration order with a no-allocation fast
        # path per field. Irregular values use the validating field decoders so
        # diagnostics identify the first invalid declaration.
        data = _map(value, where)
        get = data.get
        request_key = _fast_request_key(get("request_key"))
        if request_key is None:
            request_key = RequestKey.from_mapping(get("request_key"), f"{where}.request_key")
        op_id = get("op_id")
        if not (type(op_id) is int and op_id >= 0):
            op_id = _uint(op_id, f"{where}.op_id")
        parent = _fast_version_ref(get("parent"))
        if parent is None:
            parent = Checkpoint.from_mapping(get("parent"), f"{where}.parent")
        work = _fast_work(get("kind"))
        if work is None:
            work = _enum(RunKind, get("kind"), f"{where}.kind")
        payload_data = _map(get("payload"), f"{where}.payload")
        family = _str(payload_data.get("family"), f"{where}.payload.family")
        payload_class = _payload_class(work)
        if family != payload_class.family:
            raise invalid_descriptor(f"{where}.payload family disagrees with its kind")
        payload_get = payload_data.get
        bounds = _fast_bounds(payload_get("bounds"))
        if bounds is None:
            bounds = Bounds.from_mapping(payload_get("bounds"), f"{where}.payload.bounds")
        inputs = _fast_product_refs(payload_get("inputs", ()))
        if inputs is None:
            inputs = tuple(
                ProductRef.from_mapping(item, f"{where}.payload.inputs[{index}]")
                for index, item in enumerate(_seq(payload_get("inputs", ()), f"{where}.payload.inputs"))
            )
        outputs = _fast_product_refs(payload_get("outputs", ()))
        if outputs is None:
            outputs = tuple(
                ProductRef.from_mapping(item, f"{where}.payload.outputs[{index}]")
                for index, item in enumerate(_seq(payload_get("outputs", ()), f"{where}.payload.outputs"))
            )
        predicate_raw = payload_get("predicate")
        if predicate_raw is None:
            predicate = None
        else:
            predicate = _fast_product_ref(predicate_raw)
            if predicate is None:
                predicate = ProductRef.from_mapping(predicate_raw, f"{where}.payload.predicate")
        rng_raw = payload_get("rng")
        if rng_raw is None:
            rng = None
        else:
            rng = _fast_rng(rng_raw)
            if rng is None:
                rng = Rng.from_mapping(rng_raw, f"{where}.payload.rng")
        control_seq = payload_get("control_seq")
        if not (type(control_seq) is int and control_seq >= 0):
            control_seq = _uint(control_seq, f"{where}.payload.control_seq")
        operation = cls(
            request_key=request_key,
            op_id=op_id,
            parent=parent,
            kind=work,
            payload=payload_class(bounds, inputs, outputs, predicate, rng, control_seq),
        )
        operation.validate()
        return operation

    def to_mapping(self) -> dict[str, object]:
        """Encode operation identity, parent checkpoint, kind, and family-specific payload."""

        return {
            "request_key": self.request_key.to_mapping(),
            "op_id": self.op_id,
            "parent": self.parent.to_mapping(),
            "kind": self.kind.value,
            "payload": self.payload.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class LogicalLengths:
    """Tracks semantic token, visible-KV, computed-KV, and latent extents after an operation."""

    token_len: int = 0
    kv_visible_len: int = 0
    kv_computed_len: int = 0
    latent_len: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "logical_lengths") -> LogicalLengths:
        """Parse semantic token, KV, and latent extents from a result payload."""

        data = _map(value, where)
        return cls(
            token_len=_uint(data.get("token_len"), f"{where}.token_len"),
            kv_visible_len=_uint(data.get("kv_visible_len"), f"{where}.kv_visible_len"),
            kv_computed_len=_uint(data.get("kv_computed_len"), f"{where}.kv_computed_len"),
            latent_len=_uint(data.get("latent_len"), f"{where}.latent_len"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize semantic token, KV, and latent extents for IPC."""

        return {
            "token_len": self.token_len,
            "kv_visible_len": self.kv_visible_len,
            "kv_computed_len": self.kv_computed_len,
            "latent_len": self.latent_len,
        }


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """Identifies a contiguous token interval in a model result."""

    base: int = 0
    len: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "token_span") -> TokenSpan:
        """Parse a contiguous token base and length from a result payload."""

        data = _map(value, where)
        return cls(
            base=_uint(data.get("base"), f"{where}.base"),
            len=_uint(data.get("len"), f"{where}.len"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the contiguous token interval for IPC."""

        return {"base": self.base, "len": self.len}


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
class _ResultData:
    """Holds fields shared by autoregressive, encoder, diffusion, and transfer results."""

    family: ClassVar[str]
    logical_lengths: LogicalLengths
    token_span: TokenSpan
    committed_tokens: tuple[int, ...]
    finish_flags: FinishFlags
    media_output: MediaOutput | None = None

    def to_mapping(self) -> dict[str, object]:
        """Serialize fields shared by every result family and diffusion cursor extensions."""

        lengths = self.logical_lengths
        span = self.token_span
        flags = self.finish_flags
        value: dict[str, object] = {
            "family": self.family,
            "logical_lengths": {
                "token_len": lengths.token_len,
                "kv_visible_len": lengths.kv_visible_len,
                "kv_computed_len": lengths.kv_computed_len,
                "latent_len": lengths.latent_len,
            },
            "token_span": {"base": span.base, "len": span.len},
            "committed_tokens": list(self.committed_tokens),
            "finish_flags": {"eos": flags.eos, "length": flags.length, "stop": flags.stop},
            "media_output": None if self.media_output is None else self.media_output.to_mapping(),
        }
        if isinstance(self, DiffusionResult):
            value.update(next_cursor=self.next_cursor, done=self.done)
        return value


class ArResult(_ResultData):
    """Carries autoregressive lengths, committed tokens, finish state, and optional media output."""

    family = "ar"


class EncoderResult(_ResultData):
    """Carries encoder logical lengths and any produced feature payload metadata."""

    family = "encoder"


@dataclass(frozen=True, slots=True)
class DiffusionResult(_ResultData):
    """Carries diffusion logical lengths, committed points, and optional media output."""

    family = "diffusion"
    next_cursor: int = 0
    done: bool = False


class TransferResult(_ResultData):
    """Carries logical lengths and publication metadata from a transfer operation."""

    family = "transfer"


ResultPayload: TypeAlias = ArResult | EncoderResult | DiffusionResult | TransferResult


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Associates a typed operation result with status, error metadata, timing, and output products."""

    request_key: RequestKey
    op_id: int
    completion_slot_generation: int
    status: OpStatus
    selected_point: int
    product_generations: tuple[int, ...]
    error_code: ErrorCode | None
    timing_counters: TimingCounters
    payload: ResultPayload

    @property
    def logical_lengths(self) -> LogicalLengths:
        """Expose semantic extents carried by the typed result payload."""

        return self.payload.logical_lengths

    @property
    def token_span(self) -> TokenSpan:
        """Expose the contiguous token interval produced by the operation."""

        return self.payload.token_span

    @property
    def committed_tokens(self) -> tuple[int, ...]:
        """Expose host-visible tokens accepted into request state."""

        return self.payload.committed_tokens

    @property
    def finish_flags(self) -> FinishFlags:
        """Expose generation termination conditions from the typed payload."""

        return self.payload.finish_flags

    @property
    def media_output(self) -> MediaOutput | None:
        """Expose the completed media artifact attached to this result, if any."""

        return self.payload.media_output

    @property
    def result_family(self) -> str:
        """Expose the wire discriminator for the typed result payload."""

        return self.payload.family

    @property
    def next_cursor(self) -> int:
        """Expose the next diffusion reconstruction cursor, or zero for other families."""

        return self.payload.next_cursor if isinstance(self.payload, DiffusionResult) else 0

    @property
    def done(self) -> bool:
        """Indicate whether diffusion reconstruction is complete."""

        return self.payload.done if isinstance(self.payload, DiffusionResult) else False

    def validate(self) -> None:
        """Verify that status, result family, products, errors, and timing form a coherent completion."""

        if self.op_id < 1:
            raise invalid_descriptor("completion op id must be positive")
        if self.logical_lengths.kv_visible_len > self.logical_lengths.kv_computed_len:
            raise invalid_descriptor("completion selected KV length exceeds computed length")
        if self.completion_slot_generation < 1:
            raise invalid_descriptor("completion slot generation must be positive")
        if self.next_cursor < 0:
            raise invalid_descriptor("completion next cursor must not be negative")
        if self.status is OpStatus.ERROR:
            if self.error_code is None:
                raise invalid_descriptor("an error completion must carry an error code")
        elif self.error_code is not None:
            raise invalid_descriptor("a non-error completion must not carry an error code")
        if self.status is OpStatus.PREDICATED and (
            self.token_span.len != 0
            or self.committed_tokens
            or self.product_generations
            or self.finish_flags.eos
            or self.finish_flags.length
            or self.finish_flags.stop
        ):
            raise invalid_descriptor(
                "a predicated completion must select its parent without semantic output"
            )

    @classmethod
    def from_mapping(cls, value: object, where: str = "completion") -> ModelOutput:
        """Parse a completion and enforce its status-specific result and error contract."""

        data = _map(value, where)
        payload = _map(data.get("payload"), f"{where}.payload")
        result_family = _str(payload.get("family"), f"{where}.payload.family")
        result_type: type[_ResultData]
        if result_family == "ar":
            result_type = ArResult
        elif result_family == "encoder":
            result_type = EncoderResult
        elif result_family == "diffusion":
            result_type = DiffusionResult
        elif result_family == "transfer":
            result_type = TransferResult
        else:
            raise invalid_descriptor(f"{where}.payload family is invalid")
        result_args = (
            LogicalLengths.from_mapping(
                payload.get("logical_lengths"), f"{where}.payload.logical_lengths"
            ),
            TokenSpan.from_mapping(payload.get("token_span"), f"{where}.payload.token_span"),
            _uints(payload.get("committed_tokens", ()), f"{where}.payload.committed_tokens"),
            FinishFlags.from_mapping(payload.get("finish_flags"), f"{where}.payload.finish_flags"),
            None
            if payload.get("media_output") is None
            else MediaOutput.from_mapping(payload["media_output"], f"{where}.payload.media_output"),
        )
        result_payload = (
            DiffusionResult(
                *result_args,
                next_cursor=_uint(payload.get("next_cursor", 0), f"{where}.payload.next_cursor"),
                done=_bool(payload.get("done", False), f"{where}.payload.done"),
            )
            if result_type is DiffusionResult
            else result_type(*result_args)
        )
        record = cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            op_id=_uint(data.get("op_id"), f"{where}.op_id"),
            completion_slot_generation=_uint(
                data.get("completion_slot_generation"), f"{where}.completion_slot_generation"
            ),
            status=_enum(OpStatus, data.get("status"), f"{where}.status"),
            selected_point=_uint(data.get("selected_point"), f"{where}.selected_point"),
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
            payload=result_payload,
        )
        record.validate()
        return record

    def to_mapping(self) -> dict[str, object]:
        """Encode one operation completion with typed result, products, timing, and error metadata."""

        key = self.request_key
        timing = self.timing_counters
        error_code = self.error_code
        return {
            "request_key": {
                "authority_id": key.authority_id,
                "request_id": key.request_id,
                "epoch": key.epoch,
            },
            "op_id": self.op_id,
            "completion_slot_generation": self.completion_slot_generation,
            "status": self.status.value,
            "selected_point": self.selected_point,
            "payload": self.payload.to_mapping(),
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
        return cls(
            request=NewRequest.from_mapping(data.get("request"), f"{where}.request")
        )


@dataclass(frozen=True, slots=True)
class Commit:
    """Atomically selects a checkpoint, publishes bounded results, and advances request state."""

    request_key: RequestKey
    control_seq: int
    expected_parent: Checkpoint
    selected: Checkpoint
    public_event_limit: int
    disposition: Disposition


@dataclass(frozen=True, slots=True)
class Finish:
    """Closes a request at a validated cutoff and records its terminal reason."""

    request_key: RequestKey
    control_seq: int
    cutoff: Checkpoint
    reason: CloseReason


@dataclass(frozen=True, slots=True)
class Free:
    """Releases one scheduler-owned persistent buffer."""

    buffer: BufferId

    @property
    def request_key(self) -> RequestKey:
        """Expose the request generation that owns the freed persistent buffer."""

        return self.buffer.owner


BatchCommand: TypeAlias = Start | Commit | Finish | Free


def _command_variant_index(command: BatchCommand) -> int:
    """Return the stable wire-union variant index for a lifecycle command."""

    if isinstance(command, Start):
        return 0
    if isinstance(command, Commit):
        return 1
    if isinstance(command, Finish):
        return 2
    return 3


def command_from_mapping(
    value: object,
    where: str = "batch command",
) -> BatchCommand:
    """Parse a tagged start, commit, finish, or free lifecycle command."""

    kind, payload = _tagged(value, where)
    data = _map(payload, f"{where}.value")
    if kind == "start":
        return Start.from_mapping(data, f"{where}.value")
    request_key = _fast_request_key(data.get("request_key"))
    if kind != "free" and request_key is None:
        request_key = RequestKey.from_mapping(data.get("request_key"), f"{where}.value.request_key")
    if kind == "commit":
        expected_parent = _fast_version_ref(data.get("expected_parent"))
        if expected_parent is None:
            expected_parent = Checkpoint.from_mapping(
                data.get("expected_parent"), f"{where}.value.expected_parent"
            )
        selected = _fast_version_ref(data.get("selected"))
        if selected is None:
            selected = Checkpoint.from_mapping(data.get("selected"), f"{where}.value.selected")
        commit = Commit(
            request_key=request_key,
            control_seq=_uint(data.get("control_seq"), f"{where}.value.control_seq"),
            expected_parent=expected_parent,
            selected=selected,
            public_event_limit=_uint(
                data.get("public_event_limit"), f"{where}.value.public_event_limit"
            ),
            disposition=_enum(Disposition, data.get("disposition"), f"{where}.value.disposition"),
        )
        if not commit.selected.is_fixed():
            raise invalid_descriptor("a commit command must select a fixed version")
        command: BatchCommand = commit
    elif kind == "finish":
        cutoff = _fast_version_ref(data.get("cutoff"))
        if cutoff is None:
            cutoff = Checkpoint.from_mapping(data.get("cutoff"), f"{where}.value.cutoff")
        command = Finish(
            request_key=request_key,
            control_seq=_uint(data.get("control_seq"), f"{where}.value.control_seq"),
            cutoff=cutoff,
            reason=_enum(CloseReason, data.get("reason"), f"{where}.value.reason"),
        )
        if not command.cutoff.is_fixed():
            raise invalid_descriptor("a finish command must name a fixed cutoff version")
    elif kind == "free":
        command = Free(
            buffer=BufferId.from_mapping(data.get("buffer"), f"{where}.value.buffer")
        )
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
    if isinstance(command, Commit):
        return {
            "kind": "commit",
            "value": {
                "request_key": command.request_key.to_mapping(),
                "control_seq": command.control_seq,
                "expected_parent": command.expected_parent.to_mapping(),
                "selected": command.selected.to_mapping(),
                "public_event_limit": command.public_event_limit,
                "disposition": command.disposition.value,
            },
        }
    if isinstance(command, Finish):
        return {
            "kind": "finish",
            "value": {
                "request_key": command.request_key.to_mapping(),
                "control_seq": command.control_seq,
                "cutoff": command.cutoff.to_mapping(),
                "reason": command.reason.value,
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
    def from_mapping(cls, value: object, where: str = "autoregressive parameters") -> ArRequestParams:
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
    def from_mapping(cls, value: object, where: str = "unified-multimodal parameters") -> UmmRequestParams:
        """Parse unified multimodal image-generation parameters for admission."""

        data = _map(value, where)
        return cls(image=ImageParams.from_mapping(data.get("image", {}), f"{where}.image"))

    def to_mapping(self) -> dict[str, object]:
        """Serialize unified multimodal controls for admission."""

        return {"image": self.image.to_mapping()}


@dataclass(frozen=True, slots=True)
class MediaGeometry:
    """Fixes frame rate, frame count, and raster dimensions for bounded media generation."""

    frame_count: int
    decode_units: int
    prompt_tokens: int
    denoise_steps: int

    def __post_init__(self) -> None:
        """Validate positive frame rate, frame count, and raster dimensions."""

        for name in (
            "frame_count",
            "decode_units",
            "prompt_tokens",
            "denoise_steps",
        ):
            _nonnegative(getattr(self, name), f"media geometry {name}")
        if (
            self.frame_count == 0
            or self.decode_units == 0
            or self.prompt_tokens == 0
            or self.denoise_steps == 0
        ):
            raise invalid_descriptor("media geometry is invalid")

    @classmethod
    def from_mapping(cls, value: object, where: str = "media geometry") -> MediaGeometry:
        """Parse exact frame, prompt, denoising, and reconstruction-unit geometry."""

        data = _map(value, where)
        return cls(
            frame_count=_uint(data.get("frame_count"), f"{where}.frame_count"),
            decode_units=_uint(data.get("decode_units"), f"{where}.decode_units"),
            prompt_tokens=_uint(data.get("prompt_tokens"), f"{where}.prompt_tokens"),
            denoise_steps=_uint(data.get("denoise_steps"), f"{where}.denoise_steps"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize bounded media execution geometry for admission."""

        return {
            "frame_count": self.frame_count,
            "decode_units": self.decode_units,
            "prompt_tokens": self.prompt_tokens,
            "denoise_steps": self.denoise_steps,
        }


@dataclass(frozen=True, slots=True)
class DiffusionRequestParams:
    """Defines latent geometry, integration steps, guidance scales, and deterministic noise coordinates."""

    prompt_token_ids: tuple[int, ...]
    seed: int
    geometry: MediaGeometry

    def __post_init__(self) -> None:
        """Validate latent geometry, integration steps, guidance, and noise coordinates."""

        if not self.prompt_token_ids:
            raise invalid_descriptor("diffusion prompt tokens must not be empty")
        _nonnegative(self.seed, "diffusion seed")
        if len(self.prompt_token_ids) != self.geometry.prompt_tokens:
            raise invalid_descriptor("diffusion prompt tokens disagree with its geometry")

    @classmethod
    def from_mapping(cls, value: object, where: str = "diffusion parameters") -> DiffusionRequestParams:
        """Parse prompt tokens, deterministic seed, and media geometry for diffusion work."""

        data = _map(value, where)
        return cls(
            prompt_token_ids=_uints(data.get("prompt_token_ids", ()), f"{where}.prompt_token_ids"),
            seed=_uint(data.get("seed"), f"{where}.seed"),
            geometry=MediaGeometry.from_mapping(data.get("geometry"), f"{where}.geometry"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize diffusion admission parameters for IPC."""

        return {
            "prompt_token_ids": list(self.prompt_token_ids),
            "seed": self.seed,
            "geometry": self.geometry.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class NewRequest:
    """Binds one request key to its autoregressive, multimodal, or diffusion parameters."""

    request_key: RequestKey
    request_pool_idx: int
    ar: ArRequestParams | None
    umm: UmmRequestParams | None
    diffusion: DiffusionRequestParams | None = None

    def __post_init__(self) -> None:
        """Require exactly one parameter family compatible with the request kind."""

        if self.request_pool_idx < 1:
            raise invalid_descriptor("request-pool index must be positive")
        if self.ar is None and self.umm is None and self.diffusion is None:
            raise invalid_descriptor("request start must declare one runtime-family parameter set")

    @classmethod
    def create(
        cls,
        request_key: RequestKey,
        *,
        request_pool_idx: int,
        ar: ArRequestParams | None = None,
        umm: UmmRequestParams | None = None,
        diffusion: DiffusionRequestParams | None = None,
    ) -> NewRequest:
        """Construct a new request while requiring exactly one parameter family."""

        return cls(request_key, request_pool_idx, ar, umm, diffusion)

    @classmethod
    def from_mapping(cls, value: object, where: str = "admission") -> NewRequest:
        """Parse a request slot and its optional execution-family parameter sets."""

        data = _map(value, where)
        admission = cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            request_pool_idx=_uint(data.get("request_pool_idx"), f"{where}.request_pool_idx"),
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
                else DiffusionRequestParams.from_mapping(data["diffusion"], f"{where}.diffusion")
            ),
        )
        return admission

    def to_mapping(self) -> dict[str, object]:
        """Serialize request identity, slot, and family parameters for admission."""

        return {
            "request_key": self.request_key.to_mapping(),
            "request_pool_idx": self.request_pool_idx,
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


@dataclass(frozen=True, slots=True)
class RowGeometry:
    """Describes logical and physical row counts for one packed execution layout."""

    operation_index: int
    request_pool_index: int
    seq_len: int
    query_len: int

    def __post_init__(self) -> None:
        """Validate logical rows fit their positive physical row capacity."""

        if (
            self.operation_index < 0
            or self.request_pool_index < 1
            or self.seq_len < 0
            or self.query_len < 1
        ):
            raise invalid_descriptor("forward row is invalid")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "forward row",
    ) -> RowGeometry:
        """Parse one packed forward row's operation, slot, and sequence geometry."""

        data = _map(value, where)
        return cls(
            operation_index=_uint(data.get("operation_index"), f"{where}.operation_index"),
            request_pool_index=_uint(data.get("request_pool_index"), f"{where}.request_pool_index"),
            seq_len=_uint(data.get("seq_len"), f"{where}.seq_len"),
            query_len=_uint(data.get("query_len"), f"{where}.query_len"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize row-to-operation placement and token lengths."""

        return {
            "operation_index": self.operation_index,
            "request_pool_index": self.request_pool_index,
            "seq_len": self.seq_len,
            "query_len": self.query_len,
        }


@dataclass(frozen=True, slots=True)
class LatentPlacement:
    """Assigns a request trajectory to fixed latent pages with bounded units and width."""

    request_key: RequestKey
    op_id: int
    page_table: tuple[int, ...]
    latent_units: int
    height: int
    width: int
    start_step: int
    step_count: int

    def __post_init__(self) -> None:
        """Validate latent page ownership, bank, units, width, and raster geometry."""

        if self.op_id < 1:
            raise invalid_descriptor("latent placement operation id must be positive")
        if min(self.latent_units, self.height, self.width) < 1:
            raise invalid_descriptor("latent placement geometry must be positive")
        if (
            not self.page_table
            or any(page < 1 for page in self.page_table)
            or len(set(self.page_table)) != len(self.page_table)
        ):
            raise invalid_descriptor(
                "latent placement page table is empty, repeats a page, or carries page zero"
            )

    @classmethod
    def from_mapping(cls, value: object, where: str = "latent placement") -> LatentPlacement:
        """Parse request-owned latent pages, raster geometry, and solver-step range."""

        data = _map(value, where)
        return cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            op_id=_uint(data.get("op_id"), f"{where}.op_id"),
            page_table=_uints(data.get("page_table", ()), f"{where}.page_table"),
            latent_units=_uint(data.get("latent_units"), f"{where}.latent_units"),
            height=_uint(data.get("height"), f"{where}.height"),
            width=_uint(data.get("width"), f"{where}.width"),
            start_step=_uint(data.get("start_step"), f"{where}.start_step"),
            step_count=_uint(data.get("step_count"), f"{where}.step_count"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize a physical latent trajectory placement for lane execution."""

        return {
            "request_key": self.request_key.to_mapping(),
            "op_id": self.op_id,
            "page_table": list(self.page_table),
            "latent_units": self.latent_units,
            "height": self.height,
            "width": self.width,
            "start_step": self.start_step,
            "step_count": self.step_count,
        }


@dataclass(frozen=True, slots=True)
class DecodePlacement:
    """Assigns a reconstruction operation to one latent slice and output ring slot."""

    request_key: RequestKey
    op_id: int
    cursor: int
    max_units: int

    def __post_init__(self) -> None:
        """Validate latent slice bounds and destination output-ring slot."""

        if self.op_id < 1 or self.max_units < 1:
            raise invalid_descriptor(
                "decode placement identity and unit bound must be positive"
            )
        _nonnegative(self.cursor, "decode placement cursor")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "decode placement"
    ) -> DecodePlacement:
        """Parse the reconstruction cursor and bounded unit count for one operation."""

        data = _map(value, where)
        return cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            op_id=_uint(data.get("op_id"), f"{where}.op_id"),
            cursor=_uint(data.get("cursor"), f"{where}.cursor"),
            max_units=_uint(data.get("max_units"), f"{where}.max_units"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize a media reconstruction placement for lane execution."""

        return {
            "request_key": self.request_key.to_mapping(),
            "op_id": self.op_id,
            "cursor": self.cursor,
            "max_units": self.max_units,
        }


@dataclass(frozen=True, slots=True)
class BufferPlacement:
    """Assigns a product to a bounded slice of scheduler-managed persistent storage."""

    buffer: BufferId
    offset: int
    bytes: int

    def __post_init__(self) -> None:
        """Validate the persistent-buffer identifier, offset, and bounded shape."""

        if self.offset < 0 or self.bytes < 1:
            raise invalid_descriptor("buffer placement span is invalid")
        if self.offset + self.bytes > (1 << 64) - 1:
            raise invalid_descriptor("buffer placement span overflows")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "buffer placement"
    ) -> BufferPlacement:
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


def _validate_buffer_placements(
    operations: Sequence[Operation],
    placements: Sequence[BufferPlacement],
    where: str,
) -> None:
    """Validate persistent-buffer placements against producing operations and shape bounds."""

    by_id: dict[BufferId, BufferPlacement] = {}
    spans: list[tuple[int, int]] = []
    for placement in placements:
        if placement.buffer in by_id:
            raise invalid_descriptor(f"{where} repeats a buffer placement identity")
        by_id[placement.buffer] = placement
        spans.append((placement.offset, placement.offset + placement.bytes))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise invalid_descriptor(f"{where} buffer placements overlap")
    for operation in operations:
        for output in operation.outputs:
            if not output.uses_persistent_buffer():
                continue
            placement = by_id.get(output.buffer_id)
            if placement is None:
                raise invalid_descriptor(
                    "persistent operation output has no buffer placement"
                )
            if placement.bytes < output.max_bytes:
                raise invalid_descriptor(
                    "buffer placement is smaller than its declared output"
                )


@dataclass(frozen=True, slots=True)
class RunLane:
    """Carries one lane’s ordered operations, row tables, placements, and collective identity."""

    lane_id: int
    launch_id: int
    collective_seq: int
    domain: Domain
    route: int
    attention: AttentionRegime
    shape_class: int
    operations: tuple[Operation, ...]
    block_tables: tuple[BlockTable, ...] = ()
    new_cache_pages: tuple[CachePageAllocation, ...] = ()
    forward_rows: tuple[RowGeometry, ...] = ()
    latent_placements: tuple[LatentPlacement, ...] = ()
    decode_placements: tuple[DecodePlacement, ...] = ()
    buffer_placements: tuple[BufferPlacement, ...] = ()

    def __post_init__(self) -> None:
        """Validate aligned lane columns, operation ownership, placements, and collective identity."""

        if (
            min(
                self.lane_id,
                self.launch_id,
                self.collective_seq,
            )
            < 1
        ):
            raise invalid_descriptor("batch lane identity must be positive")
        if self.route < 0 or self.shape_class < 0:
            raise invalid_descriptor("batch lane route and shape class must be unsigned")
        if not self.operations:
            raise invalid_descriptor("batch lane must carry at least one operation")
        if any(operation.domain is not self.domain for operation in self.operations):
            raise invalid_descriptor("batch lane operation disagrees with its domain")
        operations = {
            (operation.request_key, operation.op_id): operation for operation in self.operations
        }
        tables = {(table.request_pool_idx, table.group_id): table for table in self.block_tables}
        if len(tables) != len(self.block_tables):
            raise invalid_descriptor("batch lane repeats a block table")
        allocation_ids: set[tuple[int, int]] = set()
        for allocation in self.new_cache_pages:
            identity = (allocation.request_pool_idx, allocation.group_id)
            if identity in allocation_ids:
                raise invalid_descriptor("batch lane repeats a cache-page allocation")
            allocation_ids.add(identity)
            table = tables.get(identity)
            if table is None or not set(allocation.page_ids).issubset(table.page_ids):
                raise invalid_descriptor("cache-page allocation has no matching block table")
        for row in self.forward_rows:
            if row.operation_index >= len(self.operations):
                raise invalid_descriptor("forward row operation index is outside its lane")
        latent_ids: set[tuple[RequestKey, int]] = set()
        latent_pages: set[int] = set()

        def addresses_trajectory(operation: Operation) -> bool:
            """Identify operations that require an explicit physical latent placement."""

            return operation.kind in {
                RunKind.DIFFUSION_PREPARE,
                RunKind.DIFFUSION_STEP,
            } or any(reference.kind is ProductKind.LATENT for reference in operation.inputs)

        for latent_placement in self.latent_placements:
            latent_identity = (latent_placement.request_key, latent_placement.op_id)
            if latent_identity in latent_ids:
                raise invalid_descriptor("batch lane repeats a latent placement identity")
            latent_ids.add(latent_identity)
            operation = operations.get(latent_identity)
            if operation is None:
                raise invalid_descriptor("latent placement does not name a lane operation")
            if not addresses_trajectory(operation):
                raise invalid_descriptor(
                    "latent placement names an operation that does not address a trajectory"
                )
            if not latent_pages.isdisjoint(latent_placement.page_table):
                raise invalid_descriptor("latent placements overlap physical pages")
            latent_pages.update(latent_placement.page_table)
        if any(
            addresses_trajectory(operation)
            and (operation.request_key, operation.op_id) not in latent_ids
            for operation in self.operations
        ):
            raise invalid_descriptor(
                "operation that addresses a trajectory has no latent placement"
            )
        decode_ids: set[tuple[RequestKey, int]] = set()
        for placement in self.decode_placements:
            identity = (placement.request_key, placement.op_id)
            if identity in decode_ids:
                raise invalid_descriptor(
                    "batch lane repeats a decode placement identity"
                )
            decode_ids.add(identity)
            operation = operations.get(identity)
            if operation is None or operation.kind not in {
                RunKind.DIFFUSION_DECODE,
                RunKind.DIFFUSION_FINALIZE,
            }:
                raise invalid_descriptor(
                    "decode placement does not name a media decode operation"
                )
        if any(
            operation.kind is RunKind.DIFFUSION_DECODE
            and (operation.request_key, operation.op_id) not in decode_ids
            for operation in self.operations
        ):
            raise invalid_descriptor(
                "media reconstruction operation has no decode placement"
            )
        _validate_buffer_placements(self.operations, self.buffer_placements, "batch lane")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "batch lane",
    ) -> RunLane:
        """Parse one physical lane and validate its operations, rows, cache tables, and placements."""

        data = _map(value, where)
        fields = dict(
            lane_id=_uint(data.get("lane_id"), f"{where}.lane_id"),
            launch_id=_uint(data.get("launch_id"), f"{where}.launch_id"),
            collective_seq=_uint(data.get("collective_seq"), f"{where}.collective_seq"),
            domain=Domain(_str(data.get("domain"), f"{where}.domain")),
            route=_uint(data.get("route"), f"{where}.route"),
            attention=AttentionRegime(_str(data.get("attention"), f"{where}.attention")),
            shape_class=_uint(data.get("shape_class"), f"{where}.shape_class"),
            operations=tuple(
                Operation.from_mapping(
                    item,
                    f"{where}.operations[{index}]",
                )
                for index, item in enumerate(
                    _seq(data.get("operations", ()), f"{where}.operations")
                )
            ),
            block_tables=tuple(
                BlockTable.from_mapping(
                    item,
                    f"{where}.block_tables[{index}]",
                )
                for index, item in enumerate(
                    _seq(data.get("block_tables", ()), f"{where}.block_tables")
                )
            ),
            new_cache_pages=tuple(
                CachePageAllocation.from_mapping(
                    item,
                    f"{where}.new_cache_pages[{index}]",
                )
                for index, item in enumerate(
                    _seq(data.get("new_cache_pages", ()), f"{where}.new_cache_pages")
                )
            ),
            forward_rows=tuple(
                RowGeometry.from_mapping(item, f"{where}.forward_rows[{index}]")
                for index, item in enumerate(
                    _seq(data.get("forward_rows", ()), f"{where}.forward_rows")
                )
            ),
            latent_placements=tuple(
                LatentPlacement.from_mapping(item, f"{where}.latent_placements[{index}]")
                for index, item in enumerate(
                    _seq(data.get("latent_placements", ()), f"{where}.latent_placements")
                )
            ),
            decode_placements=tuple(
                DecodePlacement.from_mapping(
                    item, f"{where}.decode_placements[{index}]"
                )
                for index, item in enumerate(
                    _seq(
                        data.get("decode_placements", ()),
                        f"{where}.decode_placements",
                    )
                )
            ),
            buffer_placements=tuple(
                BufferPlacement.from_mapping(
                    item, f"{where}.buffer_placements[{index}]"
                )
                for index, item in enumerate(
                    _seq(
                        data.get("buffer_placements", ()),
                        f"{where}.buffer_placements",
                    )
                )
            ),
        )
        return cls(**cast(Any, fields))

    def to_mapping(self) -> dict[str, object]:
        """Encode one physical lane with its routing, row, cache, and placement tables."""

        return {
            "lane_id": self.lane_id,
            "launch_id": self.launch_id,
            "collective_seq": self.collective_seq,
            "domain": self.domain.value,
            "route": self.route,
            "attention": self.attention.value,
            "shape_class": self.shape_class,
            "operations": [operation.to_mapping() for operation in self.operations],
            "block_tables": [table.to_mapping() for table in self.block_tables],
            "new_cache_pages": [allocation.to_mapping() for allocation in self.new_cache_pages],
            "forward_rows": [row.to_mapping() for row in self.forward_rows],
            "latent_placements": [placement.to_mapping() for placement in self.latent_placements],
            "decode_placements": [
                placement.to_mapping() for placement in self.decode_placements
            ],
            "buffer_placements": [
                placement.to_mapping() for placement in self.buffer_placements
            ],
        }


@dataclass(frozen=True, slots=True)
class Run:
    """Describes one scheduler-submitted collection of lane operations."""

    batch_id: int
    run_id: int
    collective_seq: int = 1
    operations: tuple[Operation, ...] = ()
    block_tables: tuple[BlockTable, ...] = ()
    new_cache_pages: tuple[CachePageAllocation, ...] = ()
    forward_rows: tuple[RowGeometry, ...] = ()
    latent_placements: tuple[LatentPlacement, ...] = ()
    decode_placements: tuple[DecodePlacement, ...] = ()
    buffer_placements: tuple[BufferPlacement, ...] = ()
    commands: tuple[BatchCommand, ...] = ()
    input_products: tuple[ProductPayload, ...] = ()
    lanes: tuple[RunLane, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate run identity, lane uniqueness, and lifecycle-operation consistency."""

        if self.lanes and not self.operations:
            set_field = object.__setattr__
            set_field(self, "collective_seq", self.lanes[0].collective_seq)
            set_field(
                self,
                "operations",
                tuple(operation for lane in self.lanes for operation in lane.operations),
            )
            set_field(self, "block_tables", tuple(value for lane in self.lanes for value in lane.block_tables))
            set_field(self, "new_cache_pages", tuple(value for lane in self.lanes for value in lane.new_cache_pages))
            set_field(self, "forward_rows", tuple(value for lane in self.lanes for value in lane.forward_rows))
            set_field(self, "latent_placements", tuple(value for lane in self.lanes for value in lane.latent_placements))
            set_field(self, "decode_placements", tuple(value for lane in self.lanes for value in lane.decode_placements))
            set_field(self, "buffer_placements", tuple(value for lane in self.lanes for value in lane.buffer_placements))
        self.validate()

    @property
    def admissions(self) -> tuple[NewRequest, ...]:
        """Extract new-request payloads from lifecycle commands in submission order."""

        return tuple(
            command.request for command in self.commands if isinstance(command, Start)
        )

    def validate(self) -> None:
        """Enforce run identity, command ordering, lane uniqueness, operation counts, and token bounds."""

        if not self.operations and not self.commands:
            raise invalid_descriptor(
                "a submission batch must carry at least one operation or command"
            )
        if self.collective_seq < 1:
            raise invalid_descriptor("run collective sequence must be positive")
        if any(row.operation_index >= len(self.operations) for row in self.forward_rows):
            raise invalid_descriptor("forward row operation index is outside its run")
        request_keys = [operation.request_key for operation in self.operations]
        if len(set(request_keys)) != len(request_keys):
            raise invalid_descriptor(
                "a submission batch carries multiple operations for one request"
            )
        admitted = [admission.request_key for admission in self.admissions]
        if len(set(admitted)) != len(admitted):
            raise invalid_descriptor("a submission batch carries a duplicate admission")
        for admission in self.admissions:
            if admission.request_key not in request_keys:
                raise invalid_descriptor("a submission batch admits a request without an operation")
        identities: dict[
            tuple[RequestKey, int | None, int, int | None], BatchCommand
        ] = {}
        for command in self.commands:
            seq = command.control_seq if isinstance(command, (Commit, Finish)) else None
            generation = command.buffer.generation if isinstance(command, Free) else None
            identity = (
                command.request_key,
                seq,
                _command_variant_index(command),
                generation,
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
            for product in (*operation.inputs, operation.predicate)
            if product is not None
        }
        for operation in self.operations:
            for product in operation.inputs:
                if product.storage_class is StorageClass.HOST_STAGING and (
                    product.request_key != operation.request_key
                    or product.producer_op_id != operation.op_id
                ):
                    raise invalid_descriptor(
                        "a host-staging input is not owned by its consuming operation"
                    )
        supplied_inputs: set[ProductRef] = set()
        for payload in self.input_products:
            product = payload.product
            if product not in declared_inputs:
                raise invalid_descriptor(
                    "an input product payload is not declared by any operation"
                )
            transferred = isinstance(payload.payload, TransferHandle)
            if product.storage_class is StorageClass.HOST_STAGING and transferred:
                raise invalid_descriptor(
                    "host-staging input cannot carry a cross-stage transfer handle"
                )
            if product.storage_class is not StorageClass.HOST_STAGING:
                if not transferred:
                    raise invalid_descriptor("cross-stage product input has no transfer handle")
            if product in supplied_inputs:
                raise invalid_descriptor("a submission batch repeats an input product payload")
            supplied_inputs.add(product)
            if product.kind is ProductKind.TOKEN and not transferred:
                if (
                    len(decode_token_product_bytes(payload.payload))
                    > product.shape_bound.max_elements
                ):
                    raise invalid_descriptor(
                        "token input product exceeds its registered element bound"
                    )
            elif not transferred and len(payload.payload) > product.max_bytes:
                raise invalid_descriptor("input product payload exceeds its registered byte bound")
        for product in declared_inputs:
            if (
                product.storage_class is StorageClass.HOST_STAGING
                and product not in supplied_inputs
            ):
                raise invalid_descriptor("a host-staging operation input has no product payload")
        _validate_buffer_placements(self.operations, self.buffer_placements, "run")

    @classmethod
    def from_mapping(cls, value: object) -> Run:
        """Parse a scheduler run and validate all lifecycle commands and physical lanes."""

        data = _map(value, "execute run")
        batch_id = _uint(data.get("batch_id"), "execute run.batch_id")
        run_id = _uint(data.get("run_id"), "execute run.run_id")
        operations = tuple(
            Operation.from_mapping(item, f"execute run.operations[{index}]")
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
            ProductPayload.from_mapping(item, f"execute run.input_products[{index}]")
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
                for index, item in enumerate(_seq(data.get("block_tables", ()), "execute run.block_tables"))
            ),
            new_cache_pages=tuple(
                CachePageAllocation.from_mapping(item, f"execute run.new_cache_pages[{index}]")
                for index, item in enumerate(_seq(data.get("new_cache_pages", ()), "execute run.new_cache_pages"))
            ),
            forward_rows=tuple(
                RowGeometry.from_mapping(item, f"execute run.forward_rows[{index}]")
                for index, item in enumerate(_seq(data.get("forward_rows", ()), "execute run.forward_rows"))
            ),
            latent_placements=tuple(
                LatentPlacement.from_mapping(item, f"execute run.latent_placements[{index}]")
                for index, item in enumerate(_seq(data.get("latent_placements", ()), "execute run.latent_placements"))
            ),
            decode_placements=tuple(
                DecodePlacement.from_mapping(item, f"execute run.decode_placements[{index}]")
                for index, item in enumerate(_seq(data.get("decode_placements", ()), "execute run.decode_placements"))
            ),
            buffer_placements=tuple(
                BufferPlacement.from_mapping(
                    item, f"execute run.buffer_placements[{index}]"
                )
                for index, item in enumerate(
                    _seq(data.get("buffer_placements", ()), "execute run.buffer_placements")
                )
            ),
            commands=commands,
            input_products=input_products,
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
            "forward_rows": [value.to_mapping() for value in self.forward_rows],
            "latent_placements": [value.to_mapping() for value in self.latent_placements],
            "decode_placements": [value.to_mapping() for value in self.decode_placements],
            "buffer_placements": [value.to_mapping() for value in self.buffer_placements],
            "commands": [command_to_mapping(value) for value in self.commands],
            "input_products": [value.to_mapping() for value in self.input_products],
        }


@dataclass(frozen=True, slots=True)
class RegistrationAck:
    """Returns the request versions made visible by a registration batch."""

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
class ProductPayload:
    """Associates one semantic product reference with its encoded payload."""

    product: ProductRef
    payload: bytes | TransferHandle

    @classmethod
    def from_mapping(cls, value: object, where: str = "product payload") -> ProductPayload:
        """Parse inline bytes or a transfer handle for one declared product reference."""

        data = _map(value, where)
        raw_value = _map(data.get("value"), f"{where}.value")
        value_kind = _str(raw_value.get("kind"), f"{where}.value.kind")
        raw = raw_value.get("value")
        if value_kind == "bytes":
            payload: bytes | TransferHandle = (
                raw
                if type(raw) is bytes
                else bytes(raw)
                if isinstance(raw, (bytearray, memoryview))
                else bytes(_uints(raw, f"{where}.value.value"))
            )
        elif value_kind == "transfer":
            payload = TransferHandle.from_mapping(raw, f"{where}.value.value")
        else:
            raise invalid_descriptor(f"{where}.value.kind is invalid")
        return cls(
            product=ProductRef.from_mapping(data.get("product"), f"{where}.product"),
            payload=payload,
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize a product reference with its tagged inline or transfer payload."""

        if isinstance(self.payload, TransferHandle):
            value = {"kind": "transfer", "value": self.payload.to_mapping()}
        else:
            value = {"kind": "bytes", "value": self.payload}
        return {"product": self.product.to_mapping(), "value": value}


def encode_token_product_bytes(tokens: Sequence[int]) -> bytes:
    """Encode a ``ProductKind.TOKEN`` product value.

    The layout is a little-endian ``u32`` count followed by that many
    little-endian ``u32`` token ids, matching the Rust ``uniserve-worker-ipc`` codec so
    the scheduler and worker share one exact format.
    """

    out = bytearray(struct.pack("<I", len(tokens)))
    for token in tokens:
        out += struct.pack("<I", token)
    return bytes(out)


def decode_token_product_bytes(data: bytes) -> tuple[int, ...]:
    """Decode a ``ProductKind.TOKEN`` product value produced by
    :func:`encode_token_product_bytes`."""

    if len(data) < 4:
        raise invalid_descriptor("token product bytes are too short to carry a count")
    (count,) = struct.unpack_from("<I", data, 0)
    expected = 4 + count * 4
    if len(data) != expected:
        raise invalid_descriptor(
            f"token product byte length {len(data)} does not match declared count {count}"
        )
    return tuple(struct.unpack_from("<I", data, 4 + index * 4)[0] for index in range(count))


@dataclass(frozen=True, slots=True)
class SamplingState:
    """Canonical branch-local token processor inputs for one operation.

    Penalty token counts are not carried here: they are a device-resident
    committed base plus bounded per-operation deltas the worker folds on commit,
    so no host token history participates in a successor's sampling input.
    """

    allowed_token_ids: tuple[int, ...] | None = None
    suppressed_token_ids: tuple[int, ...] = ()
    finish_token_ids: tuple[int, ...] = ()
    transition_token_ids: tuple[int, ...] = ()
    force_finish: bool = False


def encode_sampling_state_bytes(state: SamplingState) -> bytes:
    """Encode generated-token counts as a bounded little-endian sampling-state payload."""

    allowed = (
        None
        if state.allowed_token_ids is None
        else tuple(sorted(set(int(token) for token in state.allowed_token_ids)))
    )
    suppressed = tuple(sorted(set(int(token) for token in state.suppressed_token_ids)))
    finish = tuple(sorted(set(int(token) for token in state.finish_token_ids)))
    transition = tuple(sorted(set(int(token) for token in state.transition_token_ids)))
    out = bytearray()
    if allowed is None:
        out += b"\x00"
    else:
        out += b"\x01" + struct.pack("<I", len(allowed))
        for token in allowed:
            out += struct.pack("<I", token)
    out += struct.pack("<I", len(suppressed))
    for token in suppressed:
        out += struct.pack("<I", token)
    out += struct.pack("<I", len(finish))
    for token in finish:
        out += struct.pack("<I", token)
    out += struct.pack("<I", len(transition))
    for token in transition:
        out += struct.pack("<I", token)
    out += bytes((int(state.force_finish),))
    return bytes(out)


def decode_sampling_state_bytes(data: bytes) -> SamplingState:
    """Decode and validate a little-endian sampling-state payload."""

    offset = 0

    def take_u32() -> int:
        """Consume one little-endian unsigned integer from the sampling payload."""

        nonlocal offset
        if offset + 4 > len(data):
            raise invalid_descriptor("sampling-state bytes are truncated")
        value = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        return value

    def take_ids(count: int) -> tuple[int, ...]:
        """Consume a canonical strictly increasing token-id vector."""

        values = tuple(take_u32() for _ in range(count))
        if any(left >= right for left, right in zip(values, values[1:], strict=False)):
            raise invalid_descriptor("sampling-state token ids are not canonical")
        return values

    if offset >= len(data):
        raise invalid_descriptor("sampling-state bytes omit allowed presence")
    presence = data[offset]
    offset += 1
    if presence == 0:
        allowed = None
    elif presence == 1:
        allowed = take_ids(take_u32())
    else:
        raise invalid_descriptor(f"sampling-state allowed presence {presence} is invalid")
    suppressed = take_ids(take_u32())
    finish = take_ids(take_u32())
    transition = take_ids(take_u32())
    if offset >= len(data):
        raise invalid_descriptor("sampling-state bytes omit force-finish")
    force_finish = data[offset]
    offset += 1
    if force_finish not in (0, 1):
        raise invalid_descriptor(f"sampling-state force-finish {force_finish} is invalid")
    if offset != len(data):
        raise invalid_descriptor("sampling-state bytes contain trailing data")
    return SamplingState(allowed, suppressed, finish, transition, bool(force_finish))


@dataclass(frozen=True, slots=True)
class WorkerForwardStats:
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
    def from_mapping(cls, value: object, where: str = "worker forward stats") -> WorkerForwardStats:
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
class LaneResult:
    """Carries the ordered operation results produced by one execution lane."""

    lane_id: int
    completions: tuple[ModelOutput | CompletionState, ...]
    products: tuple[ProductPayload, ...] = ()
    registration: RegistrationAck = field(default_factory=RegistrationAck)
    worker_exec_us: int | None = None
    forward_stats: WorkerForwardStats | None = None
    publication: LanePublication | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "lane completion",
    ) -> LaneResult:
        """Parse one lane completion with ordered operation outputs and optional publication state."""

        data = _map(value, where)
        return cls(
            lane_id=_uint(data.get("lane_id"), f"{where}.lane_id"),
            completions=tuple(
                ModelOutput.from_mapping(item, f"{where}.completions[{index}]")
                for index, item in enumerate(
                    _seq(data.get("completions", ()), f"{where}.completions")
                )
            ),
            products=tuple(
                ProductPayload.from_mapping(item, f"{where}.products[{index}]")
                for index, item in enumerate(_seq(data.get("products", ()), f"{where}.products"))
            ),
            registration=RegistrationAck.from_mapping(
                data.get("registration", {}), f"{where}.registration"
            ),
            worker_exec_us=_optional_uint(data.get("worker_exec_us"), f"{where}.worker_exec_us"),
            forward_stats=(
                None
                if data.get("forward_stats") is None
                else WorkerForwardStats.from_mapping(
                    data["forward_stats"], f"{where}.forward_stats"
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode one lane’s operation completions in execution order."""

        return {
            "lane_id": self.lane_id,
            "completions": [cast(ModelOutput, value).to_mapping() for value in self.completions],
            "products": [value.to_mapping() for value in self.products],
            "registration": self.registration.to_mapping(),
            "worker_exec_us": self.worker_exec_us,
            "forward_stats": None
            if self.forward_stats is None
            else self.forward_stats.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    """Carries the completed lane results and aggregate forward statistics for one run."""

    batch_id: int
    run_id: int
    lanes: tuple[LaneResult, ...]
    done: bool = True

    @property
    def completions(self) -> tuple[ModelOutput | CompletionState, ...]:
        """Flatten lane completions in physical lane and operation order."""

        return tuple(
            completion for lane in self.lanes for completion in lane.completions
        )

    @property
    def products(self) -> tuple[ProductPayload, ...]:
        """Flatten lane product payloads in physical lane order."""

        return tuple(product for lane in self.lanes for product in lane.products)

    @property
    def registration(self) -> RegistrationAck:
        """Report visibility only after every physical lane acknowledges registration."""

        return RegistrationAck(
            visible=bool(self.lanes)
            and all(lane.registration.visible for lane in self.lanes)
        )

    @property
    def worker_exec_us(self) -> int | None:
        """Use the slowest lane duration as the run's wall-clock execution time."""

        values = tuple(
            lane.worker_exec_us
            for lane in self.lanes
            if lane.worker_exec_us is not None
        )
        return None if not values else max(values)

    @property
    def forward_stats(self) -> WorkerForwardStats | None:
        """Merge lane counters by summing scalars and keyed counter maps."""

        values = tuple(lane.forward_stats for lane in self.lanes if lane.forward_stats is not None)
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        merged: dict[str, object] = {}
        for name in WorkerForwardStats.__dataclass_fields__:
            fields = tuple(getattr(value, name) for value in values)
            if isinstance(fields[0], Mapping):
                totals: dict[str, int] = {}
                for field_value in fields:
                    for key, count in cast(Mapping[str, int], field_value).items():
                        totals[key] = totals.get(key, 0) + count
                merged[name] = totals
            else:
                merged[name] = sum(cast(tuple[int, ...], fields))
        return WorkerForwardStats(**cast(Any, merged))

    @classmethod
    def from_mapping(cls, value: object, where: str = "completion report") -> RunResult:
        """Parse a run completion and its lane reports, registration acknowledgement, and forward statistics."""

        data = _map(value, where)
        completions = tuple(
            ModelOutput.from_mapping(item, f"{where}.completions[{index}]")
            for index, item in enumerate(_seq(data.get("completions", ()), f"{where}.completions"))
        )
        products = tuple(
            ProductPayload.from_mapping(item, f"{where}.products[{index}]")
            for index, item in enumerate(_seq(data.get("products", ()), f"{where}.products"))
        )
        registration = RegistrationAck.from_mapping(data.get("registration", {}), f"{where}.registration")
        forward_stats = (
            None
            if data.get("forward_stats") is None
            else WorkerForwardStats.from_mapping(data["forward_stats"], f"{where}.forward_stats")
        )
        has_payload = bool(completions or products or data.get("worker_exec_us") is not None or forward_stats is not None)
        return cls(
            batch_id=_uint(data.get("batch_id"), f"{where}.batch_id"),
            run_id=_uint(data.get("run_id"), f"{where}.run_id"),
            lanes=(
                LaneResult(
                    lane_id=1,
                    completions=completions,
                    products=products,
                    registration=registration,
                    worker_exec_us=_optional_uint(data.get("worker_exec_us"), f"{where}.worker_exec_us"),
                    forward_stats=forward_stats,
                ),
            ) if has_payload else (),
            done=_bool(data.get("done", True), f"{where}.done"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode the completed run, lane outputs, registration state, and aggregate statistics."""

        return {
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "completions": [cast(ModelOutput, value).to_mapping() for value in self.completions],
            "products": [value.to_mapping() for value in self.products],
            "registration": self.registration.to_mapping(),
            "worker_exec_us": self.worker_exec_us,
            "forward_stats": None if self.forward_stats is None else self.forward_stats.to_mapping(),
            "done": self.done,
        }


# Each `_fast_*` helper recognizes the exact built-in IPC shape without
# allocating error-location strings. A non-matching value returns ``None`` so
# the caller applies the canonical validated constructor and its precise error.

_E = TypeVar("_E", bound=StrEnum)

_DOMAIN_BY_VALUE: Mapping[str, Domain] = Domain._value2member_map_  # type: ignore[assignment]
_PRODUCT_KIND_BY_VALUE: Mapping[str, ProductKind] = ProductKind._value2member_map_  # type: ignore[assignment]
_STORAGE_CLASS_BY_VALUE: Mapping[str, StorageClass] = StorageClass._value2member_map_  # type: ignore[assignment]
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
        return cast(bytes, value)
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
def _interned_request_key(authority_id: int, request_id: int, epoch: int) -> RequestKey:
    """Reuse an immutable request key for identical authority, request, and epoch values."""

    key = object.__new__(RequestKey)
    object.__setattr__(key, "authority_id", authority_id)
    object.__setattr__(key, "request_id", request_id)
    object.__setattr__(key, "epoch", epoch)
    return key


def _fast_request_key(value: object) -> RequestKey | None:
    """Decode a trusted compact request-key mapping."""

    if type(value) is not dict:
        return None
    authority_id = value.get("authority_id")
    request_id = value.get("request_id")
    epoch = value.get("epoch")
    if (
        type(authority_id) is int
        and authority_id >= 0
        and type(request_id) is int
        and request_id >= 0
        and type(epoch) is int
        and epoch >= 0
    ):
        return _interned_request_key(authority_id, request_id, epoch)
    return None


@lru_cache(maxsize=1024)
def _interned_point_range(base_point: int, max_points: int) -> PointRange:
    """Reuse an immutable point range for identical base and extent values."""

    point_range = object.__new__(PointRange)
    object.__setattr__(point_range, "base_point", base_point)
    object.__setattr__(point_range, "max_points", max_points)
    return point_range


def _fast_point_range(value: object) -> PointRange | None:
    """Decode a trusted compact point-range mapping."""

    if type(value) is not dict:
        return None
    base_point = value.get("base_point")
    max_points = value.get("max_points")
    if type(base_point) is int and base_point >= 0 and type(max_points) is int and max_points >= 0:
        return _interned_point_range(base_point, max_points)
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


def _fast_product_ref(value: object) -> ProductRef | None:
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
        type(producer_op_id) is int
        and producer_op_id >= 0
        and type(output_index) is int
        and output_index >= 0
        and type(generation) is int
        and generation > 0
    ):
        return None
    raw_kind = value.get("kind")
    raw_storage = value.get("storage_class")
    raw_dtype = value.get("dtype")
    if type(raw_kind) is not str or type(raw_storage) is not str or type(raw_dtype) is not str:
        return None
    kind = _PRODUCT_KIND_BY_VALUE.get(raw_kind)
    storage_class = _STORAGE_CLASS_BY_VALUE.get(raw_storage)
    dtype = _DTYPE_BY_VALUE.get(raw_dtype)
    if kind is None or storage_class is None or dtype is None:
        return None
    shape_bound = _fast_shape_bound(value.get("shape_bound"))
    if shape_bound is None:
        return None
    point_range = _fast_point_range(value.get("point_range"))
    if point_range is None:
        return None
    reference = object.__new__(ProductRef)
    set_field = object.__setattr__
    set_field(reference, "request_key", request_key)
    set_field(reference, "producer_op_id", producer_op_id)
    set_field(reference, "output_index", output_index)
    set_field(reference, "generation", generation)
    set_field(reference, "kind", kind)
    set_field(reference, "storage_class", storage_class)
    set_field(reference, "dtype", dtype)
    set_field(reference, "shape_bound", shape_bound)
    set_field(reference, "point_range", point_range)
    return reference


def _fast_product_refs(value: object) -> tuple[ProductRef, ...] | None:
    """Decode a trusted sequence of compact product references."""

    kind = type(value)
    if kind is not list and kind is not tuple:
        return None
    items = cast(list[object] | tuple[object, ...], value)
    references: list[ProductRef] = []
    for item in items:
        reference = _fast_product_ref(item)
        if reference is None:
            return None
        references.append(reference)
    return tuple(references)


def _fast_version_ref(value: object) -> Checkpoint | None:
    """Decode a trusted compact version reference from its wire mapping."""

    if type(value) is not dict:
        return None
    op_id = value.get("op_id")
    if not (type(op_id) is int and op_id >= 0):
        return None
    raw_point = value.get("point")
    if type(raw_point) is not dict:
        return None
    tag = raw_point.get("kind")
    payload = raw_point.get("value")
    if type(tag) is not str:
        return None
    point: CheckpointPoint
    if tag == "fixed":
        if not (type(payload) is int and payload >= 0):
            return None
        point = FixedCheckpoint(payload)
    elif tag == "device_selected" and payload is None:
        point = DeviceSelected()
    else:
        return None
    reference = object.__new__(Checkpoint)
    object.__setattr__(reference, "op_id", op_id)
    object.__setattr__(reference, "point", point)
    return reference


def _fast_work(value: object) -> RunKind | None:
    """Decode a trusted compact work descriptor and its kind-specific payload."""

    if type(value) is RunKind:
        return value
    if type(value) is str:
        return _FORWARD_MODE_BY_VALUE.get(value)
    return None


@lru_cache(maxsize=256)
def _interned_bounds(
    max_points: int,
    max_tokens: int,
    max_kv_pages: int,
    max_latent_bytes: int,
    max_completion_bytes: int,
    max_transfer_bytes: int,
) -> Bounds:
    """Reuse immutable execution bounds for an identical capacity tuple."""

    bounds = object.__new__(Bounds)
    set_field = object.__setattr__
    set_field(bounds, "max_points", max_points)
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
    max_points = value.get("max_points")
    max_tokens = value.get("max_tokens")
    max_kv_pages = value.get("max_kv_pages")
    max_latent_bytes = value.get("max_latent_bytes")
    max_completion_bytes = value.get("max_completion_bytes")
    max_transfer_bytes = value.get("max_transfer_bytes")
    if (
        type(max_points) is int
        and max_points >= 0
        and type(max_tokens) is int
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
            max_points,
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
