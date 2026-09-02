"""Scheduler-to-worker execution records and their validation."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from typing import Any, Protocol, TypeAlias, TypeVar, cast

from ..foundation.errors import invalid_descriptor

TRANSFER_DESCRIPTOR_PREFIX = b"uniserve-transfer\0"
MAX_TRANSFER_DESCRIPTOR_BYTES = 64 * 1024


class CompletionState(Protocol):
    """Structural interface for one pending output row."""

    def ready(self) -> bool: ...

    def finalize(self) -> object: ...


class PartitionPublication(Protocol):
    """One-shot request-state publication owned by a partition result."""

    def finish(self, completions: tuple[ModelOutput, ...]) -> None: ...

    def cancel(self) -> None: ...

    @property
    def successors_ready(self) -> bool: ...


def is_transfer_descriptor(value: bytes) -> bool:
    return value.startswith(TRANSFER_DESCRIPTOR_PREFIX)


def validate_transfer_descriptor_frame(value: bytes) -> None:
    if not is_transfer_descriptor(value):
        raise ValueError("cross-stage product input has no transfer descriptor frame")
    if len(value) > MAX_TRANSFER_DESCRIPTOR_BYTES:
        raise ValueError("cross-stage product transfer descriptor exceeds its byte bound")


class TokenMode(StrEnum):
    EXTEND = "extend"
    DECODE = "decode"
    VERIFY = "verify"


class EncodeMode(StrEnum):
    VISION = "vision"
    LATENT = "latent"


class TransferMode(StrEnum):
    PRODUCT = "product"
    KV_PUBLISH = "kv_publish"
    KV_INSTALL = "kv_install"


class MediaMode(StrEnum):
    PREPARE = "prepare"
    DENOISE = "denoise"
    RECONSTRUCT = "reconstruct"


class ReconstructionKind(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"


class MediaProfileId(StrEnum):
    MINIMAX_H3_T2VA = "minimax_h3_t2va"


class ForwardMode(StrEnum):
    TOKEN_EXTEND = "token_extend"
    TOKEN_DECODE = "token_decode"
    TOKEN_VERIFY = "token_verify"
    ENCODE_VISION = "encode_vision"
    ENCODE_LATENT = "encode_latent"
    TRANSFER_PRODUCT = "transfer_product"
    TRANSFER_KV_PUBLISH = "transfer_kv_publish"
    TRANSFER_KV_INSTALL = "transfer_kv_install"
    MEDIA_PREPARE = "media_prepare"
    MEDIA_DENOISE = "media_denoise"
    MATERIALIZE = "materialize"
    MEDIA_RECONSTRUCT = "media_reconstruct"

    @property
    def advances_state(self) -> bool:
        return self in _STATE_ADVANCING_WORK

    @property
    def requires_fixed_parent(self) -> bool:
        return self is ForwardMode.TRANSFER_KV_PUBLISH

    @property
    def token_mode(self) -> TokenMode | None:
        if self is ForwardMode.TOKEN_EXTEND:
            return TokenMode.EXTEND
        if self is ForwardMode.TOKEN_DECODE:
            return TokenMode.DECODE
        if self is ForwardMode.TOKEN_VERIFY:
            return TokenMode.VERIFY
        return None

    @property
    def encode_mode(self) -> EncodeMode | None:
        if self is ForwardMode.ENCODE_VISION:
            return EncodeMode.VISION
        if self is ForwardMode.ENCODE_LATENT:
            return EncodeMode.LATENT
        return None

    @property
    def transfer_mode(self) -> TransferMode | None:
        if self is ForwardMode.TRANSFER_PRODUCT:
            return TransferMode.PRODUCT
        if self is ForwardMode.TRANSFER_KV_PUBLISH:
            return TransferMode.KV_PUBLISH
        if self is ForwardMode.TRANSFER_KV_INSTALL:
            return TransferMode.KV_INSTALL
        return None

    @property
    def media_mode(self) -> MediaMode | None:
        if self is ForwardMode.MEDIA_PREPARE:
            return MediaMode.PREPARE
        if self is ForwardMode.MEDIA_DENOISE:
            return MediaMode.DENOISE
        if self is ForwardMode.MEDIA_RECONSTRUCT:
            return MediaMode.RECONSTRUCT
        return None

    @classmethod
    def token(cls, mode: TokenMode) -> ForwardMode:
        return {
            TokenMode.EXTEND: cls.TOKEN_EXTEND,
            TokenMode.DECODE: cls.TOKEN_DECODE,
            TokenMode.VERIFY: cls.TOKEN_VERIFY,
        }[mode]

    @classmethod
    def encode(cls, mode: EncodeMode) -> ForwardMode:
        return {
            EncodeMode.VISION: cls.ENCODE_VISION,
            EncodeMode.LATENT: cls.ENCODE_LATENT,
        }[mode]

    @classmethod
    def transfer(cls, mode: TransferMode) -> ForwardMode:
        return {
            TransferMode.PRODUCT: cls.TRANSFER_PRODUCT,
            TransferMode.KV_PUBLISH: cls.TRANSFER_KV_PUBLISH,
            TransferMode.KV_INSTALL: cls.TRANSFER_KV_INSTALL,
        }[mode]

    @classmethod
    def media(cls, mode: MediaMode) -> ForwardMode:
        return {
            MediaMode.PREPARE: cls.MEDIA_PREPARE,
            MediaMode.DENOISE: cls.MEDIA_DENOISE,
            MediaMode.RECONSTRUCT: cls.MEDIA_RECONSTRUCT,
        }[mode]


class Domain(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
    FLOW = "flow"


_DOMAIN_BY_WORK_VARIANT = {
    ForwardMode.TOKEN_EXTEND: Domain.PREFILL,
    ForwardMode.TOKEN_DECODE: Domain.DECODE,
    ForwardMode.TOKEN_VERIFY: Domain.DECODE,
    ForwardMode.ENCODE_VISION: Domain.PREFILL,
    ForwardMode.ENCODE_LATENT: Domain.PREFILL,
    ForwardMode.TRANSFER_PRODUCT: Domain.PREFILL,
    ForwardMode.TRANSFER_KV_PUBLISH: Domain.PREFILL,
    ForwardMode.TRANSFER_KV_INSTALL: Domain.PREFILL,
    ForwardMode.MEDIA_PREPARE: Domain.FLOW,
    ForwardMode.MEDIA_DENOISE: Domain.FLOW,
    ForwardMode.MATERIALIZE: Domain.FLOW,
    ForwardMode.MEDIA_RECONSTRUCT: Domain.FLOW,
}


def execution_domain(work: ForwardMode) -> Domain:
    return _DOMAIN_BY_WORK_VARIANT[work]


class AttentionRegime(StrEnum):
    NONE = "none"
    CAUSAL = "causal"
    BIDIRECTIONAL = "bidirectional"
    HYBRID = "hybrid"


class SamplingOwnership(StrEnum):
    DESIGNATED_RANK = "designated_rank"
    DETERMINISTIC_SHARDED = "deterministic_sharded"


class ProductKind(StrEnum):
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
    DEVICE_TENSOR = "device_tensor"
    REQUEST_RELAY = "request_relay"
    PAGED_KV = "paged_kv"
    LATENT_ARENA = "latent_arena"
    HOST_STAGING = "host_staging"
    PINNED_OUTPUT = "pinned_output"


class DType(StrEnum):
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    I32 = "i32"
    I64 = "i64"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"


class OpStatus(StrEnum):
    OK = "ok"
    PREDICATED = "predicated"
    ERROR = "error"


class ErrorCode(StrEnum):
    INVALID_OPERATION = "invalid_operation"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    COMPUTE_ERROR = "compute_error"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class DrawLayout(StrEnum):
    TARGET_SAMPLING = "target_sampling"
    SPECULATIVE_PROPOSAL = "speculative_proposal"
    FLOW_NOISE = "flow_noise"


class Disposition(StrEnum):
    PUBLISH = "publish"
    RETAIN = "retain"
    DISCARD = "discard"


class CloseReason(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"
    PREEMPTED = "preempted"


_STATE_ADVANCING_WORK = frozenset(
    {
        ForwardMode.TOKEN_EXTEND,
        ForwardMode.TOKEN_DECODE,
        ForwardMode.TOKEN_VERIFY,
        ForwardMode.MEDIA_PREPARE,
        ForwardMode.MEDIA_DENOISE,
        ForwardMode.MEDIA_RECONSTRUCT,
    }
)

_FORWARD_MODE_BY_VALUE = {member.value: member for member in ForwardMode}


def native_partition(
    partition_id: int,
    submission_group: int,
    collective_seq: int,
    domain: Domain,
    route: int,
    attention: AttentionRegime,
    shape_class: int,
    operations: tuple[Operation, ...],
    block_tables: tuple[BlockTable, ...],
    new_cache_pages: tuple[CachePageAllocation, ...],
    forward_rows: tuple[RowGeometry, ...],
    latent_placements: Sequence[object],
    reconstruction_placements: Sequence[object],
) -> BatchPartition:
    """Assemble a partition from transport-constructed members.

    The native transport is produced by the engine's own encoder, so members
    already carry their registered identities and the validation that guards
    untrusted IPC maps is not repeated here.
    """

    partition = object.__new__(BatchPartition)
    set_field = object.__setattr__
    set_field(partition, "partition_id", partition_id)
    set_field(partition, "submission_group", submission_group)
    set_field(partition, "collective_seq", collective_seq)
    set_field(partition, "domain", domain)
    set_field(partition, "route", route)
    set_field(partition, "attention", attention)
    set_field(partition, "shape_class", shape_class)
    set_field(partition, "operations", operations)
    set_field(partition, "block_tables", block_tables)
    set_field(partition, "new_cache_pages", new_cache_pages)
    set_field(partition, "forward_rows", forward_rows)
    set_field(
        partition,
        "latent_placements",
        tuple(
            LatentPlacement.from_mapping(item, f"partition.latent_placements[{index}]")
            for index, item in enumerate(latent_placements)
        ),
    )
    set_field(
        partition,
        "reconstruction_placements",
        tuple(
            ReconstructionPlacement.from_mapping(
                item, f"partition.reconstruction_placements[{index}]"
            )
            for index, item in enumerate(reconstruction_placements)
        ),
    )
    return partition


def native_batch(
    step_id: int,
    admissions: Sequence[object],
    partitions: tuple[BatchPartition, ...],
    controls: tuple[Control, ...],
    input_products: Sequence[object],
) -> Batch:
    """Assemble a batch from transport-constructed members."""

    batch = object.__new__(Batch)
    set_field = object.__setattr__
    set_field(batch, "step_id", step_id)
    set_field(
        batch,
        "admissions",
        tuple(
            NewRequest.from_mapping(item, f"batch.admissions[{index}]")
            for index, item in enumerate(admissions)
        ),
    )
    set_field(batch, "partitions", partitions)
    set_field(batch, "controls", controls)
    set_field(
        batch,
        "input_products",
        tuple(
            ProductPayload.from_mapping(item, f"batch.input_products[{index}]")
            for index, item in enumerate(input_products)
        ),
    )
    return batch


@dataclass(frozen=True, slots=True)
class SamplingParams:
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
    authority_id: int
    session_id: int
    epoch: int

    def __post_init__(self) -> None:
        _nonnegative(self.authority_id, "request_key.authority_id")
        _nonnegative(self.session_id, "request_key.session_id")
        _nonnegative(self.epoch, "request_key.epoch")

    @classmethod
    def from_mapping(cls, value: object, where: str = "request_key") -> RequestKey:
        key = _fast_request_key(value)
        if key is not None:
            return key
        data = _map(value, where)
        return cls(
            authority_id=_uint(data.get("authority_id"), f"{where}.authority_id"),
            session_id=_uint(data.get("session_id"), f"{where}.session_id"),
            epoch=_uint(data.get("epoch"), f"{where}.epoch"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "authority_id": self.authority_id,
            "session_id": self.session_id,
            "epoch": self.epoch,
        }


@dataclass(frozen=True, slots=True)
class StaticDim:
    extent: int


@dataclass(frozen=True, slots=True)
class DeviceDim:
    bound: int


DimBound: TypeAlias = StaticDim | DeviceDim


@dataclass(frozen=True, slots=True)
class ShapeBound:
    dims: tuple[DimBound, ...] = ()

    def __post_init__(self) -> None:
        device_dims = sum(1 for dim in self.dims if isinstance(dim, DeviceDim))
        if device_dims > 1:
            raise invalid_descriptor("a shape bound carries more than one device-actual dimension")
        if any((dim.extent if isinstance(dim, StaticDim) else dim.bound) < 1 for dim in self.dims):
            raise invalid_descriptor("a shape bound contains a zero extent")

    @property
    def max_elements(self) -> int:
        elements = 1
        for dim in self.dims:
            elements *= dim.extent if isinstance(dim, StaticDim) else dim.bound
        return elements

    @classmethod
    def from_mapping(cls, value: object, where: str = "shape_bound") -> ShapeBound:
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
        return {"dims": [_dim_to_mapping(dim) for dim in self.dims]}


def _dim_to_mapping(dim: DimBound) -> dict[str, object]:
    if isinstance(dim, StaticDim):
        return {"kind": "static", "value": dim.extent}
    return {"kind": "device", "value": {"max": dim.bound}}


@dataclass(frozen=True, slots=True)
class PointRange:
    base_point: int = 0
    max_points: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "point_range") -> PointRange:
        point_range = _fast_point_range(value)
        if point_range is not None:
            return point_range
        data = _map(value, where)
        return cls(
            base_point=_uint(data.get("base_point"), f"{where}.base_point"),
            max_points=_uint(data.get("max_points"), f"{where}.max_points"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"base_point": self.base_point, "max_points": self.max_points}


@dataclass(frozen=True, slots=True)
class ProductRef:
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
        if self.generation < 1:
            raise invalid_descriptor("product reference has no logical generation")
        self.shape_bound.__post_init__()

    @property
    def max_bytes(self) -> int:
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

    @classmethod
    def from_mapping(cls, value: object, where: str = "product_ref") -> ProductRef:
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
class FixedPoint:
    point_index: int


@dataclass(frozen=True, slots=True)
class DevicePoint:
    point_index: int
    selected_point: ProductRef | None


Point: TypeAlias = FixedPoint | DevicePoint


@dataclass(frozen=True, slots=True)
class VersionRef:
    request_key: RequestKey
    producer_op_id: int
    point: Point

    def is_fixed(self) -> bool:
        return isinstance(self.point, FixedPoint)

    @classmethod
    def from_mapping(cls, value: object, where: str = "version_ref") -> VersionRef:
        reference = _fast_version_ref(value)
        if reference is not None:
            return reference
        data = _map(value, where)
        kind, payload = _tagged(data.get("point"), f"{where}.point")
        if kind == "fixed":
            inner = _map(payload, f"{where}.point.value")
            point: Point = FixedPoint(
                point_index=_uint(inner.get("point_index"), f"{where}.point.value.point_index")
            )
        elif kind == "device":
            inner = _map(payload, f"{where}.point.value")
            point = DevicePoint(
                point_index=_uint(inner.get("point_index"), f"{where}.point.value.point_index"),
                selected_point=(
                    None
                    if inner.get("selected_point") is None
                    else ProductRef.from_mapping(
                        inner.get("selected_point"), f"{where}.point.value.selected_point"
                    )
                ),
            )
        else:
            raise invalid_descriptor(f"{where}.point has unknown variant {kind!r}")
        return cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            producer_op_id=_uint(data.get("producer_op_id"), f"{where}.producer_op_id"),
            point=point,
        )

    def to_mapping(self) -> dict[str, object]:
        if isinstance(self.point, FixedPoint):
            point = {
                "kind": "fixed",
                "value": {"point_index": self.point.point_index},
            }
        else:
            point = {
                "kind": "device",
                "value": {
                    "point_index": self.point.point_index,
                    "selected_point": (
                        None
                        if self.point.selected_point is None
                        else self.point.selected_point.to_mapping()
                    ),
                },
            }
        return {
            "request_key": self.request_key.to_mapping(),
            "producer_op_id": self.producer_op_id,
            "point": point,
        }


@dataclass(frozen=True, slots=True)
class Bounds:
    max_points: int = 0
    max_tokens: int = 0
    max_kv_pages: int = 0
    max_latent_bytes: int = 0
    max_completion_bytes: int = 0
    max_transfer_bytes: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "bounds") -> Bounds:
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
    seed: int
    semantic_index_base: int
    draw_layout: DrawLayout

    @classmethod
    def from_mapping(cls, value: object, where: str = "rng") -> Rng:
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
        return {
            "seed": self.seed,
            "semantic_index_base": self.semantic_index_base,
            "draw_layout": self.draw_layout.value,
        }


@dataclass(frozen=True, slots=True)
class Operation:
    request_key: RequestKey
    op_id: int
    parent: VersionRef
    work: ForwardMode
    route: int
    domain: Domain
    advances_state: bool
    bounds: Bounds
    inputs: tuple[ProductRef, ...]
    outputs: tuple[ProductRef, ...]
    predicate: ProductRef | None
    rng: Rng | None
    control_seq: int

    @classmethod
    def registered(
        cls,
        *,
        request_key: RequestKey,
        op_id: int,
        parent: VersionRef,
        work: ForwardMode,
        route: int,
        domain: Domain,
        bounds: Bounds,
        inputs: tuple[ProductRef, ...] = (),
        outputs: tuple[ProductRef, ...] = (),
        predicate: ProductRef | None = None,
        rng: Rng | None = None,
        control_seq: int = 0,
    ) -> Operation:
        return cls(
            request_key=request_key,
            op_id=op_id,
            parent=parent,
            work=work,
            route=route,
            domain=domain,
            advances_state=work.advances_state,
            bounds=bounds,
            inputs=inputs,
            outputs=outputs,
            predicate=predicate,
            rng=rng,
            control_seq=control_seq,
        )

    def validate(self) -> None:
        if self.op_id < 1:
            raise invalid_descriptor("operation id must be positive")
        if self.domain is not execution_domain(self.work):
            raise invalid_descriptor("operation domain is inconsistent with its work variant")
        if self.advances_state != self.work.advances_state:
            raise invalid_descriptor(
                "operation declares an advances_state inconsistent with its work variant"
            )
        if self.parent.request_key != self.request_key:
            raise invalid_descriptor("operation parent belongs to another request lineage")
        if self.work.requires_fixed_parent and not isinstance(self.parent.point, FixedPoint):
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
        if isinstance(self.parent.point, DevicePoint):
            selected = self.parent.point.selected_point
            if selected is None:
                if self.parent.point.point_index < 1:
                    raise invalid_descriptor(
                        "a static device version must name a positive producer point"
                    )
            else:
                if self.parent.point.point_index != 0:
                    raise invalid_descriptor("a dynamic device version also declares a fixed point")
                if (
                    selected.request_key != self.parent.request_key
                    or selected.producer_op_id != self.parent.producer_op_id
                ):
                    raise invalid_descriptor(
                        "device version selected point is not owned by its producer"
                    )
                if selected.generation < 1:
                    raise invalid_descriptor(
                        "device version selected point has no logical generation"
                    )
                if (
                    selected.kind is not ProductKind.SELECTED_POINT
                    or selected.storage_class
                    not in {StorageClass.DEVICE_TENSOR, StorageClass.REQUEST_RELAY}
                    or selected.dtype is not DType.U32
                    or selected.shape_bound.max_elements != 1
                ):
                    raise invalid_descriptor(
                        "device version does not name a scalar selected-point product"
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
            parent = VersionRef.from_mapping(get("parent"), f"{where}.parent")
        work = _fast_work(get("work"))
        if work is None:
            work = _enum(ForwardMode, get("work"), f"{where}.work")
        route = get("route")
        if not (type(route) is int and route >= 0):
            route = _uint(route, f"{where}.route")
        domain_raw = get("domain")
        domain = _DOMAIN_BY_VALUE.get(domain_raw) if type(domain_raw) is str else None
        if domain is None:
            domain = _enum(Domain, domain_raw, f"{where}.domain")
        advances_state = get("advances_state")
        if advances_state is not True and advances_state is not False:
            advances_state = _bool(advances_state, f"{where}.advances_state")
        bounds = _fast_bounds(get("bounds"))
        if bounds is None:
            bounds = Bounds.from_mapping(get("bounds"), f"{where}.bounds")
        inputs = _fast_product_refs(get("inputs", ()))
        if inputs is None:
            inputs = tuple(
                ProductRef.from_mapping(item, f"{where}.inputs[{index}]")
                for index, item in enumerate(_seq(data.get("inputs", ()), f"{where}.inputs"))
            )
        outputs = _fast_product_refs(get("outputs", ()))
        if outputs is None:
            outputs = tuple(
                ProductRef.from_mapping(item, f"{where}.outputs[{index}]")
                for index, item in enumerate(_seq(data.get("outputs", ()), f"{where}.outputs"))
            )
        predicate_raw = get("predicate")
        if predicate_raw is None:
            predicate = None
        else:
            predicate = _fast_product_ref(predicate_raw)
            if predicate is None:
                predicate = ProductRef.from_mapping(predicate_raw, f"{where}.predicate")
        rng_raw = get("rng")
        if rng_raw is None:
            rng = None
        else:
            rng = _fast_rng(rng_raw)
            if rng is None:
                rng = Rng.from_mapping(rng_raw, f"{where}.rng")
        control_seq = get("control_seq")
        if not (type(control_seq) is int and control_seq >= 0):
            control_seq = _uint(control_seq, f"{where}.control_seq")
        operation = cls(
            request_key=request_key,
            op_id=op_id,
            parent=parent,
            work=work,
            route=route,
            domain=domain,
            advances_state=advances_state,
            bounds=bounds,
            inputs=inputs,
            outputs=outputs,
            predicate=predicate,
            rng=rng,
            control_seq=control_seq,
        )
        operation.validate()
        return operation

    def to_mapping(self) -> dict[str, object]:
        return {
            "request_key": self.request_key.to_mapping(),
            "op_id": self.op_id,
            "parent": self.parent.to_mapping(),
            "work": self.work.value,
            "route": self.route,
            "domain": self.domain.value,
            "advances_state": self.advances_state,
            "bounds": self.bounds.to_mapping(),
            "inputs": [product.to_mapping() for product in self.inputs],
            "outputs": [product.to_mapping() for product in self.outputs],
            "predicate": None if self.predicate is None else self.predicate.to_mapping(),
            "rng": None if self.rng is None else self.rng.to_mapping(),
            "control_seq": self.control_seq,
        }


@dataclass(frozen=True, slots=True)
class LogicalLengths:
    token_len: int = 0
    kv_visible_len: int = 0
    kv_computed_len: int = 0
    latent_len: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "logical_lengths") -> LogicalLengths:
        data = _map(value, where)
        return cls(
            token_len=_uint(data.get("token_len"), f"{where}.token_len"),
            kv_visible_len=_uint(data.get("kv_visible_len"), f"{where}.kv_visible_len"),
            kv_computed_len=_uint(data.get("kv_computed_len"), f"{where}.kv_computed_len"),
            latent_len=_uint(data.get("latent_len"), f"{where}.latent_len"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "token_len": self.token_len,
            "kv_visible_len": self.kv_visible_len,
            "kv_computed_len": self.kv_computed_len,
            "latent_len": self.latent_len,
        }


@dataclass(frozen=True, slots=True)
class TokenSpan:
    base: int = 0
    len: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "token_span") -> TokenSpan:
        data = _map(value, where)
        return cls(
            base=_uint(data.get("base"), f"{where}.base"),
            len=_uint(data.get("len"), f"{where}.len"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"base": self.base, "len": self.len}


@dataclass(frozen=True, slots=True)
class FinishFlags:
    eos: bool = False
    length: bool = False
    stop: bool = False

    @classmethod
    def from_mapping(cls, value: object, where: str = "finish_flags") -> FinishFlags:
        data = _map(value, where)
        return cls(
            eos=_bool(data.get("eos", False), f"{where}.eos"),
            length=_bool(data.get("length", False), f"{where}.length"),
            stop=_bool(data.get("stop", False), f"{where}.stop"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"eos": self.eos, "length": self.length, "stop": self.stop}


@dataclass(frozen=True, slots=True)
class TimingCounters:
    queued_us: int = 0
    device_us: int = 0
    copy_us: int = 0
    host_us: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "timing_counters") -> TimingCounters:
        data = _map(value, where)
        return cls(
            queued_us=_uint(data.get("queued_us", 0), f"{where}.queued_us"),
            device_us=_uint(data.get("device_us", 0), f"{where}.device_us"),
            copy_us=_uint(data.get("copy_us", 0), f"{where}.copy_us"),
            host_us=_uint(data.get("host_us", 0), f"{where}.host_us"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "queued_us": self.queued_us,
            "device_us": self.device_us,
            "copy_us": self.copy_us,
            "host_us": self.host_us,
        }


@dataclass(frozen=True, slots=True)
class MediaOutput:
    handle: str
    bytes: int

    def __post_init__(self) -> None:
        if not self.handle or self.bytes < 1:
            raise invalid_descriptor("media output locator is invalid")

    @classmethod
    def from_mapping(cls, value: object, where: str = "media_output") -> MediaOutput:
        data = _map(value, where)
        return cls(
            handle=_str(data.get("handle"), f"{where}.handle"),
            bytes=_uint(data.get("bytes"), f"{where}.bytes"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"handle": self.handle, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class ModelOutput:
    request_key: RequestKey
    op_id: int
    completion_slot_generation: int
    status: OpStatus
    selected_point: int
    logical_lengths: LogicalLengths
    token_span: TokenSpan
    committed_tokens: tuple[int, ...]
    finish_flags: FinishFlags
    product_generations: tuple[int, ...]
    error_code: ErrorCode | None
    timing_counters: TimingCounters
    media_output: MediaOutput | None = None

    def validate(self) -> None:
        if self.op_id < 1:
            raise invalid_descriptor("completion op id must be positive")
        if self.logical_lengths.kv_visible_len > self.logical_lengths.kv_computed_len:
            raise invalid_descriptor("completion selected KV length exceeds computed length")
        if self.completion_slot_generation < 1:
            raise invalid_descriptor("completion slot generation must be positive")
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
        data = _map(value, where)
        record = cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            op_id=_uint(data.get("op_id"), f"{where}.op_id"),
            completion_slot_generation=_uint(
                data.get("completion_slot_generation"), f"{where}.completion_slot_generation"
            ),
            status=_enum(OpStatus, data.get("status"), f"{where}.status"),
            selected_point=_uint(data.get("selected_point"), f"{where}.selected_point"),
            logical_lengths=LogicalLengths.from_mapping(
                data.get("logical_lengths"), f"{where}.logical_lengths"
            ),
            token_span=TokenSpan.from_mapping(data.get("token_span"), f"{where}.token_span"),
            committed_tokens=_uints(data.get("committed_tokens", ()), f"{where}.committed_tokens"),
            finish_flags=FinishFlags.from_mapping(
                data.get("finish_flags"), f"{where}.finish_flags"
            ),
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
            media_output=(
                None
                if data.get("media_output") is None
                else MediaOutput.from_mapping(data["media_output"], f"{where}.media_output")
            ),
        )
        record.validate()
        return record

    def to_mapping(self) -> dict[str, object]:
        key = self.request_key
        lengths = self.logical_lengths
        span = self.token_span
        flags = self.finish_flags
        timing = self.timing_counters
        error_code = self.error_code
        return {
            "request_key": {
                "authority_id": key.authority_id,
                "session_id": key.session_id,
                "epoch": key.epoch,
            },
            "op_id": self.op_id,
            "completion_slot_generation": self.completion_slot_generation,
            "status": self.status.value,
            "selected_point": self.selected_point,
            "logical_lengths": {
                "token_len": lengths.token_len,
                "kv_visible_len": lengths.kv_visible_len,
                "kv_computed_len": lengths.kv_computed_len,
                "latent_len": lengths.latent_len,
            },
            "token_span": {"base": span.base, "len": span.len},
            "committed_tokens": list(self.committed_tokens),
            "finish_flags": {"eos": flags.eos, "length": flags.length, "stop": flags.stop},
            "product_generations": list(self.product_generations),
            "error_code": None if error_code is None else error_code.value,
            "timing_counters": {
                "queued_us": timing.queued_us,
                "device_us": timing.device_us,
                "copy_us": timing.copy_us,
                "host_us": timing.host_us,
            },
            "media_output": (
                None if self.media_output is None else self.media_output.to_mapping()
            ),
        }


@dataclass(frozen=True, slots=True)
class Commit:
    request_key: RequestKey
    control_seq: int
    expected_parent: VersionRef
    selected: VersionRef
    public_event_limit: int
    disposition: Disposition


@dataclass(frozen=True, slots=True)
class Close:
    request_key: RequestKey
    control_seq: int
    cutoff: VersionRef
    reason: CloseReason


@dataclass(frozen=True, slots=True)
class Release:
    request_key: RequestKey
    op_id: int


Control: TypeAlias = Commit | Close | Release


def _control_variant_index(control: Control) -> int:
    if isinstance(control, Commit):
        return 0
    if isinstance(control, Close):
        return 1
    return 2


def control_from_mapping(
    value: object,
    where: str = "control",
) -> Control:
    kind, payload = _tagged(value, where)
    data = _map(payload, f"{where}.value")
    request_key = _fast_request_key(data.get("request_key"))
    if request_key is None:
        request_key = RequestKey.from_mapping(data.get("request_key"), f"{where}.value.request_key")
    if kind == "commit":
        expected_parent = _fast_version_ref(data.get("expected_parent"))
        if expected_parent is None:
            expected_parent = VersionRef.from_mapping(
                data.get("expected_parent"), f"{where}.value.expected_parent"
            )
        selected = _fast_version_ref(data.get("selected"))
        if selected is None:
            selected = VersionRef.from_mapping(data.get("selected"), f"{where}.value.selected")
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
            raise invalid_descriptor("a commit control must select a fixed version")
        control: Control = commit
    elif kind == "close":
        cutoff = _fast_version_ref(data.get("cutoff"))
        if cutoff is None:
            cutoff = VersionRef.from_mapping(data.get("cutoff"), f"{where}.value.cutoff")
        control = Close(
            request_key=request_key,
            control_seq=_uint(data.get("control_seq"), f"{where}.value.control_seq"),
            cutoff=cutoff,
            reason=_enum(CloseReason, data.get("reason"), f"{where}.value.reason"),
        )
        if not control.cutoff.is_fixed():
            raise invalid_descriptor("a close control must name a fixed cutoff version")
    elif kind == "release":
        control = Release(
            request_key=request_key,
            op_id=_uint(data.get("op_id"), f"{where}.value.op_id"),
        )
    else:
        raise invalid_descriptor(f"{where} has unknown variant {kind!r}")
    return control


def control_to_mapping(control: Control) -> dict[str, object]:
    if isinstance(control, Commit):
        return {
            "kind": "commit",
            "value": {
                "request_key": control.request_key.to_mapping(),
                "control_seq": control.control_seq,
                "expected_parent": control.expected_parent.to_mapping(),
                "selected": control.selected.to_mapping(),
                "public_event_limit": control.public_event_limit,
                "disposition": control.disposition.value,
            },
        }
    if isinstance(control, Close):
        return {
            "kind": "close",
            "value": {
                "request_key": control.request_key.to_mapping(),
                "control_seq": control.control_seq,
                "cutoff": control.cutoff.to_mapping(),
                "reason": control.reason.value,
            },
        }
    return {
        "kind": "release",
        "value": {"request_key": control.request_key.to_mapping(), "op_id": control.op_id},
    }


@dataclass(frozen=True, slots=True)
class UndAdmission:
    sampling: SamplingParams = field(default_factory=SamplingParams)
    negative_token_ids: tuple[int, ...] = ()
    finish_token_ids: tuple[int, ...] = ()
    initial_position: int = 0

    def __post_init__(self) -> None:
        _nonnegative(self.initial_position, "und admission initial position")
        if any(
            left >= right
            for left, right in zip(
                self.finish_token_ids,
                self.finish_token_ids[1:],
                strict=False,
            )
        ):
            raise invalid_descriptor("und admission finish token ids are not canonical")

    @classmethod
    def from_mapping(cls, value: object, where: str = "und admission") -> UndAdmission:
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
        return {
            "sampling": self.sampling.to_mapping(),
            "negative_token_ids": list(self.negative_token_ids),
            "finish_token_ids": list(self.finish_token_ids),
            "initial_position": self.initial_position,
        }


@dataclass(frozen=True, slots=True)
class GenAdmission:
    image: ImageParams = field(default_factory=ImageParams)

    @classmethod
    def from_mapping(cls, value: object, where: str = "gen admission") -> GenAdmission:
        data = _map(value, where)
        return cls(image=ImageParams.from_mapping(data.get("image", {}), f"{where}.image"))

    def to_mapping(self) -> dict[str, object]:
        return {"image": self.image.to_mapping()}


@dataclass(frozen=True, slots=True)
class MediaGeometry:
    frame_count: int
    video_reconstruction_units: int
    audio_latent_frames: int
    prompt_tokens: int
    denoise_steps: int

    def __post_init__(self) -> None:
        for name in (
            "frame_count",
            "video_reconstruction_units",
            "audio_latent_frames",
            "prompt_tokens",
            "denoise_steps",
        ):
            _nonnegative(getattr(self, name), f"media geometry {name}")
        if (
            self.frame_count < 22
            or self.frame_count % 17 != 5
            or self.video_reconstruction_units != (self.frame_count - 5) // 17
            or self.audio_latent_frames != (self.frame_count * 40 + 23) // 24
            or self.prompt_tokens == 0
            or self.denoise_steps != 4
        ):
            raise invalid_descriptor("media geometry is invalid")

    @classmethod
    def from_mapping(cls, value: object, where: str = "media geometry") -> MediaGeometry:
        data = _map(value, where)
        return cls(
            frame_count=_uint(data.get("frame_count"), f"{where}.frame_count"),
            video_reconstruction_units=_uint(
                data.get("video_reconstruction_units"), f"{where}.video_reconstruction_units"
            ),
            audio_latent_frames=_uint(
                data.get("audio_latent_frames"), f"{where}.audio_latent_frames"
            ),
            prompt_tokens=_uint(data.get("prompt_tokens"), f"{where}.prompt_tokens"),
            denoise_steps=_uint(data.get("denoise_steps"), f"{where}.denoise_steps"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "video_reconstruction_units": self.video_reconstruction_units,
            "audio_latent_frames": self.audio_latent_frames,
            "prompt_tokens": self.prompt_tokens,
            "denoise_steps": self.denoise_steps,
        }


@dataclass(frozen=True, slots=True)
class MediaAdmission:
    prompt_token_ids: tuple[int, ...]
    seed: int
    profile: MediaProfileId
    geometry: MediaGeometry

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise invalid_descriptor("media admission prompt tokens must not be empty")
        _nonnegative(self.seed, "media admission seed")
        if len(self.prompt_token_ids) != self.geometry.prompt_tokens:
            raise invalid_descriptor("media admission prompt tokens disagree with its geometry")

    @classmethod
    def from_mapping(cls, value: object, where: str = "media admission") -> MediaAdmission:
        data = _map(value, where)
        return cls(
            prompt_token_ids=_uints(data.get("prompt_token_ids", ()), f"{where}.prompt_token_ids"),
            seed=_uint(data.get("seed"), f"{where}.seed"),
            profile=_enum(MediaProfileId, data.get("profile"), f"{where}.profile"),
            geometry=MediaGeometry.from_mapping(data.get("geometry"), f"{where}.geometry"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "prompt_token_ids": list(self.prompt_token_ids),
            "seed": self.seed,
            "profile": self.profile.value,
            "geometry": self.geometry.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class NewRequest:
    request_key: RequestKey
    request_pool_idx: int
    und: UndAdmission | None
    gen_admission: GenAdmission | None
    media: MediaAdmission | None = None

    def __post_init__(self) -> None:
        if self.request_pool_idx < 1:
            raise invalid_descriptor("request-pool index must be positive")
        if self.und is None and self.gen_admission is None and self.media is None:
            raise invalid_descriptor(
                "admission must declare an understanding, generation, or media branch"
            )

    @classmethod
    def create(
        cls,
        request_key: RequestKey,
        *,
        request_pool_idx: int,
        und: UndAdmission | None = None,
        gen_admission: GenAdmission | None = None,
        media: MediaAdmission | None = None,
    ) -> NewRequest:
        return cls(request_key, request_pool_idx, und, gen_admission, media)

    @classmethod
    def from_mapping(cls, value: object, where: str = "admission") -> NewRequest:
        data = _map(value, where)
        admission = cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            request_pool_idx=_uint(data.get("request_pool_idx"), f"{where}.request_pool_idx"),
            und=(
                None
                if data.get("und") is None
                else UndAdmission.from_mapping(data["und"], f"{where}.und")
            ),
            gen_admission=(
                None
                if data.get("gen_admission") is None
                else GenAdmission.from_mapping(data["gen_admission"], f"{where}.gen_admission")
            ),
            media=(
                None
                if data.get("media") is None
                else MediaAdmission.from_mapping(data["media"], f"{where}.media")
            ),
        )
        return admission

    def to_mapping(self) -> dict[str, object]:
        return {
            "request_key": self.request_key.to_mapping(),
            "request_pool_idx": self.request_pool_idx,
            "und": None if self.und is None else self.und.to_mapping(),
            "gen_admission": None
            if self.gen_admission is None
            else self.gen_admission.to_mapping(),
            "media": None if self.media is None else self.media.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class BlockTable:
    request_pool_idx: int
    group_id: int
    page_ids: tuple[int, ...]
    allocated_tokens: int

    def __post_init__(self) -> None:
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
        data = _map(value, where)

        def uint_field(name: str) -> int:
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
        return {
            "request_pool_idx": self.request_pool_idx,
            "group_id": self.group_id,
            "page_ids": list(self.page_ids),
            "allocated_tokens": self.allocated_tokens,
        }


@dataclass(frozen=True, slots=True)
class CachePageAllocation:
    request_pool_idx: int
    group_id: int
    page_ids: tuple[int, ...]

    def __post_init__(self) -> None:
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
        data = _map(value, where)

        def uint_field(name: str) -> int:
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
        return {
            "request_pool_idx": self.request_pool_idx,
            "group_id": self.group_id,
            "page_ids": list(self.page_ids),
        }


@dataclass(frozen=True, slots=True)
class RowGeometry:
    operation_index: int
    request_pool_index: int
    seq_len: int
    query_len: int

    def __post_init__(self) -> None:
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
        data = _map(value, where)
        return cls(
            operation_index=_uint(data.get("operation_index"), f"{where}.operation_index"),
            request_pool_index=_uint(data.get("request_pool_index"), f"{where}.request_pool_index"),
            seq_len=_uint(data.get("seq_len"), f"{where}.seq_len"),
            query_len=_uint(data.get("query_len"), f"{where}.query_len"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "operation_index": self.operation_index,
            "request_pool_index": self.request_pool_index,
            "seq_len": self.seq_len,
            "query_len": self.query_len,
        }


@dataclass(frozen=True, slots=True)
class LatentPlacement:
    request_key: RequestKey
    op_id: int
    page_table: tuple[int, ...]
    latent_units: int
    height: int
    width: int
    start_step: int
    step_count: int

    def __post_init__(self) -> None:
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
class ReconstructionPlacement:
    request_key: RequestKey
    op_id: int
    kind: ReconstructionKind
    start_unit: int
    unit_count: int

    def __post_init__(self) -> None:
        if self.op_id < 1 or self.unit_count < 1:
            raise invalid_descriptor(
                "reconstruction placement identity and unit count must be positive"
            )
        _nonnegative(self.start_unit, "reconstruction placement start unit")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "reconstruction placement"
    ) -> ReconstructionPlacement:
        data = _map(value, where)
        return cls(
            request_key=RequestKey.from_mapping(data.get("request_key"), f"{where}.request_key"),
            op_id=_uint(data.get("op_id"), f"{where}.op_id"),
            kind=_enum(ReconstructionKind, data.get("kind"), f"{where}.kind"),
            start_unit=_uint(data.get("start_unit"), f"{where}.start_unit"),
            unit_count=_uint(data.get("unit_count"), f"{where}.unit_count"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "request_key": self.request_key.to_mapping(),
            "op_id": self.op_id,
            "kind": self.kind.value,
            "start_unit": self.start_unit,
            "unit_count": self.unit_count,
        }


@dataclass(frozen=True, slots=True)
class BatchPartition:
    partition_id: int
    submission_group: int
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
    reconstruction_placements: tuple[ReconstructionPlacement, ...] = ()

    def __post_init__(self) -> None:
        if (
            min(
                self.partition_id,
                self.submission_group,
                self.collective_seq,
            )
            < 1
        ):
            raise invalid_descriptor("batch partition identity must be positive")
        if self.route < 0 or self.shape_class < 0:
            raise invalid_descriptor("batch partition route and shape class must be unsigned")
        if not self.operations:
            raise invalid_descriptor("batch partition must carry at least one operation")
        if any(
            operation.domain is not self.domain or operation.route != self.route
            for operation in self.operations
        ):
            raise invalid_descriptor("batch partition operation disagrees with its domain or route")
        operations = {
            (operation.request_key, operation.op_id): operation for operation in self.operations
        }
        tables = {(table.request_pool_idx, table.group_id): table for table in self.block_tables}
        if len(tables) != len(self.block_tables):
            raise invalid_descriptor("batch partition repeats a block table")
        allocation_ids: set[tuple[int, int]] = set()
        for allocation in self.new_cache_pages:
            identity = (allocation.request_pool_idx, allocation.group_id)
            if identity in allocation_ids:
                raise invalid_descriptor("batch partition repeats a cache-page allocation")
            allocation_ids.add(identity)
            table = tables.get(identity)
            if table is None or not set(allocation.page_ids).issubset(table.page_ids):
                raise invalid_descriptor("cache-page allocation has no matching block table")
        for row in self.forward_rows:
            if row.operation_index >= len(self.operations):
                raise invalid_descriptor("forward row operation index is outside its partition")
        latent_ids: set[tuple[RequestKey, int]] = set()
        latent_pages: set[int] = set()

        def addresses_trajectory(operation: Operation) -> bool:
            return operation.work in {
                ForwardMode.MEDIA_PREPARE,
                ForwardMode.MEDIA_DENOISE,
            } or any(reference.kind is ProductKind.LATENT for reference in operation.inputs)

        for latent_placement in self.latent_placements:
            latent_identity = (latent_placement.request_key, latent_placement.op_id)
            if latent_identity in latent_ids:
                raise invalid_descriptor("batch partition repeats a latent placement identity")
            latent_ids.add(latent_identity)
            operation = operations.get(latent_identity)
            if operation is None:
                raise invalid_descriptor("latent placement does not name a partition operation")
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
        reconstruction_ids: set[tuple[RequestKey, int]] = set()
        for placement in self.reconstruction_placements:
            identity = (placement.request_key, placement.op_id)
            if identity in reconstruction_ids:
                raise invalid_descriptor(
                    "batch partition repeats a reconstruction placement identity"
                )
            reconstruction_ids.add(identity)
            operation = operations.get(identity)
            if operation is None or operation.work is not ForwardMode.MEDIA_RECONSTRUCT:
                raise invalid_descriptor(
                    "reconstruction placement does not name a media reconstruction operation"
                )
        if any(
            operation.work is ForwardMode.MEDIA_RECONSTRUCT
            and (operation.request_key, operation.op_id) not in reconstruction_ids
            for operation in self.operations
        ):
            raise invalid_descriptor(
                "media reconstruction operation has no reconstruction placement"
            )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "batch partition",
    ) -> BatchPartition:
        data = _map(value, where)
        fields = dict(
            partition_id=_uint(data.get("partition_id"), f"{where}.partition_id"),
            submission_group=_uint(data.get("submission_group"), f"{where}.submission_group"),
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
            reconstruction_placements=tuple(
                ReconstructionPlacement.from_mapping(
                    item, f"{where}.reconstruction_placements[{index}]"
                )
                for index, item in enumerate(
                    _seq(
                        data.get("reconstruction_placements", ()),
                        f"{where}.reconstruction_placements",
                    )
                )
            ),
        )
        return cls(**cast(Any, fields))

    def to_mapping(self) -> dict[str, object]:
        return {
            "partition_id": self.partition_id,
            "submission_group": self.submission_group,
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
            "reconstruction_placements": [
                placement.to_mapping() for placement in self.reconstruction_placements
            ],
        }


@dataclass(frozen=True, slots=True)
class Batch:
    step_id: int
    admissions: tuple[NewRequest, ...] = ()
    partitions: tuple[BatchPartition, ...] = ()
    controls: tuple[Control, ...] = ()
    input_products: tuple[ProductPayload, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    @property
    def operations(self) -> tuple[Operation, ...]:
        return tuple(
            operation for partition in self.partitions for operation in partition.operations
        )

    def validate(self) -> None:
        if not self.partitions and not self.controls:
            raise invalid_descriptor(
                "a submission batch must carry at least one operation or control"
            )
        partition_ids = [partition.partition_id for partition in self.partitions]
        if len(set(partition_ids)) != len(partition_ids):
            raise invalid_descriptor("a submission batch repeats a partition id")
        groups: dict[int, list[BatchPartition]] = {}
        for partition in self.partitions:
            groups.setdefault(partition.submission_group, []).append(partition)
        for partitions in groups.values():
            if sum(bool(partition.latent_placements) for partition in partitions) > 1:
                raise invalid_descriptor(
                    "a physical submission group has multiple latent staging partitions"
                )
            collective_seq = partitions[0].collective_seq
            attention = partitions[0].attention
            shape_class = partitions[0].shape_class
            if any(
                partition.collective_seq != collective_seq
                or partition.attention is not attention
                or partition.shape_class != shape_class
                for partition in partitions
            ):
                raise invalid_descriptor(
                    "physical submission partitions disagree on attention, shape, or collective order"
                )
            if len(partitions) >= 2:
                domains = {partition.domain for partition in partitions}
                routes = {partition.route for partition in partitions}
                if len(domains) != len(partitions):
                    raise invalid_descriptor(
                        "a tensorized-mixed submission group must contain distinct domains"
                    )
                if len(routes) != 1:
                    raise invalid_descriptor(
                        "a tensorized-mixed submission group spans execution routes"
                    )
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
        identities: dict[tuple[RequestKey, int | None, int], Control] = {}
        for control in self.controls:
            seq = control.control_seq if isinstance(control, (Commit, Close)) else None
            identity = (control.request_key, seq, _control_variant_index(control))
            existing = identities.get(identity)
            if existing is not None and existing != control:
                raise invalid_descriptor(
                    "a submission batch reuses a control identity with different content"
                )
            identities[identity] = control
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
            transferred = is_transfer_descriptor(payload.payload)
            if product.storage_class is StorageClass.HOST_STAGING and transferred:
                raise invalid_descriptor(
                    "host-staging input cannot carry a cross-stage transfer descriptor"
                )
            if product.storage_class is not StorageClass.HOST_STAGING:
                try:
                    validate_transfer_descriptor_frame(payload.payload)
                except ValueError as error:
                    raise invalid_descriptor(str(error)) from error
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

    @classmethod
    def from_mapping(cls, value: object) -> Batch:
        data = _map(value, "execute batch")
        step_id = _uint(data.get("step_id"), "execute batch.step_id")
        admissions = tuple(
            NewRequest.from_mapping(item, f"execute batch.admissions[{index}]")
            for index, item in enumerate(
                _seq(data.get("admissions", ()), "execute batch.admissions")
            )
        )
        partitions = tuple(
            BatchPartition.from_mapping(
                item,
                f"execute batch.partitions[{index}]",
            )
            for index, item in enumerate(
                _seq(data.get("partitions", ()), "execute batch.partitions")
            )
        )
        controls = tuple(
            _fast_release_control(item)
            or control_from_mapping(
                item,
                f"execute batch.controls[{index}]",
            )
            for index, item in enumerate(_seq(data.get("controls", ()), "execute batch.controls"))
        )
        input_products = tuple(
            ProductPayload.from_mapping(item, f"execute batch.input_products[{index}]")
            for index, item in enumerate(
                _seq(data.get("input_products", ()), "execute batch.input_products")
            )
        )
        return cls(
            step_id=step_id,
            admissions=admissions,
            partitions=partitions,
            controls=controls,
            input_products=input_products,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "admissions": [value.to_mapping() for value in self.admissions],
            "partitions": [value.to_mapping() for value in self.partitions],
            "controls": [control_to_mapping(value) for value in self.controls],
            "input_products": [value.to_mapping() for value in self.input_products],
        }


@dataclass(frozen=True, slots=True)
class RegistrationAck:
    visible: bool = False

    @classmethod
    def from_mapping(cls, value: object, where: str = "registration") -> RegistrationAck:
        data = _map(value, where)
        return cls(visible=_bool(data.get("visible", False), f"{where}.visible"))

    def to_mapping(self) -> dict[str, object]:
        return {"visible": self.visible}


@dataclass(frozen=True, slots=True)
class ProductPayload:
    product: ProductRef
    payload: bytes

    @classmethod
    def from_mapping(cls, value: object, where: str = "product payload") -> ProductPayload:
        data = _map(value, where)
        return cls(
            product=ProductRef.from_mapping(data.get("product"), f"{where}.product"),
            payload=(
                raw
                if type(raw := data.get("bytes", b"")) is bytes
                else bytes(raw)
                if isinstance(raw, (bytearray, memoryview))
                else bytes(_uints(raw, f"{where}.bytes"))
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"product": self.product.to_mapping(), "bytes": self.payload}


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
    offset = 0

    def take_u32() -> int:
        nonlocal offset
        if offset + 4 > len(data):
            raise invalid_descriptor("sampling-state bytes are truncated")
        value = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        return value

    def take_ids(count: int) -> tuple[int, ...]:
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
        data = _map(value, where)

        def counter_map(name: str) -> dict[str, int]:
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
        return {
            name: dict(value) if isinstance(value, Mapping) else value
            for name, value in ((name, getattr(self, name)) for name in self.__dataclass_fields__)
        }


@dataclass(frozen=True, slots=True)
class PartitionCompletion:
    partition_id: int
    completions: tuple[ModelOutput | CompletionState, ...]
    products: tuple[ProductPayload, ...] = ()
    registration: RegistrationAck = field(default_factory=RegistrationAck)
    worker_exec_us: int | None = None
    forward_stats: WorkerForwardStats | None = None
    publication: PartitionPublication | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "partition completion",
    ) -> PartitionCompletion:
        data = _map(value, where)
        return cls(
            partition_id=_uint(data.get("partition_id"), f"{where}.partition_id"),
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
        return {
            "partition_id": self.partition_id,
            "completions": [cast(ModelOutput, value).to_mapping() for value in self.completions],
            "products": [value.to_mapping() for value in self.products],
            "registration": self.registration.to_mapping(),
            "worker_exec_us": self.worker_exec_us,
            "forward_stats": None
            if self.forward_stats is None
            else self.forward_stats.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class CompletionReport:
    step_id: int
    partitions: tuple[PartitionCompletion, ...]

    @property
    def completions(self) -> tuple[ModelOutput | CompletionState, ...]:
        return tuple(
            completion for partition in self.partitions for completion in partition.completions
        )

    @property
    def products(self) -> tuple[ProductPayload, ...]:
        return tuple(product for partition in self.partitions for product in partition.products)

    @property
    def registration(self) -> RegistrationAck:
        return RegistrationAck(
            visible=bool(self.partitions)
            and all(partition.registration.visible for partition in self.partitions)
        )

    @property
    def worker_exec_us(self) -> int | None:
        values = tuple(
            partition.worker_exec_us
            for partition in self.partitions
            if partition.worker_exec_us is not None
        )
        return None if not values else max(values)

    @classmethod
    def from_mapping(cls, value: object, where: str = "completion report") -> CompletionReport:
        data = _map(value, where)
        return cls(
            step_id=_uint(data.get("step_id"), f"{where}.step_id"),
            partitions=tuple(
                PartitionCompletion.from_mapping(item, f"{where}.partitions[{index}]")
                for index, item in enumerate(
                    _seq(data.get("partitions", ()), f"{where}.partitions")
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "partitions": [partition.to_mapping() for partition in self.partitions],
        }


# ---------------------------------------------------------------------------
# Decode helpers
#
# Each `_fast_*` helper recognizes the exact built-in IPC shape without
# allocating error-location strings. A non-matching value returns ``None`` so
# the caller applies the canonical validated constructor and its precise error.
# ---------------------------------------------------------------------------

_E = TypeVar("_E", bound=StrEnum)

_DOMAIN_BY_VALUE: Mapping[str, Domain] = Domain._value2member_map_  # type: ignore[assignment]
_PRODUCT_KIND_BY_VALUE: Mapping[str, ProductKind] = ProductKind._value2member_map_  # type: ignore[assignment]
_STORAGE_CLASS_BY_VALUE: Mapping[str, StorageClass] = StorageClass._value2member_map_  # type: ignore[assignment]
_DTYPE_BY_VALUE: Mapping[str, DType] = DType._value2member_map_  # type: ignore[assignment]
_DRAW_LAYOUT_BY_VALUE: Mapping[str, DrawLayout] = DrawLayout._value2member_map_  # type: ignore[assignment]


def _enum(kind: type[_E], value: object, where: str) -> _E:
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
    if type(value) is dict:
        return value
    if not isinstance(value, Mapping):
        raise invalid_descriptor(f"{where} must be a map")
    return cast(Mapping[str, Any], value)


def _seq(value: object, where: str) -> Sequence[Any]:
    kind = type(value)
    if kind is list or kind is tuple:
        return cast(Sequence[Any], value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{where} must be a list")
    return value


def _pair(value: object, where: str) -> Sequence[Any]:
    items = _seq(value, where)
    if len(items) != 2:
        raise invalid_descriptor(f"{where} must contain two values")
    return items


def _tagged(value: object, where: str) -> tuple[str, object]:
    data = _map(value, where)
    return _str(data.get("kind"), f"{where}.kind"), data.get("value")


def _str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    return value


def _bool(value: object, where: str) -> bool:
    if value is True or value is False:
        return value
    raise invalid_descriptor(f"{where} must be a bool")


def _uint(value: object, where: str) -> int:
    if type(value) is int and value >= 0:
        return value
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"{where} must be a non-negative integer")
    return value


def _optional_uint(value: object, where: str) -> int | None:
    return None if value is None else _uint(value, where)


def _float(value: object, where: str) -> float:
    kind = type(value)
    if kind is not float and kind is not int:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise invalid_descriptor(f"{where} must be a number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise invalid_descriptor(f"{where} must be finite")
    return result


def _uints(value: object, where: str) -> tuple[int, ...]:
    items = _seq(value, where)
    for item in items:
        if not (type(item) is int and item >= 0):
            return tuple(_uint(item, f"{where}[{index}]") for index, item in enumerate(items))
    return tuple(items)


def _nonnegative(value: int, where: str) -> None:
    _uint(value, where)


# ---------------------------------------------------------------------------
# Allocation-light record decoders for the per-batch hot path
#
# Each returns the decoded record for a well-formed IPC value and ``None``
# otherwise; the caller uses validating decoders in declaration order so the
# first invalid field receives a precise diagnostic.
# Construction bypasses ``__init__``/``__post_init__`` only where the fast
# path itself enforces everything those validators check.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8192)
def _interned_request_key(authority_id: int, session_id: int, epoch: int) -> RequestKey:
    key = object.__new__(RequestKey)
    object.__setattr__(key, "authority_id", authority_id)
    object.__setattr__(key, "session_id", session_id)
    object.__setattr__(key, "epoch", epoch)
    return key


def _fast_request_key(value: object) -> RequestKey | None:
    if type(value) is not dict:
        return None
    authority_id = value.get("authority_id")
    session_id = value.get("session_id")
    epoch = value.get("epoch")
    if (
        type(authority_id) is int
        and authority_id >= 0
        and type(session_id) is int
        and session_id >= 0
        and type(epoch) is int
        and epoch >= 0
    ):
        return _interned_request_key(authority_id, session_id, epoch)
    return None


def _fast_release_control(value: object) -> Release | None:
    if type(value) is not dict or value.get("kind") != "release":
        return None
    payload = value.get("value")
    if type(payload) is not dict:
        return None
    request_key = _fast_request_key(payload.get("request_key"))
    op_id = payload.get("op_id")
    if request_key is None or not (type(op_id) is int and op_id >= 0):
        return None
    control = object.__new__(Release)
    object.__setattr__(control, "request_key", request_key)
    object.__setattr__(control, "op_id", op_id)
    return control


@lru_cache(maxsize=1024)
def _interned_point_range(base_point: int, max_points: int) -> PointRange:
    point_range = object.__new__(PointRange)
    object.__setattr__(point_range, "base_point", base_point)
    object.__setattr__(point_range, "max_points", max_points)
    return point_range


def _fast_point_range(value: object) -> PointRange | None:
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


def _fast_version_ref(value: object) -> VersionRef | None:
    if type(value) is not dict:
        return None
    request_key = _fast_request_key(value.get("request_key"))
    if request_key is None:
        return None
    producer_op_id = value.get("producer_op_id")
    if not (type(producer_op_id) is int and producer_op_id >= 0):
        return None
    raw_point = value.get("point")
    if type(raw_point) is not dict:
        return None
    tag = raw_point.get("kind")
    payload = raw_point.get("value")
    if type(tag) is not str or type(payload) is not dict:
        return None
    point: Point
    if tag == "fixed":
        point_index = payload.get("point_index")
        if not (type(point_index) is int and point_index >= 0):
            return None
        point = FixedPoint(point_index)
    elif tag == "device":
        point_index = payload.get("point_index")
        raw_selected_point = payload.get("selected_point")
        selected_point = (
            None if raw_selected_point is None else _fast_product_ref(raw_selected_point)
        )
        if not (type(point_index) is int and point_index >= 0) or (
            raw_selected_point is not None and selected_point is None
        ):
            return None
        point = DevicePoint(point_index, selected_point)
    else:
        return None
    reference = object.__new__(VersionRef)
    object.__setattr__(reference, "request_key", request_key)
    object.__setattr__(reference, "producer_op_id", producer_op_id)
    object.__setattr__(reference, "point", point)
    return reference


def _fast_work(value: object) -> ForwardMode | None:
    if type(value) is ForwardMode:
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
    kind = type(value)
    if kind is not list and kind is not tuple:
        return None
    items = cast(list[object] | tuple[object, ...], value)
    for item in items:
        if not (type(item) is int and item >= 0):
            return None
    return cast(tuple[int, ...], tuple(items))


__all__ = [name for name in globals() if not name.startswith("_")]
