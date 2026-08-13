"""Typed values crossing the scheduler-to-worker execution boundary.

The scheduler and worker exchange four cross-layer records — :class:`Operation`,
:class:`VersionRef`, :class:`ProductRef`, and :class:`CompletionRecord` — plus a
request :class:`Control` command. Every operation names one closed :class:`Work`
variant, one exact parent version, and its declared input and output products.
Two host-computed digests fix identity: :meth:`Operation.compute_plan_digest`
over immutable registration fields, and
:meth:`CompletionRecord.compute_semantic_digest` over the selected result. The
digest byte layout matches the Rust ``worker-wire`` crate exactly so both sides
compute identical digests.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import lru_cache
from typing import Any, TypeAlias, TypeVar, cast

from .foundation.errors import invalid_descriptor
from .foundation.product_transfer import (
    is_transfer_descriptor,
    validate_transfer_descriptor_frame,
)


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


class GenMode(StrEnum):
    TRANSITION = "transition"
    FLOW = "flow"


class WorkVariant(StrEnum):
    TOKEN_EXTEND = "token_extend"
    TOKEN_DECODE = "token_decode"
    TOKEN_VERIFY = "token_verify"
    DRAFT = "draft"
    ENCODE_VISION = "encode_vision"
    ENCODE_LATENT = "encode_latent"
    TRANSFER_PRODUCT = "transfer_product"
    TRANSFER_KV_PUBLISH = "transfer_kv_publish"
    TRANSFER_KV_INSTALL = "transfer_kv_install"
    GEN_TRANSITION = "gen_transition"
    GEN_FLOW = "gen_flow"
    MATERIALIZE = "materialize"


class Domain(StrEnum):
    UND = "und"
    GEN = "gen"


class ExecutionCapability(StrEnum):
    DOMAIN_HOMOGENEOUS = "domain_homogeneous"
    TENSORIZED_MIXED = "tensorized_mixed"


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
    DRAFT = "draft"
    VISION_FEATURE = "vision_feature"
    LATENT_FEATURE = "latent_feature"
    KV = "kv"
    LATENT = "latent"
    ARTIFACT = "artifact"
    COMPLETION = "completion"
    SAMPLING_STATE = "sampling_state"
    FINISH = "finish"
    SELECTED_POINT = "selected_point"
    ACCEPTED_SPAN = "accepted_span"
    CONTINUATION = "continuation"


class StorageClass(StrEnum):
    DEVICE_TENSOR = "device_tensor"
    PAGED_KV = "paged_kv"
    LATENT_ARENA = "latent_arena"
    HOST_STAGING = "host_staging"
    COMPLETION_ARENA = "completion_arena"


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


# Ordered `(kind, mode)` for every closed `Work` leaf; the position is the
# canonical variant index used on the wire and in the plan digest.
_WORK_VARIANTS: tuple[tuple[str, str | None], ...] = (
    ("token", "extend"),
    ("token", "decode"),
    ("token", "verify"),
    ("draft", None),
    ("encode", "vision"),
    ("encode", "latent"),
    ("transfer", "product"),
    ("transfer", "kv_publish"),
    ("transfer", "kv_install"),
    ("gen", "transition"),
    ("gen", "flow"),
    ("materialize", None),
)
_STATE_ADVANCING_WORK = frozenset({0, 1, 2, 9, 10})

# Canonical variant-index tables. Digest byte layouts index enum members by
# declaration order (mirroring the Rust codec); precomputing the tables keeps
# the per-operation digest recomputation off `list(Enum).index` linear scans.
_WORK_VARIANT_MEMBERS: tuple[WorkVariant, ...] = tuple(WorkVariant)
_WORK_PAIR_INDEX: dict[tuple[str, str | None], int] = {
    pair: index for index, pair in enumerate(_WORK_VARIANTS)
}
_DOMAIN_INDEX = {member: index for index, member in enumerate(Domain)}
_PRODUCT_KIND_INDEX = {member: index for index, member in enumerate(ProductKind)}
_STORAGE_CLASS_INDEX = {member: index for index, member in enumerate(StorageClass)}
_DTYPE_INDEX = {member: index for index, member in enumerate(DType)}
_OP_STATUS_INDEX = {member: index for index, member in enumerate(OpStatus)}
_DRAW_LAYOUT_INDEX = {member: index for index, member in enumerate(DrawLayout)}
_DISPOSITION_INDEX = {member: index for index, member in enumerate(Disposition)}
_CLOSE_REASON_INDEX = {member: index for index, member in enumerate(CloseReason)}

# The native worker transport attaches this process-local token only after the
# decoded Rust Batch has passed its complete wire validation. Direct Python
# mappings never carry the token and retain the full decoder validation path.
_WIRE_VALIDATION_TOKEN = object()
_WIRE_VALIDATION_KEY = "_uniserve_wire_validation"

# Precompiled little-endian packers. Multi-field formats fuse the fixed-width
# runs of the record digests into single calls; `<` guarantees no padding, so
# the produced bytes are identical to packing each field separately.
_PACK_B = struct.Struct("<B").pack
_PACK_H = struct.Struct("<H").pack
_PACK_I = struct.Struct("<I").pack
_PACK_Q = struct.Struct("<Q").pack
_PACK_F = struct.Struct("<f").pack
_PACK_II = struct.Struct("<II").pack
_PACK_BI = struct.Struct("<BI").pack
_PACK_QQB = struct.Struct("<QQB").pack
_PACK_QQQ = struct.Struct("<QQQ").pack
_PACK_QQQQB = struct.Struct("<QQQQB").pack
_PACK_IB = struct.Struct("<IB").pack
_PACK_BBB = struct.Struct("<BBB").pack
_PACK_BIBB = struct.Struct("<BIBB").pack
_PACK_IIIII = struct.Struct("<IIIII").pack
_PACK_IIIQQQ = struct.Struct("<IIIQQQ").pack
_PACK_PRODUCT_HEAD = struct.Struct("<QQQQHIBBB").pack


class _Digest:
    """Little-endian, length-prefixed SHA-256 builder mirroring the Rust codec.

    Fields accumulate into one byte buffer hashed once at :meth:`finish`; the
    digest bytes are identical to streaming each field into the hash.
    """

    __slots__ = ("buf",)

    def __init__(self, domain: bytes) -> None:
        self.buf = bytearray(domain)

    def finish(self) -> str:
        return hashlib.sha256(self.buf).hexdigest()

    def u8(self, value: int) -> None:
        self.buf += _PACK_B(value)

    def u16(self, value: int) -> None:
        self.buf += _PACK_H(value)

    def u32(self, value: int) -> None:
        self.buf += _PACK_I(value)

    def u64(self, value: int) -> None:
        self.buf += _PACK_Q(value)

    def f32(self, value: float) -> None:
        self.buf += _PACK_F(value)

    def boolean(self, value: bool) -> None:
        self.buf += _PACK_B(int(value))

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        buf = self.buf
        buf += _PACK_Q(len(encoded))
        buf += encoded

    def u32s(self, values: Sequence[int]) -> None:
        buf = self.buf
        buf += _PACK_Q(len(values))
        for value in values:
            buf += _PACK_I(value)

    def option(self, value: object | None, encode: Any) -> None:
        if value is None:
            self.buf += b"\x00"
        else:
            self.buf += b"\x01"
            encode(value)


# Field-name lists of the protocol records, in declaration order, copied verbatim
# from the Rust `protocol_layout_digest` source. They fix the byte layout of the
# startup agreement digest and must not be reordered or extended here — the goal
# is byte-identical cross-language agreement, not layout completeness.
_LAYOUT_RECORDS: tuple[tuple[str, ...], ...] = (
    (
        "request_key",
        "op_id",
        "parent",
        "work",
        "route",
        "domain",
        "advances_state",
        "bounds",
        "inputs",
        "outputs",
        "kv_capacity_pages",
        "predicate",
        "rng",
        "control_seq",
        "plan_digest",
    ),
    ("request_key", "producer_op_id", "point"),
    (
        "request_key",
        "producer_op_id",
        "output_index",
        "generation",
        "kind",
        "storage_class",
        "dtype",
        "shape_bound",
        "point_range",
    ),
    (
        "request_key",
        "op_id",
        "completion_slot_generation",
        "status",
        "selected_point",
        "logical_lengths",
        "token_span",
        "committed_tokens",
        "finish_flags",
        "product_generations",
        "semantic_digest",
        "error_code",
        "timing_counters",
    ),
    ("version", "digest", "locator"),
    ("prefix_len", "group_id"),
    ("sampling", "negative_token_ids", "finish_token_ids", "kv"),
    ("request_key", "request_pool_idx", "digest", "und", "gen_admission"),
    (
        "request_key",
        "op_id",
        "group_id",
        "block_table",
        "pages_to_zero",
        "prefix_length",
        "input_length",
        "visible_length",
        "resulting_length",
    ),
    (
        "request_key",
        "op_id",
        "branch_index",
        "group_id",
        "block_table",
        "pages_to_zero",
    ),
    ("group_id", "page_ids", "length"),
    ("request_key", "request_pool_idx", "cache_groups"),
    ("group_id", "source_page", "destination_page"),
    (
        "request_key",
        "op_id",
        "page_table",
        "latent_units",
        "height",
        "width",
        "start_step",
        "step_count",
    ),
    (
        "partition_id",
        "submission_group",
        "collective_seq",
        "domain",
        "route",
        "execution",
        "attention",
        "shape_class",
        "operations",
        "request_pool_indices",
        "kv_placements",
        "kv_branch_placements",
        "latent_placements",
    ),
    (
        "block_size",
        "num_blocks",
        "num_layers",
        "num_kv_heads",
        "head_dim",
        "scratch_capacity_tokens",
        "supported_work",
        "latent_page_units",
        "num_latent_pages",
        "latent_width",
        "latent_dtype",
        "latent_downsample",
        "max_vae_grid_tokens",
        "max_vit_grid_tokens",
        "max_latent_feature_bytes",
        "max_vision_feature_bytes",
        "commit_marker_tokens",
        "gen_rope_advance",
        "max_cfg_branches",
        "bytes_per_token",
        "groups",
        "kv_dtype",
        "model_dtype",
        "attention_backend",
        "quantization",
        "rank",
        "pipeline_depth",
        "encoder_cache_budget",
        "supported_controls",
        "max_batch_operations",
        "max_unresolved_window",
        "incremental_kv_publication",
        "tensorized_mixed",
        "sampling_ownership",
        "resource_classes",
        "model_identity",
        "weight_digest",
        "protocol_layout_digest",
    ),
)


def protocol_layout_digest() -> str:
    """The canonical protocol-layout digest over the closed ``Work`` and
    ``Control`` variants and the fixed record field layouts.

    Mirrors the Rust ``worker-wire`` ``protocol_layout_digest`` byte-for-byte so
    a scheduler, worker, and frontend agree at admission.
    """

    digest = _Digest(b"uniserve-protocol-layout\0")
    digest.u64(len(WorkVariant))
    for variant in WorkVariant:
        digest.string(variant.value)
    digest.u64(len(ProductKind))
    for kind in ProductKind:
        digest.string(kind.value)
    for control in ("commit", "close", "release"):
        digest.string(control)
    for record in _LAYOUT_RECORDS:
        digest.u64(len(record))
        for name in record:
            digest.string(name)
    logical_lengths = (
        "token_len",
        "kv_visible_len",
        "latent_len",
        "kv_reserved_len",
        "kv_initialized_len",
        "kv_committed_len",
        "kv_published_len",
    )
    digest.u64(len(logical_lengths))
    for name in logical_lengths:
        digest.string(name)
    return digest.finish()


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
    def from_wire(cls, value: object, where: str = "sampling") -> SamplingParams:
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

    def to_wire(self) -> dict[str, object]:
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
    def from_wire(cls, value: object, where: str = "image") -> ImageParams:
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

    def to_wire(self) -> dict[str, object]:
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
class KvAdmission:
    prefix_len: int = 0
    group_id: int = 0

    def __post_init__(self) -> None:
        _nonnegative(self.prefix_len, "kv.prefix_len")
        _nonnegative(self.group_id, "kv.group_id")

    @classmethod
    def from_wire(cls, value: object, where: str = "kv") -> KvAdmission:
        data = _map(value, where)
        return cls(
            prefix_len=_uint(data.get("prefix_len", 0), f"{where}.prefix_len"),
            group_id=_uint(data.get("group_id", 0), f"{where}.group_id"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "prefix_len": self.prefix_len,
            "group_id": self.group_id,
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
    def from_wire(cls, value: object, where: str = "request_key") -> RequestKey:
        key = _fast_request_key(value)
        if key is not None:
            return key
        data = _map(value, where)
        return cls(
            authority_id=_uint(data.get("authority_id"), f"{where}.authority_id"),
            session_id=_uint(data.get("session_id"), f"{where}.session_id"),
            epoch=_uint(data.get("epoch"), f"{where}.epoch"),
        )

    def to_wire(self) -> dict[str, object]:
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
    def from_wire(cls, value: object, where: str = "shape_bound") -> ShapeBound:
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

    def to_wire(self) -> dict[str, object]:
        return {"dims": [_dim_to_wire(dim) for dim in self.dims]}


def _dim_to_wire(dim: DimBound) -> dict[str, object]:
    if isinstance(dim, StaticDim):
        return {"kind": "static", "value": dim.extent}
    return {"kind": "device", "value": {"max": dim.bound}}


@dataclass(frozen=True, slots=True)
class PointRange:
    base_point: int = 0
    max_points: int = 0

    @classmethod
    def from_wire(cls, value: object, where: str = "point_range") -> PointRange:
        point_range = _fast_point_range(value)
        if point_range is not None:
            return point_range
        data = _map(value, where)
        return cls(
            base_point=_uint(data.get("base_point"), f"{where}.base_point"),
            max_points=_uint(data.get("max_points"), f"{where}.max_points"),
        )

    def to_wire(self) -> dict[str, object]:
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
    def from_wire(cls, value: object, where: str = "product_ref") -> ProductRef:
        reference = _fast_product_ref(value)
        if reference is not None:
            return reference
        data = _map(value, where)
        return cls(
            request_key=RequestKey.from_wire(data.get("request_key"), f"{where}.request_key"),
            producer_op_id=_uint(data.get("producer_op_id"), f"{where}.producer_op_id"),
            output_index=_uint(data.get("output_index"), f"{where}.output_index"),
            generation=_uint(data.get("generation"), f"{where}.generation"),
            kind=_enum(ProductKind, data.get("kind"), f"{where}.kind"),
            storage_class=_enum(StorageClass, data.get("storage_class"), f"{where}.storage_class"),
            dtype=_enum(DType, data.get("dtype"), f"{where}.dtype"),
            shape_bound=ShapeBound.from_wire(data.get("shape_bound"), f"{where}.shape_bound"),
            point_range=PointRange.from_wire(data.get("point_range"), f"{where}.point_range"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "request_key": self.request_key.to_wire(),
            "producer_op_id": self.producer_op_id,
            "output_index": self.output_index,
            "generation": self.generation,
            "kind": self.kind.value,
            "storage_class": self.storage_class.value,
            "dtype": self.dtype.value,
            "shape_bound": self.shape_bound.to_wire(),
            "point_range": self.point_range.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class FixedPoint:
    point_index: int
    semantic_digest: str


@dataclass(frozen=True, slots=True)
class DevicePoint:
    point_index: int
    selected_point: ProductRef | None
    producer_plan_digest: str


Point: TypeAlias = FixedPoint | DevicePoint


@dataclass(frozen=True, slots=True)
class VersionRef:
    request_key: RequestKey
    producer_op_id: int
    point: Point

    def is_fixed(self) -> bool:
        return isinstance(self.point, FixedPoint)

    @classmethod
    def from_wire(cls, value: object, where: str = "version_ref") -> VersionRef:
        reference = _fast_version_ref(value)
        if reference is not None:
            return reference
        data = _map(value, where)
        kind, payload = _tagged(data.get("point"), f"{where}.point")
        if kind == "fixed":
            inner = _map(payload, f"{where}.point.value")
            point: Point = FixedPoint(
                point_index=_uint(inner.get("point_index"), f"{where}.point.value.point_index"),
                semantic_digest=_str(
                    inner.get("semantic_digest"), f"{where}.point.value.semantic_digest"
                ),
            )
        elif kind == "device":
            inner = _map(payload, f"{where}.point.value")
            point = DevicePoint(
                point_index=_uint(inner.get("point_index"), f"{where}.point.value.point_index"),
                selected_point=(
                    None
                    if inner.get("selected_point") is None
                    else ProductRef.from_wire(
                        inner.get("selected_point"), f"{where}.point.value.selected_point"
                    )
                ),
                producer_plan_digest=_str(
                    inner.get("producer_plan_digest"),
                    f"{where}.point.value.producer_plan_digest",
                ),
            )
        else:
            raise invalid_descriptor(f"{where}.point has unknown variant {kind!r}")
        return cls(
            request_key=RequestKey.from_wire(data.get("request_key"), f"{where}.request_key"),
            producer_op_id=_uint(data.get("producer_op_id"), f"{where}.producer_op_id"),
            point=point,
        )

    def to_wire(self) -> dict[str, object]:
        if isinstance(self.point, FixedPoint):
            point = {
                "kind": "fixed",
                "value": {
                    "point_index": self.point.point_index,
                    "semantic_digest": self.point.semantic_digest,
                },
            }
        else:
            point = {
                "kind": "device",
                "value": {
                    "point_index": self.point.point_index,
                    "selected_point": (
                        None
                        if self.point.selected_point is None
                        else self.point.selected_point.to_wire()
                    ),
                    "producer_plan_digest": self.point.producer_plan_digest,
                },
            }
        return {
            "request_key": self.request_key.to_wire(),
            "producer_op_id": self.producer_op_id,
            "point": point,
        }


@dataclass(frozen=True, slots=True)
class SnapshotRef:
    version: VersionRef
    digest: str
    locator: str

    def __post_init__(self) -> None:
        if not isinstance(self.version.point, FixedPoint):
            raise invalid_descriptor("snapshot reference version must be fixed")
        if not _is_digest(self.version.point.semantic_digest):
            raise invalid_descriptor("snapshot reference semantic digest is invalid")
        if not _is_digest(self.digest) or self.locator != self.digest:
            raise invalid_descriptor("snapshot reference artifact digest or locator is invalid")

    def to_wire(self) -> dict[str, object]:
        return {
            "version": self.version.to_wire(),
            "digest": self.digest,
            "locator": self.locator,
        }

    @classmethod
    def from_wire(cls, value: object, where: str = "snapshot") -> SnapshotRef:
        data = _map(value, where)
        return cls(
            version=VersionRef.from_wire(data.get("version"), f"{where}.version"),
            digest=_str(data.get("digest"), f"{where}.digest"),
            locator=_str(data.get("locator"), f"{where}.locator"),
        )


@dataclass(frozen=True, slots=True)
class Work:
    kind: str
    mode: str | None = None

    def __post_init__(self) -> None:
        if (self.kind, self.mode) not in _WORK_PAIR_INDEX:
            raise invalid_descriptor(f"unknown work variant {(self.kind, self.mode)!r}")

    @property
    def variant_index(self) -> int:
        return _WORK_PAIR_INDEX[(self.kind, self.mode)]

    @property
    def variant(self) -> WorkVariant:
        return _WORK_VARIANT_MEMBERS[self.variant_index]

    @property
    def advances_state(self) -> bool:
        return self.variant_index in _STATE_ADVANCING_WORK

    @classmethod
    def token(cls, mode: TokenMode) -> Work:
        return cls("token", mode.value)

    @classmethod
    def from_wire(cls, value: object, where: str = "work") -> Work:
        work = _fast_work(value)
        if work is not None:
            return work
        kind, payload = _tagged(value, where)
        mode = None if payload is None else _str(payload, f"{where}.value")
        return cls(kind, mode)

    def to_wire(self) -> dict[str, object]:
        if self.mode is None:
            return {"kind": self.kind}
        return {"kind": self.kind, "value": self.mode}


@dataclass(frozen=True, slots=True)
class Bounds:
    max_points: int = 0
    max_tokens: int = 0
    max_kv_pages: int = 0
    max_latent_bytes: int = 0
    max_completion_bytes: int = 0
    max_transfer_bytes: int = 0

    @classmethod
    def from_wire(cls, value: object, where: str = "bounds") -> Bounds:
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

    def to_wire(self) -> dict[str, object]:
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
    def from_wire(cls, value: object, where: str = "rng") -> Rng:
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

    def to_wire(self) -> dict[str, object]:
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
    work: Work
    route: int
    domain: Domain
    advances_state: bool
    bounds: Bounds
    inputs: tuple[ProductRef, ...]
    outputs: tuple[ProductRef, ...]
    kv_capacity_pages: int
    predicate: ProductRef | None
    rng: Rng | None
    control_seq: int
    plan_digest: str

    @classmethod
    def registered(
        cls,
        *,
        request_key: RequestKey,
        op_id: int,
        parent: VersionRef,
        work: Work,
        route: int,
        domain: Domain,
        bounds: Bounds,
        inputs: tuple[ProductRef, ...] = (),
        outputs: tuple[ProductRef, ...] = (),
        kv_capacity_pages: int = 0,
        predicate: ProductRef | None = None,
        rng: Rng | None = None,
        control_seq: int = 0,
    ) -> Operation:
        value = cls(
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
            kv_capacity_pages=kv_capacity_pages,
            predicate=predicate,
            rng=rng,
            control_seq=control_seq,
            plan_digest="",
        )
        return replace(value, plan_digest=value.compute_plan_digest())

    def compute_plan_digest(self) -> str:
        digest = _Digest(b"uniserve-operation\0")
        buf = digest.buf
        key = self.request_key
        buf += _PACK_QQQ(key.authority_id, key.session_id, key.epoch)
        buf += _PACK_Q(self.op_id)
        _digest_version_ref(digest, self.parent)
        buf += _PACK_BIBB(
            self.work.variant_index,
            self.route,
            _DOMAIN_INDEX[self.domain],
            int(self.advances_state),
        )
        _digest_bounds(digest, self.bounds)
        buf += _PACK_Q(len(self.inputs))
        for product in self.inputs:
            _digest_product_ref(digest, product)
        buf += _PACK_Q(len(self.outputs))
        for product in self.outputs:
            _digest_product_ref(digest, product)
        digest.u32(self.kv_capacity_pages)
        if self.predicate is None:
            buf += b"\x00"
        else:
            buf += b"\x01"
            _digest_product_ref(digest, self.predicate)
        if self.rng is None:
            buf += b"\x00"
        else:
            buf += b"\x01"
            _digest_rng(digest, self.rng)
        digest.u64(self.control_seq)
        return digest.finish()

    def validate(self) -> None:
        if self.op_id < 1:
            raise invalid_descriptor("operation id must be positive")
        if self.advances_state != self.work.advances_state:
            raise invalid_descriptor(
                "operation declares an advances_state inconsistent with its work variant"
            )
        if self.parent.request_key != self.request_key:
            raise invalid_descriptor("operation parent belongs to another request lineage")
        if self.bounds.max_kv_pages > self.kv_capacity_pages:
            raise invalid_descriptor("operation KV growth bound exceeds its logical capacity")
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
                product.storage_class in (StorageClass.HOST_STAGING, StorageClass.COMPLETION_ARENA)
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
                    or selected.storage_class is not StorageClass.DEVICE_TENSOR
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
                or self.predicate.storage_class is not StorageClass.DEVICE_TENSOR
                or not (self.predicate.kind is ProductKind.COMPLETION or continuation_token)
            ):
                raise invalid_descriptor(
                    "operation predicate is not a generation-tagged device decision product"
                )
        if not _is_digest(self.plan_digest):
            raise invalid_descriptor("operation plan digest is not a lowercase SHA-256 digest")
        if self.plan_digest != self.compute_plan_digest():
            raise invalid_descriptor("operation plan digest does not match its registration fields")

    @classmethod
    def from_wire(
        cls,
        value: object,
        where: str = "operation",
        *,
        _validated_wire: bool = False,
    ) -> Operation:
        # Field decoding follows declaration order with a no-allocation fast
        # path per field. Irregular values use the validating field decoders so
        # diagnostics identify the first invalid declaration.
        data = _map(value, where)
        get = data.get
        request_key = _fast_request_key(get("request_key"))
        if request_key is None:
            request_key = RequestKey.from_wire(get("request_key"), f"{where}.request_key")
        op_id = get("op_id")
        if not (type(op_id) is int and op_id >= 0):
            op_id = _uint(op_id, f"{where}.op_id")
        parent = _fast_version_ref(get("parent"))
        if parent is None:
            parent = VersionRef.from_wire(get("parent"), f"{where}.parent")
        work = _fast_work(get("work"))
        if work is None:
            work = Work.from_wire(get("work"), f"{where}.work")
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
            bounds = Bounds.from_wire(get("bounds"), f"{where}.bounds")
        inputs = _fast_product_refs(get("inputs", ()))
        if inputs is None:
            inputs = tuple(
                ProductRef.from_wire(item, f"{where}.inputs[{index}]")
                for index, item in enumerate(_seq(data.get("inputs", ()), f"{where}.inputs"))
            )
        outputs = _fast_product_refs(get("outputs", ()))
        if outputs is None:
            outputs = tuple(
                ProductRef.from_wire(item, f"{where}.outputs[{index}]")
                for index, item in enumerate(_seq(data.get("outputs", ()), f"{where}.outputs"))
            )
        kv_capacity_pages = get("kv_capacity_pages", 0)
        if not (type(kv_capacity_pages) is int and kv_capacity_pages >= 0):
            kv_capacity_pages = _uint(kv_capacity_pages, f"{where}.kv_capacity_pages")
        predicate_raw = get("predicate")
        if predicate_raw is None:
            predicate = None
        else:
            predicate = _fast_product_ref(predicate_raw)
            if predicate is None:
                predicate = ProductRef.from_wire(predicate_raw, f"{where}.predicate")
        rng_raw = get("rng")
        if rng_raw is None:
            rng = None
        else:
            rng = _fast_rng(rng_raw)
            if rng is None:
                rng = Rng.from_wire(rng_raw, f"{where}.rng")
        control_seq = get("control_seq")
        if not (type(control_seq) is int and control_seq >= 0):
            control_seq = _uint(control_seq, f"{where}.control_seq")
        plan_digest = get("plan_digest")
        if type(plan_digest) is not str:
            plan_digest = _str(plan_digest, f"{where}.plan_digest")
        if _validated_wire:
            operation = object.__new__(cls)
            set_field = object.__setattr__
            set_field(operation, "request_key", request_key)
            set_field(operation, "op_id", op_id)
            set_field(operation, "parent", parent)
            set_field(operation, "work", work)
            set_field(operation, "route", route)
            set_field(operation, "domain", domain)
            set_field(operation, "advances_state", advances_state)
            set_field(operation, "bounds", bounds)
            set_field(operation, "inputs", inputs)
            set_field(operation, "outputs", outputs)
            set_field(operation, "kv_capacity_pages", kv_capacity_pages)
            set_field(operation, "predicate", predicate)
            set_field(operation, "rng", rng)
            set_field(operation, "control_seq", control_seq)
            set_field(operation, "plan_digest", plan_digest)
            return operation
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
            kv_capacity_pages=kv_capacity_pages,
            predicate=predicate,
            rng=rng,
            control_seq=control_seq,
            plan_digest=plan_digest,
        )
        operation.validate()
        return operation

    def to_wire(self) -> dict[str, object]:
        return {
            "request_key": self.request_key.to_wire(),
            "op_id": self.op_id,
            "parent": self.parent.to_wire(),
            "work": self.work.to_wire(),
            "route": self.route,
            "domain": self.domain.value,
            "advances_state": self.advances_state,
            "bounds": self.bounds.to_wire(),
            "inputs": [product.to_wire() for product in self.inputs],
            "outputs": [product.to_wire() for product in self.outputs],
            "kv_capacity_pages": self.kv_capacity_pages,
            "predicate": None if self.predicate is None else self.predicate.to_wire(),
            "rng": None if self.rng is None else self.rng.to_wire(),
            "control_seq": self.control_seq,
            "plan_digest": self.plan_digest,
        }


@dataclass(frozen=True, slots=True)
class LogicalLengths:
    token_len: int = 0
    kv_visible_len: int = 0
    latent_len: int = 0
    kv_reserved_len: int = 0
    kv_initialized_len: int = 0
    kv_committed_len: int = 0
    kv_published_len: int = 0

    @classmethod
    def from_wire(cls, value: object, where: str = "logical_lengths") -> LogicalLengths:
        data = _map(value, where)
        return cls(
            token_len=_uint(data.get("token_len"), f"{where}.token_len"),
            kv_visible_len=_uint(data.get("kv_visible_len"), f"{where}.kv_visible_len"),
            latent_len=_uint(data.get("latent_len"), f"{where}.latent_len"),
            kv_reserved_len=_uint(data.get("kv_reserved_len", 0), f"{where}.kv_reserved_len"),
            kv_initialized_len=_uint(
                data.get("kv_initialized_len", 0), f"{where}.kv_initialized_len"
            ),
            kv_committed_len=_uint(data.get("kv_committed_len", 0), f"{where}.kv_committed_len"),
            kv_published_len=_uint(data.get("kv_published_len", 0), f"{where}.kv_published_len"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "token_len": self.token_len,
            "kv_visible_len": self.kv_visible_len,
            "latent_len": self.latent_len,
            "kv_reserved_len": self.kv_reserved_len,
            "kv_initialized_len": self.kv_initialized_len,
            "kv_committed_len": self.kv_committed_len,
            "kv_published_len": self.kv_published_len,
        }


@dataclass(frozen=True, slots=True)
class TokenSpan:
    base: int = 0
    len: int = 0

    @classmethod
    def from_wire(cls, value: object, where: str = "token_span") -> TokenSpan:
        data = _map(value, where)
        return cls(
            base=_uint(data.get("base"), f"{where}.base"),
            len=_uint(data.get("len"), f"{where}.len"),
        )

    def to_wire(self) -> dict[str, object]:
        return {"base": self.base, "len": self.len}


@dataclass(frozen=True, slots=True)
class FinishFlags:
    eos: bool = False
    length: bool = False
    stop: bool = False

    @classmethod
    def from_wire(cls, value: object, where: str = "finish_flags") -> FinishFlags:
        data = _map(value, where)
        return cls(
            eos=_bool(data.get("eos", False), f"{where}.eos"),
            length=_bool(data.get("length", False), f"{where}.length"),
            stop=_bool(data.get("stop", False), f"{where}.stop"),
        )

    def to_wire(self) -> dict[str, object]:
        return {"eos": self.eos, "length": self.length, "stop": self.stop}


@dataclass(frozen=True, slots=True)
class TimingCounters:
    queued_us: int = 0
    device_us: int = 0
    copy_us: int = 0
    host_us: int = 0

    @classmethod
    def from_wire(cls, value: object, where: str = "timing_counters") -> TimingCounters:
        data = _map(value, where)
        return cls(
            queued_us=_uint(data.get("queued_us", 0), f"{where}.queued_us"),
            device_us=_uint(data.get("device_us", 0), f"{where}.device_us"),
            copy_us=_uint(data.get("copy_us", 0), f"{where}.copy_us"),
            host_us=_uint(data.get("host_us", 0), f"{where}.host_us"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "queued_us": self.queued_us,
            "device_us": self.device_us,
            "copy_us": self.copy_us,
            "host_us": self.host_us,
        }


@dataclass(frozen=True, slots=True)
class CompletionRecord:
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
    semantic_digest: str
    error_code: ErrorCode | None
    timing_counters: TimingCounters

    def compute_semantic_digest(self, parent_semantic: str, plan_digest: str) -> str:
        digest = _Digest(b"uniserve-semantic\0")
        digest.string(parent_semantic)
        digest.string(plan_digest)
        buf = digest.buf
        lengths = self.logical_lengths
        span = self.token_span
        buf += _PACK_IB(self.selected_point, _OP_STATUS_INDEX[self.status])
        buf += _PACK_IIIII(
            lengths.token_len,
            lengths.kv_visible_len,
            lengths.latent_len,
            span.base,
            span.len,
        )
        digest.u32s(self.committed_tokens)
        flags = self.finish_flags
        buf += _PACK_BBB(int(flags.eos), int(flags.length), int(flags.stop))
        digest.u32s(self.product_generations)
        return digest.finish()

    def validate(self) -> None:
        if self.op_id < 1:
            raise invalid_descriptor("completion op id must be positive")
        if self.completion_slot_generation < 1:
            raise invalid_descriptor("completion slot generation must be positive")
        if not _is_digest(self.semantic_digest):
            raise invalid_descriptor("completion semantic digest is not a lowercase SHA-256 digest")
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
    def from_wire(cls, value: object, where: str = "completion") -> CompletionRecord:
        data = _map(value, where)
        record = cls(
            request_key=RequestKey.from_wire(data.get("request_key"), f"{where}.request_key"),
            op_id=_uint(data.get("op_id"), f"{where}.op_id"),
            completion_slot_generation=_uint(
                data.get("completion_slot_generation"), f"{where}.completion_slot_generation"
            ),
            status=_enum(OpStatus, data.get("status"), f"{where}.status"),
            selected_point=_uint(data.get("selected_point"), f"{where}.selected_point"),
            logical_lengths=LogicalLengths.from_wire(
                data.get("logical_lengths"), f"{where}.logical_lengths"
            ),
            token_span=TokenSpan.from_wire(data.get("token_span"), f"{where}.token_span"),
            committed_tokens=_uints(data.get("committed_tokens", ()), f"{where}.committed_tokens"),
            finish_flags=FinishFlags.from_wire(data.get("finish_flags"), f"{where}.finish_flags"),
            product_generations=_uints(
                data.get("product_generations", ()), f"{where}.product_generations"
            ),
            semantic_digest=_str(data.get("semantic_digest"), f"{where}.semantic_digest"),
            error_code=(
                None
                if data.get("error_code") is None
                else _enum(ErrorCode, data["error_code"], f"{where}.error_code")
            ),
            timing_counters=TimingCounters.from_wire(
                data.get("timing_counters"), f"{where}.timing_counters"
            ),
        )
        record.validate()
        return record

    def to_wire(self) -> dict[str, object]:
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
                "latent_len": lengths.latent_len,
                "kv_reserved_len": lengths.kv_reserved_len,
                "kv_initialized_len": lengths.kv_initialized_len,
                "kv_committed_len": lengths.kv_committed_len,
                "kv_published_len": lengths.kv_published_len,
            },
            "token_span": {"base": span.base, "len": span.len},
            "committed_tokens": list(self.committed_tokens),
            "finish_flags": {"eos": flags.eos, "length": flags.length, "stop": flags.stop},
            "product_generations": list(self.product_generations),
            "semantic_digest": self.semantic_digest,
            "error_code": None if error_code is None else error_code.value,
            "timing_counters": {
                "queued_us": timing.queued_us,
                "device_us": timing.device_us,
                "copy_us": timing.copy_us,
                "host_us": timing.host_us,
            },
        }


@dataclass(frozen=True, slots=True)
class Commit:
    request_key: RequestKey
    control_seq: int
    expected_parent: VersionRef
    selected: VersionRef
    public_event_limit: int
    disposition: Disposition
    _content_digest: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class Close:
    request_key: RequestKey
    control_seq: int
    cutoff: VersionRef
    reason: CloseReason
    _content_digest: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class Release:
    request_key: RequestKey
    op_id: int
    _content_digest: str | None = field(default=None, compare=False, repr=False)


Control: TypeAlias = Commit | Close | Release


def _control_variant_index(control: Control) -> int:
    if isinstance(control, Commit):
        return 0
    if isinstance(control, Close):
        return 1
    return 2


def control_content_digest(control: Control) -> str:
    if control._content_digest is not None:
        return control._content_digest
    digest = _Digest(b"uniserve-control\0")
    digest.u8(_control_variant_index(control))
    _digest_request_key(digest, control.request_key)
    if isinstance(control, Commit):
        digest.u64(control.control_seq)
        _digest_version_ref(digest, control.expected_parent)
        _digest_version_ref(digest, control.selected)
        digest.u64(control.public_event_limit)
        digest.u8(_DISPOSITION_INDEX[control.disposition])
    elif isinstance(control, Close):
        digest.u64(control.control_seq)
        _digest_version_ref(digest, control.cutoff)
        digest.u8(_CLOSE_REASON_INDEX[control.reason])
    else:
        digest.u64(control.op_id)
    return digest.finish()


def control_from_wire(
    value: object,
    where: str = "control",
    *,
    _validated_wire: bool = False,
) -> Control:
    kind, payload = _tagged(value, where)
    envelope = _map(value, where)
    data = _map(payload, f"{where}.value")
    request_key = _fast_request_key(data.get("request_key"))
    if request_key is None:
        request_key = RequestKey.from_wire(data.get("request_key"), f"{where}.value.request_key")
    cached_digest = envelope.get("_content_digest") if _validated_wire else None
    if cached_digest is not None and not _is_digest(cached_digest):
        raise invalid_descriptor(f"{where} has an invalid trusted content digest")
    if kind == "commit":
        expected_parent = _fast_version_ref(data.get("expected_parent"))
        if expected_parent is None:
            expected_parent = VersionRef.from_wire(
                data.get("expected_parent"), f"{where}.value.expected_parent"
            )
        selected = _fast_version_ref(data.get("selected"))
        if selected is None:
            selected = VersionRef.from_wire(data.get("selected"), f"{where}.value.selected")
        commit = Commit(
            request_key=request_key,
            control_seq=_uint(data.get("control_seq"), f"{where}.value.control_seq"),
            expected_parent=expected_parent,
            selected=selected,
            public_event_limit=_uint(
                data.get("public_event_limit"), f"{where}.value.public_event_limit"
            ),
            disposition=_enum(Disposition, data.get("disposition"), f"{where}.value.disposition"),
            _content_digest=cast(str | None, cached_digest),
        )
        if not commit.selected.is_fixed():
            raise invalid_descriptor("a commit control must select a fixed version")
        control: Control = commit
    elif kind == "close":
        cutoff = _fast_version_ref(data.get("cutoff"))
        if cutoff is None:
            cutoff = VersionRef.from_wire(data.get("cutoff"), f"{where}.value.cutoff")
        control = Close(
            request_key=request_key,
            control_seq=_uint(data.get("control_seq"), f"{where}.value.control_seq"),
            cutoff=cutoff,
            reason=_enum(CloseReason, data.get("reason"), f"{where}.value.reason"),
            _content_digest=cast(str | None, cached_digest),
        )
        if not control.cutoff.is_fixed():
            raise invalid_descriptor("a close control must name a fixed cutoff version")
    elif kind == "release":
        control = Release(
            request_key=request_key,
            op_id=_uint(data.get("op_id"), f"{where}.value.op_id"),
            _content_digest=cast(str | None, cached_digest),
        )
    else:
        raise invalid_descriptor(f"{where} has unknown variant {kind!r}")
    return control


def control_to_wire(control: Control) -> dict[str, object]:
    if isinstance(control, Commit):
        return {
            "kind": "commit",
            "value": {
                "request_key": control.request_key.to_wire(),
                "control_seq": control.control_seq,
                "expected_parent": control.expected_parent.to_wire(),
                "selected": control.selected.to_wire(),
                "public_event_limit": control.public_event_limit,
                "disposition": control.disposition.value,
            },
        }
    if isinstance(control, Close):
        return {
            "kind": "close",
            "value": {
                "request_key": control.request_key.to_wire(),
                "control_seq": control.control_seq,
                "cutoff": control.cutoff.to_wire(),
                "reason": control.reason.value,
            },
        }
    return {
        "kind": "release",
        "value": {"request_key": control.request_key.to_wire(), "op_id": control.op_id},
    }


@dataclass(frozen=True, slots=True)
class UndAdmission:
    sampling: SamplingParams = field(default_factory=SamplingParams)
    negative_token_ids: tuple[int, ...] = ()
    finish_token_ids: tuple[int, ...] = ()
    kv: KvAdmission = field(default_factory=KvAdmission)

    def __post_init__(self) -> None:
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
    def from_wire(cls, value: object, where: str = "und admission") -> UndAdmission:
        data = _map(value, where)
        return cls(
            sampling=SamplingParams.from_wire(data.get("sampling", {}), f"{where}.sampling"),
            negative_token_ids=_uints(
                data.get("negative_token_ids", ()), f"{where}.negative_token_ids"
            ),
            finish_token_ids=_uints(data.get("finish_token_ids", ()), f"{where}.finish_token_ids"),
            kv=KvAdmission.from_wire(data.get("kv", {}), f"{where}.kv"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "sampling": self.sampling.to_wire(),
            "negative_token_ids": list(self.negative_token_ids),
            "finish_token_ids": list(self.finish_token_ids),
            "kv": self.kv.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class GenAdmission:
    image: ImageParams = field(default_factory=ImageParams)

    @classmethod
    def from_wire(cls, value: object, where: str = "gen admission") -> GenAdmission:
        data = _map(value, where)
        return cls(image=ImageParams.from_wire(data.get("image", {}), f"{where}.image"))

    def to_wire(self) -> dict[str, object]:
        return {"image": self.image.to_wire()}


@dataclass(frozen=True, slots=True)
class Admission:
    request_key: RequestKey
    request_pool_idx: int
    digest: str
    und: UndAdmission | None
    gen_admission: GenAdmission | None

    def __post_init__(self) -> None:
        if self.request_pool_idx < 1:
            raise invalid_descriptor("request-pool index must be positive")
        if self.und is None and self.gen_admission is None:
            raise invalid_descriptor("admission must declare an understanding or generation branch")

    @classmethod
    def create(
        cls,
        request_key: RequestKey,
        *,
        request_pool_idx: int,
        und: UndAdmission | None = None,
        gen_admission: GenAdmission | None = None,
    ) -> Admission:
        value = cls(request_key, request_pool_idx, "", und, gen_admission)
        return replace(value, digest=value.payload_digest())

    @classmethod
    def from_wire(cls, value: object, where: str = "admission") -> Admission:
        data = _map(value, where)
        admission = cls(
            request_key=RequestKey.from_wire(data.get("request_key"), f"{where}.request_key"),
            request_pool_idx=_uint(data.get("request_pool_idx"), f"{where}.request_pool_idx"),
            digest=_str(data.get("digest"), f"{where}.digest"),
            und=(
                None
                if data.get("und") is None
                else UndAdmission.from_wire(data["und"], f"{where}.und")
            ),
            gen_admission=(
                None
                if data.get("gen_admission") is None
                else GenAdmission.from_wire(data["gen_admission"], f"{where}.gen_admission")
            ),
        )
        admission.validate()
        return admission

    def validate(self) -> None:
        if not _is_digest(self.digest):
            raise invalid_descriptor("admission digest must be a lowercase SHA-256 digest")
        if self.digest != self.payload_digest():
            raise invalid_descriptor(
                f"admission digest mismatch for request {self.request_key.session_id}"
            )

    def payload_digest(self) -> str:
        digest = _Digest(b"uniserve-admission\0")
        _digest_request_key(digest, self.request_key)
        digest.option(self.und, lambda value: _digest_und_admission(digest, value))
        digest.option(self.gen_admission, lambda value: _digest_image(digest, value.image))
        return digest.finish()

    def to_wire(self) -> dict[str, object]:
        return {
            "request_key": self.request_key.to_wire(),
            "request_pool_idx": self.request_pool_idx,
            "digest": self.digest,
            "und": None if self.und is None else self.und.to_wire(),
            "gen_admission": None if self.gen_admission is None else self.gen_admission.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class KvPlacement:
    request_key: RequestKey
    op_id: int
    group_id: int
    block_table: tuple[int, ...]
    pages_to_zero: tuple[int, ...]
    prefix_length: int
    input_length: int
    visible_length: int
    resulting_length: int

    def __post_init__(self) -> None:
        if self.op_id < 1:
            raise invalid_descriptor("KV placement operation id must be positive")
        if self.group_id < 0 or any(value < 1 for value in self.block_table):
            raise invalid_descriptor("KV placement contains an invalid group or reserved page zero")
        if any(value < 1 for value in self.pages_to_zero):
            raise invalid_descriptor("KV placement carries the reserved page zero")
        if len(set(self.block_table)) != len(self.block_table):
            raise invalid_descriptor("KV placement repeats a page in its block table")
        if len(set(self.pages_to_zero)) != len(self.pages_to_zero):
            raise invalid_descriptor("KV placement repeats a page-to-zero")
        if not set(self.pages_to_zero).issubset(self.block_table):
            raise invalid_descriptor("KV placement zeroes a page outside its block table")
        if (
            self.visible_length < self.prefix_length
            or self.resulting_length < self.visible_length
            or self.prefix_length + self.input_length != self.resulting_length
        ):
            raise invalid_descriptor("KV placement lengths are inconsistent")

    @classmethod
    def from_wire(
        cls,
        value: object,
        where: str = "KV placement",
        *,
        _validated_wire: bool = False,
    ) -> KvPlacement:
        data = _map(value, where)
        request_key = _fast_request_key(data.get("request_key"))
        if request_key is None:
            request_key = RequestKey.from_wire(data.get("request_key"), f"{where}.request_key")

        def uint_field(name: str) -> int:
            raw = data.get(name)
            return raw if type(raw) is int and raw >= 0 else _uint(raw, f"{where}.{name}")

        block_table = _fast_uints(data.get("block_table", ()))
        if block_table is None:
            block_table = _uints(data.get("block_table", ()), f"{where}.block_table")
        pages_to_zero = _fast_uints(data.get("pages_to_zero", ()))
        if pages_to_zero is None:
            pages_to_zero = _uints(data.get("pages_to_zero", ()), f"{where}.pages_to_zero")
        fields = (
            request_key,
            uint_field("op_id"),
            uint_field("group_id"),
            block_table,
            pages_to_zero,
            uint_field("prefix_length"),
            uint_field("input_length"),
            uint_field("visible_length"),
            uint_field("resulting_length"),
        )
        if _validated_wire:
            placement = object.__new__(cls)
            for name, item in zip(cls.__slots__, fields, strict=True):
                object.__setattr__(placement, name, item)
            return placement
        return cls(*fields)

    def to_wire(self) -> dict[str, object]:
        return {
            "request_key": self.request_key.to_wire(),
            "op_id": self.op_id,
            "group_id": self.group_id,
            "block_table": list(self.block_table),
            "pages_to_zero": list(self.pages_to_zero),
            "prefix_length": self.prefix_length,
            "input_length": self.input_length,
            "visible_length": self.visible_length,
            "resulting_length": self.resulting_length,
        }


@dataclass(frozen=True, slots=True)
class KvBranchPlacement:
    request_key: RequestKey
    op_id: int
    branch_index: int
    group_id: int
    block_table: tuple[int, ...]
    pages_to_zero: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.op_id < 1 or self.branch_index < 1 or self.group_id < 0:
            raise invalid_descriptor("KV branch placement identity is invalid")
        if (
            not self.block_table
            or any(page < 1 for page in self.block_table)
            or len(set(self.block_table)) != len(self.block_table)
        ):
            raise invalid_descriptor(
                "KV branch placement is empty, repeats a page, or carries page zero"
            )
        if (
            any(page < 1 for page in self.pages_to_zero)
            or len(set(self.pages_to_zero)) != len(self.pages_to_zero)
            or not set(self.pages_to_zero).issubset(self.block_table)
        ):
            raise invalid_descriptor("KV branch placement zero set is invalid")

    @classmethod
    def from_wire(
        cls,
        value: object,
        where: str = "KV branch placement",
        *,
        _validated_wire: bool = False,
    ) -> KvBranchPlacement:
        data = _map(value, where)
        request_key = _fast_request_key(data.get("request_key"))
        if request_key is None:
            request_key = RequestKey.from_wire(data.get("request_key"), f"{where}.request_key")

        def uint_field(name: str) -> int:
            raw = data.get(name)
            return raw if type(raw) is int and raw >= 0 else _uint(raw, f"{where}.{name}")

        block_table = _fast_uints(data.get("block_table", ()))
        if block_table is None:
            block_table = _uints(data.get("block_table", ()), f"{where}.block_table")
        pages_to_zero = _fast_uints(data.get("pages_to_zero", ()))
        if pages_to_zero is None:
            pages_to_zero = _uints(data.get("pages_to_zero", ()), f"{where}.pages_to_zero")
        fields = (
            request_key,
            uint_field("op_id"),
            uint_field("branch_index"),
            uint_field("group_id"),
            block_table,
            pages_to_zero,
        )
        if _validated_wire:
            placement = object.__new__(cls)
            for name, item in zip(cls.__slots__, fields, strict=True):
                object.__setattr__(placement, name, item)
            return placement
        return cls(*fields)

    def to_wire(self) -> dict[str, object]:
        return {
            "request_key": self.request_key.to_wire(),
            "op_id": self.op_id,
            "branch_index": self.branch_index,
            "group_id": self.group_id,
            "block_table": list(self.block_table),
            "pages_to_zero": list(self.pages_to_zero),
        }


@dataclass(frozen=True, slots=True)
class CacheGroupPlacement:
    group_id: int
    page_ids: tuple[int, ...]
    length: int

    def __post_init__(self) -> None:
        if (
            self.group_id < 0
            or self.length < 0
            or any(page < 1 for page in self.page_ids)
            or len(set(self.page_ids)) != len(self.page_ids)
        ):
            raise invalid_descriptor("cache recovery placement is invalid")

    @classmethod
    def from_wire(
        cls,
        value: object,
        where: str = "cache recovery placement",
    ) -> CacheGroupPlacement:
        data = _map(value, where)
        return cls(
            group_id=_uint(data.get("group_id"), f"{where}.group_id"),
            page_ids=_uints(data.get("page_ids", ()), f"{where}.page_ids"),
            length=_uint(data.get("length"), f"{where}.length"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "page_ids": list(self.page_ids),
            "length": self.length,
        }


@dataclass(frozen=True, slots=True)
class RecoveryPlacement:
    request_key: RequestKey
    request_pool_idx: int
    cache_groups: tuple[CacheGroupPlacement, ...]

    def __post_init__(self) -> None:
        groups = tuple(group.group_id for group in self.cache_groups)
        if self.request_pool_idx < 1 or len(set(groups)) != len(groups):
            raise invalid_descriptor("recovery placement identity is invalid")

    @classmethod
    def from_wire(
        cls,
        value: object,
        where: str = "recovery placement",
    ) -> RecoveryPlacement:
        data = _map(value, where)
        return cls(
            request_key=RequestKey.from_wire(data.get("request_key"), f"{where}.request_key"),
            request_pool_idx=_uint(
                data.get("request_pool_idx"),
                f"{where}.request_pool_idx",
            ),
            cache_groups=tuple(
                CacheGroupPlacement.from_wire(item, f"{where}.cache_groups[{index}]")
                for index, item in enumerate(
                    _seq(data.get("cache_groups", ()), f"{where}.cache_groups")
                )
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "request_key": self.request_key.to_wire(),
            "request_pool_idx": self.request_pool_idx,
            "cache_groups": [group.to_wire() for group in self.cache_groups],
        }


@dataclass(frozen=True, slots=True)
class CacheCopy:
    group_id: int
    source_page: int
    destination_page: int

    def __post_init__(self) -> None:
        if self.group_id < 0 or self.source_page < 1 or self.destination_page < 1:
            raise invalid_descriptor("cache copy identity is invalid")

    @classmethod
    def from_wire(cls, value: object, where: str = "cache copy") -> CacheCopy:
        data = _map(value, where)
        return cls(
            group_id=_uint(data.get("group_id"), f"{where}.group_id"),
            source_page=_uint(data.get("source_page"), f"{where}.source_page"),
            destination_page=_uint(
                data.get("destination_page"),
                f"{where}.destination_page",
            ),
        )

    def to_wire(self) -> dict[str, int]:
        return {
            "group_id": self.group_id,
            "source_page": self.source_page,
            "destination_page": self.destination_page,
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
    def from_wire(cls, value: object, where: str = "latent placement") -> LatentPlacement:
        data = _map(value, where)
        return cls(
            request_key=RequestKey.from_wire(data.get("request_key"), f"{where}.request_key"),
            op_id=_uint(data.get("op_id"), f"{where}.op_id"),
            page_table=_uints(data.get("page_table", ()), f"{where}.page_table"),
            latent_units=_uint(data.get("latent_units"), f"{where}.latent_units"),
            height=_uint(data.get("height"), f"{where}.height"),
            width=_uint(data.get("width"), f"{where}.width"),
            start_step=_uint(data.get("start_step"), f"{where}.start_step"),
            step_count=_uint(data.get("step_count"), f"{where}.step_count"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "request_key": self.request_key.to_wire(),
            "op_id": self.op_id,
            "page_table": list(self.page_table),
            "latent_units": self.latent_units,
            "height": self.height,
            "width": self.width,
            "start_step": self.start_step,
            "step_count": self.step_count,
        }


@dataclass(frozen=True, slots=True)
class BatchPartition:
    partition_id: int
    submission_group: int
    collective_seq: int
    domain: Domain
    route: int
    execution: ExecutionCapability
    attention: AttentionRegime
    shape_class: int
    operations: tuple[Operation, ...]
    request_pool_indices: tuple[int, ...]
    kv_placements: tuple[KvPlacement, ...] = ()
    kv_branch_placements: tuple[KvBranchPlacement, ...] = ()
    latent_placements: tuple[LatentPlacement, ...] = ()

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
        if len(self.request_pool_indices) != len(self.operations):
            raise invalid_descriptor(
                "batch partition request-pool indices are not aligned with operations"
            )
        if any(index < 1 for index in self.request_pool_indices):
            raise invalid_descriptor("batch partition carries request-pool index zero")
        if any(
            operation.domain is not self.domain or operation.route != self.route
            for operation in self.operations
        ):
            raise invalid_descriptor("batch partition operation disagrees with its domain or route")
        operations = {
            (operation.request_key, operation.op_id): operation for operation in self.operations
        }
        placements: set[tuple[RequestKey, int, int]] = set()
        for kv_placement in self.kv_placements:
            kv_identity = (
                kv_placement.request_key,
                kv_placement.op_id,
                kv_placement.group_id,
            )
            if kv_identity in placements:
                raise invalid_descriptor("batch partition repeats a KV placement identity")
            placements.add(kv_identity)
            operation = operations.get(kv_identity[:2])
            if operation is None:
                raise invalid_descriptor("KV placement does not name a partition operation")
            if len(kv_placement.block_table) != operation.kv_capacity_pages:
                raise invalid_descriptor("KV placement does not establish operation capacity")
        if any(
            operation.kv_capacity_pages > 0
            and not any(
                request_key == operation.request_key and op_id == operation.op_id
                for request_key, op_id, _group_id in placements
            )
            for operation in self.operations
        ):
            raise invalid_descriptor("operation with logical KV capacity has no placement")
        branch_ids: set[tuple[RequestKey, int, int, int]] = set()
        branch_pages: set[int] = set()
        for placement in self.kv_branch_placements:
            identity = (
                placement.request_key,
                placement.op_id,
                placement.branch_index,
                placement.group_id,
            )
            if identity in branch_ids:
                raise invalid_descriptor("batch partition repeats a KV branch placement identity")
            branch_ids.add(identity)
            operation = operations.get(identity[:2])
            if operation is None or operation.work.variant is not WorkVariant.GEN_FLOW:
                raise invalid_descriptor("KV branch placement does not name generation flow")
            if not branch_pages.isdisjoint(placement.block_table):
                raise invalid_descriptor("KV branch placements overlap physical pages")
            branch_pages.update(placement.block_table)
        latent_ids: set[tuple[RequestKey, int]] = set()
        latent_variants = {
            WorkVariant.GEN_TRANSITION,
            WorkVariant.GEN_FLOW,
            WorkVariant.MATERIALIZE,
        }
        for latent_placement in self.latent_placements:
            latent_identity = (latent_placement.request_key, latent_placement.op_id)
            if latent_identity in latent_ids:
                raise invalid_descriptor("batch partition repeats a latent placement identity")
            latent_ids.add(latent_identity)
            operation = operations.get(latent_identity)
            if operation is None:
                raise invalid_descriptor("latent placement does not name a partition operation")
            if operation.work.variant not in latent_variants:
                raise invalid_descriptor(
                    "latent placement names an operation that does not address a trajectory"
                )
        if any(
            operation.work.variant in latent_variants
            and (operation.request_key, operation.op_id) not in latent_ids
            for operation in self.operations
        ):
            raise invalid_descriptor(
                "operation that addresses a trajectory has no latent placement"
            )

    @classmethod
    def from_wire(
        cls,
        value: object,
        where: str = "batch partition",
        *,
        _validated_wire: bool = False,
    ) -> BatchPartition:
        data = _map(value, where)
        fields = dict(
            partition_id=_uint(data.get("partition_id"), f"{where}.partition_id"),
            submission_group=_uint(data.get("submission_group"), f"{where}.submission_group"),
            collective_seq=_uint(data.get("collective_seq"), f"{where}.collective_seq"),
            domain=Domain(_str(data.get("domain"), f"{where}.domain")),
            route=_uint(data.get("route"), f"{where}.route"),
            execution=ExecutionCapability(_str(data.get("execution"), f"{where}.execution")),
            attention=AttentionRegime(_str(data.get("attention"), f"{where}.attention")),
            shape_class=_uint(data.get("shape_class"), f"{where}.shape_class"),
            operations=tuple(
                Operation.from_wire(
                    item,
                    f"{where}.operations[{index}]",
                    _validated_wire=_validated_wire,
                )
                for index, item in enumerate(
                    _seq(data.get("operations", ()), f"{where}.operations")
                )
            ),
            request_pool_indices=_uints(
                data.get("request_pool_indices", ()), f"{where}.request_pool_indices"
            ),
            kv_placements=tuple(
                KvPlacement.from_wire(
                    item,
                    f"{where}.kv_placements[{index}]",
                    _validated_wire=_validated_wire,
                )
                for index, item in enumerate(
                    _seq(data.get("kv_placements", ()), f"{where}.kv_placements")
                )
            ),
            kv_branch_placements=tuple(
                KvBranchPlacement.from_wire(
                    item,
                    f"{where}.kv_branch_placements[{index}]",
                    _validated_wire=_validated_wire,
                )
                for index, item in enumerate(
                    _seq(
                        data.get("kv_branch_placements", ()),
                        f"{where}.kv_branch_placements",
                    )
                )
            ),
            latent_placements=tuple(
                LatentPlacement.from_wire(item, f"{where}.latent_placements[{index}]")
                for index, item in enumerate(
                    _seq(data.get("latent_placements", ()), f"{where}.latent_placements")
                )
            ),
        )
        if _validated_wire:
            partition = object.__new__(cls)
            for name, item in fields.items():
                object.__setattr__(partition, name, item)
            return partition
        return cls(**fields)

    def to_wire(self) -> dict[str, object]:
        return {
            "partition_id": self.partition_id,
            "submission_group": self.submission_group,
            "collective_seq": self.collective_seq,
            "domain": self.domain.value,
            "route": self.route,
            "execution": self.execution.value,
            "attention": self.attention.value,
            "shape_class": self.shape_class,
            "operations": [operation.to_wire() for operation in self.operations],
            "request_pool_indices": list(self.request_pool_indices),
            "kv_placements": [placement.to_wire() for placement in self.kv_placements],
            "kv_branch_placements": [
                placement.to_wire() for placement in self.kv_branch_placements
            ],
            "latent_placements": [placement.to_wire() for placement in self.latent_placements],
        }


@dataclass(frozen=True, slots=True)
class Batch:
    step_id: int
    admissions: tuple[Admission, ...] = ()
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
            execution = partitions[0].execution
            collective_seq = partitions[0].collective_seq
            attention = partitions[0].attention
            shape_class = partitions[0].shape_class
            if any(
                partition.execution is not execution
                or partition.collective_seq != collective_seq
                or partition.attention is not attention
                or partition.shape_class != shape_class
                for partition in partitions
            ):
                raise invalid_descriptor(
                    "physical submission partitions disagree on execution, attention, shape, or collective order"
                )
            if execution is ExecutionCapability.DOMAIN_HOMOGENEOUS:
                if len(partitions) != 1:
                    raise invalid_descriptor(
                        "a domain-homogeneous submission group must contain one partition"
                    )
            else:
                domains = {partition.domain for partition in partitions}
                routes = {partition.route for partition in partitions}
                if len(partitions) < 2 or len(domains) != len(partitions):
                    raise invalid_descriptor(
                        "a tensorized-mixed submission group must contain distinct domains"
                    )
                if len(routes) != 1:
                    raise invalid_descriptor(
                        "a tensorized-mixed submission group spans route capabilities"
                    )
        request_keys = [operation.request_key for operation in self.operations]
        if len(set(request_keys)) != len(request_keys):
            raise invalid_descriptor(
                "a submission batch carries multiple operations for one request"
            )
        request_slots = {
            operation.request_key: request_pool_idx
            for partition in self.partitions
            for operation, request_pool_idx in zip(
                partition.operations,
                partition.request_pool_indices,
                strict=True,
            )
        }
        if len(set(request_slots.values())) != len(request_slots):
            raise invalid_descriptor(
                "a submission batch assigns one request-pool index to multiple requests"
            )
        admitted = [admission.request_key for admission in self.admissions]
        if len(set(admitted)) != len(admitted):
            raise invalid_descriptor("a submission batch carries a duplicate admission")
        for admission in self.admissions:
            admission.validate()
            if admission.request_key not in request_keys:
                raise invalid_descriptor("a submission batch admits a request without an operation")
            if request_slots[admission.request_key] != admission.request_pool_idx:
                raise invalid_descriptor(
                    "an admission disagrees with its operation request-pool index"
                )
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
        declared_inputs = {product for operation in self.operations for product in operation.inputs}
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
    def from_wire(cls, value: object) -> Batch:
        data = _map(value, "execute batch")
        validated_wire = data.get(_WIRE_VALIDATION_KEY) is _WIRE_VALIDATION_TOKEN
        step_id = _uint(data.get("step_id"), "execute batch.step_id")
        admissions = tuple(
            Admission.from_wire(item, f"execute batch.admissions[{index}]")
            for index, item in enumerate(
                _seq(data.get("admissions", ()), "execute batch.admissions")
            )
        )
        partitions = tuple(
            BatchPartition.from_wire(
                item,
                f"execute batch.partitions[{index}]",
                _validated_wire=validated_wire,
            )
            for index, item in enumerate(
                _seq(data.get("partitions", ()), "execute batch.partitions")
            )
        )
        controls = tuple(
            _fast_release_control(item)
            or control_from_wire(
                item,
                f"execute batch.controls[{index}]",
                _validated_wire=validated_wire,
            )
            for index, item in enumerate(_seq(data.get("controls", ()), "execute batch.controls"))
        )
        input_products = tuple(
            ProductPayload.from_wire(item, f"execute batch.input_products[{index}]")
            for index, item in enumerate(
                _seq(data.get("input_products", ()), "execute batch.input_products")
            )
        )
        if validated_wire:
            batch = object.__new__(cls)
            set_field = object.__setattr__
            set_field(batch, "step_id", step_id)
            set_field(batch, "admissions", admissions)
            set_field(batch, "partitions", partitions)
            set_field(batch, "controls", controls)
            set_field(batch, "input_products", input_products)
            return batch
        return cls(
            step_id=step_id,
            admissions=admissions,
            partitions=partitions,
            controls=controls,
            input_products=input_products,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "admissions": [value.to_wire() for value in self.admissions],
            "partitions": [value.to_wire() for value in self.partitions],
            "controls": [control_to_wire(value) for value in self.controls],
            "input_products": [value.to_wire() for value in self.input_products],
        }


@dataclass(frozen=True, slots=True)
class RegistrationAck:
    visible: bool = False

    @classmethod
    def from_wire(cls, value: object, where: str = "registration") -> RegistrationAck:
        data = _map(value, where)
        return cls(visible=_bool(data.get("visible", False), f"{where}.visible"))

    def to_wire(self) -> dict[str, object]:
        return {"visible": self.visible}


@dataclass(frozen=True, slots=True)
class ProductPayload:
    product: ProductRef
    payload: bytes

    @classmethod
    def from_wire(cls, value: object, where: str = "product payload") -> ProductPayload:
        data = _map(value, where)
        return cls(
            product=ProductRef.from_wire(data.get("product"), f"{where}.product"),
            payload=(
                raw
                if type(raw := data.get("bytes", b"")) is bytes
                else bytes(raw)
                if isinstance(raw, (bytearray, memoryview))
                else bytes(_uints(raw, f"{where}.bytes"))
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {"product": self.product.to_wire(), "bytes": self.payload}


def encode_token_product_bytes(tokens: Sequence[int]) -> bytes:
    """Encode a ``ProductKind.TOKEN`` product value.

    The layout is a little-endian ``u32`` count followed by that many
    little-endian ``u32`` token ids, matching the Rust ``worker-wire`` codec so
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
    force_finish: bool = False


def encode_sampling_state_bytes(state: SamplingState) -> bytes:
    allowed = (
        None
        if state.allowed_token_ids is None
        else tuple(sorted(set(int(token) for token in state.allowed_token_ids)))
    )
    suppressed = tuple(sorted(set(int(token) for token in state.suppressed_token_ids)))
    finish = tuple(sorted(set(int(token) for token in state.finish_token_ids)))
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
    if offset >= len(data):
        raise invalid_descriptor("sampling-state bytes omit force-finish")
    force_finish = data[offset]
    offset += 1
    if force_finish not in (0, 1):
        raise invalid_descriptor(f"sampling-state force-finish {force_finish} is invalid")
    if offset != len(data):
        raise invalid_descriptor("sampling-state bytes contain trailing data")
    return SamplingState(allowed, suppressed, finish, bool(force_finish))


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
    def from_wire(cls, value: object, where: str = "worker forward stats") -> WorkerForwardStats:
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

    def to_wire(self) -> dict[str, object]:
        return {
            name: dict(value) if isinstance(value, Mapping) else value
            for name, value in ((name, getattr(self, name)) for name in self.__dataclass_fields__)
        }


@dataclass(frozen=True, slots=True)
class PartitionCompletion:
    partition_id: int
    completions: tuple[CompletionRecord, ...]
    products: tuple[ProductPayload, ...] = ()
    registration: RegistrationAck = field(default_factory=RegistrationAck)
    worker_exec_us: int | None = None
    forward_stats: WorkerForwardStats | None = None

    @classmethod
    def from_wire(
        cls,
        value: object,
        where: str = "partition completion",
    ) -> PartitionCompletion:
        data = _map(value, where)
        return cls(
            partition_id=_uint(data.get("partition_id"), f"{where}.partition_id"),
            completions=tuple(
                CompletionRecord.from_wire(item, f"{where}.completions[{index}]")
                for index, item in enumerate(
                    _seq(data.get("completions", ()), f"{where}.completions")
                )
            ),
            products=tuple(
                ProductPayload.from_wire(item, f"{where}.products[{index}]")
                for index, item in enumerate(_seq(data.get("products", ()), f"{where}.products"))
            ),
            registration=RegistrationAck.from_wire(
                data.get("registration", {}), f"{where}.registration"
            ),
            worker_exec_us=_optional_uint(data.get("worker_exec_us"), f"{where}.worker_exec_us"),
            forward_stats=(
                None
                if data.get("forward_stats") is None
                else WorkerForwardStats.from_wire(data["forward_stats"], f"{where}.forward_stats")
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "partition_id": self.partition_id,
            "completions": [value.to_wire() for value in self.completions],
            "products": [value.to_wire() for value in self.products],
            "registration": self.registration.to_wire(),
            "worker_exec_us": self.worker_exec_us,
            "forward_stats": None if self.forward_stats is None else self.forward_stats.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class CompletionReport:
    step_id: int
    partitions: tuple[PartitionCompletion, ...]

    @property
    def completions(self) -> tuple[CompletionRecord, ...]:
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
    def from_wire(cls, value: object, where: str = "completion report") -> CompletionReport:
        data = _map(value, where)
        return cls(
            step_id=_uint(data.get("step_id"), f"{where}.step_id"),
            partitions=tuple(
                PartitionCompletion.from_wire(item, f"{where}.partitions[{index}]")
                for index, item in enumerate(
                    _seq(data.get("partitions", ()), f"{where}.partitions")
                )
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "partitions": [partition.to_wire() for partition in self.partitions],
        }


# ---------------------------------------------------------------------------
# Digest helpers mirroring the Rust `CanonicalDigest`
# ---------------------------------------------------------------------------


def _digest_request_key(digest: _Digest, value: RequestKey) -> None:
    digest.buf += _PACK_QQQ(value.authority_id, value.session_id, value.epoch)


def _digest_shape_bound(digest: _Digest, value: ShapeBound) -> None:
    buf = digest.buf
    dims = value.dims
    buf += _PACK_Q(len(dims))
    for dim in dims:
        if isinstance(dim, StaticDim):
            buf += _PACK_BI(0, dim.extent)
        else:
            buf += _PACK_BI(1, dim.bound)


def _digest_product_ref(digest: _Digest, value: ProductRef) -> None:
    key = value.request_key
    digest.buf += _PACK_PRODUCT_HEAD(
        key.authority_id,
        key.session_id,
        key.epoch,
        value.producer_op_id,
        value.output_index,
        value.generation,
        _PRODUCT_KIND_INDEX[value.kind],
        _STORAGE_CLASS_INDEX[value.storage_class],
        _DTYPE_INDEX[value.dtype],
    )
    _digest_shape_bound(digest, value.shape_bound)
    point_range = value.point_range
    digest.buf += _PACK_II(point_range.base_point, point_range.max_points)


def _digest_version_ref(digest: _Digest, value: VersionRef) -> None:
    key = value.request_key
    point = value.point
    if isinstance(point, FixedPoint):
        digest.buf += _PACK_QQQQB(
            key.authority_id, key.session_id, key.epoch, value.producer_op_id, 0
        )
        digest.buf += _PACK_I(point.point_index)
        digest.string(point.semantic_digest)
    else:
        digest.buf += _PACK_QQQQB(
            key.authority_id, key.session_id, key.epoch, value.producer_op_id, 1
        )
        digest.buf += _PACK_IB(point.point_index, int(point.selected_point is not None))
        if point.selected_point is not None:
            _digest_product_ref(digest, point.selected_point)
        digest.string(point.producer_plan_digest)


def _digest_bounds(digest: _Digest, value: Bounds) -> None:
    digest.buf += _PACK_IIIQQQ(
        value.max_points,
        value.max_tokens,
        value.max_kv_pages,
        value.max_latent_bytes,
        value.max_completion_bytes,
        value.max_transfer_bytes,
    )


def _digest_rng(digest: _Digest, value: Rng) -> None:
    digest.buf += _PACK_QQB(
        value.seed, value.semantic_index_base, _DRAW_LAYOUT_INDEX[value.draw_layout]
    )


def _digest_sampling(digest: _Digest, value: SamplingParams) -> None:
    digest.f32(value.temperature)
    digest.u32(value.top_k)
    digest.f32(value.top_p)
    digest.boolean(value.ignore_eos)
    digest.option(value.seed, digest.u64)
    digest.f32(value.min_p)
    digest.f32(value.repetition_penalty)
    digest.f32(value.frequency_penalty)
    digest.f32(value.presence_penalty)
    digest.u64(len(value.logit_bias))
    for token, bias in value.logit_bias:
        digest.u32(token)
        digest.f32(bias)
    digest.u64(value.min_tokens)
    digest.boolean(value.return_logprobs)
    digest.u32(value.n_logprobs)
    digest.boolean(value.return_prompt_logprobs)
    digest.u32(value.n_prompt_logprobs)
    digest.u32s(value.logprob_token_ids)
    digest.u64(len(value.bad_words_ids))
    for tokens in value.bad_words_ids:
        digest.u32s(tokens)
    digest.option(value.allowed_token_ids, digest.u32s)
    digest.f32(value.typical_p)
    digest.u32s(value.forced_token_ids)


def _digest_image(digest: _Digest, value: ImageParams) -> None:
    digest.u16(value.steps)
    digest.f32(value.cfg_text_scale)
    digest.f32(value.cfg_img_scale)
    digest.string(value.cfg_renorm_type)
    digest.f32(value.cfg_renorm_min)
    digest.f32(value.cfg_interval[0])
    digest.f32(value.cfg_interval[1])
    digest.f32(value.timestep_shift)
    digest.u32(value.height)
    digest.u32(value.width)
    digest.option(value.seed, digest.u64)
    digest.string(value.negative_prompt)
    digest.u16(value.max_images)
    digest.u64(len(value.image_prompts))
    for prompt in value.image_prompts:
        digest.string(prompt)
    digest.boolean(value.retain_images)


def _digest_und_admission(digest: _Digest, value: UndAdmission) -> None:
    _digest_sampling(digest, value.sampling)
    digest.u32s(value.negative_token_ids)
    digest.u32s(value.finish_token_ids)
    digest.u32(value.kv.prefix_len)
    digest.u32(value.kv.group_id)


# ---------------------------------------------------------------------------
# Decode helpers
#
# Every helper (and every `_fast_*` record decoder below) has one contract: on
# well-formed wire values it produces exactly the value the original readable
# decode would, without allocating error-location strings on the happy path;
# on anything anomalous it falls back to the original checks so the raised
# error is identical. `type(x) is T` guards route exotic-but-valid values
# (int/str subclasses) onto the fallback, which accepts them as before.
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


_HEX64_MATCH = re.compile(r"[0-9a-f]{64}\Z").match


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and _HEX64_MATCH(value) is not None


# ---------------------------------------------------------------------------
# Allocation-light record decoders for the per-batch hot path
#
# Each returns the decoded record for a well-formed wire value and ``None``
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
        semantic_digest = payload.get("semantic_digest")
        if not (type(point_index) is int and point_index >= 0 and type(semantic_digest) is str):
            return None
        point = FixedPoint(point_index, semantic_digest)
    elif tag == "device":
        point_index = payload.get("point_index")
        raw_selected_point = payload.get("selected_point")
        selected_point = (
            None if raw_selected_point is None else _fast_product_ref(raw_selected_point)
        )
        producer_plan_digest = payload.get("producer_plan_digest")
        if (
            not (type(point_index) is int and point_index >= 0)
            or (raw_selected_point is not None and selected_point is None)
            or type(producer_plan_digest) is not str
        ):
            return None
        point = DevicePoint(point_index, selected_point, producer_plan_digest)
    else:
        return None
    reference = object.__new__(VersionRef)
    object.__setattr__(reference, "request_key", request_key)
    object.__setattr__(reference, "producer_op_id", producer_op_id)
    object.__setattr__(reference, "point", point)
    return reference


@lru_cache(maxsize=len(_WORK_VARIANTS))
def _interned_work(kind: str, mode: str | None) -> Work:
    work = object.__new__(Work)
    object.__setattr__(work, "kind", kind)
    object.__setattr__(work, "mode", mode)
    return work


def _fast_work(value: object) -> Work | None:
    if type(value) is not dict:
        return None
    kind = value.get("kind")
    mode = value.get("value")
    if (
        type(kind) is str
        and (mode is None or type(mode) is str)
        and (kind, mode) in _WORK_PAIR_INDEX
    ):
        return _interned_work(kind, mode)
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
