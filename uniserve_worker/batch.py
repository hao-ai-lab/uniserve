"""Typed values crossing the scheduler-to-worker execution boundary."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, TypeAlias, TypeVar, cast

from . import spec as _spec
from .foundation.errors import invalid_descriptor

EXECUTION_PROTOCOL_VERSION = 3


class SequenceMode(StrEnum):
    EXTEND = "extend"
    DECODE = "decode"
    VERIFY = "verify"
    SAMPLE = "sample"


class TokenSource(StrEnum):
    WIRE = "wire"
    LAST_SAMPLED = "last_sampled"


class EncodeKind(StrEnum):
    VISION = "vision"
    LATENT = "latent"


class MaterializeKind(StrEnum):
    IMAGE = "image"
    FRAME = "frame"


class TransferKind(StrEnum):
    PRODUCT = "product"
    KV = "kv"


def _operation_type(operation: Operation) -> _spec.OperationType:
    if isinstance(operation, SequenceOperation):
        return {
            SequenceMode.EXTEND: _spec.OperationType.SEQUENCE_EXTEND,
            SequenceMode.DECODE: _spec.OperationType.SEQUENCE_DECODE,
            SequenceMode.VERIFY: _spec.OperationType.SEQUENCE_VERIFY,
            SequenceMode.SAMPLE: _spec.OperationType.SEQUENCE_SAMPLE,
        }[operation.mode]
    if isinstance(operation, FlowOperation):
        return _spec.OperationType.FLOW
    if isinstance(operation, EncodeOperation):
        return {
            EncodeKind.VISION: _spec.OperationType.ENCODE_VISION,
            EncodeKind.LATENT: _spec.OperationType.ENCODE_LATENT,
        }[operation.kind]
    if isinstance(operation, MaterializeOperation):
        return {
            MaterializeKind.IMAGE: _spec.OperationType.MATERIALIZE_IMAGE,
            MaterializeKind.FRAME: _spec.OperationType.MATERIALIZE_FRAME,
        }[operation.kind]
    return {
        TransferKind.PRODUCT: _spec.OperationType.TRANSFER_PRODUCT,
        TransferKind.KV: _spec.OperationType.TRANSFER_KV,
    }[operation.kind]


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
class KvAllocation:
    block_ids: tuple[int, ...] = ()
    prefix_len: int = 0
    group_id: int = 0

    def __post_init__(self) -> None:
        if self.prefix_len and not self.block_ids:
            raise invalid_descriptor("a non-empty KV prefix requires allocated blocks")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise invalid_descriptor("KV allocation repeats a logical block")
        _nonnegative(self.prefix_len, "kv.prefix_len")
        _nonnegative(self.group_id, "kv.group_id")

    @classmethod
    def from_wire(cls, value: object, where: str = "kv") -> KvAllocation:
        data = _map(value, where)
        return cls(
            block_ids=_uints(data.get("block_ids", ()), f"{where}.block_ids"),
            prefix_len=_uint(data.get("prefix_len", 0), f"{where}.prefix_len"),
            group_id=_uint(data.get("group_id", 0), f"{where}.group_id"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "block_ids": list(self.block_ids),
            "prefix_len": self.prefix_len,
            "group_id": self.group_id,
        }


@dataclass(frozen=True, slots=True)
class SequenceAdmission:
    sampling: SamplingParams = field(default_factory=SamplingParams)
    negative_token_ids: tuple[int, ...] = ()
    kv: KvAllocation = field(default_factory=KvAllocation)

    @classmethod
    def from_wire(cls, value: object, where: str = "sequence admission") -> SequenceAdmission:
        data = _map(value, where)
        return cls(
            sampling=SamplingParams.from_wire(data.get("sampling", {}), f"{where}.sampling"),
            negative_token_ids=_uints(
                data.get("negative_token_ids", ()), f"{where}.negative_token_ids"
            ),
            kv=KvAllocation.from_wire(data.get("kv", {}), f"{where}.kv"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "sampling": self.sampling.to_wire(),
            "negative_token_ids": list(self.negative_token_ids),
            "kv": self.kv.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class FlowAdmission:
    image: ImageParams = field(default_factory=ImageParams)

    @classmethod
    def from_wire(cls, value: object, where: str = "flow admission") -> FlowAdmission:
        data = _map(value, where)
        return cls(image=ImageParams.from_wire(data.get("image", {}), f"{where}.image"))

    def to_wire(self) -> dict[str, object]:
        return {"image": self.image.to_wire()}


@dataclass(frozen=True, slots=True)
class Admission:
    session_id: int
    digest: str
    sequence: SequenceAdmission | None
    flow: FlowAdmission | None
    adapter_id: int | None = None

    def __post_init__(self) -> None:
        _nonnegative(self.session_id, "admission.session_id")
        if self.sequence is None and self.flow is None:
            raise invalid_descriptor("admission must declare sequence or flow state")
        if self.adapter_id is not None:
            _nonnegative(self.adapter_id, "admission.adapter_id")

    @classmethod
    def create(
        cls,
        session_id: int,
        *,
        sequence: SequenceAdmission | None = None,
        flow: FlowAdmission | None = None,
        adapter_id: int | None = None,
    ) -> Admission:
        value = cls(session_id, "", sequence, flow, adapter_id)
        return replace(value, digest=value.payload_digest())

    @classmethod
    def from_wire(cls, value: object, where: str = "admission") -> Admission:
        data = _map(value, where)
        admission = cls(
            session_id=_uint(data.get("session_id"), f"{where}.session_id"),
            digest=_str(data.get("digest"), f"{where}.digest"),
            sequence=(
                None
                if data.get("sequence") is None
                else SequenceAdmission.from_wire(data["sequence"], f"{where}.sequence")
            ),
            flow=(
                None
                if data.get("flow") is None
                else FlowAdmission.from_wire(data["flow"], f"{where}.flow")
            ),
            adapter_id=_optional_uint(data.get("adapter_id"), f"{where}.adapter_id"),
        )
        admission.validate()
        return admission

    def validate(self) -> None:
        if not _is_digest(self.digest):
            raise invalid_descriptor("admission digest must be a lowercase SHA-256 digest")
        if self.digest != self.payload_digest():
            raise invalid_descriptor(f"admission digest mismatch for session {self.session_id}")

    def payload_digest(self) -> str:
        digest = _Digest(b"uniserve-admission-v3\0")
        digest.u64(self.session_id)
        digest.option(self.sequence, lambda value: _digest_sequence_admission(digest, value))
        digest.option(self.flow, lambda value: _digest_image(digest, value.image))
        digest.option(self.adapter_id, digest.u32)
        return digest.finish()

    def to_wire(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "digest": self.digest,
            "sequence": None if self.sequence is None else self.sequence.to_wire(),
            "flow": None if self.flow is None else self.flow.to_wire(),
            "adapter_id": self.adapter_id,
        }


@dataclass(frozen=True, slots=True)
class KvLeaseDelta:
    group_id: int = 0
    new_blocks: tuple[int, ...] = ()

    @classmethod
    def from_wire(cls, value: object, where: str = "lease") -> KvLeaseDelta:
        data = _map(value, where)
        return cls(
            group_id=_uint(data.get("group_id", 0), f"{where}.group_id"),
            new_blocks=_uints(data.get("new_blocks", ()), f"{where}.new_blocks"),
        )

    def to_wire(self) -> dict[str, object]:
        return {"group_id": self.group_id, "new_blocks": list(self.new_blocks)}


@dataclass(frozen=True, slots=True)
class TokenPolicy:
    allowed_tokens: tuple[int, ...] = ()
    suppress_tokens: tuple[int, ...] = ()
    recent_tokens: tuple[int, ...] = ()
    publish_kv: bool = False
    publish_kv_on_tokens: tuple[int, ...] = ()

    @classmethod
    def from_wire(cls, value: object, where: str = "token policy") -> TokenPolicy:
        data = _map(value, where)
        return cls(
            allowed_tokens=_uints(data.get("allowed_tokens", ()), f"{where}.allowed_tokens"),
            suppress_tokens=_uints(data.get("suppress_tokens", ()), f"{where}.suppress_tokens"),
            recent_tokens=_uints(data.get("recent_tokens", ()), f"{where}.recent_tokens"),
            publish_kv=_bool(data.get("publish_kv", False), f"{where}.publish_kv"),
            publish_kv_on_tokens=_uints(
                data.get("publish_kv_on_tokens", ()), f"{where}.publish_kv_on_tokens"
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "allowed_tokens": list(self.allowed_tokens),
            "suppress_tokens": list(self.suppress_tokens),
            "recent_tokens": list(self.recent_tokens),
            "publish_kv": self.publish_kv,
            "publish_kv_on_tokens": list(self.publish_kv_on_tokens),
        }


@dataclass(frozen=True, slots=True)
class TokenInput:
    token_ids: tuple[int, ...]
    source: TokenSource = TokenSource.WIRE
    draft_token_ids: tuple[int, ...] = ()
    return_all_logits: bool = False

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise invalid_descriptor("model sequence requires at least one token id")

    @classmethod
    def from_wire(cls, value: object, where: str = "token input") -> TokenInput:
        data = _map(value, where)
        return cls(
            token_ids=_uints(data.get("token_ids", ()), f"{where}.token_ids"),
            source=_enum(TokenSource, data.get("source"), f"{where}.source"),
            draft_token_ids=_uints(data.get("draft_token_ids", ()), f"{where}.draft_token_ids"),
            return_all_logits=_bool(
                data.get("return_all_logits", False), f"{where}.return_all_logits"
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "token_ids": list(self.token_ids),
            "source": self.source.value,
            "draft_token_ids": list(self.draft_token_ids),
            "return_all_logits": self.return_all_logits,
        }


@dataclass(frozen=True, slots=True)
class PublishedProduct:
    handle: int
    locator: str

    def __post_init__(self) -> None:
        if self.handle <= 0 and not self.locator:
            raise invalid_descriptor("published product requires a handle or locator")

    @classmethod
    def from_wire(cls, value: object, where: str = "published product") -> PublishedProduct:
        data = _map(value, where)
        return cls(
            handle=_uint(data.get("handle", 0), f"{where}.handle"),
            locator=_str(data.get("locator", ""), f"{where}.locator"),
        )

    def to_wire(self) -> dict[str, object]:
        return {"handle": self.handle, "locator": self.locator}


@dataclass(frozen=True, slots=True)
class PublishedKv:
    handle: int
    locators: tuple[str, ...]
    source_version: int
    kv_tokens: int
    block_ids: tuple[int, ...]
    group_id: int
    position: int

    def __post_init__(self) -> None:
        if self.handle <= 0 and not self.locators:
            raise invalid_descriptor("published KV requires a local handle or data-plane locators")
        if self.source_version < 1:
            raise invalid_descriptor("published KV source version must be positive")
        if self.kv_tokens and not self.block_ids:
            raise invalid_descriptor("non-empty published KV requires logical blocks")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise invalid_descriptor("published KV repeats a logical block")
        if any(not locator for locator in self.locators):
            raise invalid_descriptor("published KV contains an empty locator")

    @classmethod
    def from_wire(cls, value: object, where: str = "published KV") -> PublishedKv:
        data = _map(value, where)
        return cls(
            handle=_uint(data.get("handle", 0), f"{where}.handle"),
            locators=tuple(
                _str(item, f"{where}.locators[{index}]")
                for index, item in enumerate(_seq(data.get("locators", ()), f"{where}.locators"))
            ),
            source_version=_uint(data.get("source_version"), f"{where}.source_version"),
            kv_tokens=_uint(data.get("kv_tokens"), f"{where}.kv_tokens"),
            block_ids=_uints(data.get("block_ids", ()), f"{where}.block_ids"),
            group_id=_uint(data.get("group_id", 0), f"{where}.group_id"),
            position=_uint(data.get("position"), f"{where}.position"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "handle": self.handle,
            "locators": list(self.locators),
            "source_version": self.source_version,
            "kv_tokens": self.kv_tokens,
            "block_ids": list(self.block_ids),
            "group_id": self.group_id,
            "position": self.position,
        }

    def for_tensor_rank(self, rank: int, size: int) -> PublishedKv:
        """Select this rank's contiguous locator group from a TP publication."""

        rank = int(rank)
        size = int(size)
        if size < 1 or rank < 0 or rank >= size:
            raise invalid_descriptor("published KV tensor-parallel rank is invalid")
        if not self.locators:
            return self
        if len(self.locators) % size:
            raise invalid_descriptor(
                "published KV locators do not divide across tensor-parallel ranks"
            )
        width = len(self.locators) // size
        start = rank * width
        return replace(self, locators=self.locators[start : start + width])


SequenceInput: TypeAlias = TokenInput | PublishedProduct


@dataclass(frozen=True, slots=True)
class SequenceOperation:
    mode: SequenceMode
    lease: KvLeaseDelta
    position: tuple[int, int]
    policy: TokenPolicy
    input: SequenceInput

    def __post_init__(self) -> None:
        if self.position[1] < self.position[0]:
            raise invalid_descriptor("sequence position range is inverted")
        if self.mode is SequenceMode.SAMPLE:
            if not isinstance(self.input, PublishedProduct):
                raise invalid_descriptor("sample sequence requires published logits")
        elif not isinstance(self.input, TokenInput):
            raise invalid_descriptor("model sequence requires token input")
        else:
            token_input = self.input
            if self.mode is SequenceMode.EXTEND:
                if token_input.source is not TokenSource.WIRE:
                    raise invalid_descriptor("sequence extend requires wire token input")
                if token_input.draft_token_ids:
                    raise invalid_descriptor("sequence extend cannot carry draft work")
            elif self.mode is SequenceMode.DECODE:
                if len(token_input.token_ids) != 1 or token_input.draft_token_ids:
                    raise invalid_descriptor(
                        "sequence decode requires one input token and no draft"
                    )
            elif self.mode is SequenceMode.VERIFY:
                if len(token_input.token_ids) != 1 or not token_input.draft_token_ids:
                    raise invalid_descriptor("sequence verify requires one input token and a draft")


@dataclass(frozen=True, slots=True)
class Guidance:
    branch_count: int
    text_scale: float
    image_scale: float
    renorm_type: str
    renorm_min: float
    interval: tuple[float, float]

    def __post_init__(self) -> None:
        if self.branch_count < 1:
            raise invalid_descriptor("flow guidance requires a branch")
        if any(
            not math.isfinite(float(value))
            for value in (*self.interval, self.text_scale, self.image_scale, self.renorm_min)
        ):
            raise invalid_descriptor("guidance values must be finite")


@dataclass(frozen=True, slots=True)
class FlowOperation:
    latent_handle: int
    position: int
    start_step: int
    step_count: int
    conditioning_position: int
    conditioning: PublishedKv | None
    guidance: Guidance
    image_prompt: str

    def __post_init__(self) -> None:
        if self.latent_handle < 1:
            raise invalid_descriptor("flow operation requires a latent handle")
        if self.step_count < 1:
            raise invalid_descriptor("flow step count must be positive")


@dataclass(frozen=True, slots=True)
class InlineImage:
    base64: str
    content_hash: int

    def __post_init__(self) -> None:
        if not self.base64 or self.content_hash < 1:
            raise invalid_descriptor("inline encode input is invalid")


@dataclass(frozen=True, slots=True)
class StagedProduct:
    handle: int
    content_hash: int

    def __post_init__(self) -> None:
        if self.handle < 1 or self.content_hash < 1:
            raise invalid_descriptor("staged encode input is invalid")


@dataclass(frozen=True, slots=True)
class CachedProduct:
    content_hash: int

    def __post_init__(self) -> None:
        if self.content_hash < 1:
            raise invalid_descriptor("cached encode input is invalid")


EncodeInput: TypeAlias = InlineImage | StagedProduct | CachedProduct


@dataclass(frozen=True, slots=True)
class EncodeOperation:
    kind: EncodeKind
    lease: KvLeaseDelta
    position: tuple[int, int]
    conditioning_position: int
    input: EncodeInput

    def __post_init__(self) -> None:
        if self.position[1] < self.position[0]:
            raise invalid_descriptor("encode position range is inverted")


@dataclass(frozen=True, slots=True)
class LatentProduct:
    handle: int

    def __post_init__(self) -> None:
        if self.handle < 1:
            raise invalid_descriptor("materialize operation requires a latent handle")


MaterializeInput: TypeAlias = LatentProduct | PublishedProduct


@dataclass(frozen=True, slots=True)
class MaterializeOperation:
    kind: MaterializeKind
    lease: KvLeaseDelta
    position: int
    conditioning_position: int
    policy: TokenPolicy
    input: MaterializeInput


@dataclass(frozen=True, slots=True)
class TransferOperation:
    kind: TransferKind
    lease: KvLeaseDelta
    position: int
    conditioning_position: int
    policy: TokenPolicy
    source: PublishedProduct


Operation: TypeAlias = (
    SequenceOperation | FlowOperation | EncodeOperation | MaterializeOperation | TransferOperation
)


@dataclass(frozen=True, slots=True)
class OperationEnvelope:
    session_id: int
    epoch: int
    op_id: int
    base_version: int
    digest: str
    admission_digest: str
    model_spec_digest: str
    weight_digest: str
    operation: Operation

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise invalid_descriptor("operation epoch must be positive")
        if self.op_id < 1:
            raise invalid_descriptor("operation id must be positive")
        _nonnegative(self.session_id, "operation.session_id")
        _nonnegative(self.base_version, "operation.base_version")

    @property
    def kind(self) -> _spec.OperationKind:
        return self.operation_type.kind

    @property
    def operation_type(self) -> _spec.OperationType:
        return _operation_type(self.operation)

    @classmethod
    def create(
        cls,
        *,
        session_id: int,
        epoch: int,
        op_id: int,
        base_version: int,
        admission_digest: str,
        model_spec_digest: str,
        weight_digest: str,
        operation: Operation,
    ) -> OperationEnvelope:
        value = cls(
            session_id,
            epoch,
            op_id,
            base_version,
            "",
            admission_digest,
            model_spec_digest,
            weight_digest,
            operation,
        )
        return replace(value, digest=value.payload_digest())

    @classmethod
    def from_wire(cls, value: object, where: str = "operation") -> OperationEnvelope:
        data = _map(value, where)
        envelope = cls(
            session_id=_uint(data.get("session_id"), f"{where}.session_id"),
            epoch=_uint(data.get("epoch"), f"{where}.epoch"),
            op_id=_uint(data.get("op_id"), f"{where}.op_id"),
            base_version=_uint(data.get("base_version"), f"{where}.base_version"),
            digest=_str(data.get("digest"), f"{where}.digest"),
            admission_digest=_str(data.get("admission_digest"), f"{where}.admission_digest"),
            model_spec_digest=_str(data.get("model_spec_digest"), f"{where}.model_spec_digest"),
            weight_digest=_str(data.get("weight_digest"), f"{where}.weight_digest"),
            operation=_parse_operation(data.get("operation"), f"{where}.operation"),
        )
        envelope.validate()
        return envelope

    def validate(self) -> None:
        for name in ("admission_digest", "model_spec_digest", "weight_digest", "digest"):
            if not _is_digest(getattr(self, name)):
                raise invalid_descriptor(f"operation {name.replace('_', ' ')} is invalid")
        if self.digest != self.payload_digest():
            raise invalid_descriptor(f"operation digest mismatch for session {self.session_id}")

    def payload_digest(self) -> str:
        digest = _Digest(b"uniserve-operation-v3\0")
        digest.u64(self.session_id)
        digest.u64(self.epoch)
        digest.u64(self.op_id)
        digest.u64(self.base_version)
        digest.string(self.admission_digest)
        digest.string(self.model_spec_digest)
        digest.string(self.weight_digest)
        _digest_operation(digest, self.operation)
        return digest.finish()

    def to_wire(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "epoch": self.epoch,
            "op_id": self.op_id,
            "base_version": self.base_version,
            "digest": self.digest,
            "admission_digest": self.admission_digest,
            "model_spec_digest": self.model_spec_digest,
            "weight_digest": self.weight_digest,
            "operation": _operation_to_wire(self.operation),
        }


@dataclass(frozen=True, slots=True)
class SessionProjection:
    session_id: int
    epoch: int
    version: int
    last_op_id: int
    admission_digest: str
    source_digest: str
    last_sampled_token: int | None = None

    def __post_init__(self) -> None:
        if self.epoch < 1 or self.version < 1 or self.last_op_id < 1:
            raise invalid_descriptor("session projection must identify committed state")
        _nonnegative(self.session_id, "session projection.session_id")
        if not _is_digest(self.admission_digest) or not _is_digest(self.source_digest):
            raise invalid_descriptor("session projection digest is invalid")
        if self.last_sampled_token is not None:
            _nonnegative(self.last_sampled_token, "session projection.last_sampled_token")

    def validate_for(self, operation: OperationEnvelope) -> None:
        if (
            self.session_id != operation.session_id
            or self.epoch != operation.epoch
            or self.version != operation.base_version
            or self.admission_digest != operation.admission_digest
        ):
            raise invalid_descriptor("session projection does not match operation identity")

    @classmethod
    def from_wire(cls, value: object, where: str = "session projection") -> SessionProjection:
        data = _map(value, where)
        return cls(
            session_id=_uint(data.get("session_id"), f"{where}.session_id"),
            epoch=_uint(data.get("epoch"), f"{where}.epoch"),
            version=_uint(data.get("version"), f"{where}.version"),
            last_op_id=_uint(data.get("last_op_id"), f"{where}.last_op_id"),
            admission_digest=_str(data.get("admission_digest"), f"{where}.admission_digest"),
            source_digest=_str(data.get("source_digest"), f"{where}.source_digest"),
            last_sampled_token=_optional_uint(
                data.get("last_sampled_token"), f"{where}.last_sampled_token"
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "epoch": self.epoch,
            "version": self.version,
            "last_op_id": self.last_op_id,
            "admission_digest": self.admission_digest,
            "source_digest": self.source_digest,
            "last_sampled_token": self.last_sampled_token,
        }


@dataclass(frozen=True, slots=True)
class Batch:
    step_id: int
    admissions: tuple[Admission, ...]
    projections: tuple[SessionProjection, ...]
    operations: tuple[OperationEnvelope, ...]
    protocol_version: int = EXECUTION_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.protocol_version != EXECUTION_PROTOCOL_VERSION:
            raise invalid_descriptor(
                f"unsupported execution protocol version {self.protocol_version}"
            )
        if not self.operations:
            raise invalid_descriptor("execution batch must contain an operation")
        for operation in self.operations:
            operation.validate()
        sessions = tuple(value.session_id for value in self.operations)
        if len(set(sessions)) != len(sessions):
            raise invalid_descriptor("execution batch contains multiple operations for one session")
        admitted = tuple(value.session_id for value in self.admissions)
        if len(set(admitted)) != len(admitted):
            raise invalid_descriptor("execution batch contains duplicate admissions")
        operations = {value.session_id: value for value in self.operations}
        for admission in self.admissions:
            admission.validate()
            admitted_operation = operations.get(admission.session_id)
            if admitted_operation is None:
                raise invalid_descriptor("execution batch admits a session without an operation")
            if admitted_operation.admission_digest != admission.digest:
                raise invalid_descriptor(
                    f"operation admission digest mismatch for session {admission.session_id}"
                )
        projected = tuple(value.session_id for value in self.projections)
        if len(set(projected)) != len(projected):
            raise invalid_descriptor("execution batch contains duplicate session projections")
        for projection in self.projections:
            projected_operation = operations.get(projection.session_id)
            if projected_operation is None:
                raise invalid_descriptor("execution batch projects a session without an operation")
            projection.validate_for(projected_operation)

    @classmethod
    def from_wire(cls, value: object) -> Batch:
        data = _map(value, "execute batch")
        protocol_version = _uint(data.get("protocol_version"), "execute batch.protocol_version")
        return cls(
            protocol_version=protocol_version,
            step_id=_uint(data.get("step_id"), "execute batch.step_id"),
            admissions=tuple(
                Admission.from_wire(item, f"execute batch.admissions[{index}]")
                for index, item in enumerate(
                    _seq(data.get("admissions", ()), "execute batch.admissions")
                )
            ),
            projections=tuple(
                SessionProjection.from_wire(item, f"execute batch.projections[{index}]")
                for index, item in enumerate(
                    _seq(data.get("projections", ()), "execute batch.projections")
                )
            ),
            operations=tuple(
                OperationEnvelope.from_wire(item, f"execute batch.operations[{index}]")
                for index, item in enumerate(
                    _seq(data.get("operations", ()), "execute batch.operations")
                )
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "step_id": self.step_id,
            "admissions": [value.to_wire() for value in self.admissions],
            "projections": [value.to_wire() for value in self.projections],
            "operations": [value.to_wire() for value in self.operations],
        }


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    token_id: int
    logprob: float
    rank: int

    @classmethod
    def from_wire(cls, value: object, where: str) -> TokenLogprob:
        items = _seq(value, where)
        if len(items) != 3:
            raise invalid_descriptor(f"{where} must contain token, logprob, and rank")
        return cls(
            _uint(items[0], f"{where}[0]"),
            _float(items[1], f"{where}[1]"),
            _uint(items[2], f"{where}[2]"),
        )

    def to_wire(self) -> list[int | float]:
        return [self.token_id, self.logprob, self.rank]


@dataclass(frozen=True, slots=True)
class SequenceEffect:
    sampled_token_ids: tuple[int, ...] = ()
    sampled_logprob: float | None = None
    top_logprobs: tuple[TokenLogprob, ...] = ()
    prompt_logprobs: tuple[tuple[TokenLogprob, ...], ...] = ()
    accepted_draft_tokens: int | None = None
    kv_tokens: int | None = None
    published_logits: PublishedProduct | None = None
    published_kv: PublishedKv | None = None

    @classmethod
    def from_wire(cls, value: object, where: str = "sequence effect") -> SequenceEffect:
        data = _map(value, where)
        return cls(
            sampled_token_ids=_uints(
                data.get("sampled_token_ids", ()), f"{where}.sampled_token_ids"
            ),
            sampled_logprob=_optional_float(
                data.get("sampled_logprob"), f"{where}.sampled_logprob"
            ),
            top_logprobs=tuple(
                TokenLogprob.from_wire(item, f"{where}.top_logprobs[{index}]")
                for index, item in enumerate(
                    _seq(data.get("top_logprobs", ()), f"{where}.top_logprobs")
                )
            ),
            prompt_logprobs=tuple(
                tuple(
                    TokenLogprob.from_wire(
                        entry, f"{where}.prompt_logprobs[{index}][{entry_index}]"
                    )
                    for entry_index, entry in enumerate(
                        _seq(row, f"{where}.prompt_logprobs[{index}]")
                    )
                )
                for index, row in enumerate(
                    _seq(data.get("prompt_logprobs", ()), f"{where}.prompt_logprobs")
                )
            ),
            accepted_draft_tokens=_optional_uint(
                data.get("accepted_draft_tokens"), f"{where}.accepted_draft_tokens"
            ),
            kv_tokens=_optional_uint(data.get("kv_tokens"), f"{where}.kv_tokens"),
            published_logits=(
                None
                if data.get("published_logits") is None
                else PublishedProduct.from_wire(
                    data["published_logits"], f"{where}.published_logits"
                )
            ),
            published_kv=(
                None
                if data.get("published_kv") is None
                else PublishedKv.from_wire(data["published_kv"], f"{where}.published_kv")
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "sampled_token_ids": list(self.sampled_token_ids),
            "sampled_logprob": self.sampled_logprob,
            "top_logprobs": [value.to_wire() for value in self.top_logprobs],
            "prompt_logprobs": [[entry.to_wire() for entry in row] for row in self.prompt_logprobs],
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "kv_tokens": self.kv_tokens,
            "published_logits": (
                None if self.published_logits is None else self.published_logits.to_wire()
            ),
            "published_kv": None if self.published_kv is None else self.published_kv.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class SequenceDelta:
    effect: SequenceEffect


@dataclass(frozen=True, slots=True)
class FlowDelta:
    steps_completed: int
    done: bool


@dataclass(frozen=True, slots=True)
class EncodeDelta:
    product_handle: int
    kv_tokens: int
    image_size: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ImageArtifact:
    png_base64: str
    height: int
    width: int
    handle: int
    locator: str


@dataclass(frozen=True, slots=True)
class FrameRecord:
    count: int


MaterializedProduct: TypeAlias = ImageArtifact | PublishedProduct | FrameRecord


@dataclass(frozen=True, slots=True)
class MaterializeDelta:
    product: MaterializedProduct
    kv_tokens: int | None = None
    sequence: SequenceEffect | None = None


@dataclass(frozen=True, slots=True)
class TransferDelta:
    product: PublishedProduct | None = None
    kv_tokens: int | None = None
    sequence: SequenceEffect | None = None


ResultDelta: TypeAlias = SequenceDelta | FlowDelta | EncodeDelta | MaterializeDelta | TransferDelta


def _delta_kind(delta: ResultDelta) -> _spec.OperationKind:
    if isinstance(delta, SequenceDelta):
        return _spec.OperationKind.SEQUENCE
    if isinstance(delta, FlowDelta):
        return _spec.OperationKind.FLOW
    if isinstance(delta, EncodeDelta):
        return _spec.OperationKind.ENCODE
    if isinstance(delta, MaterializeDelta):
        return _spec.OperationKind.MATERIALIZE
    return _spec.OperationKind.TRANSFER


@dataclass(frozen=True, slots=True)
class OperationResult:
    session_id: int
    epoch: int
    op_id: int
    base_version: int
    result_version: int
    delta: ResultDelta

    @classmethod
    def for_operation(cls, operation: OperationEnvelope, delta: ResultDelta) -> OperationResult:
        result = cls(
            session_id=operation.session_id,
            epoch=operation.epoch,
            op_id=operation.op_id,
            base_version=operation.base_version,
            result_version=operation.base_version + 1,
            delta=delta,
        )
        result.validate_for(operation)
        return result

    def validate_for(self, operation: OperationEnvelope) -> None:
        if (
            self.session_id,
            self.epoch,
            self.op_id,
            self.base_version,
            self.result_version,
        ) != (
            operation.session_id,
            operation.epoch,
            operation.op_id,
            operation.base_version,
            operation.base_version + 1,
        ):
            raise invalid_descriptor("operation result identity or version does not match")
        if _delta_kind(self.delta) is not operation.kind:
            raise invalid_descriptor("operation result delta variant does not match operation")

    @classmethod
    def from_wire(cls, value: object, where: str = "operation result") -> OperationResult:
        data = _map(value, where)
        return cls(
            session_id=_uint(data.get("session_id"), f"{where}.session_id"),
            epoch=_uint(data.get("epoch"), f"{where}.epoch"),
            op_id=_uint(data.get("op_id"), f"{where}.op_id"),
            base_version=_uint(data.get("base_version"), f"{where}.base_version"),
            result_version=_uint(data.get("result_version"), f"{where}.result_version"),
            delta=_delta_from_wire(data.get("delta"), f"{where}.delta"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "epoch": self.epoch,
            "op_id": self.op_id,
            "base_version": self.base_version,
            "result_version": self.result_version,
            "delta": _delta_to_wire(self.delta),
        }


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

    def to_wire(self) -> dict[str, object]:
        return {
            name: dict(value) if isinstance(value, Mapping) else value
            for name, value in ((name, getattr(self, name)) for name in self.__dataclass_fields__)
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    step_id: int
    operations: tuple[OperationResult, ...]
    worker_exec_us: int | None = None
    forward_stats: WorkerForwardStats | None = None

    def validate_for(self, batch: Batch) -> None:
        if self.step_id != batch.step_id:
            raise invalid_descriptor("execution result step does not match batch")
        if len(self.operations) != len(batch.operations):
            raise invalid_descriptor("execution result operation count does not match batch")
        for result, operation in zip(self.operations, batch.operations, strict=True):
            result.validate_for(operation)

    def to_wire(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "operations": [value.to_wire() for value in self.operations],
            "worker_exec_us": self.worker_exec_us,
            "forward_stats": None if self.forward_stats is None else self.forward_stats.to_wire(),
        }


class _Digest:
    def __init__(self, domain: bytes) -> None:
        self.value = hashlib.sha256()
        self.value.update(domain)
        self.u16(EXECUTION_PROTOCOL_VERSION)

    def finish(self) -> str:
        return self.value.hexdigest()

    def u8(self, value: int) -> None:
        self.value.update(struct.pack("<B", value))

    def u16(self, value: int) -> None:
        self.value.update(struct.pack("<H", value))

    def u32(self, value: int) -> None:
        self.value.update(struct.pack("<I", value))

    def u64(self, value: int) -> None:
        self.value.update(struct.pack("<Q", value))

    def f32(self, value: float) -> None:
        self.value.update(struct.pack("<f", value))

    def boolean(self, value: bool) -> None:
        self.u8(int(value))

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.u64(len(encoded))
        self.value.update(encoded)

    def u32s(self, values: Sequence[int]) -> None:
        self.u64(len(values))
        for value in values:
            self.u32(value)

    def option(self, value: object | None, encode: Any) -> None:
        if value is None:
            self.u8(0)
        else:
            self.u8(1)
            encode(value)


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


def _digest_sequence_admission(digest: _Digest, value: SequenceAdmission) -> None:
    _digest_sampling(digest, value.sampling)
    digest.u32s(value.negative_token_ids)
    digest.u32s(value.kv.block_ids)
    digest.u32(value.kv.prefix_len)
    digest.u32(value.kv.group_id)


def _digest_lease(digest: _Digest, value: KvLeaseDelta) -> None:
    digest.u32(value.group_id)
    digest.u32s(value.new_blocks)


def _digest_policy(digest: _Digest, value: TokenPolicy) -> None:
    digest.u32s(value.allowed_tokens)
    digest.u32s(value.suppress_tokens)
    digest.u32s(value.recent_tokens)
    digest.boolean(value.publish_kv)
    digest.u32s(value.publish_kv_on_tokens)


def _digest_published(digest: _Digest, value: PublishedProduct) -> None:
    digest.u64(value.handle)


def _digest_published_kv(digest: _Digest, value: PublishedKv) -> None:
    digest.u64(value.handle)
    digest.u64(value.source_version)
    digest.u32(value.kv_tokens)
    digest.u32s(value.block_ids)
    digest.u32(value.group_id)
    digest.u32(value.position)


def _digest_operation(digest: _Digest, value: Operation) -> None:
    if isinstance(value, SequenceOperation):
        digest.u8(0)
        digest.u8(list(SequenceMode).index(value.mode))
        _digest_lease(digest, value.lease)
        digest.u32(value.position[0])
        digest.u32(value.position[1])
        _digest_policy(digest, value.policy)
        if isinstance(value.input, TokenInput):
            digest.u8(0)
            digest.u32s(value.input.token_ids)
            digest.u8(list(TokenSource).index(value.input.source))
            digest.u32s(value.input.draft_token_ids)
            digest.boolean(value.input.return_all_logits)
        else:
            digest.u8(1)
            _digest_published(digest, value.input)
        return
    if isinstance(value, FlowOperation):
        digest.u8(1)
        digest.u64(value.latent_handle)
        digest.u32(value.position)
        digest.u16(value.start_step)
        digest.u16(value.step_count)
        digest.u32(value.conditioning_position)
        digest.option(
            value.conditioning, lambda conditioning: _digest_published_kv(digest, conditioning)
        )
        digest.u8(value.guidance.branch_count)
        digest.f32(value.guidance.text_scale)
        digest.f32(value.guidance.image_scale)
        digest.string(value.guidance.renorm_type)
        digest.f32(value.guidance.renorm_min)
        digest.f32(value.guidance.interval[0])
        digest.f32(value.guidance.interval[1])
        digest.string(value.image_prompt)
        return
    if isinstance(value, EncodeOperation):
        digest.u8(2)
        digest.u8(list(EncodeKind).index(value.kind))
        _digest_lease(digest, value.lease)
        digest.u32(value.position[0])
        digest.u32(value.position[1])
        digest.u32(value.conditioning_position)
        if isinstance(value.input, InlineImage):
            digest.u8(0)
            digest.string(value.input.base64)
            digest.u64(value.input.content_hash)
        elif isinstance(value.input, StagedProduct):
            digest.u8(1)
            digest.u64(value.input.handle)
            digest.u64(value.input.content_hash)
        else:
            digest.u8(2)
            digest.u64(value.input.content_hash)
        return
    if isinstance(value, MaterializeOperation):
        digest.u8(3)
        digest.u8(list(MaterializeKind).index(value.kind))
        _digest_lease(digest, value.lease)
        digest.u32(value.position)
        digest.u32(value.conditioning_position)
        _digest_policy(digest, value.policy)
        if isinstance(value.input, LatentProduct):
            digest.u8(0)
            digest.u64(value.input.handle)
        else:
            digest.u8(1)
            _digest_published(digest, value.input)
        return
    digest.u8(4)
    digest.u8(list(TransferKind).index(value.kind))
    _digest_lease(digest, value.lease)
    digest.u32(value.position)
    digest.u32(value.conditioning_position)
    _digest_policy(digest, value.policy)
    _digest_published(digest, value.source)


def _parse_operation(value: object, where: str) -> Operation:
    tagged = _tagged(value, where)
    kind = tagged[0]
    data = _map(tagged[1], f"{where}.value")
    if kind == _spec.OperationKind.SEQUENCE.value:
        position = _uint_pair(data.get("position"), f"{where}.value.position")
        input_kind, input_value = _tagged(data.get("input"), f"{where}.value.input")
        if input_kind == "tokens":
            sequence_input: SequenceInput = TokenInput.from_wire(
                input_value, f"{where}.value.input.value"
            )
        elif input_kind == "published_logits":
            sequence_input = PublishedProduct.from_wire(input_value, f"{where}.value.input.value")
        else:
            raise invalid_descriptor(f"{where}.value.input has unknown variant {input_kind!r}")
        return SequenceOperation(
            mode=_enum(SequenceMode, data.get("mode"), f"{where}.value.mode"),
            lease=KvLeaseDelta.from_wire(data.get("lease", {}), f"{where}.value.lease"),
            position=position,
            policy=TokenPolicy.from_wire(data.get("policy", {}), f"{where}.value.policy"),
            input=sequence_input,
        )
    if kind == _spec.OperationKind.FLOW.value:
        guidance_data = _map(data.get("guidance"), f"{where}.value.guidance")
        interval = _float_pair(guidance_data.get("interval"), f"{where}.value.guidance.interval")
        return FlowOperation(
            latent_handle=_uint(data.get("latent_handle"), f"{where}.value.latent_handle"),
            position=_uint(data.get("position"), f"{where}.value.position"),
            start_step=_uint(data.get("start_step"), f"{where}.value.start_step"),
            step_count=_uint(data.get("step_count"), f"{where}.value.step_count"),
            conditioning_position=_uint(
                data.get("conditioning_position"), f"{where}.value.conditioning_position"
            ),
            conditioning=(
                None
                if data.get("conditioning") is None
                else PublishedKv.from_wire(data["conditioning"], f"{where}.value.conditioning")
            ),
            guidance=Guidance(
                branch_count=_uint(
                    guidance_data.get("branch_count"), f"{where}.value.guidance.branch_count"
                ),
                text_scale=_float(
                    guidance_data.get("text_scale"), f"{where}.value.guidance.text_scale"
                ),
                image_scale=_float(
                    guidance_data.get("image_scale"), f"{where}.value.guidance.image_scale"
                ),
                renorm_type=_str(
                    guidance_data.get("renorm_type"), f"{where}.value.guidance.renorm_type"
                ),
                renorm_min=_float(
                    guidance_data.get("renorm_min"), f"{where}.value.guidance.renorm_min"
                ),
                interval=interval,
            ),
            image_prompt=_str(data.get("image_prompt", ""), f"{where}.value.image_prompt"),
        )
    if kind == _spec.OperationKind.ENCODE.value:
        input_kind, input_value = _tagged(data.get("input"), f"{where}.value.input")
        input_data = _map(input_value, f"{where}.value.input.value")
        if input_kind == "inline_image":
            encode_input: EncodeInput = InlineImage(
                _str(input_data.get("base64"), f"{where}.value.input.value.base64"),
                _uint(input_data.get("content_hash"), f"{where}.value.input.value.content_hash"),
            )
        elif input_kind == "staged_product":
            encode_input = StagedProduct(
                _uint(input_data.get("handle"), f"{where}.value.input.value.handle"),
                _uint(input_data.get("content_hash"), f"{where}.value.input.value.content_hash"),
            )
        elif input_kind == "cached_product":
            encode_input = CachedProduct(
                _uint(input_data.get("content_hash"), f"{where}.value.input.value.content_hash")
            )
        else:
            raise invalid_descriptor(f"{where}.value.input has unknown variant {input_kind!r}")
        return EncodeOperation(
            kind=_enum(EncodeKind, data.get("kind"), f"{where}.value.kind"),
            lease=KvLeaseDelta.from_wire(data.get("lease", {}), f"{where}.value.lease"),
            position=_uint_pair(data.get("position"), f"{where}.value.position"),
            conditioning_position=_uint(
                data.get("conditioning_position"), f"{where}.value.conditioning_position"
            ),
            input=encode_input,
        )
    if kind == _spec.OperationKind.MATERIALIZE.value:
        input_kind, input_value = _tagged(data.get("input"), f"{where}.value.input")
        if input_kind == "latent":
            materialize_input: MaterializeInput = LatentProduct(
                _uint(
                    _map(input_value, f"{where}.value.input.value").get("handle"),
                    f"{where}.value.input.value.handle",
                )
            )
        elif input_kind == "published":
            materialize_input = PublishedProduct.from_wire(
                input_value, f"{where}.value.input.value"
            )
        else:
            raise invalid_descriptor(f"{where}.value.input has unknown variant {input_kind!r}")
        return MaterializeOperation(
            kind=_enum(MaterializeKind, data.get("kind"), f"{where}.value.kind"),
            lease=KvLeaseDelta.from_wire(data.get("lease", {}), f"{where}.value.lease"),
            position=_uint(data.get("position"), f"{where}.value.position"),
            conditioning_position=_uint(
                data.get("conditioning_position"), f"{where}.value.conditioning_position"
            ),
            policy=TokenPolicy.from_wire(data.get("policy", {}), f"{where}.value.policy"),
            input=materialize_input,
        )
    if kind == _spec.OperationKind.TRANSFER.value:
        return TransferOperation(
            kind=_enum(TransferKind, data.get("kind"), f"{where}.value.kind"),
            lease=KvLeaseDelta.from_wire(data.get("lease", {}), f"{where}.value.lease"),
            position=_uint(data.get("position"), f"{where}.value.position"),
            conditioning_position=_uint(
                data.get("conditioning_position"), f"{where}.value.conditioning_position"
            ),
            policy=TokenPolicy.from_wire(data.get("policy", {}), f"{where}.value.policy"),
            source=PublishedProduct.from_wire(data.get("source"), f"{where}.value.source"),
        )
    raise invalid_descriptor(f"{where} has unknown variant {kind!r}")


def _operation_to_wire(value: Operation) -> dict[str, object]:
    if isinstance(value, SequenceOperation):
        input_value = (
            {"kind": "tokens", "value": value.input.to_wire()}
            if isinstance(value.input, TokenInput)
            else {"kind": "published_logits", "value": value.input.to_wire()}
        )
        payload: dict[str, object] = {
            "mode": value.mode.value,
            "lease": value.lease.to_wire(),
            "position": list(value.position),
            "policy": value.policy.to_wire(),
            "input": input_value,
        }
        return {"kind": "sequence", "value": payload}
    if isinstance(value, FlowOperation):
        return {
            "kind": "flow",
            "value": {
                "latent_handle": value.latent_handle,
                "position": value.position,
                "start_step": value.start_step,
                "step_count": value.step_count,
                "conditioning_position": value.conditioning_position,
                "conditioning": (
                    None if value.conditioning is None else value.conditioning.to_wire()
                ),
                "guidance": {
                    "branch_count": value.guidance.branch_count,
                    "text_scale": value.guidance.text_scale,
                    "image_scale": value.guidance.image_scale,
                    "renorm_type": value.guidance.renorm_type,
                    "renorm_min": value.guidance.renorm_min,
                    "interval": list(value.guidance.interval),
                },
                "image_prompt": value.image_prompt,
            },
        }
    if isinstance(value, EncodeOperation):
        if isinstance(value.input, InlineImage):
            input_value = {
                "kind": "inline_image",
                "value": {"base64": value.input.base64, "content_hash": value.input.content_hash},
            }
        elif isinstance(value.input, StagedProduct):
            input_value = {
                "kind": "staged_product",
                "value": {"handle": value.input.handle, "content_hash": value.input.content_hash},
            }
        else:
            input_value = {
                "kind": "cached_product",
                "value": {"content_hash": value.input.content_hash},
            }
        return {
            "kind": "encode",
            "value": {
                "kind": value.kind.value,
                "lease": value.lease.to_wire(),
                "position": list(value.position),
                "conditioning_position": value.conditioning_position,
                "input": input_value,
            },
        }
    if isinstance(value, MaterializeOperation):
        input_value = (
            {"kind": "latent", "value": {"handle": value.input.handle}}
            if isinstance(value.input, LatentProduct)
            else {"kind": "published", "value": value.input.to_wire()}
        )
        return {
            "kind": "materialize",
            "value": {
                "kind": value.kind.value,
                "lease": value.lease.to_wire(),
                "position": value.position,
                "conditioning_position": value.conditioning_position,
                "policy": value.policy.to_wire(),
                "input": input_value,
            },
        }
    return {
        "kind": "transfer",
        "value": {
            "kind": value.kind.value,
            "lease": value.lease.to_wire(),
            "position": value.position,
            "conditioning_position": value.conditioning_position,
            "policy": value.policy.to_wire(),
            "source": value.source.to_wire(),
        },
    }


def _delta_to_wire(value: ResultDelta) -> dict[str, object]:
    if isinstance(value, SequenceDelta):
        return {"kind": "sequence", "value": {"effect": value.effect.to_wire()}}
    if isinstance(value, FlowDelta):
        return {
            "kind": "flow",
            "value": {"steps_completed": value.steps_completed, "done": value.done},
        }
    if isinstance(value, EncodeDelta):
        return {
            "kind": "encode",
            "value": {
                "product_handle": value.product_handle,
                "kv_tokens": value.kv_tokens,
                "image_size": None if value.image_size is None else list(value.image_size),
            },
        }
    if isinstance(value, MaterializeDelta):
        if isinstance(value.product, ImageArtifact):
            product = {
                "kind": "image",
                "value": {
                    "png_base64": value.product.png_base64,
                    "height": value.product.height,
                    "width": value.product.width,
                    "handle": value.product.handle,
                    "locator": value.product.locator,
                },
            }
        elif isinstance(value.product, PublishedProduct):
            product = {"kind": "published", "value": value.product.to_wire()}
        else:
            product = {"kind": "frame", "value": {"count": value.product.count}}
        return {
            "kind": "materialize",
            "value": {
                "product": product,
                "kv_tokens": value.kv_tokens,
                "sequence": None if value.sequence is None else value.sequence.to_wire(),
            },
        }
    return {
        "kind": "transfer",
        "value": {
            "product": None if value.product is None else value.product.to_wire(),
            "kv_tokens": value.kv_tokens,
            "sequence": None if value.sequence is None else value.sequence.to_wire(),
        },
    }


def _delta_from_wire(value: object, where: str) -> ResultDelta:
    kind, raw = _tagged(value, where)
    data = _map(raw, f"{where}.value")
    if kind == "sequence":
        return SequenceDelta(SequenceEffect.from_wire(data.get("effect"), f"{where}.value.effect"))
    if kind == "flow":
        return FlowDelta(
            steps_completed=_uint(data.get("steps_completed"), f"{where}.value.steps_completed"),
            done=_bool(data.get("done"), f"{where}.value.done"),
        )
    if kind == "encode":
        raw_size = data.get("image_size")
        return EncodeDelta(
            product_handle=_uint(data.get("product_handle"), f"{where}.value.product_handle"),
            kv_tokens=_uint(data.get("kv_tokens"), f"{where}.value.kv_tokens"),
            image_size=(
                None if raw_size is None else _uint_pair(raw_size, f"{where}.value.image_size")
            ),
        )
    if kind == "materialize":
        product_kind, raw_product = _tagged(data.get("product"), f"{where}.value.product")
        product_data = _map(raw_product, f"{where}.value.product.value")
        if product_kind == "image":
            product: MaterializedProduct = ImageArtifact(
                png_base64=_str(
                    product_data.get("png_base64"),
                    f"{where}.value.product.value.png_base64",
                ),
                height=_uint(product_data.get("height"), f"{where}.value.product.value.height"),
                width=_uint(product_data.get("width"), f"{where}.value.product.value.width"),
                handle=_uint(product_data.get("handle"), f"{where}.value.product.value.handle"),
                locator=_str(
                    product_data.get("locator", ""),
                    f"{where}.value.product.value.locator",
                ),
            )
        elif product_kind == "published":
            product = PublishedProduct.from_wire(raw_product, f"{where}.value.product.value")
        elif product_kind == "frame":
            product = FrameRecord(
                _uint(product_data.get("count"), f"{where}.value.product.value.count")
            )
        else:
            raise invalid_descriptor(f"{where}.value.product has unknown variant {product_kind!r}")
        return MaterializeDelta(
            product=product,
            kv_tokens=_optional_uint(data.get("kv_tokens"), f"{where}.value.kv_tokens"),
            sequence=(
                None
                if data.get("sequence") is None
                else SequenceEffect.from_wire(data["sequence"], f"{where}.value.sequence")
            ),
        )
    if kind == "transfer":
        return TransferDelta(
            product=(
                None
                if data.get("product") is None
                else PublishedProduct.from_wire(data["product"], f"{where}.value.product")
            ),
            kv_tokens=_optional_uint(data.get("kv_tokens"), f"{where}.value.kv_tokens"),
            sequence=(
                None
                if data.get("sequence") is None
                else SequenceEffect.from_wire(data["sequence"], f"{where}.value.sequence")
            ),
        )
    raise invalid_descriptor(f"{where} has unknown variant {kind!r}")


_E = TypeVar("_E", bound=StrEnum)


def _enum(kind: type[_E], value: object, where: str) -> _E:
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    try:
        return kind(value)
    except ValueError:
        raise invalid_descriptor(f"{where} has unknown value {value!r}") from None


def _map(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise invalid_descriptor(f"{where} must be a map")
    return cast(Mapping[str, Any], value)


def _seq(value: object, where: str) -> Sequence[Any]:
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
    if not isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be a bool")
    return value


def _uint(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"{where} must be a non-negative integer")
    return value


def _optional_uint(value: object, where: str) -> int | None:
    return None if value is None else _uint(value, where)


def _float(value: object, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise invalid_descriptor(f"{where} must be finite")
    return result


def _optional_float(value: object, where: str) -> float | None:
    return None if value is None else _float(value, where)


def _uints(value: object, where: str) -> tuple[int, ...]:
    return tuple(_uint(item, f"{where}[{index}]") for index, item in enumerate(_seq(value, where)))


def _uint_pair(value: object, where: str) -> tuple[int, int]:
    items = _pair(value, where)
    return _uint(items[0], f"{where}[0]"), _uint(items[1], f"{where}[1]")


def _float_pair(value: object, where: str) -> tuple[float, float]:
    items = _pair(value, where)
    return _float(items[0], f"{where}[0]"), _float(items[1], f"{where}[1]")


def _nonnegative(value: int, where: str) -> None:
    _uint(value, where)


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [name for name in globals() if not name.startswith("_")]
