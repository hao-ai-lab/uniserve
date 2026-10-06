"""Numerical parameters and operation kinds for native worker calls.

Rust owns `Call`, its data dependencies, and its validation. Python numerical
code borrows the call's immutable parameters through these typed views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from uniserve_worker._uniserve_ipc import Call as Call
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol import tensor
from uniserve_worker.protocol.validation import (
    _bool,
    _float,
    _map,
    _nonnegative,
    _optional_uint,
    _pair,
    _seq,
    _str,
    _uint,
)


class ForwardMode(StrEnum):
    """The numerical mode of one homogeneous token-model call.

    Prefill, decode and verify extend a request's KV cache causally; token
    denoising runs one pass over canvas rows that read the request's cached
    prefix without writing it.
    """

    PREFILL = "prefill"
    DECODE = "decode"
    VERIFY = "verify"
    TOKEN_DENOISING = "token_denoising"


class MediaCall(StrEnum):
    """A concrete media reading, encoder, diffusion, or decoder computation."""

    # Decodes a video request's condition media on a host rank into the
    # inputs of the vision and latent encoders.
    MEDIA_READING = "media_reading"
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
    KV_EXPORT = "kv_export"
    KV_INSTALL = "kv_install"


CallKind: TypeAlias = ForwardMode | MediaCall | TransferMode

# Every call kind a worker may execute, in the order of the Rust
# `CallKind::ALL`. Worker reports list supported calls in this order.
CALL_KINDS: tuple[CallKind, ...] = (
    *ForwardMode,
    *MediaCall,
    *TransferMode,
)


def computation(value: object, where: str) -> CallKind:
    """Decode a call kind from an enum member or its exact wire string.

    Args:
        value: A `ForwardMode`, `MediaCall`, or `TransferMode` member, or the
            plain `str` value of one.
        where: Field path used in the error message.

    Returns:
        The matching call-kind member.

    Raises:
        WorkerError: `value` is neither a call-kind member nor the exact wire
            string of one.
    """
    if (
        isinstance(value, (ForwardMode, MediaCall, TransferMode))
        and value in CALL_KINDS
    ):
        return value
    if type(value) is str:
        member = _COMPUTATION_BY_VALUE.get(value)
        if member is not None:
            return member
    raise invalid_descriptor(f"{where} is not a supported computation")


class CallStatus(StrEnum):
    """Terminal status of one call's completion.

    `PREDICATED` means the call's device predicate disabled it: the call did
    not run and reports the request's accepted coordinates.
    """

    OK = "ok"
    PREDICATED = "predicated"
    ERROR = "error"


class ErrorCode(StrEnum):
    """Classifies bounded execution failures returned to the scheduler."""

    INVALID_CALL = "invalid_call"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    COMPUTE_ERROR = "compute_error"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class DrawLayout(StrEnum):
    """Selects how a call's random draws map to semantic indices.

    Draws are keyed by the request seed and a semantic index rather than by
    execution order, so replays and recomputation reproduce identical random
    values. The meaning of `Rng.semantic_index_base` depends on the layout:
    with `TARGET_SAMPLING` it is the first sampled position, and for
    stochastic sampling the token executor requires one consecutive index per
    sampled position; with `FLOW_NOISE` it is the positive image index the
    engine assigns, from which the diffusion executor derives the noise seed.
    """

    TARGET_SAMPLING = "target_sampling"
    SPECULATIVE_PROPOSAL = "speculative_proposal"
    FLOW_NOISE = "flow_noise"


_COMPUTATION_BY_VALUE = {member.value: member for member in CALL_KINDS}


@dataclass(frozen=True, slots=True)
class ImageParams:
    """Image-generation controls carried by a request's admission.

    Covers the diffusion schedule (steps, timestep shift), classifier-free
    guidance scales and renormalization, output dimensions, and prompt inputs.
    `NewRequest.image` holds one; `__post_init__` mirrors the limits of the
    Rust `ImageParams::validate`.
    """

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
        """Validate schedule, dimensions, guidance, image count, and seed.

        Raises:
            WorkerError: A field is out of range or not finite, a dimension
                is not a multiple of 16, the CFG interval is unordered, or
                the renorm type is blank.
        """
        if not 1 <= self.steps <= 1000:
            raise invalid_descriptor("image.steps must be in 1..=1000")

        for name in ("height", "width"):
            dimension = int(getattr(self, name))
            if not 16 <= dimension <= 4096 or dimension % 16:
                raise invalid_descriptor(
                    f"image.{name} must be a multiple of 16 in 16..=4096"
                )

        for name in ("cfg_text_scale", "cfg_img_scale"):
            scale = float(getattr(self, name))
            if not math.isfinite(scale) or not 0 <= scale <= 100:
                raise invalid_descriptor(
                    f"image.{name} must be finite and in [0, 100]"
                )

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
        """Parse image-generation controls, applying defaults when omitted."""
        data = _map(value, where)
        interval = _pair(
            data.get("cfg_interval", (0.0, 1.0)), f"{where}.cfg_interval"
        )
        return cls(
            steps=_uint(data.get("steps", 50), f"{where}.steps"),
            cfg_text_scale=_float(
                data.get("cfg_text_scale", 4.0), f"{where}.cfg_text_scale"
            ),
            cfg_img_scale=_float(
                data.get("cfg_img_scale", 1.0), f"{where}.cfg_img_scale"
            ),
            cfg_renorm_type=_str(
                data.get("cfg_renorm_type", "global"),
                f"{where}.cfg_renorm_type",
            ),
            cfg_renorm_min=_float(
                data.get("cfg_renorm_min", 0.0), f"{where}.cfg_renorm_min"
            ),
            cfg_interval=(
                _float(interval[0], f"{where}.cfg_interval[0]"),
                _float(interval[1], f"{where}.cfg_interval[1]"),
            ),
            timestep_shift=_float(
                data.get("timestep_shift", 1.0), f"{where}.timestep_shift"
            ),
            height=_uint(data.get("height", 512), f"{where}.height"),
            width=_uint(data.get("width", 512), f"{where}.width"),
            seed=_optional_uint(data.get("seed"), f"{where}.seed"),
            negative_prompt=_str(
                data.get("negative_prompt", ""), f"{where}.negative_prompt"
            ),
            max_images=_uint(data.get("max_images", 1), f"{where}.max_images"),
            image_prompts=tuple(
                _str(item, f"{where}.image_prompts[{index}]")
                for index, item in enumerate(
                    _seq(
                        data.get("image_prompts", ()), f"{where}.image_prompts"
                    )
                )
            ),
            retain_images=_bool(
                data.get("retain_images", True), f"{where}.retain_images"
            ),
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
class Bounds:
    """Resource ceilings the scheduler reserves before a call runs.

    Zero means the call may use none of that resource. `Call.validate` checks
    `max_tokens` against the input token ids and `max_latent_bytes` against
    encoder, latent, and image outputs; native batch validation checks
    `max_transfer_bytes` against installed KV transfers.
    """

    max_tokens: int = 0
    max_kv_units: int = 0
    max_latent_bytes: int = 0
    max_completion_bytes: int = 0
    max_transfer_bytes: int = 0

    def to_mapping(self) -> dict[str, object]:
        """Serialize all call resource ceilings for IPC."""
        return {
            "max_tokens": self.max_tokens,
            "max_kv_units": self.max_kv_units,
            "max_latent_bytes": self.max_latent_bytes,
            "max_completion_bytes": self.max_completion_bytes,
            "max_transfer_bytes": self.max_transfer_bytes,
        }


@dataclass(frozen=True, slots=True)
class Rng:
    """Deterministic random-draw coordinates for one call.

    The engine derives them per call; see `DrawLayout` for how each layout
    interprets `semantic_index_base`.
    """

    # Request-level seed; executors that draw from it check it against the
    # seed the request was admitted with.
    seed: int
    # First semantic index covered by this call's draws.
    semantic_index_base: int
    draw_layout: DrawLayout

    def to_mapping(self) -> dict[str, object]:
        """Serialize deterministic RNG coordinates for IPC."""
        return {
            "seed": self.seed,
            "semantic_index_base": self.semantic_index_base,
            "draw_layout": self.draw_layout.value,
        }


@dataclass(frozen=True, slots=True)
class CallCoordinates:
    """The coordinates a call executes at, as the engine states them.

    A rank would otherwise chain these from the calls that preceded it. The
    engine holds the request state they come from, so it sends them and the
    rank asserts its own ledger agrees. All lengths count tokens.
    """

    # Position of this call's first token in the request's logical sequence.
    logical_position: int = 0
    # Tokens whose KV a numerical call may attend to at submission.
    kv_visible_len: int = 0
    # Tokens whose KV is initialized at submission; never below the visible
    # extent, and above it while a verifier's rejected drafts stay initialized.
    kv_computed_len: int = 0
    # Denoising steps completed for this request at submission.
    flow_step: int = 0

    def to_mapping(self) -> dict[str, object]:
        """Serialize the coordinates a call states."""
        return {
            "logical_position": self.logical_position,
            "kv_visible_len": self.kv_visible_len,
            "kv_computed_len": self.kv_computed_len,
            "flow_step": self.flow_step,
        }


@dataclass(frozen=True, slots=True)
class CanvasStep:
    """One denoising step of a block-diffusion request's resident canvas.

    The request's admitted ``GenerationParams.canvas`` fixes the sampling;
    the worker holds the canvas and its sampler state in the request's slot.
    ``block`` counts the blocks already committed to the request's context
    and ``step`` the steps already run on this canvas; step zero starts the
    canvas from random tokens. The step that stops the canvas reports its
    argmax tokens as the completion's committed tokens.
    """

    block: int
    step: int

    def to_mapping(self) -> dict[str, object]:
        """Serialize a canvas step's counters."""
        return {"block": self.block, "step": self.step}


@dataclass(frozen=True, slots=True)
class Readout:
    """Candidate log-probabilities a token-denoising call reads at slots.

    The call's ``input_token_ids`` hold its canvas rows back to back. Slot
    ``i`` is canvas token ``slot_tokens[i]`` of that sequence, slots
    increase along it, and slot ``i`` reads the
    candidate ids ``candidate_ids[candidate_offsets[i]:candidate_offsets[i +
    1]]``. The call reports each candidate's natural-log probability under
    the log-softmax over the full vocabulary of the logits at its slot, in
    ``candidate_ids`` order (``RequestOutput.candidate_logprobs``).
    """

    slot_tokens: tuple[int, ...]
    candidate_offsets: tuple[int, ...]
    candidate_ids: tuple[int, ...]

    def to_mapping(self) -> dict[str, object]:
        """Serialize the readout's slot and candidate columns."""
        return {
            "slot_tokens": list(self.slot_tokens),
            "candidate_offsets": list(self.candidate_offsets),
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class VisionInput:
    """One image block of a context prefill.

    The block writes the vision-encoder product ``feature`` into KV as one
    attention block, before the call's input token ``offset``: the call's
    context is its input tokens ``[0, offset)``, then the block, then the
    tokens from ``offset`` on. Blocks sharing an offset follow in list
    order.
    """

    offset: int
    feature: tensor.TensorRef

    def to_mapping(self) -> dict[str, object]:
        """Serialize an image block's offset and feature."""
        return {"offset": self.offset, "feature": self.feature.to_mapping()}


@dataclass(frozen=True, slots=True)
class SamplingState:
    """Canonical branch-local token processor inputs for one call.

    Token ids in every field must be strictly increasing uint32 values.
    Native `Call.validate` checks their ordering before execution.
    Penalty token counts are not carried here: they are a device-resident
    accepted base plus bounded deltas folded when sampling accepts tokens,
    so no host token history participates in a successor's sampling input.
    """

    # Sampling whitelist; None means unrestricted, an empty tuple means
    # none allowed.
    allowed_token_ids: tuple[int, ...] | None = None
    suppressed_token_ids: tuple[int, ...] = ()
    finish_token_ids: tuple[int, ...] = ()
    transition_token_ids: tuple[int, ...] = ()
    force_finish: bool = False

    def to_mapping(self) -> dict[str, object]:
        """Serialize sampler constraints into their wire mapping."""
        return {
            "allowed_token_ids": None
            if self.allowed_token_ids is None
            else list(self.allowed_token_ids),
            "suppressed_token_ids": list(self.suppressed_token_ids),
            "finish_token_ids": list(self.finish_token_ids),
            "transition_token_ids": list(self.transition_token_ids),
            "force_finish": self.force_finish,
        }
