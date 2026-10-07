"""Call descriptors the engine submits to a worker rank.

A `Call` is one computation for one request: its identity, the coordinates it
executes at, its kind (`ForwardMode`, `MediaCall`, or `TransferMode`), the
resource `Bounds` the scheduler reserved, its tensor dataflow, and its scalar
side inputs. The wire contract and its validators live in the Rust
`uniserve_worker_ipc` crate; the `validate` methods here are a separate Python
implementation of call-local checks.

Records reach Python on two paths. The PyO3 transport (`crates/worker-ipc-py`)
decodes and validates a batch in Rust, then calls the constructors of `Call`,
`CallCoordinates`, `Bounds`, `Rng`, and `SamplingState` positionally in
field-declaration order, so their field order is part of that contract
(`ImageParams` is built from keyword arguments). None of those five classes
defines a `__post_init__`, so on that path `Call.validate`,
`CallCoordinates.validate`, and `SamplingState.validate` never run; the Rust
`Call::validate` covered them. `Call.from_mapping` runs all three.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from uniserve import sampling
from uniserve.model import DEFAULT_COMPONENT
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol import identity, tensor
from uniserve_worker.protocol.validation import (
    _bool,
    _enum,
    _float,
    _ints,
    _map,
    _nonnegative,
    _optional_uint,
    _pair,
    _seq,
    _str,
    _uint,
    _uints,
)


class ForwardMode(StrEnum):
    """The numerical mode of one homogeneous autoregressive call."""

    PREFILL = "prefill"
    DECODE = "decode"
    VERIFY = "verify"


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
    KV_PUBLISH = "kv_publish"
    KV_INSTALL = "kv_install"


CallKind: TypeAlias = ForwardMode | MediaCall | TransferMode

# Every call kind a worker may execute, in the order of the Rust
# `CallKind::ALL`. Worker reports list supported calls in this order.
CALL_KINDS: tuple[CallKind, ...] = (
    ForwardMode.PREFILL,
    ForwardMode.DECODE,
    ForwardMode.VERIFY,
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


# Computations that advance a request's accepted progress; mirrors the Rust
# `CallKind::advances_state`. `RequestPool.predecessors` chains a request's
# calls from its latest state-advancing call (on the media path only
# state-advancing calls join the chain), and `RequestPool.apply_result` moves
# the request's `state_call_id` only on an OK result of one of these.
_STATE_ADVANCING_WORK = frozenset(
    {
        ForwardMode.PREFILL,
        ForwardMode.DECODE,
        ForwardMode.VERIFY,
        MediaCall.LATENT_PREPARATION,
        MediaCall.DENOISING,
    }
)

_COMPUTATION_BY_VALUE = {member.value: member for member in CALL_KINDS}


def _sampling_params_from_mapping(
    value: object, where: str = "sampling"
) -> sampling.SamplingParams:
    """Decode wire sampling parameters, applying defaults for absent fields.

    Raises:
        WorkerError: A field is malformed or `SamplingParams` rejects the
            combination.
    """
    data = _map(value, where)
    # Field decoders raise descriptor errors directly; only the parameter
    # invariants checked by SamplingParams raise ValueError.
    try:
        return sampling.SamplingParams(
            temperature=_float(
                data.get("temperature", 0.0), f"{where}.temperature"
            ),
            top_k=_uint(data.get("top_k", 0), f"{where}.top_k"),
            top_p=_float(data.get("top_p", 1.0), f"{where}.top_p"),
            ignore_eos=_bool(
                data.get("ignore_eos", False), f"{where}.ignore_eos"
            ),
            seed=_optional_uint(data.get("seed"), f"{where}.seed"),
            min_p=_float(data.get("min_p", 0.0), f"{where}.min_p"),
            repetition_penalty=_float(
                data.get("repetition_penalty", 1.0),
                f"{where}.repetition_penalty",
            ),
            frequency_penalty=_float(
                data.get("frequency_penalty", 0.0), f"{where}.frequency_penalty"
            ),
            presence_penalty=_float(
                data.get("presence_penalty", 0.0), f"{where}.presence_penalty"
            ),
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
            return_logprobs=_bool(
                data.get("return_logprobs", False), f"{where}.return_logprobs"
            ),
            n_logprobs=_uint(data.get("n_logprobs", 0), f"{where}.n_logprobs"),
            return_prompt_logprobs=_bool(
                data.get("return_prompt_logprobs", False),
                f"{where}.return_prompt_logprobs",
            ),
            n_prompt_logprobs=_uint(
                data.get("n_prompt_logprobs", 0), f"{where}.n_prompt_logprobs"
            ),
            logprob_token_ids=_uints(
                data.get("logprob_token_ids", ()), f"{where}.logprob_token_ids"
            ),
            bad_words_ids=tuple(
                _uints(item, f"{where}.bad_words_ids[{index}]")
                for index, item in enumerate(
                    _seq(
                        data.get("bad_words_ids", ()), f"{where}.bad_words_ids"
                    )
                )
            ),
            allowed_token_ids=(
                None
                if data.get("allowed_token_ids") is None
                else _uints(
                    data["allowed_token_ids"], f"{where}.allowed_token_ids"
                )
            ),
            typical_p=_float(data.get("typical_p", 1.0), f"{where}.typical_p"),
            forced_token_ids=_uints(
                data.get("forced_token_ids", ()), f"{where}.forced_token_ids"
            ),
        )
    except ValueError as error:
        raise invalid_descriptor(f"{where}.{error}") from error


def _sampling_params_to_mapping(
    params: sampling.SamplingParams,
) -> dict[str, object]:
    """Serialize sampling parameters into their wire mapping."""
    return {
        "temperature": params.temperature,
        "top_k": params.top_k,
        "top_p": params.top_p,
        "ignore_eos": params.ignore_eos,
        "seed": params.seed,
        "min_p": params.min_p,
        "repetition_penalty": params.repetition_penalty,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
        "logit_bias": [list(value) for value in params.logit_bias],
        "min_tokens": params.min_tokens,
        "return_logprobs": params.return_logprobs,
        "n_logprobs": params.n_logprobs,
        "return_prompt_logprobs": params.return_prompt_logprobs,
        "n_prompt_logprobs": params.n_prompt_logprobs,
        "logprob_token_ids": list(params.logprob_token_ids),
        "bad_words_ids": [list(value) for value in params.bad_words_ids],
        "allowed_token_ids": (
            None
            if params.allowed_token_ids is None
            else list(params.allowed_token_ids)
        ),
        "typical_p": params.typical_p,
        "forced_token_ids": list(params.forced_token_ids),
    }


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
    encoder, latent, and image outputs; `Batch.validate` checks
    `max_transfer_bytes` against installed KV transfers.
    """

    max_tokens: int = 0
    max_kv_units: int = 0
    max_latent_bytes: int = 0
    max_completion_bytes: int = 0
    max_transfer_bytes: int = 0

    @classmethod
    def from_mapping(cls, value: object, where: str = "bounds") -> Bounds:
        """Parse scheduler-enforced resource ceilings for one call."""
        data = _map(value, where)
        return cls(
            max_tokens=_uint(data.get("max_tokens"), f"{where}.max_tokens"),
            max_kv_units=_uint(
                data.get("max_kv_units"), f"{where}.max_kv_units"
            ),
            max_latent_bytes=_uint(
                data.get("max_latent_bytes"), f"{where}.max_latent_bytes"
            ),
            max_completion_bytes=_uint(
                data.get("max_completion_bytes"),
                f"{where}.max_completion_bytes",
            ),
            max_transfer_bytes=_uint(
                data.get("max_transfer_bytes"), f"{where}.max_transfer_bytes"
            ),
        )

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

    @classmethod
    def from_mapping(cls, value: object, where: str = "rng") -> Rng:
        """Parse deterministic random seed, semantic offset, and draw layout."""
        data = _map(value, where)
        return cls(
            seed=_uint(data.get("seed"), f"{where}.seed"),
            semantic_index_base=_uint(
                data.get("semantic_index_base"), f"{where}.semantic_index_base"
            ),
            draw_layout=_enum(
                DrawLayout, data.get("draw_layout"), f"{where}.draw_layout"
            ),
        )

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

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "coordinates"
    ) -> CallCoordinates:
        """Parse the coordinates a call states."""
        data = _map(value, where)
        coordinates = cls(
            logical_position=_uint(
                data.get("logical_position"), f"{where}.logical_position"
            ),
            kv_visible_len=_uint(
                data.get("kv_visible_len"), f"{where}.kv_visible_len"
            ),
            kv_computed_len=_uint(
                data.get("kv_computed_len"), f"{where}.kv_computed_len"
            ),
            flow_step=_uint(data.get("flow_step"), f"{where}.flow_step"),
        )
        coordinates.validate()
        return coordinates

    def validate(self) -> None:
        """Enforce the containment the two KV extents must satisfy."""
        if self.kv_visible_len > self.kv_computed_len:
            raise invalid_descriptor(
                "call coordinates place visible KV beyond the computed extent"
            )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the coordinates a call states."""
        return {
            "logical_position": self.logical_position,
            "kv_visible_len": self.kv_visible_len,
            "kv_computed_len": self.kv_computed_len,
            "flow_step": self.flow_step,
        }


@dataclass(frozen=True, slots=True)
class Call:
    """One immutable computation with its identity, dataflow, and limits."""

    request_key: identity.RequestKey
    call_id: identity.CallId
    # Coordinates this call executes at, stated by the engine.
    coordinates: CallCoordinates
    kind: CallKind
    bounds: Bounds
    # Name of the worker model component that executes this computation.
    component: str = DEFAULT_COMPONENT

    # Tensor dataflow: generic inputs/outputs plus role-specific endpoints.
    inputs: tuple[tensor.TensorRef, ...] = ()
    outputs: tuple[tensor.TensorRef, ...] = ()
    # Source token scalar relayed between components by a tensor transfer.
    token_input: tensor.TensorRef | None = None
    # Sampled token and continuation bit packed into one int64 scalar (see
    # `tagged_token_values` in `uniserve_worker.sampling.sampler`).
    token_output: tensor.TensorRef | None = None
    vision_input: tensor.TensorRef | None = None
    latent_feature_input: tensor.TensorRef | None = None
    encoder_output: tensor.TensorRef | None = None
    latent_input: tensor.TensorRef | None = None
    latent_output: tensor.TensorRef | None = None
    image_input: tensor.TensorRef | None = None
    image_output: tensor.TensorRef | None = None
    # Boolean device completion consumed by a dependent computation.
    completion_output: tensor.TensorRef | None = None
    # Boolean device decision that selects an image transition.
    transition_output: tensor.TensorRef | None = None
    # Device-resident scalar that gates execution (u8 flag or packed i64
    # continuation).
    predicate: tensor.TensorRef | None = None

    # Scalar side-channel inputs carried on the wire rather than as tensors.
    rng: Rng | None = None
    # Per-call sampler constraints. The token executor treats None as an empty
    # `SamplingState` and merges these finish ids with the admission's.
    sampling_state: SamplingState | None = None
    # Host-known prompt, draft, or decode input tokens in model input order;
    # empty when a continuation reads its predecessor's device token.
    input_token_ids: tuple[int, ...] = ()
    # Encoded source image payload for a vision or latent encoding call.
    input_image: str | None = None

    # KV cache transfer endpoints, as persistent buffer identities.
    kv_input: identity.BufferId | None = None
    kv_output: identity.BufferId | None = None
    # Acknowledgment slots of the ranks that read this call's products, stated
    # by the engine; a published product retires once each has acknowledged
    # it. This rank's own slot is never listed.
    consumer_slots: tuple[int, ...] = ()

    def tensor_inputs(self) -> tuple[tensor.TensorRef, ...]:
        """Return the tensor inputs of the computation, excluding its predicate.

        Includes the token and latent inputs, which `buffer_inputs` omits.
        """
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

    def tensor_outputs(self) -> tuple[tensor.TensorRef, ...]:
        """Return every tensor output this computation produces.

        Includes the device scalar outputs (token, completion, transition)
        and the latent output, which `buffer_outputs` omits.
        """
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

    def buffer_inputs(self) -> tuple[tensor.TensorRef, ...]:
        """Return inputs that require persistent destination storage."""
        return (
            *self.inputs,
            *(
                value
                for value in (
                    self.vision_input,
                    self.latent_feature_input,
                    self.image_input,
                )
                if value is not None
            ),
        )

    def buffer_outputs(self) -> tuple[tensor.TensorRef, ...]:
        """Return outputs backed by scheduler-allocated persistent buffers.

        `Batch.validate` requires a `BufferAllocation` for each of these that
        is at least as large as the output's `max_bytes`.
        """
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
        """Indicate whether the call advances accepted request progress."""
        return self.kind in _STATE_ADVANCING_WORK

    def validate(self) -> None:
        """Enforce the call's identity, bound, dataflow, and predicate rules.

        `from_mapping` calls this; records the PyO3 transport builds were
        validated by the Rust `Call::validate` instead. `rng` is not checked
        here: the executors that consume it validate it against the admission.

        Raises:
            WorkerError: Any rule is violated.
        """
        # Identity, coordinates, component, and kind.
        if self.call_id.batch_id < 1:
            raise invalid_descriptor("call id must be positive")
        self.coordinates.validate()
        if not isinstance(self.component, str) or not self.component:
            raise invalid_descriptor("call component must not be empty")
        if self.kind not in CALL_KINDS:
            raise invalid_descriptor("call requires a valid computation tag")

        # Token and sampling inputs.
        if len(self.input_token_ids) > self.bounds.max_tokens:
            raise invalid_descriptor(
                "input token count exceeds the computation token bound"
            )
        if any(
            token < 0 or token > 0xFFFFFFFF for token in self.input_token_ids
        ):
            raise invalid_descriptor("input token id is outside uint32")
        if self.sampling_state is not None:
            self.sampling_state.validate()
        if self.input_image is not None and (
            not isinstance(self.input_image, str)
            or not self.input_image
            or self.kind
            not in {
                MediaCall.VISION_ENCODING,
                MediaCall.LATENT_ENCODING,
            }
            or self.image_input is not None
        ):
            raise invalid_descriptor(
                "encoded image requires an image encoder without another "
                "image source"
            )

        # KV cache transfer endpoints. A KV output's index shares the output
        # index space with the tensor outputs checked below.
        output_indices: set[int] = set()
        publishes_kv = self.kind in {
            TransferMode.KV_PUBLISH,
            TransferMode.KV_INSTALL,
        }
        if (self.kv_output is not None) != publishes_kv:
            raise invalid_descriptor(
                "KV publication or installation requires one cache output "
                "identity"
            )
        if self.kv_output is not None:
            if (
                self.kv_output.owner != self.request_key
                or self.kv_output.producer_call_id != self.call_id
            ):
                raise invalid_descriptor(
                    "KV output is not owned by its producing computation"
                )
            output_indices.add(self.kv_output.output_index)
        if self.kv_input is not None and (
            self.kv_input.owner != self.request_key
            or self.kind
            not in {
                TransferMode.KV_INSTALL,
                MediaCall.LATENT_PREPARATION,
                MediaCall.DENOISING,
            }
        ):
            raise invalid_descriptor(
                "KV input is incompatible with its computation or request"
            )
        if self.kind is TransferMode.KV_INSTALL and self.kv_input is None:
            raise invalid_descriptor(
                "KV installation requires a source publication"
            )

        # Output ownership, generation, and capacity.
        for product in self.tensor_outputs():
            if (
                product.request_key != self.request_key
                or product.producer_call_id != self.call_id
            ):
                raise invalid_descriptor(
                    "an output product is not owned by its producing call"
                )
            if product.generation < 1:
                raise invalid_descriptor(
                    "an output product has no logical generation"
                )
            if product.output_index in output_indices:
                raise invalid_descriptor("call repeats an output index")
            output_indices.add(product.output_index)
        for media_output in (
            self.encoder_output,
            self.latent_output,
            self.image_output,
        ):
            if (
                media_output is not None
                and media_output.max_bytes > self.bounds.max_latent_bytes
            ):
                raise invalid_descriptor(
                    "image or trajectory output exceeds its declared byte "
                    "capacity"
                )

        # Scalar relay endpoints: the token input and output are each one
        # int64 element (only a tensor transfer consumes a token input), and
        # the completion and transition outputs are each one uint8 element.
        if self.token_input is not None and (
            self.kind is not TransferMode.TENSOR
            or self.token_input.dtype is not tensor.DType.I64
            or self.token_input.shape_bound.max_elements != 1
        ):
            raise invalid_descriptor(
                "token transfer input requires a tensor transfer of one "
                "int64 element"
            )
        for output in (self.token_output,):
            if output is not None and (
                output.dtype is not tensor.DType.I64
                or output.shape_bound.max_elements != 1
            ):
                raise invalid_descriptor(
                    "device token relay requires one int64 element"
                )
        for output in (self.completion_output, self.transition_output):
            if output is not None and (
                output.dtype is not tensor.DType.U8
                or output.shape_bound.max_elements != 1
            ):
                raise invalid_descriptor(
                    "device completion requires one uint8 element"
                )

        # Inputs and predicate must belong to this request's lineage, except
        # admitted cross-request encoder products (vision, latent features).
        for product in self.tensor_inputs():
            if product.request_key != self.request_key and product not in (
                self.vision_input,
                self.latent_feature_input,
            ):
                raise invalid_descriptor(
                    "request-local tensor belongs to another request lineage"
                )
        if self.predicate is not None:
            if self.predicate.request_key != self.request_key:
                raise invalid_descriptor(
                    "computation predicate belongs to another request lineage"
                )
            if (
                self.predicate.dtype not in {tensor.DType.U8, tensor.DType.I64}
                or self.predicate.shape_bound.max_elements != 1
            ):
                raise invalid_descriptor(
                    "device predicate requires a boolean or packed "
                    "continuation scalar"
                )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "call",
    ) -> Call:
        """Parse a call mapping and run `validate` on the result.

        Raises:
            WorkerError: A field is malformed or `validate` rejects the call.
        """
        data = _map(value, where)
        get = data.get

        request_key = identity.RequestKey.from_mapping(
            get("request_key"), f"{where}.request_key"
        )
        call_id = identity.CallId.from_mapping(
            get("call_id"), f"{where}.call_id"
        )
        coordinates = CallCoordinates.from_mapping(
            get("coordinates"), f"{where}.coordinates"
        )
        component = _str(get("component"), f"{where}.component")
        work = computation(get("code"), f"{where}.code")
        bounds = Bounds.from_mapping(get("bounds"), f"{where}.bounds")

        inputs = tuple(
            tensor.TensorRef.from_mapping(item, f"{where}.inputs[{index}]")
            for index, item in enumerate(
                _seq(get("inputs", ()), f"{where}.inputs")
            )
        )
        outputs = tuple(
            tensor.TensorRef.from_mapping(item, f"{where}.outputs[{index}]")
            for index, item in enumerate(
                _seq(get("outputs", ()), f"{where}.outputs")
            )
        )

        predicate_raw = get("predicate")
        if predicate_raw is None:
            predicate = None
        else:
            predicate = tensor.TensorRef.from_mapping(
                predicate_raw, f"{where}.predicate"
            )
        rng_raw = get("rng")
        if rng_raw is None:
            rng = None
        else:
            rng = Rng.from_mapping(rng_raw, f"{where}.rng")

        call = cls(
            request_key=request_key,
            call_id=call_id,
            coordinates=coordinates,
            component=component,
            kind=work,
            bounds=bounds,
            inputs=inputs,
            outputs=outputs,
            token_input=None
            if get("token_input") is None
            else tensor.TensorRef.from_mapping(
                get("token_input"), f"{where}.token_input"
            ),
            token_output=None
            if get("token_output") is None
            else tensor.TensorRef.from_mapping(
                get("token_output"), f"{where}.token_output"
            ),
            vision_input=None
            if get("vision_input") is None
            else tensor.TensorRef.from_mapping(
                get("vision_input"), f"{where}.vision_input"
            ),
            latent_feature_input=None
            if get("latent_feature_input") is None
            else tensor.TensorRef.from_mapping(
                get("latent_feature_input"), f"{where}.latent_feature_input"
            ),
            encoder_output=None
            if get("encoder_output") is None
            else tensor.TensorRef.from_mapping(
                get("encoder_output"), f"{where}.encoder_output"
            ),
            latent_input=None
            if get("latent_input") is None
            else tensor.TensorRef.from_mapping(
                get("latent_input"), f"{where}.latent_input"
            ),
            latent_output=None
            if get("latent_output") is None
            else tensor.TensorRef.from_mapping(
                get("latent_output"), f"{where}.latent_output"
            ),
            image_input=None
            if get("image_input") is None
            else tensor.TensorRef.from_mapping(
                get("image_input"), f"{where}.image_input"
            ),
            image_output=None
            if get("image_output") is None
            else tensor.TensorRef.from_mapping(
                get("image_output"), f"{where}.image_output"
            ),
            completion_output=None
            if get("completion_output") is None
            else tensor.TensorRef.from_mapping(
                get("completion_output"), f"{where}.completion_output"
            ),
            transition_output=None
            if get("transition_output") is None
            else tensor.TensorRef.from_mapping(
                get("transition_output"), f"{where}.transition_output"
            ),
            predicate=predicate,
            rng=rng,
            input_token_ids=tuple(
                _ints(get("input_token_ids"), "input_token_ids")
            ),
            kv_input=None
            if get("kv_input") is None
            else identity.BufferId.from_mapping(
                get("kv_input"), f"{where}.kv_input"
            ),
            kv_output=None
            if get("kv_output") is None
            else identity.BufferId.from_mapping(
                get("kv_output"), f"{where}.kv_output"
            ),
            consumer_slots=tuple(
                _uint(slot, f"{where}.consumer_slots[{index}]")
                for index, slot in enumerate(
                    _seq(get("consumer_slots", ()), f"{where}.consumer_slots")
                )
            ),
            input_image=(
                None
                if get("input_image") is None
                else _str(get("input_image"), "input_image")
            ),
            sampling_state=(
                None
                if get("sampling_state") is None
                else SamplingState.from_mapping(get("sampling_state"))
            ),
        )
        call.validate()
        return call

    def to_mapping(self) -> dict[str, object]:
        """Encode every call field into its wire mapping."""
        return {
            "request_key": self.request_key.to_mapping(),
            "call_id": self.call_id.to_mapping(),
            "coordinates": self.coordinates.to_mapping(),
            "component": self.component,
            "code": self.kind.value,
            "bounds": self.bounds.to_mapping(),
            "inputs": [product.to_mapping() for product in self.inputs],
            "outputs": [product.to_mapping() for product in self.outputs],
            "token_input": None
            if self.token_input is None
            else self.token_input.to_mapping(),
            "token_output": None
            if self.token_output is None
            else self.token_output.to_mapping(),
            "vision_input": None
            if self.vision_input is None
            else self.vision_input.to_mapping(),
            "latent_feature_input": None
            if self.latent_feature_input is None
            else self.latent_feature_input.to_mapping(),
            "encoder_output": None
            if self.encoder_output is None
            else self.encoder_output.to_mapping(),
            "latent_input": None
            if self.latent_input is None
            else self.latent_input.to_mapping(),
            "latent_output": None
            if self.latent_output is None
            else self.latent_output.to_mapping(),
            "image_input": None
            if self.image_input is None
            else self.image_input.to_mapping(),
            "image_output": None
            if self.image_output is None
            else self.image_output.to_mapping(),
            "completion_output": None
            if self.completion_output is None
            else self.completion_output.to_mapping(),
            "transition_output": None
            if self.transition_output is None
            else self.transition_output.to_mapping(),
            "predicate": None
            if self.predicate is None
            else self.predicate.to_mapping(),
            "rng": None if self.rng is None else self.rng.to_mapping(),
            "input_token_ids": list(self.input_token_ids),
            "input_image": self.input_image,
            "kv_input": None
            if self.kv_input is None
            else self.kv_input.to_mapping(),
            "kv_output": None
            if self.kv_output is None
            else self.kv_output.to_mapping(),
            "consumer_slots": list(self.consumer_slots),
            "sampling_state": (
                None
                if self.sampling_state is None
                else self.sampling_state.to_mapping()
            ),
        }


@dataclass(frozen=True, slots=True)
class SamplingState:
    """Canonical branch-local token processor inputs for one call.

    Token ids in every field must be strictly increasing uint32 values; both
    `validate` and the Rust `Call::validate` reject any other ordering.
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

    def validate(self) -> None:
        """Require every token set to be strictly increasing within uint32.

        An empty `allowed_token_ids` is valid and distinct from None.

        Raises:
            WorkerError: A token id is out of range or a set is not canonical.
        """
        for ids in (
            self.allowed_token_ids,
            self.suppressed_token_ids,
            self.finish_token_ids,
            self.transition_token_ids,
        ):
            if ids is None:
                continue
            if any(token < 0 or token > 0xFFFFFFFF for token in ids):
                raise invalid_descriptor(
                    "sampling-state token id is outside uint32"
                )
            if any(
                left >= right for left, right in zip(ids, ids[1:], strict=False)
            ):
                raise invalid_descriptor(
                    "sampling-state token ids are not canonical"
                )

    @classmethod
    def from_mapping(cls, value: object) -> SamplingState:
        """Parse sampler constraints without checking token range or order.

        `Call.validate` runs `validate` on the call's sampling state.
        """
        data = _map(value, "sampling_state")
        allowed = data.get("allowed_token_ids")
        return cls(
            allowed_token_ids=(
                None
                if allowed is None
                else tuple(_ints(allowed, "allowed_token_ids"))
            ),
            suppressed_token_ids=tuple(
                _ints(data.get("suppressed_token_ids"), "suppressed_token_ids")
            ),
            finish_token_ids=tuple(
                _ints(data.get("finish_token_ids"), "finish_token_ids")
            ),
            transition_token_ids=tuple(
                _ints(data.get("transition_token_ids"), "transition_token_ids")
            ),
            force_finish=_bool(data.get("force_finish", False), "force_finish"),
        )

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
