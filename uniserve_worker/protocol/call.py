"""Call descriptors the engine submits to a worker rank.

A `Call` is one call for one request: its identity, the coordinates it
executes at, its kind (`ForwardMode`, `MediaCall`, or `TransferMode`), the
resource `Bounds` the scheduler reserved, its tensor dataflow, and its scalar
side inputs. The wire contract and its validators live in the Rust
`uniserve_worker_ipc` crate, whose `Call::validate` checks every call before
it reaches a rank.

The PyO3 transport (`crates/worker-ipc-py`) decodes and validates a batch in
Rust, then calls the constructors of `Call`, `CallCoordinates`, `Bounds`,
and `SamplingState` positionally in field-declaration order, so their
field order is part of that contract (`ImageParams` is built from keyword
arguments).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from uniserve.model import DEFAULT_COMPONENT
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol import identity, tensor


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
    """A concrete media reading, encoder, diffusion, or decoder call."""

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
    *ForwardMode,
    *MediaCall,
    *TransferMode,
)


def call(value: object, where: str) -> CallKind:
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
    raise invalid_descriptor(f"{where} is not a supported call")


class CallStatus(StrEnum):
    """Terminal status of one call's completion.

    `PREDICATED` means the call's device predicate disabled it: the call did
    not run and reports the request's accepted coordinates.
    """

    OK = "ok"
    PREDICATED = "predicated"
    ERROR = "error"


# Computations that advance a request's accepted progress, as the schema
# states them. `RequestPool.predecessors` chains a request's calls from its
# latest state-advancing call (on the media path only state-advancing calls
# join the chain), and `RequestPool.apply_result` moves the request's
# `state_call_id` only on an OK result of one of these.
_STATE_ADVANCING_WORK = frozenset(
    {
        *ForwardMode,
        MediaCall.LATENT_PREPARATION,
        MediaCall.DENOISING,
    }
)

_COMPUTATION_BY_VALUE = {member.value: member for member in CALL_KINDS}


@dataclass(frozen=True, slots=True)
class ImageParams:
    """Image-generation controls carried by a request's admission.

    Covers the diffusion schedule (steps, timestep shift), classifier-free
    guidance scales and renormalization, output dimensions, and the negative
    prompt.
    `NewRequest.image` holds one; the Rust decoder validates its ranges.
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


@dataclass(frozen=True, slots=True)
class Bounds:
    """Resource ceilings the scheduler reserves before a call runs.

    Zero means the call may use none of that resource. The Rust
    `Call::validate` checks `max_tokens` against the input token ids, and
    `Batch::validate` checks `max_transfer_bytes` against installed KV
    transfers.
    """

    max_tokens: int = 0
    max_completion_bytes: int = 0
    max_transfer_bytes: int = 0


@dataclass(frozen=True, slots=True)
class CallCoordinates:
    """The coordinates a call executes at, as the engine states them.

    A rank would otherwise chain these from the calls that preceded it. The
    engine holds the request state they come from, so it sends them. All
    lengths count tokens.
    """

    # Position of this call's first token in the request's logical sequence.
    logical_position: int = 0
    # Tokens whose KV a numerical call may attend to at submission.
    kv_visible_len: int = 0
    # Tokens whose KV is initialized at submission; never below the visible
    # extent, and above it while a verifier's rejected drafts stay initialized.
    kv_computed_len: int = 0


@dataclass(frozen=True, slots=True)
class CanvasStep:
    """One denoising step of a block-diffusion request's resident canvas.

    The worker's launch configuration (``WorkerConfig.canvas_sampling``)
    fixes the sampling; the worker holds the canvas and its sampler state in
    the request's slot.
    ``block`` counts the blocks already committed to the request's context
    and ``step`` the steps already run on this canvas; step zero starts the
    canvas from random tokens. The step that stops the canvas reports its
    argmax tokens as the completion's committed tokens.
    """

    block: int
    step: int


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


@dataclass(frozen=True, slots=True)
class Call:
    """One immutable call with its identity, dataflow, and limits."""

    request_key: identity.RequestKey
    call_id: identity.CallId
    # Coordinates this call executes at, stated by the engine.
    coordinates: CallCoordinates
    kind: CallKind
    bounds: Bounds
    # Name of the worker model component that executes this call.
    component: str = DEFAULT_COMPONENT

    # Tensor dataflow: generic inputs/outputs plus role-specific endpoints.
    inputs: tuple[tensor.TensorRef, ...] = ()
    outputs: tuple[tensor.TensorRef, ...] = ()
    # Source token scalar relayed between components by a tensor transfer.
    token_input: tensor.TensorRef | None = None
    # Sampled token and continuation bit packed into one int64 scalar (see
    # `tagged_token_values` in `uniserve_worker.sampling.sampler`).
    token_output: tensor.TensorRef | None = None
    # Image blocks a context prefill injects, in context order.
    vision_inputs: tuple[VisionInput, ...] = ()
    latent_feature_input: tensor.TensorRef | None = None
    encoder_output: tensor.TensorRef | None = None
    latent_input: tensor.TensorRef | None = None
    latent_output: tensor.TensorRef | None = None
    image_input: tensor.TensorRef | None = None
    image_output: tensor.TensorRef | None = None
    # Boolean device completion consumed by a dependent call.
    completion_output: tensor.TensorRef | None = None
    # Boolean device decision that selects an image transition.
    transition_output: tensor.TensorRef | None = None
    # Device-resident scalar that gates execution (u8 flag or packed i64
    # continuation).
    predicate: tensor.TensorRef | None = None

    # Scalar side-channel inputs carried on the wire rather than as tensors.
    # Index of the image a latent preparation initializes, positive and
    # assigned by the engine; the noise seed derives from the request's
    # admitted image seed and this index (`flow_noise_seed`).
    image_index: int | None = None
    # Per-call sampler constraints. The token executor treats None as an empty
    # `SamplingState` and merges these finish ids with the admission's.
    sampling_state: SamplingState | None = None
    # Host-known prompt, draft, or decode input tokens in model input order,
    # or a token-denoising call's canvas rows back to back; empty when a
    # continuation reads its predecessor's device token.
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
    # Candidate slots of a token-denoising call; None for every other call.
    readout: Readout | None = None
    # Step a token-denoising call runs on its request's generation canvas.
    canvas: CanvasStep | None = None

    def tensor_inputs(self) -> tuple[tensor.TensorRef, ...]:
        """Return the tensor inputs of the call, excluding its predicate.

        Includes the token and latent inputs, which `buffer_inputs` omits.
        """
        return (
            *self.inputs,
            *(
                value
                for value in (
                    self.token_input,
                    *(block.feature for block in self.vision_inputs),
                    self.latent_feature_input,
                    self.latent_input,
                    self.image_input,
                )
                if value is not None
            ),
        )

    def tensor_outputs(self) -> tuple[tensor.TensorRef, ...]:
        """Return every tensor output this call produces.

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
                    *(block.feature for block in self.vision_inputs),
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


@dataclass(frozen=True, slots=True)
class SamplingState:
    """Canonical branch-local token processor inputs for one call.

    Token ids in every field are strictly increasing uint32 values; the Rust
    `Call::validate` rejects any other ordering.
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
