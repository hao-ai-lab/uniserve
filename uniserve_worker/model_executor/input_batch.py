"""One numerical input view and the aligned rows used to construct it.

Execution code describes each request's contribution to a call as a row
(``InputRow`` and its subclasses here, in ``diffusion_inputs`` and in
``image_inputs``). Runners stage a homogeneous group of rows into their
fixed backing and evaluate the resulting ``InputBatch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from uniserve.diffusion.canvas import CanvasSampling, CanvasState
from uniserve.model import CanvasInput
from uniserve_worker.protocol.call import ForwardMode, MediaCall
from uniserve_worker.sampling.metadata import TokenSelection

InputT = TypeVar("InputT")


def int64_bits(value: int) -> int:
    """Return the int64 whose two's-complement bits equal unsigned ``value``.

    Request seeds are unsigned 64-bit values, while device columns and
    ``torch.tensor(..., dtype=torch.int64)`` hold signed int64. Seeds of
    2**63 and above map to the negative int64 with the same bits; the
    sampler kernels read the column back as unsigned, so every seed keeps
    its Philox key.

    Raises:
        ValueError: ``value`` is not an unsigned 64-bit integer.
    """
    if not 0 <= value < 1 << 64:
        raise ValueError(f"seed {value} is not an unsigned 64-bit integer")
    return value - (1 << 64) if value >= 1 << 63 else value


@dataclass(frozen=True, slots=True)
class InputBatch(Generic[InputT]):
    """One typed numerical input and the worker's aligned output controls.

    ``request_pool_indices`` is a nonempty [rows] vector naming the request
    slot that receives each row's output. Sampled text calls (a
    ``ForwardMode`` other than ``TOKEN_DENOISING``, whose canvas rows select
    their slots themselves) carry one ``TokenSelection`` per row, and
    ``decode_force_finish``, when present, is a [rows] bool mask;
    ``__post_init__`` checks both.
    """

    forward_mode: ForwardMode | MediaCall
    inputs: InputT
    request_pool_indices: torch.Tensor
    token_selections: tuple[TokenSelection, ...] = ()
    decode_force_finish: torch.Tensor | None = None

    @property
    def row_count(self) -> int:
        return self.request_pool_indices.numel()

    @property
    def query_tokens(self) -> int | None:
        """Query tokens the batch computes, including any padding sequences.

        The sum of the attention input's host query lengths: every token a
        text row appends after its cached prefix, or every latent token a
        denoising row attends from. None when the input carries no attention
        sequences with host lengths, as encoder and decoder inputs do not.
        """
        attention = getattr(self.inputs, "attention", None)
        queries = getattr(attention, "queries", None)
        lengths = None if queries is None else queries.host
        return None if lengths is None else sum(lengths)

    def __post_init__(self):
        if self.request_pool_indices.ndim != 1 or self.row_count < 1:
            raise ValueError(
                "execution requires a nonempty vector of request slots"
            )
        if (
            isinstance(self.forward_mode, ForwardMode)
            and self.forward_mode is not ForwardMode.TOKEN_DENOISING
            and len(self.token_selections) != self.row_count
        ):
            raise ValueError(
                "text output selections must align with request slots"
            )
        if self.decode_force_finish is not None and (
            self.decode_force_finish.shape != self.request_pool_indices.shape
            or self.decode_force_finish.dtype != torch.bool
        ):
            raise ValueError(
                "decode completion controls must align with request slots"
            )


@dataclass(frozen=True, slots=True)
class InputRow:
    """One numerical input and the request slot receiving its output."""

    forward_mode: ForwardMode | MediaCall
    request_pool_idx: int = 0


@dataclass(frozen=True, slots=True)
class AttentionRow(InputRow):
    """Position and cache coordinates of one attention sequence."""

    positions: torch.Tensor | None = None
    # Cached prefix length in tokens that the query follows.
    seq_len: int = 0
    write_kv: bool = False
    causal: bool = True

    @property
    def query_tokens(self) -> int:
        """Return the query length this row appends after its cached prefix."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TokenRow(AttentionRow):
    """Token views or a resident request-slot continuation.

    Indexed decode borrows tokens and positions from DecodeState during input
    preparation and carries no duplicate per-row tensor views.
    """

    token_ids: torch.Tensor | None = None
    token_embeddings: torch.Tensor | None = None
    token_embedding_mask: torch.Tensor | None = None
    selection: TokenSelection | None = None
    decode_predicate: torch.Tensor | None = None
    decode_predicate_tagged: bool = False
    decode_force_finish: bool = False
    request_indexed_decode: bool = False

    @property
    def query_tokens(self) -> int:
        """Return the query length this row appends after its cached prefix.

        One for indexed decode; otherwise the number of ``token_ids``, or zero
        without them.
        """
        if self.request_indexed_decode:
            return 1
        if self.token_ids is not None:
            return int(self.token_ids.numel())
        return 0


@dataclass(frozen=True, slots=True)
class CanvasRow(AttentionRow):
    """One token canvas denoised over its request's cached prefix.

    The canvas attends non-causally to the request's first ``seq_len``
    cached tokens and to all of its own tokens, and writes no KV, so
    ``write_kv`` and ``causal`` stay false. ``token_ids`` and ``positions``
    are host int64 ``[canvas]`` vectors. The row reads the logits of its
    tokens ``slot_tokens``: slot ``i`` reports the log-probabilities of
    ``candidate_ids[candidate_offsets[i]:candidate_offsets[i + 1]]`` under
    the full-vocabulary softmax, so its output is FP32 ``[candidates]`` in
    slot, then candidate order.
    """

    token_ids: torch.Tensor | None = None
    slot_tokens: tuple[int, ...] = ()
    candidate_offsets: tuple[int, ...] = (0,)
    candidate_ids: tuple[int, ...] = ()

    def __post_init__(self):
        tokens = self.query_tokens
        offsets = self.candidate_offsets
        if (
            self.write_kv
            or self.causal
            or tokens < 1
            or self.positions is None
            or self.positions.shape[-1] != tokens
        ):
            raise ValueError(
                "a canvas row is a read-only noncausal token sequence"
            )
        if (
            not self.slot_tokens
            or len(offsets) != len(self.slot_tokens) + 1
            or offsets[0] != 0
            or offsets[-1] != len(self.candidate_ids)
            or any(end <= start for start, end in zip(offsets, offsets[1:]))
            or any(not 0 <= token < tokens for token in self.slot_tokens)
        ):
            raise ValueError(
                "canvas slots must lie in the row and each read candidates"
            )

    @property
    def query_tokens(self) -> int:
        """Return the canvas length."""
        return 0 if self.token_ids is None else int(self.token_ids.numel())


@dataclass(frozen=True, slots=True)
class ReadoutInput:
    """Staged canvases of one token-denoising call and their candidate reads.

    ``canvas`` packs every ``CanvasRow`` back to back. ``slot_tokens`` is
    int64 ``[slots]``: each slot's index into the packed canvas tokens, in
    row then slot order. ``candidates`` is int64 ``[slots, width]``, each
    slot's candidate ids with the row padded by its first candidate, and
    ``selection`` is int64 ``[candidates]``: the flat indices into
    ``candidates`` of every real candidate, in row, slot and candidate
    order. ``row_candidates`` holds each row's candidate count on the host.
    """

    canvas: CanvasInput
    slot_tokens: torch.Tensor
    candidates: torch.Tensor
    selection: torch.Tensor
    row_candidates: tuple[int, ...]

    @property
    def attention(self):
        """The canvas rows' attention input, which execution binds."""
        return self.canvas.attention


@dataclass(frozen=True, slots=True)
class CanvasStepRow(AttentionRow):
    """One denoising step of a request's resident generation canvas.

    The canvas lives in the request slot's sampler state, so the row carries
    no tokens: it attends non-causally to the request's first ``seq_len``
    cached tokens and to its ``canvas_length`` canvas tokens at host int64
    ``positions``, and writes no KV. ``block`` counts the blocks the request
    has committed and ``step`` the steps already run on this canvas; step
    zero starts the canvas. ``seed`` and ``sampling`` are the request's
    admitted seed and the sampler constants of its canvas sampling. The
    admitted seed is an unsigned 64-bit value; ``seed`` holds the int64 with
    the same two's-complement bits, which the device columns store and the
    sampler reinterprets as unsigned (see ``int64_bits``).
    """

    canvas_length: int = 0
    seed: int = 0
    block: int = 0
    step: int = 0
    sampling: CanvasSampling | None = None

    def __post_init__(self):
        if (
            self.write_kv
            or self.causal
            or self.canvas_length < 1
            or self.request_pool_idx < 1
            or self.positions is None
            or self.positions.shape[-1] != self.canvas_length
        ):
            raise ValueError(
                "a canvas step is a read-only noncausal canvas of its slot"
            )
        if (
            self.sampling is None
            or min(self.block, self.step) < 0
            or self.step >= self.sampling.steps
        ):
            raise ValueError(
                "a canvas step runs within its request's canvas sampling"
            )

    @property
    def query_tokens(self) -> int:
        """Return the canvas length."""
        return self.canvas_length


@dataclass(frozen=True, slots=True)
class CanvasStepInput:
    """Staged canvas steps of one numerical call.

    ``canvas`` packs every row's resident canvas back to back, with its
    self-conditioning embeddings. ``state`` is the rows' gathered sampler
    state, whose canvas and self-conditioning rows ``canvas`` reads, and
    ``views`` the same staged tensors by ``CanvasSlots`` field. ``slots``
    holds the rows' request slots as a device int64 ``[rows]`` vector,
    through which the stepped state returns to its slots, and ``sampling``
    each row's sampler constants. ``first`` is whether every row starts its
    canvas (step zero), whose self-conditioning signal is zero.
    """

    canvas: CanvasInput
    state: CanvasState
    views: dict[str, torch.Tensor]
    slots: torch.Tensor
    sampling: tuple[CanvasSampling, ...]
    first: bool = False

    @property
    def attention(self):
        """The canvas rows' attention input, which execution binds."""
        return self.canvas.attention
