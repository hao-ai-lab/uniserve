"""One numerical input view and the aligned rows used to construct it.

Execution code describes each request's contribution to a call as a row
(``InputRow`` and its subclasses here, in ``diffusion_inputs`` and in
``image_inputs``). Runners pack a homogeneous group of rows into their
fixed backing and evaluate the resulting ``InputBatch``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.diffusion.canvas import CanvasSampling, CanvasState
from uniserve.model import CanvasInput
from uniserve_worker._uniserve_ipc import AttentionRow as AttentionRow
from uniserve_worker._uniserve_ipc import CanvasRow as CanvasRow
from uniserve_worker._uniserve_ipc import CanvasStepRow as CanvasStepRow
from uniserve_worker._uniserve_ipc import InputBatch as InputBatch
from uniserve_worker._uniserve_ipc import InputRow as InputRow
from uniserve_worker._uniserve_ipc import TokenRow as TokenRow


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
class ReadoutInput:
    """Input canvases of one token-denoising call and their candidate reads.

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
class CanvasStepInput:
    """Prepared canvas steps of one numerical call.

    ``canvas`` packs every row's resident canvas back to back, with its
    self-conditioning embeddings. ``state`` is the rows' gathered sampler
    state, whose canvas and self-conditioning rows ``canvas`` reads, and
    ``views`` the same input tensors by ``CanvasSlots`` field. ``slots``
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
