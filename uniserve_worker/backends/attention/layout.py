"""Shared q/k/v layout normalization for paged attention backends.

A kernel declares the layout it consumes via :class:`QKVLayout`. ``normalize_to``
reshapes an incoming q/k/v tensor into that layout and returns a
:class:`LayoutRestore` whose ``apply`` undoes the reshape on the kernel output.
The reshape, transpose, and squeeze operations define the canonical tensor
forms consumed by every paged attention backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

__all__ = [
    "QKVLayout",
    "LayoutRestore",
    "normalize_to",
    "normalize_kv",
]


class QKVLayout(Enum):
    """Kernel-consumed q/k/v layout.

    ``BLHD`` is ``[batch, seq, heads, dim]`` (flash-attn / fa4_cute kvcache
    kernels). ``BHD`` is ``[batch, heads, dim]`` with exactly one token per row
    (flashinfer paged decode).
    """

    BLHD = "blhd"
    BHD = "bhd"


@dataclass(frozen=True)
class LayoutRestore:
    """Inverse of a :func:`normalize_to` reshape, applied to the kernel output."""

    layout: QKVLayout
    # Source rank of the tensor that was normalized (3 or 4).
    source_ndim: int

    def apply(self, out: torch.Tensor) -> torch.Tensor:
        """Restore normalized attention output to the caller’s original rank and head layout."""

        if self.layout is QKVLayout.BLHD:
            if self.source_ndim == 3:
                return out.squeeze(0)
            return out.transpose(1, 2).contiguous()
        # QKVLayout.BHD
        if self.source_ndim == 3:
            return out
        return out.unsqueeze(2).contiguous()


def normalize_to(tensor: torch.Tensor, target: QKVLayout) -> tuple[torch.Tensor, LayoutRestore]:
    """Normalize ``tensor`` into ``target`` layout for a paged kernel.

    Returns the normalized tensor and the :class:`LayoutRestore` that inverts the
    transform. Raises ``ValueError`` for an unsupported source rank.
    """

    if target is QKVLayout.BLHD:
        if tensor.ndim == 3:
            return tensor.unsqueeze(0).contiguous(), LayoutRestore(target, 3)
        if tensor.ndim == 4:
            return tensor.transpose(1, 2).contiguous(), LayoutRestore(target, 4)
        raise ValueError("paged q/k/v must be [L,H,D] or [B,H,L,D]")
    # QKVLayout.BHD
    if tensor.ndim == 3:
        return tensor.contiguous(), LayoutRestore(target, 3)
    if tensor.ndim == 4:
        if int(tensor.shape[2]) != 1:
            raise ValueError("paged decode only supports one token per row")
        return tensor.transpose(1, 2).squeeze(1).contiguous(), LayoutRestore(target, 4)
    raise ValueError("paged q/k/v must be [1,H,D] or [B,H,1,D]")


def normalize_kv(
    tensor: torch.Tensor,
    target: QKVLayout,
    *,
    contiguous: bool = True,
) -> torch.Tensor:
    """Normalize a current-K/V tensor into ``target``; no restore is needed.

    Paged cache writers can set ``contiguous=False`` when they accept packed
    head/dimension rows with a larger leading stride.
    """

    if contiguous:
        normalized, _ = normalize_to(tensor, target)
        return normalized
    if target is QKVLayout.BHD:
        if tensor.ndim == 3:
            return tensor
        if tensor.ndim == 4:
            if int(tensor.shape[2]) != 1:
                raise ValueError("paged decode only supports one token per row")
            return tensor.transpose(1, 2).squeeze(1)
    if target is QKVLayout.BLHD:
        if tensor.ndim == 3:
            return tensor.unsqueeze(0)
        if tensor.ndim == 4:
            return tensor.transpose(1, 2)
    raise ValueError("paged q/k/v must use a supported rank")
