"""Causal video convolution input padding.

A causal 3D convolution over ``[batch, channels, frames, height, width]``
values reads them padded on both sides of height and width and preceded by
zero frames, so no output frame depends on a later input frame.
``frame_pad`` produces that padded input in one pass on CUDA, reading any
strided view of the source; other devices compute it with ``F.pad``. Both
give the same values bit for bit.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from uniserve_kernels.triton import require_kernel


def frame_pad(
    values: torch.Tensor,
    padding: tuple[int, int, int, int, int],
    *,
    mode: str = "reflect",
) -> torch.Tensor:
    """Pad ``[batch, channels, frames, height, width]`` values causally.

    ``padding`` is ``(left, right, top, bottom, front)``: columns and rows
    padded on each side in ``mode`` (``reflect`` or ``replicate``), then
    ``front`` zero frames before the first frame. Reflection mirrors once,
    so each reflected side must be shorter than its extent. Returns a new
    contiguous tensor.

    Raises:
        ValueError: when the padding or values are malformed, or on CUDA
            when the kernel cannot take the operands.
    """
    if len(padding) != 5 or any(
        type(value) is not int or value < 0 for value in padding
    ):
        raise ValueError(
            "frame padding is (left, right, top, bottom, front) nonnegative "
            "integers"
        )
    if mode not in ("reflect", "replicate"):
        raise ValueError("frame padding mode must be reflect or replicate")
    if values.ndim != 5:
        raise ValueError(
            "frame padding takes [batch, channels, frames, height, width]"
        )

    left, right, top, bottom, front = padding
    batch, channels, frames, height, width = values.shape
    # The kernel mirrors each coordinate once, the extent F.pad accepts.
    if mode == "reflect" and (
        max(left, right) >= width or max(top, bottom) >= height
    ):
        raise ValueError("reflected padding must be shorter than its extent")

    out = torch.empty(
        (
            batch,
            channels,
            frames + front,
            height + top + bottom,
            width + left + right,
        ),
        dtype=values.dtype,
        device=values.device,
    )
    if values.is_cuda:
        from uniserve_kernels import frame

        require_kernel(
            "frame_pad", frame.unsupported(values, out), values=values
        )
        frame.pad(
            values, out, padding=(top, left, front), reflect=mode == "reflect"
        )
        return out

    if left or right or top or bottom:
        values = F.pad(values, (left, right, top, bottom, 0, 0), mode=mode)
    if front:
        values = F.pad(values, (0, 0, 0, 0, front, 0))
    return out.copy_(values)
