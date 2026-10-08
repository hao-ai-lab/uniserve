"""Causal video convolution input padding.

A causal 3D convolution over ``[batch, channels, frames, height, width]``
values reads them padded on both sides of height and width and preceded by
zero frames, so no output frame depends on a later input frame.
``frame_pad`` produces that padded input in one pass on CUDA, reading any
strided view of the source and writing channel-first or channels-last
storage; other devices compute it with ``F.pad``. ``frame_norm_pad`` pads
the pre-activation of such a convolution, SiLU of every frame's group
normalization: on CUDA it reads the moments from the values and writes the
normalized, activated and padded input in one further pass. Each gives its
composition's values bit for bit.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from uniserve_kernels.triton import require_kernel


def _padded(
    values: torch.Tensor,
    padding: tuple[int, int, int, int, int],
    mode: str,
    memory_format: torch.memory_format,
) -> torch.Tensor:
    """Validate a causal padding of ``values`` and allocate its output."""
    if len(padding) != 5 or any(
        type(value) is not int or value < 0 for value in padding
    ):
        raise ValueError(
            "frame padding is (left, right, top, bottom, front) nonnegative "
            "integers"
        )
    if mode not in ("reflect", "replicate"):
        raise ValueError("frame padding mode must be reflect or replicate")
    if memory_format not in (torch.contiguous_format, torch.channels_last_3d):
        raise ValueError(
            "frame padding stores channel-first or channels-last values"
        )
    if values.ndim != 5:
        raise ValueError(
            "frame padding takes [batch, channels, frames, height, width]"
        )

    left, right, top, bottom, front = padding
    batch, channels, frames, height, width = values.shape
    # The kernels mirror each coordinate once, the extent F.pad accepts.
    if mode == "reflect" and (
        max(left, right) >= width or max(top, bottom) >= height
    ):
        raise ValueError("reflected padding must be shorter than its extent")

    return torch.empty(
        (
            batch,
            channels,
            frames + front,
            height + top + bottom,
            width + left + right,
        ),
        dtype=values.dtype,
        device=values.device,
        memory_format=memory_format,
    )


def frame_pad(
    values: torch.Tensor,
    padding: tuple[int, int, int, int, int],
    *,
    mode: str = "reflect",
    memory_format: torch.memory_format = torch.contiguous_format,
) -> torch.Tensor:
    """Pad ``[batch, channels, frames, height, width]`` values causally.

    ``padding`` is ``(left, right, top, bottom, front)``: columns and rows
    padded on each side in ``mode`` (``reflect`` or ``replicate``), then
    ``front`` zero frames before the first frame. Reflection mirrors once,
    so each reflected side must be shorter than its extent. Returns a new
    tensor stored in ``memory_format``: ``torch.contiguous_format`` or
    ``torch.channels_last_3d``.

    Raises:
        ValueError: when the padding or values are malformed, or on CUDA
            when the kernel cannot take the operands.
    """
    out = _padded(values, padding, mode, memory_format)
    left, right, top, bottom, front = padding
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


def frame_norm_pad(
    values: torch.Tensor,
    padding: tuple[int, int, int, int, int],
    *,
    groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    mode: str = "reflect",
    memory_format: torch.memory_format = torch.contiguous_format,
) -> torch.Tensor:
    """Pad SiLU of the per-frame group normalization of ``values``.

    Every frame of ``[batch, channels, frames, height, width]`` values is
    normalized on its own, as ``F.group_norm`` normalizes a ``[channels,
    height, width]`` sample with ``groups``, ``weight``, ``bias`` and
    ``eps``; frames never mix. The result passes through SiLU and is padded
    as ``frame_pad`` pads it with ``padding``, ``mode`` and
    ``memory_format``, and equals that composition bit for bit.

    On CUDA the kernels read the moments straight from the values and write
    the padded result in one further pass; values whose frames do not store
    each channel's pixels contiguously, such as channels-last values, are
    first copied frame-major, the copy the composition's frame folding makes.
    Other devices evaluate the composition.

    Raises:
        ValueError: when the padding, values or groups are malformed, or on
            CUDA when the kernels cannot take the operands.
    """
    out = _padded(values, padding, mode, memory_format)
    if type(groups) is not int or groups < 1 or values.shape[1] % groups:
        raise ValueError("frame normalization groups must divide the channels")
    left, right, top, bottom, front = padding
    if values.is_cuda:
        from uniserve_kernels.norm import frame

        if values.stride(4) != 1 or values.stride(3) != values.shape[4]:
            # Frame-major [batch, frames, channels, height, width] storage.
            frame_major = values.permute(0, 2, 1, 3, 4).contiguous()
            values = frame_major.permute(0, 2, 1, 3, 4)
        require_kernel(
            "frame_norm_pad",
            frame.unsupported(values, out, weight, bias),
            values=values,
        )
        frame.normalize_pad(
            values,
            out,
            groups=groups,
            weight=weight,
            bias=bias,
            eps=eps,
            padding=(top, left, front),
            reflect=mode == "reflect",
        )
        return out

    # Fold the frames into the batch, normalize and activate, then pad.
    batch, channels, frames, height, width = values.shape
    folded = values.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    activated = F.silu(F.group_norm(folded, groups, weight, bias, eps))
    return frame_pad(
        activated.view(batch, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        ),
        padding,
        mode=mode,
        memory_format=memory_format,
    )
