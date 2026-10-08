"""Per-channel bias addition, with a residual, into channel-first storage.

PyTorch adds the bias of a cuDNN convolution in a separate pass after the
convolution, and a residual block adds that result to its shortcut in a
further pass. A convolution of channels-last input computes channels-last
``[batch, channels, *spatial]`` values, which a consumer reading each
channel's positions contiguously copies channel-first in yet another pass.
:func:`add` reads the unbiased values once and writes ``values + bias``, or
``residual + (values + bias)`` with an optionally biased residual, into
contiguous channel-first storage. Every sum is one round-to-nearest FP32
addition, as each separate tensor addition rounds it, in the same grouping,
so the result equals those additions bit for bit.

The kernel backs ``uniserve.nn.functional.bias_add``, which validates the
shapes, allocates the output and raises on CUDA whenever :func:`unsupported`
reports a reason; the launcher does not revalidate.
"""

from __future__ import annotations

import math

import torch

from uniserve_kernels.triton import tl, triton, unsupported_operands

# Spatial positions and channels of one program's tile. Channels-last values
# are read along channels and the channel-first output is written along
# positions; Triton transposes the tile through shared memory in between.
_BLOCK_POSITIONS = 128
_BLOCK_CHANNELS = 64

# CUDA grids span at most 65535 programs along their second and third axes,
# which hold the channel blocks and the samples.
_GRID_LIMIT = 65535


if triton is not None:  # pragma: no cover - depends on the accelerator stack.

    @triton.jit
    def _add_rn(a, b):
        # One round-to-nearest FP32 addition, as a tensor addition computes
        # it: no flush of subnormals and no contraction with other operations.
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=r,r,r",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _bias_add_kernel(
        values,
        bias,
        residual,
        residual_bias,
        out,
        stride_batch,
        stride_channel,
        stride_position,
        residual_batch,
        residual_channel,
        residual_position,
        channels,
        positions,
        HAS_RESIDUAL: tl.constexpr,  # noqa: N803
        HAS_RESIDUAL_BIAS: tl.constexpr,  # noqa: N803
        block_positions: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        """Write one ``[positions, channels]`` tile of one sample.

        Grid: (position blocks, channel blocks, samples). Operands are
        ``[batch, channels, positions]`` through their strides; ``out`` is
        contiguous.
        """
        sample = tl.program_id(2).to(tl.int64)
        position = tl.program_id(0) * block_positions + tl.arange(
            0, block_positions
        )
        channel = tl.program_id(1) * block_channels + tl.arange(
            0, block_channels
        )
        live = channel < channels
        valid = (position < positions)[:, None] & live[None, :]
        position = position.to(tl.int64)
        channel = channel.to(tl.int64)

        hidden = tl.load(
            values
            + sample * stride_batch
            + position[:, None] * stride_position
            + channel[None, :] * stride_channel,
            mask=valid,
        )
        hidden = _add_rn(hidden, tl.load(bias + channel, mask=live)[None, :])
        if HAS_RESIDUAL:
            shortcut = tl.load(
                residual
                + sample * residual_batch
                + position[:, None] * residual_position
                + channel[None, :] * residual_channel,
                mask=valid,
            )
            if HAS_RESIDUAL_BIAS:
                shortcut = _add_rn(
                    shortcut,
                    tl.load(residual_bias + channel, mask=live)[None, :],
                )
            # The shortcut is the left operand, as in shortcut + hidden.
            hidden = _add_rn(shortcut, hidden)

        tl.store(
            out
            + (sample * channels + channel[None, :]) * positions
            + position[:, None],
            hidden,
            mask=valid,
        )


def _blocks(extent: int, block: int) -> int:
    # Plain integer ceiling division: triton.cdiv costs about a microsecond
    # per call on the host, and every launch computes two.
    return (extent + block - 1) // block


def _rows(tensor: torch.Tensor) -> torch.Tensor | None:
    """View ``[batch, channels, *spatial]`` as ``[batch, channels, positions]``.

    Returns None when the spatial axes do not merge into one strided axis.
    The view never copies.
    """
    try:
        return tensor.view(
            tensor.shape[0], tensor.shape[1], math.prod(tensor.shape[2:])
        )
    except RuntimeError:
        return None


def unsupported(
    values: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor | None,
    residual_bias: torch.Tensor | None,
) -> str | None:
    """Return why the kernel cannot add these operands, or ``None``.

    The operands are CUDA float32. ``values`` and ``residual`` are ``[batch,
    channels, *spatial]`` with any strides under which the spatial axes
    merge into one strided axis, such as contiguous or channels-last
    storage; the biases are contiguous per-channel vectors. The caller has
    checked the shapes.
    """
    reason = unsupported_operands(values, bias, residual, residual_bias)
    if reason is not None:
        return reason
    present = tuple(
        tensor
        for tensor in (values, bias, residual, residual_bias)
        if tensor is not None
    )
    if any(tensor.dtype != torch.float32 for tensor in present):
        return "the kernel adds float32 values"
    if any(
        tensor is not None and _rows(tensor) is None
        for tensor in (values, residual)
    ):
        return "the spatial axes do not merge into one strided axis"
    if any(
        not tensor.is_contiguous()
        for tensor in (bias, residual_bias)
        if tensor is not None
    ):
        return "the biases are not contiguous vectors"
    if values.shape[0] > _GRID_LIMIT or (
        _blocks(values.shape[1], _BLOCK_CHANNELS) > _GRID_LIMIT
    ):
        return "the samples or channel blocks exceed the grid's extent"
    return None


def add(
    values: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    *,
    residual: torch.Tensor | None,
    residual_bias: torch.Tensor | None,
) -> None:
    """Write ``values + bias``, after any residual, into contiguous ``out``.

    With ``residual`` the result is ``residual + (values + bias)``, and with
    ``residual_bias`` as well ``(residual + residual_bias) + (values +
    bias)``; the biases broadcast along every axis but the channels.
    """
    rows = _rows(values)
    batch, channels, positions = (int(size) for size in rows.shape)
    if rows.numel() == 0:
        return

    shortcut = rows if residual is None else _rows(residual)
    _bias_add_kernel[
        (
            _blocks(positions, _BLOCK_POSITIONS),
            _blocks(channels, _BLOCK_CHANNELS),
            batch,
        )
    ](
        rows,
        bias,
        shortcut,
        bias if residual_bias is None else residual_bias,
        out,
        *(int(stride) for stride in rows.stride()),
        *(int(stride) for stride in shortcut.stride()),
        channels,
        positions,
        residual is not None,
        residual_bias is not None,
        _BLOCK_POSITIONS,
        _BLOCK_CHANNELS,
        num_warps=4,
    )
