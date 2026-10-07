"""Causal convolution padding of video frames in one pass.

A causal 3D convolution over ``[batch, channels, frames, height, width]``
reads its input padded on both sides of height and width (reflected or
replicated) and preceded by zero frames. ``triton_frame_pad`` writes that
padded input into contiguous storage in one pass, reading the source through
its own strides, so channel-first, channels-last and permuted views all
work. The output equals the two-step ``F.pad`` result bit for bit: every
element is a copy of one source element or zero. Callers run
``can_run_triton_frame_pad`` first; the launcher does not revalidate.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

# Output pixels along the width and channels each program writes.
_PAD_PIXELS = 32
_PAD_CHANNELS = 64


if triton is not None:  # pragma: no cover - depends on the accelerator stack.

    @triton.jit
    def _frame_pad_kernel(
        values,
        out,
        stride_batch,
        stride_channel,
        stride_frame,
        stride_height,
        stride_width,
        height,
        width,
        out_frames,
        out_height,
        out_width,
        channels,
        pad_top,
        pad_left,
        pad_front,
        reflect: tl.constexpr,
        block_pixels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        """Write one row segment of the padded contiguous output.

        Grid: (samples * out_frames * out_height, width blocks, channel
        blocks). Output frames before ``pad_front`` are zero; every other
        output pixel copies the reflected or replicated source pixel.
        """
        row_id = tl.program_id(0)
        out_row = row_id % out_height
        out_frame = (row_id // out_height) % out_frames
        sample = row_id // (out_height * out_frames)
        columns = tl.program_id(1) * block_pixels + tl.arange(0, block_pixels)
        channel = tl.program_id(2) * block_channels + tl.arange(
            0, block_channels
        )
        valid = (columns < out_width)[:, None] & (channel < channels)[None, :]

        # Contiguous storage: [sample, channel, frame, row, column]; the
        # columns of one row are adjacent, so the stores coalesce.
        plane = out_frames.to(tl.int64) * out_height * out_width
        destination = (
            out
            + (sample.to(tl.int64) * channels + channel[None, :]) * plane
            + (out_frame.to(tl.int64) * out_height + out_row) * out_width
            + columns[:, None]
        )

        frame = out_frame - pad_front
        if frame < 0:
            tl.store(
                destination,
                tl.zeros((block_pixels, block_channels), dtype=tl.float32).to(
                    out.dtype.element_ty
                ),
                mask=valid,
            )
        else:
            source_row = out_row - pad_top
            source_columns = columns - pad_left
            if reflect:
                source_row = tl.where(source_row < 0, -source_row, source_row)
                source_row = tl.where(
                    source_row >= height,
                    2 * height - 2 - source_row,
                    source_row,
                )
                source_columns = tl.where(
                    source_columns < 0, -source_columns, source_columns
                )
                source_columns = tl.where(
                    source_columns >= width,
                    2 * width - 2 - source_columns,
                    source_columns,
                )
            else:
                source_row = tl.minimum(tl.maximum(source_row, 0), height - 1)
                source_columns = tl.minimum(
                    tl.maximum(source_columns, 0), width - 1
                )
            copied = tl.load(
                values
                + sample.to(tl.int64) * stride_batch
                + frame.to(tl.int64) * stride_frame
                + source_row.to(tl.int64) * stride_height
                + source_columns[:, None].to(tl.int64) * stride_width
                + channel[None, :].to(tl.int64) * stride_channel,
                mask=valid,
            )
            tl.store(destination, copied, mask=valid)


def can_run_triton_frame_pad(values: torch.Tensor, out: torch.Tensor) -> bool:
    """Check a ``[batch, channels, frames, height, width]`` input and output.

    The input may use any strides; the output must be contiguous storage of
    the same dtype on the same CUDA device. The launch records no autograd
    graph, so calls that record one take the tensor operations.
    """
    return (
        triton is not None
        and not torch.is_grad_enabled()
        and launchable(values.device)
        and values.is_cuda
        and out.device == values.device
        and values.ndim == 5
        and out.ndim == 5
        and out.dtype == values.dtype
        and out.is_contiguous()
        and out.shape[:2] == values.shape[:2]
    )


def triton_frame_pad(
    values: torch.Tensor,
    out: torch.Tensor,
    *,
    padding: tuple[int, int, int],
    reflect: bool,
) -> None:
    """Write ``values`` padded into ``out``.

    ``padding`` is ``(top, left, front)``: the rows above, columns left of
    and zero frames before the source within ``out``; the remaining rows and
    columns of ``out`` pad the bottom and right.
    """
    batch, channels, _, height, width = (int(size) for size in values.shape)
    _, _, out_frames, out_height, out_width = (int(size) for size in out.shape)
    _frame_pad_kernel[
        (
            batch * out_frames * out_height,
            triton.cdiv(out_width, _PAD_PIXELS),
            triton.cdiv(channels, _PAD_CHANNELS),
        )
    ](
        values,
        out,
        *(int(stride) for stride in values.stride()),
        height,
        width,
        out_frames,
        out_height,
        out_width,
        channels,
        *padding,
        reflect,
        _PAD_PIXELS,
        _PAD_CHANNELS,
        num_warps=4,
    )
