"""Token-to-raster rearrangement with explicit spatiotemporal patch shapes."""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

if triton is not None:

    @triton.jit
    def _unpatchify_video_tokens_kernel(
        source_ptr,
        bias_ptr,
        output_ptr,
        elements: tl.constexpr,
        sequence: tl.constexpr,
        frames: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        PATCH_FRAMES: tl.constexpr,  # noqa: N803
        PATCH_HEIGHT: tl.constexpr,  # noqa: N803
        PATCH_WIDTH: tl.constexpr,  # noqa: N803
        CHANNELS: tl.constexpr,  # noqa: N803
        HAS_BIAS: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Unpack decoder patch channels into video coordinates.

        Unpack decoder patch channels into planar channel-major video
        coordinates.
        """
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements

        output_width: tl.constexpr = width * PATCH_WIDTH
        output_height: tl.constexpr = height * PATCH_HEIGHT
        output_frames: tl.constexpr = frames * PATCH_FRAMES
        output_spatial: tl.constexpr = output_height * output_width
        output_volume: tl.constexpr = output_frames * output_spatial
        output_channels: tl.constexpr = CHANNELS * output_volume

        # Decompose a planar output offset into [batch, channel, frame,
        # row, col] coordinates of the unpacked video.
        batch = offsets // output_channels
        remainder = offsets - batch * output_channels
        channel = remainder // output_volume
        remainder -= channel * output_volume
        output_frame = remainder // output_spatial
        remainder -= output_frame * output_spatial
        output_row = remainder // output_width
        output_column = remainder - output_row * output_width

        # Invert the decoder's patch packing: locate the source patch and the
        # element's position within that patch.
        frame = output_frame // PATCH_FRAMES
        temporal_patch = output_frame - frame * PATCH_FRAMES
        patch_row = output_row // PATCH_HEIGHT
        inner_row = output_row - patch_row * PATCH_HEIGHT
        patch_column = output_column // PATCH_WIDTH
        inner_column = output_column - patch_column * PATCH_WIDTH
        patch = (frame * height + patch_row) * width + patch_column
        patch_channel = (
            (
                (channel * PATCH_FRAMES + temporal_patch) * PATCH_HEIGHT
                + inner_row
            )
            * PATCH_WIDTH
        ) + inner_column

        # Source tokens are [batch, patch, channel-major patch volume].
        source_offsets = (batch * sequence + patch) * (
            CHANNELS * PATCH_FRAMES * PATCH_HEIGHT * PATCH_WIDTH
        ) + patch_channel
        value = tl.load(source_ptr + source_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        if HAS_BIAS:
            value += tl.load(bias_ptr + patch_channel, mask=mask, other=0.0).to(
                tl.float32
            )
        tl.store(output_ptr + offsets, value, mask=mask)


def can_run(source: torch.Tensor, bias: torch.Tensor | None) -> bool:
    """Return whether contiguous CUDA tokens and bias fit the kernel."""
    return (
        launchable(source.device)
        and source.is_cuda
        and source.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and source.is_contiguous()
        and (
            bias is None
            or (bias.is_contiguous() and bias.dtype == source.dtype)
        )
    )


def unpatchify_video_tokens(
    source: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    *,
    grid_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
) -> None:
    """Store channel-major token patches into contiguous planar ``out``.

    ``out`` has shape ``[batch, channels, frames * pt, height * ph,
    width * pw]``; trailing padded tokens are ignored.
    """
    frames, height, width = grid_shape
    patch_frames, patch_height, patch_width = patch_shape
    _unpatchify_video_tokens_kernel[(triton.cdiv(out.numel(), 1024),)](
        source,
        bias,
        out,
        elements=out.numel(),
        sequence=int(source.shape[1]),
        frames=frames,
        height=height,
        width=width,
        PATCH_FRAMES=patch_frames,
        PATCH_HEIGHT=patch_height,
        PATCH_WIDTH=patch_width,
        CHANNELS=int(out.shape[1]),
        HAS_BIAS=bias is not None,
        BLOCK=1024,
        num_warps=4,
    )
