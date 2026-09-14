"""Token-to-raster transforms with explicit spatiotemporal patch geometry."""

from __future__ import annotations

import torch

from uniserve.runtime.triton import triton_available

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


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
        PATCH_FRAMES: tl.constexpr,
        PATCH_HEIGHT: tl.constexpr,
        PATCH_WIDTH: tl.constexpr,
        CHANNELS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Unpack decoder patch channels into planar channel-major video coordinates."""

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        output_width: tl.constexpr = width * PATCH_WIDTH
        output_height: tl.constexpr = height * PATCH_HEIGHT
        output_frames: tl.constexpr = frames * PATCH_FRAMES
        output_spatial: tl.constexpr = output_height * output_width
        output_volume: tl.constexpr = output_frames * output_spatial
        output_channels: tl.constexpr = CHANNELS * output_volume

        # Decompose a planar output offset, then invert the decoder's patch packing.
        batch = offsets // output_channels
        remainder = offsets - batch * output_channels
        channel = remainder // output_volume
        remainder -= channel * output_volume
        output_frame = remainder // output_spatial
        remainder -= output_frame * output_spatial
        output_row = remainder // output_width
        output_column = remainder - output_row * output_width
        frame = output_frame // PATCH_FRAMES
        temporal_patch = output_frame - frame * PATCH_FRAMES
        patch_row = output_row // PATCH_HEIGHT
        inner_row = output_row - patch_row * PATCH_HEIGHT
        patch_column = output_column // PATCH_WIDTH
        inner_column = output_column - patch_column * PATCH_WIDTH
        patch = (frame * height + patch_row) * width + patch_column
        patch_channel = (
            ((channel * PATCH_FRAMES + temporal_patch) * PATCH_HEIGHT + inner_row) * PATCH_WIDTH
        ) + inner_column
        source_offsets = (batch * sequence + patch) * (
            CHANNELS * PATCH_FRAMES * PATCH_HEIGHT * PATCH_WIDTH
        ) + patch_channel
        value = tl.load(source_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            value += tl.load(bias_ptr + patch_channel, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + offsets, value, mask=mask)


def unpatchify_video_tokens(
    source: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    grid_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Unpack ``[batch, tokens, channels * patch volume]`` into planar video.

    Grid and patch dimensions follow time, height, width order. Each token stores
    channel-major patch elements. Trailing padded tokens are ignored. Bias is
    accumulated in FP32 before storing the rearranged output in the source dtype.
    """

    if source.ndim != 3 or min(*grid_shape, *patch_shape) < 1:
        raise ValueError("video unpacking requires positive grid/patch dimensions and token rows")
    batch, sequence, token_width = source.shape
    frames, height, width = grid_shape
    patch_frames, patch_height, patch_width = patch_shape
    patch_volume = patch_frames * patch_height * patch_width
    if token_width % patch_volume or batch < 1:
        raise ValueError("token width must contain a positive integral channel count")
    channels = token_width // patch_volume
    if channels < 1 or sequence < frames * height * width:
        raise ValueError("video unpacking has fewer tokens or channels than required")
    if bias is not None and (bias.shape != (token_width,) or bias.device != source.device):
        raise ValueError("patch bias must match the token width and device")
    if not (
        source.is_cuda
        and triton_available(source.device)
        and source.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and source.is_contiguous()
        and (bias is None or (bias.is_contiguous() and bias.dtype == source.dtype))
    ):
        if bias is not None:
            source = (source + bias).to(source.dtype)
        value = source[:, : frames * height * width].reshape(
            batch, frames, height, width, channels, patch_frames, patch_height, patch_width
        )
        return (
            value.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .contiguous()
            .reshape(
                batch, channels, frames * patch_frames, height * patch_height, width * patch_width
            )
        )
    output = torch.empty(
        (batch, channels, frames * patch_frames, height * patch_height, width * patch_width),
        dtype=source.dtype,
        device=source.device,
    )
    _unpatchify_video_tokens_kernel[(triton.cdiv(output.numel(), 1024),)](
        source,
        bias,
        output,
        elements=output.numel(),
        sequence=sequence,
        frames=frames,
        height=height,
        width=width,
        PATCH_FRAMES=patch_frames,
        PATCH_HEIGHT=patch_height,
        PATCH_WIDTH=patch_width,
        CHANNELS=channels,
        HAS_BIAS=bias is not None,
        BLOCK=1024,
        num_warps=4,
    )
    return output
