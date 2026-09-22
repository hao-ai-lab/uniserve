"""Canonical image patch rows and their inverse."""

from __future__ import annotations

import torch


def patchify(images: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    """Pack CHW or NCHW images in spatial-patch, pixel, then channel order."""
    if (
        type(patch_size) is not int
        or patch_size < 1
        or images.ndim not in {3, 4}
    ):
        raise ValueError(
            "patching requires CHW/NCHW images and a positive patch size"
        )
    channels, height, width = images.shape[-3:]
    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be divisible by the patch size")

    batch = images.shape[0] if images.ndim == 4 else 1
    rows, columns = height // patch_size, width // patch_size
    # [batch, rows * columns, patch_size**2 * channels]
    result = (
        images.reshape(batch, channels, rows, patch_size, columns, patch_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, rows * columns, patch_size**2 * channels)
    )
    return result if images.ndim == 4 else result[0]


def unpatchify(
    patches: torch.Tensor, size, *, patch_size: int, channels: int
) -> torch.Tensor:
    """Restore canonical patch rows into a CHW or NCHW image of the explicit
    size.
    """  # noqa: D205
    if (
        type(patch_size) is not int
        or patch_size < 1
        or type(channels) is not int
        or channels < 1
        or patches.ndim not in {2, 3}
    ):
        raise ValueError(
            "unpatching requires token rows and positive patch/channel "
            "dimensions"
        )
    if size.height % patch_size or size.width % patch_size:
        raise ValueError("image dimensions must be divisible by the patch size")
    rows, columns = size.height // patch_size, size.width // patch_size
    if patches.shape[-2:] != (rows * columns, patch_size**2 * channels):
        raise ValueError(
            "patch rows do not match the requested image dimensions"
        )

    batch = patches.shape[0] if patches.ndim == 3 else 1
    # Inverse of patchify: [batch, channels, height, width]
    result = (
        patches.reshape(batch, rows, columns, patch_size, patch_size, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, size.height, size.width)
    )
    return result if patches.ndim == 3 else result[0]


def unpatchify_video_tokens(
    source: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    grid_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Unpack ``[batch, tokens, channels * patch volume]`` into planar video.

    Grid and patch dimensions follow time, height, width order. Each token
    stores channel-major patch elements; trailing padded tokens are ignored.
    Bias is added in FP32 before storing the rearranged output in the source
    dtype, as ``[batch, channels, frames, height, width]``.
    """
    from uniserve_kernels import patch

    if source.ndim != 3 or min(*grid_shape, *patch_shape) < 1:
        raise ValueError(
            "video unpacking requires positive grid/patch dimensions and "
            "token rows"
        )

    batch, sequence, token_width = source.shape
    frames, height, width = grid_shape
    patch_frames, patch_height, patch_width = patch_shape
    patch_volume = patch_frames * patch_height * patch_width
    if token_width % patch_volume or batch < 1:
        raise ValueError(
            "token width must contain a positive integral channel count"
        )
    channels = token_width // patch_volume
    if channels < 1 or sequence < frames * height * width:
        raise ValueError(
            "video unpacking has fewer tokens or channels than required"
        )
    if bias is not None and (
        bias.shape != (token_width,) or bias.device != source.device
    ):
        raise ValueError("patch bias must match the token width and device")

    shape = (
        batch,
        channels,
        frames * patch_frames,
        height * patch_height,
        width * patch_width,
    )
    if patch.can_run(source, bias):
        output = torch.empty(shape, dtype=source.dtype, device=source.device)
        patch.unpatchify_video_tokens(
            source,
            bias,
            output,
            grid_shape=grid_shape,
            patch_shape=patch_shape,
        )
        return output

    value = source[:, : frames * height * width]
    if bias is not None:
        value = (value.float() + bias.float()).to(source.dtype)
    value = value.reshape(
        batch,
        frames,
        height,
        width,
        channels,
        patch_frames,
        patch_height,
        patch_width,
    )
    return value.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().reshape(shape)
