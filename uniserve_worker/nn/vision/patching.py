"""Patch (de)serialization and per-patch position helpers for vision inputs."""
from __future__ import annotations

import torch

__all__ = [
    'patchify',
    'patchify_batch',
    'unpatchify_batch',
    'build_abs_positions_from_grid_hw',
]


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert ``(C,H,W)`` pixels into flattened patch rows in packed vision order."""

    p = int(patch_size)
    c, h, w = image.shape
    if h % p != 0 or w % p != 0:
        raise ValueError(f"image shape {(c, h, w)} is not divisible by patch size {p}")
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    return image.reshape(-1, p * p * c)


def patchify_batch(images: torch.Tensor, patch_size: int, *, channel_first: bool = False) -> torch.Tensor:
    """Convert ``(N,3,H,W)`` images into ``(N,L,patch_size**2*3)`` patch rows."""

    p = int(patch_size)
    batch, channels, height, width = images.shape
    if height % p != 0 or width % p != 0:
        raise ValueError(f"image shape {(batch, channels, height, width)} is not divisible by {p}")
    h = height // p
    w = width // p
    x = images.reshape(batch, channels, h, p, w, p)
    if channel_first:
        x = torch.einsum("nchpwq->nhwcpq", x)
    else:
        x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(batch, h * w, p * p * channels)


def unpatchify_batch(
    patches: torch.Tensor,
    patch_size: int,
    *,
    height: int | None = None,
    width: int | None = None,
    channels: int | None = None,
) -> torch.Tensor:
    """Convert ``(N,L,patch_size**2*C)`` patch rows back to ``(N,C,H,W)``.

    ``channels`` defaults to ``3`` (RGB) but can be set explicitly for latent
    folds (e.g. VAE latents with ``vae_z_channels`` channels); when omitted it is
    inferred from the patch-row width so the channel count is named in one place.
    """

    p = int(patch_size)
    if channels is None:
        c = int(patches.shape[-1]) // (p * p)
    else:
        c = int(channels)
    if height is None or width is None:
        h = w = int(patches.shape[1] ** 0.5)
    else:
        h = int(height) // p
        w = int(width) // p
    x = patches.reshape(patches.shape[0], h, w, p, p, c)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(patches.shape[0], c, h * p, w * p)


def build_abs_positions_from_grid_hw(
    grid_hw: torch.Tensor,
    *,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-patch x/y coordinates for one or more image grids."""

    device = device or grid_hw.device
    grid_hw = grid_hw.to(device)
    batch = grid_hw.shape[0]
    heights = grid_hw[:, 0]
    widths = grid_hw[:, 1]
    counts = heights * widths
    total = int(counts.sum().item())
    patch_to_sample = torch.repeat_interleave(torch.arange(batch, device=device), counts)
    patch_id = torch.arange(total, device=device)
    offsets = torch.cumsum(torch.cat([torch.tensor([0], device=device), counts[:-1]]), dim=0)
    patch_id = patch_id - offsets[patch_to_sample]
    width_per_patch = widths[patch_to_sample]
    abs_x = patch_id % width_per_patch
    abs_y = patch_id // width_per_patch
    return abs_x, abs_y
