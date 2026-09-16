"""Patch (de)serialization and per-patch position helpers for vision inputs."""

from __future__ import annotations

import torch

__all__ = [
    "patchify",
    "patchify_batch",
    "build_abs_positions_from_grid_hw",
]


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert ``(C,H,W)`` pixels into flattened patch rows in packed vision
    order.
    """  # noqa: D205
    p = int(patch_size)
    c, h, w = image.shape
    if h % p != 0 or w % p != 0:
        raise ValueError(
            f"image shape {(c, h, w)} is not divisible by patch size {p}"
        )
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    return image.reshape(-1, p * p * c)


def patchify_batch(
    images: torch.Tensor, patch_size: int, *, channel_first: bool = False
) -> torch.Tensor:
    """Convert ``(N,3,H,W)`` images into ``(N,L,patch_size**2*3)`` patch
    rows.
    """  # noqa: D205
    p = int(patch_size)
    batch, channels, height, width = images.shape
    if height % p != 0 or width % p != 0:
        raise ValueError(
            f"image shape {(batch, channels, height, width)} is not "
            f"divisible by {p}"
        )
    h = height // p
    w = width // p
    x = images.reshape(batch, channels, h, p, w, p)
    if channel_first:
        x = torch.einsum("nchpwq->nhwcpq", x)
    else:
        x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(batch, h * w, p * p * channels)


def build_abs_positions_from_grid_hw(
    grid_hw: torch.Tensor,
    *,
    device=None,
    total: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-patch x/y coordinates for one or more image grids.

    ``total`` is the number of patches across all grids. The caller takes it
    from a static tensor shape, so no value is read back from ``grid_hw`` and
    the reduction stays capturable inside a CUDA graph.
    """
    device = device or grid_hw.device
    grid_hw = grid_hw.to(device)
    batch = grid_hw.shape[0]
    heights = grid_hw[:, 0]
    widths = grid_hw[:, 1]
    counts = heights * widths
    total = int(total)

    patch_to_sample = torch.repeat_interleave(
        torch.arange(batch, device=device), counts, output_size=total
    )
    patch_id = torch.arange(total, device=device)
    # ``counts.new_zeros(1)`` allocates the leading zero directly on-device;
    # ``torch.tensor([0], device=device)`` would stage a host tensor and copy
    # it, which is rejected mid CUDA-graph capture.
    offsets = torch.cumsum(torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0)
    patch_id = patch_id - offsets[patch_to_sample]

    width_per_patch = widths[patch_to_sample]
    abs_x = patch_id % width_per_patch
    abs_y = patch_id // width_per_patch
    return abs_x, abs_y
