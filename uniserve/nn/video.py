"""Numerical overlap blending and decoded video pixel conversion."""

from __future__ import annotations

import torch


def blend_decoded_overlap(
    previous: torch.Tensor,
    current: torch.Tensor,
    extent: int,
    dim: int,
) -> torch.Tensor:
    """Cross-fade an overlap extent between adjacent decoded tiles along one
    dimension.
    """  # noqa: D205
    if extent < 0:
        raise ValueError("decoded overlap extent cannot be negative")
    extent = min(previous.shape[dim], current.shape[dim], extent)
    if extent == 0:
        return current

    # Linear ramp along the blend axis: the previous tile's weight falls from
    # 1 to 0 while the current tile's weight rises from 0 to (extent-1)/extent.
    positions = torch.arange(extent, device=current.device, dtype=current.dtype)
    shape = [1] * current.ndim
    shape[dim] = extent
    previous_weight = (1 - positions / extent).view(shape)
    current_weight = (positions / extent).view(shape)

    previous_slice = [slice(None)] * current.ndim
    current_slice = [slice(None)] * current.ndim
    previous_slice[dim] = slice(-extent, None)
    current_slice[dim] = slice(0, extent)
    blended = (
        previous[tuple(previous_slice)] * previous_weight
        + current[tuple(current_slice)] * current_weight
    )

    if extent == current.shape[dim]:
        return blended
    remainder = [slice(None)] * current.ndim
    remainder[dim] = slice(extent, None)
    return torch.cat((blended, current[tuple(remainder)]), dim=dim)
