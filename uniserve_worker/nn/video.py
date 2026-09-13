"""Numerical overlap blending and decoded video pixel conversion."""

from __future__ import annotations

import torch


def blend_decoded_overlap(
    previous: torch.Tensor,
    current: torch.Tensor,
    extent: int,
    dim: int,
) -> torch.Tensor:
    """Cross-fade an overlap extent between adjacent decoded tiles along one dimension."""

    if extent < 0:
        raise ValueError("decoded overlap extent cannot be negative")
    extent = min(previous.shape[dim], current.shape[dim], extent)
    if extent == 0:
        return current
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


@torch.inference_mode()
def video_segment_rgb(
    segment: torch.Tensor,
    previous_overlap: torch.Tensor | None,
    *,
    body_frames: int,
    overlap_frames: int,
    padding_frames: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    final_unit: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join one NCTHW segment and return RGB24 frames and its successor overlap.

    The caller supplies checkpoint normalization and temporal-window geometry.
    Multiplication, addition, clamping and integer rounding preserve that order;
    overlap blending retains the decoded tensor's arithmetic precision.
    """

    if segment.ndim != 5 or segment.shape[:2] != (1, 3):
        raise ValueError("decoded video segments must have shape [1, 3, frames, height, width]")
    if (
        min(body_frames, overlap_frames) < 1
        or padding_frames < 0
        or body_frames + padding_frames + overlap_frames > segment.shape[2]
    ):
        raise ValueError("decoded segment does not cover its body and successor overlap")
    if pixel_mean.shape != (1, 3, 1, 1, 1) or pixel_std.shape != pixel_mean.shape:
        raise ValueError("video pixel normalization requires one mean and scale per channel")
    body = segment[:, :, :body_frames]
    if previous_overlap is not None:
        body = blend_decoded_overlap(previous_overlap, body, overlap_frames, dim=-3)
    next_overlap = segment[:, :, body_frames + padding_frames :].contiguous()
    if final_unit:
        body = torch.cat((body, next_overlap[:, :, :overlap_frames]), dim=2)
    pixels = (body.float() * pixel_std + pixel_mean).clamp_(0.0, 1.0)
    rgb24 = pixels[0].permute(1, 2, 3, 0).mul_(255.0).round_().to(torch.uint8).contiguous()
    return rgb24, next_overlap[:, :, :overlap_frames].contiguous()
