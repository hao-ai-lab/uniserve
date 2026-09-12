# SPDX-License-Identifier: Apache-2.0
"""Ordered Ref2VA geometry in semantic (unpadded, untiled) row coordinates.

The joint transformer consumes one presentation followed by the references in
request order, then target audio and video. Modality projection order is distinct
from joint sequence order: each projection receives its reference rows first and
its target rows last. Solver state must contain only the latter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from .packing import (
    AUDIO_TAG,
    ROPE_FRAME_RESCALE,
    ROPE_FRAMES_PER_LATENT,
    TEXT_TAG,
    VIDEO_TAG,
    _spatial_grid,
    _temporal_grid,
)


@dataclass(frozen=True, slots=True)
class H3ReferenceGeometry:
    """Resolved VAE geometry for one ordered medium, including its soundtrack.

    Heights and widths are latent pixels, not input pixels or transformer
    patches. Stereo audio counts time positions once; it occupies twice as many
    rows. An image has exactly one latent frame and no soundtrack.
    """

    media_type: Literal["image", "video", "audio"]
    video_frames: int = 0
    latent_height: int = 0
    latent_width: int = 0
    audio_frames: int = 0

    def __post_init__(self) -> None:
        dimensions = (self.video_frames, self.latent_height, self.latent_width, self.audio_frames)
        if any(type(value) is not int or value < 0 for value in dimensions):
            raise ValueError("reference latent dimensions must be nonnegative integers")
        if self.media_type not in ("image", "video", "audio"):
            raise ValueError("reference type must be image, video, or audio")
        if self.media_type == "audio":
            if any(dimensions[:3]) or not self.audio_frames:
                raise ValueError("audio references require audio latents and no visual geometry")
        else:
            if min(dimensions[:3]) < 1 or self.latent_height % 2 or self.latent_width % 2:
                raise ValueError("visual reference geometry must be positive and patch-2 aligned")
            if self.media_type == "image" and (self.video_frames != 1 or self.audio_frames):
                raise ValueError("image references require one latent frame and no audio")

    @property
    def video_rows(self) -> int:
        return self.video_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def audio_rows(self) -> int:
        return 2 * self.audio_frames


def validate_reference_geometry(references: Sequence[H3ReferenceGeometry]) -> None:
    """Enforce the released ordered bundle limits without reordering media."""

    if not references or len(references) > 12:
        raise ValueError("Ref2VA requires between 1 and 12 references")
    counts = {kind: 0 for kind in ("image", "video", "audio")}
    for reference in references:
        if not isinstance(reference, H3ReferenceGeometry):
            raise TypeError("references must contain resolved H3ReferenceGeometry values")
        counts[reference.media_type] += 1
    for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
        if counts[kind] > limit:
            raise ValueError(f"Ref2VA accepts at most {limit} {kind} references")
    if counts["audio"] == len(references):
        raise ValueError("Ref2VA requires at least one visual reference")


@dataclass(frozen=True, slots=True)
class H3ReferenceLayout:
    """Joint transformer coordinates before any attention-provider tiling.

    All indices address semantic rows. Video/audio indices preserve projection
    order, so their initial condition counts can be stripped from predictions.
    `video_regions` describes reference videos and the target as (start, T, H, W)
    in transformer patch units; images are not video regions. This allows an
    attention provider to tile each region without conflating media boundaries.
    """

    position_ids: torch.Tensor
    token_tags: torch.Tensor
    text_indices: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    condition_video_rows: int
    condition_audio_rows: int
    video_regions: tuple[tuple[int, int, int, int], ...]

    @property
    def sequence_length(self) -> int:
        return int(self.token_tags.numel())

    @property
    def target_video_indices(self) -> torch.Tensor:
        return self.video_indices[self.condition_video_rows :]

    @property
    def target_audio_indices(self) -> torch.Tensor:
        return self.audio_indices[self.condition_audio_rows :]

    def row_clean_times(self, video: float, audio: float) -> torch.Tensor:
        """Return FP32 clean times, preserving fixed visual/audio conditioning.

        Vision tokens inside the Qwen presentation follow the target video
        clock. Only VAE reference rows use the condition clocks.
        """

        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (video, audio)):
            raise ValueError("H3 clean times must be finite and in [0, 1]")
        times = torch.full((self.sequence_length,), video, dtype=torch.float32)
        times[self.video_indices[: self.condition_video_rows]] = max(video, 0.999)
        times[self.audio_indices[: self.condition_audio_rows]] = 1.0
        times[self.target_audio_indices] = audio
        return times


def _frame_grid(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    area = math.sqrt(height * width)
    heights = _spatial_grid(height, 2, area)
    widths = _spatial_grid(width, 2, area)
    grid = torch.meshgrid(heights, widths, indexing="ij")
    return torch.stack([axis.reshape(-1) for axis in grid], dim=-1), widths


def build_reference_layout(
    *,
    text_token_tags: torch.Tensor,
    references: Sequence[H3ReferenceGeometry],
    video_frames: int,
    latent_height: int,
    latent_width: int,
    audio_frames: int,
) -> H3ReferenceLayout:
    """Build exact Ref2VA `[text | ordered references | target audio | video]`.

    Coordinates are CPU FP64, with the release's NumPy spatial grids and
    sequential Python summation for advancing the reference-video clock. Target
    dimensions are already VAE latent dimensions and audio length is explicit;
    this routine does not select a duration-rounding or sampling policy.
    """

    validate_reference_geometry(references)
    target = H3ReferenceGeometry("video", video_frames, latent_height, latent_width, audio_frames)
    if not target.audio_frames:
        raise ValueError("Ref2VA target audio must have positive latent length")
    if text_token_tags.ndim != 1 or not bool(
        ((text_token_tags == TEXT_TAG) | (text_token_tags == VIDEO_TAG)).all()
    ):
        raise ValueError("presentation tags must be one-dimensional text or vision tags")

    text_rows = int(text_token_tags.numel())
    condition_video_rows = sum(item.video_rows for item in references)
    condition_audio_rows = sum(item.audio_rows for item in references)
    total = (
        text_rows
        + condition_video_rows
        + condition_audio_rows
        + target.video_rows
        + target.audio_rows
    )
    positions = torch.zeros((total, 3), dtype=torch.float64)
    tags = torch.empty(total, dtype=torch.long)
    positions[:text_rows, 0] = torch.arange(text_rows, dtype=torch.float64)
    tags[:text_rows] = text_token_tags.to(device="cpu", dtype=torch.long)
    _, target_widths = _frame_grid(latent_height, latent_width)
    video_indices: list[torch.Tensor] = []
    audio_indices: list[torch.Tensor] = []
    video_regions: list[tuple[int, int, int, int]] = []
    cursor = text_rows
    clock = float(text_rows)

    def append_audio(frames: int, widths: torch.Tensor) -> None:
        nonlocal cursor
        rows = torch.arange(cursor, cursor + 2 * frames)
        positions[rows, 0] = (clock + torch.arange(frames, dtype=torch.float64)).repeat(2)
        positions[rows, 2] = torch.cat(
            (
                torch.full((frames,), float(widths[0]), dtype=torch.float64),
                torch.full((frames,), float(widths[-1]), dtype=torch.float64),
            )
        )
        tags[rows] = AUDIO_TAG
        audio_indices.append(rows)
        cursor += 2 * frames

    # Each video's soundtrack precedes that video, rather than being grouped
    # with all other audio. Both share the same rotary origin.
    for reference in (*references, target):
        if reference.media_type == "audio":
            append_audio(reference.audio_frames, target_widths)
            clock += float(reference.audio_frames)
            continue

        grid, widths = _frame_grid(reference.latent_height, reference.latent_width)
        if reference.audio_frames:
            append_audio(reference.audio_frames, widths)
        rows = torch.arange(cursor, cursor + reference.video_rows)
        tags[rows] = VIDEO_TAG
        video_indices.append(rows)
        if reference.media_type == "image":
            positions[rows, 0] = clock
            positions[rows, 1:] = grid
            clock += 1.0
        else:
            video_regions.append(
                (
                    cursor,
                    reference.video_frames,
                    reference.latent_height // 2,
                    reference.latent_width // 2,
                )
            )
            positions[rows, 0] = _temporal_grid(reference.video_frames, clock).repeat_interleave(
                grid.shape[0]
            )
            positions[rows, 1:] = grid.repeat(reference.video_frames, 1)
            # Do not use torch/NumPy reductions: their summation order changes
            # the final ULP for long references and shifts every later medium.
            duration = sum(
                ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[index % len(ROPE_FRAMES_PER_LATENT)]
                for index in range(reference.video_frames)
            )
            clock += max(float(reference.audio_frames), duration)
        cursor += reference.video_rows

    return H3ReferenceLayout(
        position_ids=positions,
        token_tags=tags,
        text_indices=torch.arange(text_rows),
        video_indices=torch.cat(video_indices),
        audio_indices=torch.cat(audio_indices),
        condition_video_rows=condition_video_rows,
        condition_audio_rows=condition_audio_rows,
        video_regions=tuple(video_regions),
    )
