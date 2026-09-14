"""Spatial sizes, video timing and mathematical reconstruction windows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageSize:
    """The output raster used by image diffusion and reconstruction."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if min(self.height, self.width) < 1:
            raise ValueError("image dimensions must be positive")


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Raster, duration and sample rates of a complete video."""

    frame_count: int
    width: int
    height: int
    frame_rate: int
    audio_rate: int

    def __post_init__(self) -> None:
        if min(self.frame_count, self.width, self.height, self.frame_rate, self.audio_rate) < 1:
            raise ValueError("video dimensions and rates must be positive")


@dataclass(frozen=True, slots=True)
class DecodeWindow:
    """One logical temporal reconstruction interval, independent of placement.

    Latent and output frame intervals are half-open. ``crop`` removes decoder
    boundary frames before postprocessing. A cropped segment contains a body,
    padding, then its successor overlap. Only the final window appends that
    overlap to its output; other windows retain it in caller-owned state.
    """

    latent_start: int
    latent_stop: int
    frame_start: int
    frame_stop: int
    body_frames: int
    overlap_frames: int
    padding_frames: int
    crop: tuple[int, int] = (0, 0)
    final: bool = False

    def __post_init__(self) -> None:
        if (
            min(self.latent_start, self.frame_start, self.padding_frames, *self.crop) < 0
            or self.latent_stop <= self.latent_start
            or min(self.body_frames, self.overlap_frames) < 1
        ):
            raise ValueError("decode window has invalid temporal extents")
        frames = self.body_frames + (self.overlap_frames if self.final else 0)
        if self.frame_stop - self.frame_start != frames:
            raise ValueError("decode window output must cover its body and final overlap")

    @property
    def segment_frames(self) -> int:
        """Length after decoder boundary cropping and before overlap processing."""

        return self.body_frames + self.padding_frames + self.overlap_frames


@dataclass(frozen=True, slots=True)
class VideoSize:
    """Generated video frames and the numerical conditioning token extent."""

    frames: int
    prompt_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.frames) is not int
            or self.frames < 1
            or type(self.prompt_tokens) is not int
            or self.prompt_tokens < 0
        ):
            raise ValueError("video size requires positive frames and nonnegative text tokens")
