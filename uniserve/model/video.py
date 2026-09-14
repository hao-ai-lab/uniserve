"""Windowed video reconstruction over borrowed numerical state."""

from __future__ import annotations

from abc import abstractmethod

import torch

from uniserve.model.batch import TensorOutput
from uniserve.model.media import DecodeWindow, VideoInfo
from uniserve.model.tensors import TensorViews
from uniserve.nn.video import video_segment_rgb
from uniserve.tensors import OutputLayout


class VideoMixin:
    """Describe temporal geometry and reconstruct ordered RGB frame windows.

    The default computation cross-fades overlaps in the decoded dtype, then
    denormalizes in float32 before clamping and rounding to RGB24. It consumes
    ``pixel_mean`` and ``pixel_std`` constants, updates ``video_overlap`` state,
    and returns a prefix of the caller's ``rgb_frames`` scratch. The caller must
    retain that storage until the result's final reader completes.
    """

    output_capacity: VideoInfo
    decode_frame_capacity: int
    min_frames: int
    text_alignment: int

    @abstractmethod
    def video_info(self, frames: int) -> VideoInfo:
        """Describe the raster and timing of a mathematically legal video."""

    @abstractmethod
    def decode_windows(self, shape: VideoInfo) -> tuple[DecodeWindow, ...]:
        """Describe overlapping latent windows in logical output order."""

    @torch.inference_mode()
    def postprocess_video(
        self,
        segments: tuple[torch.Tensor, ...],
        windows: tuple[DecodeWindow, ...],
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        """Join a contiguous window range, preserving overlap across calls.

        Segments have shape [1, 3, T, H, W] after each window's decoder crop.
        Starting at frame zero ignores previous state; any later start requires
        the predecessor's overlap in ``state``. Inputs remain unchanged. Shape
        errors are rejected before either state or output storage is written.
        """

        if not segments or len(segments) != len(windows):
            raise ValueError("video segments must align with logical decode windows")
        first = segments[0]
        if first.ndim != 5 or first.shape[:2] != (1, 3):
            raise ValueError("video segments must have shape [1, 3, T, H, W]")
        height, width = first.shape[-2:]
        overlap_state = state["video_overlap"]
        pixels = scratch["rgb_frames"]
        mean, std = constants["pixel_mean"], constants["pixel_std"]
        if mean.shape != (1, 3, 1, 1, 1) or std.shape != mean.shape:
            raise ValueError("video normalization must contain one mean and scale per channel")
        if any(value.device != first.device for value in (overlap_state, pixels, mean, std)):
            raise ValueError("video views must be bound to the input device")
        if mean.dtype != torch.float32 or std.dtype != torch.float32:
            raise ValueError("video normalization constants must use float32")
        if (
            pixels.ndim != 4
            or pixels.shape[1:] != (height, width, 3)
            or pixels.dtype != torch.uint8
            or pixels.shape[0] < windows[-1].frame_stop - windows[0].frame_start
        ):
            raise ValueError("RGB scratch must cover the complete output frame range")
        for index, (segment, window) in enumerate(zip(segments, windows, strict=True)):
            if (
                segment.shape != (1, 3, window.segment_frames, height, width)
                or segment.dtype != first.dtype
                or segment.device != first.device
                or overlap_state.shape != (1, 3, window.overlap_frames, height, width)
                or overlap_state.dtype != segment.dtype
            ):
                raise ValueError("video segment or overlap state disagrees with its window")
            if index and (
                windows[index - 1].frame_stop != window.frame_start or windows[index - 1].final
            ):
                raise ValueError("video windows must describe a contiguous ordered range")

        overlap = None if windows[0].frame_start == 0 else overlap_state
        frame_start = 0
        for segment, window in zip(segments, windows, strict=True):
            rgb, overlap = video_segment_rgb(
                segment,
                overlap,
                body_frames=window.body_frames,
                overlap_frames=window.overlap_frames,
                padding_frames=window.padding_frames,
                pixel_mean=mean,
                pixel_std=std,
                final_unit=window.final,
            )
            frames = window.frame_stop - window.frame_start
            pixels[frame_start : frame_start + frames].copy_(rgb)
            frame_start += frames
        assert overlap is not None
        overlap_state.copy_(overlap)
        output = pixels[:frame_start]
        return TensorOutput(
            {"video": (output,)},
            {"video": (OutputLayout(shape=tuple(output.shape), dtype=output.dtype),)},
        )
