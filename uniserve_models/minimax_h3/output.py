"""H3 temporal windows and RGB reconstruction over borrowed tensor views."""

from __future__ import annotations

import torch
from torch import nn

from uniserve.model.media import DecodeWindow, VideoInfo, VideoSize
from uniserve.model.tensors import TensorViews
from uniserve.model.video import VideoMixin
from uniserve.tensors import BufferConfig, OutputLayout
from uniserve_models.minimax_h3.layout import validate_frames


class VideoOutput(VideoMixin, nn.Module):
    """Combine fixed video timing with the shared window blending computation."""

    def __init__(
        self, *, width: int, height: int, frame_rate: int, audio_rate: int, frame_limit: int
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.audio_rate = audio_rate
        self.frame_limit = frame_limit
        self.output_capacity = self.video_info(frame_limit)
        self.decode_frame_capacity = frame_limit

    def video_info(self, frames: int) -> VideoInfo:
        validate_frames(frames)
        if frames > self.frame_limit:
            raise ValueError("H3 output exceeds configured frame capacity")
        return VideoInfo(
            frame_count=frames,
            width=self.width,
            height=self.height,
            frame_rate=self.frame_rate,
            audio_rate=self.audio_rate,
        )

    def decode_windows(self, video: VideoInfo) -> tuple[DecodeWindow, ...]:
        """Partition output into 17-frame bodies and the final five-frame overlap."""

        if video != self.video_info(video.frame_count):
            raise ValueError("H3 decode windows require the configured raster and frame rates")
        return decode_windows(video.frame_count)

    def output_layout(self, size: VideoSize) -> dict[str, OutputLayout]:
        video = self.video_info(size.frames)
        return {
            "video": OutputLayout(
                (video.frame_count, video.height, video.width, 3),
                torch.uint8,
                variable_axes=(0,),
            )
        }

    def state_buffers(self, size: VideoSize) -> dict[str, BufferConfig]:
        video = self.video_info(size.frames)
        return {"video_overlap": BufferConfig((1, 3, 5, video.height, video.width), torch.float16)}

    def workspace_buffers(self, size: VideoSize) -> dict[str, BufferConfig]:
        video = self.video_info(size.frames)
        return {
            "rgb_frames": BufferConfig(
                (video.frame_count, video.height, video.width, 3), torch.uint8
            )
        }

    def constant_buffers(self, size: VideoSize) -> dict[str, BufferConfig]:
        self.video_info(size.frames)
        return {
            name: BufferConfig((1, 3, 1, 1, 1), torch.float32)
            for name in ("pixel_mean", "pixel_std")
        }

    @torch.inference_mode()
    def prepare_constants(self, size: VideoSize, *, out: TensorViews) -> None:
        buffers = self.constant_buffers(size)
        if out.keys() != buffers.keys():
            raise ValueError("RGB constants must contain mean and standard deviation")
        for name, config in buffers.items():
            if tuple(out[name].shape) != config.shape or out[name].dtype != config.dtype:
                raise ValueError(f"RGB constant {name!r} has incompatible shape or dtype")
        for name, values in (
            ("pixel_mean", (0.485, 0.456, 0.406)),
            ("pixel_std", (0.229, 0.224, 0.225)),
        ):
            out[name].copy_(
                torch.tensor(values, dtype=torch.float32, device="cpu").view(1, 3, 1, 1, 1)
            )


def decode_windows(frames: int) -> tuple[DecodeWindow, ...]:
    """Map an H3 duration to its native windows and exact RGB frame intervals."""

    validate_frames(frames)
    units = (frames - 5) // 17
    return tuple(
        DecodeWindow(
            latent_start=unit * 5,
            latent_stop=unit * 5 + 7,
            frame_start=unit * 17,
            frame_stop=(unit + 1) * 17 + (5 if unit + 1 == units else 0),
            body_frames=17,
            overlap_frames=5,
            padding_frames=3,
            crop=(3, 0),
            final=unit + 1 == units,
        )
        for unit in range(units)
    )
