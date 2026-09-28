"""H3 output rasters, sampling clocks and RGB reconstruction.

A video of ``17 * n + 5`` frames is decoded as ``n`` media units. Each unit
decodes to 25 frames: a 17-frame body, three VAE padding frames, and a
five-frame tail. The next unit cross-fades that tail into its leading
frames; the last unit's tail ends the video. ``VideoPostprocessor`` supplies
these slices and blend weights.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from uniserve.media import image, video
from uniserve.model import VideoPostprocessor as BaseVideoPostprocessor
from uniserve.tensors import BufferConfig, OutputLayout


@dataclass(frozen=True, slots=True)
class Config:
    """Output rasters and the video frame and audio sample clocks in Hz.

    ``frame_sizes`` lists the served rasters: 16:9 landscape first, then
    9:16 portrait.
    """

    frame_sizes: tuple[image.Config, ...] = (
        image.Config(768, 1344),
        image.Config(1344, 768),
    )
    frame_rate: int = 24
    sample_rate: int = 32000

    def __post_init__(self):
        if not self.frame_sizes or any(
            not isinstance(frame, image.Config) for frame in self.frame_sizes
        ):
            raise ValueError("H3 output requires at least one raster")
        if any(
            type(value) is not int or value < 1
            for value in (self.frame_rate, self.sample_rate)
        ):
            raise ValueError("media sampling clocks must be positive integers")


def frame_slices(num_frames: int) -> tuple[slice, ...]:
    """Partition a complete H3 timeline into 17-frame bodies and its final overlap."""  # noqa: E501
    if type(num_frames) is not int or num_frames < 22 or num_frames % 17 != 5:
        raise ValueError(
            "H3 frame count must have the form 17 * n + 5 with n positive"
        )
    count = (num_frames - 5) // 17
    # Every unit covers a 17-frame body; the last unit adds the 5-frame tail.
    return tuple(
        slice(index * 17, (index + 1) * 17 + (5 if index + 1 == count else 0))
        for index in range(count)
    )


class VideoPostprocessor(BaseVideoPostprocessor):
    """Remove H3's three-frame decoder padding and cross-fade five-frame overlaps."""  # noqa: E501

    def __init__(self, *, frame_rate: int):
        # Weight of the current unit across the five overlap frames: 0, 0.2,
        # ..., 0.8, so the first blended frame is entirely the predecessor's.
        weights = torch.arange(5, device="cpu", dtype=torch.float16) / 5
        super().__init__(weights, frame_rate=frame_rate)

    def reconstruction_slices(
        self, frames: slice, num_frames: int
    ) -> tuple[slice, slice]:
        if frames not in frame_slices(num_frames):
            raise ValueError(
                "H3 RGB reconstruction requires a complete legal frame slice"
            )
        # Each decoded unit is 25 frames: a 17-frame body to keep, three
        # VAE padding frames to drop, and a five-frame tail overlapping the
        # next unit.
        return slice(0, 17), slice(20, 25)

    def output_layout(self, size: video.Config) -> Mapping[str, OutputLayout]:
        num_frames = size.num_frames
        frame_slices(num_frames)
        height, width = size.frame.height, size.frame.width
        return {
            "video": OutputLayout(
                (num_frames, height, width, 3),
                torch.uint8,
                (
                    slice(0, num_frames),
                    slice(0, height),
                    slice(0, width),
                    slice(0, 3),
                ),
                variable_axes=(0,),
                value_range=(0, 255),
            )
        }

    def state_buffers(self, size: video.Config) -> Mapping[str, BufferConfig]:
        frame_slices(size.num_frames)
        return {
            "video_overlap": BufferConfig(
                (1, 3, 5, size.frame.height, size.frame.width),
                torch.float16,
            )
        }

    def workspace_buffers(
        self, size: video.Config
    ) -> Mapping[str, BufferConfig]:
        layout = self.output_layout(size)["video"]
        # One call reconstructs one media unit, so the RGB workspace holds the
        # longest unit rather than the whole timeline; the last unit is the
        # longest because it keeps its own overlap tail.
        unit = max(
            window.stop - window.start
            for window in frame_slices(size.num_frames)
        )
        return {
            "rgb_frames": BufferConfig((unit, *layout.shape[1:]), layout.dtype),
            # Receives the predecessor media unit's overlap from the ring.
            "overlap_exchange": BufferConfig(
                (1, 3, 5, size.frame.height, size.frame.width),
                torch.float16,
            ),
        }

    def constant_buffers(
        self, size: video.Config
    ) -> Mapping[str, BufferConfig]:
        frame_slices(size.num_frames)
        return {
            name: BufferConfig((1, 3, 1, 1, 1), torch.float32)
            for name in ("pixel_mean", "pixel_std")
        }

    def prepare_constants(
        self, size: video.Config, *, out: Mapping[str, torch.Tensor]
    ) -> None:
        buffers = self.constant_buffers(size)
        if out.keys() != buffers.keys():
            raise ValueError(
                "RGB constants must contain mean and standard deviation"
            )
        # The decoder emits normalized RGB; the post-processor restores it as
        # ``value * std + mean`` per channel before clamping and quantizing
        # to uint8.
        for name, values in (
            ("pixel_mean", (0.485, 0.456, 0.406)),
            ("pixel_std", (0.229, 0.224, 0.225)),
        ):
            if (
                out[name].shape != buffers[name].shape
                or out[name].dtype != buffers[name].dtype
            ):
                raise ValueError(
                    f"RGB constant {name!r} has incompatible shape or dtype"
                )
            out[name].copy_(
                torch.tensor(values, device="cpu", dtype=torch.float32).view(
                    1, 3, 1, 1, 1
                )
            )
