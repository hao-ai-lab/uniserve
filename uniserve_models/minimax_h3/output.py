"""H3 RGB reconstruction.

A video of ``17 * n + 5`` frames is decoded as ``n`` clips. Each clip decodes
to 25 frames: a 17-frame body, three VAE padding frames, and a five-frame
tail. The next clip cross-fades that tail into its leading frames; the last
clip's tail ends the video. ``VideoPostprocessor`` supplies these slices and
blend weights.
"""

from collections.abc import Mapping

import torch

from uniserve.media import video
from uniserve.model import VideoPostprocessor as BaseVideoPostprocessor
from uniserve.tensors import BufferConfig, OutputLayout

from .packing import CLIP_FRAMES, CLIP_LATENTS


def video_clips(num_frames: int) -> tuple[slice, ...]:
    """Partition a complete H3 timeline into 17-frame bodies and its final overlap."""  # noqa: E501
    body, tail = CLIP_FRAMES, CLIP_LATENTS
    if (
        type(num_frames) is not int
        or num_frames < body + tail
        or num_frames % body != tail
    ):
        raise ValueError(
            "H3 frame count must have the form 17 * n + 5 with n positive"
        )
    count = (num_frames - tail) // body
    # Every clip covers a 17-frame body; the last clip adds the 5-frame tail.
    return tuple(
        slice(
            index * body,
            (index + 1) * body + (tail if index + 1 == count else 0),
        )
        for index in range(count)
    )


class VideoPostprocessor(BaseVideoPostprocessor):
    """Remove H3's three-frame decoder padding and cross-fade five-frame overlaps."""  # noqa: E501

    def __init__(self, *, frame_rate: int):
        # Weight of the current clip across the five overlap frames: 0, 0.2,
        # ..., 0.8, so the first blended frame is entirely the predecessor's.
        weights = torch.arange(5, device="cpu", dtype=torch.float16) / 5
        super().__init__(weights, frame_rate=frame_rate)

    def reconstruction_slices(
        self, clip: slice, num_frames: int
    ) -> tuple[slice, slice]:
        if clip not in video_clips(num_frames):
            raise ValueError(
                "H3 RGB reconstruction requires one of the video's clips"
            )
        # Each decoded clip is 25 frames: a 17-frame body to keep, three
        # VAE padding frames to drop, and a five-frame tail overlapping the
        # next clip.
        return slice(0, 17), slice(20, 25)

    def output_layout(self, size: video.Config) -> Mapping[str, OutputLayout]:
        num_frames = size.num_frames
        video_clips(num_frames)
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
        video_clips(size.num_frames)
        return {
            "video_overlap": BufferConfig(
                (1, 3, 5, size.frame.height, size.frame.width), torch.float16
            )
        }

    def workspace_buffers(
        self, size: video.Config
    ) -> Mapping[str, BufferConfig]:
        layout = self.output_layout(size)["video"]
        # One call reconstructs one clip, so the RGB workspace holds the
        # longest clip rather than the whole timeline; the last clip is the
        # longest because it keeps its own overlap tail.
        longest = max(
            clip.stop - clip.start for clip in video_clips(size.num_frames)
        )
        return {
            "rgb_frames": BufferConfig(
                (longest, *layout.shape[1:]), layout.dtype
            ),
            # Receives the predecessor clip's overlap from the ring.
            "overlap_exchange": BufferConfig(
                (1, 3, 5, size.frame.height, size.frame.width), torch.float16
            ),
        }

    def constant_buffers(
        self, size: video.Config
    ) -> Mapping[str, BufferConfig]:
        video_clips(size.num_frames)
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
