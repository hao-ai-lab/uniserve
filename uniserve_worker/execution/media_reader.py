"""Host execution of a video request's media reading.

The media reader runs on a host rank of the head host, where the server
published each condition's fetched bytes to shared memory. One media reading
call per conditioned request decodes every condition exactly as the server
planned it (``uniserve_worker.media.reader``) and writes the request's
condition products (``uniserve_worker.execution.conditions``):

- ``condition_pixels``: each visual condition's RGB24 frames the video
  encoder encodes, ``[pixels, 3]`` uint8, frame-major;
- ``condition_samples``: each audio track's model-rate stereo PCM,
  ``[samples, 2]`` FP32;
- ``vision_pixels``: the vision encoder's packed patch rows of each
  condition the conditioner reads (``PatchEncoder.pack_pixels``) from its
  decoded frames: an image's one frame, or a video's sampled frames.

Conditions follow each other in request order. Native host tasks retain the
borrowed output tensors until reading completes, then export them together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from uniserve.model import PatchEncoder
from uniserve_worker.media.reader import (
    open_media,
    read_audio,
    read_bytes,
    read_image,
    read_video,
)

if TYPE_CHECKING:
    from uniserve_worker.protocol.video import VideoCondition


def read_conditions(
    conditions: tuple[VideoCondition, ...],
    *,
    pixels: torch.Tensor | None,
    samples: torch.Tensor | None,
    patches: torch.Tensor | None,
    vision: PatchEncoder,
    sample_rate: int,
    ffmpeg: str,
) -> None:
    """Decode a request's conditions into its condition products.

    ``pixels``, ``samples`` and ``patches`` are the request's
    ``condition_pixels``, ``condition_samples`` and ``vision_pixels``
    storage, each None when no condition contributes to it. ``vision``
    packs the conditioner's patch rows, ``sample_rate`` is the audio
    encoder's rate and ``ffmpeg`` the executable reference videos decode
    with.

    Raises:
        ValueError: A condition decodes to other extents than planned, or
            the conditions do not fill the products exactly.
    """
    rows = {"pixels": 0, "samples": 0, "patches": 0}

    def fill(name: str, target: torch.Tensor | None, value: torch.Tensor):
        # Each product is filled in request order from its first row.
        if target is None:
            raise ValueError(f"a condition's {name} have no product")
        start = rows[name]
        if start + value.shape[0] > target.shape[0]:
            raise ValueError(f"the request's conditions exceed their {name}")
        target[start : start + value.shape[0]].copy_(value)
        rows[name] = start + value.shape[0]

    for condition in conditions:
        source = condition.source
        frames = None
        if condition.image is not None:
            frames = read_image(read_bytes(source), condition.image)
        elif condition.video is not None:
            with open_media(source) as media:
                frames = read_video(media, condition.video, ffmpeg=ffmpeg)

        if frames is not None:
            # The video encoder encodes a video's leading frames.
            encoded = condition.pixels
            assert encoded is not None
            fill(
                "pixels",
                pixels,
                torch.from_numpy(frames[: encoded.num_frames]).reshape(-1, 3),
            )
            if condition.vision is not None:
                view = condition.vision
                # The conditioner reads an image's one frame and a video's
                # sampled frames; indexing copies them.
                sampled = (
                    frames
                    if condition.video is None
                    else frames[list(view.frame_indices)]
                )
                fill(
                    "patches",
                    patches,
                    vision.pack_pixels(torch.from_numpy(sampled), view.grid),
                )

        if condition.audio is not None:
            # An audio reference's track is its file's; a video's
            # soundtrack is its container's first audio stream.
            with open_media(source) as media:
                fill(
                    "samples",
                    samples,
                    read_audio(media, condition.audio, rate=sample_rate),
                )

    for name, target in (
        ("pixels", pixels),
        ("samples", samples),
        ("patches", patches),
    ):
        if target is not None and rows[name] != target.shape[0]:
            raise ValueError(f"the request's conditions leave {name} unfilled")
