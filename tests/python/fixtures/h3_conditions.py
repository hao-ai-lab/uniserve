"""Recorded MiniMax-H3 reference requests as a worker admits them.

A recorded diffusers reference run holds its request and the official media
it conditions on. Each recorded case states the sizes the reference
conditioned it at (the canvas, the frame count, how every condition is
prepared and what the conditioner reads of it, as the recorded run's
``metadata.json``, ``presentation.json`` and condition latents show). These
helpers describe each condition as admission carries it and read the
conditions through the media reader into products sized as the engine
reserves them.
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from tests.python.fixtures.h3 import text_encoder_config
from uniserve.media import image
from uniserve.model import ConditionRole
from uniserve_models import qwen3_vl
from uniserve_models.minimax_h3.packing import AUDIO_SAMPLE_RATE, FPS
from uniserve_worker.execution.media_reader import read_conditions
from uniserve_worker.media.reader import SHARED_MEMORY
from uniserve_worker.protocol.video import (
    AudioClip,
    ConditionVision,
    ImageFit,
    MediaLocator,
    VideoClip,
    VideoCondition,
    VideoTask,
)

# The model's audio rate, the rate every soundtrack is resampled to.
SAMPLE_RATE = AUDIO_SAMPLE_RATE

# The 16:9 canvas of the canvas rule: 24 x 42 denoiser tokens of 32 pixels,
# which the conditioner reads as a 48 x 84 grid of 16-pixel patches merged
# 2 x 2 into 1008 vision tokens.
_CANVAS = image.Config(768, 1344)
_CANVAS_ROWS = (768 // 32) * (1344 // 32)


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    """What the reference generated: its task, canvas and frame count."""

    task: VideoTask
    canvas: image.Config
    num_frames: int


# The official soundtrack is 44.1 kHz. Five seconds keep its first 227,850
# native samples, which resample to 165,334 samples at 32 kHz and pad to 207
# audio latents per stereo channel.
_SOUNDTRACK = AudioClip(
    sample_rate=44_100, start_sample=0, source_samples=227_850, samples=165_334
)
_SOUNDTRACK_ROWS = 2 * 207


def _keyframe(role: ConditionRole, source: MediaLocator) -> VideoCondition:
    """A keyframe stretched onto the canvas: one image unit of one frame."""
    return VideoCondition(
        role=role,
        source=source,
        image=ImageFit(_CANVAS, 0, 0, _CANVAS),
        video=None,
        audio=None,
        vision=ConditionVision((1, 48, 84), _CANVAS_ROWS),
        latent_units=(_CANVAS_ROWS,),
        audio_rows=0,
    )


def _image_reference(source: MediaLocator, image_bands: int) -> VideoCondition:
    """A 16:9 image reference enlarged to its 2048-pixel short edge.

    It is 64 patch rows of 114 patches. The patch rows divide evenly into the
    bands a latent encoder of ``image_bands`` units takes.
    """
    size = image.Config(2048, 3648)
    return VideoCondition(
        role=ConditionRole.REFERENCE,
        source=source,
        image=ImageFit(size, 0, 0, size),
        video=None,
        audio=None,
        vision=ConditionVision((1, 128, 228), 7296),
        latent_units=(64 // image_bands * 114,) * image_bands,
        audio_rows=0,
    )


def _video_reference(source: MediaLocator) -> VideoCondition:
    """The official 16:9 clip's 124 frames with its soundtrack.

    The video encoder encodes seven 17-frame windows of five latent frames
    and a padded last one of two. The conditioner samples every twelfth frame
    into six blocks of 1008 vision tokens.
    """
    return VideoCondition(
        role=ConditionRole.REFERENCE,
        source=source,
        image=None,
        video=VideoClip(_CANVAS, start_frame=0, frames=124, vae_frames=124),
        audio=_SOUNDTRACK,
        vision=ConditionVision(
            (6, 48, 84), 6 * _CANVAS_ROWS, tuple(range(0, 121, 12))
        ),
        latent_units=(5 * _CANVAS_ROWS,) * 7 + (2 * _CANVAS_ROWS,),
        audio_rows=_SOUNDTRACK_ROWS,
    )


def _audio_reference(source: MediaLocator) -> VideoCondition:
    return VideoCondition(
        role=ConditionRole.REFERENCE,
        source=source,
        image=None,
        video=None,
        audio=_SOUNDTRACK,
        vision=None,
        latent_units=(),
        audio_rows=_SOUNDTRACK_ROWS,
    )


def vision_encoder() -> qwen3_vl.VisionEncoder:
    """The conditioner's vision encoder; packing reads no parameter."""
    config = text_encoder_config()
    with torch.device("meta"):
        return qwen3_vl.VisionEncoder(
            config.vision, config.pixels, dtype=torch.bfloat16
        )


@contextmanager
def published() -> Iterator:
    """Publish media files to shared memory as the server does.

    Yields a function from a file path to its ``MediaLocator``; every
    object it published is unlinked on exit.
    """
    names = []

    def publish(path: Path) -> MediaLocator:
        name = f"uniserve-test-media-{uuid.uuid4().hex}"
        shutil.copyfile(path, SHARED_MEMORY / name)
        names.append(name)
        return MediaLocator(name, path.stat().st_size)

    try:
        yield publish
    finally:
        for name in names:
            (SHARED_MEMORY / name).unlink(missing_ok=True)


def media_file(root: Path, uri: str) -> Path:
    return root / "inputs" / "media" / Path(uri).name


def read(conditions, vision, ffmpeg):
    """Read conditions into products sized as the engine reserves them."""

    def rows(count, width, dtype):
        return torch.empty((count, width), dtype=dtype) if count else None

    pixels = rows(
        sum(condition.pixel_bytes // 3 for condition in conditions),
        3,
        torch.uint8,
    )
    samples = rows(
        sum(
            condition.audio.samples
            for condition in conditions
            if condition.audio
        ),
        2,
        torch.float32,
    )
    patches = rows(
        sum(
            condition.vision.patches
            for condition in conditions
            if condition.vision
        ),
        vision.pixels_layout(1).shape[1],
        torch.float32,
    )
    read_conditions(
        conditions,
        pixels=pixels,
        samples=samples,
        patches=patches,
        vision=vision,
        sample_rate=SAMPLE_RATE,
        frame_rate=FPS,
        ffmpeg=ffmpeg,
    )
    return pixels, samples, patches


def recorded_request(root: Path, case: str, publish, image_bands: int = 1):
    """Describe a recorded run's request and its conditions.

    Returns the run directory, the request's task, canvas and frame count,
    each condition's media file and the conditions as admission carries
    them, published with ``publish``, reference images in ``image_bands``
    bands.
    """
    run = root / "reference" / "diffusers" / case / "seed42"
    items = json.loads((run / "request.json").read_text())["conditions"]
    paths = [media_file(root, item["uri"]) for item in items]
    sources = [publish(path) for path in paths]
    if case == "fl2va_first_8s":
        request = RecordedRequest(VideoTask.FL2VA, _CANVAS, 192)
        conditions = (_keyframe(ConditionRole.FIRST_FRAME, sources[0]),)
    elif case == "ref2va_image_audio_5s":
        request = RecordedRequest(VideoTask.REF2VA, _CANVAS, 124)
        conditions = (
            _image_reference(sources[0], image_bands),
            _audio_reference(sources[1]),
        )
    elif case == "ref2va_video_audio_5s":
        request = RecordedRequest(VideoTask.REF2VA, _CANVAS, 124)
        conditions = (
            _video_reference(sources[0]),
            _audio_reference(sources[1]),
        )
    else:
        raise KeyError(f"no recorded sizes for the case {case!r}")
    assert len(conditions) == len(paths)
    return run, request, paths, conditions
