"""Recorded MiniMax-H3 reference requests as a worker admits them.

A recorded diffusers reference run holds its request and the official media
it conditions on. These helpers probe that media, plan the request with
``processing.plan_request``, the planner the server's rules are checked
against, describe each condition as admission carries it, and read the
conditions through the media reader into products sized as the engine
reserves them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import torch

from uniserve.media import image
from uniserve.model import ConditionRole
from uniserve_models import qwen3_vl
from uniserve_models.minimax_h3 import processing, video_vae
from uniserve_models.minimax_h3.encoder import TextEncoderConfig
from uniserve_models.minimax_h3.encoding import VideoEncoder
from uniserve_worker.execution.media_reader import read_conditions
from uniserve_worker.media.reader import SHARED_MEMORY
from uniserve_worker.protocol.video import (
    AudioClip,
    ConditionVision,
    ImageFit,
    MediaLocator,
    VideoClip,
    VideoCondition,
)

_VISION = processing.VisionConfig(
    **json.loads((Path(__file__).parent / "minimax_h3_plan.json").read_text())[
        "vision"
    ]
)
# The model's audio rate, the rate every soundtrack is resampled to.
SAMPLE_RATE = 32_000
FPS = 24


def vision_encoder() -> qwen3_vl.VisionEncoder:
    """The conditioner's vision encoder; packing reads no parameter."""
    config = TextEncoderConfig()
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


def _audio_facts(path: Path) -> processing.AudioFacts | None:
    from diffusers.modular_pipelines.minimax_h3 import references

    try:
        waveform, rate = references._decode_audio_file(path)
    except ValueError:
        return None
    return processing.AudioFacts(rate, int(waveform.shape[1]))


def _video_facts(path: Path, ffmpeg: str) -> processing.VideoFacts:
    """Probe a video's display aspect, frame rate, frames and soundtrack."""
    stream = json.loads(
        subprocess.run(
            [
                str(Path(ffmpeg).with_name("ffprobe")),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    )["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    aspect = stream.get("sample_aspect_ratio", "1:1")
    if aspect not in ("0:1", "N/A"):
        numerator, denominator = (int(part) for part in aspect.split(":"))
        width = width * numerator // denominator
    if any(
        int(data.get("rotation", 0)) % 180
        for data in stream.get("side_data_list", ())
    ):
        width, height = height, width
    return processing.VideoFacts(
        display_width=width,
        display_height=height,
        frame_rate=Fraction(stream["avg_frame_rate"]),
        frames=int(stream["nb_read_frames"]),
        soundtrack=_audio_facts(path),
    )


def _facts(path: Path, kind: processing.ConditionType, ffmpeg: str):
    from PIL import Image, ImageOps

    if kind is processing.ConditionType.IMAGE:
        with Image.open(path) as value:
            width, height = ImageOps.exif_transpose(value).size
        return processing.ImageFacts(width, height)
    if kind is processing.ConditionType.AUDIO:
        return _audio_facts(path)
    return _video_facts(path, ffmpeg)


def plan(request: dict, root: Path, ffmpeg: str):
    """Plan a recorded request from its media's probed facts."""
    conditions = []
    for item in request["conditions"]:
        kind = processing.ConditionType(item["type"])
        conditions.append(
            processing.Condition(
                type=kind,
                role=processing.Role(item["role"]),
                media=_facts(media_file(root, item["uri"]), kind, ffmpeg),
                frame_index=item.get("frame_index"),
                start_seconds=item.get("start_time_seconds"),
            )
        )
    target = processing.Target(**request["target"])
    return processing.plan_request(
        processing.Task(request["task"]), target, conditions, _VISION
    )


def _audio(clip: processing.AudioClip) -> AudioClip:
    return AudioClip(
        clip.sample_rate, clip.start_sample, clip.source_samples, clip.samples
    )


def describe(
    plan: processing.ConditionPlan,
    canvas: image.Config,
    source: MediaLocator,
    image_bands: int = 1,
) -> VideoCondition:
    """Describe one planned condition as admission carries it.

    A reference image is encoded in ``image_bands`` bands of its patch rows,
    as the server bands it for a latent encoder of that many units a round.
    """
    with torch.device("meta"):
        encoder = VideoEncoder(video_vae.Config())
    prepared, seen = plan.prepared, plan.vision
    fit = video = audio = None
    role = ConditionRole.REFERENCE
    if isinstance(prepared, processing.KeyframeFit):
        role = (
            ConditionRole.FIRST_FRAME
            if prepared.position is processing.FramePosition.FIRST
            else ConditionRole.LAST_FRAME
        )
        crop = prepared.cover_crop
        fit = (
            ImageFit(canvas, 0, 0, canvas)
            if crop is None
            else ImageFit(
                image.Config(crop.height, crop.width),
                crop.left,
                crop.top,
                canvas,
            )
        )
    elif isinstance(prepared, image.Config):
        fit = ImageFit(prepared, 0, 0, prepared)
    elif isinstance(prepared, processing.VideoClip):
        video = VideoClip(
            prepared.canvas,
            prepared.start_frame,
            prepared.frames,
            prepared.vae_frames,
        )
        if prepared.soundtrack is not None:
            audio = _audio(prepared.soundtrack)
    else:
        audio = _audio(prepared)

    # Each temporal unit of the encoded pixels yields its latent frames'
    # rows; a keyframe is one unit of one latent frame and a reference image
    # one unit per band of its patch rows.
    if isinstance(prepared, image.Config):
        per_group = processing.rows_per_frame(prepared) // (
            prepared.height // processing.CANVAS_MULTIPLE
        )
        units = tuple(
            (band.stop - band.start) // encoder.row_group * per_group
            for band in encoder.row_bands(prepared, image_bands)
        )
    elif fit is not None:
        units = (processing.rows_per_frame(fit.size),)
    elif video is not None:
        units = tuple(
            (window.stop - window.start)
            * processing.rows_per_frame(video.canvas)
            for window in encoder.latent_slices(video.vae_frames)
        )
    else:
        units = ()
    if isinstance(seen, processing.VideoVision):
        view = ConditionVision(
            seen.grid, seen.block_tokens * seen.grid[0], seen.frame_indices
        )
    elif seen is not None:
        view = ConditionVision(seen.grid, seen.tokens)
    else:
        view = None
    return VideoCondition(
        role=role,
        source=source,
        image=fit,
        video=video,
        audio=audio,
        vision=view,
        latent_units=units,
        audio_rows=plan.audio_rows,
    )


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
        ffmpeg=ffmpeg,
    )
    return pixels, samples, patches


def recorded_request(
    root: Path, case: str, ffmpeg: str, publish, image_bands: int = 1
):
    """Plan a recorded run's request and describe its conditions.

    Returns the run directory, the plan, each condition's media file and
    the conditions as admission carries them, published with ``publish``,
    reference images in ``image_bands`` bands.
    """
    run = root / "reference" / "diffusers" / case / "seed42"
    request = json.loads((run / "request.json").read_text())
    planned = plan(request, root, ffmpeg)
    paths = [media_file(root, item["uri"]) for item in request["conditions"]]
    conditions = tuple(
        describe(condition, planned.canvas, publish(path), image_bands)
        for condition, path in zip(planned.conditions, paths, strict=True)
    )
    return run, planned, paths, conditions
