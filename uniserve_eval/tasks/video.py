"""Build MiniMax-H3 video requests and inspect complete MP4s after measurement.

Every request is the official MiniMax-H3 body that UniServe and SGLang
accept: a ``task``, its ``conditions`` and a ``target`` of short edge, aspect
ratio and duration. A t2va row carries no conditions; an fl2va or ref2va row
lists its condition media, which the body names by ``file://`` URI under the
point's ``video.condition_root``. The canvas and frame count a target
resolves to follow the model package's request rules
(``uniserve_models.minimax_h3.processing``), which the server's planner
shares; validation holds every MP4 to its own request's canvas and frames.
Media files are read when the request is built, before measurement, and
inspection uses each request's shape and never runs inside a load slot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from ..transport.native_video import native_request
from ..transport.video import VideoOutputError, inspect_video_bytes
from ..types import (
    VIDEO_SCHEDULE_FIELDS,
    VIDEOS_SYNC,
    BenchmarkPoint,
    ConditionMedia,
    Example,
    RequestRecord,
    TaskName,
    TaskRequest,
    ValidationResult,
    VideoConfig,
    VideoShape,
)
from .base import BenchmarkTask

# Every MiniMax-H3 output runs at 24 frames per second.
VIDEO_FPS = 24
# The only target short edge the canvas rule serves, in pixels.
TARGET_SHORT_EDGE = 768

# Media types of the condition files the request API accepts, by suffix:
# JPEG, PNG and WEBP images, MP4 and QuickTime videos, MP3 and WAV audio.
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
}

# The media type family each condition type requires.
_CONDITION_MEDIA = {
    "image": "image/",
    "video": "video/",
    "video_audio": "video/",
    "audio": "audio/",
}

# The EXIF orientations that turn an image by 90 degrees, so its displayed
# width is its stored height.
_TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})
_EXIF_ORIENTATION = 0x0112


def target_canvas(
    task: str, aspect_ratio: str, keyframe: tuple[int, int] | None = None
) -> tuple[int, int]:
    """Resolve the canvas of a target.

    A target names ``auto`` or one of the named ratios, which the
    adapt_shape_v1 canvas rule turns into pixels. ``auto`` is 16:9 for t2va
    and ref2va, and for fl2va the displayed aspect of the first keyframe.

    Args:
        task: ``t2va``, ``fl2va`` or ``ref2va``.
        aspect_ratio: The target's aspect ratio.
        keyframe: The first keyframe's displayed ``(width, height)``, which
            an fl2va ``auto`` target requires.

    Returns:
        The canvas as ``(width, height)`` in pixels.

    Raises:
        ValueError: The task does not accept the aspect ratio, or an fl2va
            ``auto`` target has no keyframe or one outside 1:4 to 4:1.
    """
    # The model package loads torch, which only video points need, so the
    # rules are imported on first use.
    from uniserve_models.minimax_h3 import processing

    ratio = _named_ratio(task, aspect_ratio)
    if ratio is None:
        if task != processing.Task.FL2VA:
            ratio = processing.DEFAULT_ASPECT_RATIO
        elif keyframe is None:
            raise ValueError("an fl2va auto target follows its first keyframe")
        else:
            ratio = keyframe
    size = processing.canvas(*ratio)
    return size.width, size.height


def target_frames(seconds: float) -> int:
    """Return the frames a target duration generates.

    The duration times 24 rounds half to even and aligns up to the next
    ``17 n + 5`` frames.

    Raises:
        ValueError: The duration lies outside 4 to 15 seconds.
    """
    from uniserve_models.minimax_h3 import processing

    return processing.frame_count(seconds)


def _named_ratio(task: str, aspect_ratio: str) -> tuple[int, int] | None:
    """Return a named target ratio, or ``None`` for ``auto``.

    Raises:
        ValueError: The ratio is neither ``auto`` nor a named ratio.
    """
    from uniserve_models.minimax_h3 import processing

    if aspect_ratio == "auto":
        return None
    # The API spells a ratio in canonical decimal form, so a named ratio is
    # accepted exactly as written here.
    named = {
        f"{width}:{height}": (width, height)
        for width, height in processing.NAMED_ASPECT_RATIOS
    }
    if aspect_ratio not in named:
        raise ValueError(
            f"aspect_ratio must be auto or one of {', '.join(named)} for "
            f"{task}, got {aspect_ratio!r}"
        )
    return named[aspect_ratio]


def _displayed_size(path: Path) -> tuple[int, int]:
    """Return an image's ``(width, height)`` after its EXIF orientation.

    Only the header is read; the orientation decides whether the stored
    sides swap, as the server's probe does before it plans the canvas.
    """
    with Image.open(path) as picture:
        width, height = picture.size
        orientation = picture.getexif().get(_EXIF_ORIENTATION)
    if orientation in _TRANSPOSED_ORIENTATIONS:
        return height, width
    return width, height


class VideoTask(BenchmarkTask):
    """Builds MiniMax-H3 video requests and validates the media contract."""

    name: ClassVar[TaskName] = TaskName.VIDEO
    allowed_endpoints: ClassVar[tuple[str, ...]] = (VIDEOS_SYNC, "/v1/videos")
    default_endpoint: ClassVar[str] = VIDEOS_SYNC
    default_stream: ClassVar[bool] = False

    def __init__(self, point: BenchmarkPoint) -> None:
        """Bind the point to the condition media root it reads rows from."""
        super().__init__(point)
        root = point.video.condition_root
        self.condition_root = Path(root).resolve() if root else None

    @classmethod
    def check_video(cls, video: VideoConfig, context: str) -> None:
        """Reject a target ratio the task does not accept."""
        try:
            _named_ratio(video.task, video.aspect_ratio)
        except ValueError as error:
            raise ValueError(f"{context}: {error}") from error

    def build_request(self, example: Example) -> TaskRequest:
        """Build one request from the row's duration, seed and conditions.

        A row without its own duration or seed takes the point's video
        duration and load seed. The stated schedule fields are sent when the
        point sets them. The row's condition media are read here, so a
        backend that uploads them sends bytes already in memory.

        Raises:
            ValueError: The duration lies outside 4 to 15 seconds, the
                row's conditions do not fit the point's task, or the backend
                cannot be sent them as the same work.
            OSError: A condition media file cannot be read.
        """
        video = self.point.video
        seconds = float(
            example.seconds if example.seconds is not None else video.seconds
        )
        frames = target_frames(seconds)
        conditions, media = self._conditions(example)

        # Only an fl2va auto target reads its canvas from the first keyframe,
        # whose displayed size the server probes the same way.
        keyframe = (
            _displayed_size(Path(media[0].path))
            if video.task == "fl2va"
            else None
        )
        width, height = target_canvas(video.task, video.aspect_ratio, keyframe)
        payload = {
            "model": self.point.model,
            "prompt": example.prompt,
            "task": video.task,
            "conditions": conditions,
            "target": {
                "short_edge": TARGET_SHORT_EDGE,
                "aspect_ratio": video.aspect_ratio,
                "duration_seconds": seconds,
            },
            "seed": int(
                example.seed
                if example.seed is not None
                else self.point.load.seed
            ),
        }
        for name in VIDEO_SCHEDULE_FIELDS:
            value = getattr(video, name)
            if value is not None:
                payload[name] = value
        request = TaskRequest(
            self.point.endpoint,
            payload,
            stream=False,
            video_backend=video.backend,
            poll_interval_s=video.poll_interval_s,
            video_extra_params=dict(video.extra_params),
            video_shape=VideoShape(width=width, height=height, frames=frames),
            condition_media=tuple(media),
            video_parallel_decoding=video.parallel_decoding,
        )
        # A request its backend cannot take as the same work fails here,
        # before measurement, rather than as a measured transport failure.
        native_request(request)
        return request

    def _conditions(
        self, example: Example
    ) -> tuple[list[dict[str, Any]], list[ConditionMedia]]:
        """Return a row's request conditions and their media, in row order.

        Each condition names its file by an absolute ``file://`` URI. The
        rows' structure is checked only as far as the canvas and the
        backends' request forms depend on it: t2va takes no conditions,
        fl2va only image keyframes, and ref2va at least one reference. The
        server enforces the remaining request rules.

        Raises:
            ValueError: The conditions do not fit the point's task.
            OSError: A media file cannot be read.
        """
        task = self.point.video.task
        rows = example.conditions or []
        if task == "t2va":
            if rows:
                raise ValueError(f"row {example.id}: t2va takes no conditions")
            return [], []
        if not rows:
            raise ValueError(f"row {example.id}: {task} requires conditions")

        assert self.condition_root is not None
        conditions: list[dict[str, Any]] = []
        media: list[ConditionMedia] = []
        for index, row in enumerate(rows):
            context = f"row {example.id} conditions[{index}]"
            kind, role = row.get("type"), row.get("role")
            if kind not in _CONDITION_MEDIA:
                raise ValueError(f"{context}: unknown type {kind!r}")
            if role not in {"keyframe", "reference"}:
                raise ValueError(f"{context}: unknown role {role!r}")
            if task == "fl2va" and (role != "keyframe" or kind != "image"):
                raise ValueError(f"{context}: fl2va takes image keyframes")

            path = (self.condition_root / str(row["media"])).resolve()
            mime = MEDIA_TYPES.get(path.suffix.lower())
            if mime is None or not mime.startswith(_CONDITION_MEDIA[kind]):
                raise ValueError(f"{context}: {path.name} is not {kind} media")

            condition: dict[str, Any] = {
                "type": kind,
                "uri": path.as_uri(),
                "role": role,
            }
            for name in ("frame_index", "start_time_seconds"):
                if row.get(name) is not None:
                    condition[name] = row[name]
            conditions.append(condition)
            media.append(ConditionMedia(str(path), mime, path.read_bytes()))

        if task == "ref2va" and not any(
            condition["role"] == "reference" for condition in conditions
        ):
            raise ValueError(f"row {example.id}: ref2va requires a reference")
        return conditions, media

    def inspect_output(self, record: RequestRecord) -> None:
        """Decode and classify a response outside the load window."""
        if not record.success:
            return
        try:
            record.decoded_video = inspect_video_bytes(
                record.video_body or b"", declared_mime=record.video_mime
            )
        except VideoOutputError as error:
            record.mark_failure(error.classifier, str(error))
        record.media_checks = self.validate_output([record]).checks
        if not all(record.media_checks.values()):
            failed = [
                key for key, passed in record.media_checks.items() if not passed
            ]
            if record.success:
                record.mark_failure("invalid_video_output", ", ".join(failed))

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Validate canvas, codecs, audio, and duration alignment.

        Each output is held to the canvas and frame count its own request
        resolved to (``RequestRecord.video_shape``); a record without one
        fails every shape check.
        """
        outputs = [record.decoded_video for record in records]
        shapes = [record.video_shape for record in records]
        present = (
            bool(outputs)
            and all(output is not None for output in outputs)
            and all(shape is not None for shape in shapes)
        )
        pairs = [
            (video, shape)
            for video, shape in zip(outputs, shapes, strict=True)
            if video is not None and shape is not None
        ]
        videos = [video for video, _ in pairs]

        # Both streams are measured against the aligned media length rather
        # than each other: the video may differ from the aligned frame count
        # by one frame, and the audio duration from the aligned duration by
        # one frame period. A one-frame-short video with full-length audio is
        # therefore valid, while a missing or truncated stream is not.
        frame_tolerance = 1
        duration_tolerance_s = 1.0 / VIDEO_FPS
        return ValidationResult(
            checks={
                "decoded_video": present,
                "h264_video": present
                and all(video.video_codec == "h264" for video in videos),
                "target_canvas": present
                and all(
                    (video.width, video.height) == (shape.width, shape.height)
                    for video, shape in pairs
                ),
                # The container frame rate is a rational; require exactly 24.
                "24fps_video": present
                and all(
                    video.fps_numerator == VIDEO_FPS * video.fps_denominator
                    for video in videos
                ),
                "aligned_frame_count": present
                and all(
                    abs(video.frame_count - shape.frames) <= frame_tolerance
                    for video, shape in pairs
                ),
                "aac_stereo_32khz_audio": present
                and all(
                    video.audio_codec == "aac"
                    and video.audio_channels == 2
                    and video.audio_sample_rate == 32_000
                    for video in videos
                ),
                "aligned_audio_duration": present
                and all(
                    abs(video.audio_duration_s - shape.frames / VIDEO_FPS)
                    <= duration_tolerance_s
                    for video, shape in pairs
                ),
                "nonzero_video_variance": present
                and all(
                    math.isfinite(video.video_variance)
                    and video.video_variance > 0
                    for video in videos
                ),
                "nonzero_audio_rms": present
                and all(
                    math.isfinite(video.audio_rms) and video.audio_rms > 0
                    for video in videos
                ),
            },
            statistics={
                "completed_videos": len(videos),
                "media_bytes": sum(video.byte_size for video in videos),
            },
        )


__all__ = ["MEDIA_TYPES", "VideoTask", "target_canvas", "target_frames"]
