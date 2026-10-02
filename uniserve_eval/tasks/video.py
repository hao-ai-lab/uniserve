"""Build MiniMax-H3 video requests and inspect complete MP4s after measurement.

Every request is the official MiniMax-H3 body that UniServe and SGLang
accept: a ``task``, its ``conditions`` and a ``target`` of short edge, aspect
ratio and duration. The canvas and frame count a target resolves to follow
the model package's request rules (``uniserve_models.minimax_h3.processing``),
which the server's planner shares; validation holds every MP4 to them.
Inspection uses each request's duration and never runs inside a load slot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from ..transport.video import VideoOutputError, inspect_video_bytes
from ..types import (
    VIDEO_SCHEDULE_FIELDS,
    VIDEOS_SYNC,
    BenchmarkPoint,
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


def target_canvas(task: str, aspect_ratio: str) -> tuple[int, int]:
    """Resolve the canvas of a target without conditions.

    A t2va target names ``auto``, which is 16:9, or one of the named ratios;
    the adapt_shape_v1 canvas rule turns the ratio into pixels.

    Returns:
        The canvas as ``(width, height)`` in pixels.

    Raises:
        ValueError: The task does not accept the aspect ratio.
    """
    # The model package loads torch, which only video points need, so the
    # rules are imported on first use.
    from uniserve_models.minimax_h3 import processing

    if task != processing.Task.T2VA:
        raise ValueError(f"{task} targets depend on condition media")
    # The API spells a ratio in canonical decimal form, so a named ratio is
    # accepted exactly as written here.
    named = {
        f"{width}:{height}": (width, height)
        for width, height in processing.NAMED_ASPECT_RATIOS
    }
    if aspect_ratio == "auto":
        ratio = processing.DEFAULT_ASPECT_RATIO
    elif aspect_ratio in named:
        ratio = named[aspect_ratio]
    else:
        raise ValueError(
            f"aspect_ratio must be auto or one of {', '.join(named)} for "
            f"{task}, got {aspect_ratio!r}"
        )
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


class VideoTask(BenchmarkTask):
    """Builds MiniMax-H3 video requests and validates the media contract."""

    name: ClassVar[TaskName] = TaskName.VIDEO
    allowed_endpoints: ClassVar[tuple[str, ...]] = (VIDEOS_SYNC, "/v1/videos")
    default_endpoint: ClassVar[str] = VIDEOS_SYNC
    default_stream: ClassVar[bool] = False

    def __init__(self, point: BenchmarkPoint) -> None:
        """Bind the point and resolve the canvas every request targets."""
        super().__init__(point)
        self.canvas = target_canvas(point.video.task, point.video.aspect_ratio)

    @classmethod
    def check_video(cls, video: VideoConfig, context: str) -> None:
        """Reject a target the task cannot build or the canvas rule refuses."""
        try:
            target_canvas(video.task, video.aspect_ratio)
        except ValueError as error:
            raise ValueError(f"{context}: {error}") from error

    def build_request(self, example: Example) -> TaskRequest:
        """Build one request from the row's duration and seed.

        A row without its own duration or seed takes the point's video
        duration and load seed. The stated schedule fields are sent when the
        point sets them.

        Raises:
            ValueError: The duration lies outside 4 to 15 seconds.
        """
        video = self.point.video
        seconds = float(
            example.seconds if example.seconds is not None else video.seconds
        )
        frames = target_frames(seconds)
        payload = {
            "model": self.point.model,
            "prompt": example.prompt,
            "task": video.task,
            # Rows carry no condition media; t2va takes none.
            "conditions": [],
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
        width, height = self.canvas
        return TaskRequest(
            self.point.endpoint,
            payload,
            stream=False,
            video_backend=video.backend,
            poll_interval_s=video.poll_interval_s,
            video_extra_params=dict(video.extra_params),
            video_shape=VideoShape(width=width, height=height, frames=frames),
        )

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
        """Validate canvas, codecs, audio, and duration alignment."""
        expected_frames = [
            target_frames(
                record.requested_seconds
                if record.requested_seconds is not None
                else self.point.video.seconds
            )
            for record in records
        ]
        outputs = [record.decoded_video for record in records]
        present = bool(outputs) and all(
            output is not None for output in outputs
        )
        videos = [output for output in outputs if output is not None]
        width, height = self.canvas

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
                    (video.width, video.height) == (width, height)
                    for video in videos
                ),
                # The container frame rate is a rational; require exactly 24.
                "24fps_video": present
                and all(
                    video.fps_numerator == VIDEO_FPS * video.fps_denominator
                    for video in videos
                ),
                "aligned_frame_count": present
                and all(
                    abs(video.frame_count - frames) <= frame_tolerance
                    for video, frames in zip(
                        videos, expected_frames, strict=True
                    )
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
                    abs(video.audio_duration_s - frames / VIDEO_FPS)
                    <= duration_tolerance_s
                    for video, frames in zip(
                        videos, expected_frames, strict=True
                    )
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


__all__ = ["VideoTask", "target_canvas", "target_frames"]
