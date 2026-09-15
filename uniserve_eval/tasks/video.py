"""Defines synchronous text-to-video-with-audio benchmark behavior."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from ..types import (
    VIDEOS_SYNC,
    Example,
    RequestRecord,
    TaskName,
    TaskRequest,
    ValidationResult,
)
from .base import BenchmarkTask


class VideoTask(BenchmarkTask):
    """Builds synchronous video requests and validates the media contract."""

    name: ClassVar[TaskName] = TaskName.VIDEO
    allowed_endpoints: ClassVar[tuple[str, ...]] = (VIDEOS_SYNC,)
    default_endpoint: ClassVar[str] = VIDEOS_SYNC
    default_stream: ClassVar[bool] = False

    def build_request(self, example: Example) -> TaskRequest:
        """Build a deterministic duration- and seed-qualified video request."""
        return TaskRequest(
            self.point.endpoint,
            {
                "model": self.point.model,
                "prompt": example.prompt,
                "seed": int(
                    example.seed
                    if example.seed is not None
                    else self.point.load.seed
                ),
                "seconds": float(
                    example.seconds
                    if example.seconds is not None
                    else self.point.video.seconds
                ),
            },
            stream=False,
        )

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Validate fixed video geometry, codecs, audio, and duration alignment."""  # noqa: E501
        raw_frames = math.floor(float(self.point.video.seconds) * 24.0 + 0.5)
        expected_frames = int(raw_frames + (5 - raw_frames) % 17)
        outputs = [record.decoded_video for record in records]
        present = bool(outputs) and all(
            output is not None for output in outputs
        )
        videos = [output for output in outputs if output is not None]
        duration_tolerance_s = 1.0 / 24.0
        return ValidationResult(
            checks={
                "decoded_video": present,
                "h264_video": present
                and all(video.video_codec == "h264" for video in videos),
                "fixed_video_geometry": present
                and all(
                    video.width == 1344
                    and video.height == 768
                    and video.frame_count == expected_frames
                    and video.fps_numerator == 24 * video.fps_denominator
                    for video in videos
                ),
                "aac_stereo_32khz_audio": present
                and all(
                    video.audio_codec == "aac"
                    and video.audio_channels == 2
                    and video.audio_sample_rate == 32_000
                    for video in videos
                ),
                "audio_spans_video": present
                and all(
                    abs(video.audio_duration_s - video.video_duration_s)
                    <= duration_tolerance_s
                    for video in videos
                ),
            },
            statistics={
                "completed_videos": len(videos),
                "media_bytes": sum(video.byte_size for video in videos),
            },
        )


__all__ = ["VideoTask"]
