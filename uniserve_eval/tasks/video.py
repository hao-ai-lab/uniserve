"""Build native video requests and inspect complete MP4s after measurement.

Synchronous and asynchronous adapters share the MiniMax H3 media contract.
Inspection uses each request's duration and never runs inside a load slot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from ..transport.video import VideoOutputError, inspect_video_bytes
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
    """Builds native video requests and validates the media contract."""

    name: ClassVar[TaskName] = TaskName.VIDEO
    allowed_endpoints: ClassVar[tuple[str, ...]] = (VIDEOS_SYNC, "/v1/videos")
    default_endpoint: ClassVar[str] = VIDEOS_SYNC
    default_stream: ClassVar[bool] = False

    def build_request(self, example: Example) -> TaskRequest:
        """Build a deterministic duration- and seed-qualified video request."""
        seconds = float(
            example.seconds
            if example.seconds is not None
            else self.point.video.seconds
        )
        if not math.isfinite(seconds) or not 4 <= seconds <= 15:
            raise ValueError("H3 seconds must be finite and in [4, 15]")
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
                "seconds": seconds,
            },
            stream=False,
            video_backend=self.point.video.backend,
            poll_interval_s=self.point.video.poll_interval_s,
            video_extra_params=dict(self.point.video.extra_params),
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
        """Validate fixed video geometry, codecs, audio, and duration alignment."""  # noqa: E501
        # Python round uses the API's ties-to-even rule before upward alignment.
        expected_frames = []
        for record in records:
            seconds = record.requested_seconds
            raw_frames = round(
                (seconds if seconds is not None else self.point.video.seconds)
                * 24
            )
            expected_frames.append(raw_frames + (5 - raw_frames) % 17)
        outputs = [record.decoded_video for record in records]
        present = bool(outputs) and all(
            output is not None for output in outputs
        )
        videos = [output for output in outputs if output is not None]

        # Both streams are measured against the aligned media length rather
        # than each other: the video may differ from the aligned frame count
        # by one frame, and the audio duration from the aligned duration by
        # one frame period. A one-frame-short video with full-length audio is
        # therefore valid, while a missing or truncated stream is not.
        frame_tolerance = 1
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
                    # The container frame rate is a rational; require exactly
                    # 24 fps.
                    and video.fps_numerator == 24 * video.fps_denominator
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
                    abs(video.audio_duration_s - frames / 24.0)
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


__all__ = ["VideoTask"]
