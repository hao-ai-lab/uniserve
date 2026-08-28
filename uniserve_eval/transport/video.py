"""Inspection of synchronous raw-MP4 responses."""

from __future__ import annotations

import hashlib
import io
from fractions import Fraction

from ..types import DecodedVideo


class VideoOutputError(ValueError):
    def __init__(self, classifier: str, message: str) -> None:
        self.classifier = classifier
        super().__init__(message)


def inspect_video_bytes(data: bytes, *, declared_mime: str) -> DecodedVideo:
    if not data:
        raise VideoOutputError("response_empty_video", "video response body is empty")
    mime = declared_mime.partition(";")[0].strip().lower()
    if mime != "video/mp4":
        raise VideoOutputError(
            "response_invalid_video_mime", f"expected video/mp4, received {declared_mime!r}"
        )

    try:
        import av

        with av.open(io.BytesIO(data), mode="r", format="mp4") as container:
            video_streams = list(container.streams.video)
            audio_streams = list(container.streams.audio)
            if len(video_streams) != 1 or len(audio_streams) != 1:
                raise VideoOutputError(
                    "response_invalid_media_streams",
                    "fixed H3 output must contain one video and one audio stream",
                )
            video_stream = video_streams[0]
            audio_stream = audio_streams[0]
            frame_count = 0
            audio_samples = 0
            audio_channels: int | None = None
            audio_sample_rate: int | None = None
            for frame in container.decode(video=0, audio=0):
                if isinstance(frame, av.VideoFrame):
                    frame_count += 1
                    continue
                if isinstance(frame, av.AudioFrame):
                    audio_samples += int(frame.samples)
                    channels = len(frame.layout.channels)
                    rate = int(frame.sample_rate)
                    if audio_channels is None:
                        audio_channels = channels
                    elif channels != audio_channels:
                        raise VideoOutputError(
                            "response_inconsistent_audio_layout",
                            "audio channel layout changes within the response",
                        )
                    if audio_sample_rate is None:
                        audio_sample_rate = rate
                    elif rate != audio_sample_rate:
                        raise VideoOutputError(
                            "response_inconsistent_audio_rate",
                            "audio sample rate changes within the response",
                        )

            rate_value = video_stream.average_rate
            if rate_value is None:
                raise VideoOutputError(
                    "response_missing_video_rate", "video stream has no average frame rate"
                )
            frame_rate = Fraction(rate_value)
            video_codec = str(video_stream.codec_context.name or "")
            audio_codec = str(audio_stream.codec_context.name or "")
            if frame_count < 1 or audio_samples < 1:
                raise VideoOutputError(
                    "response_undecodable_video", "video or audio stream decoded no frames"
                )
            if audio_channels is None or audio_sample_rate is None:
                raise VideoOutputError(
                    "response_undecodable_audio", "audio stream decoded no valid PCM frames"
                )
            checksum = hashlib.sha256(data).hexdigest()
            return DecodedVideo(
                data=data,
                sha256=checksum,
                byte_size=len(data),
                mime=mime,
                width=int(video_stream.codec_context.width),
                height=int(video_stream.codec_context.height),
                frame_count=frame_count,
                fps_numerator=frame_rate.numerator,
                fps_denominator=frame_rate.denominator,
                video_codec=video_codec,
                audio_codec=audio_codec,
                audio_channels=audio_channels,
                audio_sample_rate=audio_sample_rate,
                audio_samples=audio_samples,
                sample_filename=f"{checksum}.mp4",
            )
    except VideoOutputError:
        raise
    except Exception as error:
        raise VideoOutputError(
            "response_undecodable_video", f"MP4 decode failed: {type(error).__name__}: {error}"
        ) from error


__all__ = ["VideoOutputError", "inspect_video_bytes"]
