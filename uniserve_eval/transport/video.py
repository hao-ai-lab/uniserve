"""Decodes and validates synchronous raw-MP4 responses.

``/v1/videos/sync`` returns the encoded MP4 itself as the response body.
``inspect_video_bytes`` checks the declared media type and stream layout,
fully decodes both streams, and returns a ``DecodedVideo`` whose metadata is
derived from the bytes and the normalized media type. Model-specific
expectations such as codecs, raster, frame count, and audio format are
checked by ``VideoTask.validate_output``, not here.

Inspection runs after transport completion for the entire measured batch.
Original response bytes are persisted even when inspection rejects them.
"""

from __future__ import annotations

import hashlib
import io
from fractions import Fraction

import numpy as np

from ..types import DecodedVideo


class VideoOutputError(ValueError):
    """Carries a stable classifier and detail for invalid video output.

    The video transport in ``uniserve_eval.transport.client`` records
    ``classifier`` on the ``RequestRecord``, and ``build_summary`` in
    ``uniserve_eval.pipeline.report`` counts records per classifier.
    """

    def __init__(self, classifier: str, message: str) -> None:
        """Initialize the error with its classifier and diagnostic detail."""
        self.classifier = classifier
        super().__init__(message)


def inspect_video_bytes(data: bytes, *, declared_mime: str) -> DecodedVideo:
    """Decode one video and audio stream and return verified media metadata.

    Args:
        data: The complete response body.
        declared_mime: The response ``Content-Type``; parameters after ``;``
            are ignored.

    Returns:
        The body with its metadata. Frame and sample counts come from full
        decoding; ``sample_filename`` is named by the SHA-256 of ``data``.

    Raises:
        VideoOutputError: The body is empty, the media type is not
            ``video/mp4``, the container does not hold exactly one video and
            one audio stream, the audio channel count or sample rate changes
            between frames, the video stream has no average frame rate,
            either stream decodes nothing, or any other exception is raised
            inside the decode block, including a failed ``av`` import, which
            is wrapped with the ``response_undecodable_video`` classifier.
    """
    # Validate the HTTP-level envelope before invoking the media decoder.
    if not data:
        raise VideoOutputError(
            "response_empty_video", "video response body is empty"
        )
    mime = declared_mime.partition(";")[0].strip().lower()
    if mime != "video/mp4":
        raise VideoOutputError(
            "response_invalid_video_mime",
            f"expected video/mp4, received {declared_mime!r}",
        )

    try:
        # A missing av module is caught below and classified
        # response_undecodable_video.
        import av

        with av.open(io.BytesIO(data), mode="r", format="mp4") as container:
            # The metadata below describes one stream per modality, and
            # `container.decode(video=0, audio=0)` reads only the first stream
            # of each, so additional streams are rejected rather than ignored.
            video_streams = list(container.streams.video)
            audio_streams = list(container.streams.audio)
            if len(video_streams) != 1 or len(audio_streams) != 1:
                raise VideoOutputError(
                    "response_invalid_media_streams",
                    "fixed H3 output must contain one video and one audio "
                    "stream",
                )
            video_stream = video_streams[0]
            audio_stream = audio_streams[0]
            frame_count = 0
            audio_samples = 0
            audio_channels: int | None = None
            audio_sample_rate: int | None = None
            pixel_count = 0
            pixel_sum = pixel_squares = 0.0
            pcm_count = 0
            pcm_squares = 0.0

            # Fully decode both streams; container metadata alone cannot prove
            # that frames and PCM samples are readable.
            for frame in container.decode(video=0, audio=0):
                if isinstance(frame, av.VideoFrame):
                    frame_count += 1
                    pixels = frame.to_ndarray(format="rgb24").astype(np.float64)
                    pixel_count += pixels.size
                    pixel_sum += float(pixels.sum())
                    pixel_squares += float(np.square(pixels).sum())
                    if (frame.width, frame.height) != (
                        video_stream.codec_context.width,
                        video_stream.codec_context.height,
                    ):
                        raise VideoOutputError(
                            "response_inconsistent_video_shape",
                            "video raster changes within the response",
                        )
                    continue
                if isinstance(frame, av.AudioFrame):
                    # `samples` counts samples per channel, so the total over
                    # sample rate is the audio duration in seconds.
                    audio_samples += int(frame.samples)
                    pcm = frame.to_ndarray()
                    # Report RMS in normalized PCM units for integer formats.
                    scale = (
                        max(
                            abs(np.iinfo(pcm.dtype).min),
                            np.iinfo(pcm.dtype).max,
                        )
                        if np.issubdtype(pcm.dtype, np.integer)
                        else 1.0
                    )
                    normalized = pcm.astype(np.float64) / scale
                    pcm_count += pcm.size
                    pcm_squares += float(np.square(normalized).sum())
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

            # Derive content metadata only after all frames satisfy the stream
            # invariants, then name the sample by its encoded bytes.
            rate_value = video_stream.average_rate
            if rate_value is None:
                raise VideoOutputError(
                    "response_missing_video_rate",
                    "video stream has no average frame rate",
                )
            frame_rate = Fraction(rate_value)
            video_codec = str(video_stream.codec_context.name or "")
            audio_codec = str(audio_stream.codec_context.name or "")
            if frame_count < 1 or audio_samples < 1:
                raise VideoOutputError(
                    "response_undecodable_video",
                    "video or audio stream decoded no frames",
                )
            # A positive sample count implies an audio frame set both values;
            # this check narrows their types.
            if audio_channels is None or audio_sample_rate is None:
                raise VideoOutputError(
                    "response_undecodable_audio",
                    "audio stream decoded no valid PCM frames",
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
                video_variance=max(
                    0.0,
                    pixel_squares / pixel_count
                    - (pixel_sum / pixel_count) ** 2,
                ),
                audio_rms=float(np.sqrt(pcm_squares / pcm_count)),
            )
    except VideoOutputError:
        raise
    except Exception as error:
        # Normalize decoder-specific failures into the evaluator classifier set.
        raise VideoOutputError(
            "response_undecodable_video",
            f"MP4 decode failed: {type(error).__name__}: {error}",
        ) from error


__all__ = ["VideoOutputError", "inspect_video_bytes"]
