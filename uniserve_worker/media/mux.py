"""Thread-safe in-memory audio/video encoding and MP4 mux sessions."""

from __future__ import annotations

import io
from dataclasses import dataclass
from fractions import Fraction
from threading import RLock

import numpy as np

__all__ = ["AvMuxConfig", "AvMuxSession", "require_media_codecs"]


def require_media_codecs(video_codec: str, audio_codec: str) -> None:
    """Verify that the configured video and audio encoders are available through PyAV."""

    try:
        import av
    except ImportError as error:
        raise RuntimeError("media output requires the PyAV runtime") from error
    missing = [
        name for name in (video_codec, audio_codec) if name not in av.codecs_available
    ]
    if missing:
        raise RuntimeError(f"media output is missing required encoders {missing!r}")
    for name in (video_codec, audio_codec):
        av.CodecContext.create(name, "w")


@dataclass(frozen=True, slots=True)
class AvMuxConfig:
    """Codec, frame, audio, and decode-unit geometry for one media container."""

    width: int
    height: int
    frame_count: int
    frame_rate: int
    audio_rate: int
    video_unit_frames: tuple[int, ...]
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_frame_samples: int = 1024

    def __post_init__(self) -> None:
        """Validate positive media geometry and complete decode-unit coverage."""

        if (
            min(
                self.width,
                self.height,
                self.frame_count,
                self.frame_rate,
                self.audio_rate,
            )
            < 1
        ):
            raise ValueError("media mux geometry and rates must be positive")
        if (
            not self.video_unit_frames
            or sum(self.video_unit_frames) != self.frame_count
        ):
            raise ValueError("media mux decode units must cover the output frame count")


class AvMuxSession:
    """Request-owned media container with independent audio and video producers."""

    def __init__(self, config: AvMuxConfig) -> None:
        """Create an in-memory container session with independent audio and video locks."""

        self.config = config
        self._buffer = io.BytesIO()
        self._container = None
        self._video = None
        self._audio = None
        self._next_unit = 0
        self._video_frames = 0
        self._audio_written = False
        self._audio_packets: list[object] = []
        self._closed = False
        self._video_lock = RLock()
        self._audio_lock = RLock()
        self._container_lock = RLock()

    def _open(self) -> None:
        """Create the container and both encoder streams exactly once."""

        with self._container_lock:
            if self._container is not None:
                return
            import av

            # Stream creation is atomic under the container lock so either producer may
            # be the first to initialize their shared mux destination.
            config = self.config
            container = av.open(self._buffer, mode="w", format="mp4")
            video = container.add_stream(config.video_codec, rate=config.frame_rate)
            video.width = config.width
            video.height = config.height
            video.pix_fmt = "yuv420p"
            video.options = {"preset": "ultrafast", "tune": "zerolatency"}
            audio = container.add_stream(config.audio_codec, rate=config.audio_rate)
            audio.layout = "stereo"
            audio.sample_rate = config.audio_rate
            audio.bit_rate = 144_000
            audio.options = {"aac_coder": "fast"}
            self._container = container
            self._video = video
            self._audio = audio

    def write_video(self, start_unit: int, unit_count: int, rgb24: np.ndarray) -> None:
        """Validate and encode the next ordered unit of packed RGB frames."""

        import av

        config = self.config
        with self._video_lock:
            # Decode-unit ordering makes frame timestamps independent of producer timing.
            stop_unit = int(start_unit) + int(unit_count)
            if (
                self._closed
                or int(start_unit) != self._next_unit
                or int(unit_count) < 1
                or stop_unit > len(config.video_unit_frames)
            ):
                raise RuntimeError("video mux units are not request-ordered")
            expected_frames = sum(config.video_unit_frames[int(start_unit) : stop_unit])
            if (
                rgb24.ndim != 4
                or rgb24.shape[1:] != (config.height, config.width, 3)
                or int(rgb24.shape[0]) != expected_frames
            ):
                raise RuntimeError("video capture has invalid RGB24 geometry")

            # Initialize shared streams only after the input has passed validation.
            self._open()
            container, stream = self._container, self._video
            if container is None or stream is None:
                raise RuntimeError("video stream was not initialized")

            # Packet muxing is serialized with audio while frame preparation remains local.
            for pixels in rgb24:
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = self._video_frames
                frame.time_base = Fraction(1, config.frame_rate)
                for packet in stream.encode(frame):
                    with self._container_lock:
                        container.mux(packet)
                self._video_frames += 1
            self._next_unit = stop_unit

    def write_audio(self, pcm: np.ndarray) -> None:
        """Validate and encode the request's complete stereo PCM track exactly once."""

        import av

        config = self.config
        with self._audio_lock:
            # Audio is a single request-owned contribution rather than an ordered unit stream.
            if self._closed or self._audio_written:
                raise RuntimeError("audio was muxed more than once")
            if pcm.ndim != 2 or pcm.shape[1] != 2:
                raise RuntimeError("audio capture has invalid stereo geometry")
            self._open()
            container, stream = self._container, self._audio
            if container is None or stream is None:
                raise RuntimeError("audio stream was not initialized")

            # Align audio duration to the video timeline, truncating or zero-padding PCM.
            target_samples = round(
                config.frame_count * config.audio_rate / config.frame_rate
            )
            source = pcm[:target_samples]
            if source.shape[0] < target_samples:
                source = np.pad(source, ((0, target_samples - source.shape[0]), (0, 0)))

            # Encode fixed-size planar frames; the final frame carries zero padding only.
            packets: list[object] = []
            pts = 0
            for start in range(0, target_samples, config.audio_frame_samples):
                stop = min(start + config.audio_frame_samples, target_samples)
                planar = np.zeros((2, config.audio_frame_samples), dtype=np.int16)
                planar[:, : stop - start] = source[start:stop].T
                frame = av.AudioFrame.from_ndarray(
                    planar, format="s16p", layout="stereo"
                )
                frame.sample_rate = config.audio_rate
                frame.pts = pts
                frame.time_base = Fraction(1, config.audio_rate)
                packets.extend(stream.encode(frame))
                pts += config.audio_frame_samples

            # Defer audio packet muxing until close so video units can arrive independently.
            self._audio_packets = packets
            self._audio_written = True

    def close(self) -> bytes:
        """Flush encoders, finalize the container, and return its bytes exactly once."""

        config = self.config
        with self._video_lock, self._audio_lock:
            # Repeated close calls expose the same finalized immutable container bytes.
            if self._closed:
                value = self._buffer.getvalue()
                if not value:
                    raise RuntimeError("media mux produced an empty container")
                return value
            if self._video_frames != config.frame_count or not self._audio_written:
                raise RuntimeError("media materialization is incomplete")
            container, video, audio = self._container, self._video, self._audio
            if container is None or video is None or audio is None:
                raise RuntimeError("media mux session was never initialized")

            # Drain delayed video packets before committing queued and delayed audio packets.
            for packet in video.encode(None):
                with self._container_lock:
                    container.mux(packet)
            with self._container_lock:
                for packet in self._audio_packets:
                    container.mux(packet)
                for packet in audio.encode(None):
                    container.mux(packet)
                container.close()

            # Closing the PyAV container commits the complete MP4 structure to the buffer.
            self._closed = True
            value = self._buffer.getvalue()
            if not value:
                raise RuntimeError("media mux produced an empty container")
            return value

    def abort(self) -> None:
        """Close the container and discard its buffered output after a failed request."""

        with self._video_lock, self._audio_lock:
            completed = self._closed
            if not self._closed and self._container is not None:
                try:
                    with self._container_lock:
                        self._container.close()
                except Exception:
                    # Teardown is best-effort because aborted output is never published.
                    pass
            self._closed = True
        if not completed:
            self._buffer.close()
