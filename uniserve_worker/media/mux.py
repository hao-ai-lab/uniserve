"""Thread-safe in-memory audio/video encoding and MP4 mux sessions."""

from __future__ import annotations

import concurrent.futures
import io
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from threading import RLock
from typing import TYPE_CHECKING, cast

import numpy as np

from uniserve.media.video import Config

from ..foundation.errors import invalid_descriptor
from ..protocol.batch import MediaTrack
from ..protocol.identity import ComputationId, RequestKey
from ..protocol.output import MediaOutput, PosixShmArtifact
from .storage import publish_media_bytes

if TYPE_CHECKING:
    import torch
    from av.audio.stream import AudioStream
    from av.container.output import OutputContainer
    from av.packet import Packet
    from av.video.stream import VideoStream

    from ..execution.output import OutputBuffer
    from ..runtime.host_lane import HostTask
    from .buffers import MediaLease

__all__ = ["AvMuxConfig", "AvMuxSession", "require_media_codecs"]


def require_media_codecs(video_codec: str, audio_codec: str) -> None:
    """Verify that the configured encoders are available through PyAV.

    Both the video and the audio encoder are checked.
    """
    try:
        import av
    except ImportError as error:
        raise RuntimeError("media output requires the PyAV runtime") from error
    missing = [
        name
        for name in (video_codec, audio_codec)
        if name not in av.codecs_available
    ]
    if missing:
        raise RuntimeError(
            f"media output is missing required encoders {missing!r}"
        )
    for name in (video_codec, audio_codec):
        av.CodecContext.create(name, "w")


@dataclass(frozen=True, slots=True)
class AvMuxConfig:
    """Codec, frame, audio, and decode-unit settings for one media container."""

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
        """Validate positive media dimensions.

        Also validate complete decode-unit coverage.
        """
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
            raise ValueError("media mux dimensions and rates must be positive")
        if (
            not self.video_unit_frames
            or sum(self.video_unit_frames) != self.frame_count
        ):
            raise ValueError(
                "media mux decode units must cover the output frame count"
            )


class AvMuxSession:
    """Request-owned media container.

    Audio and video producers are independent.
    """

    def __init__(self, config: AvMuxConfig) -> None:
        """Create an in-memory container session.

        Audio and video use independent locks.
        """
        self.config = config
        self._buffer = io.BytesIO()
        self._container: OutputContainer | None = None
        self._video: VideoStream | None = None
        self._audio: AudioStream | None = None
        self._next_unit = 0
        self._video_frames = 0
        self._audio_written = False
        self._audio_packets: list[Packet] = []
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

            # Stream creation is atomic under the container lock so either
            # producer may be the first to initialize their shared mux
            # destination.
            config = self.config
            container = av.open(self._buffer, mode="w", format="mp4")
            video = cast(
                "VideoStream",
                container.add_stream(
                    config.video_codec, rate=config.frame_rate
                ),
            )
            video.width = config.width
            video.height = config.height
            video.pix_fmt = "yuv420p"
            video.options = {"preset": "ultrafast", "tune": "zerolatency"}
            audio = cast(
                "AudioStream",
                container.add_stream(
                    config.audio_codec, rate=config.audio_rate
                ),
            )
            audio.layout = "stereo"
            audio.sample_rate = config.audio_rate
            audio.bit_rate = 144_000
            audio.options = {"aac_coder": "fast"}
            self._container = container
            self._video = video
            self._audio = audio

    def write_video(
        self, start_unit: int, unit_count: int, rgb24: np.ndarray
    ) -> None:
        """Validate and encode the next ordered unit of packed RGB frames."""
        import av

        config = self.config
        with self._video_lock:
            # Decode-unit ordering makes frame timestamps independent of
            # producer timing.
            stop_unit = int(start_unit) + int(unit_count)
            if (
                self._closed
                or int(start_unit) != self._next_unit
                or int(unit_count) < 1
                or stop_unit > len(config.video_unit_frames)
            ):
                raise RuntimeError("video mux units are not request-ordered")
            expected_frames = sum(
                config.video_unit_frames[int(start_unit) : stop_unit]
            )
            if (
                rgb24.ndim != 4
                or rgb24.shape[1:] != (config.height, config.width, 3)
                or int(rgb24.shape[0]) != expected_frames
            ):
                raise RuntimeError("video capture has invalid RGB24 dimensions")

            # Initialize shared streams only after the input has passed
            # validation.
            self._open()
            container, stream = self._container, self._video
            if container is None or stream is None:
                raise RuntimeError("video stream was not initialized")

            # Packet muxing is serialized with audio while frame
            # preparation remains local.
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
        """Validate and encode the request's stereo PCM track.

        The complete track is encoded exactly once.
        """
        import av

        config = self.config
        with self._audio_lock:
            # Audio is a single request-owned contribution rather than an
            # ordered unit stream.
            if self._closed or self._audio_written:
                raise RuntimeError("audio was muxed more than once")
            if pcm.ndim != 2 or pcm.shape[1] != 2:
                raise RuntimeError(
                    "audio capture has invalid stereo dimensions"
                )

            self._open()
            container, stream = self._container, self._audio
            if container is None or stream is None:
                raise RuntimeError("audio stream was not initialized")

            # Align audio duration to the video timeline, truncating or
            # zero-padding PCM.
            target_samples = round(
                config.frame_count * config.audio_rate / config.frame_rate
            )
            source = pcm[:target_samples]
            if source.shape[0] < target_samples:
                source = np.pad(
                    source, ((0, target_samples - source.shape[0]), (0, 0))
                )

            # Encode fixed-size planar frames; the final frame carries
            # zero padding only.
            packets: list[Packet] = []
            pts = 0
            for start in range(0, target_samples, config.audio_frame_samples):
                stop = min(start + config.audio_frame_samples, target_samples)
                planar = np.zeros(
                    (2, config.audio_frame_samples), dtype=np.int16
                )
                planar[:, : stop - start] = source[start:stop].T
                frame = av.AudioFrame.from_ndarray(
                    planar, format="s16p", layout="stereo"
                )
                frame.sample_rate = config.audio_rate
                frame.pts = pts
                frame.time_base = Fraction(1, config.audio_rate)
                packets.extend(stream.encode(frame))
                pts += config.audio_frame_samples

            # Defer audio packet muxing until close so video units can
            # arrive independently.
            self._audio_packets = packets
            self._audio_written = True

    def close(self) -> bytes:
        """Flush encoders, finalize the container, and return its bytes.

        Finalization happens exactly once.
        """
        config = self.config
        with self._video_lock, self._audio_lock:
            # Repeated close calls expose the same finalized immutable
            # container bytes.
            if self._closed:
                value = self._buffer.getvalue()
                if not value:
                    raise RuntimeError("media mux produced an empty container")
                return value
            if (
                self._video_frames != config.frame_count
                or not self._audio_written
            ):
                raise RuntimeError("media materialization is incomplete")

            container, video, audio = self._container, self._video, self._audio
            if container is None or video is None or audio is None:
                raise RuntimeError("media mux session was never initialized")

            # Drain delayed video packets before committing queued and
            # delayed audio packets.
            for packet in video.encode(None):
                with self._container_lock:
                    container.mux(packet)
            with self._container_lock:
                for packet in self._audio_packets:
                    container.mux(packet)
                for packet in audio.encode(None):
                    container.mux(packet)
                container.close()

            # Closing the PyAV container commits the complete MP4
            # structure to the buffer.
            self._closed = True
            value = self._buffer.getvalue()
            if not value:
                raise RuntimeError("media mux produced an empty container")
            return value

    def abort(self) -> None:
        """Close the container and discard its buffered output.

        Used after a failed request.
        """
        with self._video_lock, self._audio_lock:
            completed = self._closed
            if not self._closed and self._container is not None:
                try:
                    with self._container_lock:
                        self._container.close()
                except Exception:
                    # Teardown is best-effort because aborted output is
                    # never published.
                    pass
            self._closed = True
        if not completed:
            self._buffer.close()


@dataclass(slots=True)
class MuxSession:
    """Container and accepted track submissions for one request epoch.

    Submission cursors are advanced by the execution thread. Encoder futures
    preserve independent video/audio ordering until finalization consumes both.
    """

    container: AvMuxSession
    video_tail: concurrent.futures.Future[object] | None = None
    audio_tail: concurrent.futures.Future[object] | None = None
    video_units: int = 0
    audio_written: bool = False
    finalized: bool = False


class MediaMux:
    """Request-indexed mux sessions with independent video and audio tails."""

    def __init__(self, *, rank: int) -> None:
        """Initialize per-request mux sessions and temporal overlap tails."""
        self.rank = rank
        self._sessions: dict[RequestKey, MuxSession] = {}

    def open(
        self,
        request_key: RequestKey,
        *,
        video: Config,
        frame_rate: int,
        audio_rate: int,
        video_unit_frames: tuple[int, ...],
    ) -> None:
        """Create the request-owned mux session.

        Output dimensions are validated beforehand.
        """
        if request_key in self._sessions:
            return
        self._sessions[request_key] = MuxSession(
            AvMuxSession(
                AvMuxConfig(
                    width=int(video.frame.width),
                    height=int(video.frame.height),
                    frame_count=int(video.num_frames),
                    frame_rate=int(frame_rate),
                    audio_rate=int(audio_rate),
                    video_unit_frames=video_unit_frames,
                )
            )
        )

    def validate_track(
        self,
        request_key: RequestKey,
        track: MediaTrack,
        cursor: int,
        count: int,
    ) -> None:
        """Reject duplicate tracks and temporal gaps.

        Validation runs before numerical assembly.
        """
        session = self._sessions.get(request_key)
        if session is None:
            raise invalid_descriptor("media output has no active session")
        if session.finalized:
            raise invalid_descriptor("media output is already finalized")
        if track is MediaTrack.VIDEO and (
            cursor != session.video_units
            or count < 1
            or cursor + count > len(session.container.config.video_unit_frames)
        ):
            raise invalid_descriptor(
                "video assembly requires the next temporal range"
            )
        if track is MediaTrack.AUDIO and session.audio_written:
            raise invalid_descriptor("audio output is already written")

    def _task(
        self,
        request_key: RequestKey,
        reservation: HostTask,
        action: Callable[[AvMuxSession], object],
        output: OutputBuffer | None,
        dependencies: tuple[concurrent.futures.Future[object], ...],
        ring_lease: MediaLease | None = None,
        *,
        profile_name: str,
    ) -> HostTask:
        """Submit one ordered mux action.

        Its reservation and ring lease are released on completion.
        """
        session = self._sessions.get(request_key)
        if session is None:
            raise RuntimeError("video mux session is not active")
        return reservation.configure(
            lambda: action(session.container),
            dependencies=dependencies,
            input_ready=None if output is None else output.ready,
            input_completion=None
            if output is None
            else output.completion_future,
            release=None if ring_lease is None else ring_lease.release,
            profile_name=profile_name,
        )

    def video(
        self,
        request_key: RequestKey,
        start_unit: int,
        unit_count: int,
        frames: torch.Tensor,
        output: OutputBuffer,
        reservation: HostTask,
        ring_lease: MediaLease,
        operation_id: ComputationId,
    ) -> HostTask:
        """Schedule ordered RGB frame encoding.

        Frames come from a captured output-ring slot.
        """
        self.validate_track(
            request_key, MediaTrack.VIDEO, start_unit, unit_count
        )
        dependency = self._sessions[request_key].video_tail

        task = self._task(
            request_key,
            reservation,
            lambda session: session.write_video(
                start_unit, unit_count, frames.numpy()
            ),
            output,
            () if dependency is None else (dependency,),
            ring_lease,
            profile_name=(
                f"uniserve.video.mux request={_key_label(request_key)} "
                f"step={operation_id.batch_id} op={operation_id.request_index} "
                f"kind=video start_unit={start_unit} "
                f"unit_count={unit_count} rank={self.rank}"
            ),
        )
        self._sessions[request_key].video_tail = task.promise
        self._sessions[request_key].video_units += unit_count
        return task

    def audio(
        self,
        request_key: RequestKey,
        pcm: torch.Tensor,
        output: OutputBuffer,
        reservation: HostTask,
        ring_lease: MediaLease,
        operation_id: ComputationId,
    ) -> HostTask:
        """Schedule PCM encoding from a captured output-ring slot."""
        self.validate_track(request_key, MediaTrack.AUDIO, 0, 1)
        dependency = self._sessions[request_key].audio_tail

        # The ring slot holds raw PCM bytes; reinterpret them as [samples, 2]
        # int16 stereo samples for the encoder.
        task = self._task(
            request_key,
            reservation,
            lambda session: session.write_audio(
                pcm.numpy().reshape(-1).view(np.int16).reshape(-1, 2)
            ),
            output,
            () if dependency is None else (dependency,),
            ring_lease,
            profile_name=(
                f"uniserve.video.mux request={_key_label(request_key)} "
                f"step={operation_id.batch_id} "
                f"op={operation_id.request_index} "
                f"kind=audio rank={self.rank}"
            ),
        )
        self._sessions[request_key].audio_tail = task.promise
        self._sessions[request_key].audio_written = True
        return task

    def finalize_artifact(
        self,
        request_key: RequestKey,
        reservation: HostTask,
        operation_id: ComputationId,
    ) -> HostTask:
        """Schedule mux finalization and shared-memory publication.

        Finalization runs after all segment jobs.
        """
        state = self._sessions.get(request_key)
        if (
            state is None
            or state.video_units
            != len(state.container.config.video_unit_frames)
            or not state.audio_written
            or state.finalized
        ):
            raise invalid_descriptor(
                "media finalization requires both completed output tracks"
            )

        dependencies = tuple(
            tail
            for tail in (
                self._sessions[request_key].video_tail,
                self._sessions[request_key].audio_tail,
            )
            if tail is not None
        )

        def publish(session: AvMuxSession) -> MediaOutput:
            """Close the container and publish its final bytes.

            Publication happens after both tracks complete.
            """
            payload = session.close()
            name = publish_media_bytes(payload)
            return MediaOutput(
                handle=PosixShmArtifact(name=name),
                bytes=len(payload),
            )

        task = self._task(
            request_key,
            reservation,
            publish,
            None,
            dependencies,
            profile_name=(
                f"uniserve.video.mux request={_key_label(request_key)} "
                f"step={operation_id.batch_id} "
                f"op={operation_id.request_index} "
                f"kind=artifact rank={self.rank}"
            ),
        )
        state.finalized = True
        return task

    def drop(self, request_id: int) -> None:
        """Abort and remove every mux session owned by a request identifier."""
        selected = [
            key for key in self._sessions if key.request_id == int(request_id)
        ]
        for key in selected:
            session = self._sessions.pop(key)
            session.container.abort()

    def close(self) -> None:
        """Abort all active mux sessions and reject new media work."""
        for session in self._sessions.values():
            session.container.abort()
        self._sessions.clear()


def _key_label(request_key: RequestKey) -> str:
    """Format a stable request key for media task profiling."""
    return (
        f"{request_key.engine_id}:{request_key.request_id}:"
        f"{request_key.request_epoch}"
    )
