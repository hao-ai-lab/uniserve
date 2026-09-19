"""In-memory media unit encoding and artifact assembly.

Each media unit is encoded where it was reconstructed, as a self-contained MP4
whose first frame is a keyframe, and the audio track is encoded the same way on
the muxer rank. The muxer concatenates them without re-encoding: it creates its
output streams from the first payload of each track and remuxes every packet
with a running timestamp offset.
"""

from __future__ import annotations

import concurrent.futures
import io
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from uniserve.media.video import Config

from ..foundation.errors import invalid_descriptor
from ..protocol.identity import CallId, RequestKey
from ..protocol.output import MediaOutput, PosixShmArtifact
from .storage import publish_media_bytes

if TYPE_CHECKING:
    import torch

    from ..execution.output import OutputBuffer
    from ..runtime.host_lane import HostTask
    from .buffers import MediaLease

__all__ = [
    "AvMuxConfig",
    "AvMuxSession",
    "MediaEncoder",
    "MediaMux",
    "encoded_unit_bytes",
    "require_media_codecs",
]

# A framed media unit carries its own length because the product row that holds
# it is sized for the largest unit a request can produce.
_LENGTH_BYTES = 8
# Container structure and the codec's parameter sets cost about a kilobyte and a
# half regardless of raster, so a row is never smaller than this.
_CONTAINER_FLOOR = 1 << 16
# Each media unit is its own encoder session, and several ranks encode at once
# on one host, so a session must not size its thread pool to the machine.
# Measured at this preset and raster, eight threads encode a unit as fast as an
# unbounded pool does, while an unbounded pool costs an order of magnitude more
# than the encoding itself to create inside a worker process.
_ENCODER_THREADS = 8


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
    """Codec, frame, audio, and media unit settings for one media container."""

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

        Also validate complete media unit coverage.
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
                "media mux media units must cover the output frame count"
            )


def encoded_unit_bytes(frames: int, height: int, width: int) -> int:
    """Bytes a framed encoded media unit occupies, including its length.

    The bound is a quarter of the planar raster the encoder consumes, which is
    about seven times the largest unit measured on real output at the serving
    encoder's settings. A small unit is dominated by the container and its
    parameter sets rather than by its raster, so the bound does not fall below
    the floor those cost. A unit that exceeds the bound fails by name.
    """
    raster = int(frames) * int(height) * int(width)
    return _LENGTH_BYTES + max(raster * 3 // 8, _CONTAINER_FLOOR)


def frame_encoded_unit(payload: bytes, destination: torch.Tensor) -> None:
    """Write one encoded media unit and its length into a product row."""
    import torch

    capacity = int(destination.numel()) - _LENGTH_BYTES
    if len(payload) > capacity:
        raise invalid_descriptor(
            f"encoded media unit of {len(payload)} bytes exceeds the "
            f"{capacity} bytes reserved for it"
        )
    header = np.frombuffer(
        np.uint64(len(payload)).tobytes(), dtype=np.uint8
    ).copy()
    destination[:_LENGTH_BYTES].copy_(torch.from_numpy(header))
    body = np.frombuffer(payload, dtype=np.uint8).copy()
    destination[_LENGTH_BYTES : _LENGTH_BYTES + len(payload)].copy_(
        torch.from_numpy(body)
    )


def read_encoded_unit(row: torch.Tensor) -> bytes:
    """Return the encoded media unit a product row carries."""
    values = row.numpy()
    length = int(
        np.frombuffer(values[:_LENGTH_BYTES].tobytes(), dtype=np.uint64)[0]
    )
    if length > len(values) - _LENGTH_BYTES:
        raise invalid_descriptor("encoded media unit names an invalid length")
    return values[_LENGTH_BYTES : _LENGTH_BYTES + length].tobytes()


def encode_video_unit(config: AvMuxConfig, rgb24: np.ndarray) -> bytes:
    """Encode one media unit as a self-contained MP4 starting at a keyframe."""
    import av

    if (
        rgb24.ndim != 4
        or rgb24.shape[1:] != (config.height, config.width, 3)
        or rgb24.shape[0] < 1
    ):
        raise invalid_descriptor("video capture has invalid RGB24 dimensions")
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    try:
        stream = container.add_stream(
            config.video_codec, rate=config.frame_rate
        )
        stream.width, stream.height = config.width, config.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
        stream.codec_context.thread_count = _ENCODER_THREADS
        for index, pixels in enumerate(rgb24):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, config.frame_rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return buffer.getvalue()


def encode_audio_track(config: AvMuxConfig, pcm: np.ndarray) -> bytes:
    """Encode the request's stereo PCM as a self-contained MP4.

    The track is aligned to the video timeline, truncated or zero-padded, and
    encoded in fixed-size planar frames whose last one carries padding only.
    """
    import av

    if pcm.ndim != 2 or pcm.shape[1] != 2:
        raise invalid_descriptor("audio capture has invalid stereo dimensions")
    target = round(config.frame_count * config.audio_rate / config.frame_rate)
    source = pcm[:target]
    if source.shape[0] < target:
        source = np.pad(source, ((0, target - source.shape[0]), (0, 0)))

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    try:
        stream = container.add_stream(
            config.audio_codec, rate=config.audio_rate
        )
        stream.layout = "stereo"
        stream.sample_rate = config.audio_rate
        stream.bit_rate = 144_000
        stream.options = {"aac_coder": "fast"}
        stream.codec_context.thread_count = _ENCODER_THREADS
        pts = 0
        for start in range(0, target, config.audio_frame_samples):
            stop = min(start + config.audio_frame_samples, target)
            planar = np.zeros((2, config.audio_frame_samples), dtype=np.int16)
            planar[:, : stop - start] = source[start:stop].T
            frame = av.AudioFrame.from_ndarray(
                planar, format="s16p", layout="stereo"
            )
            frame.sample_rate = config.audio_rate
            frame.pts = pts
            frame.time_base = Fraction(1, config.audio_rate)
            for packet in stream.encode(frame):
                container.mux(packet)
            pts += config.audio_frame_samples
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return buffer.getvalue()


class AvMuxSession:
    """Assembles encoded media units and an encoded audio track into one MP4.

    Both tracks arrive already encoded, so assembly creates each output stream
    from a template and copies packets across with a running timestamp
    offset; no frame is decoded or re-encoded. Media units are appended as
    their rounds complete, and the audio track is muxed when the artifact is
    finalized, so the container is open from the first unit to the last.
    """

    def __init__(self, config: AvMuxConfig) -> None:
        self.config = config
        self._buffer: io.BytesIO | None = None
        self._container: Any = None
        self._video_out: Any = None
        self._audio_out: Any = None
        self._offset = 0
        self.units_appended = 0

    @property
    def total_units(self) -> int:
        """Return the number of media units the request's video divides into."""
        return len(self.config.video_unit_frames)

    def _open(self, first_unit: bytes) -> None:
        """Create the container and both output streams before any packet.

        The video stream is templated from the first unit. The audio stream
        must exist before the header is written, and the audio track is not
        encoded until every unit has been reconstructed, so its template is a
        brief silent track encoded under the same settings, whose codec
        parameters are the ones the real track carries.
        """
        import av

        self._buffer = io.BytesIO()
        self._container = av.open(self._buffer, mode="w", format="mp4")
        source = av.open(io.BytesIO(first_unit))
        try:
            self._video_out = self._container.add_stream_from_template(
                source.streams.video[0]
            )
        finally:
            source.close()
        template = encode_audio_track(
            replace(self.config, frame_count=1, video_unit_frames=(1,)),
            np.zeros((1, 2), dtype=np.int16),
        )
        track = av.open(io.BytesIO(template))
        try:
            self._audio_out = self._container.add_stream_from_template(
                track.streams.audio[0]
            )
        finally:
            track.close()

    def append(self, units: tuple[bytes, ...]) -> None:
        """Copy the packets of the next media units, in order."""
        import av

        if self.units_appended + len(units) > self.total_units:
            raise invalid_descriptor(
                "artifact assembly received more media units than the request"
            )
        for payload in units:
            if self._container is None:
                self._open(payload)
            source = av.open(io.BytesIO(payload))
            try:
                stream = source.streams.video[0]
                last = self._offset
                for packet in source.demux(stream):
                    if packet.dts is None:
                        continue
                    packet.stream = self._video_out
                    packet.pts = (packet.pts or 0) + self._offset
                    packet.dts = packet.dts + self._offset
                    self._container.mux(packet)
                    last = max(last, packet.dts + (packet.duration or 1))
                self._offset = last
            finally:
                source.close()
            self.units_appended += 1

    def finalize(self, audio: bytes) -> bytes:
        """Mux the audio track after every unit and return the artifact."""
        import av

        if self.units_appended != self.total_units or self._container is None:
            raise invalid_descriptor(
                "artifact assembly requires every media unit of the request"
            )
        track = av.open(io.BytesIO(audio))
        try:
            for packet in track.demux(track.streams.audio[0]):
                if packet.dts is None:
                    continue
                packet.stream = self._audio_out
                self._container.mux(packet)
        finally:
            track.close()
        self._container.close()
        assert self._buffer is not None
        value = self._buffer.getvalue()
        self.close()
        if not value:
            raise RuntimeError("media mux produced an empty container")
        return value

    def close(self) -> None:
        """Discard an open container, for a request that ends early."""
        if self._container is not None:
            try:
                self._container.close()
            except Exception:  # noqa: BLE001 - closing a discarded container
                pass
        self._container = None
        self._buffer = None
        self._video_out = None
        self._audio_out = None


@dataclass(slots=True)
class MuxSession:
    """One request epoch's assembly state on the muxer rank."""

    container: AvMuxSession
    audio: bytes | None = None
    audio_tail: concurrent.futures.Future[object] | None = None
    #: The last append task, which the next append and the finalization follow.
    mux_tail: concurrent.futures.Future[object] | None = None
    finalized: bool = False


class MediaEncoder:
    """One rank's host encoder for the media units it reconstructs."""

    def __init__(self, *, rank: int) -> None:
        self.rank = rank

    def unit(
        self,
        request_key: RequestKey,
        *,
        config: AvMuxConfig,
        unit_index: int,
        frames: torch.Tensor,
        output: OutputBuffer,
        reservation: HostTask,
        ring_lease: MediaLease,
        call_id: CallId,
    ) -> HostTask:
        """Schedule one media unit's encode from a captured output-ring slot."""
        height, width = config.height, config.width

        def encode() -> bytes:
            # The bytes become this call's product when it completes; the
            # encoded length is not known until here.
            pixels = frames.numpy().reshape(-1, height, width, 3)
            return encode_video_unit(config, pixels)

        return reservation.configure(
            encode,
            dependencies=(),
            input_ready=output.ready,
            input_completion=output.completion_future,
            release=ring_lease.release,
            profile_name=(
                f"uniserve.host.encode request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"op={call_id.request_index} "
                f"kind=video unit={unit_index} rank={self.rank}"
            ),
        )


class MediaMux:
    """Request-indexed artifact assembly on the muxer rank."""

    def __init__(self, *, rank: int) -> None:
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
        """Create the request-owned assembly session."""
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

    def config(self, request_key: RequestKey) -> AvMuxConfig:
        """Return the container settings this request assembles under."""
        session = self._sessions.get(request_key)
        if session is None:
            raise invalid_descriptor("media output has no active session")
        return session.container.config

    def audio(
        self,
        request_key: RequestKey,
        pcm: torch.Tensor,
        output: OutputBuffer,
        reservation: HostTask,
        ring_lease: MediaLease,
        call_id: CallId,
    ) -> HostTask:
        """Schedule the audio track's encode from a captured ring slot."""
        session = self._sessions.get(request_key)
        if session is None:
            raise invalid_descriptor("media output has no active session")
        if session.finalized or session.audio is not None:
            raise invalid_descriptor("audio output is already written")
        config = session.container.config

        def encode() -> None:
            # The ring slot holds raw PCM bytes; reinterpret them as stereo
            # int16 samples for the encoder. The encoded track is the muxer's
            # own state rather than a result, because only the assembled
            # artifact is this request's media output.
            track = pcm.numpy().reshape(-1).view(np.int16).reshape(-1, 2)
            session.audio = encode_audio_track(config, track)

        task = reservation.configure(
            encode,
            dependencies=(),
            input_ready=output.ready,
            input_completion=output.completion_future,
            release=ring_lease.release,
            profile_name=(
                f"uniserve.host.encode request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"op={call_id.request_index} "
                f"kind=audio rank={self.rank}"
            ),
        )
        session.audio_tail = task.promise
        return task

    def append_units(
        self,
        request_key: RequestKey,
        units: tuple[bytes, ...],
        reservation: HostTask,
        call_id: CallId,
    ) -> HostTask:
        """Schedule the next media units into the request's container.

        Units of consecutive rounds are appended in order, so a round's task
        follows the previous round's on the host lane through its dependency.
        """
        session = self._sessions.get(request_key)
        if session is None or session.finalized:
            raise invalid_descriptor(
                "media assembly requires an open assembly session"
            )
        if not units:
            raise invalid_descriptor("media assembly received no media units")
        dependencies = () if session.mux_tail is None else (session.mux_tail,)

        def append() -> None:
            session.container.append(units)

        task = reservation.configure(
            append,
            dependencies=dependencies,
            input_ready=None,
            input_completion=None,
            release=None,
            profile_name=(
                f"uniserve.host.mux request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"op={call_id.request_index} "
                f"kind=units rank={self.rank}"
            ),
        )
        session.mux_tail = task.promise
        return task

    def finalize_artifact(
        self,
        request_key: RequestKey,
        reservation: HostTask,
        call_id: CallId,
    ) -> HostTask:
        """Schedule the artifact's assembly after every unit and the audio.

        The engine schedules this call after the last encode round and the
        audio track have completed, so the task depends on the last append
        and on the audio encode rather than on any input of its own.
        """
        session = self._sessions.get(request_key)
        if session is None or session.finalized:
            raise invalid_descriptor(
                "media finalization requires an open assembly session"
            )
        if session.audio is None and session.audio_tail is None:
            raise invalid_descriptor(
                "artifact finalization precedes the audio track's encode"
            )
        dependencies = tuple(
            promise
            for promise in (session.mux_tail, session.audio_tail)
            if promise is not None
        )

        def publish() -> MediaOutput:
            if session.audio is None:
                raise RuntimeError("artifact assembly has no encoded audio")
            payload = session.container.finalize(session.audio)
            return MediaOutput(
                handle=PosixShmArtifact(name=publish_media_bytes(payload)),
                bytes=len(payload),
            )

        task = reservation.configure(
            publish,
            dependencies=dependencies,
            input_ready=None,
            input_completion=None,
            release=None,
            profile_name=(
                f"uniserve.host.mux request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"op={call_id.request_index} "
                f"kind=artifact rank={self.rank}"
            ),
        )
        session.finalized = True
        return task

    def drop(self, request_id: int) -> None:
        """Remove every assembly session owned by a request identifier."""
        for key in [
            key for key in self._sessions if key.request_id == int(request_id)
        ]:
            self._sessions.pop(key).container.close()

    def close(self) -> None:
        """Discard all active assembly sessions and reject new media work."""
        for session in self._sessions.values():
            session.container.close()
        self._sessions.clear()


def _key_label(request_key: RequestKey) -> str:
    """Format a stable request key for media task profiling."""
    return (
        f"{request_key.engine_id}:{request_key.request_id}:"
        f"{request_key.request_epoch}"
    )
