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
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from uniserve.media.video import Config

from ..foundation.errors import invalid_descriptor
from ..protocol.identity import ComputationId, RequestKey
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
    from its first payload's parameters and copies packets across with a running
    timestamp offset. No frame is decoded or re-encoded.
    """

    def __init__(self, config: AvMuxConfig) -> None:
        self.config = config

    def assemble(self, units: tuple[bytes, ...], audio: bytes) -> bytes:
        """Return the artifact for one request's ordered units and its audio."""
        import av

        if len(units) != len(self.config.video_unit_frames):
            raise invalid_descriptor(
                "artifact assembly requires every media unit of the request"
            )
        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        try:
            video_out = audio_out = None
            offset = 0
            for payload in units:
                source = av.open(io.BytesIO(payload))
                try:
                    stream = source.streams.video[0]
                    if video_out is None:
                        # Both output streams exist before any packet is
                        # muxed, and video leads so the artifact keeps its
                        # stream order.
                        video_out = container.add_stream_from_template(stream)
                        track = av.open(io.BytesIO(audio))
                        try:
                            audio_out = container.add_stream_from_template(
                                track.streams.audio[0]
                            )
                        finally:
                            track.close()
                    last = offset
                    for packet in source.demux(stream):
                        if packet.dts is None:
                            continue
                        packet.stream = video_out
                        packet.pts = (packet.pts or 0) + offset
                        packet.dts = packet.dts + offset
                        container.mux(packet)
                        last = max(last, packet.dts + (packet.duration or 1))
                    offset = last
                finally:
                    source.close()

            track = av.open(io.BytesIO(audio))
            try:
                for packet in track.demux(track.streams.audio[0]):
                    if packet.dts is None:
                        continue
                    packet.stream = audio_out
                    container.mux(packet)
            finally:
                track.close()
        finally:
            container.close()

        value = buffer.getvalue()
        if not value:
            raise RuntimeError("media mux produced an empty container")
        return value


@dataclass(slots=True)
class MuxSession:
    """One request epoch's assembly state on the muxer rank."""

    container: AvMuxSession
    audio: bytes | None = None
    audio_tail: concurrent.futures.Future[object] | None = None
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
        operation_id: ComputationId,
    ) -> HostTask:
        """Schedule one media unit's encode from a captured output-ring slot."""
        height, width = config.height, config.width

        def encode() -> bytes:
            # The bytes become this operation's product when it completes; the
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
                f"step={operation_id.batch_id} "
                f"op={operation_id.request_index} "
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
        operation_id: ComputationId,
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
                f"step={operation_id.batch_id} "
                f"op={operation_id.request_index} "
                f"kind=audio rank={self.rank}"
            ),
        )
        session.audio_tail = task.promise
        return task

    def finalize_artifact(
        self,
        request_key: RequestKey,
        units: tuple[bytes, ...],
        reservation: HostTask,
        operation_id: ComputationId,
    ) -> HostTask:
        """Schedule assembly of the artifact from every encoded track."""
        session = self._sessions.get(request_key)
        if session is None or session.finalized:
            raise invalid_descriptor(
                "media finalization requires an open assembly session"
            )
        dependencies = (
            () if session.audio_tail is None else (session.audio_tail,)
        )

        def publish() -> MediaOutput:
            if session.audio is None:
                raise RuntimeError("artifact assembly has no encoded audio")
            payload = session.container.assemble(units, session.audio)
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
                f"step={operation_id.batch_id} "
                f"op={operation_id.request_index} "
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
            self._sessions.pop(key)

    def close(self) -> None:
        """Discard all active assembly sessions and reject new media work."""
        self._sessions.clear()


def _key_label(request_key: RequestKey) -> str:
    """Format a stable request key for media task profiling."""
    return (
        f"{request_key.engine_id}:{request_key.request_id}:"
        f"{request_key.request_epoch}"
    )
