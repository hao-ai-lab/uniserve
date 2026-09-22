"""Codec work of a rank's host lane, executed in the lane's codec processes.

A rank's interpreter must not run a media codec. Closing an x264 encoder joins
its thread pool while PyAV holds the interpreter lock, and inside a rank
process that join takes hundreds of milliseconds, during which the rank's
service thread can neither launch device work nor report completions. The
host lane therefore owns codec processes, and this module is both sides of
that boundary: the jobs a rank describes, the loop a codec process runs, and
the client the lane talks to it through.

A codec process reads its media unit inputs from shared mappings the rank
pins for device-to-host copies, so a unit is handed over by offset rather than
copied. The module imports no torch, so a codec process stays small.
"""

from __future__ import annotations

import io
import mmap
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from multiprocessing.connection import Connection
from threading import Lock
from typing import Any

import numpy as np

from .storage import publish_media_bytes

__all__ = [
    "AvMuxConfig",
    "AvMuxSession",
    "CodecJob",
    "CodecProcess",
    "EncodeAudioTrack",
    "EncodeVideoUnit",
    "MuxAppend",
    "MuxClose",
    "MuxFinalize",
    "Probe",
    "SessionKey",
    "SharedSlice",
    "encode_audio_track",
    "encode_video_unit",
    "require_media_codecs",
]

# Each media unit is its own encoder session, and several ranks encode at once
# on one host, so a session must not size its thread pool to the machine.
# Measured at this preset and raster, eight threads encode a unit as fast as an
# unbounded pool does; one thread takes four to five times as long.
_ENCODER_THREADS = 8


def encoded_video_bytes(frames: int, height: int, width: int) -> int:
    """Bound the MP4 bytes of the serving libx264/yuv420p/ultrafast encoder.

    This preset uses baseline CAVLC. A padded 16x16 macroblock has at most
    27 residual blocks (including the separate DC blocks), each with up to
    16 coefficients. Baseline level codes occupy at most 28 bits; x264
    re-encodes CAVLC overflows at a higher QP. Per-block token, zero and run
    codes need at most 16 + 9 + 15*11 bits. A further 1536 bits covers all
    macroblock headers and 16 pairs of motion-vector differences. NAL escaping
    adds at most one byte per two source bytes.

    Eight slice headers per frame need at most 1024 bytes each. MP4 sample
    tables need at most 64 bytes per packet; parameter sets, encoder SEI and
    fixed container boxes fit in 64 KiB. These are syntax bounds, independent
    of image entropy or the achieved compression ratio. See x264's
    encoder/cavlc.c and encoder/encoder.c and FFmpeg's libavformat/movenc.c.
    """
    if min(frames, height, width) < 1:
        raise ValueError("encoded video dimensions must be positive")
    blocks = ((height + 15) // 16) * ((width + 15) // 16)
    residual_bits = 27 * (16 + 16 * 28 + 9 + 15 * 11)
    macroblock_bytes = (residual_bits + 1536 + 7) // 8
    escaped_bytes = (3 * macroblock_bytes + 1) // 2
    return (1 << 16) + frames * (
        blocks * escaped_bytes + _ENCODER_THREADS * 1024 + 64
    )


# A mux session is identified by the request it assembles: the engine id, the
# request id and the request epoch, which is what a RequestKey carries.
SessionKey = tuple[int, int, int]


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
            "media output requires the codecs "
            + ", ".join(missing)
            + " which this PyAV runtime does not provide"
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


def encode_video_unit(config: AvMuxConfig, rgb24: np.ndarray) -> bytes:
    """Encode one media unit as a self-contained MP4 starting at a keyframe."""
    import av

    if (
        rgb24.ndim != 4
        or rgb24.shape[1:] != (config.height, config.width, 3)
        or rgb24.shape[0] < 1
    ):
        raise ValueError("video capture has invalid RGB24 dimensions")
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
        raise ValueError("audio capture has invalid stereo dimensions")
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
            raise ValueError(
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
            raise ValueError(
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


@dataclass(frozen=True, slots=True)
class SharedSlice:
    """A media unit's bytes inside a POSIX shared-memory segment.

    The segment is a host product's publication, named in this host's
    shared-memory namespace; a codec process maps it for the job that reads
    it and unmaps it afterwards, so the producer's retirement of the segment
    never waits on a codec process.
    """

    segment: str
    offset: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class Probe:
    """Confirm the process serves and its codecs load; the result is True."""

    video_codec: str = "libx264"
    audio_codec: str = "aac"


@dataclass(frozen=True, slots=True)
class EncodeVideoUnit:
    """Encode the RGB24 frames of one media unit; the result is its bytes."""

    config: AvMuxConfig
    source: SharedSlice


@dataclass(frozen=True, slots=True)
class EncodeAudioTrack:
    """Encode a request's complete stereo int16 PCM timeline into its track."""

    session: SessionKey
    config: AvMuxConfig
    source: SharedSlice


@dataclass(frozen=True, slots=True)
class MuxAppend:
    """Append encoded media units, in order, to a request's container."""

    session: SessionKey
    config: AvMuxConfig
    units: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class MuxFinalize:
    """Mux the session's audio track and publish the artifact.

    The result is the artifact's shared-memory name and its byte count; the
    session is gone afterwards.
    """

    session: SessionKey


@dataclass(frozen=True, slots=True)
class MuxClose:
    """Discard a session whose request ended before its artifact."""

    session: SessionKey


CodecJob = (
    Probe
    | EncodeVideoUnit
    | EncodeAudioTrack
    | MuxAppend
    | MuxFinalize
    | MuxClose
)


class _Session:
    """One request's assembly state inside a codec process."""

    def __init__(self, config: AvMuxConfig) -> None:
        self.container = AvMuxSession(config)
        self.audio: bytes | None = None


def _read(source: SharedSlice) -> np.ndarray:
    """View a media unit's bytes in its shared-memory segment without copying.

    The segment is opened by name in this host's shared-memory namespace,
    read-only; the mapping lives as long as the returned view and is released
    with it, so a codec that keeps a frame's storage alive keeps the mapping.
    """
    if source.offset < 0 or source.nbytes < 0:
        raise ValueError("media unit lies outside its shared-memory segment")
    name = source.segment.removeprefix("/")
    descriptor = os.open(f"/dev/shm/{name}", os.O_RDONLY)
    try:
        size = os.fstat(descriptor).st_size
        if source.offset + source.nbytes > size:
            raise ValueError(
                "media unit lies outside its shared-memory segment"
            )
        mapping = mmap.mmap(
            descriptor,
            source.offset + source.nbytes,
            prot=mmap.PROT_READ,
        )
    finally:
        os.close(descriptor)
    return np.frombuffer(
        mapping, dtype=np.uint8, count=source.nbytes, offset=source.offset
    )


def _execute(job: CodecJob, sessions: dict[SessionKey, _Session]) -> object:
    """Run one job against the process's sessions."""
    if isinstance(job, Probe):
        require_media_codecs(job.video_codec, job.audio_codec)
        return True

    if isinstance(job, EncodeVideoUnit):
        raster = job.config.height * job.config.width * 3
        if job.source.nbytes % raster != 0:
            raise ValueError("video capture has invalid RGB24 dimensions")
        pixels = _read(job.source).reshape(
            -1, job.config.height, job.config.width, 3
        )
        return encode_video_unit(job.config, pixels)

    if isinstance(job, EncodeAudioTrack):
        session = sessions.get(job.session)
        if session is None:
            session = sessions[job.session] = _Session(job.config)
        if session.audio is not None:
            raise ValueError("audio output is already written")
        track = _read(job.source).view(np.int16).reshape(-1, 2)
        session.audio = encode_audio_track(job.config, track)
        return None

    if isinstance(job, MuxAppend):
        session = sessions.get(job.session)
        if session is None:
            session = sessions[job.session] = _Session(job.config)
        session.container.append(job.units)
        return None

    if isinstance(job, MuxFinalize):
        session = sessions.pop(job.session, None)
        if session is None:
            raise ValueError("artifact finalization has no assembly session")
        if session.audio is None:
            raise ValueError("artifact assembly has no encoded audio")
        payload = session.container.finalize(session.audio)
        return publish_media_bytes(payload), len(payload)

    if isinstance(job, MuxClose):
        session = sessions.pop(job.session, None)
        if session is not None:
            session.container.close()
        return None

    raise TypeError(f"unsupported codec job {type(job).__name__}")


def codec_main(connection: Connection) -> None:
    """Serve jobs over one connection until it closes or sends None.

    Every message is answered with ("ok", value) or ("error", exception), so
    a failure of one job reaches its task without ending the process.
    """
    sessions: dict[SessionKey, _Session] = {}
    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            if message is None:
                break

            kind = message[0]
            try:
                if kind == "job":
                    value = _execute(message[1], sessions)
                else:
                    raise TypeError(f"unsupported codec message {kind!r}")
            except BaseException as error:  # noqa: BLE001 - answered
                connection.send(("error", error))
            else:
                connection.send(("ok", value))
    finally:
        for session in sessions.values():
            session.container.close()
        connection.close()


class CodecProcess:
    """One codec process and the rank's side of its connection.

    The process is started as its own interpreter running this module, so it
    inherits nothing of the rank but its environment; the connection is one
    end of a socket pair. A job names the shared-memory segment it reads, so
    nothing but the job travels. Calls are serialized by a lock, because a
    process runs one job at a time; the host lane keeps one process per
    worker so its capacity is the lane's. Transfer releases the interpreter
    lock, so a job in flight costs the rank nothing but the bytes it
    exchanges.
    """

    def __init__(self) -> None:
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    _MODULE,
                    str(theirs.fileno()),
                ],
                pass_fds=(theirs.fileno(),),
                stdin=subprocess.DEVNULL,
            )
        finally:
            theirs.close()
        self._connection = Connection(ours.detach())
        self._lock = Lock()

    def execute(self, job: CodecJob) -> object:
        """Run one job to completion and return its result."""
        with self._lock:
            self._connection.send(("job", job))
            return self._reply()

    def _reply(self) -> object:
        try:
            kind, value = self._connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(
                "codec process ended before answering"
            ) from error
        if kind == "error":
            raise value
        return value

    def abort(self) -> None:
        """Stop the codec without acquiring its in-flight job lock."""
        if self._process.poll() is None:
            self._process.kill()

    def close(self) -> None:
        """End the process, forcibly if it does not exit on request."""
        with self._lock:
            try:
                self._connection.send(None)
            except (BrokenPipeError, OSError):
                pass
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._connection.close()


_MODULE = "uniserve_worker.media.codec_process"


def _serve(argv: list[str]) -> None:
    """Entry point of a codec process: serve the connection it was handed."""
    if len(argv) != 2:
        raise SystemExit(f"usage: python -m {_MODULE} <connection-fd>")
    codec_main(Connection(int(argv[1])))


if __name__ == "__main__":
    # Serve through the module under its import name, so the jobs a rank sends
    # unpickle as the classes the executor tests them against.
    from uniserve_worker.media import codec_process

    codec_process._serve(sys.argv)
