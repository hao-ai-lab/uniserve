"""Media codecs and MP4 assembly run on a host worker's ranks.

A host rank is one codec slot: it encodes one media unit, the audio track, or
one container step at a time on its host lane's thread, in its own process.
Closing an x264 encoder joins its thread pool while PyAV holds the
interpreter lock, which stalls the rank's service thread for that time; a
host rank has no device work to launch and no other task in flight, so the
stall only extends the task it belongs to.

This module holds the PyAV work itself; `uniserve_worker.media.mux` decides
which of these functions runs as which lane task and for which request.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any

import numpy as np

__all__ = [
    "AUDIO_CODEC",
    "VIDEO_CODEC",
    "AvMuxConfig",
    "AvMuxSession",
    "encode_audio_track",
    "encode_video_unit",
    "encoded_video_bytes",
    "require_media_codecs",
]

# The encoders every served container uses.
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"

# Each media unit is its own encoder session, and several ranks encode at once
# on one host, so a session must not size its thread pool to the machine.
# Measured at this preset and raster, eight threads encode a unit as fast as an
# unbounded pool does; one thread takes four to five times as long.
# `encoded_video_bytes` budgets one slice header per thread per frame, so this
# value also sizes the encoded unit rows reserved for a round's product
# (`encoded_unit_bytes` in `uniserve_worker.media.mux`).
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

    Up to `_ENCODER_THREADS` slice headers per frame need at most 1024 bytes
    each. MP4 sample tables need at most 64 bytes per packet; parameter sets,
    encoder SEI and fixed container boxes fit in 64 KiB. These are syntax
    bounds, independent of image entropy or the achieved compression ratio.
    See x264's encoder/cavlc.c and encoder/encoder.c and FFmpeg's
    libavformat/movenc.c.
    """
    if any(
        type(value) is not int or value < 1 for value in (frames, height, width)
    ):
        raise ValueError("encoded video dimensions must be positive integers")
    blocks = ((height + 15) // 16) * ((width + 15) // 16)
    residual_bits = 27 * (16 + 16 * 28 + 9 + 15 * 11)
    macroblock_bytes = (residual_bits + 1536 + 7) // 8
    escaped_bytes = (3 * macroblock_bytes + 1) // 2
    return (1 << 16) + frames * (
        blocks * escaped_bytes + _ENCODER_THREADS * 1024 + 64
    )


def require_media_codecs(video_codec: str, audio_codec: str) -> None:
    """Verify that the configured encoders are available through PyAV.

    Both codec names must be listed by PyAV and open as an encoder context.

    Raises:
        RuntimeError: When PyAV is not installed or does not list a codec.
            PyAV raises its own error when a listed codec cannot be created
            as an encoder.
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
    video_codec: str = VIDEO_CODEC
    audio_codec: str = AUDIO_CODEC
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
    """Encode one media unit as a self-contained MP4 starting at a keyframe.

    ``rgb24`` holds the unit's uint8 frames as ``[frames, height, width, 3]``.
    Frame timestamps start at zero in every unit; `AvMuxSession.append`
    shifts them onto the request's timeline.

    Raises:
        ValueError: When ``rgb24`` does not match the configured raster or
            holds no frame, or ``config.video_codec`` is not a video encoder.
            PyAV raises its own errors for codec and encoding failures.
    """
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
        if not isinstance(stream, av.VideoStream):
            raise ValueError(f"{config.video_codec} is not a video encoder")
        stream.width, stream.height = config.width, config.height
        # `encoded_video_bytes` derives its size bound from these encoder
        # settings. A change it does not account for can overflow the
        # reserved product row, which `frame_encoded_unit` in
        # `uniserve_worker.media.mux` rejects.
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

    ``pcm`` holds interleaved int16 samples as ``[samples, 2]``. The track is
    truncated or zero-padded to the video's duration, ``frame_count /
    frame_rate`` seconds at ``audio_rate``, and encoded in planar frames of
    ``audio_frame_samples`` samples; the last frame is zero-padded to that
    size when the track does not fill it.

    Raises:
        ValueError: When ``pcm`` is not a ``[samples, 2]`` array, or
            ``config.audio_codec`` is not an audio encoder. PyAV raises its
            own errors for codec and encoding failures.
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
        if not isinstance(stream, av.AudioStream):
            raise ValueError(f"{config.audio_codec} is not an audio encoder")
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
        # The container and its streams are created by the first `append`.
        self._buffer: io.BytesIO | None = None
        self._container: Any = None
        self._video_out: Any = None
        self._audio_out: Any = None
        # End of the video copied so far, in the unit packets' time base; the
        # next unit's timestamps are shifted by it.
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
        source = av.open(io.BytesIO(first_unit), mode="r")
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
        track = av.open(io.BytesIO(template), mode="r")
        try:
            self._audio_out = self._container.add_stream_from_template(
                track.streams.audio[0]
            )
        finally:
            track.close()

    def append(self, units: tuple[bytes, ...]) -> None:
        """Copy the packets of the next media units, in order.

        The first unit ever appended also creates the container (`_open`).

        Raises:
            ValueError: When the units would exceed the request's unit count;
                nothing is appended in that case.
        """
        import av

        if self.units_appended + len(units) > self.total_units:
            raise ValueError(
                "artifact assembly received more media units than the request"
            )
        for payload in units:
            if self._container is None:
                self._open(payload)
            source = av.open(io.BytesIO(payload), mode="r")
            try:
                stream = source.streams.video[0]
                last = self._offset
                for packet in source.demux(stream):
                    # ``demux`` ends a stream with an empty flush packet that
                    # has no timestamp and nothing to copy.
                    if packet.dts is None:
                        continue
                    packet.stream = self._video_out
                    packet.pts = (packet.pts or 0) + self._offset
                    packet.dts = packet.dts + self._offset
                    self._container.mux(packet)
                    # A packet without a recorded duration counts one tick.
                    last = max(last, packet.dts + (packet.duration or 1))
                self._offset = last
            finally:
                source.close()
            self.units_appended += 1

    def finalize(self, audio: bytes) -> bytes:
        """Mux the audio track after every unit and return the artifact.

        The session is closed once its container is finalized, so a later
        call raises.

        Raises:
            ValueError: When not every media unit has been appended, or the
                session was closed.
            RuntimeError: When the finished container is empty.
        """
        import av

        if self.units_appended != self.total_units or self._container is None:
            raise ValueError(
                "artifact assembly requires every media unit of the request"
            )
        # The audio track starts at time zero, as the first unit does, so
        # its packets are copied without an offset.
        track = av.open(io.BytesIO(audio), mode="r")
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
