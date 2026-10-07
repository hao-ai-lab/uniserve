"""Media codecs and MP4 assembly run on a host worker's ranks.

A host rank is one codec slot: it encodes one media unit, the audio track, or
one container step at a time on its host lane's thread, in its own process.
Closing an x264 encoder joins its thread pool while PyAV holds the
interpreter lock, which stalls the rank's service thread for that time; a
host rank has no device work to launch and no other task in flight, so the
stall only extends the task it belongs to.

This module holds the PyAV work itself. Native requests retain containers;
the executor schedules codec operations on the host lane.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from uniserve_worker._uniserve_ipc import MuxSession

__all__ = [
    "AUDIO_CODEC",
    "VIDEO_CODEC",
    "AvMuxConfig",
    "MuxSession",
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
    Frame timestamps start at zero in every unit; `MuxSession.append`
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
