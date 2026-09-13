# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU preparation of inline H3 reference media.

Canvas and frame selection follow FastVideo's released Ref2VA preparation.
Presentation frames and complete visual-VAE chunks are deliberately distinct;
the soundtrack follows the presentation window rather than the chunk trim.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import av
import numpy as np
from PIL import Image, ImageOps

MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_EDGE = 8192
MAX_SOURCE_PIXELS = 4096 * 4096
MAX_REFERENCE_FRAMES = 345
FPS = 24
AUDIO_RATE = 32_000


@dataclass(frozen=True, slots=True)
class PreparedReferenceVideo:
    """RGB uint8 THWC presentation frames and optional stereo float32 audio.

    Audio is channels-first at 32 kHz. The visual VAE consumes ``vae_frames``;
    Qwen sampling consumes ``frames``. All arrays are CPU-owned.
    """

    frames: np.ndarray
    waveform: np.ndarray | None

    @property
    def vae_frames(self) -> np.ndarray:
        return self.frames[: trim_reference_num_frames(len(self.frames))]


def trim_reference_num_frames(num_frames: int) -> int:
    """Select the released causal-VAE 17n+5 window (minimum 22)."""

    if num_frames < 1:
        raise ValueError("reference video must contain frames")
    return max(1, (num_frames - 5) // 17) * 17 + 5


def reference_canvas(width: int, height: int, *, video: bool) -> tuple[int, int]:
    """Return (height, width) using the released canvas rounding and area cap."""

    if min(width, height) <= 0:
        raise ValueError("reference dimensions must be positive")
    if not video and (width > 4 * height or height > 4 * width):
        raise ValueError("reference image aspect must be within 1:4 through 4:1")
    scale = (768 if video else 2048) / min(width, height)
    h, w = height * scale, width * scale
    if video and h * w > 768 * 1344:
        scale = math.sqrt(768 * 1344 / (h * w))
        h, w = h * scale, w * scale
    return max(32, round(h / 32) * 32), max(32, round(w / 32) * 32)


def _validate_source(source: bytes) -> None:
    if not source or len(source) > MAX_SOURCE_BYTES:
        raise ValueError("reference source must contain 1 through 32 MiB of encoded media")


def _validate_raster(width: int, height: int) -> None:
    if (
        min(width, height) <= 0
        or max(width, height) > MAX_SOURCE_EDGE
        or width * height > MAX_SOURCE_PIXELS
    ):
        raise ValueError("reference source raster exceeds the decode budget")


def prepare_reference_image(source: bytes) -> np.ndarray:
    """Decode bounded PNG/JPEG bytes to the EXIF-corrected 2048-edge RGB canvas."""

    _validate_source(source)
    with Image.open(io.BytesIO(source)) as image:
        if image.format not in ("PNG", "JPEG"):
            raise ValueError("reference image requires PNG or JPEG")
        _validate_raster(*image.size)
        image = ImageOps.exif_transpose(image).convert("RGB")
        height, width = reference_canvas(*image.size, video=False)
        return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS)).copy()


def sample_reference_video_frames(frames: np.ndarray) -> tuple[np.ndarray, tuple[float, ...]]:
    """Select 2-fps Qwen frames and temporal-patch block midpoint timestamps."""

    if frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames):
        raise ValueError("reference frames must be nonempty THWC RGB")
    sampled = frames[::12]
    timestamps = [index / 2 for index in range(len(sampled))]
    if len(timestamps) % 2:
        timestamps.append(timestamps[-1])
    blocks = tuple(
        (timestamps[index] + timestamps[index + 1]) / 2 for index in range(0, len(timestamps), 2)
    )
    return sampled, blocks


def _decode_soundtrack(source: bytes, duration: float) -> np.ndarray | None:
    with av.open(io.BytesIO(source)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        rate = int(stream.codec_context.sample_rate)
        if not 1 <= rate <= 192_000:
            raise ValueError("reference audio sample rate exceeds the decode budget")
        # FFmpeg's channel-layout-aware mix preserves stereo directly and
        # downmixes multichannel sources rather than dropping rear/center audio.
        layout = stream.layout if len(stream.layout.channels) <= 2 else "stereo"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
        limit = int(duration * rate)
        chunks = []
        samples = 0
        for frame in container.decode(stream):
            for item in resampler.resample(frame):
                chunk = item.to_ndarray()[:, : max(0, limit - samples)]
                if chunk.shape[-1]:
                    chunks.append(chunk)
                    samples += chunk.shape[-1]
            if samples >= limit:
                break
        if samples < limit:
            for item in resampler.resample(None):
                chunk = item.to_ndarray()[:, : max(0, limit - samples)]
                if chunk.shape[-1]:
                    chunks.append(chunk)
                    samples += chunk.shape[-1]
        if not chunks:
            raise ValueError("reference audio stream contains no samples")
        waveform = np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)
        if waveform.shape[0] == 1:
            waveform = np.repeat(waveform, 2, axis=0)
        if rate != AUDIO_RATE:
            try:
                import torchaudio
            except (ImportError, OSError):
                from scipy.signal import resample_poly

                divisor = math.gcd(rate, AUDIO_RATE)
                waveform = resample_poly(waveform, AUDIO_RATE // divisor, rate // divisor, axis=-1)
            else:
                import torch

                waveform = torchaudio.transforms.Resample(rate, AUDIO_RATE)(
                    torch.from_numpy(waveform)
                ).numpy()
        return np.ascontiguousarray(waveform, dtype=np.float32)


def prepare_reference_video(source: bytes, *, num_frames: int) -> PreparedReferenceVideo:
    """Decode an inline video and its embedded audio within the target window.

    The target frame cap is selected before decoding. Input dimensions and rate
    are bounded before RGB materialization, and each raster is resized before
    accumulation. Invalid, empty, or sub-chunk videos raise ValueError; decoder
    format errors propagate from PyAV. No temporary files or child processes
    are created.
    """

    _validate_source(source)
    if not 22 <= num_frames <= MAX_REFERENCE_FRAMES:
        raise ValueError("reference frame budget must be within 22 through 345")
    duration = num_frames / FPS
    frames: list[np.ndarray] = []
    with av.open(io.BytesIO(source)) as container:
        if not container.streams.video:
            raise ValueError("reference media contains no video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate or getattr(stream, "guessed_rate", None)
        if rate is None or not math.isfinite(float(rate)) or not 1 <= float(rate) <= 240:
            raise ValueError("reference video requires a frame rate within 1 through 240")
        rate = float(rate)
        _validate_raster(stream.codec_context.width, stream.codec_context.height)
        for index, frame in enumerate(container.decode(stream)):
            timestamp = frame.time if frame.time is not None else index / rate
            if timestamp >= duration:
                break
            if index >= math.ceil(duration * 240):
                raise ValueError("reference video timestamps exceed the decode budget")
            _validate_raster(frame.width, frame.height)
            pixels = frame.to_ndarray(format="rgb24")
            turns = round(float(getattr(frame, "rotation", 0) or 0) / 90) % 4
            if turns:
                pixels = np.rot90(pixels, k=-turns, axes=(0, 1))
            height, width = reference_canvas(pixels.shape[1], pixels.shape[0], video=True)
            pixels = np.asarray(
                Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
            )
            # FastVideo rounds source frame slots to the nearest 24-fps slot,
            # then repeats each frame by the difference between adjacent slots.
            repeats = math.floor((index + 1) * FPS / rate + 0.5) - math.floor(
                index * FPS / rate + 0.5
            )
            frames.extend([pixels] * min(repeats, num_frames - len(frames)))
            if len(frames) == num_frames:
                break
    if len(frames) < 22:
        raise ValueError("reference video must contain at least 22 resampled frames")
    return PreparedReferenceVideo(np.stack(frames), _decode_soundtrack(source, duration))
