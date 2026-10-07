"""Decode a video request's condition media exactly as the server planned.

The server fetches every condition's bytes, plans how each is prepared and
publishes the bytes to POSIX shared memory (``MediaLocator``). The functions
here turn those bytes into the encoders' inputs with the operations the
reference conditioning applies, in its order, so the results equal the
reference's own:

- ``read_image``: PIL decoding with the EXIF orientation applied and RGB
  conversion (diffusers' ``load_image``), then a LANCZOS resize and the
  planned crop (``ImageFit``): a stretched keyframe, a cover-cropped one,
  or an image reference at its 2048-pixel short edge.
- ``read_video``: one FFmpeg pass resampling the video to 24 fps, skipping
  the planned start frames and scaling onto the planned canvas with LANCZOS
  (``fps=24,trim=start_frame=S,scale=W:H:flags=lanczos,setsar=1``) to RGB24.
- ``read_audio``: PyAV decoding to planar float at the track's native rate
  and channel layout, the planned native samples kept, a mono track
  duplicated to stereo, and one torchaudio resample to the model's rate.

Every function checks that it produced exactly the planned extent and
raises ``ValueError`` otherwise, so media that decodes differently from what
the server probed fails the request instead of misaligning its rows.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import torch

from uniserve_worker.protocol.video import (
    AudioClip,
    ImageFit,
    MediaLocator,
    VideoClip,
)

#: Where Linux mounts POSIX shared memory: a published object's file path.
SHARED_MEMORY = Path("/dev/shm")

#: The frame rate the model conditions on, and video references resample to.
FRAME_RATE = 24


def media_path(locator: MediaLocator) -> Path:
    """Return the file path of a published condition's bytes.

    The publisher keeps the object until the request retires, so readers
    open it by name for the request's lifetime and never unlink it.
    """
    return SHARED_MEMORY / locator.name


def read_bytes(locator: MediaLocator) -> bytes:
    """Read a published condition's bytes.

    Raises:
        ValueError: The object holds fewer bytes than its locator states.
    """
    with media_path(locator).open("rb") as file:
        data = file.read(locator.bytes)
    if len(data) != locator.bytes:
        raise ValueError("condition media is shorter than its locator")
    return data


def read_image(data: bytes, fit: ImageFit) -> np.ndarray:
    """Decode an image and put it on its planned raster.

    Returns:
        ``[1, height, width, 3]`` uint8 RGB at ``fit.size``.
    """
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    image = image.convert("RGB")
    resized = (fit.resized.width, fit.resized.height)
    # Pillow returns a copy for a resize to the image's own size, so an
    # image already at its raster passes through unchanged.
    if image.size != resized:
        image = image.resize(resized, Image.Resampling.LANCZOS)
    size = fit.size
    if (fit.left, fit.top, size.width, size.height) != (0, 0, *resized):
        image = image.crop(
            (fit.left, fit.top, fit.left + size.width, fit.top + size.height)
        )
    # A writable copy, which tensors may view.
    pixels = np.array(image)
    if pixels.shape != (size.height, size.width, 3):
        raise ValueError("a condition image decoded to another raster")
    return pixels[None]


def read_video(path: Path, clip: VideoClip, *, ffmpeg: str) -> np.ndarray:
    """Decode a reference video's kept frames on its planned canvas.

    The video is resampled to 24 fps before its start frames are skipped,
    so the offset counts frames of the 24 fps timeline. Scaling follows the
    trim and is per frame, so it never sees a skipped frame.

    Returns:
        ``[frames, height, width, 3]`` uint8 RGB, ``clip.frames`` frames at
        ``clip.canvas``.

    Raises:
        ValueError: FFmpeg failed or produced another frame count.
    """
    canvas = clip.canvas
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={FRAME_RATE},trim=start_frame={clip.start_frame},"
        f"scale={canvas.width}:{canvas.height}:flags=lanczos,setsar=1",
        "-frames:v",
        str(clip.frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    frames = np.empty(
        (clip.frames, canvas.height, canvas.width, 3), dtype=np.uint8
    )
    # FFmpeg's frames stream straight into the result; its diagnostics go to
    # a file, so a full pipe never stalls it.
    with (
        tempfile.TemporaryFile() as errors,
        subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors
        ) as process,
    ):
        assert process.stdout is not None
        # A binary pipe at the default buffering is an ``io.BufferedReader``
        # (``subprocess.Popen.stdout``), which ``IO[bytes]`` does not spell.
        stdout = cast(io.BufferedReader, process.stdout)
        target = memoryview(frames).cast("B")
        filled = 0
        while filled < len(target):
            count = stdout.readinto(target[filled:])
            if not count:
                break
            filled += count
        # ``-frames:v`` stops the output at the planned count.
        trailing = stdout.read()
        code = process.wait()
        errors.seek(0)
        message = errors.read().decode(errors="replace").strip()
    if code:
        raise ValueError(f"decoding a reference video failed: {message}")
    frame_bytes = canvas.height * canvas.width * 3
    if filled != len(target) or trailing:
        raise ValueError(
            f"a reference video decoded to "
            f"{(filled + len(trailing)) / frame_bytes:g} frames, "
            f"{clip.frames} were planned"
        )
    return frames


def read_audio(source: bytes | Path, clip: AudioClip, *, rate: int):
    """Decode an audio track's kept samples at the model's rate.

    ``source`` is an audio file's bytes or a video file's path; the first
    audio stream is read. Decoding keeps the native rate and channel layout
    as planar float; the planned native samples are kept, a mono track is
    duplicated to stereo, and one resample brings the track to ``rate``.

    Returns:
        ``[samples, 2]`` FP32 sample-major PCM, ``clip.samples`` samples.

    Raises:
        ValueError: The track has no audio stream, another rate, too few
            samples, or resamples to another count.
    """
    import av

    container = av.open(
        io.BytesIO(source) if isinstance(source, bytes) else str(source),
        mode="r",
    )
    with container:
        if not container.streams.audio:
            raise ValueError("a condition carries no audio stream")
        stream = container.streams.audio[0]
        native = int(stream.codec_context.sample_rate)
        if native != clip.sample_rate:
            raise ValueError(
                f"an audio track decodes at {native} Hz, "
                f"{clip.sample_rate} Hz were planned"
            )
        # Planar float is a format conversion only: rate and layout stay
        # the stream's own.
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout=stream.layout, rate=native
        )
        chunks: list[torch.Tensor] = []
        for frame in container.decode(stream):
            chunks.extend(
                torch.from_numpy(part.to_ndarray())
                for part in resampler.resample(frame)
            )
        chunks.extend(
            torch.from_numpy(part.to_ndarray())
            for part in resampler.resample(None)
        )
    # [channels, samples] at the native rate.
    waveform = torch.cat(chunks, dim=-1).to(torch.float32)
    stop = clip.start_sample + clip.source_samples
    if waveform.shape[-1] < stop or waveform.shape[0] not in (1, 2):
        raise ValueError("an audio track holds fewer samples than planned")
    waveform = waveform[:, clip.start_sample : stop]
    if waveform.shape[0] != 2:
        waveform = waveform.expand(2, -1).contiguous()
    if native != rate:
        import torchaudio

        waveform = torchaudio.transforms.Resample(native, rate)(waveform)
    if waveform.shape[-1] != clip.samples:
        raise ValueError(
            f"an audio track resampled to {waveform.shape[-1]} samples, "
            f"{clip.samples} were planned"
        )
    return waveform.transpose(0, 1).contiguous()


__all__ = [
    "FRAME_RATE",
    "SHARED_MEMORY",
    "media_path",
    "read_audio",
    "read_bytes",
    "read_image",
    "read_video",
]
