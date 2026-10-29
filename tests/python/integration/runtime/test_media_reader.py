"""The media reader decodes MiniMax-H3 conditions as the reference does.

Each case reads the media of a recorded diffusers reference run through the
media reader's ``read_conditions``, from shared memory, with the descriptors
the reference's sizes give (``tests/python/fixtures/h3_conditions.py``). The
reader's operations are the reference conditioning's own, in its order, so
every product is compared bit for bit:

- the conditioner's patch rows with the recorded processor output
  (``qwen_pixel_values`` and ``qwen_pixel_values_videos``): the same decode,
  LANCZOS resize and crop, frame sampling, and Qwen3-VL resize, rescale,
  normalization and patch order;
- an image's encoded pixels with the diffusers pipeline's prepared image;
- every soundtrack's PCM with the diffusers normalization of the
  reference's own decode: PyAV planar float at the native rate, truncation
  there, and one torchaudio resample to 32 kHz.

A video's encoded frames come from the same single FFmpeg pass as the frames
the conditioner samples, so the sampled frames' equality also checks the
pass.

Every image reference the MiniMax-H3 workloads condition on is also read at
its planned raster and compared with Pillow's own LANCZOS resize of the
decoded file, byte for byte, since the reader runs that resize as bands on
several threads.

A reference video decoded on a GPU's NVDEC is compared with the CPU decode
of the same file, byte for byte; that case needs a CUDA device and an FFmpeg
with the ``cuda`` hardware accelerator (``UNISERVE_FFMPEG``, or ``ffmpeg``
on ``PATH``).

The cases need the reference artifacts (``UNISERVE_MINIMAX_H3_REFERENCE``,
holding ``inputs/media`` and ``reference/diffusers``); the recorded runs'
cases also need the FFmpeg build the reference decoded with
(``UNISERVE_FFMPEG``, with ``ffprobe`` beside it).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from tests.python.fixtures.h3_conditions import (
    SAMPLE_RATE,
    published,
    read,
    recorded_request,
    vision_encoder,
)
from uniserve.media import image
from uniserve_models.minimax_h3.packing import FPS
from uniserve_worker.media.reader import (
    read_image,
    read_video,
    require_hwaccel,
)
from uniserve_worker.protocol.video import ImageFit, VideoClip

pytestmark = [
    pytest.mark.integration,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]


def _nearest_multiple(value: float) -> int:
    """Round a side to the nearest multiple of 32, half to even."""
    return max(32, round(value / 32) * 32)


@pytest.fixture(scope="module")
def root() -> Path:
    value = os.environ.get("UNISERVE_MINIMAX_H3_REFERENCE")
    if not value:
        pytest.skip("UNISERVE_MINIMAX_H3_REFERENCE names no reference run")
    return Path(value)


@pytest.fixture(scope="module")
def ffmpeg(root: Path) -> str:
    """The FFmpeg build the reference decoded its videos with."""
    value = os.environ.get("UNISERVE_FFMPEG")
    if not value:
        pytest.skip("UNISERVE_FFMPEG names no FFmpeg build")
    version = subprocess.run(
        [value, "-version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    metadata = json.loads(
        (
            root
            / "reference/diffusers/ref2va_video_audio_5s/seed42"
            / "metadata.json"
        ).read_text()
    )
    assert version == metadata["versions"]["ffmpeg"]
    return value


@pytest.mark.parametrize(
    "case",
    ["fl2va_first_8s", "ref2va_image_audio_5s", "ref2va_video_audio_5s"],
)
def test_conditions_read_as_the_reference_conditioning(case, root, ffmpeg):
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.modular_pipelines.minimax_h3 import references
    from diffusers.modular_pipelines.minimax_h3.before_encoder import (
        MiniMaxH3Ref2VASetupStep,
    )
    from diffusers.utils import load_image

    with published() as publish:
        run, request, paths, conditions = recorded_request(root, case, publish)
        pixels, samples, patches = read(conditions, vision_encoder(), ffmpeg)

    # The conditioner's rows, split by modality as the processor batches
    # them; each modality's rows follow request order.
    recorded = load_file(run / "text.safetensors")
    rows = {"image": [], "video": []}
    offset = 0
    for condition in conditions:
        if condition.vision is None:
            continue
        count = condition.vision.patches
        kind = "image" if condition.video is None else "video"
        rows[kind].append(patches[offset : offset + count])
        offset += count
    for kind, name in (
        ("image", "qwen_pixel_values"),
        ("video", "qwen_pixel_values_videos"),
    ):
        assert (name in recorded) == bool(rows[kind])
        if rows[kind]:
            assert torch.equal(torch.cat(rows[kind]), recorded[name])

    # The diffusers pipeline prepares an image with PIL's LANCZOS resize of
    # its decoded file: a keyframe onto the canvas, a reference image onto
    # its 2048-pixel short edge.
    resize = VaeImageProcessor(vae_scale_factor=16).resize
    offset = 0
    for condition, path in zip(conditions, paths, strict=True):
        if condition.image is not None:
            size = condition.image.size
            prepared = resize(
                load_image(str(path)), height=size.height, width=size.width
            )
            expected = torch.from_numpy(np.array(prepared)).reshape(-1, 3)
            count = condition.pixel_bytes // 3
            assert torch.equal(pixels[offset : offset + count], expected)
        offset += condition.pixel_bytes // 3

    # Every soundtrack as the diffusers setup step normalizes it, truncated
    # to the generated duration.
    offset = 0
    for condition, path in zip(conditions, paths, strict=True):
        if condition.audio is None:
            continue
        waveform, rate = references._decode_audio_file(path)
        expected = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
            waveform, rate, SAMPLE_RATE, max_duration=request.num_frames / FPS
        )
        count = condition.audio.samples
        assert torch.equal(samples[offset : offset + count], expected.T)
        offset += count


@pytest.mark.parametrize(
    "name",
    [
        "hf_character_action_reference.png",
        "official_fl2va_0_image.png",
        "hf_fashion_glasses_reference_3.png",
        "square_character_action_reference.png",
        "square_official_fl2va_0_image.png",
        "square_fashion_glasses_reference_1.png",
        "square_fashion_glasses_reference_2.png",
        "square_fashion_glasses_reference_3.png",
        "square_fl2va_clay_fox_reference.png",
        "square_h3_cinematic_shot_3s.png",
        "square_h3_suspense_title_2s.png",
        "square_robot_arm_red_cube_1s.png",
    ],
)
@pytest.mark.parametrize("enlarge", (False, True))
def test_image_references_resize_as_pillow(name, enlarge, root):
    from PIL import Image, ImageOps

    data = (root / "inputs/media" / name).read_bytes()
    decoded = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    decoded = decoded.convert("RGB")
    # Every image here is smaller than its 2048 short edge: it keeps its
    # own size on the 32-pixel grid, or is enlarged to that edge as the
    # reference pipeline enlarges it.
    width, height = decoded.size
    scale = 2048 / min(width, height) if enlarge else 1.0
    raster = image.Config(
        height=_nearest_multiple(height * scale),
        width=_nearest_multiple(width * scale),
    )
    fit = ImageFit(resized=raster, left=0, top=0, size=raster)

    expected = decoded.resize(
        (raster.width, raster.height), Image.Resampling.LANCZOS
    )
    pixels = read_image(data, fit)
    np.testing.assert_array_equal(pixels[0], np.asarray(expected))


def _cuda_ffmpeg() -> str | None:
    """An FFmpeg that decodes on NVDEC here, or None."""
    value = os.environ.get("UNISERVE_FFMPEG") or shutil.which("ffmpeg")
    if value is None or not torch.cuda.is_available():
        return None
    listed = subprocess.run(
        [value, "-hide_banner", "-hwaccels"], capture_output=True, text=True
    ).stdout.split()
    return value if "cuda" in listed else None


@pytest.mark.skipif(_cuda_ffmpeg() is None, reason="NVDEC is unavailable")
def test_nvdec_reads_a_reference_video_as_the_cpu_does(tmp_path):
    import av

    # A 30 fps H.264 clip of moving gradients, resampled to the model's
    # 24 fps, trimmed and scaled onto a smaller canvas of another aspect.
    height, width, frames = 720, 1280, 60
    path = tmp_path / "reference.mp4"
    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(frames):
            pixels = np.stack(
                (
                    (rows + 4 * index) % 256 + 0 * columns,
                    (columns + 7 * index) % 256 + 0 * rows,
                    (rows + columns + 3 * index) % 256,
                ),
                axis=-1,
            ).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    clip = VideoClip(
        canvas=image.Config(height=480, width=832),
        start_frame=5,
        frames=33,
        vae_frames=33,
    )
    ffmpeg = _cuda_ffmpeg()
    cpu = read_video(path, clip, frame_rate=FPS, ffmpeg=ffmpeg)
    nvdec = read_video(path, clip, frame_rate=FPS, ffmpeg=ffmpeg, gpu=0)
    np.testing.assert_array_equal(nvdec, cpu)


@pytest.mark.skipif(_cuda_ffmpeg() is None, reason="NVDEC is unavailable")
def test_a_reader_accelerator_must_name_a_gpu_ffmpeg_can_open():
    require_hwaccel(_cuda_ffmpeg(), 0)
    with pytest.raises(RuntimeError, match="cannot decode on cuda:99"):
        require_hwaccel(_cuda_ffmpeg(), 99)
