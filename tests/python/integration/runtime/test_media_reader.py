"""The media reader decodes MiniMax-H3 conditions as the reference does.

Each case reads the media of a recorded diffusers reference run through the
media reader's ``read_conditions``, from shared memory, with the descriptors
the planner derives from the media's probed facts
(``processing.plan_request``). The reader's operations are the reference
conditioning's own, in its order, so every product is compared bit for bit:

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

The cases need the reference artifacts (``UNISERVE_MINIMAX_H3_REFERENCE``,
holding ``inputs/media`` and ``reference/diffusers``) and the FFmpeg build
the reference decoded with (``UNISERVE_FFMPEG``, with ``ffprobe`` beside it).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from tests.python.fixtures.h3_conditions import (
    FPS,
    SAMPLE_RATE,
    published,
    read,
    recorded_request,
    vision_encoder,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]


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
        run, plan, paths, conditions = recorded_request(
            root, case, ffmpeg, publish
        )
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
            waveform, rate, SAMPLE_RATE, max_duration=plan.num_frames / FPS
        )
        count = condition.audio.samples
        assert torch.equal(samples[offset : offset + count], expected.T)
        offset += count
