"""The MiniMax-H3 library generates a clip from a diffusers root eagerly.

Loads the base checkpoint's text encoder, denoiser, decoders and
post-processor on one GPU through the public loader, tokenizes the prompt of a
four-second 9:16 t2va request and runs ``generation.generate`` at the canvas
and frame count the model package's rules give. The clip has those frames at
that canvas and a stereo track covering them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from uniserve_models import loading as models
from uniserve_models.minimax_h3 import config as h3_config
from uniserve_models.minimax_h3 import processing, weight_config
from uniserve_models.minimax_h3.generation import generate

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("minimax_h3")]


def test_library_generates_a_portrait_clip() -> None:
    root = os.environ.get("UNISERVE_MINIMAX_H3_MODEL")
    if not root or not Path(root).is_dir():
        pytest.fail(
            "UNISERVE_MINIMAX_H3_MODEL must name a MiniMax-H3 diffusers root"
        )
    transformers = pytest.importorskip("transformers")

    config = models.read_config(
        root,
        modules=frozenset(
            {
                "text_encoder",
                "transformer",
                "video_decoder",
                "video_postprocessor",
                "audio_decoder",
            }
        ),
    )
    model = models.load_model(
        config, device="cuda", weights=weight_config(config.model)
    ).model

    canvas = h3_config.canvas(9, 16)
    num_frames = processing.frame_count(4.0)
    # A text-to-video presentation is the prompt's tokens alone.
    token_ids = transformers.AutoTokenizer.from_pretrained(
        Path(root) / "tokenizer"
    ).encode(
        "A lighthouse keeper climbs a spiral staircase at dusk while waves "
        "crash against the rocks below.",
        add_special_tokens=False,
    )
    generation = generate(
        model,
        prompt_token_ids=tuple(token_ids),
        num_frames=num_frames,
        canvas=canvas,
        seed=7,
    )

    # Four seconds at 24 fps align up to 107 frames of the 768x1344 canvas.
    assert (canvas.width, canvas.height) == (768, 1344)
    assert num_frames == 107
    assert generation.frames.shape == (107, 1344, 768, 3)
    assert generation.frames.dtype == torch.uint8
    # The stereo track spans the clip at the audio decoder's 32 kHz.
    samples, channels = generation.audio.shape
    assert channels == 2
    assert abs(samples / 32_000 - 107 / 24) <= 1 / 24
    assert generation.frames.float().std() > 0
    assert generation.audio.float().abs().max() > 0
