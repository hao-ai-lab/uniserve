"""Supported checkpoint metadata controls capabilities.

Control is independent of directory names.
"""

import json
import shutil
from pathlib import Path

import pytest
import torch

from uniserve.loading import Config as IOConfig
from uniserve_models.minimax_h3 import Model
from uniserve_models.minimax_h3.config import Config, read_config

pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint(tmp_path):
    source = Path(__file__).parents[2] / "fixtures/models/fasth3"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_full_vsa_checkpoint_resolves_its_trained_rungs(checkpoint):
    config = read_config(checkpoint, IOConfig())
    assert config.diffusion.ladder == (999, 749, 500, 250)
    assert config.diffusion.time_scale == 1000
    assert config.diffusion.video_shift == 12
    assert config.diffusion.audio_shift == 3
    assert config.denoiser.vsa_sparsity == 0.9


def test_eight_step_checkpoint_owns_its_ladder_and_shifts(checkpoint):
    inference_path = checkpoint / "fastvideo_inference.json"
    inference = json.loads(inference_path.read_text())
    inference.update(
        {
            "transformer_forwards": 8,
            "num_inference_steps": 9,
            "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
            "vsa_sparsity": 0.8,
            "video_scheduler_shift": 10.0,
            "audio_scheduler_shift": 3.0,
        }
    )
    inference_path.write_text(json.dumps(inference))
    scheduler_path = checkpoint / "scheduler/scheduler_config.json"
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["shift"] = 10.0
    scheduler_path.write_text(json.dumps(scheduler))

    config = read_config(checkpoint, IOConfig())

    assert config.diffusion.ladder == (999, 874, 749, 624, 500, 375, 250, 125)
    assert config.diffusion.video_shift == 10
    assert config.diffusion.audio_shift == 3
    assert config.denoiser.vsa_sparsity == 0.8
    with torch.device("meta"):
        model = Model(config)

    # The first and last evaluations follow the shifted timestep equation
    # sigma = s t / (1 + (s - 1) t) with t = step / 1000 and each modality's
    # trained shift; the clean endpoint follows the last evaluation.
    schedules = model.denoiser.make_schedules(8, shift=None, device="cpu")
    for name, shift in (("video", 10.0), ("audio", 3.0)):
        sigmas = schedules[name].sigmas
        assert sigmas.shape == (9,)
        for index, step in ((0, 999), (7, 125)):
            t = step / 1000
            expected = shift * t / (1 + (shift - 1) * t)
            torch.testing.assert_close(
                sigmas[index], torch.tensor(expected), rtol=0, atol=1e-7
            )
        assert sigmas[8] == 0
    with pytest.raises(ValueError, match="8 evaluations"):
        model.denoiser.make_schedules(4, shift=None, device="cpu")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "ref2va"),
        ("attention_backend", "FLASH_ATTN"),
        ("guidance_scale", 3.0),
        ("transformer_forwards", 50),
        ("num_inference_steps", 4),
        ("dmd_denoising_steps", [999, 999, 500, 250]),
        ("dmd_denoising_steps", [1001, 749, 500, 250]),
        ("vsa_sparsity", 1.0),
        ("video_scheduler_shift", 10.0),
        ("schema_version", "unknown"),
    ],
)
def test_incompatible_checkpoint_is_rejected(checkpoint, field, value):
    path = checkpoint / "fastvideo_inference.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        read_config(checkpoint, IOConfig())


def test_architecture_without_variant_metadata_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="fastvideo_inference.json"):
        read_config(tmp_path, IOConfig())


@pytest.mark.parametrize(
    "sidecar, field, value, error",
    [
        (
            "vae/config.json",
            "decoder_num_layers",
            35,
            "video_decoder.decoder_num_layers",
        ),
        (
            "audio_vae/config.json",
            "latents_std",
            [0.0] * 32,
            "standard deviations",
        ),
        (
            "vae/config.json",
            "spatial_downsample_factors",
            16,
            "must be a sequence",
        ),
    ],
)
def test_h3_reader_rejects_unsupported_decoder_math(
    checkpoint, sidecar, field, value, error
):
    path = checkpoint / sidecar
    values = json.loads(path.read_text())
    values[field] = value
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match=error):
        read_config(checkpoint, IOConfig())


def test_h3_reader_reports_missing_decoder_field(checkpoint):
    path = checkpoint / "vae/config.json"
    values = json.loads(path.read_text())
    del values["latent_channels"]
    path.write_text(json.dumps(values))
    with pytest.raises(
        ValueError, match="video_vae is missing field latent_channels"
    ):
        read_config(checkpoint, IOConfig())


def test_h3_direct_configuration_preserves_cross_component_dimensions():
    from dataclasses import replace

    from uniserve_models.minimax_h3.encoder import TextEncoderConfig

    with pytest.raises(ValueError, match="conditioning width"):
        Config(text_encoder=replace(TextEncoderConfig(), hidden_size=4096))
