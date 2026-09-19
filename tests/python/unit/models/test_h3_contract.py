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


def test_full_vsa_checkpoint_resolves_t2va_and_inference_grid(checkpoint):
    config = read_config(checkpoint, IOConfig())
    assert config.diffusion.ladder == (1000, 750, 500, 250)
    assert config.diffusion.time_scale == 1000
    assert config.diffusion.video_shift == 12
    assert config.diffusion.audio_shift == 3


def test_eight_step_checkpoint_owns_its_ladder_and_shifts(checkpoint):
    inference_path = checkpoint / "fastvideo_inference.json"
    inference = json.loads(inference_path.read_text())
    inference.update(
        {
            "model_id": "FastVideo/FastVideo-FastH3-8-Step-V2",
            "checkpoint_content_sha256": (
                "516323fa396fa5dff4e82669d4e9a08a5791692a3d3b98ff6bc3de3fc6a33d11"
            ),
            "checkpoint_metadata_sha256": (
                "ca9f2d609c05742ba465d24989981ec02cca26acb6ca2f163dc0f6dc8d11c27b"
            ),
            "fastvideo_commit": "24bbe7fddd05ca6f2c34b3dbed06ac1c75b72086",
            "transformer_forwards": 8,
            "num_inference_steps": 9,
            "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
            "vsa_sparsity": 0.8,
        }
    )
    inference_path.write_text(json.dumps(inference))
    scheduler_path = checkpoint / "scheduler/scheduler_config.json"
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["shift"] = 10.0
    scheduler_path.write_text(json.dumps(scheduler))

    config = read_config(checkpoint, IOConfig())

    assert config.diffusion.ladder == (1000, 875, 750, 625, 500, 375, 250, 125)
    assert config.diffusion.video_shift == 10
    assert config.diffusion.audio_shift == 3
    with torch.device("meta"):
        model = Model(config)
    assert model.denoiser.transformer.modulation.products.shape[0] == 8
    assert model.denoiser.transformer.modulation.output_products.shape[0] == 8
    assert model.denoiser.transformer.layers["0"].attention.sparsity == 0.8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "ref2va"),
        ("attention_backend", "FLASH_ATTN"),
        ("transformer_forwards", 50),
        ("vsa_sparsity", 0.8),
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
