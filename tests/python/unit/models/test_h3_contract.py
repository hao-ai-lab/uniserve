"""Supported checkpoint metadata controls capabilities.

Control is independent of directory names.
"""

import json
import shutil
from pathlib import Path

import pytest

from uniserve.loading import Config as IOConfig
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
