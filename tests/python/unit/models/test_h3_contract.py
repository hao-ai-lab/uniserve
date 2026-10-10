"""Supported checkpoint metadata controls capabilities.

Control is independent of directory names.
"""

import json
import shutil
from pathlib import Path

import pytest
import torch

from uniserve.diffusion import BlockGrid, RungGrid, UniformGrid
from uniserve.loading import Config as IOConfig
from uniserve.nn.functional import Rounding
from uniserve_models.minimax_h3 import (
    DenseAttention,
    Model,
    SparseAttention,
    entry_points,
)
from uniserve_models.minimax_h3.checkpoint import base_checkpoint, detect
from uniserve_models.minimax_h3.config import Config, read_config

pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint(tmp_path):
    source = Path(__file__).parents[2] / "fixtures/models/fasth3"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    return tmp_path


@pytest.fixture
def base(checkpoint):
    """The base release: both DiT partitions and no FastH3 contract."""
    (checkpoint / "fastvideo_inference.json").unlink()
    shutil.copytree(checkpoint / "transformer", checkpoint / "transformer_ref")
    component = ["diffusers", "MiniMaxH3Transformer3DModel", {}]
    (checkpoint / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ModularPipeline",
                "transformer": component,
                "transformer_ref": component,
            }
        )
    )
    return checkpoint


def test_t2va_fasth3_resolves_its_trained_rungs(checkpoint):
    config = read_config(checkpoint, IOConfig(), sources={})
    # A t2va student replaces the release's transformer partition.
    assert set(config.denoisers) == {"transformer"}
    denoiser = config.denoisers["transformer"]
    assert denoiser.grids == {
        "video": RungGrid((999, 749, 500, 250), shift=12.0, clock=1000.0),
        "audio": RungGrid((999, 749, 500, 250), shift=3.0, clock=1000.0),
    }
    assert denoiser.attention == SparseAttention(tile=64, sparsity=0.9)
    assert denoiser.tasks == ("t2va",)
    # The student generates its 12 (width, height) training buckets: 21:9,
    # 16:9, 4:3, 1:1, 3:4 and 9:16 at 768p, then at 480p.
    assert [(canvas.width, canvas.height) for canvas in denoiser.canvases] == [
        (1536, 672),
        (1344, 768),
        (1024, 768),
        (768, 768),
        (768, 1024),
        (768, 1344),
        (992, 416),
        (832, 480),
        (640, 480),
        (480, 480),
        (480, 640),
        (480, 832),
    ]
    # Single-segment sparse attention keeps the single-rounding epilogues.
    assert denoiser.transformer.rounding is Rounding.ONCE


def test_base_release_serves_both_task_families(base):
    config = read_config(base, IOConfig(), sources={})
    assert set(config.denoisers) == {"transformer", "transformer_ref"}
    for name, tasks in (
        ("transformer", ("t2va", "fl2va")),
        ("transformer_ref", ("ref2va",)),
    ):
        denoiser = config.denoisers[name]
        assert denoiser.tasks == tasks
        assert denoiser.attention == DenseAttention()
        assert denoiser.grids == {
            "video": UniformGrid(50, shift=12.0),
            "audio": UniformGrid(50, shift=3.0),
        }
        assert denoiser.canvases is None
        # The released DiTs follow the diffusers eager BF16 arithmetic.
        assert denoiser.transformer.rounding is Rounding.STEPWISE
    points = entry_points(config)
    assert {"transformer", "transformer_ref"} <= set(points)
    with torch.device("meta"):
        model = Model(config)
    assert model.transformer.num_steps == model.transformer_ref.num_steps == 49


def _ref2va_export(checkpoint, **fields):
    """Turn the fixture into a ref2va student of the release's partition."""
    contract = {
        "model_type": "ref2va",
        "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693",
        "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
        "grid_max_t": 0.999,
        "vsa_ref_keep_rate": 0.1,
        "vsa_sparsity": 0.9,
        "vsa_tile_size": 128,
        **fields,
    }
    (checkpoint / "fastvideo_inference.json").write_text(json.dumps(contract))
    shutil.move(checkpoint / "transformer", checkpoint / "transformer_ref")
    path = checkpoint / "transformer_ref/config.json"
    transformer = json.loads(path.read_text())
    path.write_text(json.dumps({**transformer, "pdd_steps": 32}))
    return checkpoint


def test_ref2va_fasth3_branches_from_the_reference_partition(checkpoint):
    export = _ref2va_export(checkpoint)
    layout = detect(export)
    assert dict(layout.partitions) == {"transformer_ref": ("ref2va",)}
    assert base_checkpoint(export) == ("MiniMaxAI/MiniMax-H3", "9bfb6693")

    config = read_config(export, IOConfig(), sources={})
    denoiser = config.denoisers["transformer_ref"]
    assert denoiser.transformer.output_heads == 32
    nodes = (0, 4, 8, 12, 16, 20, 24, 28, 32)
    assert denoiser.grids == {
        "video": BlockGrid(32, nodes, shift=12.0, max_t=0.999),
        "audio": BlockGrid(32, nodes, shift=3.0, max_t=0.999),
    }
    assert denoiser.attention == SparseAttention(
        tile=128, sparsity=0.9, reference_keep=0.1
    )
    assert denoiser.tasks == ("ref2va",)
    assert denoiser.canvases is None
    # Reference-segment sparse attention follows the reference's eager BF16
    # arithmetic.
    assert denoiser.transformer.rounding is Rounding.STEPWISE


def test_parallel_decoding_blocks_must_cover_the_heads(checkpoint):
    export = _ref2va_export(checkpoint, pdd_step_indices=[0, 4, 8, 16])
    with pytest.raises(ValueError, match="block grid"):
        read_config(export, IOConfig(), sources={})


def test_eight_step_checkpoint_owns_its_ladder_and_shifts(checkpoint):
    inference_path = checkpoint / "fastvideo_inference.json"
    inference = json.loads(inference_path.read_text())
    inference.update(
        {
            "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
            "vsa_sparsity": 0.8,
        }
    )
    inference_path.write_text(json.dumps(inference))
    scheduler_path = checkpoint / "scheduler/scheduler_config.json"
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["shift"] = 10.0
    scheduler_path.write_text(json.dumps(scheduler))

    config = read_config(checkpoint, IOConfig(), sources={})

    denoiser = config.denoisers["transformer"]
    rungs = (999, 874, 749, 624, 500, 375, 250, 125)
    assert denoiser.grids == {
        "video": RungGrid(rungs, shift=10, clock=1000.0),
        "audio": RungGrid(rungs, shift=3, clock=1000.0),
    }
    assert denoiser.attention.sparsity == 0.8
    with torch.device("meta"):
        model = Model(config)

    # The first and last evaluations follow the shifted timestep equation
    # sigma = s t / (1 + (s - 1) t) with t = step / 1000 and each modality's
    # trained shift; the clean endpoint follows the last evaluation.
    schedules = model.transformer.make_schedules(8, shift=None, device="cpu")
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
    with pytest.raises(ValueError, match="evaluates the network 8 times"):
        model.transformer.make_schedules(4, shift=None, device="cpu")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("task", "t2i", "task"),
        ("dmd_denoising_steps", [999, 999, 500, 250], "rungs"),
        ("dmd_denoising_steps", [1001, 749, 500, 250], "rungs"),
        ("vsa_sparsity", 1.0, "sparsity"),
        ("vsa_tile_size", 32, "64 or 128 rows"),
    ],
)
def test_a_contract_the_model_cannot_compute_is_rejected(
    checkpoint, field, value, error
):
    path = checkpoint / "fastvideo_inference.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=error):
        read_config(checkpoint, IOConfig(), sources={})


def test_a_contract_without_a_schedule_is_rejected(checkpoint):
    path = checkpoint / "fastvideo_inference.json"
    manifest = json.loads(path.read_text())
    del manifest["dmd_denoising_steps"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dmd_denoising_steps"):
        read_config(checkpoint, IOConfig(), sources={})


def test_architecture_without_variant_metadata_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="fastvideo_inference.json"):
        read_config(tmp_path, IOConfig(), sources={})


@pytest.mark.parametrize(
    "sidecar, field, value, error",
    [
        (
            "vae/config.json",
            "decoder_num_layers",
            35,
            "video_vae.decoder_num_layers",
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
        read_config(checkpoint, IOConfig(), sources={})


def test_h3_reader_reports_missing_decoder_field(checkpoint):
    path = checkpoint / "vae/config.json"
    values = json.loads(path.read_text())
    del values["latent_channels"]
    path.write_text(json.dumps(values))
    with pytest.raises(
        ValueError, match="video_vae is missing field latent_channels"
    ):
        read_config(checkpoint, IOConfig(), sources={})


def test_h3_direct_configuration_preserves_cross_component_dimensions():
    from dataclasses import replace

    from tests.python.fixtures.h3 import dmd_denoiser
    from uniserve_models.minimax_h3 import audio_vae, video_vae
    from uniserve_models.minimax_h3.encoder import TextEncoderConfig

    with pytest.raises(ValueError, match="conditioning width"):
        Config(
            text_encoder=replace(TextEncoderConfig(), hidden_size=4096),
            denoisers={"transformer": dmd_denoiser()},
            video_vae=video_vae.Config(),
            audio_vae=audio_vae.Config(),
        )
