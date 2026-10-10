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
from uniserve_models.minimax_h3.checkpoint import Kind, detect
from uniserve_models.minimax_h3.config import Config, normalize, read_config

pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint(tmp_path):
    source = Path(__file__).parents[2] / "fixtures/models/fasth3"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    return tmp_path


@pytest.fixture
def base(checkpoint):
    """A diffusers root holding both DiT partitions and no contract."""
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


def test_full_vsa_checkpoint_resolves_its_trained_rungs(checkpoint):
    config = read_config(checkpoint, IOConfig(), sources={})
    assert set(config.denoisers) == {"denoiser"}
    denoiser = config.denoisers["denoiser"]
    assert denoiser.grids == {
        "video": RungGrid((999, 749, 500, 250), shift=12.0, clock=1000.0),
        "audio": RungGrid((999, 749, 500, 250), shift=3.0, clock=1000.0),
    }
    assert denoiser.attention == SparseAttention(tile=64, sparsity=0.9)
    assert denoiser.tasks == ("t2va",)
    # The export generates its 12 (width, height) training buckets: 21:9,
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
    # FastH3 exports keep the single-rounding block epilogues.
    assert denoiser.transformer.rounding is Rounding.ONCE


def test_diffusers_root_serves_both_task_families(base):
    assert detect(base).kind is Kind.DIFFUSERS_ROOT
    config = read_config(base, IOConfig(), sources={})
    assert set(config.denoisers) == {"denoiser", "reference_denoiser"}
    for name, tasks in (
        ("denoiser", ("t2va", "fl2va")),
        ("reference_denoiser", ("ref2va",)),
    ):
        denoiser = config.denoisers[name]
        assert denoiser.tasks == tasks
        assert denoiser.attention == DenseAttention()
        assert denoiser.grids == {
            "video": UniformGrid(50, shift=12.0),
            "audio": UniformGrid(50, shift=3.0),
        }
        assert denoiser.canvases is None
        assert denoiser.max_sequence_rows is None
        # The released DiTs follow the diffusers eager BF16 arithmetic.
        assert denoiser.transformer.rounding is Rounding.STEPWISE
    points = entry_points(config)
    assert {"denoiser", "reference_denoiser"} <= set(points)
    with torch.device("meta"):
        model = Model(config)
    assert model.denoiser.num_steps == model.reference_denoiser.num_steps == 49


def test_component_export_states_its_parallel_decoding_contract(checkpoint):
    contract = {
        "schema_version": "fasth3-inference-contract-v1",
        "model_type": "ref2va",
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "conditioning": "fixed_ordered_references_target_only_flow",
        "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693",
        "transformer_component": "transformer_ref",
        "guidance_scale": 1.0,
        "pdd_steps": 32,
        "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
        "transformer_forwards": 8,
        "num_inference_steps": 8,
        "grid_max_t": 0.999,
        "video_scheduler_shift": 12.0,
        "audio_scheduler_shift": 3.0,
        "vsa_ref_policy": "p2_multi_region",
        "vsa_ref_keep_rate": 0.1,
        "vsa_sparsity": 0.9,
        "vsa_tile_size": 128,
    }
    (checkpoint / "fastvideo_inference.json").write_text(json.dumps(contract))
    shutil.move(checkpoint / "transformer", checkpoint / "transformer_ref")
    layout = detect(checkpoint)
    assert layout.kind is Kind.COMPONENT_EXPORT
    assert dict(layout.denoisers) == {"reference_denoiser": "transformer_ref"}
    assert (layout.base.repository, layout.base.revision) == (
        "MiniMaxAI/MiniMax-H3",
        "9bfb6693",
    )
    metadata = {
        name: json.loads((checkpoint / relative).read_text())
        for name, relative in (
            ("text_encoder", "text_encoder/config.json"),
            ("video_vae", "vae/config.json"),
            ("audio_vae", "audio_vae/config.json"),
            ("scheduler", "scheduler/scheduler_config.json"),
            ("audio_scheduler", "audio_scheduler/scheduler_config.json"),
        )
    }
    transformer = json.loads(
        (checkpoint / "transformer_ref/config.json").read_text()
    )
    metadata["reference_denoiser"] = {**transformer, "pdd_steps": 32}
    config = normalize(layout, metadata)
    denoiser = config.denoisers["reference_denoiser"]
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
    # Region selection's 8192 tiles of 128 rows, not a checkpoint bound.
    assert denoiser.max_sequence_rows == 1_048_576
    # Component exports follow their reference's eager BF16 arithmetic.
    assert denoiser.transformer.rounding is Rounding.STEPWISE
    # A student whose heads disagree with its contract is rejected.
    metadata["reference_denoiser"] = {**transformer, "pdd_steps": 16}
    with pytest.raises(ValueError, match="pdd_steps"):
        normalize(layout, metadata)


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

    config = read_config(checkpoint, IOConfig(), sources={})

    denoiser = config.denoisers["denoiser"]
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
    with pytest.raises(ValueError, match="evaluates the network 8 times"):
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
            denoisers={"denoiser": dmd_denoiser()},
            video_vae=video_vae.Config(),
            audio_vae=audio_vae.Config(),
        )
