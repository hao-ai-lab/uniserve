from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from uniserve_worker.models.minimax_h3.packing import audio_latent_frames
from uniserve_worker.models.minimax_h3.schedule import load_fasth3_schedule
from uniserve_worker.nn.diffusion.integrator import clean_sample_euler_step_

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("modality", "expected"),
    (
        (0, (1.0, 36.0 / 37.0, 12.0 / 13.0, 0.8, 0.0)),
        (1, (1.0, 0.9, 0.75, 0.5, 0.0)),
    ),
)
def test_fasth3_checkpoint_schedule(
    tmp_path: Path, modality: int, expected: tuple[float, ...]
) -> None:
    schedule = load_fasth3_schedule(tmp_path, torch.device("cpu"))
    torch.testing.assert_close(schedule.sigmas[modality], torch.tensor(expected))


@pytest.fixture
def dmd_checkpoint(tmp_path: Path) -> Path:
    contract = {
        "schema_version": "fasth3-inference-contract-v1",
        "task": "t2av",
        "transformer_forwards": 4,
        "num_inference_steps": 5,
        "dmd_denoising_steps": [999, 749, 500, 250],
        "guidance_scale": 1.0,
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "vsa_tile_size": 64,
        "vsa_sparsity": 0.9,
    }
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps(contract))
    return tmp_path


def test_fasth3_explicit_dmd_coordinates(dmd_checkpoint: Path) -> None:
    schedule = load_fasth3_schedule(dmd_checkpoint, torch.device("cpu"))
    for modality, expected in enumerate(
        (
            (11988.0 / 11989.0, 8988.0 / 9239.0, 12.0 / 13.0, 0.8, 0.0),
            (2997.0 / 2998.0, 2247.0 / 2498.0, 0.75, 0.5, 0.0),
        )
    ):
        torch.testing.assert_close(schedule.sigmas[modality], torch.tensor(expected))
        torch.testing.assert_close(
            schedule.timesteps[modality], 1.0 - schedule.sigmas[modality][:-1], rtol=0, atol=0
        )


@pytest.mark.parametrize("ladder", ([999, 749, 500], [999, 500, 749, 250], [999.0, 749, 500, 250]))
def test_fasth3_rejects_invalid_dmd_grid(dmd_checkpoint: Path, ladder: list[int | float]) -> None:
    path = dmd_checkpoint / "fastvideo_inference.json"
    contract = json.loads(path.read_text())
    contract["dmd_denoising_steps"] = ladder
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="DMD ladder"):
        load_fasth3_schedule(dmd_checkpoint, torch.device("cpu"))


@pytest.mark.parametrize(("video_frames", "expected"), ((124, 207), (362, 604)))
def test_fasth3_audio_duration_geometry(video_frames: int, expected: int) -> None:
    assert audio_latent_frames(video_frames) == expected


def test_fasth3_solver_follows_clean_time_interval() -> None:
    sample = torch.tensor([2.0, -1.0], dtype=torch.float32)
    velocity = torch.tensor([0.5, 2.0], dtype=torch.float32)

    clean_sample_euler_step_(
        sample,
        velocity,
        timestep=torch.tensor(0.25),
        sigma=torch.tensor(0.75),
        sigma_next=torch.tensor(0.5),
    )

    assert sample.tolist() == pytest.approx([2.125, -0.5])
