from __future__ import annotations

import pytest
import torch

from uniserve_worker.models.minimax_h3.packing import audio_latent_frames
from uniserve_worker.models.minimax_h3.schedule import shifted_sigmas, solver_step

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("shift", "expected"),
    (
        (12.0, (1.0, 36.0 / 37.0, 12.0 / 13.0, 0.8, 0.0)),
        (3.0, (1.0, 0.9, 0.75, 0.5, 0.0)),
    ),
)
def test_fasth3_checkpoint_schedule(shift: float, expected: tuple[float, ...]) -> None:
    assert shifted_sigmas(shift) == pytest.approx(expected)


@pytest.mark.parametrize(("video_frames", "expected"), ((124, 207), (362, 603)))
def test_fasth3_audio_duration_geometry(video_frames: int, expected: int) -> None:
    assert audio_latent_frames(video_frames) == expected


def test_fasth3_solver_follows_clean_time_interval() -> None:
    sample = torch.tensor([2.0, -1.0], dtype=torch.float32)
    velocity = torch.tensor([0.5, 2.0], dtype=torch.float32)

    solver_step(
        sample,
        velocity,
        timestep=torch.tensor(0.25),
        sigma=torch.tensor(0.75),
        sigma_next=torch.tensor(0.5),
    )

    assert sample.tolist() == pytest.approx([2.125, -0.5])
