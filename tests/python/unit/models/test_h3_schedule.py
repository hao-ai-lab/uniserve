"""H3's trained four-evaluation endpoints retain their materialization order."""

import pytest
import torch

from uniserve_models.minimax_h3.config import DiffusionConfig, TransformerConfig
from uniserve_models.minimax_h3.denoiser import Denoiser
from uniserve_models.minimax_h3.packing import audio_latent_frames

pytestmark = pytest.mark.unit


def test_fixed_modality_endpoints():
    with torch.device("meta"):
        model = Denoiser(TransformerConfig(), DiffusionConfig())
    schedules = model.make_schedules(4, shift=None, device="cpu")
    assert tuple(schedules) == ("video", "audio")
    for name, expected in (
        ("video", (1.0, 36.0 / 37.0, 12.0 / 13.0, 0.8, 0.0)),
        ("audio", (1.0, 0.9, 0.75, 0.5, 0.0)),
    ):
        sigma = torch.tensor(expected, dtype=torch.float32)
        torch.testing.assert_close(
            schedules[name].sigmas, sigma, rtol=0, atol=0
        )
        torch.testing.assert_close(
            schedules[name].timesteps, 1.0 - sigma, rtol=0, atol=0
        )
        assert schedules[name].coordinates == pytest.approx(
            tuple(1.0 - value for value in expected)
        )
        assert schedules[name].num_steps == 4


@pytest.mark.parametrize(
    "steps,shift", [(1, None), (2, None), (3, None), (5, None), (4, 1.0)]
)
def test_rejects_untrained_schedule(steps, shift):
    with torch.device("meta"):
        model = Denoiser(TransformerConfig(), DiffusionConfig())
    with pytest.raises(ValueError, match="four evaluations"):
        model.make_schedules(steps, shift=shift, device="cpu")


@pytest.mark.parametrize("video_frames,expected", [(124, 207), (362, 604)])
def test_audio_duration(video_frames, expected):
    assert audio_latent_frames(video_frames) == expected
