"""H3's trained four-evaluation rungs retain their materialization order."""

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
    for name, shift in (("video", 12.0), ("audio", 3.0)):
        # Each rung is an unshifted noise level on the 1000-step clock,
        # shifted once per modality and materialized in double precision
        # before rounding to FP32; the clean endpoint follows the last rung.
        expected = tuple(
            shift * (rung / 1000) / (1 + (shift - 1) * (rung / 1000))
            for rung in (999, 749, 500, 250, 0)
        )
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
    with pytest.raises(ValueError, match="4 evaluations"):
        model.make_schedules(steps, shift=shift, device="cpu")


@pytest.mark.parametrize("video_frames,expected", [(124, 207), (362, 604)])
def test_audio_duration(video_frames, expected):
    assert audio_latent_frames(video_frames) == expected
