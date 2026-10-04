"""H3's schedules match their checkpoints' contracts."""

import pytest
import torch
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

from tests.python.fixtures.h3 import base_denoiser, dmd_denoiser
from uniserve_models.minimax_h3.denoiser import Denoiser
from uniserve_models.minimax_h3.packing import audio_latent_frames

pytestmark = pytest.mark.unit


def test_fixed_modality_endpoints():
    with torch.device("meta"):
        model = Denoiser(dmd_denoiser())
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
        model = Denoiser(dmd_denoiser())
    with pytest.raises(ValueError, match="4 evaluations"):
        model.make_schedules(steps, shift=shift, device="cpu")


def test_released_grid_matches_the_reference_scheduler():
    """The full-step schedule is the diffusers scheduler's, bit for bit."""
    with torch.device("meta"):
        model = Denoiser(base_denoiser())
    schedules = model.make_schedules(49, shift=None, device="cpu")
    for name, shift in (("video", 12.0), ("audio", 3.0)):
        reference = MiniMaxH3Scheduler(shift=shift)
        reference.set_timesteps(50, device="cpu")
        torch.testing.assert_close(
            schedules[name].sigmas, reference.sigmas, rtol=0, atol=0
        )
        torch.testing.assert_close(
            schedules[name].timesteps[:-1], reference.timesteps, rtol=0, atol=0
        )


# Nearest-integer audio latents per aligned frame count: 107 frames last
# 4.458 s, or 178.3 latents at 40 Hz; 362 frames last 15.083 s, or 603.3.
@pytest.mark.parametrize(
    "video_frames,expected", [(107, 178), (124, 207), (243, 405), (362, 603)]
)
def test_audio_duration(video_frames, expected):
    assert audio_latent_frames(video_frames) == expected


def test_audio_track_decodes_exactly_the_generated_latent_timeline():
    """The decoded track and the denoiser's audio rows describe one timeline.

    Every aligned frame count's track spans whole latent frames, the same
    count the denoiser generates, so the decoder never reads past the
    generated latents or leaves requested samples uncovered.
    """
    from uniserve_models.minimax_h3 import audio_vae
    from uniserve_models.minimax_h3.decoding import AudioDecoder

    with torch.device("meta"):
        decoder = AudioDecoder(audio_vae.Config(), sample_rate=32000)
    for frames in range(107, 363, 17):
        samples = decoder.track_samples(frames, 24)
        assert samples == audio_latent_frames(frames) * decoder.latent_rate
        assert decoder.latent_frames(samples) == audio_latent_frames(frames)
