"""Reference video posteriors use complete clips and one seeded whole-video draw."""

import pytest
import torch

from uniserve_worker.models.minimax_h3.video_vae import H3VisualPosterior, MiniMaxH3VideoVAE


@pytest.fixture(scope="module")
def vae():
    posterior = H3VisualPosterior(parameter_device="cpu")
    with torch.no_grad():
        for parameter in posterior.parameters():
            parameter.zero_()
    return MiniMaxH3VideoVAE(posterior, linear_precision="fp32")


@pytest.mark.parametrize("frames,latent_frames", [(22, 7), (39, 12)])
def test_video_posterior_geometry_and_seed(vae, frames, latent_frames):
    pixels = torch.zeros((frames, 32, 32, 3), dtype=torch.uint8)
    result = vae.encode_video(pixels)
    assert result.shape == (1, 24, latent_frames, 2, 2)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    # Zero posterior weights mean mean=0, logvar=0. The released CPU seed and
    # FP16 round-trip therefore determine the first latent channel exactly;
    # its normalization constants are part of the pinned H3 checkpoint.
    noise = torch.randn(result.shape, generator=torch.Generator("cpu").manual_seed(42))
    expected = (noise[:, 0].half().float() - 0.858090341091156) / 1.2223774194717407
    torch.testing.assert_close(result[:, 0], expected, rtol=0, atol=0)


def test_incomplete_video_window_is_rejected(vae):
    with pytest.raises(ValueError, match="complete 17n\\+5"):
        vae.encode_video(torch.zeros((23, 32, 32, 3), dtype=torch.uint8))
