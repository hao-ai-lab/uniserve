"""Reference audio normalization at the third-party audio-VAE boundary."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from uniserve_worker.models.minimax_h3.audio_vae import MiniMaxH3AudioVAE

pytestmark = pytest.mark.unit


class AudioVAE(nn.Module):
    """Small external VAE double with independent mono posterior channels."""

    def __init__(self, *, invalid=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(latents_mean=[2.0] * 32, latents_std=[4.0] * 32)
        self.invalid = invalid

    def encode(self, samples):
        latents = samples[:, :, ::800].expand(-1, 32, -1)
        if self.invalid:
            latents = latents[:1]
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latents))

    def decode(self, latents):
        return SimpleNamespace(sample=latents[:, :1].repeat_interleave(800, dim=-1))


def test_encode_preserves_stereo_time_order_and_normalizes_posterior_mode():
    model = MiniMaxH3AudioVAE(AudioVAE())
    waveform = torch.stack((torch.linspace(-1, 1, 2400), torch.linspace(1, -1, 2400)))
    encoded = model.encode(waveform)
    expected = ((waveform[:, ::800] - 2) / 4)[:, None].expand(2, 32, 3)
    torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
    assert encoded.dtype == torch.float32
    assert not encoded.requires_grad


@pytest.mark.parametrize(
    "waveform",
    [
        torch.zeros(1, 800),
        torch.zeros(2, 0),
        torch.zeros(2, 1, 800),
        torch.zeros(2, 800, dtype=torch.int16),
    ],
)
def test_encode_rejects_unprepared_waveforms(waveform):
    with pytest.raises(ValueError):
        MiniMaxH3AudioVAE(AudioVAE()).encode(waveform)


def test_encode_rejects_invalid_posterior_geometry():
    with pytest.raises(RuntimeError, match="stereo geometry"):
        MiniMaxH3AudioVAE(AudioVAE(invalid=True)).encode(torch.zeros(2, 800))
