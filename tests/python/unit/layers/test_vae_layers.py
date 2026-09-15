"""Behavior tests for shared latent normalization and posterior sampling."""

import pytest
import torch

from uniserve.nn.vae.layers import DiagonalGaussian

pytestmark = pytest.mark.unit


def test_latent_decoder_preserves_float32_normalization():
    from uniserve.nn.vae import LatentDecoder

    linear = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])))
    decoder = LatentDecoder(
        linear,
        latent_shape=(1, 3, 4),
        mean=torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1),
        std=torch.tensor([0.5, 1.5, 2.5]).view(1, 3, 1),
    )
    source = torch.linspace(-1, 1, 12).reshape(1, 3, 4).bfloat16()
    expected = (source.float() * decoder.std + decoder.mean) * torch.tensor(
        [1.0, 2.0, 3.0, 4.0]
    )
    torch.testing.assert_close(decoder(source), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="shape"):
        decoder(source[:, :, :3])


def test_diagonal_gaussian_disabled_returns_mean_and_halves_channels():
    """With sampling disabled the regularizer is the deterministic mean.

    The output is the first channel-chunk; the channel dim is halved.
    """
    torch.manual_seed(8)
    reg = DiagonalGaussian(sample=False)

    z = torch.randn(2, 8, 4, 4)
    mean, _logvar = torch.chunk(z, 2, dim=1)
    out = reg(z)
    assert out.shape == mean.shape
    torch.testing.assert_close(out, mean)
