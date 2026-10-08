"""Behavior tests for shared latent normalization and posterior sampling."""

import pytest
import torch

from uniserve.nn.vae.layers import DiagonalGaussian

pytestmark = pytest.mark.unit


def test_latent_decoder_preserves_float32_normalization():
    from uniserve.nn.vae import ChannelStatistics, LatentDecoder

    linear = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])))
    decoder = LatentDecoder(
        linear,
        latent_shape=(1, 3, 4),
        normalization=ChannelStatistics(
            mean=torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1),
            std=torch.tensor([0.5, 1.5, 2.5]).view(1, 3, 1),
        ),
    )
    source = torch.linspace(-1, 1, 12).reshape(1, 3, 4).bfloat16()
    expected = (
        source.float() * decoder.normalization.std + decoder.normalization.mean
    ) * torch.tensor([1.0, 2.0, 3.0, 4.0])
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


def test_diagonal_gaussian_scales_supplied_noise_by_clamped_deviation():
    posterior = DiagonalGaussian(log_variance_range=(-30.0, 20.0))
    mean = torch.tensor([0.5, -1.0, 2.0]).view(1, 3, 1)
    log_variance = torch.tensor([-2.0, 40.0, -50.0]).view(1, 3, 1)
    noise = torch.tensor([1.5, -0.5, 2.0]).view(1, 3, 1)

    sample = posterior(torch.cat((mean, log_variance), dim=1), noise=noise)

    bounded = torch.tensor([-2.0, 20.0, -30.0]).view(1, 3, 1)
    expected = mean + torch.exp(0.5 * bounded) * noise
    torch.testing.assert_close(sample, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="shape"):
        posterior(torch.cat((mean, log_variance), dim=1), noise=noise[:, :2])


def test_spatial_encoder_tiles_reproduce_a_local_encoding():
    """Overlapping tiles of a blockwise encoder assemble its untiled output.

    Every latent of an average pool reads only its own pixel block, so tiles
    placed at their raster positions agree on every overlap and the blended
    seams reproduce the whole-raster encoding.
    """
    from uniserve.nn.vae import SpatialEncoder

    encoder = SpatialEncoder(
        torch.nn.AvgPool2d(4),
        spatial_compression=4,
        tile_height=32,
        tile_width=16,
        overlap_height=8,
        overlap_width=4,
    )
    pixels = torch.randn(2, 3, 72, 60)

    tiled = encoder(pixels)

    assert tiled.shape == (2, 3, 18, 15)
    torch.testing.assert_close(tiled, encoder.encode_tile(pixels))


def test_spatial_encoder_bands_are_rows_of_the_whole_encoding():
    """Every band of latent rows equals those rows of the whole encoding.

    A padded convolution reads across tile edges, so tiles disagree on their
    overlaps and each band's seams must blend exactly as the whole raster's
    do, including bands that start inside a cross-fade.
    """
    from uniserve.nn.vae import SpatialEncoder

    torch.manual_seed(0)
    encoder = SpatialEncoder(
        torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.AvgPool2d(4)
        ),
        spatial_compression=4,
        tile_height=32,
        tile_width=16,
        overlap_height=8,
        overlap_width=4,
    )
    pixels = torch.randn(2, 3, 76, 40)

    with torch.inference_mode():
        whole = encoder(pixels)
        for start in range(whole.shape[-2]):
            for stop in range(start + 1, whole.shape[-2] + 1):
                band = encoder(pixels, rows=slice(start, stop))
                assert torch.equal(band, whole[..., start:stop, :])

    with pytest.raises(ValueError, match="latent rows"):
        encoder(pixels, rows=slice(4, 4))
    with pytest.raises(ValueError, match="latent rows"):
        encoder(pixels, rows=slice(0, whole.shape[-2] + 1))
