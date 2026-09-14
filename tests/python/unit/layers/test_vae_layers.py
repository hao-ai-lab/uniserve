"""Behavior tests for the shared VAE layer."""

from __future__ import annotations

import pytest
import torch

from uniserve.nn.vae import AutoEncoder, AutoEncoderConfig
from uniserve.nn.vae.autoencoder import AttnBlock, DiagonalGaussian
from uniserve.nn.vae.patch import PatchAutoencoder
from uniserve.nn.vision.patching import unpatchify_batch

pytestmark = pytest.mark.unit


def test_latent_decoder_preserves_float32_normalization_and_scope_restoration():
    from tests.python.fixtures.decoding import ChannelDecoder
    from uniserve.nn.vae.decoder import decoder_scope

    decoder = ChannelDecoder()
    source = torch.linspace(-1, 1, 12).reshape(1, 3, 4).bfloat16()
    expected = source.float() * torch.tensor([0.5, 1.5, 2.5]).view(1, 3, 1)
    expected += torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1)
    expected *= torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = expected.unsqueeze(-1).unsqueeze(-1)
    torch.testing.assert_close(decoder(source), expected, rtol=0, atol=0)
    with decoder_scope({}), pytest.raises(RuntimeError, match="missing"):
        decoder(source)
    torch.testing.assert_close(decoder(source), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="shape"):
        decoder(source[:, :, :3])


def _tiny_params():
    return AutoEncoderConfig(
        resolution=8,
        in_channels=3,
        downsample=2,
        ch=32,
        out_ch=3,
        ch_mult=(1, 1),
        num_res_blocks=1,
        z_channels=4,
        scale_factor=0.5,
        shift_factor=0.25,
    )


def test_autoencoder_encode_decode_is_deterministic_with_sampling_disabled():
    torch.manual_seed(8)
    params = _tiny_params()
    shared = AutoEncoder(params)
    shared.reg.sample = False

    x = torch.randn(1, 3, 8, 8)
    z1 = shared.encode(x)
    z2 = shared.encode(x)
    torch.testing.assert_close(z1, z2)
    out = shared.decode(z1)
    assert out.shape == x.shape


def test_encode_sampling_is_stochastic_but_seed_reproducible():
    """With sampling on, encode() draws noise: distinct RNG states give distinct
    latents, but a fixed seed is reproducible."""
    torch.manual_seed(8)
    shared = AutoEncoder(_tiny_params())
    shared.reg.sample = True

    x = torch.randn(1, 3, 8, 8)
    torch.manual_seed(0)
    z_a = shared.encode(x)
    torch.manual_seed(1)
    z_b = shared.encode(x)
    # Different RNG state must perturb the latent (the sampling branch is live).
    assert not torch.allclose(z_a, z_b)

    torch.manual_seed(0)
    z_a2 = shared.encode(x)
    torch.testing.assert_close(z_a, z_a2)


def test_patch_autoencoder_preserves_posterior_and_reconstruction():
    """Patch representation preserves the seeded posterior and native VAE math."""

    torch.manual_seed(8)
    native = AutoEncoder(_tiny_params())
    codec = PatchAutoencoder(
        native, patch_size=2, downsample=4, channels=4, latent_dtype=torch.bfloat16
    )
    pixels = torch.randn(2, 3, 8, 12)
    generator = torch.Generator().manual_seed(31)
    posterior = native.encode(pixels, generator)
    patches = codec.encode(pixels, torch.Generator().manual_seed(31))
    restored = unpatchify_batch(patches, 2, height=4, width=6, channels=4)
    torch.testing.assert_close(restored, posterior.to(torch.bfloat16), rtol=0, atol=0)

    expected = (native.decode(restored.float()) * 0.5 + 0.5).clamp(0, 1)
    actual = codec.decode(patches, 8, 12)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_diagonal_gaussian_disabled_returns_mean_and_halves_channels():
    """With sampling disabled the regularizer is the deterministic mean of the
    first channel-chunk; the channel dim is halved."""
    torch.manual_seed(8)
    reg = DiagonalGaussian(sample=False)

    z = torch.randn(2, 8, 4, 4)
    mean, _logvar = torch.chunk(z, 2, dim=1)
    out = reg(z)
    assert out.shape == mean.shape
    torch.testing.assert_close(out, mean)


def test_packed_attention_block_is_shape_preserving_residual():
    """The packed scaled-dot-product attention block keeps spatial shape and is a
    pure residual: zeroing proj_out leaves the input unchanged."""
    torch.manual_seed(8)
    # GroupNorm in AttnBlock hardcodes num_groups=32, so channels must be a
    # multiple of 32.
    attn = AttnBlock(32).eval()

    h = torch.randn(1, 32, 5, 7)
    with torch.no_grad():
        out = attn(h)
    assert out.shape == h.shape

    torch.nn.init.zeros_(attn.proj_out.weight)
    torch.nn.init.zeros_(attn.proj_out.bias)
    with torch.no_grad():
        identity = attn(h)
    torch.testing.assert_close(identity, h)
