"""Conformance for the shared VAE layer."""
from __future__ import annotations

import pytest
import torch

from uniserve_worker.nn.vae import AutoEncoder, AutoEncoderParams
from uniserve_worker.nn.vae.autoencoder import AttnBlock, DiagonalGaussian

pytestmark = pytest.mark.unit


def _tiny_params():
    return AutoEncoderParams(
        resolution=8,
        in_channels=3,
        downsample=4,
        ch=32,
        out_ch=3,
        ch_mult=[1, 1],
        num_res_blocks=1,
        z_channels=4,
        scale_factor=0.5,
        shift_factor=0.25,
    )


def test_autoencoder_is_shared_vae_layer():
    assert issubclass(AutoEncoder, torch.nn.Module)


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


def test_encode_applies_scale_shift_transform():
    """encode() must apply scale_factor * (z - shift_factor) on top of the raw
    encoder output; this is the numeric contract production relies on."""
    torch.manual_seed(8)
    shared = AutoEncoder(_tiny_params())
    shared.reg.sample = False

    x = torch.randn(1, 3, 8, 8)
    raw = shared.reg(shared.encoder(x))
    expected = shared.scale_factor * (raw - shared.shift_factor)
    torch.testing.assert_close(shared.encode(x), expected)


def test_decode_inverts_scale_shift_transform():
    """decode() must undo the scale/shift before running the decoder, so that
    decode(z) == decoder(z / scale_factor + shift_factor)."""
    torch.manual_seed(8)
    shared = AutoEncoder(_tiny_params())
    shared.reg.sample = False

    x = torch.randn(1, 3, 8, 8)
    z = shared.encode(x)
    expected = shared.decoder(z / shared.scale_factor + shared.shift_factor)
    torch.testing.assert_close(shared.decode(z), expected)


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
