"""FLUX checkpoint loading preserves an independent autoencoder's equations."""

import re

import pytest
import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.media import image
from uniserve.nn.vae.patch import PatchAutoencoder
from uniserve_models.bagel import vae

pytestmark = pytest.mark.integration


def _checkpoint(root):
    torch.manual_seed(451)
    reference = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(32, 64),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=32,
        sample_size=8,
        scaling_factor=0.5,
        shift_factor=0.25,
        use_quant_conv=False,
        use_post_quant_conv=False,
    ).eval()
    # The test compares both autoencoders in float64.
    reference.double()
    # Serialize the independently initialized model in the FLUX source format.
    tensors = {}
    for name, value in reference.state_dict().items():
        name = re.sub(
            r"down_blocks\.(\d+)\.resnets\.(\d+)", r"down.\1.block.\2", name
        )
        name = re.sub(
            r"down_blocks\.(\d+)\.downsamplers\.0", r"down.\1.downsample", name
        )
        name = re.sub(
            r"up_blocks\.(\d+)\.resnets\.(\d+)",
            lambda match: f"up.{1 - int(match[1])}.block.{match[2]}",
            name,
        )
        name = re.sub(
            r"up_blocks\.(\d+)\.upsamplers\.0",
            lambda match: f"up.{1 - int(match[1])}.upsample",
            name,
        )
        name = re.sub(
            r"mid_block\.resnets\.(\d+)",
            lambda match: f"mid.block_{int(match[1]) + 1}",
            name,
        )
        name = name.replace("conv_shortcut", "nin_shortcut").replace(
            "conv_norm_out", "norm_out"
        )
        if "mid_block.attentions.0" in name:
            name = name.replace("mid_block.attentions.0", "mid.attn_1")
            name = name.replace("group_norm", "norm").replace(
                "to_out.0", "proj_out"
            )
            for branch in ("q", "k", "v"):
                name = name.replace(f"to_{branch}", branch)
            if value.ndim == 2:
                value = value[:, :, None, None]
        tensors[name] = value.contiguous()
    save_file(tensors, root / "model.safetensors")
    return reference


def test_posterior_and_reconstruction_match_flux_checkpoint(tmp_path):
    reference = _checkpoint(tmp_path)
    config = vae.Config(8, 3, 2, 32, 3, (1, 2), 1, 4, 0.5, 0.25)
    model = loading.load_model(
        vae.Model,
        config,
        checkpoint=(
            checkpoint.Config("vae").resolve(tmp_path, io=loading.Config()),
        ),
        mapping=lambda model: (
            weights.ModuleMapping(
                model,
                "vae",
                lambda reader: vae.assignments(model, reader),
                frozenset(dict(model.named_parameters())),
            ),
        ),
        device="cpu",
        weights=weights.Config(dtype=torch.float64),
    ).model

    # Both autoencoders evaluate the same equations with different kernels
    # (a fused 1x1 QKV convolution here, separate linear projections in
    # diffusers), so their float32 results differ by accumulation-order
    # roundoff that depends on the platform's CPU kernels. Comparing in float64
    # keeps that roundoff far below the default float64 tolerances, so the
    # comparison checks the equations rather than the kernels.
    pixels = torch.randn(2, 3, 8, 12, dtype=torch.float64)
    with torch.no_grad():
        posterior = reference.encode(pixels).latent_dist
        expected = 0.5 * (
            posterior.sample(torch.Generator().manual_seed(8)) - 0.25
        )
        latents = model.encode(
            pixels, generator=torch.Generator().manual_seed(8)
        )
        torch.testing.assert_close(latents, expected)
        reconstructed = reference.decode(latents / 0.5 + 0.25).sample
        torch.testing.assert_close(model.decode(latents), reconstructed)
        torch.testing.assert_close(
            model(pixels, generator=torch.Generator().manual_seed(8)),
            reconstructed,
        )

        codec = PatchAutoencoder(
            model.encoder,
            model.decoder,
            model.posterior,
            patch_size=2,
            latent_channels=4,
            latent_dtype=torch.bfloat16,
            downsample=4,
            scale=0.5,
            shift=0.25,
        )
        patches = codec.encode(
            pixels, generator=torch.Generator().manual_seed(8)
        )
        restored = codec.unpatchify(patches, image.Config(8, 12))
        torch.testing.assert_close(restored, latents.bfloat16(), rtol=0, atol=0)
        expected_pixels = (
            reference.decode(restored.double() / 0.5 + 0.25).sample * 0.5 + 0.5
        ).clamp(0, 1)
        torch.testing.assert_close(
            codec.decode(patches, image.Config(8, 12)),
            expected_pixels,
        )
