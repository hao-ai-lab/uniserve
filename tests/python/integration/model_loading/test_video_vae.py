"""H3 video projections and temporal normalization through the public loader."""

from dataclasses import asdict

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    AutoencoderKLMiniMaxH3,
)
from safetensors.torch import save_file

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve_models.minimax_h3 import video_vae

pytestmark = pytest.mark.integration


def test_video_reconstruction_matches_independent_transformer_and_crop(
    tmp_path,
):
    config = video_vae.Config(
        latent_channels=2,
        block_out_channels=(8,),
        layers_per_block=1,
        spatial_downsample_factors=(2,),
        temporal_downsample_factors=(2,),
        norm_num_groups=4,
        decoder_num_layers=2,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=16,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
        clip_length=3,
        token_drop=1,
        latents_mean=(0.1, -0.2),
        latents_std=(0.5, 1.5),
    )
    torch.manual_seed(195)
    reference = AutoencoderKLMiniMaxH3(**asdict(config)).eval()
    # Exercise both residual branches and the learned suffix tokens.
    with torch.no_grad():
        reference.decoder.register_tokens.normal_(0, 0.02)
        for layer in reference.decoder.transformer_blocks:
            layer.scale1.uniform_(-0.2, 0.2)
            layer.scale2.uniform_(-0.2, 0.2)
    save_file(
        {
            name: value
            for name, value in reference.state_dict().items()
            if name.startswith(("decoder.", "post_quant_conv."))
        },
        tmp_path / "model.safetensors",
    )

    def mapping(model):
        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: video_vae.assignments(model, reader),
                frozenset(name for name, _ in model.named_parameters()),
            ),
        )

    model = loading.load_model(
        video_vae.Model,
        config,
        checkpoint=(
            checkpoint.Config().resolve(tmp_path, io=loading.Config()),
        ),
        mapping=mapping,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    ).model
    latents = torch.randn(1, 2, 3, 4, 6).bfloat16()
    normalized = latents.float() * torch.tensor(config.latents_std).view(
        1, 2, 1, 1, 1
    ) + torch.tensor(config.latents_mean).view(1, 2, 1, 1, 1)
    with torch.no_grad():
        expected = reference.decoder(reference.post_quant_conv(normalized))
        actual = model.decoder.decode(normalized, tiled=False)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        reconstructed = model(latents)
    assert reconstructed.shape == (1, 3, 5, 8, 12)
    torch.testing.assert_close(
        reconstructed, expected[:, :, 1:].half(), rtol=1e-3, atol=1e-3
    )
