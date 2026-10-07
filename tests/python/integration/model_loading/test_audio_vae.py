"""H3 public audio reconstruction agrees with the native checkpoint decoder."""

from dataclasses import asdict

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
    AutoencoderKLMiniMaxH3Audio,
)
from safetensors.torch import save_file

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve_models.minimax_h3 import audio_vae
from uniserve_models.minimax_h3.encoding import AudioEncoder

pytestmark = pytest.mark.integration


def test_loaded_audio_statistics_and_interleaved_pcm_match_diffusers(tmp_path):
    config = audio_vae.Config(
        encoder_dim=8,
        encoder_rates=(2,),
        latent_dim=8,
        latent_channels=2,
        decoder_dim=16,
        decoder_rates=(2,),
        decoder_kernel_sizes=(4,),
        num_attention_heads=2,
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
        latents_mean=(0.1, -0.2),
        latents_std=(0.5, 1.5),
    )
    torch.manual_seed(975)
    reference = AutoencoderKLMiniMaxH3Audio(**asdict(config)).eval()
    save_file(
        {
            name: value
            for name, value in reference.state_dict().items()
            if name.startswith(("dec_in_proj.", "decoder."))
        },
        tmp_path / "model.safetensors",
    )

    def mapping(model):
        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: audio_vae.assignments(model, reader),
                frozenset(name for name, _ in model.named_parameters()),
            ),
        )

    model = loading.load_model(
        audio_vae.Model,
        config,
        checkpoint=(
            checkpoint.Config().resolve(tmp_path, io=loading.Config()),
        ),
        mapping=mapping,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    ).model
    latents = torch.randn(2, 2, 16).bfloat16()
    normalized = latents.float() * torch.tensor(config.latents_std).view(
        1, 2, 1
    ) + torch.tensor(config.latents_mean).view(1, 2, 1)
    with torch.no_grad():
        decoded = reference.decode(normalized).sample
        expected = (
            (decoded[:, 0].T.clamp(-1, 1) * 32767)
            .round()
            .to(torch.int16)
            .contiguous()
        )
        actual = model(latents)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (32, 2)


@pytest.mark.parametrize("channels", (1, 2))
def test_encoded_rows_match_native_stereo_posterior_mean(tmp_path, channels):
    config = audio_vae.Config(
        encoder_dim=4,
        encoder_rates=(2, 3),
        latent_dim=8,
        latent_channels=2,
        decoder_dim=8,
        decoder_rates=(2, 3),
        decoder_kernel_sizes=(4, 6),
        num_attention_heads=2,
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
        latents_mean=(0.1, -0.2),
        latents_std=(0.5, 1.5),
    )
    torch.manual_seed(976)
    reference = AutoencoderKLMiniMaxH3Audio(**asdict(config)).eval()
    # Perturb every encoder field, including the weight-norm magnitudes and
    # the attention biases the native model initializes to constants.
    with torch.no_grad():
        for name, value in reference.named_parameters():
            if not name.startswith(("decoder.", "dec_in_proj.")):
                value.add_(0.1 * torch.randn_like(value))
    save_file(
        {
            name: value
            for name, value in reference.state_dict().items()
            if not name.startswith(("decoder.", "dec_in_proj."))
        },
        tmp_path / "model.safetensors",
    )

    def mapping(model):
        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: audio_vae.encoder_assignments(
                    model.encoder, reader
                ),
                frozenset(name for name, _ in model.named_parameters()),
                nonresident=frozenset({"logs_proj.weight", "logs_proj.bias"}),
            ),
        )

    model = loading.load_model(
        lambda config: AudioEncoder(config, sample_rate=32000),
        config,
        checkpoint=(
            checkpoint.Config().resolve(tmp_path, io=loading.Config()),
        ),
        mapping=mapping,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    ).model

    # 40 samples are six whole latent frames of 6 samples and a partial one.
    generator = torch.Generator().manual_seed(977)
    track = torch.rand((40, channels), generator=generator) * 1.6 - 0.8
    waveform = track.T.expand(2, -1).contiguous()
    with torch.no_grad():
        posterior = reference.encode(waveform[:, None]).latent_dist
        latents = posterior.mode().transpose(1, 2)
        expected = (
            (latents - torch.tensor(config.latents_mean).view(1, 1, -1))
            / torch.tensor(config.latents_std).view(1, 1, -1)
        ).reshape(-1, config.latent_channels)
    (actual,) = model.encode((track,))
    assert actual.shape == (2 * 7, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
