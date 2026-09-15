"""Checkpoint-exact resident MiniMax H3 audio decoder."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from uniserve.nn.vae.decoder import LatentDecoder

__all__ = ["Config", "Model"]


@dataclass(frozen=True, slots=True)
class Config:
    """Native audio network and channel normalization."""

    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    latent_dim: int = 2048
    latent_channels: int = 32
    decoder_dim: int = 1024
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    num_attention_heads: int = 8
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    latents_mean: tuple[float, ...] = (
        -0.020211687488382354,
        0.3876466479950502,
        -0.04398279799186767,
        -0.28591514936373,
        0.08179686214561671,
        -0.35782641352446604,
        0.040623809960919084,
        -0.01552534501956604,
        -0.223362481667332,
        0.1821006842509091,
        0.2941778783780663,
        -0.07901167601970885,
        -0.056815072777201,
        -0.3699028221860095,
        -0.31616315591624855,
        0.5905951377425391,
        -0.052139568068853864,
        0.013673160263486295,
        -0.03691647864630577,
        0.09732660653298163,
        -0.3394662328788498,
        -0.30685677538541667,
        -0.24504598907458763,
        -0.034698524462007344,
        0.02868032184767538,
        -0.21217779266454084,
        -0.1678263169941987,
        0.3221287889040614,
        -0.1223055851554907,
        0.4356604928128464,
        -0.0502599202236253,
        0.3979258376211797,
    )
    latents_std: tuple[float, ...] = (
        1.6895524230479284,
        2.76263727217653,
        1.7945344281264435,
        1.6801681847309828,
        1.6390226546605453,
        2.7788298348882177,
        1.7659090095747236,
        1.6199757612137327,
        2.6336525640336896,
        1.8539356672817833,
        2.5056497896915633,
        1.811019237886178,
        1.9579657790720237,
        1.6685498243529284,
        1.4922469314453364,
        3.298670198067373,
        1.9491804496832168,
        1.8720003270431442,
        1.8334080103291832,
        1.6488070416529093,
        1.6176957696319716,
        1.9131449234774398,
        1.5695245398428617,
        1.6943659940415912,
        1.8318420762504692,
        1.5540637421583379,
        1.9344930328968526,
        1.599198216109855,
        1.718045989838149,
        1.6307219190837705,
        1.8661226051202384,
        1.5613768203168363,
    )

    def __post_init__(self) -> None:
        for name in ("latents_mean", "latents_std"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != self.latent_channels:
                raise ValueError(f"H3 {name} must describe every latent channel")
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in values
            ):
                raise ValueError(f"H3 {name} must be finite")
        if any(value <= 0 for value in self.latents_std):
            raise ValueError("H3 latent standard deviations must be positive")

        for name in (
            "encoder_dim",
            "latent_dim",
            "latent_channels",
            "decoder_dim",
            "num_attention_heads",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"H3 audio {name} must be a positive integer")
        for name in (
            "encoder_rates",
            "decoder_rates",
            "decoder_kernel_sizes",
            "resblock_kernel_sizes",
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or not values
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in values
                )
            ):
                raise ValueError(f"H3 audio {name} must contain positive integers")

        if len(self.decoder_rates) != len(self.decoder_kernel_sizes):
            raise ValueError("H3 audio decoder rates and kernels must have matching stages")
        if not isinstance(self.resblock_dilation_sizes, tuple) or len(
            self.resblock_dilation_sizes
        ) != len(self.resblock_kernel_sizes):
            raise ValueError("H3 audio residual kernels and dilations must have matching stages")
        if any(
            not isinstance(row, tuple)
            or not row
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in row
            )
            for row in self.resblock_dilation_sizes
        ):
            raise ValueError("H3 audio residual dilations must contain positive integers")
        if self.latent_dim % self.num_attention_heads:
            raise ValueError("H3 audio latent width must be divisible by attention heads")


class Model(LatentDecoder):
    """Invert audio latent statistics and reconstruct interleaved stereo PCM."""

    def __init__(self, config: Config):
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
            MiniMaxH3AudioBigVGANDecoder,
        )

        decoder = nn.Sequential(
            OrderedDict(
                (
                    ("input", nn.Conv1d(config.latent_channels, config.latent_dim, 1)),
                    (
                        "network",
                        MiniMaxH3AudioBigVGANDecoder(
                            in_channels=config.latent_dim,
                            upsample_initial_channel=config.decoder_dim,
                            upsample_rates=config.decoder_rates,
                            upsample_kernel_sizes=config.decoder_kernel_sizes,
                            resblock_kernel_sizes=config.resblock_kernel_sizes,
                            resblock_dilation_sizes=config.resblock_dilation_sizes,
                        ),
                    ),
                )
            )
        )
        # The checkpoint supplies the exact anti-alias filters as numerical
        # weights. Register them as non-trainable Parameters so the ordinary
        # assignment path loads their values alongside the convolutions.
        for module in decoder.modules():
            for name, value in tuple(module.named_buffers(recurse=False)):
                if name not in module._non_persistent_buffers_set:
                    delattr(module, name)
                    module.register_parameter(name, nn.Parameter(value, requires_grad=False))
        super().__init__(
            decoder,
            latent_shape=(2, config.latent_channels, None),
            mean=torch.tensor(config.latents_mean, dtype=torch.float32, device="cpu").view(
                1, config.latent_channels, 1
            ),
            std=torch.tensor(config.latents_std, dtype=torch.float32, device="cpu").view(
                1, config.latent_channels, 1
            ),
        )
        self.config = config

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = super().forward(latents).float()
        if decoded.ndim != 3 or decoded.shape[:2] != (2, 1):
            raise RuntimeError("audio decoder must produce two mono channel timelines")
        # [stereo, samples] float waveforms become [samples, 2] int16 PCM.
        return (
            decoded[:, 0]
            .transpose(0, 1)
            .clamp(-1.0, 1.0)
            .mul(32767.0)
            .round()
            .to(torch.int16)
            .contiguous()
        )


def assignments(model: Model, reader):
    """Map native decoder fields without retaining the checkpoint encoder."""
    from uniserve.loading import weights

    available = frozenset(reader.names())
    values = []
    # Translate the module's sequential container names to native field names.
    for name, parameter in model.named_parameters():
        source = name.replace("decoder.input.", "dec_in_proj.").replace(
            "decoder.network.", "decoder."
        )
        if source in available:
            values.append(weights.Assignment(parameter, reader.get(source)))
    return tuple(values)
