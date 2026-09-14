"""Checkpoint-exact resident MiniMax H3 audio decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from uniserve.nn.vae.decoder import LatentDecoder

__all__ = ["AudioDecoderConfig", "MiniMaxH3AudioVAE"]


@dataclass(frozen=True, slots=True)
class AudioDecoderConfig:
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
    sampling_rate: int = 32000
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
            "sampling_rate",
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


class MiniMaxH3AudioVAE(LatentDecoder):
    """Decodes H3 audio latents into bounded stereo PCM waveforms."""

    latent_shape = (2, 32, None)
    vae: Any
    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(self, vae: nn.Module, config: AudioDecoderConfig) -> None:
        """Compose an audio decoder with its configured latent normalization statistics."""

        super().__init__()
        self.config = config
        self.latent_shape = (2, config.latent_channels, None)
        self.vae = vae.float()
        if not hasattr(vae, "decode"):
            raise TypeError("MiniMax H3 audio VAE does not expose decode")
        mean, std = config.latents_mean, config.latents_std
        # Keep constant values when parameter storage is deferred. The public
        # loader stages graph buffers after materializing the learned modules.
        statistics_device = "cpu" if self.device.type == "meta" else self.device
        self.register_buffer(
            "latents_mean",
            torch.tensor(mean, dtype=torch.float32, device=statistics_device).view(
                1, config.latent_channels, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(std, dtype=torch.float32, device=statistics_device).view(
                1, config.latent_channels, 1
            ),
            persistent=False,
        )

    def _reconstruct(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert native decoder output to interleaved signed-16 stereo."""

        decoded = self.vae.decode(latents).sample.float()
        if decoded.ndim != 3 or decoded.shape[:2] != (2, 1):
            raise RuntimeError("MiniMax H3 audio decoder returned invalid stereo geometry")
        # PyAV accepts interleaved signed-16 stereo as [samples, channels].
        return (
            decoded[:, 0]
            .transpose(0, 1)
            .clamp_(-1.0, 1.0)
            .mul_(32767.0)
            .round_()
            .to(torch.int16)
            .contiguous()
        )
