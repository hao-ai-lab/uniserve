"""SenseNova's adaptive flow head and convolutional image reconstruction."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.diffusion import NoiseScale
from uniserve.nn.linear import Linear
from uniserve.nn.timestep import TimestepEmbedding


@dataclass(frozen=True)
class HeadConfig:
    hidden_size: int
    num_layers: int
    mlp_ratio: float

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1 for value in (self.hidden_size, self.num_layers)
        ):
            raise ValueError("flow head width and depth must be positive integers")
        if (
            not math.isfinite(self.mlp_ratio)
            or self.mlp_ratio <= 0
            or int(self.hidden_size * self.mlp_ratio) < 1
        ):
            raise ValueError("flow head expansion must produce a positive finite width")


@dataclass(frozen=True)
class Config:
    head: HeadConfig
    use_pixel_head: bool
    add_noise_scale_embedding: bool
    noise: NoiseScale

    def __post_init__(self):
        if (
            type(self.use_pixel_head) is not bool
            or type(self.add_noise_scale_embedding) is not bool
        ):
            raise ValueError("flow head selection and noise embedding must be boolean")


class _Residual(nn.Module):
    """Apply adaptive shift/scale and a gated residual MLP."""

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        width = int(config.hidden_size * config.mlp_ratio)
        self.mlp = nn.Sequential(
            Linear(config.hidden_size, width), nn.SiLU(), Linear(width, config.hidden_size)
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), Linear(config.hidden_size, 3 * config.hidden_size)
        )

    def forward(self, hidden, time):
        shift, scale, gate = self.modulation(time).chunk(3, dim=-1)
        return hidden + gate * self.mlp(self.norm(hidden) * (1 + scale) + shift)


class _Output(nn.Module):
    """Project to patch values after adaptive shift/scale normalization."""

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), Linear(hidden_size, 2 * hidden_size))
        self.projection = Linear(hidden_size, output_size)

    def forward(self, hidden, time):
        shift, scale = self.modulation(time).chunk(2, dim=-1)
        return self.projection(self.norm(hidden) * (1 + scale) + shift)


class Head(nn.Module):
    """Predict clean patch values using the checkpoint's adaptive time network."""

    def __init__(self, config: HeadConfig, *, input_size: int, output_size: int):
        super().__init__()
        self.config = config
        self.input = Linear(input_size, config.hidden_size)
        self.time_embedding = TimestepEmbedding(config.hidden_size)
        self.blocks = nn.ModuleList(_Residual(config) for _ in range(config.num_layers))
        self.output = _Output(config.hidden_size, output_size)

    def forward(self, hidden: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        hidden = self.input(hidden)
        time = self.time_embedding(timestep)
        for block in self.blocks:
            hidden = block(hidden, time)
        return self.output(hidden, time)


class Decoder(nn.Module):
    """Reconstruct image pixels with three channel-to-spatial rearrangements."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 1024,
        *,
        out_channels: int = 3,
        final_upscale: int = 8,
    ):
        super().__init__()
        if input_size % 4 or hidden_size % 4:
            raise ValueError(
                "pixel decoder widths must be divisible by the intermediate 2x2 shuffle"
            )
        self.blocks = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(input_size // 4, hidden_size, 3, padding=1),
            nn.GELU(),
            nn.PixelShuffle(2),
        )
        self.output = nn.Conv2d(hidden_size // 4, out_channels * final_upscale**2, 3, padding=1)
        self._final_upscale = final_upscale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.output(self.blocks(hidden)), self._final_upscale)


class Velocity(nn.Module):
    """Convert clean image predictions into velocity in canonical patch order.

    Conditioning pixels are the complete noisy image [1, 3, height, width].
    The predictor can be an adaptive Head, a shallow ordinary MLP, or Identity
    followed by the convolutional Decoder. The endpoint denominator is bounded
    by the checkpoint's 0.02 clean-fraction interval.
    """

    def __init__(self, head: nn.Module, decoder: nn.Module, *, patch_size: int):
        super().__init__()
        self.head, self.decoder, self.patch_size = head, decoder, patch_size

    def forward(self, hidden: torch.Tensor, timestep: torch.Tensor, patches) -> torch.Tensor:
        from uniserve.nn.functional import patchify

        pixels = patches.pixels
        if pixels.ndim != 4 or pixels.shape[:2] != (1, 3):
            raise ValueError("image conditioning requires one complete NCHW RGB image")
        height, width = pixels.shape[-2:]
        rows, columns = height // self.patch_size, width // self.patch_size
        if hidden.ndim != 2 or hidden.shape[0] != rows * columns:
            raise ValueError("flow hidden rows must cover the image patch grid")

        # hidden is [rows*columns, text_hidden]; predictions are per-patch
        # clean values in the same canonical patch order as the sample below.
        if isinstance(self.decoder, Decoder):
            spatial = (
                self.head(hidden).reshape(1, rows, columns, -1).permute(0, 3, 1, 2).contiguous()
            )
            prediction = patchify(self.decoder(spatial), patch_size=self.patch_size)[0]
        else:
            time = timestep.reshape(1).expand(hidden.shape[0])
            prediction = (
                self.head(hidden, time) if isinstance(self.head, Head) else self.head(hidden)
            )
            prediction = self.decoder(prediction)

        sample = patchify(pixels, patch_size=self.patch_size)[0]
        # A one-dimensional FP32 time retains the checkpoint's FP32 velocity
        # promotion after the BF16 clean-sample subtraction.
        return (prediction - sample) / (1 - timestep.reshape(1)).clamp_min(0.02)
