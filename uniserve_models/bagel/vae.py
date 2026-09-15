"""BAGEL's FLUX image autoencoder architecture and checkpoint mapping."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.loading import checkpoint, weights
from uniserve.nn.vae.layers import (
    AttentionBlock,
    DiagonalGaussian,
    Downsample,
    ResidualBlock,
    Upsample,
)


@dataclass(frozen=True)
class Config:
    resolution: int
    in_channels: int
    downsample: int
    base_channels: int
    out_channels: int
    channel_multipliers: tuple[int, ...]
    num_res_blocks: int
    latent_channels: int
    scale_factor: float
    shift_factor: float

    def __post_init__(self):
        for name in (
            "resolution",
            "in_channels",
            "downsample",
            "base_channels",
            "out_channels",
            "num_res_blocks",
            "latent_channels",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"VAE {name} must be a positive integer")
        if (
            not isinstance(self.channel_multipliers, tuple)
            or not self.channel_multipliers
            or any(type(value) is not int or value < 1 for value in self.channel_multipliers)
        ):
            raise ValueError("VAE channel multipliers must be a nonempty positive integer tuple")
        if self.downsample != 2 ** (len(self.channel_multipliers) - 1):
            raise ValueError("VAE downsample must match its resolution stages")
        if self.base_channels % 32:
            raise ValueError("VAE base channels must be divisible by 32 GroupNorm groups")
        if (
            not math.isfinite(self.scale_factor)
            or self.scale_factor <= 0
            or not math.isfinite(self.shift_factor)
        ):
            raise ValueError("VAE scale must be finite and positive, and shift must be finite")


class Encoder(nn.Module):
    """Compress NCHW pixels to concatenated posterior mean/log-variance."""

    def __init__(self, config: Config):
        super().__init__()
        channels = config.base_channels
        self.input = nn.Conv2d(config.in_channels, channels, 3, padding=1)
        self.levels = nn.ModuleList()
        for index, multiplier in enumerate(config.channel_multipliers):
            width = config.base_channels * multiplier
            blocks = []
            for _ in range(config.num_res_blocks):
                blocks.append(ResidualBlock(channels, width))
                channels = width
            if index + 1 < len(config.channel_multipliers):
                blocks.append(Downsample(channels))
            self.levels.append(nn.Sequential(*blocks))
        self.middle = nn.Sequential(
            ResidualBlock(channels, channels),
            AttentionBlock(channels),
            ResidualBlock(channels, channels),
        )
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.output = nn.Conv2d(channels, 2 * config.latent_channels, 3, padding=1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        hidden = self.input(pixels)
        for level in self.levels:
            hidden = level(hidden)
        return self.output(F.silu(self.norm(self.middle(hidden))))


class Decoder(nn.Module):
    """Reconstruct NCHW pixels from unnormalized spatial latents."""

    def __init__(self, config: Config):
        super().__init__()
        channels = config.base_channels * config.channel_multipliers[-1]
        self.input = nn.Conv2d(config.latent_channels, channels, 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(channels, channels),
            AttentionBlock(channels),
            ResidualBlock(channels, channels),
        )
        self.levels = nn.ModuleList()
        for index in reversed(range(len(config.channel_multipliers))):
            width = config.base_channels * config.channel_multipliers[index]
            blocks = []
            for _ in range(config.num_res_blocks + 1):
                blocks.append(ResidualBlock(channels, width))
                channels = width
            if index:
                blocks.append(Upsample(channels))
            self.levels.append(nn.Sequential(*blocks))
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.output = nn.Conv2d(channels, config.out_channels, 3, padding=1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.middle(self.input(latents))
        for level in self.levels:
            hidden = level(hidden)
        return self.output(F.silu(self.norm(hidden)))


class Model(nn.Module):
    """Encode normalized posterior samples and decode their inverse transform."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.posterior = DiagonalGaussian()
        self.decoder = Decoder(config)

    def encode(
        self, pixels: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        latent = self.posterior(self.encoder(pixels), generator=generator)
        return self.config.scale_factor * (latent - self.config.shift_factor)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents / self.config.scale_factor + self.config.shift_factor)

    def forward(
        self, pixels: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return self.decode(self.encode(pixels, generator=generator))


def assignments(module: nn.Module, reader: checkpoint.Reader) -> tuple[weights.Assignment, ...]:
    """Map FLUX tensors into encoder/decoder composition, packing spatial QKV.

    Both Model and PatchAutoencoder expose the same encoder/decoder modules.
    Checkpoint decoder levels count from the pixel end; module levels execute
    from the latent end. Q/K/V fragments cover disjoint output channels.
    """
    result = []
    available = frozenset(reader.names())
    for tower_name in ("encoder", "decoder"):
        tower = getattr(module, tower_name)
        parameters = dict(tower.named_parameters())
        names = {"input": "conv_in", "norm": "norm_out", "output": "conv_out"}
        for name, layer in tower.named_modules():
            if isinstance(layer, ResidualBlock):
                if name.startswith("middle."):
                    source = f"mid.block_{1 if name == 'middle.0' else 2}"
                else:
                    _, level, block = name.split(".")
                    index = (
                        int(level)
                        if tower_name == "encoder"
                        else len(tower.levels) - 1 - int(level)
                    )
                    source = f"{'down' if tower_name == 'encoder' else 'up'}.{index}.block.{block}"
                for target, suffix in (
                    ("norms.0", "norm1"),
                    ("norms.1", "norm2"),
                    ("convolutions.0", "conv1"),
                    ("convolutions.1", "conv2"),
                    ("shortcut", "nin_shortcut"),
                ):
                    names[f"{name}.{target}"] = f"{source}.{suffix}"
            elif isinstance(layer, (Downsample, Upsample)):
                index = int(name.split(".")[1])
                if tower_name == "decoder":
                    index = len(tower.levels) - 1 - index
                direction = "down" if tower_name == "encoder" else "up"
                names[f"{name}.convolution"] = f"{direction}.{index}.{direction}sample.conv"
            elif isinstance(layer, AttentionBlock):
                names[f"{name}.norm"] = "mid.attn_1.norm"
                names[f"{name}.output"] = "mid.attn_1.proj_out"
                for field in ("weight", "bias"):
                    target = parameters[f"{name}.qkv.{field}"]
                    channels = target.shape[0] // 3
                    for index, branch in enumerate(("q", "k", "v")):
                        source_name = f"{tower_name}.mid.attn_1.{branch}.{field}"
                        if source_name in available:
                            source = reader.get(source_name)
                            region = (slice(index * channels, (index + 1) * channels),) + tuple(
                                slice(0, size) for size in target.shape[1:]
                            )
                            result.append(weights.Assignment(target, source, target_slice=region))
        for name, target in parameters.items():
            parent, field = name.rsplit(".", 1)
            if parent in names:
                source_name = f"{tower_name}.{names[parent]}.{field}"
                if source_name in available:
                    source = reader.get(source_name)
                    result.append(weights.Assignment(target, source))
    return tuple(result)
