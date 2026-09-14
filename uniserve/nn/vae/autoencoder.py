"""Single-rank variational autoencoder layers for image latent encoding and decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from einops import rearrange
from torch import Tensor, nn

__all__ = [
    "AutoEncoderConfig",
    "FLUX_VAE_CONFIG",
    "AttnBlock",
    "ResnetBlock",
    "Downsample",
    "Upsample",
    "Encoder",
    "Decoder",
    "DiagonalGaussian",
    "AutoEncoder",
]

# GroupNorm group count used throughout the FLUX-VAE blocks (fixed by the
# checkpoint architecture).
_GN_GROUPS = 32
_GN_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class AutoEncoderConfig:
    """Defines VAE channel widths, residual depth, latent channels, scaling, and spatial-resolution policy."""

    resolution: int
    in_channels: int
    downsample: int
    ch: int
    out_ch: int
    ch_mult: tuple[int, ...]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float

    def __post_init__(self) -> None:
        for name in (
            "resolution",
            "in_channels",
            "downsample",
            "ch",
            "out_ch",
            "num_res_blocks",
            "z_channels",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"VAE {name} must be a positive integer")
        if (
            not isinstance(self.ch_mult, tuple)
            or not self.ch_mult
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in self.ch_mult
            )
        ):
            raise ValueError("VAE ch_mult must be a nonempty tuple of positive integers")
        if self.downsample != 2 ** (len(self.ch_mult) - 1):
            raise ValueError("VAE downsample must match its channel stages")
        if self.ch % _GN_GROUPS:
            raise ValueError("VAE channel width must be divisible by GroupNorm groups")
        for name in ("scale_factor", "shift_factor"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"VAE {name} must be finite")
        if self.scale_factor <= 0:
            raise ValueError("VAE scale_factor must be positive")


FLUX_VAE_CONFIG = AutoEncoderConfig(
    resolution=256,
    in_channels=3,
    downsample=8,
    ch=128,
    out_ch=3,
    ch_mult=(1, 2, 4, 4),
    num_res_blocks=2,
    z_channels=16,
    scale_factor=0.3611,
    shift_factor=0.1159,
)


# Query chunks bound the score-matrix workspace while each chunk still attends
# to every key/value row, preserving the full spatial-attention equation.
_ATTN_QUERY_CHUNK = 4096


class AttnBlock(nn.Module):
    """Applies normalized spatial self-attention within a VAE resolution stage."""

    def __init__(self, in_channels: int):
        """Build channel-preserving spatial QKV projections around group normalization."""

        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=_GN_GROUPS, num_channels=in_channels, eps=_GN_EPS, affine=True
        )
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        """Apply full spatial attention with bounded query-axis workspace."""

        h_ = self.norm(h_)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        # Channels form the head dimension, so SDPA's implicit scale is ``c**-0.5``.
        n_tokens = q.shape[-2]
        if n_tokens <= _ATTN_QUERY_CHUNK:
            h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            # Every query chunk retains the complete key/value context.
            chunks = [
                nn.functional.scaled_dot_product_attention(q_chunk, k, v)
                for q_chunk in torch.split(q, _ATTN_QUERY_CHUNK, dim=-2)
            ]
            h_ = torch.cat(chunks, dim=-2)
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        """Add a spatial self-attention update to a ``[batch, channels, height, width]`` tensor."""

        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """Applies a two-convolution residual transform with optional channel projection."""

    def __init__(self, in_channels: int, out_channels: int):
        """Build a two-convolution residual path with a channel-matching shortcut."""

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(
            num_groups=_GN_GROUPS, num_channels=in_channels, eps=_GN_EPS, affine=True
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(
            num_groups=_GN_GROUPS, num_channels=out_channels, eps=_GN_EPS, affine=True
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply two normalized convolutions and add the channel-aligned residual."""

        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = self.conv2(nn.functional.silu(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    """Halves spatial resolution with asymmetric padding and a strided convolution."""

    def __init__(self, in_channels: int):
        """Build the stride-two convolution used after asymmetric spatial padding."""

        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        """Halve spatial dimensions using asymmetric padding and stride-two convolution."""

        x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    """Doubles spatial resolution and refines features with a convolution."""

    def __init__(self, in_channels: int):
        """Build the convolution that refines nearest-neighbor upsampled features."""

        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        """Double spatial dimensions with nearest-neighbor expansion and convolution."""

        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class _EncoderLevel(nn.Module):
    """Groups the residual blocks and downsampling stage at one VAE encoder resolution."""

    def __init__(self, block: nn.ModuleList, downsample: Downsample | None) -> None:
        """Own one encoder resolution's residual blocks and optional downsampler."""

        super().__init__()
        self.block = block
        self.downsample = downsample


class _DecoderLevel(nn.Module):
    """Groups the residual blocks and upsampling stage at one VAE decoder resolution."""

    def __init__(self, block: nn.ModuleList, upsample: Upsample | None) -> None:
        """Own one decoder resolution's residual blocks and optional upsampler."""

        super().__init__()
        self.block = block
        self.upsample = upsample


class _MiddleBlocks(nn.Module):
    """Groups the residual and attention blocks at the VAE bottleneck."""

    def __init__(self, channels: int) -> None:
        """Build the residual-attention-residual sequence at the latent bottleneck."""

        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)


class Encoder(nn.Module):
    """Compresses RGB images into moments of a diagonal latent distribution."""

    def __init__(self, resolution, in_channels, ch, ch_mult, num_res_blocks, z_channels):
        """Build the multiresolution image encoder and moments projection."""

        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            downsample = Downsample(block_in) if i_level != self.num_resolutions - 1 else None
            self.down.append(_EncoderLevel(block, downsample))
        self.mid = _MiddleBlocks(block_in)
        self.norm_out = nn.GroupNorm(
            num_groups=_GN_GROUPS, num_channels=block_in, eps=_GN_EPS, affine=True
        )
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Encode image features into concatenated latent mean and log-variance channels."""

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            level = cast(_EncoderLevel, self.down[i_level])
            for i_block in range(self.num_res_blocks):
                h = level.block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                if level.downsample is None:
                    raise RuntimeError("encoder level is missing its downsample module")
                hs.append(level.downsample(hs[-1]))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(hs[-1])))
        return self.conv_out(nn.functional.silu(self.norm_out(h)))


class Decoder(nn.Module):
    """Reconstructs RGB images from sampled or deterministic latent tensors."""

    def __init__(self, ch, out_ch, ch_mult, num_res_blocks, in_channels, resolution, z_channels):
        """Build the latent bottleneck, multiresolution upsampler, and RGB projection."""

        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = _MiddleBlocks(block_in)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            upsample = Upsample(block_in) if i_level != 0 else None
            self.up.insert(0, _DecoderLevel(block, upsample))
        self.norm_out = nn.GroupNorm(
            num_groups=_GN_GROUPS, num_channels=block_in, eps=_GN_EPS, affine=True
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        """Decode latent feature maps through residual resolution stages into image channels."""

        h = self.conv_in(z)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for i_level in reversed(range(self.num_resolutions)):
            level = cast(_DecoderLevel, self.up[i_level])
            for i_block in range(self.num_res_blocks + 1):
                h = level.block[i_block](h)
            if i_level != 0:
                if level.upsample is None:
                    raise RuntimeError("decoder level is missing its upsample module")
                h = level.upsample(h)
        return self.conv_out(nn.functional.silu(self.norm_out(h)))


class DiagonalGaussian(nn.Module):
    """Splits encoded moments into mean and log variance for latent sampling."""

    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        """Configure the moment axis and deterministic-versus-sampled latent policy."""

        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Return the Gaussian mean or a generator-controlled reparameterized sample."""

        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            # Draw the reparameterization noise from the per-request generator when
            # supplied so sampling is reproducible; randn_like ignores any generator,
            # so build the noise explicitly to honor it.
            noise = torch.randn(
                mean.shape, generator=generator, device=mean.device, dtype=mean.dtype
            )
            return mean + torch.exp(0.5 * logvar) * noise
        return mean


class AutoEncoder(nn.Module):
    """Encodes images to scaled latent samples and decodes scaled latents to images."""

    def __init__(self, config: AutoEncoderConfig):
        """Build paired encoder and decoder towers with checkpoint latent scaling."""

        super().__init__()
        self.config = config
        self.encoder = Encoder(
            config.resolution,
            config.in_channels,
            config.ch,
            config.ch_mult,
            config.num_res_blocks,
            config.z_channels,
        )
        self.decoder = Decoder(
            config.ch,
            config.out_ch,
            config.ch_mult,
            config.num_res_blocks,
            config.in_channels,
            config.resolution,
            config.z_channels,
        )
        self.reg = DiagonalGaussian()
        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor

    def encode(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Sample and normalize image latents, optionally using a request generator."""

        z = self.reg(self.encoder(x), generator)
        return self.scale_factor * (z - self.shift_factor)

    def decode(self, z: Tensor) -> Tensor:
        """Undo latent normalization and reconstruct an image tensor."""

        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)
