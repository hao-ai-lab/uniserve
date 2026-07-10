"""Single-rank VAE autoencoder layer.

Numerics match the current production autoencoder path; the implementation lives in
the shared layer library so commit and final image decode use one code path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from einops import rearrange
from torch import Tensor, nn

__all__ = [
    'AutoEncoderParams',
    'FLUX_VAE_PARAMS',
    'default_ae_params',
    'AttnBlock',
    'ResnetBlock',
    'Downsample',
    'Upsample',
    'Encoder',
    'Decoder',
    'DiagonalGaussian',
    'AutoEncoder',
]

# GroupNorm group count used throughout the FLUX-VAE blocks (fixed by the
# checkpoint architecture).
_GN_GROUPS = 32
_GN_EPS = 1e-6


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    downsample: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


# Canonical FLUX-VAE geometry. This is the single named source for the default
# autoencoder configuration; callers that have a checkpoint ``vae`` config should
# build :class:`AutoEncoderParams` from it and fall back to this constant. Kept
# primarily for tests and the model-neutral default path.
def _flux_vae_params() -> AutoEncoderParams:
    return AutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )


FLUX_VAE_PARAMS = _flux_vae_params()


def default_ae_params() -> AutoEncoderParams:
    # Return a fresh instance so callers can mutate without aliasing the constant.
    return _flux_vae_params()


# Largest number of query tokens attended in a single SDPA call. Above this the
# query dimension is processed in chunks so a fallback (math) SDPA backend cannot
# materialize the full (h*w, h*w) score matrix and OOM at high resolution. Each
# query row attends independently to all keys, so chunking over queries is exact.
_ATTN_QUERY_CHUNK = 4096


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=_GN_GROUPS, num_channels=in_channels, eps=_GN_EPS, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        # Channels are the per-token feature here, so the head dimension is c and
        # SDPA's implicit 1/sqrt(c) scale is the intended one (matches the original
        # einsum attention's c**-0.5 scaling); no explicit scale override is needed.
        n_tokens = q.shape[-2]
        if n_tokens <= _ATTN_QUERY_CHUNK:
            h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            # Tile over the query axis; each chunk attends to all keys/values, so the
            # concatenation is numerically identical to the single-call result.
            chunks = [
                nn.functional.scaled_dot_product_attention(q_chunk, k, v)
                for q_chunk in torch.split(q, _ATTN_QUERY_CHUNK, dim=-2)
            ]
            h_ = torch.cat(chunks, dim=-2)
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=_GN_GROUPS, num_channels=in_channels, eps=_GN_EPS, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=_GN_GROUPS, num_channels=out_channels, eps=_GN_EPS, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = self.conv2(nn.functional.silu(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class _EncoderLevel(nn.Module):
    def __init__(self, block: nn.ModuleList, downsample: Downsample | None) -> None:
        super().__init__()
        self.block = block
        self.downsample = downsample


class _DecoderLevel(nn.Module):
    def __init__(self, block: nn.ModuleList, upsample: Upsample | None) -> None:
        super().__init__()
        self.block = block
        self.upsample = upsample


class _MiddleBlocks(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)


class Encoder(nn.Module):
    def __init__(self, resolution, in_channels, ch, ch_mult, num_res_blocks, z_channels):
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
        self.norm_out = nn.GroupNorm(num_groups=_GN_GROUPS, num_channels=block_in, eps=_GN_EPS, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
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
    def __init__(self, ch, out_ch, ch_mult, num_res_blocks, in_channels, resolution, z_channels):
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
        self.norm_out = nn.GroupNorm(num_groups=_GN_GROUPS, num_channels=block_in, eps=_GN_EPS, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
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
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor, generator: torch.Generator | None = None) -> Tensor:
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
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            params.resolution,
            params.in_channels,
            params.ch,
            params.ch_mult,
            params.num_res_blocks,
            params.z_channels,
        )
        self.decoder = Decoder(
            params.ch,
            params.out_ch,
            params.ch_mult,
            params.num_res_blocks,
            params.in_channels,
            params.resolution,
            params.z_channels,
        )
        self.reg = DiagonalGaussian()
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        z = self.reg(self.encoder(x), generator)
        return self.scale_factor * (z - self.shift_factor)

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)
