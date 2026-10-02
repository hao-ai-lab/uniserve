"""Reusable spatial residual, attention and posterior computation."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.nn.attention import Attention, DenseInput


class ResidualBlock(nn.Module):
    """Apply two normalized convolutions and an optional channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norms = nn.ModuleList(
            (
                nn.GroupNorm(groups, in_channels, eps=eps),
                nn.GroupNorm(groups, out_channels, eps=eps),
            )
        )
        self.convolutions = nn.ModuleList(
            (
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            )
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for norm, convolution in zip(
            self.norms, self.convolutions, strict=True
        ):
            hidden = convolution(F.silu(norm(hidden)))
        return self.shortcut(x) + hidden


class AttentionBlock(nn.Module):
    """Add full spatial attention, with channels forming a single head."""

    def __init__(self, channels: int, *, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=eps)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.attention = Attention(1, 1, channels)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        # [batch, 1 head, H*W tokens, channels] per projection.
        query, key, value = (
            tensor.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
            for tensor in self.qkv(self.norm(x)).chunk(3, dim=1)
        )

        # Every query chunk attends to the full spatial context. This bounds
        # portable score workspace without changing the attention equation.
        result = torch.cat(
            tuple(
                self.attention(chunk, key, value, DenseInput(False, None))
                for chunk in query.split(4096, dim=2)
            ),
            dim=2,
        )
        hidden = (
            result.squeeze(1)
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )
        return x + self.output(hidden)


class Downsample(nn.Module):
    """Halve the spatial extent with a strided convolution and asymmetric
    pad.
    """  # noqa: D205

    def __init__(self, channels: int):
        super().__init__()
        self.convolution = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad bottom/right only so the output extent follows the input parity.
        return self.convolution(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    """Double the spatial extent with nearest interpolation and a
    convolution.
    """  # noqa: D205

    def __init__(self, channels: int):
        super().__init__()
        self.convolution = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convolution(
            F.interpolate(x, scale_factor=2.0, mode="nearest")
        )


class DiagonalGaussian(nn.Module):
    """Split mean/log-variance moments and optionally sample the posterior.

    ``log_variance_range`` clamps the log-variance before it scales the
    noise, as latent diffusion codecs bound it against overflow; None leaves
    it unbounded. A sample draws standard normal noise from ``generator`` on
    the moments' device unless the caller supplies ``noise``, a draw with the
    mean's shape, dtype and device. A caller that reproduces a reference
    draw, such as one fixed-seed host draw over a longer timeline than these
    moments cover, supplies its share of that draw.
    """

    def __init__(
        self,
        sample: bool = True,
        chunk_dim: int = 1,
        log_variance_range: tuple[float, float] | None = None,
    ):
        super().__init__()
        if log_variance_range is not None and (
            len(log_variance_range) != 2
            or not all(math.isfinite(bound) for bound in log_variance_range)
            or log_variance_range[0] >= log_variance_range[1]
        ):
            raise ValueError(
                "posterior log-variance range must be a finite increasing "
                "interval"
            )
        self.sample, self.chunk_dim = sample, chunk_dim
        self.log_variance_range = log_variance_range

    def forward(
        self,
        moments: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if moments.shape[self.chunk_dim] % 2:
            raise ValueError(
                "posterior moments require equally sized mean and "
                "log-variance fields"
            )

        mean, log_variance = moments.chunk(2, dim=self.chunk_dim)
        if not self.sample:
            if noise is not None:
                raise ValueError("the posterior mean takes no noise")
            return mean

        if self.log_variance_range is not None:
            log_variance = log_variance.clamp(*self.log_variance_range)
        if noise is None:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
        elif (
            noise.shape != mean.shape
            or noise.dtype != mean.dtype
            or noise.device != mean.device
        ):
            raise ValueError(
                "posterior noise must match the mean's shape, dtype and device"
            )
        return mean + torch.exp(0.5 * log_variance) * noise
