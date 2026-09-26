"""Reusable spatial residual, attention and posterior computation."""

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.nn.attention import Attention, AttentionBatch, DenseInput


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
                self.attention(
                    chunk,
                    key,
                    value,
                    AttentionBatch.single(DenseInput(False, None)),
                )
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
    """Split mean/log-variance moments and optionally sample the posterior."""

    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample, self.chunk_dim = sample, chunk_dim

    def forward(
        self, moments: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if moments.shape[self.chunk_dim] % 2:
            raise ValueError(
                "posterior moments require equally sized mean and "
                "log-variance fields"
            )

        mean, log_variance = moments.chunk(2, dim=self.chunk_dim)
        if not self.sample:
            return mean

        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return mean + torch.exp(0.5 * log_variance) * noise
