"""Reusable image patch projection and feature connection layers."""

import torch
from torch import nn

from uniserve.nn.activation import get_act_fn
from uniserve.nn.linear import Linear


class PatchEmbed(nn.Module):
    """Project NCHW pixels into non-overlapping spatial token rows."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        *,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size, self.in_channels = patch_size, in_channels
        self.projection = nn.Conv2d(
            in_channels, embed_dim, patch_size, stride=patch_size, bias=bias
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, H*W tokens, embed_dim]`` patch rows."""
        return self.projection(pixels).flatten(2).transpose(1, 2)


class MLPConnector(nn.Module):
    """Map vision features to language width through a two-layer MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "gelu_pytorch_tanh",
    ):
        super().__init__()
        self.projection = nn.Sequential(
            Linear(input_dim, output_dim, bias=True),
            get_act_fn(activation),
            Linear(output_dim, output_dim, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)
