"""Reusable image patch projection and feature connection layers."""

from math import prod, sqrt

import torch
from torch import nn
from torch.nn import functional as F

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


class TubeletEmbed(nn.Module):
    """Project flattened spatiotemporal patches (tubelets) to token rows.

    A tubelet holds ``in_channels x temporal_size x patch_size x patch_size``
    pixels flattened channel-major, the layout Qwen-VL image processors emit:
    an image repeats its frame ``temporal_size`` times, and a video tubelet
    spans ``temporal_size`` consecutive frames. ``weight`` keeps the Conv3d
    layout ``[embed_dim, in_channels, temporal_size, patch_size,
    patch_size]`` of the checkpoints. Their convolution kernel equals its
    stride, so one matrix product over the flattened tubelets evaluates it.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        temporal_size: int,
        *,
        bias: bool = True,
    ):
        super().__init__()
        if any(
            type(value) is not int or value < 1
            for value in (in_channels, embed_dim, patch_size, temporal_size)
        ):
            raise ValueError("tubelet dimensions must be positive integers")
        shape = (embed_dim, in_channels, temporal_size, patch_size, patch_size)
        self.weight = nn.Parameter(torch.empty(shape), requires_grad=False)
        # The Conv3d initialization: fan-in is one tubelet's pixel count.
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if bias:
            bound = 1 / sqrt(prod(shape[1:]))
            self.bias = nn.Parameter(
                torch.empty(embed_dim).uniform_(-bound, bound),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @property
    def in_features(self) -> int:
        """Pixel values of one flattened tubelet."""
        return prod(self.weight.shape[1:])

    def forward(self, tubelets: torch.Tensor) -> torch.Tensor:
        """Return ``[..., embed_dim]`` rows of ``[..., in_features]`` tubelets.

        Pixels convert to the weight dtype before the projection.
        """
        if tubelets.shape[-1] != self.in_features:
            raise ValueError("tubelet rows must hold one complete tubelet")
        return F.linear(
            tubelets.to(self.weight.dtype), self.weight.flatten(1), self.bias
        )


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
