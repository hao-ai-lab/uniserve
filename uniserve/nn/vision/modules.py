"""Reusable image patch projection and feature connection layers."""

from math import prod, sqrt

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.nn.activation import get_act_fn
from uniserve.nn.linear import Linear


class PatchEmbed(nn.Module):
    """Project flattened pixel patches to token rows.

    A patch holds ``in_channels`` channels over ``patch_shape`` pixels:
    ``(height, width)``, or ``(frames, height, width)`` for a spatiotemporal
    patch (a tubelet). Each input row flattens one patch channel-major, the
    packed layout patch encoders read. ``weight`` keeps the checkpoint
    layout ``[embed_dim, in_channels, *patch_shape]`` of a Conv2d or Conv3d
    whose kernel equals its stride, so one matrix product over the flattened
    patches evaluates that convolution.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_shape: tuple[int, ...],
        *,
        bias: bool = True,
    ):
        super().__init__()
        if (
            not isinstance(patch_shape, tuple)
            or len(patch_shape) not in (2, 3)
            or any(
                type(value) is not int or value < 1
                for value in (in_channels, embed_dim, *patch_shape)
            )
        ):
            raise ValueError(
                "patch embedding requires positive channel, width and 2-D or "
                "3-D patch dimensions"
            )
        shape = (embed_dim, in_channels, *patch_shape)
        self.weight = nn.Parameter(torch.empty(shape), requires_grad=False)
        # The convolution initialization: fan-in is one patch's pixel count.
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
        """Pixel values of one flattened patch."""
        return prod(self.weight.shape[1:])

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Return ``[..., embed_dim]`` rows of ``[..., in_features]`` patches.

        Pixels convert to the weight dtype before the projection.
        """
        if patches.shape[-1] != self.in_features:
            raise ValueError("patch rows must hold one complete patch")
        return F.linear(
            patches.to(self.weight.dtype), self.weight.flatten(1), self.bias
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
