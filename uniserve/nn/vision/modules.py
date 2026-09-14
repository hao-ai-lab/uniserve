"""Reusable ``nn.Module`` vision building blocks shared by multimodal models."""

from __future__ import annotations

import torch
import torch.nn as nn

from uniserve.nn.activation import get_act_fn

__all__ = [
    "PatchEmbed",
    "MLPConnector",
]


class PatchEmbed(nn.Module):
    """Converts images into a flattened sequence of projected non-overlapping patches."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        *,
        bias: bool = True,
        flatten: bool = True,
    ) -> None:
        """Configure non-overlapping convolutional patch projection and output layout."""

        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image patches and optionally flatten their spatial grid into token rows."""

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


class MLPConnector(nn.Module):
    """ViT-to-language connector (fc1 + activation + fc2)."""

    def __init__(
        self, input_dim: int, output_dim: int, activation: str = "gelu_pytorch_tanh"
    ) -> None:
        """Build a two-layer projection from vision features to language width."""

        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)
        self.act = get_act_fn(activation)
        self.fc2 = nn.Linear(output_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map vision features into the language hidden width through an activated MLP."""

        return self.fc2(self.act(self.fc1(x)))
