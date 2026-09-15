"""Sinusoidal scalar time features and their learned projection."""

import math

import torch
from torch import nn

from .linear import Linear


def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    """Return cosine-then-sine FP32 features in GLIDE/DiT frequency order."""
    half = dim // 2
    frequency = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    angles = timesteps.reshape(-1, 1).float() * frequency[None]
    values = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if dim % 2:
        values = torch.cat((values, torch.zeros_like(values[:, :1])), dim=-1)
    return values


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        super().__init__()
        if type(frequency_dim) is not int or frequency_dim < 2:
            raise ValueError("timestep frequency width must be at least two")
        self.frequency_dim = frequency_dim
        self.projection = nn.Sequential(
            Linear(frequency_dim, hidden_size, bias=True),
            nn.SiLU(),
            Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        features = timestep_embedding(timesteps, self.frequency_dim)
        return self.projection(features.to(self.projection[0].weight.dtype))
