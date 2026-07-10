"""Sinusoidal timestep embeddings for flow-matching modules."""
from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn

__all__ = [
    'timestep_embedding',
    'TimestepEmbedder',
]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """GLIDE/DiT sinusoidal timestep embedding with cosine-then-sine layout."""

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Two-layer MLP over sinusoidal scalar timestep features."""

    timestep_embedding = staticmethod(timestep_embedding)

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        input_layer = cast(nn.Linear, self.mlp[0])
        return self.mlp(t_freq.to(input_layer.weight.dtype))
