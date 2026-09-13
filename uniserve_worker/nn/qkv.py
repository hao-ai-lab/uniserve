"""Complete QKV projection branches for shared attention computations."""

from __future__ import annotations

import torch
from torch import nn

from .linear import QKVParallelLinear
from .rope import apply_rotary_emb


class QKV(nn.Module):
    """Project hidden rows and normalize rotary query/key heads as one module.

    Input and output tensors belong to the caller's numerical device. Runtime
    may bind the complete module to another device, including its rotary inputs.
    Subclasses implement the model's normalization and rotary equations.
    """

    def __init__(self, projection: QKVParallelLinear, *, separate: bool = False) -> None:
        super().__init__()
        self.projection = projection
        self.separate = separate
        self.head_dim = projection.head_size

    def forward(
        self,
        hidden: torch.Tensor,
        cos: tuple[torch.Tensor, ...],
        sin: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sizes = self.projection.output_sizes
        projected = (
            self.projection.forward_branches(hidden)
            if self.separate
            else self.projection(hidden).split(sizes, dim=-1)
        )
        query, key, value = (
            tensor.view(-1, size // self.head_dim, self.head_dim)
            for tensor, size in zip(projected, sizes, strict=True)
        )
        query, key = self.normalize(query, key, cos, sin)
        return query, key, value.to(query.dtype)

    def normalize(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: tuple[torch.Tensor, ...],
        sin: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class RotaryQKV(QKV):
    """Apply FP32 head normalization and one rotary axis, returning BF16 heads."""

    def __init__(self, projection: QKVParallelLinear, query_norm: nn.Module, key_norm: nn.Module):
        super().__init__(projection)
        self.query_norm = query_norm
        self.key_norm = key_norm

    def normalize(self, query, key, cos, sin):
        query = self.query_norm(query.float())
        key = self.key_norm(key.float())
        return (
            apply_rotary_emb(query, cos[0], sin[0]).to(torch.bfloat16),
            apply_rotary_emb(key, cos[0], sin[0]).to(torch.bfloat16),
        )
