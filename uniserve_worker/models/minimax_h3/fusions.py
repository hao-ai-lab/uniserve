"""H3 inference operations with checkpoint-defined FP32 accumulation boundaries."""

from __future__ import annotations

import torch
from torch.nn import functional as F

__all__ = [
    "apply_partial_rope",
    "qk_rmsnorm_rope",
    "row_modulated_rmsnorm",
    "value_first_swiglu",
]


def _rmsnorm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(value.dtype)


def row_modulated_rmsnorm(
    value: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    normalized = _rmsnorm(value, weight, eps)
    selected_shift = shift.index_select(0, row_indices)
    selected_scale = scale.index_select(0, row_indices)
    return normalized * (1.0 + selected_scale) + selected_shift


def apply_partial_rope(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    rotary = cosine.shape[-1]
    head = value[..., :rotary]
    tail = value[..., rotary:]
    first, second = head.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((head * cosine + rotated * sine, tail), dim=-1)


def qk_rmsnorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = _rmsnorm(query, query_weight, eps)
    key = _rmsnorm(key, key_weight, eps)
    return apply_partial_rope(query, cosine, sine), apply_partial_rope(key, cosine, sine)


def value_first_swiglu(value_gate: torch.Tensor) -> torch.Tensor:
    value, gate = value_gate.chunk(2, dim=-1)
    return value * F.silu(gate.float()).to(gate.dtype)
