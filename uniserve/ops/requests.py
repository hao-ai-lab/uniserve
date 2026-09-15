"""Typed operator requests."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RmsNormReq:
    """Carries an input tensor, normalization weight, and epsilon for RMS normalization."""

    hidden_states: torch.Tensor
    weight: torch.Tensor
    eps: float


@dataclass(frozen=True)
class AddRmsNormReq:
    """Carries mutable input and residual tensors for fused residual addition and RMS normalization."""

    hidden_states: torch.Tensor
    residual: torch.Tensor
    weight: torch.Tensor
    eps: float
    in_place: bool = False


@dataclass(frozen=True)
class SiluAndMulReq:
    """Carries a packed gate/value tensor for gated SiLU activation."""

    x: torch.Tensor


@dataclass(frozen=True)
class QKNormReq:
    """Carries query/key tensors, independent RMS weights, and epsilon for QK normalization."""

    q: torch.Tensor
    k: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    eps: float


@dataclass(frozen=True)
class MultiAxisQKNormReq:
    """Carries Q/K tensors and per-axis dimensions and weights for grouped normalization."""

    q: torch.Tensor
    k: torch.Tensor
    axis_dims: tuple[int, ...]
    q_weights: tuple[torch.Tensor, ...]
    k_weights: tuple[torch.Tensor, ...]
    eps: float


QKNormRequest = QKNormReq | MultiAxisQKNormReq


@dataclass(frozen=True)
class QKNormRopeReq:
    """Carries Q/K tensors, normalization weights, rotary tables, and interleaving policy for fused execution."""

    q: torch.Tensor
    k: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    eps: float
    position_ids: torch.Tensor | None = None
    unsqueeze_dim: int = 1
    in_place: bool = False


@dataclass(frozen=True)
class MultiAxisQKNormRopeReq:
    """Carries per-axis QK normalization and rotary metadata for fused multimodal execution."""

    q: torch.Tensor
    k: torch.Tensor
    axis_dims: tuple[int, ...]
    q_weights: tuple[torch.Tensor, ...]
    k_weights: tuple[torch.Tensor, ...]
    cos_tables: tuple[torch.Tensor, ...]
    sin_tables: tuple[torch.Tensor, ...]
    eps: float
    identity_axes: tuple[int, ...] = ()
    position_ids: torch.Tensor | None = None
    unsqueeze_dim: int = 1


QKNormRopeRequest = QKNormRopeReq | MultiAxisQKNormRopeReq


@dataclass(frozen=True)
class PackedRopeReq:
    """Carries packed Q/K tensors, rotary tables, positions, and rotary-dimension policy."""

    x: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
