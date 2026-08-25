"""Typed operator requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..execution.forward_batch import ForwardBatch


@dataclass(frozen=True)
class RmsNormReq:
    hidden_states: torch.Tensor
    weight: torch.Tensor
    eps: float


@dataclass(frozen=True)
class AddRmsNormReq:
    hidden_states: torch.Tensor
    residual: torch.Tensor
    weight: torch.Tensor
    eps: float
    in_place: bool = False


@dataclass(frozen=True)
class SiluAndMulReq:
    x: torch.Tensor


@dataclass(frozen=True)
class QKNormReq:
    q: torch.Tensor
    k: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    eps: float


@dataclass(frozen=True)
class MultiAxisQKNormReq:
    q: torch.Tensor
    k: torch.Tensor
    axis_dims: tuple[int, ...]
    q_weights: tuple[torch.Tensor, ...]
    k_weights: tuple[torch.Tensor, ...]
    eps: float


QKNormRequest = QKNormReq | MultiAxisQKNormReq


@dataclass(frozen=True)
class QKNormRopeReq:
    q: torch.Tensor
    k: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    eps: float
    position_ids: torch.Tensor | None = None
    unsqueeze_dim: int = 1


@dataclass(frozen=True)
class MultiAxisQKNormRopeReq:
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
    x: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


@dataclass(frozen=True)
class DenseAttention:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    causal: bool
    scale: float
    attn_mask: torch.Tensor | None = None
    ctx: ForwardBatch | None = None


@dataclass(frozen=True)
class PagedDecodeAttention:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    causal: bool
    scale: float
    current_k: torch.Tensor | None = None
    current_v: torch.Tensor | None = None
    kv_cache: Any | None = None
    ctx: ForwardBatch | None = None


@dataclass(frozen=True)
class VarlenAttention:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    causal: bool
    scale: float
    block_table: torch.Tensor | None = None
    kv_cache: Any | None = None
    ctx: ForwardBatch | None = None


@dataclass(frozen=True)
class VisibleEndAttention:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    visible_end: torch.Tensor
    scale: float
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    page_table: torch.Tensor | None = None
    seqused_k: torch.Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    use_prefix_bounds: bool = False
    fully_visible: bool = False
    prefix_k: torch.Tensor | None = None
    prefix_v: torch.Tensor | None = None
    prefix_lens: torch.Tensor | None = None
    ctx: ForwardBatch | None = None


AttentionReq = (
    DenseAttention
    | PagedDecodeAttention
    | VarlenAttention
    | VisibleEndAttention
)
