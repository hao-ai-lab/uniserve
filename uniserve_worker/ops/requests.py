"""Typed operator requests."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from ..forward import ForwardContext


class AttentionRegime(str, Enum):
    DENSE = "dense"
    EXTEND = "extend"
    DECODE = "decode"
    MIXED = "mixed"
    VISIBLE_END = "visible_end"


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
    q_weight: torch.Tensor | tuple[torch.Tensor, ...]
    k_weight: torch.Tensor | tuple[torch.Tensor, ...]
    eps: float
    axis_dims: tuple[int, ...] | None = None


@dataclass(frozen=True)
class QKNormRopeReq:
    q: torch.Tensor
    k: torch.Tensor
    q_weight: torch.Tensor | tuple[torch.Tensor, ...]
    k_weight: torch.Tensor | tuple[torch.Tensor, ...]
    cos: torch.Tensor | tuple[torch.Tensor, ...]
    sin: torch.Tensor | tuple[torch.Tensor, ...]
    eps: float
    position_ids: torch.Tensor | None = None
    unsqueeze_dim: int = 1
    axis_dims: tuple[int, ...] | None = None
    # Caller-declared axes whose positions are all zero for every token in this
    # call (a zero-angle rotation is the identity). A pure optimization hint:
    # providers may fuse or skip those axes' rotations; ignoring it is always
    # correct because the supplied cos/sin tables already encode the identity.
    identity_axes: tuple[int, ...] | None = None
    quant: Any | None = None


@dataclass(frozen=True)
class PackedRopeReq:
    x: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


@dataclass(frozen=True)
class AttentionReq:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    regime: AttentionRegime
    causal: bool
    scale: float
    attn_mask: torch.Tensor | None = None
    kv_cache: Any | None = None
    metadata: Any | None = None
    ctx: ForwardContext | None = None
    stats: Any | None = None
    block_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None
    current_k: torch.Tensor | None = None
    current_v: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    visible_end: torch.Tensor | None = None
    page_table: torch.Tensor | None = None
    seqused_k: torch.Tensor | None = None
    use_prefix_bounds: bool = False
    fully_visible: bool = False


@dataclass(frozen=True)
class TpAllReduceReq:
    tensor: torch.Tensor
    op: str
    axis: Any
