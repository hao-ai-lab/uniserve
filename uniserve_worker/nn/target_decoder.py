"""Shared route-aware decoder root for target family adapters (dormant).

The reusable numerical composition family roots build on (Stage 5/6 of
``specs/unified_forward_execution.md``): token embedding, RMSNorm, rotary
embedding from the packed position column, grouped QKV/O and MLP projections
dispatched by ``SegmentTable.route_id``, and the injected shared attention
seam. The root is fully table-driven — routes and overlay slots are read
from the segment table, never from operation kinds — so text decode rows and
generation-route denoise rows traverse the same layers in one invocation.

Family files own their configuration, weights, and output projection; this
module owns only the packed traversal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..contracts.residency_batch import ResidencyBatchArrays
from ..contracts.segment_table import (
    AttentionLayerSpec,
    GraphCapacity,
    SegmentTableArrays,
)
from .grouped_routing import GroupedLinear, WeightOverlayBank

__all__ = [
    "DecoderLayerWeights",
    "SharedAttention",
    "TargetDecoderConfig",
    "TargetDecoderRoot",
]


class SharedAttention(Protocol):
    """The injected shared attention seam; providers live in backends/."""

    def prepare(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
    ) -> None: ...

    def forward(
        self,
        layer: AttentionLayerSpec,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class TargetDecoderConfig:
    vocab_size: int
    hidden_size: int
    layers: int
    routes: int
    query_heads: int
    kv_heads: int
    head_dim: int
    mlp_hidden: int
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6
    # Qwen3-style per-head RMSNorm on Q and K before rotary embedding.
    qk_norm: bool = False


@dataclass(slots=True)
class DecoderLayerWeights:
    input_norm: torch.Tensor
    q: GroupedLinear
    k: GroupedLinear
    v: GroupedLinear
    o: GroupedLinear
    post_norm: torch.Tensor
    gate: GroupedLinear
    up: GroupedLinear
    down: GroupedLinear
    q_norm: torch.Tensor | None = None
    k_norm: torch.Tensor | None = None


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Neox-style rotary embedding over ``[tokens, heads, head_dim]``."""

    head_dim = x.shape[-1]
    half = head_dim // 2
    freqs = theta ** (
        -torch.arange(0, half, device=x.device, dtype=torch.float32) / half
    )
    angles = positions.float()[:, None] * freqs[None, :]
    cos = torch.cos(angles)[:, None, :]
    sin = torch.sin(angles)[:, None, :]
    first, second = x[..., :half].float(), x[..., half:].float()
    return torch.cat(
        [first * cos - second * sin, second * cos + first * sin], dim=-1
    ).to(x.dtype)


class TargetDecoderRoot:
    """One packed route-grouped traversal producing logits per token."""

    def __init__(
        self,
        config: TargetDecoderConfig,
        attention: SharedAttention,
        *,
        device: torch.device | str = "cuda",
        seed: int = 0,
        overlay_bank: WeightOverlayBank | None = None,
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.attention = attention
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def weight(*shape: int) -> torch.Tensor:
            if zero_init:
                return torch.zeros(*shape, device=self.device, dtype=dtype)
            return (
                (torch.randn(*shape, generator=generator) * (shape[-1] ** -0.5))
                .to(dtype)
                .to(self.device)
            )

        c = config
        q_dim = c.query_heads * c.head_dim
        kv_dim = c.kv_heads * c.head_dim
        self.embedding = weight(c.vocab_size, c.hidden_size)
        self.layers: list[DecoderLayerWeights] = []
        for _ in range(c.layers):
            self.layers.append(
                DecoderLayerWeights(
                    input_norm=torch.ones(c.hidden_size, device=self.device, dtype=dtype),
                    q=GroupedLinear(
                        weight(c.routes, q_dim, c.hidden_size), overlay_bank
                    ),
                    k=GroupedLinear(weight(c.routes, kv_dim, c.hidden_size)),
                    v=GroupedLinear(weight(c.routes, kv_dim, c.hidden_size)),
                    o=GroupedLinear(weight(c.routes, c.hidden_size, q_dim)),
                    post_norm=torch.ones(c.hidden_size, device=self.device, dtype=dtype),
                    gate=GroupedLinear(weight(c.routes, c.mlp_hidden, c.hidden_size)),
                    up=GroupedLinear(weight(c.routes, c.mlp_hidden, c.hidden_size)),
                    down=GroupedLinear(weight(c.routes, c.hidden_size, c.mlp_hidden)),
                    q_norm=(
                        torch.ones(c.head_dim, device=self.device, dtype=dtype)
                        if c.qk_norm
                        else None
                    ),
                    k_norm=(
                        torch.ones(c.head_dim, device=self.device, dtype=dtype)
                        if c.qk_norm
                        else None
                    ),
                )
            )
        self.final_norm = torch.ones(c.hidden_size, device=self.device, dtype=dtype)
        self.lm_head = weight(c.vocab_size, c.hidden_size)

    # ------------------------------------------------------------------ #

    def token_columns(
        self,
        segments: SegmentTableArrays,
        capacity: GraphCapacity,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Expand per-segment route and overlay columns to token columns."""

        active_tokens = 0
        route_values: list[int] = []
        overlay_values: list[int] = []
        for index in range(capacity.segments):
            if not segments.segment_active[index]:
                continue
            count = segments.token_count[index]
            route_values.extend([segments.route_id[index]] * count)
            overlay_values.extend([segments.overlay_slot[index]] * count)
            active_tokens += count
        return (
            active_tokens,
            torch.tensor(route_values, device=self.device, dtype=torch.long),
            torch.tensor(overlay_values, device=self.device, dtype=torch.long),
        )

    def logits(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
        capacity: GraphCapacity,
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> torch.Tensor:
        config = self.config
        active_tokens, route_of_token, overlay_of_token = self.token_columns(
            segments, capacity
        )
        ids = torch.tensor(
            token_ids[:active_tokens], device=self.device, dtype=torch.long
        )
        position_column = torch.tensor(
            positions[:active_tokens], device=self.device, dtype=torch.long
        )
        hidden = self.embedding.index_select(0, ids)
        route_of_token = route_of_token.to(self.device)
        if overlay_of_token is not None:
            overlay_of_token = overlay_of_token.to(self.device)
        self.attention.prepare(segments, residency)
        for layer_id, layer in enumerate(self.layers):
            normed = _rms_norm(hidden, layer.input_norm, config.rms_eps)
            q = layer.q.forward(normed, route_of_token, overlay_of_token)
            k = layer.k.forward(normed, route_of_token)
            v = layer.v.forward(normed, route_of_token)
            q = q.view(active_tokens, config.query_heads, config.head_dim)
            k = k.view(active_tokens, config.kv_heads, config.head_dim)
            v = v.view(active_tokens, config.kv_heads, config.head_dim)
            if layer.q_norm is not None:
                q = _rms_norm(q, layer.q_norm, config.rms_eps)
            if layer.k_norm is not None:
                k = _rms_norm(k, layer.k_norm, config.rms_eps)
            q = _rope(q, position_column, config.rope_theta)
            k = _rope(k, position_column, config.rope_theta)
            spec = AttentionLayerSpec(
                layer_id=layer_id,
                site_id=1,
                domain_id=1,
                query_heads=config.query_heads,
                kv_heads=config.kv_heads,
                qk_head_dim=config.head_dim,
                value_head_dim=config.head_dim,
                scale=config.head_dim**-0.5,
            )
            attended = self.attention.forward(spec, q, k, v)
            hidden = hidden + layer.o.forward(
                attended.reshape(active_tokens, -1), route_of_token
            )
            normed = _rms_norm(hidden, layer.post_norm, config.rms_eps)
            gate = layer.gate.forward(normed, route_of_token)
            up = layer.up.forward(normed, route_of_token)
            hidden = hidden + layer.down.forward(
                torch.nn.functional.silu(gate) * up, route_of_token
            )
        return _rms_norm(hidden, self.final_norm, config.rms_eps) @ self.lm_head.T
