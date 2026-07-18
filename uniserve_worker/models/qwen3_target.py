"""Target Qwen3 family adapter over the dormant unified stack.

Stage 6 family port of ``specs/unified_forward_execution.md`` in its
architectural entirety: one adapter, one resident root, one packed
traversal. The root owns the Qwen3 neural topology — token embedding,
RMSNorm, rotary embedding from the packed position column, grouped QKV/O
projections (single text route today, the operator is route-general), the
shared attention layer delegating to the injected backend, gated MLP, final
norm, logits head — and interprets nothing about caches, plans, providers,
or graphs.

The adapter translates the packed device tables into one root invocation and
projects compact outcomes: greedy sampled tokens for the requested output
positions and greedy candidate acceptance for verification spans. Weights
arrive through the constructor (checkpoint reuse is explicitly permitted by
the spec); tests instantiate a small random-weight configuration and prove
numerical conformance against an independent dense recompute of the same
function, which is the family conformance shape Stage 11 scales up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..contracts.cache_schema import CacheEffect
from ..contracts.residency_batch import ResidencyBatchArrays
from ..contracts.segment_table import (
    AttentionLayerSpec,
    GraphCapacity,
    SegmentTableArrays,
)
from ..execution.transaction import AdapterPayload, AdapterRowOutcome
from ..nn.grouped_routing import GroupedLinear, WeightOverlayBank

__all__ = ["Qwen3Target", "Qwen3TargetConfig", "SharedAttention"]


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
class Qwen3TargetConfig:
    vocab_size: int
    hidden_size: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    mlp_hidden: int
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6


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


@dataclass(slots=True)
class _DecoderLayer:
    input_norm: torch.Tensor
    q: GroupedLinear
    k: GroupedLinear
    v: GroupedLinear
    o: GroupedLinear
    post_norm: torch.Tensor
    gate: GroupedLinear
    up: GroupedLinear
    down: GroupedLinear


class Qwen3Target:
    """One resident root, one packed traversal, compact projected outcomes."""

    def __init__(
        self,
        config: Qwen3TargetConfig,
        attention: SharedAttention,
        *,
        device: torch.device | str = "cuda",
        seed: int = 0,
        overlay_bank: WeightOverlayBank | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.backend = attention
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def weight(*shape: int) -> torch.Tensor:
            return (
                torch.randn(*shape, generator=generator) * (shape[-1] ** -0.5)
            ).to(self.device)

        c = config
        q_dim = c.query_heads * c.head_dim
        kv_dim = c.kv_heads * c.head_dim
        self.embedding = weight(c.vocab_size, c.hidden_size)
        self.layers: list[_DecoderLayer] = []
        for _ in range(c.layers):
            self.layers.append(
                _DecoderLayer(
                    input_norm=torch.ones(c.hidden_size, device=self.device),
                    q=GroupedLinear(weight(1, q_dim, c.hidden_size), overlay_bank),
                    k=GroupedLinear(weight(1, kv_dim, c.hidden_size)),
                    v=GroupedLinear(weight(1, kv_dim, c.hidden_size)),
                    o=GroupedLinear(weight(1, c.hidden_size, q_dim)),
                    post_norm=torch.ones(c.hidden_size, device=self.device),
                    gate=GroupedLinear(weight(1, c.mlp_hidden, c.hidden_size)),
                    up=GroupedLinear(weight(1, c.mlp_hidden, c.hidden_size)),
                    down=GroupedLinear(weight(1, c.hidden_size, c.mlp_hidden)),
                )
            )
        self.final_norm = torch.ones(c.hidden_size, device=self.device)
        self.lm_head = weight(c.vocab_size, c.hidden_size)

    # ------------------------------------------------------------------ #
    # The family root: hidden states for every packed token.
    # ------------------------------------------------------------------ #

    def hidden_and_logits(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
        payload: AdapterPayload,
        *,
        active_tokens: int,
        route_of_token: torch.Tensor,
        overlay_of_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config = self.config
        token_ids = torch.tensor(
            payload.token_ids[:active_tokens], device=self.device, dtype=torch.long
        )
        positions = torch.tensor(
            payload.positions[:active_tokens], device=self.device, dtype=torch.long
        )
        hidden = self.embedding.index_select(0, token_ids)
        self.backend.prepare(segments, residency)
        for layer_id, layer in enumerate(self.layers):
            normed = _rms_norm(hidden, layer.input_norm, config.rms_eps)
            q = layer.q.forward(normed, route_of_token, overlay_of_token)
            k = layer.k.forward(normed, route_of_token)
            v = layer.v.forward(normed, route_of_token)
            q = q.view(active_tokens, config.query_heads, config.head_dim)
            k = k.view(active_tokens, config.kv_heads, config.head_dim)
            v = v.view(active_tokens, config.kv_heads, config.head_dim)
            q = _rope(q, positions, config.rope_theta)
            k = _rope(k, positions, config.rope_theta)
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
            attended = self.backend.forward(spec, q, k, v)
            hidden = hidden + layer.o.forward(
                attended.reshape(active_tokens, -1), route_of_token
            )
            normed = _rms_norm(hidden, layer.post_norm, config.rms_eps)
            gate = layer.gate.forward(normed, route_of_token)
            up = layer.up.forward(normed, route_of_token)
            hidden = hidden + layer.down.forward(
                torch.nn.functional.silu(gate) * up, route_of_token
            )
        return _rms_norm(hidden, self.final_norm, self.config.rms_eps) @ self.lm_head.T

    # ------------------------------------------------------------------ #
    # ResidentAdapter: compact projected outcomes per row.
    # ------------------------------------------------------------------ #

    def forward(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
        capacity: GraphCapacity,
        payload: AdapterPayload,
    ) -> tuple[AdapterRowOutcome, ...]:
        active = [
            index
            for index in range(capacity.segments)
            if segments.segment_active[index]
        ]
        active_tokens = sum(segments.token_count[index] for index in active)
        route_of_token = torch.zeros(active_tokens, device=self.device, dtype=torch.long)
        logits = self.hidden_and_logits(
            segments,
            residency,
            payload,
            active_tokens=active_tokens,
            route_of_token=route_of_token,
        )
        greedy = logits.argmax(dim=-1)
        sampled: dict[int, int | None] = {}
        accepted: dict[int, int] = {}
        for index in active:
            row = segments.row_id[index]
            sampled.setdefault(row, None)
            accepted.setdefault(row, 0)
            begin = segments.token_begin[index]
            count = segments.token_count[index]
            if segments.candidate_count[index]:
                # Greedy verification: accept while each candidate matches the
                # prediction from the previous position.
                matched = 0
                for offset in range(count):
                    predicted = int(greedy[begin + offset - 1])
                    if predicted == payload.token_ids[begin + offset]:
                        matched += 1
                    else:
                        break
                accepted[row] = matched
                sampled[row] = int(greedy[begin + count - 1])
            elif segments.cache_effect[index] == int(CacheEffect.PERSISTENT_APPEND):
                sampled[row] = int(greedy[begin + count - 1])
        outcomes: list[AdapterRowOutcome] = []
        for row in sorted(sampled):
            token = sampled[row]
            outcomes.append(
                AdapterRowOutcome(
                    sampled_tokens=(token,) if token is not None else (),
                    accepted_candidates=accepted[row],
                )
            )
        return tuple(outcomes)
