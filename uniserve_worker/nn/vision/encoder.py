"""Shared configurable vision encoder stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

import uniserve_worker.ops as ops

from ..attention import RadixAttention
from ..linear import LinearBase

__all__ = [
    'VisionEncoderConfig',
    'VisionSelfAttention',
    'max_seqlen_from_cu',
    'VisionEncoderLayer',
    'VisionEncoder',
]


@dataclass(frozen=True)
class VisionEncoderConfig:
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    num_hidden_layers: int
    layer_norm_eps: float = 1e-6


class VisionSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("vision hidden_size must be divisible by num_attention_heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size) // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = LinearBase(hidden_size, hidden_size)
        self.k_proj = LinearBase(hidden_size, hidden_size)
        self.v_proj = LinearBase(hidden_size, hidden_size)
        self.out_proj = LinearBase(hidden_size, hidden_size)
        self.attn = RadixAttention(self.num_heads, self.num_heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        varlen_backend: Any | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        n_tokens = x.shape[0]
        q = self.q_proj(x).view(n_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(n_tokens, self.num_heads, self.head_dim)
        cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
        if max_seqlen is None:
            max_seqlen = max_seqlen_from_cu(cu, n_tokens)
        backend = varlen_backend
        override = "context" if backend is not None else "flash_attn"
        if ops.can_run_attention(
            q,
            k,
            v,
            regime=ops.AttentionRegime.EXTEND,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
            scale=self.scale,
            backend=backend,
            override=override,
        ):
            # Single varlen attention call over the whole packed batch; the
            # per-image isolation is enforced by ``cu_seqlens`` instead of a
            # Python loop with one dense kernel launch per image.
            out = ops.attention(
                q,
                k,
                v,
                regime=ops.AttentionRegime.EXTEND,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
                scale=self.scale,
                backend=backend,
                override=override,
            )
            return self.out_proj(out.reshape(n_tokens, -1))
        # Portable fallback (e.g. CPU / SDPA-only backends without a varlen
        # kernel): attend each packed image segment independently.
        out = torch.empty_like(q)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            q_block = q[start:end].permute(1, 0, 2).unsqueeze(0)
            k_block = k[start:end].permute(1, 0, 2).unsqueeze(0)
            v_block = v[start:end].permute(1, 0, 2).unsqueeze(0)
            attended = self.attn(q_block, k_block, v_block, causal=False, scale=self.scale)
            out[start:end] = attended.squeeze(0).permute(1, 0, 2)
        return self.out_proj(out.reshape(n_tokens, -1))

def max_seqlen_from_cu(cu_seqlens: torch.Tensor, n_tokens: int) -> int:
    """Host-side max segment length from a cumulative-seqlen tensor (one sync)."""
    if cu_seqlens.numel() > 1:
        return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return n_tokens


class VisionEncoderLayer(nn.Module):
    def __init__(self, cfg: VisionEncoderConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.self_attn = VisionSelfAttention(cfg.hidden_size, cfg.num_attention_heads)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = nn.Sequential(
            LinearBase(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(approximate="tanh"),
            LinearBase(cfg.intermediate_size, cfg.hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        varlen_backend: Any | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.layer_norm1(x),
            cu_seqlens,
            varlen_backend=varlen_backend,
            max_seqlen=max_seqlen,
        )
        return x + self.mlp(self.layer_norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, cfg: VisionEncoderConfig, *, post_norm: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(VisionEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.post_layernorm = (
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps) if post_norm else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
        varlen_backend = None
        max_seqlen = None
        if self.layers:
            cu = cu_seqlens.to(device=x.device, dtype=torch.int32)
            max_seqlen = max_seqlen_from_cu(cu, x.shape[0])
        for layer in self.layers:
            x = layer(x, cu_seqlens, varlen_backend=varlen_backend, max_seqlen=max_seqlen)
        return self.post_layernorm(x)
