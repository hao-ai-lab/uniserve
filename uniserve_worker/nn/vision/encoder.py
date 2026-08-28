"""Shared configurable vision encoder stack."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

import uniserve_worker.ops as ops

from ...execution.forward_batch import ForwardBatch
from ..attention import RadixAttention
from ..layer import LayerConfig
from ..linear import LinearBase

__all__ = [
    "VisionEncoderConfig",
    "VisionSelfAttention",
    "VisionEncoderLayer",
    "VisionEncoder",
]


@dataclass(frozen=True)
class VisionEncoderConfig:
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    num_hidden_layers: int
    layer_norm_eps: float = 1e-6


class VisionSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, *, layer_config: LayerConfig) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("vision hidden_size must be divisible by num_attention_heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size) // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = LinearBase(hidden_size, hidden_size, layer_config=layer_config)
        self.k_proj = LinearBase(hidden_size, hidden_size, layer_config=layer_config)
        self.v_proj = LinearBase(hidden_size, hidden_size, layer_config=layer_config)
        self.out_proj = LinearBase(hidden_size, hidden_size, layer_config=layer_config)
        self.attn = RadixAttention(self.num_heads, self.num_heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        context: ForwardBatch,
        *,
        max_seqlen: int,
        seq_lens: Sequence[int],
    ) -> torch.Tensor:
        n_tokens = x.shape[0]
        q = self.q_proj(x).view(n_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(n_tokens, self.num_heads, self.head_dim)
        cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
        varlen = ops.VarlenAttention(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
            scale=self.scale,
            ctx=context,
        )
        provider = self.attn.varlen_provider
        if provider is not None and ops.can_run_attention(varlen, provider=provider):
            # Single varlen attention call over the whole packed batch; the
            # per-image isolation is enforced by ``cu_seqlens`` instead of a
            # Python loop with one dense kernel launch per image.
            out = ops.attention(varlen, provider=provider)
            return self.out_proj(out.reshape(n_tokens, -1))
        # Portable fallback (e.g. CPU / SDPA-only backends without a varlen
        # kernel): attend each packed image segment independently, walking the
        # host-known segment lengths so no offset is read back from the device.
        out = torch.empty_like(q)
        start = 0
        for length in seq_lens:
            end = start + int(length)
            q_block = q[start:end].permute(1, 0, 2).unsqueeze(0)
            k_block = k[start:end].permute(1, 0, 2).unsqueeze(0)
            v_block = v[start:end].permute(1, 0, 2).unsqueeze(0)
            attended = self.attn(
                q_block,
                k_block,
                v_block,
                context,
                causal=False,
                scale=self.scale,
            )
            out[start:end] = attended.squeeze(0).permute(1, 0, 2)
            start = end
        return self.out_proj(out.reshape(n_tokens, -1))


class VisionEncoderLayer(nn.Module):
    def __init__(self, cfg: VisionEncoderConfig, *, layer_config: LayerConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.self_attn = VisionSelfAttention(
            cfg.hidden_size,
            cfg.num_attention_heads,
            layer_config=layer_config,
        )
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = nn.Sequential(
            LinearBase(cfg.hidden_size, cfg.intermediate_size, layer_config=layer_config),
            nn.GELU(approximate="tanh"),
            LinearBase(cfg.intermediate_size, cfg.hidden_size, layer_config=layer_config),
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        context: ForwardBatch,
        *,
        max_seqlen: int,
        seq_lens: Sequence[int],
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.layer_norm1(x),
            cu_seqlens,
            context,
            max_seqlen=max_seqlen,
            seq_lens=seq_lens,
        )
        return x + self.mlp(self.layer_norm2(x))


class VisionEncoder(nn.Module):
    def __init__(
        self,
        cfg: VisionEncoderConfig,
        *,
        layer_config: LayerConfig,
        post_norm: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            VisionEncoderLayer(cfg, layer_config=layer_config) for _ in range(cfg.num_hidden_layers)
        )
        self.post_layernorm = (
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps) if post_norm else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        context: ForwardBatch,
        cu_seqlens: torch.Tensor | None = None,
        *,
        seq_lens: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
            if seq_lens is None:
                seq_lens = (int(x.shape[0]),)
        if seq_lens is None:
            raise ValueError(
                "VisionEncoder requires host-known seq_lens for a segmented packed batch"
            )
        seq_lens = tuple(int(length) for length in seq_lens)
        max_seqlen = max(seq_lens) if seq_lens else int(x.shape[0])
        for layer in self.layers:
            x = layer(x, cu_seqlens, context, max_seqlen=max_seqlen, seq_lens=seq_lens)
        return self.post_layernorm(x)
