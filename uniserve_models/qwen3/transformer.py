"""Qwen3 attention, experts and decoder layer composition."""

from __future__ import annotations

import torch
from torch import nn

from uniserve.model import TransformerDecoder
from uniserve.nn.attention import Attention as ScaledAttention
from uniserve.nn.attention import (
    AttentionInput,
    RotaryQKVProjection,
)
from uniserve.nn.functional import add_rms_norm
from uniserve.nn.linear import (
    Linear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.moe import FusedMoE
from uniserve.nn.norm import RMSNorm
from uniserve.nn.rope import RotaryEmbedding

from .config import Config


def _activation(name: str) -> str:
    if name in {"silu", "swish", "silu_and_mul", "swiglu"}:
        return "silu"
    if name in {"gelu", "gelu_and_mul", "geglu"}:
        return "gelu"
    return "gelu_pytorch_tanh"


class Attention(nn.Module):
    """QK-normalized rotary attention followed by the output contraction."""

    def __init__(self, config: Config, layer: int):
        super().__init__()
        self.qkv = RotaryQKVProjection(
            QKVParallelLinear(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                bias=config.attention_bias,
            ),
            RMSNorm(config.head_dim, config.rms_norm_eps),
            RMSNorm(config.head_dim, config.rms_norm_eps),
        )
        self.attention = ScaledAttention(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            cache_name=f"backbone.layers.{layer}.attention.attention",
        )
        self.output = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rotary = RotaryEmbedding(config.head_dim, theta=config.rope_theta)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, attention: AttentionInput):
        # This unscaled recipe depends only on positions, so computing factors
        # does not require a host mirror of the attention lengths.
        cos, sin = self.rotary(
            positions.reshape(-1), dtype=torch.float32, sequence_length=positions.numel()
        )
        query, key, value = self.qkv(hidden, (cos,), (sin,))
        attended = self.attention(query, key, value, attention)
        return self.output(attended.flatten(1))


class MoE(nn.Module):
    """Top-k expert selection with a replicated mathematical router."""

    def __init__(self, config: Config):
        super().__init__()
        self.router = Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = FusedMoE(
            [
                GatedMLP(config.hidden_size, config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ],
            top_k=config.num_experts_per_tok,
            norm_topk_prob=True,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.experts(hidden, self.router(hidden))


class TransformerLayer(nn.Module):
    """Qwen pre-normalization equations with a separate residual stream."""

    def __init__(self, config: Config, layer: int):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config, layer)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = (
            MoE(config)
            if config.num_experts
            else GatedMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=_activation(config.hidden_act),
            )
        )

    def forward(self, hidden, residual, positions, attention):
        if residual is None:
            residual = hidden
            hidden = self.input_norm(hidden)
        else:
            hidden, residual = add_rms_norm(
                hidden, residual, self.input_norm.weight, self.input_norm.eps
            )
        hidden = self.attention(hidden, positions, attention)
        hidden, residual = add_rms_norm(
            hidden, residual, self.post_attention_norm.weight, self.post_attention_norm.eps
        )
        return self.mlp(hidden), residual


class Transformer(TransformerDecoder):
    def __init__(self, config: Config):
        super().__init__(
            VocabParallelEmbedding(config.vocab_size, config.hidden_size),
            nn.ModuleDict(
                {
                    str(index): TransformerLayer(config, index)
                    for index in range(config.num_hidden_layers)
                }
            ),
            RMSNorm(config.hidden_size, config.rms_norm_eps),
        )
        self.config = config
