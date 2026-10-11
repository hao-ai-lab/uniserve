"""BAGEL text and flow experts over a shared attention domain."""

from __future__ import annotations

import torch
from torch import nn

from uniserve.model.mot import MoTDecoder, MoTLayer
from uniserve.nn.attention import Attention, RotaryQKVProjection
from uniserve.nn.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.norm import RMSNorm
from uniserve.nn.rope import RotaryEmbedding

from .config import TransformerConfig


class TransformerLayer(MoTLayer):
    """Route projection/MLP experts around one shared token-attention domain.

    The ``text`` and ``flow`` routes each own an input norm, a QKV projection
    with optional QK normalization, an attention output projection, a
    post-attention norm and a gated MLP. One attention call covers the tokens
    of both routes in their packed order. The checkpoint's expert MLPs
    consume BF16 normalization results, including when their surrounding
    accumulation is higher precision.
    """

    def __init__(self, config: TransformerConfig, index: int):
        hidden, dim = config.hidden_size, config.head_dim

        def projection():
            return RotaryQKVProjection(
                QKVParallelLinear(
                    hidden,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    dim,
                    bias=True,
                ),
                RMSNorm(dim, config.rms_norm_eps)
                if config.qk_norm
                else nn.Identity(),
                RMSNorm(dim, config.rms_norm_eps)
                if config.qk_norm
                else nn.Identity(),
            )

        super().__init__(
            # The denoiser shares this layer with the text LM, so both
            # address the K/V cache entry named here.
            attention=Attention(
                config.num_attention_heads,
                config.num_key_value_heads,
                dim,
                cache_name=f"text.backbone.layers.{index}.attention",
            ),
            hidden_size=hidden,
            eps=config.rms_norm_eps,
            projection=projection,
            output=lambda: RowParallelLinear(
                config.num_attention_heads * dim, hidden, bias=False
            ),
            mlp=lambda: GatedMLP(hidden, config.intermediate_size),
            mlp_dtype=torch.bfloat16,
        )
        self.rotary = RotaryEmbedding(dim, theta=config.rope_theta)

    def position_embeddings(self, positions):
        # All routes share one temporal rotary domain; only the expert
        # projections, outputs, and MLPs differ per route. Positions are either
        # 1-D or [3, tokens], of which RoPE uses only the temporal row.
        temporal = positions if positions.ndim == 1 else positions[0]
        cosine, sine = self.rotary(
            temporal, dtype=torch.float32, sequence_length=temporal.numel()
        )
        return (cosine,), (sine,)


class Transformer(MoTDecoder):
    """Stack routed MoT layers over a shared embedding with per-route norms."""

    def __init__(self, config: TransformerConfig):
        super().__init__(
            VocabParallelEmbedding(config.vocab_size, config.hidden_size),
            nn.ModuleDict(
                {
                    str(index): TransformerLayer(config, index)
                    for index in range(config.num_hidden_layers)
                }
            ),
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.config = config
