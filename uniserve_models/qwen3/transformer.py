"""Qwen3 attention, experts and decoder layer composition.

``Transformer`` is the Qwen3 backbone: ``Model`` places it under a vocabulary
head, and the MiniMax H3 text encoder uses it without the head. Traversal,
pipeline exchange and the final norm belong to the shared
``TransformerDecoder``; this module supplies the Qwen3 layer equations.
"""

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
    """Map checkpoint activation aliases onto the library's implemented kernels."""  # noqa: E501
    if name in {"silu", "swish", "silu_and_mul", "swiglu"}:
        return "silu"
    if name in {"gelu", "gelu_and_mul", "geglu"}:
        return "gelu"
    # Config accepts only the aliases above and the two tanh GELU spellings.
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
        self.rotary = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta,
            sections=config.mrope_sections,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionInput,
    ):
        """Attend over packed tokens at their rotary ``positions``.

        Positions are ``[tokens]``; a model with ``mrope_sections`` also
        accepts ``[axes, tokens]`` multimodal coordinates, and its
        one-dimensional positions place every axis at the same coordinate.
        """
        # This unscaled recipe depends only on positions, so computing factors
        # does not require a host mirror of the attention lengths.
        if self.rotary.sections is None:
            cos, sin = self.rotary(
                positions.reshape(-1),
                dtype=torch.float32,
                sequence_length=positions.numel(),
            )
        else:
            if positions.ndim == 1:
                positions = positions.expand(len(self.rotary.sections), -1)
            cos, sin = self.rotary(
                positions,
                dtype=torch.float32,
                sequence_length=positions.shape[-1],
            )

        # hidden: [tokens, hidden_size]; q/k/v: [tokens, TP-local heads,
        # head_dim]. Flattening the attended heads feeds the row-parallel
        # output projection, which sums the TP shards.
        query, key, value = self.qkv(hidden, (cos,), (sin,))
        attended = self.attention(query, key, value, attention)
        return self.output(attended.flatten(1))


class MoE(nn.Module):
    """Top-k expert selection with a replicated mathematical router.

    Expert weights are softmax probabilities renormalized over the selected
    top-k. Every expert is a SiLU ``GatedMLP``, whatever
    ``Config.hidden_act`` names; only the dense MLP reads it.
    """

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
        self.post_attention_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
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
        # Only the first layer of the first pipeline stage receives no
        # residual; every other call fuses the residual add into the norm to
        # keep one kernel per normalization.
        if residual is None:
            residual = hidden
            hidden = self.input_norm(hidden)
        else:
            hidden, residual = add_rms_norm(
                hidden, residual, self.input_norm.weight, self.input_norm.eps
            )

        hidden = self.attention(hidden, positions, attention)
        hidden, residual = add_rms_norm(
            hidden,
            residual,
            self.post_attention_norm.weight,
            self.post_attention_norm.eps,
        )

        # The MLP output stays unsummed: the next layer's input norm, possibly
        # on the next pipeline stage, or ``TransformerDecoder.forward`` after
        # the last layer adds the residual.
        return self.mlp(hidden), residual


class Transformer(TransformerDecoder):
    """Qwen3 decoder stack: embedding, pre-norm layers, and the final RMS norm."""  # noqa: E501

    def __init__(self, config: Config):
        # Layer keys are global layer indices. Pipeline binding keeps the
        # keys of its resident layers, so parameter paths still match the
        # checkpoint names in ``weights.parameter_sources``.
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
