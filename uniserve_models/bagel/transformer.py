"""BAGEL text and flow experts over a shared attention domain."""

from __future__ import annotations

import torch
from torch import nn

from uniserve.model import TransformerDecoder
from uniserve.nn.attention import Attention, AttentionInput, RotaryQKVProjection
from uniserve.nn.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.norm import RMSNorm
from uniserve.nn.rope import RotaryEmbedding
from uniserve.nn.routing import RoutedTensor, RouteSpan

from .config import TransformerConfig


class TransformerLayer(nn.Module):
    """Route projection/MLP experts around one shared token-attention domain."""

    def __init__(self, config: TransformerConfig, index: int):
        super().__init__()
        hidden, dim = config.hidden_size, config.head_dim
        self.input_norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        self.post_attention_norms = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        for route in ("text", "flow"):
            self.input_norms[route] = RMSNorm(hidden, config.rms_norm_eps)
            self.projections[route] = RotaryQKVProjection(
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
            self.outputs[route] = RowParallelLinear(
                config.num_attention_heads * dim, hidden, bias=False
            )
            self.post_attention_norms[route] = RMSNorm(
                hidden, config.rms_norm_eps
            )
            self.mlps[route] = GatedMLP(hidden, config.intermediate_size)
        self.attention = Attention(
            config.num_attention_heads,
            config.num_key_value_heads,
            dim,
            cache_name=f"text.backbone.layers.{index}.attention",
        )
        self.rotary = RotaryEmbedding(dim, theta=config.rope_theta)

    def forward(
        self,
        hidden: RoutedTensor,
        residual: RoutedTensor | None,
        positions: torch.Tensor,
        attention: AttentionInput,
        *,
        routes: tuple[RouteSpan, ...],
    ):
        hidden = hidden if residual is None else hidden.add(residual)
        normalized = hidden.apply(self.input_norms)

        # All routes share one temporal rotary domain; only the expert
        # projections, outputs, and MLPs differ per route.
        temporal = positions if positions.ndim == 1 else positions[0]
        cosine, sine = self.rotary(
            temporal, dtype=torch.float32, sequence_length=temporal.numel()
        )
        route_names = frozenset(hidden.values)
        cos = RoutedTensor.from_packed(cosine, routes, routes=route_names)
        sin = RoutedTensor.from_packed(sine, routes, routes=route_names)

        projected = {
            route: self.projections[route](
                value, (cos.values[route],), (sin.values[route],)
            )
            for route, value in normalized.values.items()
        }
        query, key, value = (
            RoutedTensor(
                {route: values[index] for route, values in projected.items()}
            ).packed(routes)
            for index in range(3)
        )

        attended = self.attention(query, key, value, attention).flatten(1)
        update = RoutedTensor.from_packed(
            attended, routes, routes=route_names
        ).apply(self.outputs)
        residual = hidden.add(update)

        normalized = residual.apply(self.post_attention_norms)
        # The checkpoint's expert FFNs consume BF16 normalization results,
        # including when their surrounding accumulation is higher precision.
        normalized = RoutedTensor(
            {
                route: value.to(torch.bfloat16)
                for route, value in normalized.values.items()
            }
        )
        return normalized.apply(self.mlps), residual


class Transformer(TransformerDecoder):
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
            nn.ModuleDict(
                {
                    route: RMSNorm(config.hidden_size, config.rms_norm_eps)
                    for route in ("text", "flow")
                }
            ),
            default_route="text",
        )
        self.config = config
