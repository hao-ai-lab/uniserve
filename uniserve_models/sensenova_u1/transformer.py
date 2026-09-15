"""SenseNova U1 text and flow experts with axial rotary attention."""

from __future__ import annotations

import torch
from torch import nn

from uniserve.model import TransformerDecoder
from uniserve.nn.attention import (
    Attention,
    AttentionInput,
    AxialQKVProjection,
    DenseInput,
    PagedInput,
    SegmentedInput,
)
from uniserve.nn.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.norm import RMSNorm
from uniserve.nn.rope import (
    DynamicScaling,
    LongRoPEScaling,
    RotaryEmbedding,
)
from uniserve.nn.routing import RoutedTensor, RouteSpan

from .config import TransformerConfig


class TransformerLayer(nn.Module):
    """Compose route-specific experts around three-axis token attention."""

    def __init__(self, config: TransformerConfig, index: int):
        super().__init__()
        hidden, dim, eps = config.hidden_size, config.head_dim, config.rms_norm_eps
        self.input_norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        self.post_attention_norms = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        activation = (
            "silu"
            if config.hidden_act in {"silu", "swish", "silu_and_mul", "swiglu"}
            else "gelu"
            if config.hidden_act in {"gelu", "gelu_and_mul", "geglu"}
            else "gelu_pytorch_tanh"
        )
        for route in ("text", "flow"):
            self.input_norms[route] = RMSNorm(hidden, eps)
            self.projections[route] = AxialQKVProjection(
                QKVParallelLinear(
                    hidden,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    dim,
                    bias=config.attention_bias,
                ),
                nn.ModuleList((RMSNorm(dim // 2, eps), RMSNorm(dim // 2, eps))),
                nn.ModuleList((RMSNorm(dim // 2, eps), RMSNorm(dim // 2, eps))),
                axis_dims=(dim // 2, dim // 4, dim // 4),
                rotations=("split",) * 3,
            )
            self.outputs[route] = RowParallelLinear(
                config.num_attention_heads * dim, hidden, bias=config.attention_bias
            )
            self.post_attention_norms[route] = RMSNorm(hidden, eps)
            self.mlps[route] = GatedMLP(hidden, config.intermediate_size, activation=activation)
        self.attention = Attention(
            config.num_attention_heads,
            config.num_key_value_heads,
            dim,
            cache_name=f"text.backbone.layers.{index}.attention",
        )
        self.window = (
            config.sliding_window if config.layer_types[index] == "sliding_attention" else None
        )
        self.temporal_rotary = RotaryEmbedding(
            dim // 2,
            theta=config.rope_theta,
            scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            partial_rotary_factor=config.partial_rotary_factor,
            keep_freq_range=True,
        )
        self.spatial_rotary = RotaryEmbedding(
            dim // 4,
            theta=config.rope_theta_hw,
            scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings_hw,
            partial_rotary_factor=config.partial_rotary_factor,
            keep_freq_range=True,
        )

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
        if positions.ndim == 1:
            positions = torch.stack(
                (positions, torch.zeros_like(positions), torch.zeros_like(positions))
            )
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError("SenseNova positions require temporal, height and width axes")
        dynamic = isinstance(self.temporal_rotary.scaling, (DynamicScaling, LongRoPEScaling))
        if not dynamic:
            length = positions.shape[1]
        elif isinstance(attention, (PagedInput, SegmentedInput)):
            if attention.queries.host is None or attention.prefixes.host is None:
                raise ValueError("dynamic rotary scaling requires exact host sequence lengths")
            length = max(
                (
                    query + prefix
                    for query, prefix in zip(
                        attention.queries.host, attention.prefixes.host, strict=True
                    )
                ),
                default=0,
            )
        else:
            length = (
                positions.shape[1]
                if isinstance(attention, DenseInput)
                else attention.queries.maximum
            )
        if length is None:
            raise ValueError("dynamic rotary scaling requires exact host sequence lengths")
        names = frozenset(hidden.values)
        # Height and width use the same frequency recipe. Evaluate their
        # independent coordinates in one call, then restore the two axes.
        spatial = tuple(
            table.reshape(2, positions.shape[1], table.shape[-1])
            for table in self.spatial_rotary(
                positions[1:].reshape(-1), dtype=torch.float32, sequence_length=length
            )
        )
        pairs = (
            self.temporal_rotary(positions[0], dtype=torch.float32, sequence_length=length),
            (spatial[0][0], spatial[1][0]),
            (spatial[0][1], spatial[1][1]),
        )
        cos, sin = (
            tuple(RoutedTensor.from_packed(pair[index], routes, routes=names) for pair in pairs)
            for index in (0, 1)
        )
        projected = {
            route: self.projections[route](
                value,
                tuple(axis.values[route] for axis in cos),
                tuple(axis.values[route] for axis in sin),
            )
            for route, value in normalized.values.items()
        }
        query, key, value = (
            RoutedTensor({route: values[index] for route, values in projected.items()}).packed(
                routes
            )
            for index in range(3)
        )
        attended = self.attention(query, key, value, attention).flatten(1)
        update = RoutedTensor.from_packed(attended, routes, routes=names).apply(self.outputs)
        residual = hidden.add(update)
        return residual.apply(self.post_attention_norms).apply(self.mlps), residual


class Transformer(TransformerDecoder):
    def __init__(self, config: TransformerConfig):
        super().__init__(
            VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
            ),
            nn.ModuleDict(
                (str(index), TransformerLayer(config, index))
                for index in range(config.num_hidden_layers)
            ),
            nn.ModuleDict(
                (route, RMSNorm(config.hidden_size, config.rms_norm_eps))
                for route in ("text", "flow")
            ),
            default_route="text",
        )
        self.config = config
