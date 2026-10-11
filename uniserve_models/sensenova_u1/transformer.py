"""SenseNova U1 text and flow experts with axial rotary attention.

The backbone is a mixture-of-transformers (MoT): every layer holds separate
``text`` and ``flow`` weights for its norms, Q/K/V and output projections and
MLP, but runs one attention call over the packed sequence, so both routes
share each layer's K/V cache. ``RouteSpan`` values assign packed token ranges
to routes. Text calls take the decoder's default ``text`` route, and
``Denoiser.forward`` routes every image token through ``flow``.

Attention heads carry three rotary axes: temporal over the first half of the
head width and height and width over one quarter each. Image tokens carry
their patch row and column on the spatial axes; one-dimensional positions
leave both spatial coordinates at zero.
"""

from __future__ import annotations

import torch
from torch import nn

from uniserve.model import TransformerDecoder
from uniserve.nn.attention import (
    Attention,
    AttentionBatch,
    AxialQKVProjection,
)
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
    """Compose route-specific experts around three-axis token attention."""

    def __init__(self, config: TransformerConfig, index: int):
        super().__init__()
        hidden, dim, eps = (
            config.hidden_size,
            config.head_dim,
            config.rms_norm_eps,
        )
        self.input_norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        self.post_attention_norms = nn.ModuleDict()
        self.mlps = nn.ModuleDict()

        for route in ("text", "flow"):
            self.input_norms[route] = RMSNorm(hidden, eps)
            # Head width splits into temporal / height / width rotary
            # partitions (a half, then two quarters). Q and K each have two
            # RMS norm domains: the temporal half and the combined spatial
            # half, so height and width share one RMS denominator.
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
                config.num_attention_heads * dim,
                hidden,
                bias=config.attention_bias,
            )
            self.post_attention_norms[route] = RMSNorm(hidden, eps)
            self.mlps[route] = GatedMLP(
                hidden,
                config.intermediate_size,
                activation=config.hidden_act,
            )

        # One attention module serves both routes. Its cache name is its
        # module path under ``Model``; the denoiser calls the same module, so
        # text and denoising calls address the same K/V cache layer.
        self.attention = Attention(
            config.num_attention_heads,
            config.num_key_value_heads,
            dim,
            cache_name=f"text.backbone.layers.{index}.attention",
        )
        self.temporal_rotary = RotaryEmbedding(
            dim // 2,
            theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            partial_rotary_factor=config.partial_rotary_factor,
            keep_freq_range=True,
        )
        # Height and width share one rotary table with the spatial theta and
        # context length.
        self.spatial_rotary = RotaryEmbedding(
            dim // 4,
            theta=config.rope_theta_hw,
            max_position_embeddings=config.max_position_embeddings_hw,
            partial_rotary_factor=config.partial_rotary_factor,
            keep_freq_range=True,
        )

    def forward(
        self,
        hidden: RoutedTensor,
        residual: RoutedTensor | None,
        positions: torch.Tensor,
        attention: AttentionBatch,
        *,
        routes: tuple[RouteSpan, ...],
    ):
        """Apply one routed layer to this rank's token shard.

        ``positions`` is ``[3, tokens]`` or ``[tokens]`` for the shard and
        ``routes`` are shard-local spans. Returns ``(mlp_output, residual)``
        as per-route tensors; the next layer or the decoder's final norm adds
        them.
        """
        hidden = hidden if residual is None else hidden.add(residual)
        normalized = hidden.apply(self.input_norms)

        # Text-only callers pass one temporal axis; height and width default to
        # zero so text tokens share one spatial origin.
        if positions.ndim == 1:
            positions = torch.stack(
                (
                    positions,
                    torch.zeros_like(positions),
                    torch.zeros_like(positions),
                )
            )
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                "SenseNova positions require temporal, height and width axes"
            )

        # The default recipe's factors depend only on the positions; the
        # shard width stands in for the unused sequence length.
        length = positions.shape[1]

        names = frozenset(hidden.values)
        # Height and width use the same frequency recipe. Evaluate their
        # independent coordinates in one call, then restore the two axes.
        spatial = tuple(
            table.reshape(2, positions.shape[1], table.shape[-1])
            for table in self.spatial_rotary(
                positions[1:].reshape(-1),
                dtype=torch.float32,
                sequence_length=length,
            )
        )
        pairs = (
            self.temporal_rotary(
                positions[0], dtype=torch.float32, sequence_length=length
            ),
            (spatial[0][0], spatial[1][0]),
            (spatial[0][1], spatial[1][1]),
        )
        # Split each axis's factors with the hidden stream's route spans so
        # every expert rotates exactly its own tokens.
        cos, sin = (
            tuple(
                RoutedTensor.from_packed(pair[index], routes, routes=names)
                for pair in pairs
            )
            for index in (0, 1)
        )

        # Experts project their own tokens; Q/K/V are then repacked into
        # sequence order for the single shared attention call, and the
        # result is split again for the per-route output projections.
        projected = {
            route: self.projections[route](
                value,
                tuple(axis.values[route] for axis in cos),
                tuple(axis.values[route] for axis in sin),
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
        update = RoutedTensor.from_packed(attended, routes, routes=names).apply(
            self.outputs
        )
        residual = hidden.add(update)
        return residual.apply(self.post_attention_norms).apply(
            self.mlps
        ), residual


class Transformer(TransformerDecoder):
    """Stack routed MoT layers over a shared embedding with per-route norms.

    Calls without explicit routes take the ``text`` route.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id,
            ),
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
