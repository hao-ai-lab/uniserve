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

from uniserve.model.mot import MoTDecoder, MoTLayer
from uniserve.nn.attention import Attention, AxialQKVProjection
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
    """Compose route-specific experts around three-axis token attention."""

    def __init__(self, config: TransformerConfig, index: int):
        hidden, dim, eps = (
            config.hidden_size,
            config.head_dim,
            config.rms_norm_eps,
        )

        def projection():
            # Head width splits into temporal / height / width rotary
            # partitions (a half, then two quarters). Q and K each have two
            # RMS norm domains: the temporal half and the combined spatial
            # half, so height and width share one RMS denominator.
            return AxialQKVProjection(
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

        super().__init__(
            # One attention module serves both routes. Its cache name is its
            # module path under ``Model``; the denoiser calls the same module,
            # so text and denoising calls address the same K/V cache layer.
            attention=Attention(
                config.num_attention_heads,
                config.num_key_value_heads,
                dim,
                cache_name=f"text.backbone.layers.{index}.attention",
            ),
            hidden_size=hidden,
            eps=eps,
            projection=projection,
            output=lambda: RowParallelLinear(
                config.num_attention_heads * dim,
                hidden,
                bias=config.attention_bias,
            ),
            mlp=lambda: GatedMLP(
                hidden,
                config.intermediate_size,
                activation=config.hidden_act,
            ),
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

    def position_embeddings(self, positions):
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

        # Height and width use the same frequency recipe. Evaluate their
        # independent coordinates in one call, then restore the two axes.
        spatial = tuple(
            table.reshape(2, length, table.shape[-1])
            for table in self.spatial_rotary(
                positions[1:].reshape(-1),
                dtype=torch.float32,
                sequence_length=length,
            )
        )
        temporal = self.temporal_rotary(
            positions[0], dtype=torch.float32, sequence_length=length
        )
        return (
            (temporal[0], spatial[0][0], spatial[0][1]),
            (temporal[1], spatial[1][0], spatial[1][1]),
        )


class Transformer(MoTDecoder):
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
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.config = config
