# SPDX-License-Identifier: Apache-2.0
"""UniServe-owned MiniMax H3 video VAE decoder."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from ...backends.attention.torch_sdpa import TorchSDPAAttentionBackend
from ...media.codec import blend_decoded_overlap
from ...nn.attention import RadixAttention
from ...nn.layer import LayerConfig
from ...nn.linear import LinearBase, project_with_deferred_bias
from ...nn.mlp import GatedMLP
from ...ops.patch import unpatchify_video_tokens
from ...ops.residual import (
    scaled_residual_,
    scaled_residual_layer_norm,
    scaled_residual_layer_norm_absmax,
    scaled_residual_rms_norm_,
    scaled_residual_rms_norm_absmax_,
    weighted_rms_norm,
    weighted_rms_norm_absmax,
)
from ...ops.rope import qk_rms_norm_partial_rope_

__all__ = ["MiniMaxH3VideoDecoder"]


class _RotaryEmbedding(nn.Module):
    """Applies partial rotary coordinates to the video decoder’s query and key heads."""

    inv_freq: torch.Tensor

    def __init__(
        self,
        dimension: int = 48,
        theta: float = 100.0,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        """Precompute partial three-axis rotary frequencies for flattened video patches."""

        super().__init__()
        if dimension % 6:
            raise ValueError("H3 video VAE rotary width must divide three axes")
        inverse = 1.0 / theta ** torch.arange(
            0,
            1,
            6 / dimension,
            dtype=torch.float32,
            device=device,
        )
        self.register_buffer("inv_freq", inverse, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build temporal-height-width rotary tables for flattened video patches."""

        angles = 2.0 * math.pi * positions[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class _Attention(nn.Module):
    """Computes self-attention over packed spatiotemporal video tokens."""

    def __init__(
        self, layer_config: LayerConfig, width: int = 2048, heads: int = 32, head_dim: int = 64
    ) -> None:
        """Build dense video self-attention projections and head-wise normalization."""

        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.attention = RadixAttention(
            heads, heads, head_dim, dense_provider=TorchSDPAAttentionBackend()
        )
        self.to_q = LinearBase(width, width, layer_config=layer_config, prefix="to_q")
        self.to_k = LinearBase(width, width, layer_config=layer_config, prefix="to_k")
        self.to_v = LinearBase(width, width, layer_config=layer_config, prefix="to_v")
        self.to_out = nn.ModuleList(
            (LinearBase(width, width, layer_config=layer_config, prefix="to_out.0"),)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        *,
        hidden_absmax: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run dense self-attention and optionally propagate the input activation scale."""

        batch, sequence, _ = hidden.shape
        if self.to_q.quant_method.is_quantized:
            activation = self.to_q.prepare_input(hidden, absmax=hidden_absmax)
            query = self.to_q.forward_prepared(activation, include_bias=False)
            key = self.to_k.forward_prepared(activation, include_bias=False)
            value = self.to_v.forward_prepared(activation, include_bias=False)
            query_bias: torch.Tensor | None = self.to_q.bias
            key_bias: torch.Tensor | None = self.to_k.bias
            value_bias = self.to_v.bias
        else:
            query, query_bias = project_with_deferred_bias(self.to_q, hidden)
            key, key_bias = project_with_deferred_bias(self.to_k, hidden)
            value = self.to_v(hidden)
            value_bias = None
        query = query.view(batch, sequence, self.heads, self.head_dim)
        key = key.view(batch, sequence, self.heads, self.head_dim)
        value = value.view(batch, sequence, self.heads, self.head_dim)
        cosine, sine = rotary
        qk_rms_norm_partial_rope_(
            query,
            key,
            cosine,
            sine,
            query_bias=query_bias,
            key_bias=key_bias,
            value=value if value_bias is not None else None,
            value_bias=value_bias,
        )
        attended = self.attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            None,
            causal=False,
        )
        return project_with_deferred_bias(
            self.to_out[0],
            attended.transpose(1, 2).reshape(batch, sequence, -1),
        )


class _TransformerBlock(nn.Module):
    """Composes video self-attention and feed-forward residual updates."""

    def __init__(self, layer_config: LayerConfig, width: int = 2048) -> None:
        """Assemble one scaled attention and feed-forward residual block."""

        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-5, elementwise_affine=True)
        self.attn = _Attention(layer_config.child("attn"), width)
        self.scale1 = nn.Parameter(torch.empty(width))
        self.norm2 = nn.RMSNorm(width, eps=1e-5, elementwise_affine=True)
        self.ff = GatedMLP(
            width, 8192, layer_config=layer_config.child("ff"), order="value_gate", bias=True
        )
        self.scale2 = nn.Parameter(torch.empty(width))

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Normalize hidden rows and apply one attention-plus-MLP residual block."""

        quantized_attention = self.attn.to_q.quant_method.is_quantized
        if quantized_attention:
            normalized, normalized_absmax = weighted_rms_norm_absmax(
                hidden,
                self.norm1.weight,
                eps=cast(float, self.norm1.eps),
            )
        else:
            normalized = weighted_rms_norm(
                hidden, self.norm1.weight, eps=cast(float, self.norm1.eps)
            )
            normalized_absmax = None
        hidden, feed_forward, feed_forward_bias = self.forward_normalized(
            hidden,
            normalized,
            rotary,
            normalized_absmax=normalized_absmax,
        )
        return scaled_residual_(
            hidden,
            feed_forward,
            self.scale2,
            update_bias=feed_forward_bias,
        )

    def forward_normalized(
        self,
        hidden: torch.Tensor,
        normalized: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        *,
        normalized_absmax: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Advance a block from caller-normalized input and prepare the next normalized view."""

        attention, attention_bias = self.attn(
            normalized,
            rotary,
            hidden_absmax=normalized_absmax,
        )
        quantized_feed_forward = self.ff.gate_up_proj.quant_method.is_quantized
        if quantized_feed_forward:
            hidden, normalized, feed_forward_absmax = scaled_residual_rms_norm_absmax_(
                hidden,
                attention,
                self.scale1,
                self.norm2.weight,
                update_bias=attention_bias,
                eps=cast(float, self.norm2.eps),
            )
        else:
            hidden, normalized = scaled_residual_rms_norm_(
                hidden,
                attention,
                self.scale1,
                self.norm2.weight,
                update_bias=attention_bias,
                eps=cast(float, self.norm2.eps),
            )
            feed_forward_absmax = None
        feed_forward, feed_forward_bias = self.ff.forward_deferred(
            normalized,
            input_absmax=feed_forward_absmax,
        )
        return hidden, feed_forward, feed_forward_bias


class _VideoTransformer(nn.Module):
    """Checkpoint video token projections and residual transformer stack."""

    def __init__(self, layer_config: LayerConfig, buffer_device: torch.device | str | None) -> None:
        super().__init__()
        width = 2048
        self.proj_in = LinearBase(24, width, layer_config=layer_config, prefix="proj_in")
        self.register_tokens = nn.Parameter(torch.empty(1, 4, width))
        self.transformer_blocks = nn.ModuleList(
            _TransformerBlock(layer_config.child(f"transformer_blocks.{index}"), width)
            for index in range(36)
        )
        self.norm_out = nn.LayerNorm(width, eps=1e-5)
        self.proj_out = LinearBase(
            width, 3 * 4 * 16 * 16, layer_config=layer_config, prefix="proj_out"
        )
        self.rope = _RotaryEmbedding(device=buffer_device)


class MiniMaxH3VideoDecoder(nn.Module):
    """Checkpoint-defined 36-layer ViT decoder and latent-channel projection."""

    spatial_compression_ratio = 16
    temporal_compression_ratio = 4
    tokens_chunk_size = 5
    token_overlap = 2
    frame_pre_padding = 3
    frame_overlap = 5
    use_tiling = True
    tile_sample_min_height = 256
    tile_sample_min_width = 256
    tile_sample_min_overlap_height = 64
    tile_sample_min_overlap_width = 64

    def __init__(
        self,
        *,
        layer_config: LayerConfig,
        parameter_device: torch.device | str = "meta",
        buffer_device: torch.device | str | None = None,
    ) -> None:
        """Allocate the checkpoint-defined decoder on its parameter and buffer devices."""

        super().__init__()
        with torch.device(parameter_device):
            self.post_quant_conv = nn.Conv3d(24, 24, kernel_size=1)
            self.decoder = _VideoTransformer(layer_config.child("decoder"), buffer_device)

    @staticmethod
    def _split_tiles(
        length: int,
        tile_size: int,
        minimum_overlap: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """Partition one spatial extent into aligned tiles with bounded pairwise overlap."""

        if tile_size >= length:
            return [0], [length], []
        tile_count = math.ceil(length / tile_size)
        while tile_size * tile_count - minimum_overlap * (tile_count - 1) < length:
            tile_count += 1
        overlaps = [minimum_overlap] * (tile_count - 1)
        remaining = tile_size * tile_count - sum(overlaps) - length
        for index in range(remaining // 16):
            overlaps[index % (tile_count - 1)] += 16
        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1] + tile_size - overlap)
        return starts, [tile_size] * tile_count, overlaps

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        """Blend a two-dimensional tile grid and concatenate it into one decoded frame tensor."""

        assembled_rows: list[torch.Tensor] = []
        for row_index, row in enumerate(tiles):
            assembled: list[torch.Tensor] = []
            for column_index, tile in enumerate(row):
                if row_index:
                    tile = blend_decoded_overlap(
                        tiles[row_index - 1][column_index],
                        tile,
                        height_overlaps[row_index - 1],
                        -2,
                    )
                if column_index:
                    tile = blend_decoded_overlap(
                        row[column_index - 1],
                        tile,
                        width_overlaps[column_index - 1],
                        -1,
                    )
                if row_index + 1 < len(tiles):
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index + 1 < len(row):
                    tile = tile[..., :, : -width_overlaps[column_index]]
                assembled.append(tile)
            assembled_rows.append(torch.cat(assembled, dim=-1))
        return torch.cat(assembled_rows, dim=-2)

    def forward(self, projected_latents: torch.Tensor) -> torch.Tensor:
        """Decode projected `[B, 24, T, H, W]` latents into full-resolution RGB tensors."""

        batch, channels, frames, height, width = projected_latents.shape
        hidden = projected_latents.permute(0, 2, 3, 4, 1).reshape(
            batch,
            frames * height * width,
            channels,
        )
        hidden = self.decoder.proj_in(hidden)
        compute_dtype = hidden.dtype

        # Register and sentinel tokens participate in attention but are excluded
        # from the final spatiotemporal patch projection.
        registers = self.decoder.register_tokens.expand(batch, -1, -1)
        hidden = torch.cat((hidden, registers, torch.zeros_like(hidden[:, :1])), dim=1)
        axes = tuple(
            2.0 * (torch.arange(0.5, size, device=hidden.device, dtype=torch.float32) / size) - 1.0
            for size in (frames, height, width)
        )
        positions = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, 2)
        positions = positions.unsqueeze(0).expand(batch, -1, -1)
        suffix = positions.new_zeros((batch, 5, 3))
        rotary = self.decoder.rope(torch.cat((positions, suffix), dim=1))
        rotary = (rotary[0].to(compute_dtype), rotary[1].to(compute_dtype))
        # The first block establishes the pipelined residual/feed-forward state;
        # subsequent blocks fuse the previous residual with the next normalization.
        first = cast(_TransformerBlock, self.decoder.transformer_blocks[0])
        quantized_attention = first.attn.to_q.quant_method.is_quantized
        if quantized_attention:
            normalized, normalized_absmax = weighted_rms_norm_absmax(
                hidden,
                first.norm1.weight,
                eps=cast(float, first.norm1.eps),
            )
        else:
            normalized = weighted_rms_norm(
                hidden, first.norm1.weight, eps=cast(float, first.norm1.eps)
            )
            normalized_absmax = None
        hidden, feed_forward, feed_forward_bias = first.forward_normalized(
            hidden,
            normalized,
            rotary,
            normalized_absmax=normalized_absmax,
        )
        previous = first
        for module in self.decoder.transformer_blocks[1:]:
            block = cast(_TransformerBlock, module)
            quantized_attention = block.attn.to_q.quant_method.is_quantized
            if quantized_attention:
                hidden, normalized, normalized_absmax = scaled_residual_rms_norm_absmax_(
                    hidden,
                    feed_forward,
                    previous.scale2,
                    block.norm1.weight,
                    update_bias=feed_forward_bias,
                    eps=cast(float, block.norm1.eps),
                )
            else:
                hidden, normalized = scaled_residual_rms_norm_(
                    hidden,
                    feed_forward,
                    previous.scale2,
                    block.norm1.weight,
                    update_bias=feed_forward_bias,
                    eps=cast(float, block.norm1.eps),
                )
                normalized_absmax = None
            hidden, feed_forward, feed_forward_bias = block.forward_normalized(
                hidden,
                normalized,
                rotary,
                normalized_absmax=normalized_absmax,
            )
            previous = block
        # Final layer normalization and projection preserve the selected linear
        # method's activation-scale path before patch rows become RGB volumes.
        quantized_output = self.decoder.proj_out.quant_method.is_quantized
        if quantized_output:
            hidden, hidden_absmax = scaled_residual_layer_norm_absmax(
                hidden,
                feed_forward,
                previous.scale2,
                self.decoder.norm_out.weight,
                self.decoder.norm_out.bias,
                update_bias=feed_forward_bias,
                eps=float(self.decoder.norm_out.eps),
            )
            hidden, output_bias = project_with_deferred_bias(
                self.decoder.proj_out,
                hidden,
                absmax=hidden_absmax,
            )
        else:
            hidden = scaled_residual_layer_norm(
                hidden,
                feed_forward,
                previous.scale2,
                self.decoder.norm_out.weight,
                self.decoder.norm_out.bias,
                update_bias=feed_forward_bias,
                eps=float(self.decoder.norm_out.eps),
            )
            hidden, output_bias = project_with_deferred_bias(
                self.decoder.proj_out,
                hidden,
            )
        return unpatchify_video_tokens(
            hidden,
            output_bias,
            grid_shape=(frames, height, width),
            patch_shape=(4, 16, 16),
        )
