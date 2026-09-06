# SPDX-License-Identifier: Apache-2.0
"""UniServe-owned MiniMax H3 video VAE decoder."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .video_vae_fusions import (
    qk_rmsnorm_partial_rope_,
    scaled_residual_,
    scaled_residual_layernorm,
    scaled_residual_layernorm_absmax,
    scaled_residual_rmsnorm_,
    scaled_residual_rmsnorm_absmax_,
    value_first_swiglu,
    value_first_swiglu_absmax,
    video_patch_output,
    video_rmsnorm,
    video_rmsnorm_absmax,
)

__all__ = ["MiniMaxH3VideoDecoder"]


def _linear_with_deferred_bias(
    linear: nn.Module,
    hidden: torch.Tensor,
    *,
    absmax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a linear projection while returning bias separately for a fused consumer."""

    quantize = getattr(linear, "quantize_activation", None)
    prequantized = getattr(linear, "forward_prequantized", None)
    if absmax is not None and callable(quantize) and callable(prequantized):
        activation = quantize(hidden, absmax=absmax)
        return prequantized(activation, include_bias=False), getattr(linear, "bias", None)
    execute = getattr(linear, "forward_unbiased", None)
    if callable(execute):
        return execute(hidden), getattr(linear, "bias", None)
    return linear(hidden), None


class _RotaryEmbedding(nn.Module):
    """Applies partial rotary coordinates to the video decoder’s query and key heads."""

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


def _apply_rotary(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Rotate the configured prefix of each head while preserving remaining channels."""

    width = cosine.shape[-1]
    rotary, passthrough = value[..., :width], value[..., width:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cosine + rotated * sine, passthrough), dim=-1)


class _Attention(nn.Module):
    """Computes self-attention over packed spatiotemporal video tokens."""

    def __init__(self, width: int = 2048, heads: int = 32, head_dim: int = 64) -> None:
        """Build dense video self-attention projections and head-wise normalization."""

        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=False)
        self.to_q = nn.Linear(width, width, bias=True)
        self.to_k = nn.Linear(width, width, bias=True)
        self.to_v = nn.Linear(width, width, bias=True)
        self.to_out = nn.ModuleList((nn.Linear(width, width, bias=True), nn.Dropout(0.0)))

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        *,
        hidden_absmax: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run dense self-attention and optionally propagate the input activation scale."""

        batch, sequence, _ = hidden.shape
        quantize = getattr(self.to_q, "quantize_activation", None)
        q_prequantized = getattr(self.to_q, "forward_prequantized", None)
        k_prequantized = getattr(self.to_k, "forward_prequantized", None)
        v_prequantized = getattr(self.to_v, "forward_prequantized", None)
        if all(
            callable(execute)
            for execute in (quantize, q_prequantized, k_prequantized, v_prequantized)
        ):
            activation = quantize(hidden, absmax=hidden_absmax)
            query = q_prequantized(activation, include_bias=False)
            key = k_prequantized(activation, include_bias=False)
            value = v_prequantized(activation, include_bias=False)
            query_bias = getattr(self.to_q, "bias", None)
            key_bias = getattr(self.to_k, "bias", None)
            value_bias = getattr(self.to_v, "bias", None)
        else:
            query, query_bias = _linear_with_deferred_bias(self.to_q, hidden)
            key, key_bias = _linear_with_deferred_bias(self.to_k, hidden)
            value = self.to_v(hidden)
            value_bias = None
        query = query.view(batch, sequence, self.heads, self.head_dim)
        key = key.view(batch, sequence, self.heads, self.head_dim)
        value = value.view(batch, sequence, self.heads, self.head_dim)
        cosine, sine = rotary
        qk_rmsnorm_partial_rope_(
            query,
            key,
            cosine,
            sine,
            query_bias=query_bias,
            key_bias=key_bias,
            value=value if value_bias is not None else None,
            value_bias=value_bias,
        )
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        return _linear_with_deferred_bias(
            self.to_out[0],
            attended.transpose(1, 2).reshape(batch, sequence, -1),
        )


class _SwiGLU(nn.Module):
    """Applies value-first SwiGLU gating with optional activation-scale output."""

    def __init__(self, width: int = 2048, intermediate: int = 8192) -> None:
        """Build the value-first gated expansion projection."""

        super().__init__()
        self.proj = nn.Linear(width, intermediate * 2, bias=True)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        hidden_absmax: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project hidden rows and apply the decoder's value-first SwiGLU gate."""

        projected, bias = _linear_with_deferred_bias(
            self.proj,
            hidden,
            absmax=hidden_absmax,
        )
        return value_first_swiglu(projected, bias)

    def forward_with_absmax(
        self,
        hidden: torch.Tensor,
        *,
        hidden_absmax: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply SwiGLU and return its row-wise absolute maximum for FP8 projection."""

        projected, bias = _linear_with_deferred_bias(
            self.proj,
            hidden,
            absmax=hidden_absmax,
        )
        return value_first_swiglu_absmax(projected, bias)


class _FeedForward(nn.Module):
    """Projects video tokens through gated activation and output linear layers."""

    def __init__(self, width: int = 2048, intermediate: int = 8192) -> None:
        """Build the gated expansion and hidden-width output projection."""

        super().__init__()
        self.net = nn.ModuleList(
            (_SwiGLU(width, intermediate), nn.Dropout(0.0), nn.Linear(intermediate, width))
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        hidden_absmax: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the gated expansion and return any scale reusable by the next block."""

        linear = self.net[2]
        quantize = getattr(linear, "quantize_activation", None)
        prequantized = getattr(linear, "forward_prequantized", None)
        if callable(quantize) and callable(prequantized):
            hidden, absmax = self.net[0].forward_with_absmax(
                hidden,
                hidden_absmax=hidden_absmax,
            )
            hidden = self.net[1](hidden)
            activation = quantize(hidden, absmax=absmax)
            return prequantized(activation, include_bias=False), getattr(linear, "bias", None)
        hidden = self.net[0](hidden, hidden_absmax=hidden_absmax)
        hidden = self.net[1](hidden)
        return _linear_with_deferred_bias(linear, hidden)


class _TransformerBlock(nn.Module):
    """Composes video self-attention and feed-forward residual updates."""

    def __init__(self, width: int = 2048) -> None:
        """Assemble one scaled attention and feed-forward residual block."""

        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-5, elementwise_affine=True)
        self.attn = _Attention(width)
        self.scale1 = nn.Parameter(torch.empty(width))
        self.norm2 = nn.RMSNorm(width, eps=1e-5, elementwise_affine=True)
        self.ff = _FeedForward(width)
        self.scale2 = nn.Parameter(torch.empty(width))

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Normalize hidden rows and apply one attention-plus-MLP residual block."""

        quantized_attention = callable(getattr(self.attn.to_q, "quantize_activation", None))
        if quantized_attention:
            normalized, normalized_absmax = video_rmsnorm_absmax(
                hidden,
                self.norm1.weight,
                eps=float(self.norm1.eps),
            )
        else:
            normalized = video_rmsnorm(hidden, self.norm1.weight, eps=float(self.norm1.eps))
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
        quantized_feed_forward = callable(getattr(self.ff.net[0].proj, "quantize_activation", None))
        if quantized_feed_forward:
            hidden, normalized, feed_forward_absmax = scaled_residual_rmsnorm_absmax_(
                hidden,
                attention,
                self.scale1,
                self.norm2.weight,
                update_bias=attention_bias,
                eps=float(self.norm2.eps),
            )
        else:
            hidden, normalized = scaled_residual_rmsnorm_(
                hidden,
                attention,
                self.scale1,
                self.norm2.weight,
                update_bias=attention_bias,
                eps=float(self.norm2.eps),
            )
            feed_forward_absmax = None
        feed_forward, feed_forward_bias = self.ff(
            normalized,
            hidden_absmax=feed_forward_absmax,
        )
        return hidden, feed_forward, feed_forward_bias


def blend_decoded_overlap(
    previous: torch.Tensor,
    current: torch.Tensor,
    extent: int,
    dim: int,
) -> torch.Tensor:
    """Cross-fade an overlap extent between adjacent decoded tiles along one dimension."""

    extent = min(previous.shape[dim], current.shape[dim], extent)
    positions = torch.arange(extent, device=current.device, dtype=current.dtype)
    shape = [1] * current.ndim
    shape[dim] = extent
    previous_weight = (1 - positions / extent).view(shape)
    current_weight = (positions / extent).view(shape)
    previous_slice = [slice(None)] * current.ndim
    current_slice = [slice(None)] * current.ndim
    previous_slice[dim] = slice(-extent, None)
    current_slice[dim] = slice(0, extent)
    blended = (
        previous[tuple(previous_slice)] * previous_weight
        + current[tuple(current_slice)] * current_weight
    )
    if extent == current.shape[dim]:
        return blended
    remainder = [slice(None)] * current.ndim
    remainder[dim] = slice(extent, None)
    return torch.cat((blended, current[tuple(remainder)]), dim=dim)


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
        parameter_device: torch.device | str = "meta",
        buffer_device: torch.device | str | None = None,
    ) -> None:
        """Allocate the checkpoint-defined decoder on its parameter and buffer devices."""

        super().__init__()
        width = 2048
        with torch.device(parameter_device):
            self.post_quant_conv = nn.Conv3d(24, 24, kernel_size=1)
            self.decoder = nn.Module()
            self.decoder.proj_in = nn.Linear(24, width)
            self.decoder.register_tokens = nn.Parameter(torch.empty(1, 4, width))
            self.decoder.transformer_blocks = nn.ModuleList(
                _TransformerBlock(width) for _ in range(36)
            )
            self.decoder.norm_out = nn.LayerNorm(width, eps=1e-5)
            self.decoder.proj_out = nn.Linear(width, 3 * 4 * 16 * 16)
        self.decoder.rope = _RotaryEmbedding(device=buffer_device)

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
        rotary = tuple(value.to(compute_dtype) for value in rotary)
        # The first block establishes the pipelined residual/feed-forward state;
        # subsequent blocks fuse the previous residual with the next normalization.
        first = self.decoder.transformer_blocks[0]
        quantized_attention = callable(getattr(first.attn.to_q, "quantize_activation", None))
        if quantized_attention:
            normalized, normalized_absmax = video_rmsnorm_absmax(
                hidden,
                first.norm1.weight,
                eps=float(first.norm1.eps),
            )
        else:
            normalized = video_rmsnorm(hidden, first.norm1.weight, eps=float(first.norm1.eps))
            normalized_absmax = None
        hidden, feed_forward, feed_forward_bias = first.forward_normalized(
            hidden,
            normalized,
            rotary,
            normalized_absmax=normalized_absmax,
        )
        previous = first
        for block in self.decoder.transformer_blocks[1:]:
            quantized_attention = callable(getattr(block.attn.to_q, "quantize_activation", None))
            if quantized_attention:
                hidden, normalized, normalized_absmax = scaled_residual_rmsnorm_absmax_(
                    hidden,
                    feed_forward,
                    previous.scale2,
                    block.norm1.weight,
                    update_bias=feed_forward_bias,
                    eps=float(block.norm1.eps),
                )
            else:
                hidden, normalized = scaled_residual_rmsnorm_(
                    hidden,
                    feed_forward,
                    previous.scale2,
                    block.norm1.weight,
                    update_bias=feed_forward_bias,
                    eps=float(block.norm1.eps),
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
        quantized_output = callable(getattr(self.decoder.proj_out, "quantize_activation", None))
        if quantized_output:
            hidden, hidden_absmax = scaled_residual_layernorm_absmax(
                hidden,
                feed_forward,
                previous.scale2,
                self.decoder.norm_out.weight,
                self.decoder.norm_out.bias,
                update_bias=feed_forward_bias,
                eps=float(self.decoder.norm_out.eps),
            )
            hidden, output_bias = _linear_with_deferred_bias(
                self.decoder.proj_out,
                hidden,
                absmax=hidden_absmax,
            )
        else:
            hidden = scaled_residual_layernorm(
                hidden,
                feed_forward,
                previous.scale2,
                self.decoder.norm_out.weight,
                self.decoder.norm_out.bias,
                update_bias=feed_forward_bias,
                eps=float(self.decoder.norm_out.eps),
            )
            hidden, output_bias = _linear_with_deferred_bias(
                self.decoder.proj_out,
                hidden,
            )
        return video_patch_output(
            hidden,
            output_bias,
            frames=frames,
            height=height,
            width=width,
        )
