# SPDX-License-Identifier: Apache-2.0
"""UniServe-owned H3 causal image encoding and temporal video reconstruction."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from ...media.codec import blend_decoded_overlap, video_segment_rgb
from ...nn.attention import RadixAttention
from ...nn.layer import LayerConfig
from ...nn.linear import LinearBase, project_with_deferred_bias
from ...nn.mlp import GatedMLP
from ...nn.quant.config import LinearPrecision
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
from .image_vae import H3ImageEncoder
from .layout import PROFILE_HEIGHT, PROFILE_WIDTH, H3ComputeInputs, H3Tensors
from .packing import unpatchify_video_into

__all__ = ["H3VideoAssembler", "MiniMaxH3VideoDecoder", "MiniMaxH3VideoVAE"]


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
        self.attention = RadixAttention(heads, heads, head_dim)
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
    """Checkpoint VAE weights: causal image encoder and 36-layer ViT decoder."""

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
            self.encoder = H3ImageEncoder()
            self.quant_conv = nn.Conv3d(48, 48, kernel_size=1)
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


class MiniMaxH3VideoVAE(nn.Module):
    """Own normalized single-image encoding and temporal-segment decoding."""

    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(self, vae: MiniMaxH3VideoDecoder, *, linear_precision: LinearPrecision) -> None:
        """Prepare one resident video decoder with fixed precision and normalization buffers."""

        super().__init__()
        self.vae = vae
        self.linear_precision = linear_precision
        self.autocast_dtype = torch.float16 if linear_precision == "fp16" else torch.bfloat16

        # Channel-wise latent statistics invert checkpoint normalization before
        # reconstruction; pixel statistics restore the decoder's RGB domain.
        mean = (
            0.858090341091156,
            -0.9606591463088989,
            1.0661640167236328,
            -0.5090325474739075,
            -0.2727581858634949,
            -1.3675414323806763,
            -0.2553254961967468,
            -0.26907554268836975,
            -0.5376840829849243,
            -0.0464097298681736,
            0.6657370328903198,
            0.19690127670764923,
            -0.5460608005523682,
            -0.4035342037677765,
            -0.23683024942874908,
            0.25928452610969543,
            -0.30133944749832153,
            0.211341992020607,
            -1.1206848621368408,
            0.3581933379173279,
            -0.04225143790245056,
            0.2604829967021942,
            0.22864092886447906,
            0.7056031823158264,
        )
        std = (
            1.2223774194717407,
            1.2767263650894165,
            1.6831774711608887,
            1.7549455165863037,
            1.5636216402053833,
            2.194143533706665,
            0.9653137922286987,
            1.0569885969161987,
            0.841948926448822,
            0.7729952931404114,
            1.8955937623977661,
            0.946841835975647,
            0.7996809482574463,
            0.44988900423049927,
            0.7197399735450745,
            0.6936293244361877,
            2.961095094680786,
            2.7694199085235596,
            3.0496184825897217,
            2.1088054180145264,
            3.276226282119751,
            3.1627357006073,
            2.2816812992095947,
            2.6127843856811523,
        )
        if len(mean) != 24 or len(std) != 24:
            raise ValueError("MiniMax H3 video VAE must declare 24-channel latent statistics")
        self.register_buffer(
            "latents_mean",
            torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 24, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 24, 1, 1, 1),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        """Return the device that owns the decoder's learned parameters."""

        return next(self.vae.parameters()).device

    @torch.inference_mode()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one HWC uint8 RGB image to normalized [1,24,1,H/16,W/16].

        The causal CNN runs in FP32, with spatial tiling identical to decoding.
        Posterior sampling uses the released independent CPU seed 42 and FP16
        round-trip before channel normalization, not the target-noise RNG.
        """

        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
            raise ValueError("H3 reference image must be HWC uint8 RGB")
        height, width = image.shape[:2]
        if min(height, width) < 32 or height % 32 or width % 32:
            raise ValueError("H3 reference image dimensions must be divisible by 32")
        pixels = image.permute(2, 0, 1)[None, :, None].to(self.device, torch.float32) / 255.0
        mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        pixels = (pixels - mean) / std
        with torch.autocast(device_type=self.device.type, enabled=False):
            if self.vae.use_tiling:
                ys, hs, yo = self.vae._split_tiles(
                    height, self.vae.tile_sample_min_height, self.vae.tile_sample_min_overlap_height
                )
                xs, ws, xo = self.vae._split_tiles(
                    width, self.vae.tile_sample_min_width, self.vae.tile_sample_min_overlap_width
                )
                tiles = [
                    [
                        self.vae.quant_conv(self.vae.encoder(pixels[..., y : y + h, x : x + w]))
                        for x, w in zip(xs, ws, strict=True)
                    ]
                    for y, h in zip(ys, hs, strict=True)
                ]
                moments = self.vae._stitch_tiles(
                    tiles, [v // 16 for v in yo], [v // 16 for v in xo]
                )
            else:
                moments = self.vae.quant_conv(self.vae.encoder(pixels))
        expected = (1, 48, 1, height // 16, width // 16)
        if tuple(moments.shape) != expected or moments.dtype != torch.float32:
            raise ValueError(f"image posterior must be FP32 with shape {expected}")
        mean, logvar = moments.chunk(2, dim=1)
        noise = torch.randn(
            mean.shape, generator=torch.Generator("cpu").manual_seed(42), dtype=mean.dtype
        ).to(mean.device)
        sampled = (mean + (0.5 * logvar.clamp(-30, 20)).exp() * noise).half().float()
        return (sampled - self.latents_mean) / self.latents_std

    def _decode_segment(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode one temporal latent segment and remove its prepended overlap frames."""

        span = int(self.vae.tokens_chunk_size) + int(self.vae.token_overlap)
        clip = self._decode_spatial_tiles(latents[:, :, :span])
        return clip[:, :, int(self.vae.frame_pre_padding) :].contiguous()

    def _decode_spatial_tiles(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode one latent clip directly or tile and blend it across spatial overlap regions."""

        if not bool(self.vae.use_tiling):
            return self.vae(self.vae.post_quant_conv(latents))

        ratio = int(self.vae.spatial_compression_ratio)
        height = int(latents.shape[-2]) * ratio
        width = int(latents.shape[-1]) * ratio
        y_indices, y_lengths, y_overlaps = self.vae._split_tiles(
            height,
            int(self.vae.tile_sample_min_height),
            int(self.vae.tile_sample_min_overlap_height),
        )
        x_indices, x_lengths, x_overlaps = self.vae._split_tiles(
            width,
            int(self.vae.tile_sample_min_width),
            int(self.vae.tile_sample_min_overlap_width),
        )
        # Decode all spatial tiles as one batch, then blend them back into the
        # full-resolution temporal segment using the decoder's overlap contract.
        tiles = torch.cat(
            tuple(
                latents[
                    ...,
                    y_pos // ratio : y_pos // ratio + y_length // ratio,
                    x_pos // ratio : x_pos // ratio + x_length // ratio,
                ]
                for y_pos, y_length in zip(y_indices, y_lengths, strict=True)
                for x_pos, x_length in zip(x_indices, x_lengths, strict=True)
            ),
            dim=0,
        )
        decoded = self.vae(self.vae.post_quant_conv(tiles))
        flat_tiles = decoded.split(1, dim=0)
        columns = len(x_indices)
        rows = [
            list(flat_tiles[start : start + columns])
            for start in range(0, len(flat_tiles), columns)
        ]
        return self.vae._stitch_tiles(rows, y_overlaps, x_overlaps)

    def forward(
        self,
        normalized_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Denormalize one latent segment and decode it through the spatial tiling path."""

        if normalized_latents.shape != (1, 24, 7, 48, 84):
            raise ValueError("an H3 video decode unit must have shape [1, 24, 7, 48, 84]")
        latents = normalized_latents.to(device=self.device, dtype=torch.float32)
        latents = latents * self.latents_std + self.latents_mean
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.device.type == "cuda" and self.linear_precision != "fp32",
        ):
            decoded = self._decode_segment(latents)
            return decoded if self.linear_precision == "fp32" else decoded.to(torch.float16)

    def prepare_input(
        self,
        execution: H3ComputeInputs,
        latents: torch.Tensor,
        cursor: int,
        max_units: int,
        rank: int,
    ) -> torch.Tensor:
        """Pack this decoder rank's temporal latent window into VAE layout."""

        layout, scratch = execution.layout, execution.media
        if rank >= max_units or cursor + max_units > layout.video_reconstruction_units:
            raise ValueError("video decode exceeds its temporal range")
        if tuple(latents.shape) != (int(layout.packed.video_indices.numel()), 96):
            raise ValueError("video decoder requires complete final latent rows")
        start = (cursor + rank) * 5 * 24 * 42
        selected = scratch.video_raster_order[start : start + 7 * 24 * 42]
        torch.index_select(latents, 0, selected, out=scratch.reconstruction_rows)
        unpatchify_video_into(
            scratch.reconstruction_rows,
            scratch.video_input,
            frames=7,
            height=48,
            width=84,
        )
        return scratch.video_input


class H3VideoAssembler(nn.Module):
    """Own H3 overlap state transforms and checkpoint pixel normalization."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        for name, values in (
            ("pixel_mean", (0.485, 0.456, 0.406)),
            ("pixel_std", (0.229, 0.224, 0.225)),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1),
                persistent=False,
            )

    def assemble(
        self,
        slot: H3Tensors,
        execution: H3ComputeInputs,
        segments: torch.Tensor,
        start_unit: int,
        unit_count: int,
    ) -> torch.Tensor:
        """Blend ordered decoder windows and convert them into RGB byte frames."""

        layout, scratch = execution.layout, execution.media
        expected = (unit_count, 1, 3, 25, PROFILE_HEIGHT, PROFILE_WIDTH)
        if tuple(segments.shape) != expected:
            raise ValueError("video assembly requires complete decoded segments")
        if start_unit + unit_count > layout.video_reconstruction_units:
            raise ValueError("video assembly exceeds its temporal extent")
        overlap = None if start_unit == 0 else slot.video_overlap
        frame_start = 0
        for offset in range(unit_count):
            unit = start_unit + offset
            rgb, overlap = video_segment_rgb(
                segments[offset],
                overlap,
                body_frames=(
                    MiniMaxH3VideoDecoder.tokens_chunk_size
                    * MiniMaxH3VideoDecoder.temporal_compression_ratio
                    - MiniMaxH3VideoDecoder.frame_pre_padding
                ),
                overlap_frames=MiniMaxH3VideoDecoder.frame_overlap,
                padding_frames=MiniMaxH3VideoDecoder.frame_pre_padding,
                pixel_mean=self.pixel_mean,
                pixel_std=self.pixel_std,
                final_unit=unit + 1 == layout.video_reconstruction_units,
            )
            valid_frames = layout.reconstruction_unit_frames[unit]
            if int(rgb.shape[0]) != valid_frames:
                raise RuntimeError("H3 video decoder returned an unexpected frame count")
            scratch.rgb_round[frame_start : frame_start + valid_frames].copy_(rgb)
            frame_start += valid_frames
        assert overlap is not None
        slot.video_overlap.copy_(overlap)
        return scratch.rgb_round[:frame_start]

    @torch.inference_mode()
    def warmup(self) -> None:
        """Compile the output pixel transform without retaining request state."""

        segment = torch.zeros(
            (1, 3, 25, PROFILE_HEIGHT, PROFILE_WIDTH),
            dtype=torch.float16,
            device=self.pixel_mean.device,
        )
        video_segment_rgb(
            segment,
            None,
            body_frames=(
                MiniMaxH3VideoDecoder.tokens_chunk_size
                * MiniMaxH3VideoDecoder.temporal_compression_ratio
                - MiniMaxH3VideoDecoder.frame_pre_padding
            ),
            overlap_frames=MiniMaxH3VideoDecoder.frame_overlap,
            padding_frames=MiniMaxH3VideoDecoder.frame_pre_padding,
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
            final_unit=False,
        )
