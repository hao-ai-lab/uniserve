# SPDX-License-Identifier: Apache-2.0
"""UniServe-owned MiniMax H3 video VAE decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from uniserve.nn.attention import RadixAttention
from uniserve.nn.layer import LayerConfig
from uniserve.nn.linear import LinearBase, project_with_deferred_bias
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.quant.config import LinearPrecision
from uniserve.nn.vae.decoder import LatentDecoder
from uniserve.nn.vae.spatial import SpatialDecoder
from uniserve.ops.patch import unpatchify_video_tokens
from uniserve.ops.residual import (
    scaled_residual_,
    scaled_residual_layer_norm,
    scaled_residual_layer_norm_absmax,
    scaled_residual_rms_norm_,
    scaled_residual_rms_norm_absmax_,
    weighted_rms_norm,
    weighted_rms_norm_absmax,
)
from uniserve.ops.rope import qk_rms_norm_partial_rope_

__all__ = ["VideoDecoderConfig", "MiniMaxH3VideoDecoder", "MiniMaxH3VideoVAE"]


@dataclass(frozen=True, slots=True)
class VideoDecoderConfig:
    """Video VAE network, channel normalization, and reconstruction raster."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 24
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024)
    layers_per_block: int = 2
    spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    norm_num_groups: int = 32
    norm_eps: float = 1e-06
    spatial_padding_mode: str = "reflect"
    decoder_num_layers: int = 36
    decoder_num_attention_heads: int = 32
    decoder_attention_head_dim: int = 64
    decoder_num_register_tokens: int = 4
    decoder_ffn_mult: int = 4
    decoder_rope_theta: float = 100.0
    decoder_rope_dim_ratio: float = 0.75
    decoder_norm_eps: float = 1e-05
    clip_length: int = 17
    token_drop: int = 3
    latents_mean: tuple[float, ...] = (
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
    latents_std: tuple[float, ...] = (
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
    width: int = 1344
    height: int = 768
    fps: int = 24

    def __post_init__(self) -> None:
        for name in ("latents_mean", "latents_std"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != self.latent_channels:
                raise ValueError(f"H3 {name} must describe every latent channel")
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in values
            ):
                raise ValueError(f"H3 {name} must be finite")
        if any(value <= 0 for value in self.latents_std):
            raise ValueError("H3 latent standard deviations must be positive")
        for name in (
            "in_channels",
            "out_channels",
            "latent_channels",
            "layers_per_block",
            "norm_num_groups",
            "decoder_num_layers",
            "decoder_num_attention_heads",
            "decoder_attention_head_dim",
            "decoder_num_register_tokens",
            "decoder_ffn_mult",
            "clip_length",
            "width",
            "height",
            "fps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"H3 video {name} must be a positive integer")
        if (
            not isinstance(self.token_drop, int)
            or isinstance(self.token_drop, bool)
            or self.token_drop < 0
        ):
            raise ValueError("H3 video token_drop must be a non-negative integer")
        for name in (
            "block_out_channels",
            "spatial_downsample_factors",
            "temporal_downsample_factors",
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or not values
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in values
                )
            ):
                raise ValueError(f"H3 video {name} must contain positive integers")
        if (
            len(
                {
                    len(self.block_out_channels),
                    len(self.spatial_downsample_factors),
                    len(self.temporal_downsample_factors),
                }
            )
            != 1
        ):
            raise ValueError("H3 video encoder stages must have matching widths and strides")
        for name in (
            "norm_eps",
            "decoder_norm_eps",
            "decoder_rope_theta",
            "decoder_rope_dim_ratio",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"H3 video {name} must be finite and positive")
        rotary_width = self.decoder_attention_head_dim * self.decoder_rope_dim_ratio
        if (
            self.decoder_rope_dim_ratio > 1
            or rotary_width != int(rotary_width)
            or int(rotary_width) % 6
        ):
            raise ValueError("H3 video rotary width must divide into three even axes")
        if self.spatial_padding_mode not in {"reflect", "replicate", "constant"}:
            raise ValueError("unsupported H3 video spatial padding mode")
        if self.width % self.spatial_compression or self.height % self.spatial_compression:
            raise ValueError("H3 raster dimensions must align with video compression")

    @property
    def spatial_compression(self) -> int:
        return math.prod(self.spatial_downsample_factors)

    @property
    def temporal_compression(self) -> int:
        return math.prod(self.temporal_downsample_factors)


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

    def __init__(self, config: VideoDecoderConfig, layer_config: LayerConfig) -> None:
        """Assemble one scaled attention and feed-forward residual block."""

        super().__init__()
        width = config.decoder_num_attention_heads * config.decoder_attention_head_dim
        self.norm1 = nn.RMSNorm(width, eps=config.decoder_norm_eps, elementwise_affine=True)
        self.attn = _Attention(
            layer_config.child("attn"),
            width,
            config.decoder_num_attention_heads,
            config.decoder_attention_head_dim,
        )
        self.scale1 = nn.Parameter(torch.empty(width))
        self.norm2 = nn.RMSNorm(width, eps=config.decoder_norm_eps, elementwise_affine=True)
        self.ff = GatedMLP(
            width,
            width * config.decoder_ffn_mult,
            layer_config=layer_config.child("ff"),
            order="value_gate",
            bias=True,
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

    def __init__(
        self,
        config: VideoDecoderConfig,
        layer_config: LayerConfig,
        buffer_device: torch.device | str | None,
    ) -> None:
        super().__init__()
        width = config.decoder_num_attention_heads * config.decoder_attention_head_dim
        self.proj_in = LinearBase(
            config.latent_channels, width, layer_config=layer_config, prefix="proj_in"
        )
        self.register_tokens = nn.Parameter(
            torch.empty(1, config.decoder_num_register_tokens, width)
        )
        self.transformer_blocks = nn.ModuleList(
            _TransformerBlock(config, layer_config.child(f"transformer_blocks.{index}"))
            for index in range(config.decoder_num_layers)
        )
        self.norm_out = nn.LayerNorm(width, eps=config.decoder_norm_eps)
        self.proj_out = LinearBase(
            width,
            config.out_channels * config.temporal_compression * config.spatial_compression**2,
            layer_config=layer_config,
            prefix="proj_out",
        )
        self.rope = _RotaryEmbedding(
            int(config.decoder_attention_head_dim * config.decoder_rope_dim_ratio),
            config.decoder_rope_theta,
            device=buffer_device,
        )


class MiniMaxH3VideoDecoder(SpatialDecoder):
    """Checkpoint-defined 36-layer ViT decoder and latent-channel projection."""

    use_tiling = True
    tile_sample_min_height = 256
    tile_sample_min_width = 256
    tile_sample_min_overlap_height = 64
    tile_sample_min_overlap_width = 64

    def __init__(
        self,
        config: VideoDecoderConfig,
        *,
        layer_config: LayerConfig,
        parameter_device: torch.device | str = "meta",
        buffer_device: torch.device | str | None = None,
    ) -> None:
        """Allocate the checkpoint-defined decoder on its parameter and buffer devices."""

        super().__init__()
        self.config = config
        self.spatial_compression_ratio = config.spatial_compression
        self.temporal_compression_ratio = config.temporal_compression
        # The native clip leaves a leading temporal pad and overlaps latent tokens.
        self.frame_pre_padding = (-config.clip_length) % config.temporal_compression
        self.tokens_chunk_size = math.ceil(config.clip_length / config.temporal_compression)
        self.token_overlap = (-config.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(
            self.token_overlap * config.temporal_compression - self.frame_pre_padding, 0
        )
        with torch.device(parameter_device):
            self.post_quant_conv = nn.Conv3d(
                config.latent_channels, config.latent_channels, kernel_size=1
            )
            self.decoder = _VideoTransformer(config, layer_config.child("decoder"), buffer_device)

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
        suffix = positions.new_zeros((batch, self.config.decoder_num_register_tokens + 1, 3))
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
            patch_shape=(
                self.temporal_compression_ratio,
                self.spatial_compression_ratio,
                self.spatial_compression_ratio,
            ),
        )


class MiniMaxH3VideoVAE(LatentDecoder):
    """Own one resident checkpoint VAE and decode temporal segments."""

    vae: MiniMaxH3VideoDecoder
    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(self, vae: MiniMaxH3VideoDecoder, *, linear_precision: LinearPrecision) -> None:
        """Prepare one resident video decoder with fixed precision and normalization buffers."""

        super().__init__()
        self.vae = vae
        self.latent_shape = (
            1,
            vae.config.latent_channels,
            vae.tokens_chunk_size + vae.token_overlap,
            vae.config.height // vae.spatial_compression_ratio,
            vae.config.width // vae.spatial_compression_ratio,
        )
        self.linear_precision = linear_precision
        self.autocast_dtype = torch.float16 if linear_precision == "fp16" else torch.bfloat16

        # Channel-wise latent statistics invert checkpoint normalization before
        # reconstruction; pixel statistics restore the decoder's RGB domain.
        config = vae.config
        mean, std = config.latents_mean, config.latents_std
        # Keep constant values when parameter storage is deferred. The public
        # loader stages graph buffers after materializing the learned modules.
        statistics_device = "cpu" if self.device.type == "meta" else self.device
        self.register_buffer(
            "latents_mean",
            torch.tensor(mean, dtype=torch.float32, device=statistics_device).view(
                1, config.latent_channels, 1, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(std, dtype=torch.float32, device=statistics_device).view(
                1, config.latent_channels, 1, 1, 1
            ),
            persistent=False,
        )

    def _decode_segment(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode one temporal latent segment and remove its prepended overlap frames."""

        span = int(self.vae.tokens_chunk_size) + int(self.vae.token_overlap)
        clip = self.vae.decode(latents[:, :, :span])
        return clip[:, :, int(self.vae.frame_pre_padding) :].contiguous()

    def _reconstruct(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply native decoder precision and temporal crop after denormalization."""

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.device.type == "cuda",
        ):
            return self._decode_segment(latents).to(torch.float16)
