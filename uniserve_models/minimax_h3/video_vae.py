# SPDX-License-Identifier: Apache-2.0
"""H3 video reconstruction with spatial tiling and fused residual arithmetic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from uniserve.media import image
from uniserve.nn import functional
from uniserve.nn.attention import Attention, DenseInput
from uniserve.nn.functional import (
    qk_bias_rms_norm_rope_,
    scaled_residual_,
    scaled_residual_layer_norm,
    scaled_residual_layer_norm_absmax,
    scaled_residual_rms_norm_,
    scaled_residual_rms_norm_absmax_,
    swiglu,
    swiglu_absmax,
    unpatchify_video_tokens,
    weighted_rms_norm,
    weighted_rms_norm_absmax,
)
from uniserve.nn.linear import Linear, QKVParallelLinear, RowParallelLinear
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.rope import RotaryEmbedding
from uniserve.nn.vae import LatentDecoder, SpatialDecoder
from uniserve.quantization import QuantizedTensor


@dataclass(frozen=True, slots=True)
class Config:
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

    def __post_init__(self) -> None:
        for name in ("latents_mean", "latents_std"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or len(values) != self.latent_channels
            ):
                raise ValueError(
                    f"H3 {name} must describe every latent channel"
                )
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
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"H3 video {name} must be a positive integer")
        if (
            not isinstance(self.token_drop, int)
            or isinstance(self.token_drop, bool)
            or self.token_drop < 0
        ):
            raise ValueError(
                "H3 video token_drop must be a non-negative integer"
            )
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
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value <= 0
                    for value in values
                )
            ):
                raise ValueError(
                    f"H3 video {name} must contain positive integers"
                )
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
            raise ValueError(
                "H3 video encoder stages must have matching widths and strides"
            )
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
        rotary_width = (
            self.decoder_attention_head_dim * self.decoder_rope_dim_ratio
        )
        if (
            self.decoder_rope_dim_ratio > 1
            or rotary_width != int(rotary_width)
            or int(rotary_width) % 6
        ):
            raise ValueError(
                "H3 video rotary width must divide into three even axes"
            )
        if self.spatial_padding_mode not in {
            "reflect",
            "replicate",
            "constant",
        }:
            raise ValueError("unsupported H3 video spatial padding mode")

    @property
    def spatial_compression(self) -> int:
        return math.prod(self.spatial_downsample_factors)

    @property
    def temporal_compression(self) -> int:
        return math.prod(self.temporal_downsample_factors)


def _encoded_input(layer, hidden, maximum):
    """Quantize a projection input, reusing the caller's tensor-wide amax if any."""  # noqa: E501
    quantizer = layer.input_quantizer
    if quantizer is None or isinstance(hidden, QuantizedTensor):
        return hidden
    # Fused normalization supplies a tensor-wide magnitude only. Row and block
    # formats obtain their own complete statistical domains from the Quantizer.
    if quantizer.axis is not None or quantizer.format == "mxfp8":
        maximum = None
    return quantizer.quantize(
        hidden.reshape(-1, hidden.shape[-1]),
        distribution=layer.input_distribution,
        amax=maximum,
    )


def _project(layer, hidden, maximum=None):
    """Keep encoded projection bias for the VAE's FP32 affine/residual equation."""  # noqa: E501
    if not isinstance(layer.weight, QuantizedTensor):
        return layer(hidden), None
    inputs = _encoded_input(layer, hidden, maximum)
    output = functional.linear(inputs, layer.weight, output_dtype=hidden.dtype)
    output = output.reshape(*hidden.shape[:-1], output.shape[-1])
    if isinstance(layer, RowParallelLinear):
        output = layer.group.all_reduce(output)
    return output, layer.bias


def _project_branches(projection, hidden, maximum=None):
    branches = tuple(projection.projections.values())
    if len({branch.input_quantizer for branch in branches}) != 1:
        results = {
            name: _project(branch, hidden, maximum)
            for name, branch in projection.projections.items()
        }
        return {name: result[0] for name, result in results.items()}, {
            name: result[1] for name, result in results.items()
        }
    inputs = _encoded_input(branches[0], hidden, maximum)
    deferred = {
        name: branch.bias
        if isinstance(branch.weight, QuantizedTensor)
        else None
        for name, branch in projection.projections.items()
    }
    outputs = functional.merged_linear(
        inputs,
        {
            name: branch.weight
            for name, branch in projection.projections.items()
        },
        {
            name: None
            if isinstance(branch.weight, QuantizedTensor)
            else branch.bias
            for name, branch in projection.projections.items()
        },
        branch_width=projection.branch_width,
        output_dtype=hidden.dtype,
    )
    return {
        name: value.reshape(*hidden.shape[:-1], value.shape[-1])
        for name, value in outputs.items()
    }, deferred


def _feed_forward(mlp, hidden, maximum=None):
    projections = (*mlp.gate_up.projections.values(), mlp.down)
    if all(
        not isinstance(projection.weight, QuantizedTensor)
        and projection.input_quantizer is None
        for projection in projections
    ):
        # Dense projections already apply their biases before activation.
        # The shared MLP borrows the merged gate/up output without repacking.
        return mlp(hidden), None

    branches, biases = _project_branches(mlp.gate_up, hidden, maximum)
    # Checkpoint feed-forward rows are value-first. Consume the two adjacent
    # projection views directly: reversing them into one packed tensor would
    # copy the full activation before every decoder MLP.
    options = {
        "value_bias": biases["up"],
        "gate_bias": biases["gate"],
    }
    quantizer = mlp.down.input_quantizer
    if (
        quantizer is not None
        and quantizer.axis is None
        and quantizer.format != "mxfp8"
    ):
        activated, maximum = swiglu_absmax(
            branches["up"], branches["gate"], **options
        )
    else:
        activated = swiglu(branches["up"], branches["gate"], **options)
        maximum = None

    return _project(mlp.down, activated, maximum)


class TransformerLayer(nn.Module):
    """Apply channel-scaled attention and feed-forward residual updates."""

    def __init__(self, config: Config):
        super().__init__()
        width = (
            config.decoder_num_attention_heads
            * config.decoder_attention_head_dim
        )
        self.norms = nn.ModuleList(
            nn.RMSNorm(width, eps=config.decoder_norm_eps) for _ in range(2)
        )
        self.scales = nn.ParameterList(
            nn.Parameter(torch.empty(width)) for _ in range(2)
        )
        self.qkv = QKVParallelLinear(
            width,
            config.decoder_num_attention_heads,
            config.decoder_num_attention_heads,
            config.decoder_attention_head_dim,
        )
        self.attention = Attention(
            config.decoder_num_attention_heads,
            config.decoder_num_attention_heads,
            config.decoder_attention_head_dim,
        )
        self.output = RowParallelLinear(width, width)
        self.mlp = GatedMLP(width, width * config.decoder_ffn_mult, bias=True)

    def _advance(self, hidden, normalized, cos, sin, maximum=None):
        projections, biases = _project_branches(self.qkv, normalized, maximum)
        batch, sequence, _ = hidden.shape
        dim = self.qkv.head_dim
        query, key, value = (
            projections[name].reshape(batch, sequence, -1, dim)
            for name in ("q", "k", "v")
        )
        qk_bias_rms_norm_rope_(
            query,
            key,
            cos,
            sin,
            query_bias=biases["q"],
            key_bias=biases["k"],
            value=value if biases["v"] is not None else None,
            value_bias=biases["v"],
        )
        attended = self.attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            DenseInput(causal=False, mask=None),
        )
        attended, bias = _project(
            self.output, attended.transpose(1, 2).reshape(batch, sequence, -1)
        )

        quantized = isinstance(
            self.mlp.gate_up.projections["gate"].weight, QuantizedTensor
        )
        if quantized:
            hidden, normalized, maximum = scaled_residual_rms_norm_absmax_(
                hidden,
                attended,
                self.scales[0],
                self.norms[1].weight,
                update_bias=bias,
                eps=self.norms[1].eps,
            )
        else:
            hidden, normalized = scaled_residual_rms_norm_(
                hidden,
                attended,
                self.scales[0],
                self.norms[1].weight,
                update_bias=bias,
                eps=self.norms[1].eps,
            )
            maximum = None
        update, bias = _feed_forward(self.mlp, normalized, maximum)
        return hidden, update, bias

    def forward(self, hidden, cos, sin):
        if isinstance(self.qkv.projections["q"].weight, QuantizedTensor):
            normalized, maximum = weighted_rms_norm_absmax(
                hidden, self.norms[0].weight, eps=self.norms[0].eps
            )
        else:
            normalized = weighted_rms_norm(
                hidden, self.norms[0].weight, eps=self.norms[0].eps
            )
            maximum = None
        hidden, update, bias = self._advance(
            hidden, normalized, cos, sin, maximum
        )
        return scaled_residual_(
            hidden, update, self.scales[1], update_bias=bias
        )


class Transformer(nn.Module):
    """Decode latent tokens and register tokens into spatiotemporal RGB patches."""  # noqa: E501

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        width = (
            config.decoder_num_attention_heads
            * config.decoder_attention_head_dim
        )
        self.input = Linear(config.latent_channels, width)
        self.register_tokens = nn.Parameter(
            torch.empty(1, config.decoder_num_register_tokens, width)
        )
        self.position = RotaryEmbedding(
            int(
                config.decoder_attention_head_dim
                * config.decoder_rope_dim_ratio
            )
            // 3,
            theta=config.decoder_rope_theta,
        )
        self.layers = nn.ModuleList(
            TransformerLayer(config) for _ in range(config.decoder_num_layers)
        )
        self.norm = nn.LayerNorm(width, eps=config.decoder_norm_eps)
        self.output = Linear(
            width,
            config.out_channels
            * config.temporal_compression
            * config.spatial_compression**2,
        )

    def forward(self, latents):
        # [B, C, T, H, W] latents become [B, T*H*W, C] token rows.
        batch, channels, frames, height, width = latents.shape
        hidden = self.input(
            latents.permute(0, 2, 3, 4, 1).reshape(
                batch, frames * height * width, channels
            )
        )
        compute_dtype = hidden.dtype
        # FP32 register parameters promote the residual stream. The fused
        # normalizations separately select the activation compute dtype.
        # One trailing zero slot completes the checkpoint's token rows.
        registers = self.register_tokens.expand(batch, -1, -1)
        hidden = torch.cat(
            (hidden, registers, torch.zeros_like(hidden[:, :1])), dim=1
        )

        # Cell-center coordinates in [-1, 1) along each spatiotemporal axis.
        axes = tuple(
            2.0
            * (
                torch.arange(
                    0.5, size, device=hidden.device, dtype=torch.float32
                )
                / size
            )
            - 1.0
            for size in (frames, height, width)
        )
        positions = (
            torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
            .flatten(0, 2)
            .unsqueeze(0)
            .expand(batch, -1, -1)
        )
        suffix = positions.new_zeros(
            (batch, self.config.decoder_num_register_tokens + 1, 3)
        )
        positions = torch.cat((positions, suffix), dim=1)
        cos, sin = self.position(
            positions * (2.0 * math.pi),
            dtype=torch.float32,
            sequence_length=max(frames, height, width),
        )
        # Compact factors in the activation dtype, one per rotate-half pair:
        # [B, tokens, rope_width / 2], broadcast over heads.
        cos, sin = (
            value.flatten(2, 3).to(compute_dtype) for value in (cos, sin)
        )

        first = self.layers[0]
        if isinstance(first.qkv.projections["q"].weight, QuantizedTensor):
            normalized, maximum = weighted_rms_norm_absmax(
                hidden, first.norms[0].weight, eps=first.norms[0].eps
            )
        else:
            normalized = weighted_rms_norm(
                hidden, first.norms[0].weight, eps=first.norms[0].eps
            )
            maximum = None
        hidden, update, bias = first._advance(
            hidden, normalized, cos, sin, maximum
        )
        previous = first
        # Keep the unrounded residual sum available to the next normalization.
        for layer in self.layers[1:]:
            if isinstance(layer.qkv.projections["q"].weight, QuantizedTensor):
                hidden, normalized, maximum = scaled_residual_rms_norm_absmax_(
                    hidden,
                    update,
                    previous.scales[1],
                    layer.norms[0].weight,
                    update_bias=bias,
                    eps=layer.norms[0].eps,
                )
            else:
                hidden, normalized = scaled_residual_rms_norm_(
                    hidden,
                    update,
                    previous.scales[1],
                    layer.norms[0].weight,
                    update_bias=bias,
                    eps=layer.norms[0].eps,
                )
                maximum = None
            hidden, update, bias = layer._advance(
                hidden, normalized, cos, sin, maximum
            )
            previous = layer

        if isinstance(self.output.weight, QuantizedTensor):
            hidden, maximum = scaled_residual_layer_norm_absmax(
                hidden,
                update,
                previous.scales[1],
                self.norm.weight,
                self.norm.bias,
                update_bias=bias,
                eps=self.norm.eps,
            )
        else:
            hidden = scaled_residual_layer_norm(
                hidden,
                update,
                previous.scales[1],
                self.norm.weight,
                self.norm.bias,
                update_bias=bias,
                eps=self.norm.eps,
            )
            maximum = None
        hidden, bias = _project(self.output, hidden, maximum)
        return unpatchify_video_tokens(
            hidden,
            bias,
            grid_shape=(frames, height, width),
            patch_shape=(
                self.config.temporal_compression,
                self.config.spatial_compression,
                self.config.spatial_compression,
            ),
        )


class Decoder(SpatialDecoder):
    """Project latent channels and reconstruct overlapping spatial tiles."""

    def __init__(self, config: Config):
        super().__init__(
            Transformer(config),
            spatial_compression=config.spatial_compression,
            tile_height=256,
            tile_width=256,
            overlap_height=64,
            overlap_width=64,
        )
        self.config = config
        self.post_quant_conv = nn.Conv3d(
            config.latent_channels, config.latent_channels, 1
        )

    def forward(self, latents):
        return self.decoder(self.post_quant_conv(latents))


class Model(LatentDecoder):
    """Denormalize one native temporal segment, decode tiles, and crop its pad."""  # noqa: E501

    def __init__(self, config: Config, *, frame_size: image.Config):
        if (
            frame_size.height % config.spatial_compression
            or frame_size.width % config.spatial_compression
        ):
            raise ValueError("video raster must align with spatial compression")
        self.frame_size = frame_size
        # A clip covers `span` latent frames; cropping `token_drop` frames
        # per clip leaves consecutive native windows sharing `overlap`
        # latent frames.
        span = math.ceil(config.clip_length / config.temporal_compression)
        overlap = (-config.token_drop) % span
        super().__init__(
            Decoder(config),
            latent_shape=(
                1,
                config.latent_channels,
                span + overlap,
                frame_size.height // config.spatial_compression,
                frame_size.width // config.spatial_compression,
            ),
            mean=torch.tensor(
                config.latents_mean, dtype=torch.float32, device="cpu"
            ).view(1, config.latent_channels, 1, 1, 1),
            std=torch.tensor(
                config.latents_std, dtype=torch.float32, device="cpu"
            ).view(1, config.latent_channels, 1, 1, 1),
        )

    @property
    def compute_dtype(self):
        return self.decoder.decoder.input.weight.dtype

    def forward(self, latents):
        with torch.autocast(
            device_type=latents.device.type,
            dtype=self.compute_dtype,
            enabled=latents.device.type == "cuda",
        ):
            decoded = super().forward(latents)
        # The VAE left-pads each clip to a multiple of its temporal compression;
        # crop those leading frames from the decoded timeline.
        padding = (
            -self.decoder.config.clip_length
        ) % self.decoder.config.temporal_compression
        return decoded[:, :, padding:].to(torch.float16).contiguous()


def assignments(model: Decoder | Model, reader):
    """Map native VAE tensors, splitting value-first feed-forward branches."""
    from uniserve.loading import weights

    decoder = model.decoder if isinstance(model, Model) else model
    available = frozenset(reader.names())
    assignments = []
    for name, parameter in decoder.named_parameters():
        branch = None
        source = name
        if name.startswith("decoder.layers."):
            _, _, index, kind, *parts = name.split(".")
            prefix = f"decoder.transformer_blocks.{index}."
            if kind == "norms":
                source = (
                    prefix + f"norm{int(parts[0]) + 1}." + ".".join(parts[1:])
                )
            elif kind == "scales":
                source = prefix + f"scale{int(parts[0]) + 1}"
            elif kind == "qkv":
                source = prefix + f"attn.to_{parts[1]}.{parts[2]}"
            elif kind == "output":
                source = prefix + "attn.to_out.0." + ".".join(parts)
            elif kind == "mlp":
                if parts[0] == "gate_up":
                    branch = 0 if parts[2] == "up" else 1
                    source = prefix + "ff.net.0.proj." + parts[3]
                else:
                    source = prefix + "ff.net.2." + ".".join(parts[1:])
            else:
                raise ValueError(f"unmapped video transformer weight {name!r}")
        else:
            source = (
                source.replace("decoder.input.", "decoder.proj_in.")
                .replace("decoder.norm.", "decoder.norm_out.")
                .replace("decoder.output.", "decoder.proj_out.")
            )
        if source not in available:
            continue
        value = reader.get(source)

        region = None
        if branch is not None:
            if value.shape[0] % 2:
                raise ValueError(
                    "video feed-forward checkpoint must contain "
                    "two equal branches"
                )
            width = value.shape[0] // 2
            region = (
                slice(branch * width, (branch + 1) * width),
                *(slice(0, size) for size in value.shape[1:]),
            )
        assignments.append(
            weights.Assignment(parameter, value, source_slice=region)
        )
    return tuple(assignments)
