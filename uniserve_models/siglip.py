"""SigLIP-NaViT image transformer composition and patch coordinates."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from uniserve.loading import checkpoint, weights
from uniserve.model import TransformerEncoder
from uniserve.nn.attention import Attention as ScaledAttention
from uniserve.nn.attention import SequenceLengths, VarlenInput
from uniserve.nn.functional import patchify
from uniserve.nn.linear import ColumnParallelLinear, Linear, QKVParallelLinear, RowParallelLinear
from uniserve.nn.vision.patching import build_abs_positions_from_grid_hw


@dataclass(frozen=True)
class TransformerConfig:
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    num_hidden_layers: int
    layer_norm_eps: float

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value < 1
                for value in (
                    self.hidden_size,
                    self.num_attention_heads,
                    self.intermediate_size,
                    self.num_hidden_layers,
                )
            )
            or self.hidden_size % self.num_attention_heads
        ):
            raise ValueError("SigLIP widths, layers and heads must be positive and compatible")
        if not math.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0:
            raise ValueError("SigLIP layer norm epsilon must be finite and positive")


@dataclass(frozen=True)
class Config:
    patch_size: int
    image_size: int
    num_channels: int
    encoder: TransformerConfig

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value < 1
                for value in (self.patch_size, self.image_size, self.num_channels)
            )
            or self.image_size % self.patch_size
        ):
            raise ValueError("SigLIP image and patch dimensions must be positive and aligned")


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        heads = config.num_attention_heads
        dim = config.hidden_size // heads
        self.qkv = QKVParallelLinear(config.hidden_size, heads, heads, dim, bias=True)
        self.attention = ScaledAttention(heads, heads, dim)
        self.output = RowParallelLinear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, patches: torch.Tensor, attention: VarlenInput) -> torch.Tensor:
        projections = self.qkv(patches)
        query, key, value = (
            projections[name].reshape(patches.shape[0], -1, self.attention.head_dim)
            for name in ("q", "k", "v")
        )
        hidden = self.attention(query, key, value, attention)
        return self.output(hidden.flatten(1))


class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Attention(config)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=True),
            nn.GELU(approximate="tanh"),
            RowParallelLinear(config.intermediate_size, config.hidden_size, bias=True),
        )

    def forward(self, patches: torch.Tensor, attention: VarlenInput) -> torch.Tensor:
        patches = patches + self.attention(self.input_norm(patches), attention)
        return patches + self.mlp(self.output_norm(patches))


class Encoder(nn.Module):
    """Encode canonical HWC patch rows with separate attention per image.

    grids supplies one (height, width) pair per image. grid_shapes supplies
    the corresponding host dimensions without reading device values. NCHW pixels may also be supplied with those same coordinates.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_embedding = Linear(
            config.num_channels * config.patch_size**2, config.encoder.hidden_size, bias=True
        )
        self.position_embedding = nn.Embedding(
            (config.image_size // config.patch_size) ** 2, config.encoder.hidden_size
        )
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                TransformerLayer(config.encoder) for _ in range(config.encoder.num_hidden_layers)
            ),
            nn.LayerNorm(config.encoder.hidden_size, eps=config.encoder.layer_norm_eps),
        )

    def forward(
        self, pixels: torch.Tensor, grids: torch.Tensor, grid_shapes: tuple[tuple[int, int], ...]
    ) -> torch.Tensor:
        if pixels.ndim == 4:
            pixels = patchify(pixels, patch_size=self.config.patch_size).flatten(0, 1)
        counts = tuple(height * width for height, width in grid_shapes)
        side = self.config.image_size // self.config.patch_size
        if (
            pixels.ndim != 2
            or pixels.shape != (sum(counts), self.patch_embedding.in_features)
            or grids.shape != (len(counts), 2)
            or grids.dtype not in (torch.int32, torch.int64)
            or any(min(shape) < 1 or max(shape) > side for shape in grid_shapes)
        ):
            raise ValueError(
                "SigLIP patches and grid coordinates must cover the declared image grids"
            )
        columns, rows = build_abs_positions_from_grid_hw(grids, total=sum(counts))
        positions = rows * side + columns
        features = self.patch_embedding(pixels.to(self.patch_embedding.weight.dtype))
        features = features + self.position_embedding(positions)
        values = grids.prod(dim=1).to(torch.int32)
        offsets = torch.cat((values.new_zeros(1), values.cumsum(0, dtype=torch.int32)))
        lengths = SequenceLengths(host=counts, values=values, offsets=offsets)
        return self.encoder(features, VarlenInput(lengths, lengths, (False,) * len(counts)))


def assignments(
    model: Encoder, reader: checkpoint.Reader, *, prefix: str = ""
) -> tuple[weights.Assignment, ...]:
    """Map SigLIP's checkpoint tower, whose patch matrix is already HWC-packed."""
    result = []
    available = frozenset(reader.names())
    for name, parameter in model.named_parameters():
        if name.startswith(("patch_embedding.", "position_embedding.")):
            source = "embeddings." + name
        elif name.startswith("encoder.norm."):
            source = "post_layernorm." + name.removeprefix("encoder.norm.")
        else:
            source = name.replace(".input_norm.", ".layer_norm1.")
            source = source.replace(".output_norm.", ".layer_norm2.")
            source = source.replace(".attention.output.", ".self_attn.out_proj.")
            for branch in ("q", "k", "v"):
                source = source.replace(
                    f".attention.qkv.projections.{branch}.", f".self_attn.{branch}_proj."
                )
            source = source.replace(".mlp.0.", ".mlp.fc1.").replace(".mlp.2.", ".mlp.fc2.")
        if prefix + source in available:
            result.append(weights.Assignment(parameter, reader.get(prefix + source)))
    return tuple(result)
