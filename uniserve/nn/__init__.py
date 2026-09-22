"""Neural layers, parallel primitives and model operators.

Numerical formulas live in :mod:`uniserve.nn.functional`; the modules exported
here compose them with parameters, partitioning and bound execution resources.
"""

from uniserve.nn.activation import GELUAndMul, SiLUAndMul, get_act_fn
from uniserve.nn.attention import Attention
from uniserve.nn.linear import (
    ColumnParallelLinear,
    Linear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelHead,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.modulation import Modulation
from uniserve.nn.moe import FusedMoE, TopK
from uniserve.nn.norm import RMSNorm
from uniserve.nn.rope import RotaryEmbedding
from uniserve.nn.routing import RoutedTensor, RouteSpan
from uniserve.nn.timestep import TimestepEmbedding
from uniserve.nn.vae.layers import (
    AttentionBlock,
    DiagonalGaussian,
    Downsample,
    ResidualBlock,
    Upsample,
)
from uniserve.nn.vae.patch import PatchAutoencoder, RGBDecoder
from uniserve.nn.vision import MLPConnector, PatchEmbed, PositionEmbedding

__all__ = [
    "Attention",
    "AttentionBlock",
    "ColumnParallelLinear",
    "DiagonalGaussian",
    "Downsample",
    "FusedMoE",
    "GELUAndMul",
    "GatedMLP",
    "Linear",
    "MLPConnector",
    "MergedColumnParallelLinear",
    "Modulation",
    "PatchAutoencoder",
    "PatchEmbed",
    "PositionEmbedding",
    "QKVParallelLinear",
    "RGBDecoder",
    "RMSNorm",
    "ResidualBlock",
    "RotaryEmbedding",
    "RouteSpan",
    "RoutedTensor",
    "RowParallelLinear",
    "SiLUAndMul",
    "TimestepEmbedding",
    "TopK",
    "Upsample",
    "VocabParallelEmbedding",
    "VocabParallelHead",
    "get_act_fn",
]
