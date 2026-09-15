"""Lazily exported neural layers, parallel primitives, and model operators."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Each public symbol names the submodule imported on its first package lookup.
_EXPORTS: dict[str, str] = {
    # activation
    "GELUAndMul": "activation",
    "SiLUAndMul": "activation",
    "get_act_fn": "activation",
    # attention
    "Attention": "attention",
    # decoder
    "GatedMLP": "mlp",
    "RouteSpan": "routing",
    "RoutedTensor": "routing",
    # linear
    "ColumnParallelLinear": "linear",
    "Linear": "linear",
    "MergedColumnParallelLinear": "linear",
    "QKVParallelLinear": "linear",
    "RowParallelLinear": "linear",
    # mesh (parallelism topology + transports)
    # moe
    "FusedMoE": "moe",
    "TopK": "moe",
    # norm
    "RMSNorm": "norm",
    # rope
    "RotaryEmbedding": "rope",
    "TimestepEmbedding": "timestep",
    "Modulation": "modulation",
    "PositionEmbedding": "vision",
    # vae
    "AttentionBlock": "vae.layers",
    "ResidualBlock": "vae.layers",
    "Downsample": "vae.layers",
    "Upsample": "vae.layers",
    "DiagonalGaussian": "vae.layers",
    "PatchAutoencoder": "vae.patch",
    "RGBDecoder": "vae.patch",
    # vision
    "MLPConnector": "vision",
    "PatchEmbed": "vision",
    # vocab_parallel_embedding
    "VocabParallelHead": "linear",
    "VocabParallelEmbedding": "linear",
}

# The literal export list remains visible to static tooling without eager imports.
__all__ = [
    "AttentionBlock",
    "ResidualBlock",
    "Downsample",
    "Upsample",
    "DiagonalGaussian",
    "PatchAutoencoder",
    "RGBDecoder",
    "GatedMLP",
    "RouteSpan",
    "RoutedTensor",
    "ColumnParallelLinear",
    "FusedMoE",
    "GELUAndMul",
    "Linear",
    "MLPConnector",
    "MergedColumnParallelLinear",
    "VocabParallelHead",
    "PatchEmbed",
    "QKVParallelLinear",
    "RMSNorm",
    "RotaryEmbedding",
    "TimestepEmbedding",
    "Modulation",
    "PositionEmbedding",
    "RowParallelLinear",
    "Attention",
    "SiLUAndMul",
    "TopK",
    "VocabParallelEmbedding",
    "get_act_fn",
]

# Lazy resolution and the advertised public surface must describe the same names.
assert set(__all__) == set(_EXPORTS), sorted(set(__all__) ^ set(_EXPORTS))


def __getattr__(name: str):
    """Resolve a lazily exported neural-network symbol from its owning module."""

    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(module, name)
    globals()[name] = value  # Cache resolved symbols for normal module lookup.
    return value


def __dir__() -> list[str]:
    """List eager globals and all supported lazy exports."""

    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:  # Expose concrete definitions to type checkers without importing at runtime.
    from uniserve.nn.activation import GELUAndMul, SiLUAndMul, get_act_fn
    from uniserve.nn.attention import Attention
    from uniserve.nn.linear import (
        ColumnParallelLinear,
        Linear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from uniserve.nn.moe import FusedMoE, TopK
    from uniserve.nn.norm import RMSNorm
    from uniserve.nn.rope import (
        RotaryEmbedding,
    )
    from uniserve.nn.vision import (
        MLPConnector,
        PatchEmbed,
    )
    from uniserve.nn.linear import (
        VocabParallelHead,
        VocabParallelEmbedding,
    )
