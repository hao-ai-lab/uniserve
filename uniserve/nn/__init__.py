"""Lazily exported neural layers, parallel primitives, and model operators."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Each public symbol names the submodule imported on its first package lookup.
_EXPORTS: dict[str, str] = {
    # activation
    "GeluAndMul": "activation",
    "SiluAndMul": "activation",
    "get_act_fn": "activation",
    # attention
    "RadixAttention": "attention",
    # decoder
    "MoTDecoderLayer": "decoder",
    "MoTModel": "decoder",
    # linear
    "ColumnParallelLinear": "linear",
    "LinearBase": "linear",
    "InterleavedMergedColumnParallelLinear": "linear",
    "MergedColumnParallelLinear": "linear",
    "QKVParallelLinear": "linear",
    "local_attention_head_count": "linear",
    "local_kv_head_count": "linear",
    "local_kv_head_offset": "linear",
    "RowParallelLinear": "linear",
    # immutable layer construction
    "LayerConfig": "layer",
    # logits
    "LogitsProcessor": "logits",
    # mesh (parallelism topology + transports)
    # moe
    "FusedMoE": "moe",
    "TopK": "moe",
    # norm
    "RMSNorm": "norm",
    # parameter shards and modality coordinates
    "ShardPlan": "shard",
    "ShardSlot": "shard",
    "Shard": "shard",
    "WeightMode": "shard",
    "get_shard_plan": "shard",
    "set_shard_plan": "shard",
    "shard_for": "shard",
    # rope
    "HFRotaryEmbedding": "rope",
    "RotaryEmbedding": "rope",
    "apply_rotary_emb": "rope",
    "apply_rotary_pos_emb": "rope",
    "get_rope": "rope",
    "qk_norm_rope": "rope",
    "rotate_half": "rope",
    # vae
    "AutoEncoder": "vae",
    "AutoEncoderConfig": "vae",
    # vision
    "MLPConnector": "vision",
    "NeoVitEncoder": "vision",
    "PatchEmbed": "vision",
    "SiglipNavitEncoder": "vision",
    "VisionEncoder": "vision",
    # vocab_parallel_embedding
    "ParallelLMHead": "vocab_parallel_embedding",
    "VocabParallelEmbedding": "vocab_parallel_embedding",
    "pad_vocab_size": "vocab_parallel_embedding",
}

# The literal export list remains visible to static tooling without eager imports.
__all__ = [
    "AutoEncoder",
    "AutoEncoderConfig",
    "ColumnParallelLinear",
    "FusedMoE",
    "GeluAndMul",
    "HFRotaryEmbedding",
    "LinearBase",
    "InterleavedMergedColumnParallelLinear",
    "LayerConfig",
    "LogitsProcessor",
    "MLPConnector",
    "MergedColumnParallelLinear",
    "MoTDecoderLayer",
    "MoTModel",
    "NeoVitEncoder",
    "ParallelLMHead",
    "PatchEmbed",
    "QKVParallelLinear",
    "local_attention_head_count",
    "local_kv_head_count",
    "local_kv_head_offset",
    "RMSNorm",
    "RotaryEmbedding",
    "RowParallelLinear",
    "ShardPlan",
    "ShardSlot",
    "Shard",
    "RadixAttention",
    "SiglipNavitEncoder",
    "SiluAndMul",
    "TopK",
    "VisionEncoder",
    "VocabParallelEmbedding",
    "WeightMode",
    "apply_rotary_emb",
    "apply_rotary_pos_emb",
    "get_act_fn",
    "get_rope",
    "qk_norm_rope",
    "get_shard_plan",
    "pad_vocab_size",
    "rotate_half",
    "set_shard_plan",
    "shard_for",
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
    from uniserve.nn.activation import GeluAndMul, SiluAndMul, get_act_fn
    from uniserve.nn.attention import RadixAttention
    from uniserve.nn.decoder import MoTDecoderLayer, MoTModel
    from uniserve.nn.layer import LayerConfig
    from uniserve.nn.linear import (
        ColumnParallelLinear,
        InterleavedMergedColumnParallelLinear,
        LinearBase,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from uniserve.nn.logits import LogitsProcessor
    from uniserve.nn.moe import FusedMoE, TopK
    from uniserve.nn.norm import RMSNorm
    from uniserve.nn.rope import (
        HFRotaryEmbedding,
        RotaryEmbedding,
        apply_rotary_emb,
        apply_rotary_pos_emb,
        get_rope,
        qk_norm_rope,
        rotate_half,
    )
    from uniserve.nn.shard import (
        Shard,
        ShardPlan,
        ShardSlot,
        WeightMode,
        get_shard_plan,
        set_shard_plan,
        shard_for,
    )
    from uniserve.nn.vae import AutoEncoder, AutoEncoderConfig
    from uniserve.nn.vision import (
        MLPConnector,
        NeoVitEncoder,
        PatchEmbed,
        SiglipNavitEncoder,
        VisionEncoder,
    )
    from uniserve.nn.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
        pad_vocab_size,
    )
