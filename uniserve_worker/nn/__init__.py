"""Shared model layers for the worker stack.

This barrel resolves its public names lazily (PEP 562 ``__getattr__``): importing
``uniserve_worker.nn`` does not eagerly pull the entire layer zoo (attention,
decoder, vae, vision, moe, quant, ...) and its torch/backends dependencies. A
    name listed in ``__all__`` is imported from its owning submodule on first
    access, so canonical package imports load only the layer they use.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Public name -> owning submodule. The single source of truth for what the
# package re-exports; ``__all__`` is derived from it so the two cannot drift.
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
    "MergedColumnParallelLinear": "linear",
    "QKVParallelLinear": "linear",
    "local_attention_head_count": "linear",
    "local_kv_head_count": "linear",
    "RowParallelLinear": "linear",
    # immutable layer construction
    "LayerSpec": "layer",
    # logits
    "LogitsProcessor": "logits",
    # mesh (parallelism topology + transports)
    "DeviceMesh": "mesh",
    "MeshAxis": "mesh",
    "TensorParallelSpec": "mesh",
    "divide": "mesh",
    # moe
    "FusedMoE": "moe",
    "TopK": "moe",
    # norm
    "RMSNorm": "norm",
    # parameter and modality-tower placement
    "ShardPlan": "placement",
    "ShardSlot": "placement",
    "ShardSpec": "placement",
    "WeightMode": "placement",
    "get_shard_plan": "placement",
    "get_tower_coord": "placement",
    "place_partitioned_tensor": "placement",
    "place_towers": "placement",
    "set_shard_plan": "placement",
    "set_tower_coord": "placement",
    "shard_spec": "placement",
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
    "AutoEncoderParams": "vae",
    "default_ae_params": "vae",
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

# Literal list (kept in sync with ``_EXPORTS`` by the assert below) so static
# tooling sees the package's public surface and the TYPE_CHECKING re-exports are
# recognised as exported rather than unused.
__all__ = [
    "AutoEncoder",
    "AutoEncoderParams",
    "ColumnParallelLinear",
    "DeviceMesh",
    "FusedMoE",
    "GeluAndMul",
    "HFRotaryEmbedding",
    "LinearBase",
    "LayerSpec",
    "LogitsProcessor",
    "MLPConnector",
    "MergedColumnParallelLinear",
    "MeshAxis",
    "MoTDecoderLayer",
    "MoTModel",
    "NeoVitEncoder",
    "ParallelLMHead",
    "PatchEmbed",
    "QKVParallelLinear",
    "local_attention_head_count",
    "local_kv_head_count",
    "RMSNorm",
    "RotaryEmbedding",
    "RowParallelLinear",
    "ShardPlan",
    "ShardSlot",
    "ShardSpec",
    "RadixAttention",
    "SiglipNavitEncoder",
    "SiluAndMul",
    "TensorParallelSpec",
    "TopK",
    "VisionEncoder",
    "VocabParallelEmbedding",
    "WeightMode",
    "apply_rotary_emb",
    "apply_rotary_pos_emb",
    "default_ae_params",
    "divide",
    "get_act_fn",
    "get_rope",
    "qk_norm_rope",
    "get_shard_plan",
    "get_tower_coord",
    "pad_vocab_size",
    "place_partitioned_tensor",
    "place_towers",
    "rotate_half",
    "set_shard_plan",
    "set_tower_coord",
    "shard_spec",
]

# Single-source-of-truth guard: the lazy resolver map and the advertised surface
# must list exactly the same names.
assert set(__all__) == set(_EXPORTS), sorted(set(__all__) ^ set(_EXPORTS))


def __getattr__(name: str):
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:  # let type-checkers see the concrete exports without eager cost
    from .activation import GeluAndMul, SiluAndMul, get_act_fn
    from .attention import RadixAttention
    from .decoder import MoTDecoderLayer, MoTModel
    from .layer import LayerSpec
    from .linear import (
        ColumnParallelLinear,
        LinearBase,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from .logits import LogitsProcessor
    from .mesh import (
        DeviceMesh,
        MeshAxis,
        TensorParallelSpec,
        divide,
    )
    from .moe import FusedMoE, TopK
    from .norm import RMSNorm
    from .placement import (
        ShardPlan,
        ShardSlot,
        ShardSpec,
        WeightMode,
        get_shard_plan,
        get_tower_coord,
        place_partitioned_tensor,
        place_towers,
        set_shard_plan,
        set_tower_coord,
        shard_spec,
    )
    from .rope import (
        HFRotaryEmbedding,
        RotaryEmbedding,
        apply_rotary_emb,
        apply_rotary_pos_emb,
        get_rope,
        qk_norm_rope,
        rotate_half,
    )
    from .vae import AutoEncoder, AutoEncoderParams, default_ae_params
    from .vision import MLPConnector, NeoVitEncoder, PatchEmbed, SiglipNavitEncoder, VisionEncoder
    from .vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding, pad_vocab_size
