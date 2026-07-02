"""Shared model layers for the worker stack.

This barrel resolves its public names lazily (PEP 562 ``__getattr__``): importing
``uniserve_worker.nn`` does not eagerly pull the entire layer zoo (attention,
decoder, vae, vision, moe, quant, ...) and its torch/backends dependencies. A
name listed in ``__all__`` is imported from its owning submodule on first
access, so ``from ..nn import LinearBase`` keeps working while a consumer that
only needs one layer pays only for that submodule.
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
    "KVCache": "decoder",
    "MoTDecoderLayer": "decoder",
    "MoTLayer": "decoder",
    "MoTMLP": "decoder",
    "MoTModel": "decoder",
    "Segment": "decoder",
    # linear
    "ColumnParallelLinear": "linear",
    "LinearBase": "linear",
    "MergedColumnParallelLinear": "linear",
    "QKVParallelLinear": "linear",
    "local_attention_head_count": "linear",
    "local_kv_head_count": "linear",
    "RowParallelLinear": "linear",
    "default_weight_loader": "linear",
    # logits
    "LogitsProcessor": "logits",
    # mesh (parallelism topology + transports)
    "DataPlaneTowerTransport": "mesh",
    "DeviceMesh": "mesh",
    "MeshAxis": "mesh",
    "divide": "mesh",
    "get_current_mesh": "mesh",
    "reset_current_mesh": "mesh",
    "set_current_mesh": "mesh",
    "use_mesh": "mesh",
    # moe
    "FusedMoE": "moe",
    "TopK": "moe",
    # norm
    "RMSNorm": "norm",
    "try_triton_qk_rms_norm": "norm",
    # placement (per-tensor placement + reshard + load-time shard sidecar)
    "Partial": "placement",
    "Pinned": "placement",
    "Placement": "placement",
    "Region": "placement",
    "Replicate": "placement",
    "Router": "placement",
    "Shard": "placement",
    "ShardPlan": "placement",
    "ShardSlot": "placement",
    "ShardSpec": "placement",
    "Sharding": "placement",
    "TensorParallelMode": "placement",
    "WeightMode": "placement",
    "get_shard_plan": "placement",
    "get_tower_coord": "placement",
    "place_partitioned_tensor": "placement",
    "place_towers": "placement",
    "reshard": "placement",
    "set_shard_plan": "placement",
    "set_tower_coord": "placement",
    "shard_spec": "placement",
    # rope
    "HFRotaryEmbedding": "rope",
    "RotaryEmbedding": "rope",
    "apply_rotary_emb": "rope",
    "apply_rotary_pos_emb": "rope",
    "get_rope": "rope",
    "rotate_half": "rope",
    "try_triton_qk_rms_norm_rope": "rope",
    # sampler
    "Sampler": "sampler",
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
    "DataPlaneTowerTransport",
    "DeviceMesh",
    "FusedMoE",
    "GeluAndMul",
    "HFRotaryEmbedding",
    "KVCache",
    "LinearBase",
    "LogitsProcessor",
    "MLPConnector",
    "MergedColumnParallelLinear",
    "MeshAxis",
    "MoTDecoderLayer",
    "MoTLayer",
    "MoTMLP",
    "MoTModel",
    "NeoVitEncoder",
    "Partial",
    "ParallelLMHead",
    "PatchEmbed",
    "Pinned",
    "Placement",
    "QKVParallelLinear",
    "local_attention_head_count",
    "local_kv_head_count",
    "RMSNorm",
    "Region",
    "Replicate",
    "RotaryEmbedding",
    "Router",
    "RowParallelLinear",
    "Sampler",
    "Segment",
    "Shard",
    "ShardPlan",
    "ShardSlot",
    "ShardSpec",
    "RadixAttention",
    "Sharding",
    "SiglipNavitEncoder",
    "SiluAndMul",
    "TensorParallelMode",
    "TopK",
    "VisionEncoder",
    "VocabParallelEmbedding",
    "WeightMode",
    "apply_rotary_emb",
    "apply_rotary_pos_emb",
    "default_ae_params",
    "default_weight_loader",
    "divide",
    "get_act_fn",
    "get_current_mesh",
    "get_rope",
    "get_shard_plan",
    "get_tower_coord",
    "pad_vocab_size",
    "place_partitioned_tensor",
    "place_towers",
    "reset_current_mesh",
    "reshard",
    "rotate_half",
    "set_current_mesh",
    "set_shard_plan",
    "set_tower_coord",
    "shard_spec",
    "try_triton_qk_rms_norm",
    "try_triton_qk_rms_norm_rope",
    "use_mesh",
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
    from .decoder import KVCache, MoTDecoderLayer, MoTLayer, MoTMLP, MoTModel, Segment
    from .linear import (
        ColumnParallelLinear,
        LinearBase,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
        default_weight_loader,
    )
    from .logits import LogitsProcessor
    from .mesh import (
        DataPlaneTowerTransport,
        DeviceMesh,
        MeshAxis,
        divide,
        get_current_mesh,
        reset_current_mesh,
        set_current_mesh,
        use_mesh,
    )
    from .moe import FusedMoE, TopK
    from .norm import RMSNorm, try_triton_qk_rms_norm
    from .placement import (
        Partial,
        Pinned,
        Placement,
        Region,
        Replicate,
        Router,
        Shard,
        Sharding,
        ShardPlan,
        ShardSlot,
        ShardSpec,
        TensorParallelMode,
        WeightMode,
        get_shard_plan,
        get_tower_coord,
        place_partitioned_tensor,
        place_towers,
        reshard,
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
        rotate_half,
        try_triton_qk_rms_norm_rope,
    )
    from .sampler import Sampler
    from .vae import AutoEncoder, AutoEncoderParams, default_ae_params
    from .vision import MLPConnector, NeoVitEncoder, PatchEmbed, SiglipNavitEncoder, VisionEncoder
    from .vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding, pad_vocab_size
