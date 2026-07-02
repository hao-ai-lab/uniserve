"""Tensor-parallel local geometry: sharded attention heads, KV pools, loader.

Pins the TP4-enabling contracts: SenseNova derives local head counts from the
sharded QKV projection (like Qwen3), KV-pool geometry divides KV heads with the
"too small to split stays whole" rule, and the native loader routes
global-shaped checkpoint tensors through the shard-narrowing weight loader.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from uniserve_worker.loader.transformers import _shard_plan_expects_global_shape
from uniserve_worker.nn.mesh import DeviceMesh, use_mesh
from uniserve_worker.nn.placement import ShardPlan, ShardSpec, set_shard_plan


def _tp_mesh(rank: int, size: int) -> DeviceMesh:
    return DeviceMesh.tp(rank, size, device="cpu")


def test_local_kv_head_count_divides_and_keeps_small_groups_whole():
    from uniserve_worker.nn import local_kv_head_count

    with use_mesh(_tp_mesh(0, 4)):
        assert local_kv_head_count(8) == 2
        # A KV group smaller than tp stays whole (replicated), matching
        # QKVParallelLinear's sharding rule.
        assert local_kv_head_count(2) == 2
    with use_mesh(_tp_mesh(0, 1)):
        assert local_kv_head_count(8) == 8


def test_local_kv_head_count_rejects_indivisible_groups():
    from uniserve_worker.nn import local_kv_head_count

    with use_mesh(_tp_mesh(0, 4)):
        with pytest.raises(ValueError):
            local_kv_head_count(6)


def test_qwen3_attention_uses_local_head_counts_under_tp():
    from types import SimpleNamespace

    from uniserve_worker.models.qwen3 import Qwen3Attention

    cfg = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=4096,
    )
    with use_mesh(_tp_mesh(2, 4)):
        attn = Qwen3Attention(cfg, layer_id=0)
    assert attn.total_num_heads == 8 and attn.total_num_kv_heads == 4
    assert attn.num_heads == 2 and attn.num_kv_heads == 1
    assert attn.attn.num_heads == 2 and attn.attn.num_kv_heads == 1


def test_mot_decoder_layer_uses_local_head_counts_under_tp():
    from types import SimpleNamespace

    from uniserve_worker.nn.decoder.mot import MoTDecoderLayer
    from uniserve_worker.nn.linear import RowParallelLinear

    cfg = SimpleNamespace(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    with use_mesh(_tp_mesh(1, 4)):
        layer = MoTDecoderLayer(cfg)
    assert layer.total_n_heads == 8 and layer.total_n_kv == 4
    assert layer.n_heads == 2 and layer.n_kv == 1
    assert layer.rep == 2
    # o_proj consumes the sharded attention output and must re-reduce: it is a
    # RowParallelLinear over the *global* q width for both experts.
    for o_proj in (layer.o_proj, layer.o_proj_moe_gen):
        assert isinstance(o_proj, RowParallelLinear)
        assert o_proj.global_input_size == 8 * 8
        assert o_proj.input_size == (8 * 8) // 4
    assert layer.attn.num_heads == 2 and layer.attn_moe_gen.num_kv_heads == 1


def test_sensenova_attention_uses_local_head_counts_under_tp():
    from types import SimpleNamespace

    from uniserve_worker.models.sensenova.model import _SenseNovaAttention

    config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        rope_theta_hw=10_000.0,
        max_position_embeddings=4096,
        max_position_embeddings_hw=1024,
    )
    with use_mesh(_tp_mesh(1, 4)):
        attn = _SenseNovaAttention(config, layer_idx=0)
    assert attn.total_num_heads == 8 and attn.total_num_kv_heads == 4
    assert attn.num_heads == 2 and attn.num_kv_heads == 1
    assert attn.attn.num_heads == 2 and attn.attn.num_kv_heads == 1
    # Local weights: q rows = 2 heads * 8 dims, kv rows = 1 head * 8 dims each.
    assert attn.qkv_proj.weight.shape[0] == (2 + 1 + 1) * 8


def test_shard_plan_accepts_exactly_the_narrowed_global_shape():
    param = nn.Parameter(torch.empty(4, 12), requires_grad=False)
    set_shard_plan(param, ShardPlan(spec=ShardSpec(axis=1, rank=0, size=4)))

    assert _shard_plan_expects_global_shape(torch.empty(4, 48), param)
    # Wrong global shape must still fail loudly on the direct-write path.
    assert not _shard_plan_expects_global_shape(torch.empty(4, 40), param)
    assert not _shard_plan_expects_global_shape(torch.empty(2, 48), param)

    replicated = nn.Parameter(torch.empty(4, 12), requires_grad=False)
    set_shard_plan(replicated, ShardPlan(spec=ShardSpec(axis=1, rank=0, size=1)))
    assert not _shard_plan_expects_global_shape(torch.empty(4, 48), replicated)

    unplanned = nn.Parameter(torch.empty(4, 12), requires_grad=False)
    assert not _shard_plan_expects_global_shape(torch.empty(4, 48), unplanned)
