"""Unified MoT tower-framework conformance (CPU).

Covers the model-agnostic tower pieces that both BAGEL and SenseNova ride:
the tower-aware ``route_by_modality`` Router, the ``Pinned(tower)`` module tag +
``place_towers`` pass, and the ``reshard(Pinned(primary) -> Pinned(gen))`` KV
snapshot. A trivial tower must be byte-identical; a two-coordinate tower must
route/place/reshard across coordinates. SenseNova checkpoint keys must survive
the routing refactor (the expert containers are plain reference dicts, not
submodules).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from uniserve_worker.nn import (
    DeviceMesh,
    MeshAxis,
    get_tower_coord,
    place_towers,
    set_tower_coord,
)
from uniserve_worker.nn.decoder import Modality, route_by_modality, tower_modality_coords
from uniserve_worker.nn.mesh import LocalP2PTransport
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import PagedTextCache
from uniserve_worker.runtime.tower_kv import reshard_kv_snapshot, wait_kv_snapshot_ready

pytestmark = pytest.mark.integration


def _cpu_tower(*, size: int = 2, coord: int = 0, gen_device: str = "cpu") -> DeviceMesh:
    devices = tuple(torch.device(d) for d in (["cpu"] * (size - 1) + [gen_device])[:size])
    if size == 2:
        devices = (torch.device("cpu"), torch.device(gen_device))
    transport = LocalP2PTransport(axis="tower", devices=devices, _coord=coord)
    axis = MeshAxis(name="tower", size=size, coord=coord, transport=transport)
    return DeviceMesh.of(axis, device="cpu")


def test_route_by_modality_trivial_matches_manual_scatter():
    torch.manual_seed(0)
    src = torch.randn(6, 4)
    gen = torch.tensor([False, True, False, True, True, False])
    text_fn = nn.Linear(4, 4)
    gen_fn = nn.Linear(4, 4)

    out = route_by_modality(
        src,
        {Modality.TEXT: (~gen, text_fn), Modality.GEN: (gen, gen_fn)},
        out=torch.zeros_like(src),
    )
    manual = torch.zeros_like(src)
    manual[~gen] = text_fn(src[~gen])
    manual[gen] = gen_fn(src[gen])
    torch.testing.assert_close(out, manual)


def test_tower_modality_coords_trivial_is_none_and_two_coord_maps_gen():
    assert tower_modality_coords(DeviceMesh.trivial("cpu")) is None
    coords = tower_modality_coords(_cpu_tower())
    assert coords == {Modality.TEXT: 0, Modality.GEN: 1}


def test_place_towers_moves_only_tagged_subtree():
    # Coordinate 1 is a distinct (meta) device so a move is observable on CPU.
    mesh = _cpu_tower(gen_device="meta")
    root = nn.Module()
    root.text = nn.Linear(2, 2)
    root.gen = set_tower_coord(nn.Linear(2, 2), 1)
    assert get_tower_coord(root.gen) == 1

    place_towers(root, mesh)

    assert next(root.gen.parameters()).device.type == "meta"
    assert next(root.text.parameters()).device.type == "cpu"


def test_option_a_cross_process_tower_keeps_model_code_transport_agnostic():
    """Option A (cross-process tower) builds a DataPlaneTowerTransport axis; the
    model code is unchanged: place_towers must NOT move params (no in-process peer
    device) and route_by_modality must take the in-place path (one modality per
    worker), so each worker runs its own coordinate's compute locally."""
    from uniserve_worker.server.distributed import build_device_mesh

    gen_mesh = build_device_mesh(
        tp_rank=0,
        tp_size=1,
        device="cpu",
        tower_coord=1,
        tower_size=2,
    )
    axis = gen_mesh.axis("tower")
    assert axis is not None and axis.size == 2 and axis.coord == 1
    transport = axis.transport
    # Cross-process transport exposes no in-process per-coordinate device.
    assert not hasattr(transport, "device")
    assert tower_modality_coords(gen_mesh) == {Modality.TEXT: 0, Modality.GEN: 1}

    # place_towers does not move a tagged module (each worker loads only its own
    # coordinate's params; placement is a load-time property, not a device move).
    root = nn.Module()
    root.gen = set_tower_coord(nn.Linear(2, 2), 1)
    place_towers(root, gen_mesh)
    assert next(root.gen.parameters()).device.type == "cpu"

    # route_by_modality takes the in-place path (no .device() -> no dispatch).
    src = torch.randn(4, 4)
    gen = torch.tensor([False, True, False, True])
    text_fn = nn.Linear(4, 4)
    gen_fn = nn.Linear(4, 4)
    out = route_by_modality(
        src,
        {Modality.TEXT: (~gen, text_fn), Modality.GEN: (gen, gen_fn)},
        out=torch.zeros_like(src),
        transport=transport,
        coords={Modality.TEXT: 0, Modality.GEN: 1},
    )
    manual = torch.zeros_like(src)
    manual[~gen] = text_fn(src[~gen])
    manual[gen] = gen_fn(src[gen])
    torch.testing.assert_close(out, manual)


def _seed_cache(pool: PagedKVPool, block: int, length: int, layers: int) -> PagedTextCache:
    cache = PagedTextCache(pool, [block], num_layers=layers, length=0, allocate_blocks=lambda n: [])
    for layer in range(layers):
        k = torch.arange(length * pool.n_kv * pool.head_dim, dtype=torch.float32).reshape(
            length, pool.n_kv, pool.head_dim
        ) + layer * 100
        pool.write(layer, [block], start=0, k=k, v=k + 0.5)
    cache.length = length
    return cache


def test_reshard_kv_snapshot_cross_coord_copies_values_and_arms_barrier():
    layers, length = 2, 5
    src_pool = PagedKVPool(2, 4, 8, num_kv_heads=2, head_dim=4, device="cpu", dtype=torch.float32)
    dst_pool = PagedKVPool(
        2, 4, 8, num_kv_heads=2, head_dim=4, device="cpu", dtype=torch.float32, tower_coord=1
    )
    assert dst_pool.tower_coord == 1
    src_cache = _seed_cache(src_pool, block=0, length=length, layers=layers)

    free = [0, 1, 2, 3]
    transport = LocalP2PTransport(
        axis="tower", devices=(torch.device("cpu"), torch.device("cpu")), _coord=0
    )
    snap = reshard_kv_snapshot(
        src_cache,
        target_pool=dst_pool,
        allocate_blocks=lambda n: [free.pop(0) for _ in range(n)],
        num_layers=layers,
        block_size=8,
        target_device="cpu",
        transport=transport,
        src_coord=0,
        dst_coord=1,
    )
    wait_kv_snapshot_ready(snap, transport=transport, coord=1)
    assert snap.pool is dst_pool
    assert snap.length == length
    for layer in range(layers):
        sk, sv = src_pool.read(layer, [0], start=0, length=length)
        dk, dv = dst_pool.read(layer, snap.block_ids, start=0, length=length)
        torch.testing.assert_close(sk, dk)
        torch.testing.assert_close(sv, dv)


def _sensenova_layer():
    from transformers import Qwen3Config

    from uniserve_worker.models.sensenova import model as sensenova

    cfg = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 4
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    return sensenova._SenseNovaDecoderLayer(cfg, layer_idx=0)


def test_sensenova_checkpoint_keys_survive_routing_refactor():
    """The per-modality expert containers are plain reference dicts: the unified
    routing must not add any ``state_dict`` keys."""
    layer = _sensenova_layer()
    keys = set(layer.state_dict().keys())

    # The generation twins keep their ``*_mot_gen`` ownership in the state dict.
    for expected in (
        "mlp_mot_gen.gate_up_proj.weight",
        "input_layernorm_mot_gen.weight",
        "post_attention_layernorm_mot_gen.weight",
        "self_attn.qkv_proj_mot_gen.weight",
        "self_attn.o_proj_mot_gen.weight",
        "self_attn.q_norm_hw_mot_gen.weight",
    ):
        assert expected in keys, expected
    # The expert/routing containers must NOT leak into the state dict.
    assert not any("_by_modality" in k for k in keys)
