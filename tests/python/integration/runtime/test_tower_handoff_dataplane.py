"""DataPlaneTowerHandoff (Mode A): und publishes the conditioning KV, gen fetches it.

The two-sided crossing (publish on the und pool, fetch into the gen replica) is the
register-once point-to-point data plane. With an in-memory fake transport the
publish->fetch round-trip must reconstruct the conditioning KV byte-for-byte into a
fresh writable replica (INV-2), the same payload :func:`reshard_kv_snapshot`
produces in-process for Mode C.
"""
from __future__ import annotations

import torch

from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import PagedTextCache
from uniserve_worker.runtime.tower_handoff import (
    DataPlaneTowerHandoff,
    TowerBinding,
)


class _FakeDataPlane:
    """In-memory stand-in for a register-once Transport (publish/fetch by locator)."""

    def __init__(self) -> None:
        self._store: dict[int, torch.Tensor] = {}
        self._next = 0

    def publish(self, tensor: torch.Tensor):
        loc = self._next
        self._next += 1
        self._store[loc] = tensor.clone()
        return loc

    def fetch(self, locator):
        return self._store[locator]


def _pool() -> PagedKVPool:
    return PagedKVPool(
        num_layers=2,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )


def test_publish_then_fetch_reconstructs_conditioning_kv():
    src_pool = _pool()
    gen_pool = _pool()
    gen_free = [0, 1, 2, 3]

    source = PagedTextCache(src_pool, [0], num_layers=2)
    for layer in range(2):
        k = torch.arange(8, dtype=torch.float32).view(1, 1, 4, 2) + layer
        v = -k
        source.update(k, v, layer_idx=layer)

    binding = TowerBinding(
        transport=None,
        primary_coord=0,
        gen_coord=1,
        num_layers=2,
        block_size=4,
        target_pool=gen_pool,
        target_device="cpu",
        allocate_blocks=lambda n: [gen_free.pop(0) for _ in range(n)],
    )
    plane = _FakeDataPlane()
    handoff = DataPlaneTowerHandoff(data_plane=plane, bind=lambda: binding)

    snapshot = handoff.publish_conditioning(source.past if hasattr(source, "past") else source)
    assert snapshot is not None
    assert len(snapshot.locators) == 2 * 2  # (k, v) per layer
    assert snapshot.length == source.get_seq_length()

    replica = handoff.stage_conditioning(snapshot)
    assert replica is not None
    assert replica.pool is gen_pool  # landed in the gen residency, not the und pool
    assert replica.get_seq_length() == source.get_seq_length()

    for layer in range(2):
        ks, vs = src_pool.read(layer, source.block_ids, start=0, length=4)
        kr, vr = gen_pool.read(layer, replica.block_ids, start=0, length=4)
        assert torch.equal(ks, kr)
        assert torch.equal(vs, vr)


def test_publish_none_and_empty_cache():
    binding = TowerBinding(
        transport=None, primary_coord=0, gen_coord=1, num_layers=2, block_size=4,
        target_pool=_pool(), target_device="cpu", allocate_blocks=lambda n: list(range(n)),
    )
    handoff = DataPlaneTowerHandoff(data_plane=_FakeDataPlane(), bind=lambda: binding)
    assert handoff.publish_conditioning(None) is None
    assert handoff.stage_conditioning(None) is None
    assert handoff.active is True


def test_publish_serialize_wire_fetch_reconstructs_kv_and_scalars():
    # The full Mode-A control-plane round-trip: und publishes + serializes the
    # snapshot to the SeqResult.locator wire string, the gen side deserializes it
    # and fetches into its own replica pool; KV must match byte-for-byte and the
    # decode-state scalars must survive the wire.
    src_pool = _pool()
    gen_pool = _pool()
    gen_free = [0, 1, 2, 3]
    source = PagedTextCache(src_pool, [0], num_layers=2)
    for layer in range(2):
        k = torch.arange(8, dtype=torch.float32).view(1, 1, 4, 2) + (layer + 1)
        source.update(k, -k, layer_idx=layer)

    und_binding = TowerBinding(
        transport=None, primary_coord=0, gen_coord=1, num_layers=2, block_size=4,
        target_pool=None, target_device="cpu", allocate_blocks=lambda n: [],
    )
    plane = _FakeDataPlane()
    und = DataPlaneTowerHandoff(data_plane=plane, bind=lambda: und_binding)
    snapshot = und.publish_conditioning(source, t_index=11, last_token_id=42)

    # Serialize to the wire string and back (what rides SeqResult.locator).
    from uniserve_worker.runtime.tower_handoff import ConditioningSnapshot
    wire = snapshot.to_wire()
    assert isinstance(wire, str)
    restored = ConditioningSnapshot.from_wire(wire)
    assert restored.length == snapshot.length
    assert restored.num_layers == 2
    assert restored.t_index == 11
    assert restored.last_token_id == 42

    gen_binding = TowerBinding(
        transport=None, primary_coord=0, gen_coord=1, num_layers=2, block_size=4,
        target_pool=gen_pool, target_device="cpu",
        allocate_blocks=lambda n: [gen_free.pop(0) for _ in range(n)],
    )
    gen = DataPlaneTowerHandoff(data_plane=plane, bind=lambda: gen_binding)
    replica = gen.stage_conditioning(restored)
    assert replica.pool is gen_pool
    for layer in range(2):
        ks, vs = src_pool.read(layer, source.block_ids, start=0, length=4)
        kr, vr = gen_pool.read(layer, replica.block_ids, start=0, length=4)
        assert torch.equal(ks, kr)
        assert torch.equal(vs, vr)


def test_commit_latent_round_trips_over_the_data_plane():
    binding = TowerBinding(
        transport=None, primary_coord=0, gen_coord=1, num_layers=1, block_size=4,
        target_pool=_pool(), target_device="cpu", allocate_blocks=lambda n: list(range(n)),
    )
    plane = _FakeDataPlane()
    handoff = DataPlaneTowerHandoff(data_plane=plane, bind=lambda: binding)
    latent = torch.randn(1, 3, 8, 8)
    loc = handoff.writeback_commit(latent, device="cpu", dtype=torch.float32)
    assert torch.equal(plane.fetch(loc), latent)
