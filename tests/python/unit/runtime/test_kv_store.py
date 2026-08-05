from __future__ import annotations

import pytest
import torch

from uniserve_worker.batch import (
    Admission,
    DeviceDim,
    DType,
    FixedPoint,
    KvAllocation,
    PointRange,
    ProductKind,
    ProductRef,
    RequestKey,
    ShapeBound,
    StorageClass,
    UndAdmission,
    VersionRef,
)
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.kv_store import KvExtents, KvSnapshot, KvStore
from uniserve_worker.runtime.transfer import LocalTransport, Locator


def _pool(*, layers: int = 2, branch_blocks: int = 0) -> PagedKVPool:
    return PagedKVPool(
        num_layers=layers,
        num_blocks=8,
        branch_blocks=branch_blocks,
        block_size=2,
        num_kv_heads=1,
        head_dim=1,
        device="cpu",
        dtype=torch.float32,
    )


def _admit(
    store: KvStore,
    session_id: int,
    *,
    block_ids: tuple[int, ...],
    prefix_len: int,
) -> RequestKey:
    request_key = RequestKey(0, session_id, 1)
    store.admit(
        Admission.create(
            request_key,
            und=UndAdmission(kv=KvAllocation(block_ids=block_ids, prefix_len=prefix_len)),
        )
    )
    return request_key


def _version(request_key: RequestKey, op_id: int, point: int, byte: str) -> VersionRef:
    return VersionRef(request_key, op_id, FixedPoint(point, byte * 64))


def _publication_product(request_key: RequestKey, op_id: int) -> ProductRef:
    return ProductRef(
        request_key=request_key,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id,
        kind=ProductKind.KV,
        storage_class=StorageClass.PAGED_KV,
        dtype=DType.U8,
        shape_bound=ShapeBound((DeviceDim(1 << 20),)),
        point_range=PointRange(),
    )


def _latent_product(request_key: RequestKey, op_id: int, generation: int) -> ProductRef:
    return ProductRef(
        request_key=request_key,
        producer_op_id=op_id,
        output_index=0,
        generation=generation,
        kind=ProductKind.LATENT,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(4),)),
        point_range=PointRange(),
    )


def test_scratch_branches_follow_exact_latent_product_ownership() -> None:
    pool = _pool(layers=1, branch_blocks=4)
    store = KvStore(pool)
    request_key = _admit(store, 1, block_ids=(0,), prefix_len=0)
    first = _latent_product(request_key, op_id=10, generation=7)
    second = _latent_product(request_key, op_id=11, generation=7)

    store.scratch_entry(first, "text", capacity_tokens=3, copy_conditioning=False)
    store.scratch_entry(second, "text", capacity_tokens=3, copy_conditioning=False)
    assert store.scratch_token_count() == 8
    assert pool.branch_blocks_available == 0

    store.release_operations(((request_key, 10),))
    assert store.scratch_token_count() == 4
    assert pool.branch_blocks_available == 2

    store.release_operations(((request_key, 11),))
    assert pool.branch_blocks_available == 4


def test_scratch_branches_rebind_to_a_successor_and_release_at_flow_end() -> None:
    pool = _pool(layers=1, branch_blocks=4)
    store = KvStore(pool)
    request_key = _admit(store, 1, block_ids=(0,), prefix_len=0)
    source = _latent_product(request_key, op_id=10, generation=7)
    successor = _latent_product(request_key, op_id=11, generation=8)

    entry, created = store.scratch_entry(
        source,
        "text",
        capacity_tokens=3,
        copy_conditioning=False,
    )
    store.rebind_scratch_owner(source, successor)
    rebound, rebound_created = store.scratch_entry(
        successor,
        "text",
        capacity_tokens=3,
        copy_conditioning=False,
    )

    assert created
    assert not rebound_created
    assert rebound is entry
    assert pool.branch_blocks_available == 2
    store.release_scratch_owner(successor)
    assert pool.branch_blocks_available == 4


def test_committed_scratch_snapshot_restores_exact_latent_owner() -> None:
    pool = _pool(layers=1, branch_blocks=4)
    store = KvStore(pool)
    request_key = _admit(store, 1, block_ids=(0,), prefix_len=0)
    owner = _latent_product(request_key, op_id=10, generation=7)
    store.scratch_entry(owner, "text", capacity_tokens=3, copy_conditioning=False)
    state = store.snapshot_committed({1})

    assert state[0].branches[0].owner == owner
    store.release_operations(((request_key, 10),))
    assert pool.branch_blocks_available == 4

    store.restore_committed(state, {1})
    assert store.snapshot_committed({1})[0].branches[0].owner == owner
    assert pool.branch_blocks_available == 2


def test_committed_snapshot_replaces_session_scratch_state() -> None:
    pool = _pool(layers=1, branch_blocks=4)
    store = KvStore(pool)
    request_key = _admit(store, 1, block_ids=(0,), prefix_len=0)
    committed_owner = _latent_product(request_key, op_id=10, generation=7)
    replacement_owner = _latent_product(request_key, op_id=11, generation=8)
    store.scratch_entry(committed_owner, "text", capacity_tokens=3, copy_conditioning=False)
    state = store.snapshot_committed({1})
    store.release_operations(((request_key, 10),))
    store.scratch_entry(replacement_owner, "text", capacity_tokens=3, copy_conditioning=False)

    store.restore_committed(state, {1})

    branches = store.snapshot_committed({1})[0].branches
    assert tuple(branch.owner for branch in branches) == (committed_owner,)
    assert store.scratch_token_count() == 4
    assert pool.branch_blocks_available == 2


def test_kv_view_uses_visible_extents_and_device_append_offsets() -> None:
    pool = _pool(layers=1)
    store = KvStore(pool)
    _admit(store, 1, block_ids=(0, 1), prefix_len=1)
    _admit(store, 2, block_ids=(2, 3), prefix_len=2)
    view = store.view((1, 2), query_lens=(1, 2))

    block_table = view.block_table(torch.device("cpu"))
    cache_seqlens = view.cache_seqlens(torch.device("cpu"))
    values = torch.tensor((10.0, 20.0, 30.0)).reshape(3, 1, 1)
    view.append_varlen(
        0,
        values,
        -values,
        (1, 2),
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        query_offsets=torch.tensor((0, 1, 3), dtype=torch.int32),
    )

    assert view.base_lens == (1, 2)
    torch.testing.assert_close(block_table, torch.tensor(((0, 1), (2, 3)), dtype=torch.int32))
    torch.testing.assert_close(cache_seqlens, torch.tensor((1, 2), dtype=torch.int32))
    first_key, first_value = pool.read(0, [0, 1], start=1, length=1)
    second_key, second_value = pool.read(0, [2, 3], start=2, length=2)
    assert first_key is not None and first_value is not None
    assert second_key is not None and second_value is not None
    torch.testing.assert_close(first_key, values[:1])
    torch.testing.assert_close(first_value, -values[:1])
    torch.testing.assert_close(second_key, values[1:])
    torch.testing.assert_close(second_value, -values[1:])


def test_five_extents_select_and_commit_one_initialized_prefix() -> None:
    store = KvStore(_pool(layers=1))
    _admit(store, 1, block_ids=(0, 1, 2), prefix_len=2)

    initialized = store.initialize(1, 3)
    assert initialized == 5
    assert store.get(1).extents() == KvExtents(6, 5, 2, 2, 0)

    store.select(1, 4)
    assert store.get(1).extents() == KvExtents(6, 5, 4, 2, 0)
    store.commit(1, 4)
    assert store.get(1).extents() == KvExtents(6, 5, 4, 4, 0)


def test_step_rollback_restores_every_extent_and_mapping() -> None:
    store = KvStore(_pool(layers=1))
    _admit(store, 1, block_ids=(0, 1), prefix_len=2)
    baseline = store.get(1).extents()
    transaction = store.begin_step({1})

    transaction.initialize(1, 2)
    transaction.select(1, 3)
    transaction.rollback()

    assert store.get(1).extents() == baseline
    assert store.get(1).block_ids == [0, 1]


def test_step_rollback_restores_the_exact_scratch_owner_and_capacity() -> None:
    pool = _pool(layers=1, branch_blocks=4)
    store = KvStore(pool)
    request_key = _admit(store, 1, block_ids=(0,), prefix_len=0)
    committed_owner = _latent_product(request_key, op_id=10, generation=7)
    provisional_owner = _latent_product(request_key, op_id=11, generation=8)
    store.scratch_entry(committed_owner, "text", capacity_tokens=3, copy_conditioning=False)
    transaction = store.begin_step({1})

    transaction.scratch_entry(
        provisional_owner,
        "text",
        capacity_tokens=3,
        copy_conditioning=False,
    )
    transaction.rollback()

    branches = store.snapshot_committed({1})[0].branches
    assert tuple(branch.owner for branch in branches) == (committed_owner,)
    assert store.scratch_token_count() == 4
    assert pool.branch_blocks_available == 2


def test_publication_is_incremental_immutable_and_exact_base_bound() -> None:
    pool = _pool()
    store = KvStore(pool)
    request_key = _admit(store, 1, block_ids=(0, 1, 2), prefix_len=2)
    transport = LocalTransport(byte_capacity=1 << 20)
    for layer in range(pool.num_layers):
        prefix = torch.tensor((1.0, 2.0)).reshape(2, 1, 1) + layer * 10
        pool.write(layer, [0, 1, 2], start=0, k=prefix, v=-prefix)

    first_version = _version(request_key, 10, 1, "a")
    first_product = _publication_product(request_key, 20)
    first = store.publish(
        1,
        source_version=first_version,
        source_digest="a" * 64,
        destination="gen",
        expected_base=None,
        product=first_product,
        transport=transport,
    )
    first_values = tuple(
        transport.fetch(Locator.from_wire_json(value)).clone() for value in first.locators
    )

    for layer in range(pool.num_layers):
        suffix = torch.tensor((3.0,)).reshape(1, 1, 1) + layer * 10
        pool.write(layer, [0, 1, 2], start=2, k=suffix, v=-suffix)
    store.advance(1, 1)
    store.commit(1, 3)
    second_version = _version(request_key, 11, 1, "b")
    second_product = _publication_product(request_key, 21)
    second = store.publish(
        1,
        source_version=second_version,
        source_digest="b" * 64,
        destination="gen",
        expected_base=first_version,
        product=second_product,
        transport=transport,
    )

    assert (first.base_extent, first.published_extent) == (0, 2)
    assert (second.base_extent, second.published_extent) == (2, 3)
    assert second.base_version == first_version
    assert all(Locator.from_wire_json(value).shape[0] == 1 for value in second.locators)
    for encoded, expected in zip(first.locators, first_values, strict=True):
        torch.testing.assert_close(transport.fetch(Locator.from_wire_json(encoded)), expected)
    assert store.destination_base(1, "gen") == second_version
    assert store.validate_conditioning(1, second_product) == second
    assert store.get(1).extents() == KvExtents(6, 3, 3, 3, 3)

    with pytest.raises(WorkerError, match="expected base does not match destination"):
        store.publish(
            1,
            source_version=_version(request_key, 12, 1, "c"),
            source_digest="c" * 64,
            destination="gen",
            expected_base=first_version,
            product=_publication_product(request_key, 22),
            transport=transport,
        )


def test_publication_preserves_the_committed_watermark_for_a_provisional_suffix() -> None:
    store = KvStore(_pool(layers=1))
    request_key = _admit(store, 1, block_ids=(0, 1), prefix_len=2)
    transport = LocalTransport(byte_capacity=1 << 20)
    committed_version = _version(request_key, 9, 1, "9")
    store.publish(
        1,
        source_version=committed_version,
        source_digest="9" * 64,
        destination="gen",
        expected_base=None,
        product=_publication_product(request_key, 19),
        transport=transport,
    )
    store.initialize(1, 2)
    store.select(1, 3)

    with pytest.raises(WorkerError, match="not the committed visible version"):
        store.publish(
            1,
            source_version=_version(request_key, 10, 1, "a"),
            source_digest="a" * 64,
            destination="gen",
            expected_base=committed_version,
            product=_publication_product(request_key, 20),
            transport=transport,
        )

    assert store.get(1).extents() == KvExtents(4, 4, 3, 2, 2)
    assert store.destination_base(1, "gen") == committed_version


def test_publication_descriptor_requires_an_exact_fixed_digest() -> None:
    request_key = RequestKey(0, 1, 1)
    with pytest.raises(WorkerError, match="source digest is not exact"):
        KvSnapshot(
            locators=(),
            source_version=_version(request_key, 10, 1, "a"),
            source_digest="b" * 64,
            destination="gen",
            base_version=None,
            base_extent=0,
            published_extent=0,
            block_ids=(0,),
            group_id=0,
            mapping_generation=1,
            scale_identity="float32",
        )


def test_drop_releases_every_immutable_transport_entry() -> None:
    class RecordingTransport(LocalTransport):
        def __init__(self) -> None:
            super().__init__(byte_capacity=1 << 20)
            self.released: list[Locator] = []

        def release(self, locator: Locator) -> None:
            self.released.append(locator)
            super().release(locator)

    store = KvStore(_pool(layers=1))
    request_key = _admit(store, 1, block_ids=(0,), prefix_len=1)
    transport = RecordingTransport()
    snapshot = store.publish(
        1,
        source_version=_version(request_key, 10, 1, "a"),
        source_digest="a" * 64,
        destination="gen",
        expected_base=None,
        product=_publication_product(request_key, 20),
        transport=transport,
    )

    store.drop(1)

    assert tuple(locator.to_wire_json() for locator in transport.released) == snapshot.locators
