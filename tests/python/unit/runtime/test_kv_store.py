from __future__ import annotations

import pytest
import torch

from uniserve_worker.batch import (
    Admission,
    DeviceDim,
    DType,
    FixedPoint,
    KvAdmission,
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
from uniserve_worker.runtime.transfer import LocalTransport

pytestmark = pytest.mark.unit


def _pool(*, blocks: int = 8, layers: int = 1, branch_blocks: int = 0) -> PagedKVPool:
    return PagedKVPool(
        num_layers=layers,
        num_blocks=blocks,
        branch_blocks=branch_blocks,
        block_size=2,
        num_kv_heads=1,
        head_dim=1,
        device="cpu",
        dtype=torch.float32,
    )


def _admission(session_id: int, *, prefix_len: int = 0, group_id: int = 0) -> Admission:
    return Admission.create(
        RequestKey(0, session_id, 1),
        request_pool_idx=session_id + 1,
        und=UndAdmission(kv=KvAdmission(prefix_len=prefix_len, group_id=group_id)),
    )


def _register(
    store: KvStore,
    session_id: int,
    logical_blocks: tuple[int, ...],
    *,
    prefix_len: int = 0,
    group_id: int = 0,
) -> RequestKey:
    admission = _admission(session_id, prefix_len=prefix_len, group_id=group_id)
    transaction = store.begin_step({session_id})
    store.admit(admission)
    transaction.reserve_logical_page_delta(
        admission.request_key,
        logical_blocks,
        expected_capacity_pages=len(logical_blocks),
    )
    transaction.finalize()
    return admission.request_key


def _version(request_key: RequestKey, op_id: int, point: int, byte: str) -> VersionRef:
    return VersionRef(request_key, op_id, FixedPoint(point, byte * 64))


def _product(
    request_key: RequestKey,
    op_id: int,
    *,
    kind: ProductKind,
    storage_class: StorageClass,
) -> ProductRef:
    return ProductRef(
        request_key=request_key,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id,
        kind=kind,
        storage_class=storage_class,
        dtype=DType.U8 if kind is ProductKind.KV else DType.BF16,
        shape_bound=ShapeBound((DeviceDim(1 << 20),)),
        point_range=PointRange(),
    )


def _write(store: KvStore, session_id: int, values: tuple[float, ...]) -> None:
    pool = store.pool
    assert pool is not None
    entry = store.get(session_id)
    tensor = torch.tensor(values, dtype=pool.dtype).reshape(-1, 1, 1)
    for layer in range(pool.num_layers):
        pool.write(layer, entry.block_ids, start=entry.visible_len, k=tensor, v=-tensor)
    store.advance(session_id, len(values))
    store.commit(session_id, entry.visible_len)


def test_registration_maps_logical_leases_inside_each_worker_pool() -> None:
    pool = _pool()
    store = KvStore(pool)
    _register(store, 1, (0, 1), prefix_len=2, group_id=7)
    _register(store, 2, (0, 2), prefix_len=2, group_id=7)
    conflicting = _admission(3, group_id=8)
    transaction = store.begin_step({3})
    store.admit(conflicting)
    with pytest.raises(WorkerError, match="another cache group"):
        transaction.reserve_logical_page_delta(
            conflicting.request_key,
            (0,),
            expected_capacity_pages=1,
        )
    transaction.rollback()
    _register(store, 3, (3,), group_id=8)

    first = store.get(1)
    second = store.get(2)
    other_group = store.get(3)
    assert first.logical_blocks == [0, 1]
    assert second.logical_blocks == [0, 2]
    assert first.block_ids[0] == second.block_ids[0]
    assert first.block_ids[1] != second.block_ids[1]
    assert other_group.block_ids[0] not in first.block_ids + second.block_ids
    assert store.resident_block_count() == 4
    assert pool.session_blocks_available == pool.leasable_num_blocks - 4


def test_writable_logical_page_cannot_be_shared_between_sessions() -> None:
    store = KvStore(_pool())
    _register(store, 1, (4,))
    admission = _admission(2)
    transaction = store.begin_step({2})
    store.admit(admission)

    with pytest.raises(WorkerError, match="writable KV block"):
        transaction.reserve_logical_page_delta(
            admission.request_key,
            (4,),
            expected_capacity_pages=1,
        )
    transaction.rollback()

    with pytest.raises(WorkerError, match="has no KV state"):
        store.get(2)
    assert store.resident_block_count() == 1


def test_reused_logical_prefix_retains_its_worker_page_contents() -> None:
    pool = _pool()
    store = KvStore(pool)
    _register(store, 1, (4,))
    _write(store, 1, (3.0, 4.0))
    page = store.get(1).block_ids[0]
    store.drop(1)

    assert store.resident_block_count() == 0
    _register(store, 2, (4,), prefix_len=2)
    assert store.get(2).block_ids == [page]
    key, value = pool.read(0, [page], start=0, length=2)
    assert key is not None and value is not None
    torch.testing.assert_close(key.flatten(), torch.tensor((3.0, 4.0)))
    torch.testing.assert_close(value.flatten(), torch.tensor((-3.0, -4.0)))


def test_scheduler_zero_set_clears_reassigned_page_contents() -> None:
    pool = _pool(blocks=1)
    store = KvStore(pool)
    first = _admission(1)
    first_transaction = store.begin_step({1})
    store.admit(first)
    first_transaction.apply_placement(
        first.request_key,
        group_id=0,
        block_table=(0,),
        pages_to_zero=(0,),
        expected_capacity_pages=1,
    )
    first_transaction.finalize()
    _write(store, 1, (3.0, 4.0))
    store.drop(1)

    second = _admission(2)
    second_transaction = store.begin_step({2})
    store.admit(second)
    second_transaction.apply_placement(
        second.request_key,
        group_id=0,
        block_table=(0,),
        pages_to_zero=(0,),
        expected_capacity_pages=1,
    )
    second_transaction.finalize()

    page = store.get(2).block_ids[0]
    assert torch.count_nonzero(pool.k[:, page]).item() == 0
    assert torch.count_nonzero(pool.v[:, page]).item() == 0


def test_registration_exhaustion_rolls_back_pages_and_session_atomically() -> None:
    pool = _pool(blocks=2)
    store = KvStore(pool)
    pool.allocate_session_blocks(1)
    available = pool.session_blocks_available
    admission = _admission(1)
    transaction = store.begin_step({1})
    store.admit(admission)

    with pytest.raises(RuntimeError, match="session KV pages exhausted"):
        transaction.reserve_logical_page_delta(
            admission.request_key,
            (0, 1),
            expected_capacity_pages=2,
        )
    transaction.rollback()

    assert pool.session_blocks_available == available
    assert store.resident_block_count() == 0
    with pytest.raises(WorkerError, match="has no KV state"):
        store.get(1)


def test_transaction_rollback_restores_extents_and_logical_mapping() -> None:
    pool = _pool()
    store = KvStore(pool)
    key = _register(store, 1, (0,), prefix_len=1)
    baseline = store.get(1).extents()
    available = pool.session_blocks_available
    transaction = store.begin_step({1})

    transaction.reserve_logical_page_delta(
        key,
        (1,),
        expected_capacity_pages=2,
    )
    transaction.initialize(1, 2)
    transaction.select(1, 2)
    transaction.rollback()

    entry = store.get(1)
    assert entry.logical_blocks == [0]
    assert entry.extents() == baseline
    assert pool.session_blocks_available == available


def test_kv_view_writes_at_visible_extents_through_worker_mappings() -> None:
    pool = _pool()
    store = KvStore(pool)
    _register(store, 1, (0, 1), prefix_len=1)
    _register(store, 2, (2, 3), prefix_len=2)
    view = store.view((1, 2), query_lens=(1, 2))
    values = torch.tensor((10.0, 20.0, 30.0)).reshape(3, 1, 1)
    table = view.block_table(torch.device("cpu"))
    lengths = view.cache_seqlens(torch.device("cpu"))

    view.append_varlen(
        0,
        values,
        -values,
        (1, 2),
        block_table=table,
        cache_seqlens=lengths,
        query_offsets=torch.tensor((0, 1, 3), dtype=torch.int32),
    )

    assert view.base_lens == (1, 2)
    torch.testing.assert_close(table[0], torch.tensor(store.get(1).block_ids, dtype=torch.int32))
    torch.testing.assert_close(table[1], torch.tensor(store.get(2).block_ids, dtype=torch.int32))
    first_key, first_value = pool.read(0, store.get(1).block_ids, start=1, length=1)
    second_key, second_value = pool.read(0, store.get(2).block_ids, start=2, length=2)
    assert first_key is not None and first_value is not None
    assert second_key is not None and second_value is not None
    torch.testing.assert_close(first_key, values[:1])
    torch.testing.assert_close(first_value, -values[:1])
    torch.testing.assert_close(second_key, values[1:])
    torch.testing.assert_close(second_value, -values[1:])


def test_extent_transitions_select_one_initialized_prefix() -> None:
    store = KvStore(_pool())
    _register(store, 1, (0, 1, 2), prefix_len=2)

    assert store.initialize(1, 3) == 5
    assert store.get(1).extents() == KvExtents(6, 5, 2, 2, 0)
    store.select(1, 4)
    store.commit(1, 4)
    assert store.get(1).extents() == KvExtents(6, 5, 4, 4, 0)


def test_incremental_publication_binds_logical_lease_and_exact_base() -> None:
    store = KvStore(_pool(layers=2))
    key = _register(store, 1, (0, 1, 2))
    transport = LocalTransport(byte_capacity=1 << 20)
    _write(store, 1, (1.0, 2.0))
    first_version = _version(key, 10, 1, "a")
    first_product = _product(
        key,
        20,
        kind=ProductKind.KV,
        storage_class=StorageClass.PAGED_KV,
    )
    first = store.publish(
        1,
        source_version=first_version,
        source_digest="a" * 64,
        destination="generation",
        expected_base=None,
        product=first_product,
        transport=transport,
    )
    _write(store, 1, (3.0,))
    second_version = _version(key, 11, 2, "b")
    second_product = _product(
        key,
        21,
        kind=ProductKind.KV,
        storage_class=StorageClass.PAGED_KV,
    )
    second = store.publish(
        1,
        source_version=second_version,
        source_digest="b" * 64,
        destination="generation",
        expected_base=first_version,
        product=second_product,
        transport=transport,
    )

    assert (first.base_extent, first.published_extent) == (0, 2)
    assert (second.base_extent, second.published_extent) == (2, 3)
    assert first.logical_blocks == second.logical_blocks == (0, 1, 2)
    assert store.validate_conditioning(1, second_product) == second
    with pytest.raises(WorkerError, match="expected base does not match destination"):
        store.publish(
            1,
            source_version=_version(key, 12, 3, "c"),
            source_digest="c" * 64,
            destination="generation",
            expected_base=first_version,
            product=_product(
                key,
                22,
                kind=ProductKind.KV,
                storage_class=StorageClass.PAGED_KV,
            ),
            transport=transport,
        )


def test_failed_publication_returns_all_transport_capacity() -> None:
    store = KvStore(_pool(layers=2))
    key = _register(store, 1, (0,))
    _write(store, 1, (1.0, 2.0))
    transport = LocalTransport(byte_capacity=24)

    with pytest.raises(WorkerError, match="transfer byte capacity"):
        store.publish(
            1,
            source_version=_version(key, 10, 1, "a"),
            source_digest="a" * 64,
            destination="generation",
            expected_base=None,
            product=_product(
                key,
                20,
                kind=ProductKind.KV,
                storage_class=StorageClass.PAGED_KV,
            ),
            transport=transport,
        )

    assert store.published_locator_count() == 0
    probe = transport.publish(torch.ones(6, dtype=torch.float32))
    transport.release(probe)


def test_snapshot_install_uses_destination_physical_mapping() -> None:
    source = KvStore(_pool())
    key = _register(source, 1, (0, 1))
    _write(source, 1, (4.0, 5.0, 6.0))
    transport = LocalTransport(byte_capacity=1 << 20)
    source_product = _product(
        key,
        20,
        kind=ProductKind.KV,
        storage_class=StorageClass.PAGED_KV,
    )
    snapshot = source.publish(
        1,
        source_version=_version(key, 10, 1, "a"),
        source_digest="a" * 64,
        destination="generation",
        expected_base=None,
        product=source_product,
        transport=transport,
    )

    destination = KvStore(_pool())
    _register(destination, 9, (7,))
    _register(destination, 1, (*snapshot.logical_blocks, 2))
    destination.stage_publication(source_product, snapshot)
    installed_product = _product(
        key,
        21,
        kind=ProductKind.KV,
        storage_class=StorageClass.PAGED_KV,
    )
    installed = destination.install_publication(
        1,
        source_product,
        installed_product,
        transport,
    )
    entry = destination.get(1)
    assert tuple(entry.block_ids[: len(snapshot.logical_blocks)]) != snapshot.block_ids
    assert tuple(entry.logical_blocks[: len(snapshot.logical_blocks)]) == snapshot.logical_blocks
    assert installed.logical_blocks == snapshot.logical_blocks
    assert installed.block_ids == tuple(entry.block_ids[: len(snapshot.logical_blocks)])
    assert destination.validate_installed(1, installed_product) == installed
    key_values, value_values = destination.pool.read(  # type: ignore[union-attr]
        0,
        entry.block_ids,
        start=0,
        length=3,
    )
    assert key_values is not None and value_values is not None
    torch.testing.assert_close(key_values.flatten(), torch.tensor((4.0, 5.0, 6.0)))
    torch.testing.assert_close(value_values.flatten(), torch.tensor((-4.0, -5.0, -6.0)))


def test_scratch_ownership_survives_committed_snapshot_restore() -> None:
    pool = _pool(branch_blocks=4)
    store = KvStore(pool)
    key = _register(store, 1, (0,))
    owner = _product(
        key,
        10,
        kind=ProductKind.LATENT,
        storage_class=StorageClass.LATENT_ARENA,
    )
    store.scratch_entry(owner, "text", capacity_tokens=3, copy_conditioning=False)
    committed = store.snapshot_committed({1})
    store.release_operations(((key, 10),))
    assert pool.branch_blocks_available == 4

    store.restore_committed(committed, {1})

    restored = store.snapshot_committed({1})[0]
    assert restored.logical_blocks == (0,)
    assert tuple(branch.owner for branch in restored.branches) == (owner,)
    assert pool.branch_blocks_available == 2


def test_snapshot_descriptor_requires_matching_logical_and_physical_leases() -> None:
    key = RequestKey(0, 1, 1)
    with pytest.raises(WorkerError, match="logical lease is invalid"):
        KvSnapshot(
            locators=(),
            source_version=_version(key, 10, 1, "a"),
            source_digest="a" * 64,
            destination="generation",
            base_version=None,
            base_extent=0,
            published_extent=0,
            block_ids=(0,),
            logical_blocks=(),
            group_id=0,
            mapping_generation=1,
            scale_identity="float32",
        )
