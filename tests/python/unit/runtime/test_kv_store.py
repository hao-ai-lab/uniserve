from __future__ import annotations

import pytest
import torch

from uniserve_worker.batch import Admission, KvAllocation, SequenceAdmission
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.kv_store import KvStore
from uniserve_worker.runtime.transfer import LocalTransport


def test_bounded_kv_view_stages_rows_and_writes_ragged_tokens() -> None:
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=2,
        num_kv_heads=1,
        head_dim=1,
        device="cpu",
        dtype=torch.float32,
    )
    store = KvStore(pool)
    store.admit(
        Admission.create(
            1,
            sequence=SequenceAdmission(kv=KvAllocation(block_ids=(0, 1), prefix_len=1)),
        )
    )
    store.admit(
        Admission.create(
            2,
            sequence=SequenceAdmission(kv=KvAllocation(block_ids=(2, 3), prefix_len=2)),
        )
    )
    view = store.view((1, 2), query_lens=(1, 2))

    block_table = view.block_table(torch.device("cpu"))
    cache_seqlens = view.cache_seqlens(torch.device("cpu"))
    query_offsets = torch.tensor((0, 1, 3), dtype=torch.int32)
    values = torch.tensor((10.0, 20.0, 30.0)).reshape(3, 1, 1)
    view.append_varlen(
        0,
        values,
        -values,
        (1, 2),
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        query_offsets=query_offsets,
    )

    assert view.base_lens == (1, 2)
    assert view.supports_paged_attention_storage is True
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


def test_snapshot_import_accepts_the_sessions_own_partial_page() -> None:
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=2,
        num_kv_heads=1,
        head_dim=1,
        device="cpu",
        dtype=torch.float32,
    )
    store = KvStore(pool)
    store.admit(
        Admission.create(
            1,
            sequence=SequenceAdmission(kv=KvAllocation(block_ids=(0, 1), prefix_len=0)),
        )
    )
    store.advance(1, 3)
    transport = LocalTransport()
    snapshot = store.publish(
        1,
        source_version=1,
        position=3,
        transport=transport,
    )

    store.import_snapshot(1, snapshot, transport)

    entry = store.get(1)
    assert entry.block_ids == [0, 1]
    assert entry.length == 3


def test_snapshot_import_rejects_another_sessions_partial_page() -> None:
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=2,
        num_kv_heads=1,
        head_dim=1,
        device="cpu",
        dtype=torch.float32,
    )
    store = KvStore(pool)
    for session_id, blocks in ((1, (0, 1)), (2, (2, 3))):
        store.admit(
            Admission.create(
                session_id,
                sequence=SequenceAdmission(
                    kv=KvAllocation(block_ids=blocks, prefix_len=0)
                ),
            )
        )
    store.advance(1, 3)
    transport = LocalTransport()
    snapshot = store.publish(
        1,
        source_version=1,
        position=3,
        transport=transport,
    )

    with pytest.raises(WorkerError, match="writable KV block 1 held by another session"):
        store.import_snapshot(2, snapshot, transport)


def _published_store() -> tuple[KvStore, LocalTransport]:
    pool = PagedKVPool(
        num_layers=2,
        num_blocks=4,
        block_size=2,
        num_kv_heads=1,
        head_dim=1,
        device="cpu",
        dtype=torch.float32,
    )
    store = KvStore(pool)
    store.admit(
        Admission.create(
            1,
            sequence=SequenceAdmission(kv=KvAllocation(block_ids=(0, 1), prefix_len=2)),
        )
    )
    return store, LocalTransport()


def test_a_new_publication_returns_the_copies_the_previous_one_holds() -> None:
    store, transport = _published_store()

    first = store.publish(1, source_version=1, position=2, transport=transport)
    second = store.publish(1, source_version=2, position=2, transport=transport)

    assert first.locators and second.locators
    assert len(transport._table) == len(second.locators)


def test_dropping_a_session_returns_the_copies_its_publication_holds() -> None:
    store, transport = _published_store()
    store.publish(1, source_version=1, position=2, transport=transport)
    assert transport._table

    store.drop(1)

    assert not transport._table
