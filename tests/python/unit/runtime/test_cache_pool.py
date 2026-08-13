"""Behavioral contracts for scheduler-indexed physical KV storage."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.runtime.cache_pool import CacheBatchView, CachePool, CacheRow

pytestmark = pytest.mark.unit


def _pool(*, grouped: bool = False) -> CachePool:
    return CachePool(
        num_layers=2,
        request_pages=8,
        scratch_pages=2,
        page_size=4,
        num_kv_heads=1,
        head_dim=2,
        group_ranges=((0, 4), (4, 4)) if grouped else None,
        device="cpu",
        dtype=torch.float32,
    )


def _tokens(count: int, *, offset: float = 0.0) -> torch.Tensor:
    return torch.arange(
        offset,
        offset + count * 2,
        dtype=torch.float32,
    ).reshape(count, 1, 2)


def test_zero_on_reuse_removes_all_preceding_page_content() -> None:
    pool = _pool()
    stale = _tokens(4, offset=20)
    pool.write(0, (1,), start=0, k=stale, v=stale + 100)

    pool.zero_pages(0, (1,))
    current = _tokens(1, offset=7)
    pool.write(0, (1,), start=0, k=current, v=current + 100)

    key, value = pool.read(0, (1,), start=0, length=4)
    assert key is not None and value is not None
    torch.testing.assert_close(key[:1], current)
    torch.testing.assert_close(value[:1], current + 100)
    torch.testing.assert_close(key[1:], torch.zeros_like(key[1:]))
    torch.testing.assert_close(value[1:], torch.zeros_like(value[1:]))
    assert torch.count_nonzero(pool.field(0, 0, "key")[0]) == 0
    assert torch.count_nonzero(pool.field(0, 0, "value")[0]) == 0


def test_group_commands_address_only_their_declared_physical_ranges() -> None:
    pool = _pool(grouped=True)
    first = _tokens(2, offset=1)
    second = _tokens(2, offset=50)

    pool.write(0, (1,), group=0, start=0, k=first, v=first + 10)
    pool.write(0, (4,), group=1, start=0, k=second, v=second + 10)

    first_key, _ = pool.read(0, (1,), group=0, start=0, length=2)
    second_key, _ = pool.read(0, (4,), group=1, start=0, length=2)
    assert first_key is not None and second_key is not None
    torch.testing.assert_close(first_key, first)
    torch.testing.assert_close(second_key, second)
    with pytest.raises(Exception, match="another cache group"):
        pool.zero_pages(1, (1,))


def test_copy_and_restore_target_only_the_named_group_and_pages() -> None:
    pool = _pool(grouped=True)
    source = _tokens(4, offset=3)
    pool.write(1, (1,), group=0, start=0, k=source, v=source + 100)

    pool.copy_pages(0, (1,), (2,))
    copied_key, copied_value = pool.read(1, (2,), group=0, start=0, length=4)
    assert copied_key is not None and copied_value is not None
    torch.testing.assert_close(copied_key, source)
    torch.testing.assert_close(copied_value, source + 100)

    payload = pool.page_view(0, (1,))
    pool.restore_pages(1, (4,), payload)
    restored_key, restored_value = pool.read(1, (4,), group=1, start=0, length=4)
    assert restored_key is not None and restored_value is not None
    torch.testing.assert_close(restored_key, source)
    torch.testing.assert_close(restored_value, source + 100)


def test_batch_view_derives_attention_metadata_and_writes_from_scheduler_rows() -> None:
    pool = _pool()
    rows = (CacheRow((1,), 2, 4), CacheRow((2,), 1, 4))
    view = CacheBatchView(pool, rows, query_lengths=(1, 2))

    block_table = view.block_table(torch.device("cpu"))
    assert tuple(block_table[:, 0].tolist()) == (1, 2)
    assert tuple(view.cache_seqlens(torch.device("cpu")).tolist()) == (2, 1)
    page_ids, page_offsets, token_indices = view.write_plan(torch.device("cpu"))
    assert tuple(page_ids.tolist()) == (1, 2, 2)
    assert tuple(page_offsets.tolist()) == (2, 1, 2)
    assert tuple(token_indices.tolist()) == (0, 1, 2)

    key = _tokens(3, offset=9)
    view.append_packed(
        0,
        key,
        key + 100,
        page_ids=page_ids,
        page_offsets=page_offsets,
        token_indices=token_indices,
    )
    first, _ = pool.read(0, (1,), start=2, length=1)
    second, _ = pool.read(0, (2,), start=1, length=2)
    assert first is not None and second is not None
    torch.testing.assert_close(first, key[:1])
    torch.testing.assert_close(second, key[1:])


def test_varlen_batch_write_consumes_staged_physical_indices() -> None:
    pool = _pool()
    view = CacheBatchView(
        pool,
        (CacheRow((1,), 0, 4), CacheRow((2,), 0, 4)),
        query_lengths=(1, 2),
    )
    key = _tokens(3, offset=12)

    view.append_varlen(
        0,
        key,
        key + 100,
        (1, 2),
        block_table=torch.tensor(((3,), (4,)), dtype=torch.int32),
        cache_seqlens=torch.tensor((1, 0), dtype=torch.int32),
        query_offsets=torch.tensor((0, 1, 3), dtype=torch.int32),
    )

    first, _ = pool.read(0, (3,), start=1, length=1)
    second, _ = pool.read(0, (4,), start=0, length=2)
    assert first is not None and second is not None
    torch.testing.assert_close(first, key[:1])
    torch.testing.assert_close(second, key[1:])


def test_repeated_page_zero_is_valid_only_for_padded_attention_rows() -> None:
    pool = _pool()
    live = CacheBatchView(pool, (CacheRow((1,), 0, 4),), query_lengths=(1,))

    padded = live.with_synthetic_row((0, 0), base_len=0, query_len=5)

    assert tuple(padded.block_table(torch.device("cpu"))[1].tolist()) == (0, 0)
    with pytest.raises(Exception, match="repeats a physical page"):
        CacheBatchView(pool, (CacheRow((1, 1), 0, 8),), query_lengths=(1,))


def test_invalid_page_commands_fail_before_mutating_valid_storage() -> None:
    pool = _pool()
    source = _tokens(2, offset=4)
    pool.write(0, (1,), start=0, k=source, v=source + 1)

    with pytest.raises(Exception, match="fixed physical pool"):
        pool.zero_pages(0, (pool.num_pages,))
    with pytest.raises(Exception, match="repeats a physical page"):
        pool.copy_pages(0, (1, 1), (2, 3))

    key, value = pool.read(0, (1,), start=0, length=2)
    assert key is not None and value is not None
    torch.testing.assert_close(key, source)
    torch.testing.assert_close(value, source + 1)
