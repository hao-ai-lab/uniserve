from __future__ import annotations

import pytest

from uniserve_worker.runtime.block_allocator import BlockFreeList

pytestmark = pytest.mark.unit


def test_block_free_list_allocates_lowest_available_ids_and_reuses_released_ids():
    allocator = BlockFreeList(5)

    assert allocator.allocate(2, label="scratch") == [0, 1]
    allocator.release([1, 0, 1])

    assert allocator.allocate(3, label="scratch") == [0, 1, 2]
    assert allocator.available == 2


def test_block_free_list_noops_for_non_positive_allocations():
    allocator = BlockFreeList(2)

    assert allocator.allocate(0) == []
    assert allocator.allocate(-4) == []
    assert allocator.available == 2


def test_block_free_list_reports_exhaustion_with_pool_label():
    allocator = BlockFreeList(1)

    with pytest.raises(RuntimeError, match="SenseNova scratch KV pool exhausted"):
        allocator.allocate(2, label="SenseNova scratch KV pool")
