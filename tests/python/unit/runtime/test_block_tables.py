"""Page tables and allocated lengths retain independent update coordinates."""

import pytest
import torch

from uniserve_worker.storage.block_tables import BlockTables

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
@pytest.mark.parametrize("depth", (1, 3))
def test_table_updates_and_release_preserve_all_queued_snapshots(device, depth):
    tables = BlockTables(
        group_count=2,
        request_pool_size=3,
        max_blocks_per_request=3,
        block_size=4,
        device=device,
        staging_depth=depth,
    )
    expected_pages = torch.zeros((2, 4, 3), dtype=torch.int32)
    expected_lengths = torch.zeros(4, dtype=torch.int32)
    snapshots = []
    try:
        # Changed page rows and changed allocated lengths deliberately identify
        # different slot sets, including repeated slots across cache groups.
        updates = (
            ((1, 0, (2, 3), 4), (1, 1, (6, 7), 4), (2, 0, (4, 5), 4)),
            ((1, 1, (8, 9), 4), (2, 0, (4, 5), 5)),
            ((2, 0, (4, 5), 6), (3, 1, (10,), 3)),
            ((1, 0, (2, 3), 7),),
            ((3, 1, (11, 12), 3),),
        )
        for update in updates:
            tables.install(update)
            for slot, group, pages, allocated in update:
                expected_pages[group, slot].zero_()
                expected_pages[group, slot, : len(pages)] = torch.tensor(pages)
                expected_lengths[slot] = allocated
                assert tables.pages(slot, group) == pages
                assert tables.allocated_length(slot) == allocated
            # These are real consumers on the producer stream. Queue all of
            # them before host observation to exercise asynchronous source
            # retirement.
            snapshots.append(
                (
                    tables.page_tables.clone(),
                    tables.alloced_lens.clone(),
                    expected_pages.clone(),
                    expected_lengths.clone(),
                )
            )
        tables.set_verified(torch.tensor([1, 2]), torch.tensor([6, 5]))
        tables.release((1, 3))
        expected_pages[:, (1, 3)] = 0
        expected_lengths[[1, 3]] = 0
        for actual_pages, actual_lengths, pages, lengths in snapshots:
            torch.testing.assert_close(actual_pages.cpu(), pages)
            torch.testing.assert_close(actual_lengths.cpu(), lengths)
        torch.testing.assert_close(tables.page_tables.cpu(), expected_pages)
        torch.testing.assert_close(tables.alloced_lens.cpu(), expected_lengths)
        assert tables.verified_lengths.tolist() == [0, 0, 5, 0]
        assert tables.allocated_length(1) == tables.allocated_length(3) == 0
        assert tables.pages(2, 0) == (4, 5)
    finally:
        tables.close()
