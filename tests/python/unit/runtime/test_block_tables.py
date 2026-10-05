"""Unit tables, start pages and allocated lengths keep their coordinates."""

import pytest
import torch

from uniserve_worker.errors import WorkerError
from uniserve_worker.storage.block_tables import BlockTables, GroupShape

pytestmark = pytest.mark.unit

# A full group with one-unit pages of four tokens and a windowed group whose
# eight-token pages span two units, so the tables are (group 0), (group 1,
# unit 0) and (group 1, unit 1).
GROUPS = (GroupShape(4, 1), GroupShape(8, 2, window=8))


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
@pytest.mark.parametrize("depth", (1, 3))
def test_table_updates_and_release_preserve_all_queued_snapshots(device, depth):
    tables = BlockTables(
        groups=GROUPS,
        num_units=16,
        request_pool_size=3,
        width=3,
        device=device,
        host_buffer_depth=depth,
    )
    expected_units = torch.zeros((3, 4, 3), dtype=torch.int32)
    expected_starts = torch.zeros((2, 4), dtype=torch.int32)
    expected_lengths = torch.zeros(4, dtype=torch.int32)
    installed = {}
    snapshots = []
    try:
        # Changed unit rows, start pages and allocated lengths deliberately
        # identify different slot sets, including repeated slots across
        # groups and a windowed table whose start page advances.
        updates = (
            ((1, 0, 0, (2, 3), 8), (1, 1, 0, (6, 7), 8), (2, 0, 0, (4, 5), 8)),
            ((1, 1, 1, (8, 9), 16), (2, 0, 0, (4, 5), 6)),
            ((2, 0, 0, (4, 5, 13), 12), (3, 1, 0, (10, 11), 5)),
            ((1, 0, 0, (2, 3), 7),),
            ((3, 1, 2, (14, 15), 20),),
        )
        for update in updates:
            tables.install(update)
            for slot, group, start, units, allocated in update:
                installed[(slot, group)] = allocated
                if group == 0:
                    expected_units[0, slot].zero_()
                    expected_units[0, slot, : len(units)] = torch.tensor(units)
                else:
                    # Page-major units of a two-unit page split across the
                    # group's two tables.
                    for position in range(2):
                        row = units[position::2]
                        expected_units[1 + position, slot].zero_()
                        expected_units[1 + position, slot, : len(row)] = (
                            torch.tensor(row)
                        )
                expected_starts[group, slot] = start
                table = tables.table(slot, group)
                assert (table.start_page, table.units) == (start, units)
                assert table.allocated_tokens == allocated
            # A slot covers the tokens every installed group of it covers.
            for slot in {entry[0] for entry in update}:
                expected_lengths[slot] = min(
                    length
                    for (owner, _), length in installed.items()
                    if owner == slot
                )
                assert tables.allocated_length(slot) == expected_lengths[slot]
            # These are real consumers on the producer stream. Queue all of
            # them before host observation to exercise asynchronous source
            # retirement.
            snapshots.append(
                (
                    tables.unit_tables.clone(),
                    tables.start_pages.clone(),
                    tables.alloced_lens.clone(),
                    expected_units.clone(),
                    expected_starts.clone(),
                    expected_lengths.clone(),
                )
            )
        tables.set_verified(torch.tensor([1, 2]), torch.tensor([6, 5]))
        tables.release((1, 3))
        expected_units[:, (1, 3)] = 0
        expected_starts[:, (1, 3)] = 0
        expected_lengths[[1, 3]] = 0
        for (
            units,
            starts,
            lengths,
            want_units,
            want_starts,
            want_lengths,
        ) in snapshots:
            torch.testing.assert_close(units.cpu(), want_units)
            torch.testing.assert_close(starts.cpu(), want_starts)
            torch.testing.assert_close(lengths.cpu(), want_lengths)
        torch.testing.assert_close(tables.unit_tables.cpu(), expected_units)
        torch.testing.assert_close(tables.start_pages.cpu(), expected_starts)
        torch.testing.assert_close(tables.alloced_lens.cpu(), expected_lengths)
        assert tables.verified_lengths.tolist() == [0, 0, 5, 0]
        assert tables.allocated_length(1) == tables.allocated_length(3) == 0
        assert tables.table(2, 0).units == (4, 5, 13)
        # Every table names its group, page tokens and window (-1 for full).
        assert tables.table_shapes.cpu().tolist() == [
            [0, 4, -1],
            [1, 8, 8],
            [1, 8, 8],
        ]
    finally:
        tables.close()


@pytest.mark.gpu
def test_release_bursts_do_not_wait_for_queued_device_work():
    tables = BlockTables(
        groups=GROUPS,
        num_units=16,
        request_pool_size=3,
        width=3,
        device="cuda:0",
        host_buffer_depth=1,
    )
    try:
        tables.install(
            (
                (1, 0, 0, (2, 3), 8),
                (2, 0, 0, (4, 5), 8),
                (3, 0, 0, (6, 7), 8),
            )
        )
        stream = torch.cuda.current_stream()
        stream.synchronize()

        # Requests finish in bursts while earlier batches still run: each
        # release must be enqueued behind that work, never wait for it.
        torch.cuda._sleep(1_000_000_000)
        queued = torch.cuda.Event()
        queued.record(stream)
        for slot in (1, 2, 3):
            tables.release((slot,))
        assert not queued.query(), "a release waited for queued device work"

        stream.synchronize()
        assert torch.count_nonzero(tables.unit_tables[:, 1:]).item() == 0
        assert tables.alloced_lens.tolist() == [0, 0, 0, 0]
    finally:
        tables.close()


@pytest.mark.parametrize(
    "entry",
    (
        # Units that do not fill whole pages of their group.
        (1, 1, 0, (6, 7, 8), 8),
        # An allocated extent beyond the installed pages.
        (1, 0, 1, (2, 3), 13),
        # A unit table wider than the table capacity.
        (1, 0, 0, (2, 3, 4, 5), 8),
        # The padding sentinel.
        (1, 0, 0, (0, 3), 8),
        # A physical unit beyond the backing KV pool.
        (1, 0, 0, (2, 16), 8),
    ),
)
def test_rejected_installation_changes_nothing(entry):
    tables = BlockTables(
        groups=GROUPS, num_units=16, request_pool_size=2, width=3, device="cpu"
    )
    try:
        tables.install(((2, 0, 0, (9,), 4),))
        before = (
            tables.unit_tables.clone(),
            tables.start_pages.clone(),
            tables.alloced_lens.clone(),
        )
        with pytest.raises(WorkerError):
            tables.install(((2, 0, 0, (9, 10), 8), entry))
        torch.testing.assert_close(tables.unit_tables, before[0])
        torch.testing.assert_close(tables.start_pages, before[1])
        torch.testing.assert_close(tables.alloced_lens, before[2])
        assert tables.table(2, 0).units == (9,)
    finally:
        tables.close()
