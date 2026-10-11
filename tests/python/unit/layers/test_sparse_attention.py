"""VSA numerical tile domains can be inspected without execution resources."""

import pytest
import torch

from uniserve.nn.attention import vsa
from uniserve.nn.attention.vsa.segments import select

pytestmark = pytest.mark.unit


def _segments(tile_segments, variable_block_sizes, keep, **fields):
    """Segment tables over ``tile_segments`` (-1 dense or empty) on the host."""
    segments = torch.tensor(tile_segments, dtype=torch.int32)
    starts = torch.zeros_like(segments)
    keeps = torch.ones_like(segments)
    for segment, count in enumerate(keep):
        starts[segment] = int((segments < segment).sum())
        keeps[segment] = count
    return vsa.Segments(
        128,
        segments.numel() * 128,
        torch.tensor(variable_block_sizes, dtype=torch.int32),
        segments,
        starts,
        keeps,
        **fields,
    )


def test_pattern_counts_bound_each_query_tile_by_its_declared_keys():
    pattern = vsa.Pattern(((3, 2, 2, 1),), 1, 3, 64)
    output = torch.empty((2, 4), dtype=torch.int32)

    pattern.counts(num_heads=2, query_tiles=4, index_width=3, out=output)

    torch.testing.assert_close(
        output, torch.tensor([[3, 2, 2, 1], [3, 2, 2, 1]], dtype=torch.int32)
    )
    with pytest.raises(ValueError, match="dimensions"):
        pattern.counts(num_heads=2, query_tiles=4, index_width=2, out=output)


def test_dense_prefix_queries_must_declare_every_dense_key():
    with pytest.raises(ValueError, match="dense visibility"):
        vsa.Pattern(((3, 2, 2),), 2, 3, 64)


def test_segment_tables_hold_one_int32_entry_per_tile():
    with pytest.raises(ValueError, match="one int32 entry per tile"):
        vsa.Segments(
            128,
            256,
            torch.ones(3, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"dense_prefix_tiles": 7},
        {"key_bounds": (3, 0, 1)},
        {"key_bounds": (3, 1)},
        # A dense prefix tile attends every tile.
        {"dense_prefix_tiles": 1, "key_bounds": (2, 2, 1)},
    ],
)
def test_dense_prefix_and_key_bounds_lie_within_the_tiles(fields):
    with pytest.raises(ValueError, match="within the tiles"):
        _segments([-1, 0, 0], [128, 128, 128], [1], **fields)


def test_a_query_window_selects_as_its_rows_of_the_whole_sequence():
    torch.manual_seed(3)
    layout = [-1, -1, 0, 0, 0, -1, 1, 1, 1, 1]
    valid = [128, 5, 128, 40, 128, 0, 128, 128, 77, 128]
    segments = _segments(layout, valid, [2, 3])
    scores = torch.randn(3, len(layout), len(layout))
    indices, counts = select(scores, segments)

    window = slice(4, 9)
    local_indices, local_counts = select(
        scores[:, window], segments, query_tile_offset=window.start
    )

    assert torch.equal(local_counts, counts[:, window])
    assert torch.equal(local_indices, indices[:, window])
