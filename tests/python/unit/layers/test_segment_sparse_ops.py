"""Segment attention row kernels reproduce the eager row passes they replace.

Tile padding zeroes exactly the rows past each tile's valid size, and the
gated tile compression rounds its product and sum as the two eager BF16
operations do, so both leave segment attention's results bit for bit as the
eager expressions compute them. Tile selection keeps, per segment, the
best-scoring key tiles in the order a stable descending sort ranks them.
"""

import pytest
import torch
from uniserve_kernels.attention import vsa_segments

from uniserve.nn.attention.vsa.inputs import Segments
from uniserve.nn.attention.vsa.segments import select

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

TILE, HEADS, WIDTH = 128, 6, 128


def test_tile_padding_zeroes_only_rows_past_each_valid_size():
    torch.manual_seed(11)
    valid = torch.tensor(
        [128, 0, 1, 77, 128, 16, 127], device="cuda", dtype=torch.int32
    )
    rows = valid.numel() * TILE
    # Q of a [rows, q | k] matrix: rows and heads are strided.
    merged = torch.randn(
        rows, 2 * HEADS * WIDTH, device="cuda", dtype=torch.bfloat16
    )
    before = merged.clone()
    query = merged[:, : HEADS * WIDTH].view(rows, HEADS, WIDTH)

    vsa_segments.zero_tile_padding(query, valid, TILE)

    live = (torch.arange(TILE, device="cuda") < valid[:, None]).reshape(-1)
    assert torch.equal(
        query[live], before[:, : HEADS * WIDTH].view(rows, HEADS, WIDTH)[live]
    )
    assert bool((query[~live] == 0).all())
    assert torch.equal(merged[:, HEADS * WIDTH :], before[:, HEADS * WIDTH :])


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_gated_tiles_round_as_the_eager_product_and_sum(scale):
    torch.manual_seed(int(scale * 1000) % 97)
    tiles = 5
    rows = tiles * TILE
    gate = (torch.randn(rows, 2, HEADS, WIDTH, device="cuda") * scale).to(
        torch.bfloat16
    )[:, 1]
    output = (torch.randn(rows, HEADS, WIDTH, device="cuda") * 3).to(
        torch.bfloat16
    )
    compressed = torch.randn(HEADS, tiles, WIDTH, device="cuda").to(
        torch.bfloat16
    )
    tiled = (tiles, TILE, HEADS, WIDTH)
    expected = output.clone()
    expected.view(tiled).add_(
        compressed.permute(1, 0, 2)[:, None] * gate.view(tiled)
    )

    vsa_segments.add_gated_tiles(output, compressed, gate, TILE)

    assert torch.equal(output, expected)


def _segments(tile_segments, valid_sizes, keep):
    """Segment tables over ``tile_segments`` (-1 dense or empty) on the host."""
    segments = torch.tensor(tile_segments, dtype=torch.int32)
    starts = torch.zeros_like(segments)
    keeps = torch.ones_like(segments)
    for segment, count in enumerate(keep):
        starts[segment] = int((segments < segment).sum())
        keeps[segment] = count
    valid = torch.tensor(valid_sizes, dtype=torch.int32)
    return Segments(
        TILE, segments.numel() * TILE, valid, segments, starts, keeps
    )


def _on_device(segments):
    return Segments(
        segments.tile,
        segments.padded_tokens,
        *(
            table.cuda()
            for table in (
                segments.valid_sizes,
                segments.tile_segments,
                segments.segment_starts,
                segments.segment_keep,
            )
        ),
    )


def test_tile_selection_keeps_each_segments_best_scores_in_stable_order():
    # Tile 0 dense, tile 1 empty, tiles 2-4 segment 0 (keep 2), tile 5
    # segment 1 (keep 1).
    segments = _segments([-1, -1, 0, 0, 0, 1], [5, 0, 9, 128, 1, 64], [2, 1])
    nan = float("nan")
    scores = torch.zeros(1, 6, 6)
    # A video query: NaN ranks first and equal scores keep tile order.
    scores[0, 2] = torch.tensor([0.5, 9.0, 1.0, nan, 1.0, -3.0])
    # Signed zeros are equal scores.
    scores[0, 3] = torch.tensor([0.0, 0.0, -0.0, 0.0, -0.0, 2.0])
    scores[0, 4] = torch.tensor([0.0, 0.0, -1.0, 2.0, 3.0, 0.0])

    indices, counts = select(scores.cuda(), _on_device(segments))

    # Dense query: every live tile; empty query: none, every tile unkept.
    assert counts[0, :2].tolist() == [5, 0]
    assert indices[0, 0].tolist() == [0, 2, 3, 4, 5, 1]
    assert indices[0, 1].tolist() == [0, 1, 2, 3, 4, 5]
    # Video queries: the dense tile, two of segment 0, the one of segment 1.
    assert counts[0, 2:5].tolist() == [4, 4, 4]
    assert indices[0, 2].tolist() == [0, 2, 3, 5, 1, 4]
    assert indices[0, 3].tolist() == [0, 2, 3, 5, 1, 4]
    assert indices[0, 4].tolist() == [0, 3, 4, 5, 1, 2]


@pytest.mark.parametrize(
    ("layout", "empty", "keep"),
    [
        # Dense tiles first, as a prompt precedes the latents.
        (
            [-1] * 40 + [0] * 120 + [-1] * 6 + [1] * 120 + [2] * 30,
            (3, 7, 39),
            [12, 12, 3],
        ),
        # A video segment first, its first tile empty.
        ([0] * 70 + [-1] * 9 + [1] * 50, (0, 7, 75), [12, 5]),
        # A 15-second target beside a 15-second reference video: more tiles
        # than one 2048-lane sort holds.
        (
            [-1] * 120 + [0] * 972 + [-1] * 10 + [1] * 998,
            (5, 300, 2099),
            [98, 100],
        ),
    ],
    ids=["dense_first", "segment_first", "long_video_pair"],
)
@pytest.mark.parametrize("ties", [False, True])
def test_tile_selection_matches_the_eager_sort_composition(
    layout, empty, keep, ties
):
    torch.manual_seed(5)
    valid = [int(v) for v in torch.randint(1, TILE + 1, (len(layout),))]
    for tile in empty:
        valid[tile] = 0
    segments = _segments(layout, valid, keep)
    scores = torch.randn(HEADS, len(layout), len(layout))
    if ties:
        scores = torch.round(scores * 2) / 2
        zeros = scores == 0
        scores[zeros] = torch.where(
            torch.rand(int(zeros.sum())) < 0.5, -0.0, 0.0
        )
        scores[torch.rand(scores.shape) < 0.01] = float("nan")
        scores[torch.rand(scores.shape) < 0.01] = float("-inf")

    expected = select(scores, segments)
    indices, counts = select(scores.cuda(), _on_device(segments))

    assert torch.equal(counts.cpu(), expected[1])
    assert torch.equal(indices.cpu(), expected[0])
