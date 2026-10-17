"""Segment attention row kernels reproduce the eager row passes they replace.

Tile padding zeroes exactly the rows past each tile's valid size, and the
gated tile compression rounds its product and sum as the two eager BF16
operations do, so both leave segment attention's results bit for bit as the
eager expressions compute them. Tile preparation stores, in one pass, the
stepwise Q/K normalization and rotation, the padding and the eager tile means;
gated row composition stores each member's interval of the gated sum in its
exchange slot. Tile selection keeps, per segment, the best-scoring key tiles
in the order a stable descending sort ranks them.
"""

import pytest
import torch
from uniserve_kernels.attention import vsa_segments

from uniserve.nn import functional
from uniserve.nn.attention.vsa.inputs import Segments
from uniserve.nn.attention.vsa.segments import pool, select, zero_padding

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


@pytest.mark.parametrize("tiles", [9, 600], ids=["short", "long"])
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("norm", [False, True], ids=["plain", "norm_rope"])
def test_tile_preparation_stores_the_eager_row_passes(tiles, tile, norm):
    torch.manual_seed(tiles + tile + norm)
    heads = 14
    sizes = torch.randint(1, tile + 1, (tiles,), device="cuda")
    sizes[1] = 0
    sizes[2] = tile
    sizes[3] = 1
    valid = sizes.to(torch.int32)
    rows = tiles * tile
    # Q | K | V | gate of one merged projection: rows and heads are strided.
    merged = (torch.randn(rows, 4 * heads * WIDTH, device="cuda") * 3).to(
        torch.bfloat16
    )
    expected = merged.clone()

    def branches(matrix):
        return tuple(
            matrix[:, index * heads * WIDTH : (index + 1) * heads * WIDTH].view(
                rows, heads, WIDTH
            )
            for index in range(3)
        )

    operands = {}
    query, key, value = branches(expected)
    if norm:
        # H3's normalization: one RMS domain over the head and a rotated
        # leading 96 features with FP32 compact factors.
        phases = torch.rand(rows, 48, device="cuda") * 40
        operands = {
            "query_weight": (torch.rand(WIDTH, device="cuda") + 0.5).to(
                torch.bfloat16
            ),
            "key_weight": (torch.rand(WIDTH, device="cuda") + 0.5).to(
                torch.bfloat16
            ),
            "cosine": phases.cos(),
            "sine": phases.sin(),
            "eps": 1e-6,
        }
        functional.qk_norm_rope(
            query,
            key,
            (operands["query_weight"],),
            (operands["key_weight"],),
            (operands["cosine"],),
            (operands["sine"],),
            eps=1e-6,
            axis_dims=(WIDTH,),
            out=(query, key),
            rounding=functional.Rounding.STEPWISE,
        )
    for rows_of in (query, key, value):
        zero_padding(rows_of, valid, tile)
    means = [pool(rows_of, valid, tile) for rows_of in (query, key, value)]

    pooled = [torch.empty(tiles, heads, WIDTH, device="cuda") for _ in range(3)]
    vsa_segments.prepare_tiles(
        *branches(merged), valid, tile, *pooled, **operands
    )

    # The gate branch is untouched; Q, K, V and every mean are the eager bits.
    assert torch.equal(merged, expected)
    for actual, mean in zip(pooled, means, strict=True):
        assert torch.equal(actual, mean)


@pytest.mark.parametrize("slots", [(0, 1, 2, 3), (2, 0, 3, 1)])
def test_gated_rows_compose_each_members_interval_into_its_slot(slots):
    torch.manual_seed(len(slots) + slots[0])
    members, owner_tiles, tile = 4, 6, 64
    owner_rows = owner_tiles * tile
    rows = members * owner_rows
    projected = (torch.randn(rows, 2, HEADS, WIDTH, device="cuda") * 2).to(
        torch.bfloat16
    )
    fine, gate = projected[:, 0], projected[:, 1]
    compressed = torch.randn(HEADS, rows // tile, WIDTH, device="cuda").to(
        torch.bfloat16
    )
    composed = fine.clone()
    vsa_segments.add_gated_tiles(composed, compressed, gate, tile)
    before = projected.clone()

    start, count = 128, 160
    payload = torch.empty(
        members, count, HEADS, WIDTH, device="cuda", dtype=torch.bfloat16
    )
    vsa_segments.compose_gated_rows(
        payload,
        fine,
        compressed,
        gate,
        tile,
        owner_rows=owner_rows,
        start=start,
        slots=slots,
    )

    for member, slot in enumerate(slots):
        first = member * owner_rows + start
        assert torch.equal(payload[slot], composed[first : first + count])
    assert torch.equal(projected, before)


def _segments(tile_segments, variable_block_sizes, keep):
    """Segment tables over ``tile_segments`` (-1 dense or empty) on the host."""
    segments = torch.tensor(tile_segments, dtype=torch.int32)
    starts = torch.zeros_like(segments)
    keeps = torch.ones_like(segments)
    for segment, count in enumerate(keep):
        starts[segment] = int((segments < segment).sum())
        keeps[segment] = count
    valid = torch.tensor(variable_block_sizes, dtype=torch.int32)
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
                segments.variable_block_sizes,
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


def test_a_query_window_selects_as_its_rows_of_the_whole_sequence():
    layout = [-1] * 4 + [0] * 40 + [-1] * 2 + [1] * 50
    torch.manual_seed(9)
    valid = [int(v) for v in torch.randint(1, TILE + 1, (len(layout),))]
    valid[7] = valid[50] = 0
    segments = _segments(layout, valid, [11, 9])
    scores = torch.randn(HEADS, len(layout), len(layout))
    expected = select(scores, segments)

    window = slice(33, 76)
    indices, counts = select(
        scores[:, window].cuda(),
        _on_device(segments),
        query_tile_offset=window.start,
    )

    assert torch.equal(counts.cpu(), expected[1][:, window])
    assert torch.equal(indices.cpu(), expected[0][:, window])
