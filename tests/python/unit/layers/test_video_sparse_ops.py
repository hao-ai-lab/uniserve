"""Video sparse selection ops normalize scores over the valid key tiles."""

import pytest
import torch
from uniserve_kernels.attention import vsa_tiles as ops

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def test_tile_softmax_excludes_empty_key_tiles_and_zeroes_dead_rows():
    torch.manual_seed(3)
    heads, tiles = 3, 70
    scores = torch.randn(heads, tiles, tiles, device="cuda") * 4
    valid = torch.full((tiles,), 64, device="cuda", dtype=torch.int32)
    valid[5] = 0
    valid[60:] = 0

    # Empty key tiles weigh nothing, as if their score were -inf.
    expected = torch.softmax(
        scores.masked_fill(valid.view(1, 1, -1) == 0, -torch.inf), dim=-1
    )
    actual = scores.clone()
    ops.tile_softmax(actual, valid)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert bool((actual[:, :, 5] == 0).all()) and bool(
        (actual[:, :, 60:] == 0).all()
    )

    # A row without any valid key tile is written as zeros, not NaN.
    dead = scores.clone()
    ops.tile_softmax(dead, torch.zeros_like(valid))
    assert bool((dead == 0).all())

    with pytest.raises(ValueError, match="tile softmax"):
        ops.tile_softmax(scores.half(), valid)


@pytest.mark.parametrize("spread", [4.0, 1e-3, 1e-6])
def test_block_map_selects_the_exact_top_scoring_video_tiles(spread):
    """Each video query tile keeps its live prefix and the exact top-k videos.

    Scores packed into a narrow band are where a bracketing search stops with
    more candidates than the budget; the selection must still equal the
    highest-scoring tiles, as the checkpoint's top-k policy defines.
    """
    torch.manual_seed(7)
    heads, prefix, video, selected = 4, 3, 190, 38
    tiles = prefix + video
    scores = 0.25 + spread * torch.randn(
        heads, video, video, device="cuda", dtype=torch.float32
    )
    prefix_indices = torch.arange(prefix, device="cuda", dtype=torch.int32)
    dense_indices = torch.arange(tiles, device="cuda", dtype=torch.int32)
    live = torch.tensor([prefix], device="cuda", dtype=torch.int32)
    valid = torch.full((tiles,), 64, device="cuda", dtype=torch.int32)
    indices = torch.full(
        (heads, tiles, tiles), -1, device="cuda", dtype=torch.int32
    )
    counts = torch.zeros(heads, tiles, device="cuda", dtype=torch.int32)

    ops.write_block_map(
        scores,
        prefix_indices,
        dense_indices,
        live,
        valid,
        indices,
        counts,
        query_tile_offset=0,
        local_prefix=prefix,
        local_video=tiles,
        prefix_tiles=prefix,
        valid_tiles=tiles,
        selected=selected,
    )

    video_counts = counts[:, prefix:]
    assert bool((video_counts == prefix + selected).all())
    rows = indices[:, prefix:, : prefix + selected].long()
    assert bool((rows[..., :prefix] == prefix_indices.long()).all())
    chosen = torch.sort(rows[..., prefix:] - prefix, dim=-1).values
    expected = torch.sort(scores.topk(selected, dim=-1).indices, dim=-1).values
    assert torch.equal(chosen, expected)
