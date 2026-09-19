"""Video sparse selection ops normalize scores over the valid key tiles."""

import pytest
import torch

from uniserve.ops import video_sparse as ops

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
