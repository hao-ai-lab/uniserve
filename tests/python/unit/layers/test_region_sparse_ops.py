"""Region attention row kernels reproduce the eager row passes they replace.

Tile padding zeroes exactly the rows past each tile's valid size, and the
gated tile compression rounds its product and sum as the two eager BF16
operations do, so both leave region attention's results bit for bit as the
eager expressions compute them.
"""

import pytest
import torch
from uniserve_kernels.attention import vsa_regions

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

    vsa_regions.zero_tile_padding(query, valid, TILE)

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

    vsa_regions.add_gated_tiles(output, compressed, gate, TILE)

    assert torch.equal(output, expected)
