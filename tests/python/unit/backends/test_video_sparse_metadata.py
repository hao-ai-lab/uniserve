"""CPU tile metadata contracts consumed by the native sparse providers."""

import math

import pytest
import torch

from uniserve_worker.backends.attention.video_sparse import build_video_sparse_metadata
from uniserve_worker.models.minimax_h3.packing import build_packed_layout

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sparsity", [0.0, 0.8, 0.9])
def test_h3_ragged_tile_counts_and_exempt_prefix(sparsity):
    # FastVideo test_vsa_h3_metadata's tile64 example: prefix (70,5,130),
    # video grid (9,10,13). Each prefix segment has its own partial tail.
    prefix = [64, 6, 5, 64, 64, 2]
    video = [
        min(4, 9 - t) * min(4, 10 - h) * min(4, 13 - w)
        for t in range(0, 9, 4)
        for h in range(0, 10, 4)
        for w in range(0, 13, 4)
    ]
    sizes = torch.tensor(prefix + video + [0, 0])
    metadata = build_video_sparse_metadata(
        padded_rows=sizes.numel() * 64,
        prefix_tiles=6,
        video_tiles=36,
        valid_sizes=sizes,
        device=torch.device("cpu"),
        sparsity=sparsity,
        attention_backend="VIDEO_SPARSE_ATTN_H3",
    )
    assert metadata.valid_sizes.sum() == 70 + 5 + 130 + 9 * 10 * 13
    selected = max(1, math.ceil((1 - sparsity) * 36))
    pattern = metadata.pattern(44)
    counts = pattern.counts(heads=2, query_tiles=44, index_width=44)
    assert torch.all(counts[:, :6] == 42)
    assert torch.all(counts[:, 6:42] == 6 + selected)
    assert pattern.dense_prefix_tiles == 6
    assert pattern.dense_key_tiles == 42
    # Context-parallel owners see the same global key budget, not a local top-k.
    shard = metadata.pattern(8, query_tile_offset=4)
    assert shard.row_counts[0] == (42, 42, *((6 + selected,) * 6))


def test_h3_production_tiling_preserves_raster_rows_and_partial_tails():
    packed = build_packed_layout(text_rows=320, num_frames=124, audio_frames=207)
    assert packed.prefix_tiles == 5 + 7
    assert packed.video_tiles == 10 * 6 * 11
    assert packed.tile_valid_sizes.sum() == 320 + 414 + 37 * 24 * 42
    assert packed.tile_valid_sizes[:12].tolist() == [64] * 11 + [30]
    video_sizes = packed.tile_valid_sizes[12:672]
    assert video_sizes.min() == 8
    assert video_sizes.max() == 64
    raster = torch.arange(37 * 24 * 42)
    tiled = torch.zeros(packed.padded_rows, dtype=torch.long)
    tiled[packed.video_indices] = raster[packed.video_raster_indices]
    assert torch.equal(tiled[packed.video_untile_indices], raster)
    assert torch.unique(packed.video_indices).numel() == raster.numel()
    assert bool(
        (packed.video_indices % 64 < packed.tile_valid_sizes[packed.video_indices // 64]).all()
    )


@pytest.mark.parametrize("sizes", [[65, 64], [-1, 64], [0, 64]])
def test_invalid_logical_tile_sizes_are_rejected(sizes):
    with pytest.raises(ValueError, match="tile|valid sizes"):
        build_video_sparse_metadata(
            padded_rows=128,
            prefix_tiles=1,
            video_tiles=1,
            valid_sizes=torch.tensor(sizes),
            device=torch.device("cpu"),
            sparsity=0.8,
            attention_backend="VIDEO_SPARSE_ATTN_H3",
        )
