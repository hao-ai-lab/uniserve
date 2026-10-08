"""Region-wise video sparse attention over a tile-major packed sequence.

``RegionAttention`` evaluates VSA over ``Regions``: dense tiles (text, audio,
images) attend and are attended densely, and every video query tile keeps,
of each video region independently, the key tiles whose pooled scores rank
highest. A compression branch adds, to every row of a tile, the gated
softmax-weighted mean of every live tile's mean value under the pooled
scores. Selection is computed from device tables alone, so one call serves
every assignment of tiles to regions within a padded sequence.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from uniserve.distributed import Communicator, DeviceMesh
from uniserve.distributed.tokens import HeadExchange

from ..config import AttentionParallelConfig
from .inputs import BlockInput, Pattern, Regions
from .layer import BlockAttention


def zero_padding(rows: torch.Tensor, regions: Regions) -> None:
    """Store zeros to the rows past each tile's valid size, in place.

    ``rows`` is ``[padded_tokens, heads, width]`` with any row and head
    strides; rows within a tile's valid size are not touched.
    """
    if rows.is_cuda:
        from uniserve_kernels.attention import vsa_regions

        vsa_regions.zero_tile_padding(rows, regions.valid_sizes, regions.tile)
        return
    live = torch.arange(regions.tile) < regions.valid_sizes[:, None]
    rows.view(regions.tiles, regions.tile, *rows.shape[1:]).masked_fill_(
        ~live[:, :, None, None], 0
    )


def pool(rows: torch.Tensor, regions: Regions) -> torch.Tensor:
    """Average each tile's valid rows of every head in FP32.

    ``rows`` is ``[padded_tokens, heads, width]`` whose rows past each tile's
    valid size hold zeros (``zero_padding``); the result is ``[tiles, heads,
    width]`` FP32, zero for an empty tile. The zeros add nothing, so the sum
    is the valid rows' sum in the order of a masked copy's.
    """
    values = rows.view(regions.tiles, regions.tile, *rows.shape[1:])
    total = values.sum(1, dtype=torch.float32)
    return total / regions.valid_sizes.clamp_min(1).view(-1, 1, 1)


def select(
    scores: torch.Tensor, regions: Regions
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose every query tile's key tiles from its pooled scores.

    ``scores`` is ``[heads, tiles, tiles]`` (query tile, key tile). A dense
    query tile keeps every live key tile; a video query tile keeps every live
    dense tile and, of each region ``r``, the ``region_keep[r]`` key tiles of
    that region with the highest scores; an empty query tile keeps none.

    Returns ``[heads, tiles, tiles]`` int32 key-tile indices, each query
    tile's kept tiles first in ascending order, and ``[heads, tiles]`` int32
    counts. Entries past a count are unread; they hold the unkept tiles.
    """
    if scores.is_cuda:
        from uniserve_kernels.attention import vsa_regions

        return vsa_regions.select_tiles(
            scores.contiguous(),
            regions.tile_regions,
            regions.valid_sizes,
            regions.region_starts,
            regions.region_keep,
        )
    tiles = regions.tiles
    device = scores.device
    tile_regions = regions.tile_regions.long()
    live = regions.valid_sizes > 0
    dense = live & (tile_regions < 0)
    video = live & (tile_regions >= 0)

    # Order every query's key tiles by region index, and within a region by
    # descending score: a stable sort by score followed by a stable sort by
    # region. The rank of a key within its region is its position in that
    # order less the region's first position.
    by_score = scores.argsort(dim=-1, descending=True, stable=True)
    key_regions = tile_regions[by_score]
    by_region = key_regions.argsort(dim=-1, stable=True)
    order = by_score.gather(-1, by_region)
    ordered_regions = key_regions.gather(-1, by_region)
    region = ordered_regions.clamp_min(0)
    rank = (
        torch.arange(tiles, device=device)
        - regions.region_starts.long()[region]
    )
    kept = (ordered_regions >= 0) & (rank < regions.region_keep.long()[region])
    chosen = torch.zeros_like(kept).scatter_(-1, order, kept)

    mask = (video[:, None] & dense[None, :]) | (dense[:, None] & live[None, :])
    mask = mask[None] | (chosen & video[None, :, None])
    counts = mask.sum(-1, dtype=torch.int32)

    # Compact each query tile's kept key tiles to the front in ascending
    # order: a kept tile's slot is the count of kept tiles before it, and
    # the unkept tiles fill the remaining slots in order.
    kept_before = mask.cumsum(-1) - 1
    unkept_before = (~mask).cumsum(-1) - 1
    slots = torch.where(mask, kept_before, counts[..., None] + unkept_before)
    keys = torch.arange(tiles, dtype=torch.int32, device=device)
    indices = torch.empty(mask.shape, dtype=torch.int32, device=device)
    indices.scatter_(-1, slots, keys.expand(mask.shape))
    return indices, counts


class RegionAttention(nn.Module):
    """Attend every row through region-wise VSA and gated tile compression.

    The inputs cover the complete packed sequence for this rank's heads, as
    a head-parallel projection of the gathered sequence produces them. Under
    Ulysses, the attended rows are exchanged back to the token shard each
    rank owns with every head; ``parallelize_`` binds ``exchange`` and
    declares its group in ``communication_groups``.
    """

    # Recorded by parallelize_ once the layer is bound to its partition.
    _parallel_mesh: DeviceMesh
    _attention_parallel: AttentionParallelConfig

    def __init__(self, attention: BlockAttention):
        super().__init__()
        self.attention = attention
        self.exchange = HeadExchange(Communicator())

    @torch.inference_mode()
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        regions: Regions,
    ) -> torch.Tensor:
        """Attend ``[padded_tokens, heads, width]`` projections of all rows.

        ``q`` and ``k`` are normalized and rotated; ``gate`` weights the
        compression branch elementwise. Every projection may use its own row
        and head strides; the rows of ``q``, ``k`` and ``v`` past a tile's
        valid size are overwritten with zeros. Returns BF16 ``[padded_tokens
        / members, heads * members, width]`` rows of this rank's token shard,
        ``members`` being the Ulysses group size. Rows past a tile's valid
        size hold unspecified values.

        Raises:
            ValueError: Projections that do not cover the region tiles, or a
                tile size other than the block attention's.
        """
        tile = regions.tile
        if (
            tile != self.attention.tile_size
            or q.ndim != 3
            or any(value.shape != q.shape for value in (k, v, gate))
            or q.shape[0] != regions.padded_tokens
        ):
            raise ValueError(
                "region attention projections must cover the region tiles"
            )
        tiles, heads, width = regions.tiles, q.shape[1], q.shape[2]

        # Pooled scores [heads, query tile, key tile] in FP32: the tile means
        # of the queries and keys, scaled by the inverse square root of the
        # head width. Attention never reads padding rows (keys past a tile's
        # valid size are masked and padded query rows are unspecified), so
        # zeroing them leaves every valid result as it was.
        for value in (q, k, v):
            zero_padding(value, regions)
        pooled_query, pooled_key, pooled_value = (
            pool(value, regions).permute(1, 0, 2) for value in (q, k, v)
        )
        scores = torch.matmul(
            pooled_query, pooled_key.transpose(-1, -2)
        ) / math.sqrt(width)

        indices, counts = select(scores, regions)
        # Every query tile keeps at most every tile; the device counts state
        # how many each actually keeps.
        pattern = Pattern(((tiles,) * tiles,), 0, 0, tile)
        fine = self.attention(
            q,
            k,
            v,
            BlockInput(pattern, indices, counts, regions.valid_sizes, 0),
        )

        # Compression: each query tile's softmax over the live key tiles
        # weights their mean values. The BF16 tile result then scales the
        # gate and adds to the fine rows, each operation rounding to BF16.
        live = regions.valid_sizes > 0
        weights = scores.masked_fill(~live, -torch.inf).softmax(-1)
        compressed = torch.matmul(weights, pooled_value).to(q.dtype)
        if fine.is_cuda:
            from uniserve_kernels.attention import vsa_regions

            vsa_regions.add_gated_tiles(fine, compressed, gate, tile)
        else:
            tiled = (tiles, tile, heads, width)
            gated = compressed.permute(1, 0, 2)[:, None] * gate.view(tiled)
            fine.view(tiled).add_(gated)
        return self.exchange.tokens(fine)
