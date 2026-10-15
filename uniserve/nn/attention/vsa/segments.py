"""Segment-wise video sparse attention over a tile-major packed sequence.

``SegmentAttention`` evaluates VSA over ``Segments``: dense tiles (text, audio,
images) attend and are attended densely, and every video query tile keeps,
of each video segment independently, the key tiles whose pooled scores rank
highest. A compression branch adds, to every row of a tile, the gated
softmax-weighted mean of every live tile's mean value under the pooled
scores. Selection is computed from device tables alone, so one call serves
every assignment of tiles to segments within a padded sequence.

The layer consumes its projections as they land: each projected row interval
is prepared (Q/K normalization and rotation, padding and pooling) while later
intervals are still being gathered and projected, and under Ulysses the
attended rows return to their token owners interval by interval, so the
output projection and feed-forward update of an early interval overlap the
exchange of the later ones.
"""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import torch
from torch import nn

from uniserve.distributed import Communicator, DeviceMesh
from uniserve.distributed.tokens import HeadExchange
from uniserve.nn import _binding, functional

from ..config import AttentionParallelConfig
from .inputs import BlockInput, NormRope, Pattern, Segments
from .layer import BlockAttention

if TYPE_CHECKING:
    from uniserve.nn.attention._parallel import ParallelAttention


def zero_padding(
    rows: torch.Tensor, variable_block_sizes: torch.Tensor, tile: int
) -> None:
    """Store zeros to the rows past each tile's valid size, in place.

    ``rows`` is ``[tiles * tile, heads, width]`` with any row and head
    strides and ``variable_block_sizes`` holds the valid rows of each of its
    tiles; rows within a tile's valid size are not touched.
    """
    if rows.is_cuda:
        from uniserve_kernels.attention import vsa_segments

        vsa_segments.zero_tile_padding(rows, variable_block_sizes, tile)
        return
    live = torch.arange(tile) < variable_block_sizes[:, None]
    rows.view(variable_block_sizes.numel(), tile, *rows.shape[1:]).masked_fill_(
        ~live[:, :, None, None], 0
    )


def pool(
    rows: torch.Tensor, variable_block_sizes: torch.Tensor, tile: int
) -> torch.Tensor:
    """Average each tile's valid rows of every head in FP32.

    ``rows`` is ``[tiles * tile, heads, width]`` whose rows past each tile's
    valid size hold zeros (``zero_padding``); the result is ``[tiles, heads,
    width]`` FP32, zero for an empty tile. The zeros add nothing, so the sum
    is the valid rows' sum in the order of a masked copy's.
    """
    values = rows.view(variable_block_sizes.numel(), tile, *rows.shape[1:])
    total = values.sum(1, dtype=torch.float32)
    return total / variable_block_sizes.clamp_min(1).view(-1, 1, 1)


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    tile: int,
    pooled: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    norm: NormRope | None,
    rows: slice,
) -> None:
    """Prepare one projected row interval of whole tiles in place.

    ``q``, ``k`` and ``v`` are the interval's ``[rows, heads, width]``
    projections, ``rows`` its packed rows and ``variable_block_sizes`` the
    valid rows of its tiles. With ``norm``, Q and K are normalized and
    rotated in place (``NormRope``). The rows past each tile's valid size of
    Q, K and V are then overwritten with zeros (``zero_padding``) and the
    contiguous FP32 ``[tiles, heads, width]`` ``pooled`` query, key and value
    outputs receive each tile's means (``pool``). Attention never reads
    padding rows (keys past a tile's valid size are masked and padded query
    rows are unspecified), so zeroing them leaves every valid result as it
    was. CUDA tensors take one fused pass (``vsa_segments.prepare_tiles``)
    that stores the same bits.
    """
    if q.is_cuda:
        from uniserve_kernels.attention import vsa_segments

        if norm is None:
            vsa_segments.prepare_tiles(
                q, k, v, variable_block_sizes, tile, *pooled
            )
        else:
            vsa_segments.prepare_tiles(
                q,
                k,
                v,
                variable_block_sizes,
                tile,
                *pooled,
                query_weight=norm.query_weight,
                key_weight=norm.key_weight,
                cosine=norm.cos[rows],
                sine=norm.sin[rows],
                eps=norm.eps,
            )
        return

    if norm is not None:
        functional.qk_norm_rope(
            q,
            k,
            (norm.query_weight,),
            (norm.key_weight,),
            (norm.cos[rows],),
            (norm.sin[rows],),
            eps=norm.eps,
            axis_dims=(q.shape[-1],),
            out=(q, k),
            rounding=functional.Rounding.STEPWISE,
        )
    for value, target in zip((q, k, v), pooled, strict=True):
        zero_padding(value, variable_block_sizes, tile)
        target.copy_(pool(value, variable_block_sizes, tile))


def compose_rows(
    rows: slice,
    payload: torch.Tensor,
    slots: tuple[int, ...],
    *,
    fine: torch.Tensor,
    compressed: torch.Tensor,
    gate: torch.Tensor,
    tile: int,
    owner_rows: int,
) -> None:
    """Write one exchanged interval of every member's gated rows.

    ``fine`` and ``gate`` are this rank's ``[members * owner_rows, heads,
    width]`` attended rows and gate, member ``m`` owning rows ``m *
    owner_rows`` onward, and ``compressed`` the ``[heads, tiles, width]``
    compressed tiles. Slot ``slots[m]`` of the ``[members, count, heads,
    width]`` ``payload`` receives member ``m``'s rows ``rows`` of ``fine +
    compressed[:, tile] * gate``, the product and the sum each rounded to
    the rows' dtype as two eager operations round them. CUDA tensors take
    one launch (``vsa_segments.compose_gated_rows``) that stores the same
    bits.
    """
    if payload.is_cuda:
        from uniserve_kernels.attention import vsa_segments

        vsa_segments.compose_gated_rows(
            payload,
            fine,
            compressed,
            gate,
            tile,
            owner_rows=owner_rows,
            start=rows.start,
            slots=slots,
        )
        return
    count = rows.stop - rows.start
    tiled = (count // tile, tile, *fine.shape[1:])
    for member, slot in enumerate(slots):
        start = member * owner_rows + rows.start
        tiles = slice(start // tile, (start + count) // tile)
        gated = compressed[:, tiles].permute(1, 0, 2)[:, None] * gate[
            start : start + count
        ].view(tiled)
        payload[slot].view(tiled).copy_(
            fine[start : start + count].view(tiled) + gated
        )


def select(
    scores: torch.Tensor, segments: Segments, *, query_tile_offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose every query tile's key tiles from its pooled scores.

    ``scores`` is ``[heads, query tiles, tiles]`` (query tile, key tile) and
    its query tile ``q`` is tile ``query_tile_offset + q`` of the sequence. A
    dense query tile keeps every live key tile; a video query tile keeps
    every live dense tile and, of each segment ``r``, the ``segment_keep[r]``
    key tiles of that segment with the highest scores; an empty query tile
    keeps none.

    Returns ``[heads, query tiles, tiles]`` int32 key-tile indices, each
    query tile's kept tiles first in ascending order, and ``[heads, query
    tiles]`` int32 counts. Entries past a count are unread; they hold the
    unkept tiles.
    """
    if scores.is_cuda:
        from uniserve_kernels.attention import vsa_segments

        return vsa_segments.select_tiles(
            scores.contiguous(),
            segments.tile_segments,
            segments.variable_block_sizes,
            segments.segment_starts,
            segments.segment_keep,
            query_tile_offset=query_tile_offset,
        )
    tiles = segments.tiles
    queries = slice(query_tile_offset, query_tile_offset + scores.shape[1])
    device = scores.device
    tile_segments = segments.tile_segments.long()
    live = segments.variable_block_sizes > 0
    dense = live & (tile_segments < 0)
    video = live & (tile_segments >= 0)

    # Order every query's key tiles by segment index, and within a segment by
    # descending score: a stable sort by score followed by a stable sort by
    # segment. The rank of a key within its segment is its position in that
    # order less the segment's first position.
    by_score = scores.argsort(dim=-1, descending=True, stable=True)
    key_segments = tile_segments[by_score]
    by_segment = key_segments.argsort(dim=-1, stable=True)
    order = by_score.gather(-1, by_segment)
    ordered_segments = key_segments.gather(-1, by_segment)
    segment = ordered_segments.clamp_min(0)
    rank = (
        torch.arange(tiles, device=device)
        - segments.segment_starts.long()[segment]
    )
    kept = (ordered_segments >= 0) & (
        rank < segments.segment_keep.long()[segment]
    )
    chosen = torch.zeros_like(kept).scatter_(-1, order, kept)

    mask = (video[queries, None] & dense[None, :]) | (
        dense[queries, None] & live[None, :]
    )
    mask = mask[None] | (chosen & video[None, queries, None])
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


class SegmentAttention(nn.Module):
    """Attend every row through segment-wise VSA and gated tile compression.

    The inputs cover the rows of this rank's context interval for its heads,
    as a head-parallel projection of the gathered interval produces them:
    the complete packed sequence without a context partition. Keys and
    values are then gathered across the context group, so every query
    attends to the keys of the whole sequence. Under Ulysses, the attended
    rows are exchanged back to the token shard each rank owns with every
    head; ``parallelize_`` binds ``exchange`` and ``context_parallel`` and
    declares their groups in ``communication_groups``.
    """

    # Recorded by parallelize_ once the layer is bound to its partition.
    _parallel_mesh: DeviceMesh
    _attention_parallel: AttentionParallelConfig

    def __init__(self, attention: BlockAttention):
        super().__init__()
        self.attention = attention
        self.exchange = HeadExchange(Communicator())
        self.context_parallel: ParallelAttention | None = None

    def query_rows(self, padded_tokens: int) -> slice:
        """Packed rows of the interval this rank's projections cover.

        The whole sequence without a context partition; otherwise this
        context coordinate's equal contiguous share of it.
        """
        parallel = self.context_parallel
        if parallel is None:
            return slice(0, padded_tokens)
        group = parallel.context_group
        count = padded_tokens // group.size
        return slice(group.rank * count, (group.rank + 1) * count)

    @torch.inference_mode()
    def forward_chunks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        segments: Segments,
        *,
        norm: NormRope | None = None,
        ready: Iterable[slice] | None = None,
    ) -> Iterator[tuple[slice, torch.Tensor]]:
        """Attend ``[rows, heads, width]`` projections of ``query_rows``.

        ``ready`` yields the packed row intervals of ``query_rows`` as their
        projections land in ``q``, ``k``, ``v`` and ``gate``, in any order,
        each whole tiles and every row once; without it every row is ready.
        Each interval is prepared as it arrives, while later projections may
        still be in flight: with ``norm`` its Q and K rows are normalized and
        rotated in place (``NormRope``), otherwise they arrive normalized and
        rotated; its Q, K and V rows past a tile's valid size are overwritten
        with zeros, and its tiles are pooled. ``gate`` weights the compression
        branch elementwise. Every projection may use its own row and head
        strides.

        Yields ``(packed rows, values)`` intervals covering this rank's token
        shard in order: BF16 ``[count, heads * members, width]`` rows,
        ``members`` being the Ulysses group size, with member ``m``'s heads
        at head offset ``m * heads``. Under Ulysses every interval's exchange
        starts as soon as it is composed, so the consumer of an early
        interval overlaps the exchange of the later ones. Rows past a tile's
        valid size hold unspecified values.

        Raises:
            ValueError: Projections or ready intervals that do not cover
                this rank's interval of the segment tiles.
            RuntimeError: A context partition outside an execution context.
        """
        tile = segments.tile
        interval = self.query_rows(segments.padded_tokens)
        if (
            q.ndim != 3
            or any(value.shape != q.shape for value in (k, v, gate))
            or q.shape[0] != interval.stop - interval.start
        ):
            raise ValueError(
                "segment attention projections must cover this rank's "
                "segment tiles"
            )
        tiles, heads, width = segments.tiles, q.shape[1], q.shape[2]
        queries = slice(interval.start // tile, interval.stop // tile)
        query_tiles = queries.stop - queries.start
        sizes = segments.variable_block_sizes

        # FP32 tile means [tiles, heads, width] of the prepared rows, written
        # interval by interval: queries of this rank's interval, and keys and
        # values of its tiles until a context partition gathers the rest.
        pooled_query, pooled_key, pooled_value = (
            torch.empty(
                (query_tiles, heads, width),
                dtype=torch.float32,
                device=q.device,
            )
            for _ in range(3)
        )
        covered = []
        for rows in (interval,) if ready is None else ready:
            if (
                rows.step not in (None, 1)
                or rows.start % tile
                or rows.stop % tile
                or not interval.start <= rows.start < rows.stop <= interval.stop
            ):
                raise ValueError(
                    "ready projections must be whole tiles of this rank's "
                    "interval"
                )
            local = slice(
                rows.start - interval.start, rows.stop - interval.start
            )
            local_tiles = slice(local.start // tile, local.stop // tile)
            prepare(
                q[local],
                k[local],
                v[local],
                sizes[rows.start // tile : rows.stop // tile],
                tile,
                (
                    pooled_query[local_tiles],
                    pooled_key[local_tiles],
                    pooled_value[local_tiles],
                ),
                norm=norm,
                rows=rows,
            )
            covered.append((rows.start, rows.stop))
        covered.sort()
        if (
            not covered
            or [start for start, _ in covered]
            != [interval.start, *(stop for _, stop in covered[:-1])]
            or covered[-1][1] != interval.stop
        ):
            raise ValueError(
                "ready projections must cover every row of this rank's "
                "interval once"
            )

        # Every query attends to the keys of the whole sequence, which a
        # context partition gathers into this rank's storage together with
        # the means every member pooled from its own tiles.
        parallel = self.context_parallel
        if parallel is not None and parallel.context_group.size > 1:
            binding = _binding.vsa.get().get(id(self.attention))
            if binding is None:
                raise RuntimeError(
                    "context attention requires an active ExecutionContext"
                )
            binding.exchange(self, q.shape[0], heads, width, q.dtype)
            k, v = parallel.distribute_key_value(k, v)
            group = parallel.context_group
            pooled_key, pooled_value = (
                group.all_gather(value, dim=0)
                for value in (pooled_key, pooled_value)
            )

        # Pooled scores [heads, query tile, key tile] in FP32: the tile means
        # of the queries and keys, scaled by the inverse square root of the
        # head width.
        scores = torch.matmul(
            pooled_query.permute(1, 0, 2), pooled_key.permute(1, 2, 0)
        ) / math.sqrt(width)

        indices, counts = select(
            scores, segments, query_tile_offset=queries.start
        )
        # The pattern bounds each query tile's key tiles; the device counts
        # state how many each actually keeps. Leading dense tiles attend
        # every key tile, which a backend may evaluate as one dense prefix.
        bounds = (
            (tiles,) * query_tiles
            if segments.key_bounds is None
            else segments.key_bounds[queries]
        )
        prefix = max(
            0, min(query_tiles, segments.dense_prefix_tiles - queries.start)
        )
        pattern = Pattern((bounds,), prefix, tiles, tile)
        fine = self.attention(
            q,
            k,
            v,
            BlockInput(
                pattern,
                indices,
                counts,
                segments.variable_block_sizes,
                queries.start,
            ),
        )

        # Compression: each query tile's softmax over the live key tiles
        # weights their mean values. The BF16 tile result then scales the
        # gate and adds to the fine rows, each operation rounding to BF16.
        live = sizes > 0
        weights = scores.masked_fill(~live, -torch.inf).softmax(-1)
        compressed = torch.matmul(weights, pooled_value.permute(1, 0, 2)).to(
            q.dtype
        )
        del q, k, v, scores, weights, pooled_query, pooled_key, pooled_value

        group = self.exchange.group
        if group.size == 1:
            if fine.is_cuda:
                from uniserve_kernels.attention import vsa_segments

                vsa_segments.add_gated_tiles(fine, compressed, gate, tile)
            else:
                tiled = (query_tiles, tile, heads, width)
                gated = compressed.permute(1, 0, 2)[:, None] * gate.view(tiled)
                fine.view(tiled).add_(gated)
            yield interval, fine
            return

        # Under Ulysses, member m owns the m-th equal share of the interval's
        # rows. Each exchanged interval is composed straight into its send
        # payload, so the gated sum is never stored in place.
        owner_rows = fine.shape[0] // group.size
        shard = interval.start + group.rank * owner_rows
        compose = functools.partial(
            compose_rows,
            fine=fine,
            compressed=compressed,
            gate=gate,
            tile=tile,
            owner_rows=owner_rows,
        )
        chunks = self.exchange.token_chunks(
            owner_rows,
            heads,
            width,
            dtype=fine.dtype,
            device=fine.device,
            compose=compose,
        )
        # Every interval is composed before the first one is yielded; the
        # attended rows are then released before the consumers allocate.
        first = next(chunks)
        del compose, fine, compressed, gate
        for rows, values in itertools.chain((first,), chunks):
            yield slice(shard + rows.start, shard + rows.stop), values

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        segments: Segments,
        *,
        norm: NormRope | None = None,
    ) -> torch.Tensor:
        """Attend and return this rank's whole token shard.

        The rows ``forward_chunks`` yields for projections that are all
        ready, concatenated: BF16 ``[rows / members, heads * members,
        width]``.
        """
        chunks = [
            values
            for _, values in self.forward_chunks(
                q, k, v, gate, segments, norm=norm
            )
        ]
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)
