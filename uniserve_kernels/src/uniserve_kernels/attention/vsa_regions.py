"""Row kernels of region-wise video sparse attention over 128-row tiles.

Region attention (``uniserve.nn.attention.vsa.RegionAttention``) keeps the
eager arithmetic of its checkpoint's recipe; these kernels perform two of its
row passes without materializing full-sequence temporaries and reproduce the
eager results bit for bit:

- ``zero_tile_padding`` writes zeros to the rows past each tile's valid size,
  so a plain per-tile sum of the rows equals the sum of the masked rows. Only
  padding rows are stored; a launch's shape depends on the tile count alone.
- ``add_gated_tiles`` adds each tile's gated compression to its rows,
  ``out + compressed[tile] * gate`` with the product and the sum each rounded
  to the tensors' dtype, as two eager element-wise operations round them.
- ``select_tiles`` chooses every query tile's key tiles from its pooled
  scores in one pass, the result of the stable sorts, cumulative sums and
  scatters of ``uniserve.nn.attention.vsa.regions.select``.

Tensors are ``[rows, heads, width]`` with any row and head strides and a unit
column stride; compressed tiles are ``[heads, tiles, width]``.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

# Rows one program covers: a tile is a whole number of row blocks.
_BLOCK_ROWS = 16

# Bits of a selection's packed sort key that carry the key tile. The 32 bits
# above them carry the score and the bits above those the region, below bit
# 62, which marks a lane past the last tile: 13 tile bits order 8192 tiles
# and leave 17 bits for the regions, which number at most the tiles.
_TILE_BITS = 13

if triton is not None:

    @triton.jit
    def _add_rn(a, b):
        # Round-to-nearest FP32 sum the compiler cannot contract with a
        # neighbouring multiply.
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=r,r,r",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=r,r,r",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _zero_tile_padding_kernel(
        values,
        valid_sizes,
        stride_row,
        stride_head,
        heads: tl.constexpr,
        tile_rows: tl.constexpr,
        width: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Store zeros to one tile's row block past the tile's valid size."""
        tile = tl.program_id(0)
        block = tl.program_id(1)
        valid = tl.load(valid_sizes + tile)
        # A row block wholly inside the valid rows stores nothing.
        if (block + 1) * block_rows > valid:
            local = block * block_rows + tl.arange(0, block_rows)
            rows = (tile * tile_rows + local).to(tl.int64)
            columns = tl.arange(0, width)
            padding = (local >= valid)[:, None]
            zeros = tl.zeros((block_rows, width), dtype=values.dtype.element_ty)
            for head in tl.static_range(heads):
                offsets = (
                    rows[:, None] * stride_row
                    + head * stride_head
                    + columns[None, :]
                )
                tl.store(values + offsets, zeros, mask=padding)

    @triton.jit
    def _add_gated_tiles_kernel(
        output,
        compressed,
        gate,
        output_stride_row,
        output_stride_head,
        gate_stride_row,
        gate_stride_head,
        compressed_stride_head,
        compressed_stride_tile,
        tile_rows: tl.constexpr,
        width: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Add one row block's gated tile compression for one head."""
        block = tl.program_id(0)
        head = tl.program_id(1)
        rows = (block * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        columns = tl.arange(0, width)
        # A row block lies within one tile.
        tile = (block * block_rows) // tile_rows
        tile_values = tl.load(
            compressed
            + head * compressed_stride_head
            + tile * compressed_stride_tile
            + columns
        ).to(tl.float32)
        gate_values = tl.load(
            gate
            + rows[:, None] * gate_stride_row
            + head * gate_stride_head
            + columns[None, :]
        ).to(tl.float32)
        destination = (
            output
            + rows[:, None] * output_stride_row
            + head * output_stride_head
            + columns[None, :]
        )
        # The product rounds to the storage dtype before the sum, as the two
        # eager operations store it in between.
        product = _mul_rn(tile_values[None, :], gate_values)
        product = product.to(
            output.dtype.element_ty, fp_downcast_rounding="rtne"
        ).to(tl.float32)
        total = _add_rn(tl.load(destination).to(tl.float32), product)
        tl.store(
            destination,
            total.to(output.dtype.element_ty, fp_downcast_rounding="rtne"),
        )

    @triton.jit
    def _select_tiles_kernel(
        scores,
        tile_regions,
        valid_sizes,
        region_starts,
        region_keep,
        kept,
        indices,
        counts,
        tiles,
        block: tl.constexpr,
        tile_bits: tl.constexpr,
    ):
        """Choose one query tile's key tiles for one head.

        A video query tile ranks the key tiles of every region by descending
        score, ties by ascending tile, and keeps each region's leading
        ``region_keep`` tiles besides every live dense tile; a dense query
        tile keeps every live tile and an empty one none. The kept tiles are
        stored first in ascending order, then the others in ascending order.
        """
        query = tl.program_id(0)
        head = tl.program_id(1)
        row = (head * tiles + query).to(tl.int64)
        keys = tl.arange(0, block)
        present = keys < tiles
        regions = tl.load(tile_regions + keys, mask=present, other=-1)
        live = tl.load(valid_sizes + keys, mask=present, other=0) > 0
        query_region = tl.load(tile_regions + query)
        query_live = tl.load(valid_sizes + query) > 0

        mask = live & query_live
        if query_live & (query_region >= 0):
            values = tl.load(
                scores + row * tiles + keys, mask=present, other=0.0
            )
            # Ascending order key of the FP32 score: the sign-flipped bits,
            # with -0 as +0 and every NaN above +inf, as the stable sort
            # orders them. Descending order is its complement.
            bits = (values + 0.0).to(tl.int32, bitcast=True).to(tl.int64)
            bits = bits & 0xFFFFFFFF
            ascending = tl.where(
                bits >= 0x80000000, 0xFFFFFFFF - bits, bits | 0x80000000
            )
            ascending = tl.where(values != values, 0xFFFFFFFF, ascending)
            # One packed key orders the key tiles by region (dense and empty
            # tiles first), then descending score, then ascending tile.
            packed = (
                ((regions + 1).to(tl.int64) << (32 + tile_bits))
                | ((0xFFFFFFFF - ascending) << tile_bits)
                | keys.to(tl.int64)
            )
            # Lanes past the last tile sort after every key tile, so the
            # first ``tiles`` positions of the order hold the key tiles and
            # ``present`` marks them by position as well as by tile.
            packed = tl.where(present, packed, 1 << 62)
            ordered = tl.sort(packed)

            # A key's rank within its region is its position in that order
            # less the region's first position. The trailing lanes decode to
            # tile 0 of a region past every table, so they neither read the
            # region tables nor store a choice.
            ordered_regions = (ordered >> (32 + tile_bits)).to(tl.int32) - 1
            ordered_keys = (ordered & ((1 << tile_bits) - 1)).to(tl.int32)
            ranked = present & (ordered_regions >= 0)
            region = tl.maximum(ordered_regions, 0)
            first = tl.load(region_starts + region, mask=ranked, other=0)
            keep = tl.load(region_keep + region, mask=ranked, other=0)
            chosen = ranked & (keys - first < keep)

            # Return the choice to tile order through this program's row of
            # scratch; the barrier publishes the stores to the whole program.
            scratch = kept + row * block
            tl.store(scratch + ordered_keys, chosen.to(tl.int8), mask=present)
            tl.debug_barrier()
            chosen = tl.load(scratch + keys, mask=present, other=0) != 0
            mask = (live & (regions < 0)) | chosen
        mask = mask & present

        count = tl.sum(mask.to(tl.int32), axis=0)
        kept_before = tl.cumsum(mask.to(tl.int32), axis=0) - 1
        others_before = tl.cumsum((present & ~mask).to(tl.int32), axis=0) - 1
        slots = tl.where(mask, kept_before, count + others_before)
        tl.store(indices + row * tiles + slots, keys, mask=present)
        tl.store(counts + row, count)


def _rows_layout(value: torch.Tensor, name: str) -> None:
    if value.ndim != 3 or value.stride(2) != 1:
        raise ValueError(
            f"{name} must be [rows, heads, width] with unit column stride"
        )


def zero_tile_padding(
    values: torch.Tensor, valid_sizes: torch.Tensor, tile: int
) -> None:
    """Store zeros to every row past its tile's valid size, in place.

    ``values`` is ``[tiles * tile, heads, width]`` with unit column stride
    and ``valid_sizes`` the int32 valid rows of each ``tile``-row tile, read
    on the device. Rows within a tile's valid size are not touched.

    Raises:
        RuntimeError: Triton cannot launch on the tensors' device.
        ValueError: The rows are not whole tiles of the table's tiles.
    """
    if not launchable(values.device):
        raise RuntimeError("tile padding requires Triton")
    _rows_layout(values, "tile rows")
    tiles = int(valid_sizes.numel())
    if (
        tile % _BLOCK_ROWS
        or values.shape[0] != tiles * tile
        or valid_sizes.dtype != torch.int32
    ):
        raise ValueError(
            "tile padding needs whole tiles and an int32 valid-size table"
        )
    rows, heads, width = (int(size) for size in values.shape)
    if rows == 0:
        return
    _zero_tile_padding_kernel[(tiles, tile // _BLOCK_ROWS)](
        values,
        valid_sizes,
        int(values.stride(0)),
        int(values.stride(1)),
        heads,
        tile,
        width,
        _BLOCK_ROWS,
        num_warps=4,
    )


def add_gated_tiles(
    output: torch.Tensor,
    compressed: torch.Tensor,
    gate: torch.Tensor,
    tile: int,
) -> None:
    """Add each tile's gated compression to its rows, in place.

    For every row ``r`` of tile ``t``, ``output[r] = output[r] +
    compressed[:, t] * gate[r]``, the product and the sum each rounded to
    nearest even in the tensors' common dtype: the result of
    ``output.view(tiles, tile, heads, width).add_(compressed.permute(1, 0,
    2)[:, None] * gate.view(tiles, tile, heads, width))`` evaluated eagerly.
    ``output`` and ``gate`` are ``[tiles * tile, heads, width]`` and
    ``compressed`` is ``[heads, tiles, width]``, each with unit column
    stride.

    Raises:
        RuntimeError: Triton cannot launch on the tensors' device.
        ValueError: The shapes, dtypes or devices disagree.
    """
    if not launchable(output.device):
        raise RuntimeError("gated tile compression requires Triton")
    for value, name in ((output, "output"), (gate, "gate")):
        _rows_layout(value, name)
    rows, heads, width = (int(size) for size in output.shape)
    if (
        gate.shape != output.shape
        or compressed.ndim != 3
        or compressed.stride(2) != 1
        or compressed.shape != (heads, rows // tile, width)
        or tile % _BLOCK_ROWS
        or rows % tile
        or len({value.dtype for value in (output, compressed, gate)}) != 1
        or len({value.device for value in (output, compressed, gate)}) != 1
    ):
        raise ValueError(
            "gated tile compression needs whole tiles of one dtype and device"
        )
    if rows == 0:
        return
    _add_gated_tiles_kernel[(rows // _BLOCK_ROWS, heads)](
        output,
        compressed,
        gate,
        int(output.stride(0)),
        int(output.stride(1)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        int(compressed.stride(0)),
        int(compressed.stride(1)),
        tile,
        width,
        _BLOCK_ROWS,
        num_warps=4,
    )


def select_tiles(
    scores: torch.Tensor,
    tile_regions: torch.Tensor,
    valid_sizes: torch.Tensor,
    region_starts: torch.Tensor,
    region_keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose every query tile's key tiles from its pooled scores.

    ``scores`` is the contiguous FP32 ``[heads, tiles, tiles]`` (query tile,
    key tile) matrix and the tables are the int32 per-tile vectors of
    ``uniserve.nn.attention.vsa.Regions``. A dense query tile keeps every
    live key tile; a video query tile keeps every live dense tile and, of
    each region ``r``, the ``region_keep[r]`` tiles of that region with the
    highest scores, equal scores in ascending tile order, NaN above every
    number and -0 equal to +0; an empty query tile keeps none.

    Returns ``[heads, tiles, tiles]`` int32 key-tile indices, each query
    tile's kept tiles first in ascending order followed by the unkept tiles
    in ascending order, and ``[heads, tiles]`` int32 counts.

    Raises:
        RuntimeError: Triton cannot launch on the scores' device.
        ValueError: The scores or tables do not describe one tile set, or
            there are more tiles than the packed key's 13 tile bits hold.
    """
    if not launchable(scores.device):
        raise RuntimeError("tile selection requires Triton")
    heads, tiles = int(scores.shape[0]), int(scores.shape[-1])
    tables = (tile_regions, valid_sizes, region_starts, region_keep)
    if (
        scores.ndim != 3
        or scores.shape[1] != tiles
        or scores.dtype != torch.float32
        or not scores.is_contiguous()
        or any(
            table.shape != (tiles,)
            or table.dtype != torch.int32
            or table.device != scores.device
            for table in tables
        )
        or tiles > 1 << _TILE_BITS
    ):
        raise ValueError(
            "tile selection needs [heads, tiles, tiles] FP32 scores, int32 "
            f"tables of the same tiles and at most {1 << _TILE_BITS} tiles"
        )
    indices = torch.empty(
        (heads, tiles, tiles), dtype=torch.int32, device=scores.device
    )
    counts = torch.empty(
        (heads, tiles), dtype=torch.int32, device=scores.device
    )
    if heads == 0 or tiles == 0:
        return indices, counts
    block = triton.next_power_of_2(tiles)
    # Each program returns its region choice to tile order through one row.
    kept = torch.empty(
        (heads * tiles, block), dtype=torch.int8, device=scores.device
    )
    _select_tiles_kernel[(tiles, heads)](
        scores,
        tile_regions,
        valid_sizes,
        region_starts,
        region_keep,
        kept,
        indices,
        counts,
        tiles,
        block,
        _TILE_BITS,
        num_warps=8 if block >= 1024 else 4,
    )
    return indices, counts


__all__ = ["add_gated_tiles", "select_tiles", "zero_tile_padding"]
