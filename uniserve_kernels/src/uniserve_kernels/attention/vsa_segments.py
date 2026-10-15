"""Row kernels of segment-wise video sparse attention over row tiles.

Segment attention (``uniserve.nn.attention.vsa.SegmentAttention``) keeps the
eager arithmetic of its checkpoint's recipe; these kernels perform its row
passes without materializing full-sequence temporaries and reproduce the
eager results bit for bit:

- ``prepare_tiles`` prepares one projected row interval in a single pass:
  it normalizes and rotates Q and K as ``qk_norm_rope`` evaluates
  ``Rounding.STEPWISE`` (``uniserve_kernels.rope``), stores zeros to the Q,
  K and V rows past each tile's valid size, and writes each tile's FP32
  means of the stored rows in the summation order of
  ``rows.view(tiles, tile, heads, width).sum(1, dtype=torch.float32)``.
- ``zero_tile_padding`` writes zeros to the rows past each tile's valid size,
  so a plain per-tile sum of the rows equals the sum of the masked rows. Only
  padding rows are stored; a launch's shape depends on the tile count alone.
- ``add_gated_tiles`` adds each tile's gated compression to its rows,
  ``out + compressed[tile] * gate`` with the product and the sum each rounded
  to the tensors' dtype, as two eager element-wise operations round them.
- ``compose_gated_rows`` evaluates the same sum for one row interval of every
  Ulysses member's token shard and stores it member-major, the send payload
  of that interval's exchange.
- ``select_tiles`` chooses every query tile's key tiles from its pooled
  scores in one pass, the result of the stable sorts, cumulative sums and
  scatters of ``uniserve.nn.attention.vsa.segments.select``. The query tiles
  may be a window of the sequence's tiles, as a context partition holds.

Tensors are ``[rows, heads, width]`` with any row and head strides and a unit
column stride; compressed tiles are ``[heads, tiles, width]``.
"""

from __future__ import annotations

import torch

from uniserve_kernels.rope import token_rows
from uniserve_kernels.triton import launchable, tl, triton

if triton is not None:  # pragma: no cover - depends on the accelerator stack.
    # The stepwise split-half Q/K row routine of ``qk_norm_rope``; calling it
    # on the same 16-row blocks with the same warps stores the same bits.
    from uniserve_kernels.rope import _norm_rope_halves

# Rows one program covers: a tile is a whole number of row blocks.
_BLOCK_ROWS = 16

# Bits of a selection's packed sort key that carry the key tile. The 32 bits
# above them carry the score and the bits above those the segment, below bit
# 62, which marks a lane past the last tile: 13 tile bits order 8192 tiles
# and leave 17 bits for the segments, which number at most the tiles.
# FIXME: the packed key bounds selection at 8192 tiles (1,048,576 rows of
# 128); a longer sequence needs a wider key or a multi-pass sort.
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
        variable_block_sizes,
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
        valid = tl.load(variable_block_sizes + tile)
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

    @triton.jit(do_not_specialize=["owner_rows", "start", "count"])
    def _compose_gated_rows_kernel(
        output,
        fine,
        compressed,
        gate,
        fine_stride_row,
        fine_stride_head,
        gate_stride_row,
        gate_stride_head,
        compressed_stride_head,
        compressed_stride_tile,
        owner_rows,
        start,
        count,
        heads: tl.constexpr,
        tile_rows: tl.constexpr,
        width: tl.constexpr,
        block_rows: tl.constexpr,
        slots: tl.constexpr,
    ):
        """Compose one row block of one member's interval for one head.

        Row ``i`` of member ``m``'s interval is row ``m * owner_rows + start
        + i`` of ``fine`` and ``gate``; its sum lands in row ``i`` of slot
        ``slots[m]`` of the contiguous ``[members, count, heads, width]``
        payload, with the arithmetic of :func:`_add_gated_tiles_kernel`.
        """
        block = tl.program_id(0)
        head = tl.program_id(1)
        member = tl.program_id(2).to(tl.int64)
        local = (block * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        first = member * owner_rows + start + block * block_rows
        rows = first + tl.arange(0, block_rows)
        columns = tl.arange(0, width)
        # Owner rows, interval starts and counts are whole row blocks, so a
        # row block lies within one tile.
        tile = first // tile_rows
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
        fine_values = tl.load(
            fine
            + rows[:, None] * fine_stride_row
            + head * fine_stride_head
            + columns[None, :]
        ).to(tl.float32)
        product = _mul_rn(tile_values[None, :], gate_values)
        product = product.to(
            output.dtype.element_ty, fp_downcast_rounding="rtne"
        ).to(tl.float32)
        total = _add_rn(fine_values, product)

        slot = member
        for index in tl.static_range(len(slots)):
            slot = tl.where(member == index, slots[index], slot)
        destination = (
            output
            + ((slot * count + local)[:, None] * heads + head) * width
            + columns[None, :]
        )
        tl.store(
            destination,
            total.to(output.dtype.element_ty, fp_downcast_rounding="rtne"),
        )

    @triton.jit
    def _phase_sum(
        rows,
        row_stride,
        columns,
        phase: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
    ):
        """Sum the rows ``phase``, ``phase + 4``, ... of one tile.

        Four accumulators take, in row order, the phase's rows ``phase + 4 *
        i + 16 * m`` for ``i`` = 0, 1, 2 and 3, and their sums then add left
        to right: one thread's share of ``torch.sum``'s reduction below.
        """
        first = tl.zeros((width,), dtype=tl.float32)
        second = tl.zeros((width,), dtype=tl.float32)
        third = tl.zeros((width,), dtype=tl.float32)
        fourth = tl.zeros((width,), dtype=tl.float32)
        for step in tl.static_range(tile_rows // 16):
            base = rows + (16 * step + phase) * row_stride + columns
            first = _add_rn(first, tl.load(base).to(tl.float32))
            second = _add_rn(
                second, tl.load(base + 4 * row_stride).to(tl.float32)
            )
            third = _add_rn(
                third, tl.load(base + 8 * row_stride).to(tl.float32)
            )
            fourth = _add_rn(
                fourth, tl.load(base + 12 * row_stride).to(tl.float32)
            )
        return _add_rn(_add_rn(_add_rn(first, second), third), fourth)

    @triton.jit
    def _tile_mean(
        rows,
        row_stride,
        columns,
        divisor,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
    ):
        """Return the FP32 mean of the ``tile_rows`` rows at ``rows``.

        Row ``r`` starts at ``rows + r * row_stride``. The sum follows the
        order in which ``torch.sum(dtype=torch.float32)`` reduces a strided
        row axis of 64 or 128 rows: four threads each take the rows of one
        phase modulo four (:func:`_phase_sum`), and a halving tree adds their
        sums as ``(p0 + p2) + (p1 + p3)``. Every sum and the division round
        to nearest, so the mean equals the eager mean bit for bit.
        """
        first = _phase_sum(rows, row_stride, columns, 0, width, tile_rows)
        second = _phase_sum(rows, row_stride, columns, 1, width, tile_rows)
        third = _phase_sum(rows, row_stride, columns, 2, width, tile_rows)
        fourth = _phase_sum(rows, row_stride, columns, 3, width, tile_rows)
        total = _add_rn(_add_rn(first, third), _add_rn(second, fourth))
        return tl.math.div_rn(total, divisor)

    @triton.jit
    def _prepare_tiles_kernel(
        query,
        key,
        value,
        query_weight,
        key_weight,
        cosine,
        sine,
        variable_block_sizes,
        pooled_query,
        pooled_key,
        pooled_value,
        QUERY_STRIDES: tl.constexpr,  # noqa: N803
        KEY_STRIDES: tl.constexpr,  # noqa: N803
        VALUE_STRIDES: tl.constexpr,  # noqa: N803
        QUERY_ROWS: tl.constexpr,  # noqa: N803
        KEY_ROWS: tl.constexpr,  # noqa: N803
        FACTOR_STRIDE: tl.constexpr,  # noqa: N803
        HEADS: tl.constexpr,  # noqa: N803
        DIM: tl.constexpr,  # noqa: N803
        ROTATED: tl.constexpr,  # noqa: N803
        EPS: tl.constexpr,  # noqa: N803
        TILE: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
        NORM: tl.constexpr,  # noqa: N803
    ):
        """Prepare one tile of one head: rotate Q/K, zero padding, pool.

        ``*_STRIDES`` are the (row, head) strides of Q, K and V; ``QUERY_ROWS``
        and ``KEY_ROWS`` restate the row strides as the (token, head) strides
        of a single-head row sequence, the form :func:`_norm_rope_halves`
        reads. With ``NORM``, each ``BLOCK``-row block's valid rows of Q and
        K are normalized and rotated in place by that routine on the blocks
        and warps of ``qk_norm_rope``'s stepwise launch, which stores its
        bits. The rows past the tile's valid size of Q, K and V then hold
        zeros, and once every store of the program is visible the tile's
        means of the stored rows are written to the contiguous ``[tiles,
        HEADS, DIM]`` pooled outputs.
        """
        tile = tl.program_id(0)
        head = tl.program_id(1)
        valid = tl.load(variable_block_sizes + tile)
        columns = tl.arange(0, DIM)
        first = tile.to(tl.int64) * TILE
        query_tile = query + first * QUERY_STRIDES[0] + head * QUERY_STRIDES[1]
        key_tile = key + first * KEY_STRIDES[0] + head * KEY_STRIDES[1]
        value_tile = value + first * VALUE_STRIDES[0] + head * VALUE_STRIDES[1]

        for block in tl.static_range(TILE // BLOCK):
            offset = block * BLOCK
            if NORM:
                # Only the block's valid rows are normalized and stored, so
                # the padding stores below touch disjoint rows.
                count = tl.minimum(tl.maximum(valid - offset, 0), BLOCK)
                factors = (first + offset) * FACTOR_STRIDE
                block_query = query_tile + offset * QUERY_STRIDES[0]
                block_key = key_tile + offset * KEY_STRIDES[0]
                _norm_rope_halves(
                    block_query,
                    query_weight,
                    cosine + factors,
                    sine + factors,
                    block_query,
                    0,
                    count,
                    1,
                    QUERY_ROWS,
                    QUERY_ROWS,
                    DIM,
                    ROTATED,
                    FACTOR_STRIDE,
                    EPS,
                    BLOCK,
                )
                _norm_rope_halves(
                    block_key,
                    key_weight,
                    cosine + factors,
                    sine + factors,
                    block_key,
                    0,
                    count,
                    1,
                    KEY_ROWS,
                    KEY_ROWS,
                    DIM,
                    ROTATED,
                    FACTOR_STRIDE,
                    EPS,
                    BLOCK,
                )

            local = offset + tl.arange(0, BLOCK)
            padding = (local >= valid)[:, None]
            zeros = tl.zeros((BLOCK, DIM), dtype=query.dtype.element_ty)
            tl.store(
                query_tile
                + local[:, None] * QUERY_STRIDES[0]
                + columns[None, :],
                zeros,
                mask=padding,
            )
            tl.store(
                key_tile + local[:, None] * KEY_STRIDES[0] + columns[None, :],
                zeros,
                mask=padding,
            )
            tl.store(
                value_tile
                + local[:, None] * VALUE_STRIDES[0]
                + columns[None, :],
                zeros,
                mask=padding,
            )

        # Other threads of the program stored rows the means read.
        tl.debug_barrier()
        divisor = tl.maximum(valid, 1).to(tl.float32)
        pooled = (tile.to(tl.int64) * HEADS + head) * DIM + columns
        tl.store(
            pooled_query + pooled,
            _tile_mean(
                query_tile, QUERY_STRIDES[0], columns, divisor, DIM, TILE
            ),
        )
        tl.store(
            pooled_key + pooled,
            _tile_mean(key_tile, KEY_STRIDES[0], columns, divisor, DIM, TILE),
        )
        tl.store(
            pooled_value + pooled,
            _tile_mean(
                value_tile, VALUE_STRIDES[0], columns, divisor, DIM, TILE
            ),
        )

    @triton.jit(do_not_specialize=["query_offset"])
    def _select_tiles_kernel(
        scores,
        tile_segments,
        variable_block_sizes,
        segment_starts,
        segment_keep,
        kept,
        indices,
        counts,
        query_tiles,
        tiles,
        query_offset,
        block: tl.constexpr,
        tile_bits: tl.constexpr,
    ):
        """Choose one query tile's key tiles for one head.

        A video query tile ranks the key tiles of every segment by descending
        score, ties by ascending tile, and keeps each segment's leading
        ``segment_keep`` tiles besides every live dense tile; a dense query
        tile keeps every live tile and an empty one none. The kept tiles are
        stored first in ascending order, then the others in ascending order.
        Query ``q`` of the program grid is tile ``query_offset + q`` of the
        sequence; the tables hold every tile.
        """
        query = tl.program_id(0)
        head = tl.program_id(1)
        row = (head * query_tiles + query).to(tl.int64)
        keys = tl.arange(0, block)
        present = keys < tiles
        segments = tl.load(tile_segments + keys, mask=present, other=-1)
        live = tl.load(variable_block_sizes + keys, mask=present, other=0) > 0
        query_segment = tl.load(tile_segments + query_offset + query)
        query_live = tl.load(variable_block_sizes + query_offset + query) > 0

        mask = live & query_live
        if query_live & (query_segment >= 0):
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
            # One packed key orders the key tiles by segment (dense and empty
            # tiles first), then descending score, then ascending tile.
            packed = (
                ((segments + 1).to(tl.int64) << (32 + tile_bits))
                | ((0xFFFFFFFF - ascending) << tile_bits)
                | keys.to(tl.int64)
            )
            # Lanes past the last tile sort after every key tile, so the
            # first ``tiles`` positions of the order hold the key tiles and
            # ``present`` marks them by position as well as by tile.
            packed = tl.where(present, packed, 1 << 62)
            ordered = tl.sort(packed)

            # A key's rank within its segment is its position in that order
            # less the segment's first position. The trailing lanes decode to
            # tile 0 of a segment past every table, so they neither read the
            # segment tables nor store a choice.
            ordered_segments = (ordered >> (32 + tile_bits)).to(tl.int32) - 1
            ordered_keys = (ordered & ((1 << tile_bits) - 1)).to(tl.int32)
            ranked = present & (ordered_segments >= 0)
            segment = tl.maximum(ordered_segments, 0)
            first = tl.load(segment_starts + segment, mask=ranked, other=0)
            keep = tl.load(segment_keep + segment, mask=ranked, other=0)
            chosen = ranked & (keys - first < keep)

            # Return the choice to tile order through this program's row of
            # scratch; the barrier publishes the stores to the whole program.
            scratch = kept + row * block
            tl.store(scratch + ordered_keys, chosen.to(tl.int8), mask=present)
            tl.debug_barrier()
            chosen = tl.load(scratch + keys, mask=present, other=0) != 0
            mask = (live & (segments < 0)) | chosen
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
    values: torch.Tensor, variable_block_sizes: torch.Tensor, tile: int
) -> None:
    """Store zeros to every row past its tile's valid size, in place.

    ``values`` is ``[tiles * tile, heads, width]`` with unit column stride
    and ``variable_block_sizes`` the int32 valid rows of each ``tile``-row
    tile, read on the device. Rows within a tile's valid size are not touched.

    Raises:
        RuntimeError: Triton cannot launch on the tensors' device.
        ValueError: The rows are not whole tiles of the table's tiles.
    """
    if not launchable(values.device):
        raise RuntimeError("tile padding requires Triton")
    _rows_layout(values, "tile rows")
    tiles = int(variable_block_sizes.numel())
    if (
        tile % _BLOCK_ROWS
        or values.shape[0] != tiles * tile
        or variable_block_sizes.dtype != torch.int32
    ):
        raise ValueError(
            "tile padding needs whole tiles and an int32 valid-size table"
        )
    rows, heads, width = (int(size) for size in values.shape)
    if rows == 0:
        return
    _zero_tile_padding_kernel[(tiles, tile // _BLOCK_ROWS)](
        values,
        variable_block_sizes,
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


def prepare_tiles(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    tile: int,
    pooled_query: torch.Tensor,
    pooled_key: torch.Tensor,
    pooled_value: torch.Tensor,
    *,
    query_weight: torch.Tensor | None = None,
    key_weight: torch.Tensor | None = None,
    cosine: torch.Tensor | None = None,
    sine: torch.Tensor | None = None,
    eps: float | None = None,
) -> None:
    """Prepare whole tiles of projected Q/K/V rows for segment attention.

    ``query``, ``key`` and ``value`` are ``[tiles * tile, heads, width]``
    views of one projected row interval with unit column stride, and
    ``variable_block_sizes`` holds the int32 valid rows of each of its tiles.
    With the normalization operands, Q and K are first normalized and rotated
    in place exactly as ``uniserve_kernels.rope.qk_norm_rope`` evaluates the
    stepwise recipe for one RMS domain over the head, weighted by
    ``query_weight`` and ``key_weight``, and one split-half axis whose
    ``[rows, rotated / 2]`` factor tables ``cosine`` and ``sine`` hold one
    row per interval row; without them Q and K are used as they are. Then
    the Q, K and V rows past each tile's valid size hold zeros, and each
    tile's FP32 means of every head's stored rows are written to the
    contiguous ``[tiles, heads, width]`` pooled outputs, the means
    ``uniserve.nn.attention.vsa.segments.pool`` computes from the stored
    rows, bit for bit. An empty tile's means are zeros.

    Raises:
        RuntimeError: Triton cannot launch on the tensors' device.
        ValueError: The rows are not whole tiles of the table, a tensor's
            shape, dtype, device or layout disagrees, or the normalization
            operands are partial or fall outside the stepwise split-half
            layout of ``qk_norm_rope`` (16-bit Q and K rows on 16-byte
            boundaries, a power-of-two head of 64 to 256 features and a
            rotated width that is a positive multiple of four).
    """
    from uniserve_kernels.rope import _uses_halves

    if not launchable(query.device):
        raise RuntimeError("tile preparation requires Triton")
    for tensor, name in ((query, "query"), (key, "key"), (value, "value")):
        _rows_layout(tensor, name)
    rows, heads, width = (int(size) for size in query.shape)
    tiles = int(variable_block_sizes.numel())
    pooled = (pooled_query, pooled_key, pooled_value)
    if (
        key.shape != query.shape
        or value.shape != query.shape
        or len({tensor.dtype for tensor in (query, key, value)}) != 1
        or query.dtype not in (torch.bfloat16, torch.float16)
        or tile % _BLOCK_ROWS
        or rows != tiles * tile
        or variable_block_sizes.dtype != torch.int32
        or width & (width - 1)
        or any(
            tensor.shape != (tiles, heads, width)
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            for tensor in pooled
        )
        or len(
            {
                tensor.device
                for tensor in (query, key, value, variable_block_sizes, *pooled)
            }
        )
        != 1
    ):
        raise ValueError(
            "tile preparation needs whole tiles of equal 16-bit Q/K/V rows, "
            "an int32 valid-size table and contiguous FP32 means of the same "
            "tiles"
        )

    operands = (query_weight, key_weight, cosine, sine, eps)
    norm = query_weight is not None
    if any((operand is None) == norm for operand in operands):
        raise ValueError("Q/K normalization needs every operand or none")
    rotated = 0
    factor_stride = 0
    if norm:
        assert query_weight is not None and key_weight is not None
        assert cosine is not None and sine is not None
        rotated = int(cosine.shape[-1]) * 2
        factor_rows = token_rows(cosine, 1)
        if (
            query_weight.shape != (width,)
            or key_weight.shape != (width,)
            or not query_weight.is_contiguous()
            or not key_weight.is_contiguous()
            or cosine.ndim != 2
            or cosine.shape[0] != rows
            or sine.shape != cosine.shape
            or sine.stride() != cosine.stride()
            or factor_rows is None
            or not _uses_halves(
                (query, key),
                (query, key),
                (width,),
                (query_weight, key_weight, cosine, sine),
                (rotated,),
            )
        ):
            raise ValueError(
                "fused Q/K normalization needs the stepwise split-half "
                "layout of qk_norm_rope"
            )
        factor_stride = int(factor_rows[0])
    if rows == 0:
        return

    query_strides = (int(query.stride(0)), int(query.stride(1)))
    key_strides = (int(key.stride(0)), int(key.stride(1)))
    # Unused pointers of the plain launch stand in for the absent operands.
    _prepare_tiles_kernel[(tiles, heads)](
        query,
        key,
        value,
        query_weight if norm else query,
        key_weight if norm else key,
        cosine if norm else query,
        sine if norm else query,
        variable_block_sizes,
        pooled_query,
        pooled_key,
        pooled_value,
        query_strides,
        key_strides,
        (int(value.stride(0)), int(value.stride(1))),
        (query_strides[0], 0),
        (key_strides[0], 0),
        factor_stride,
        heads,
        width,
        rotated,
        float(eps) if norm else 0.0,
        tile,
        _BLOCK_ROWS,
        norm,
        # The stepwise routine's warps and unfused arithmetic, which fix the
        # bits it stores; the means round every sum explicitly.
        num_warps=2,
        enable_fp_fusion=False,
    )


def compose_gated_rows(
    output: torch.Tensor,
    fine: torch.Tensor,
    compressed: torch.Tensor,
    gate: torch.Tensor,
    tile: int,
    *,
    owner_rows: int,
    start: int,
    slots: tuple[int, ...],
) -> None:
    """Store one row interval of every member's gated rows member-major.

    ``fine`` and ``gate`` are ``[members * owner_rows, heads, width]`` with
    unit column stride: member ``m`` owns rows ``m * owner_rows`` onward.
    ``output`` is the contiguous ``[members, count, heads, width]`` payload of
    the interval ``start:start + count`` of each member's rows; slot
    ``slots[m]`` receives member ``m``'s rows. Each value is ``fine +
    compressed[:, tile] * gate`` rounded as :func:`add_gated_tiles` rounds
    it; ``fine`` is not modified.

    Raises:
        RuntimeError: Triton cannot launch on the tensors' device.
        ValueError: The interval leaves the owners' rows, is not whole
            16-row blocks, or the tensors disagree in shape, dtype or device.
    """
    if not launchable(output.device):
        raise RuntimeError("gated row composition requires Triton")
    for value, name in ((fine, "fine rows"), (gate, "gate")):
        _rows_layout(value, name)
    members, count, heads, width = (int(size) for size in output.shape)
    if (
        output.ndim != 4
        or not output.is_contiguous()
        or fine.shape != (members * owner_rows, heads, width)
        or gate.shape != fine.shape
        or compressed.ndim != 3
        or compressed.stride(2) != 1
        or compressed.shape != (heads, fine.shape[0] // tile, width)
        or sorted(slots) != list(range(members))
        or tile % _BLOCK_ROWS
        or owner_rows % _BLOCK_ROWS
        or start % _BLOCK_ROWS
        or count % _BLOCK_ROWS
        or not 0 <= start <= owner_rows - count
        or len({value.dtype for value in (output, fine, compressed, gate)}) != 1
        or len({value.device for value in (output, fine, compressed, gate)})
        != 1
    ):
        raise ValueError(
            "gated row composition needs whole row blocks of every owner's "
            "tiles, one dtype and one device"
        )
    if count == 0:
        return
    _compose_gated_rows_kernel[(count // _BLOCK_ROWS, heads, members)](
        output,
        fine,
        compressed,
        gate,
        int(fine.stride(0)),
        int(fine.stride(1)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        int(compressed.stride(0)),
        int(compressed.stride(1)),
        owner_rows,
        start,
        count,
        heads,
        tile,
        width,
        _BLOCK_ROWS,
        tuple(int(slot) for slot in slots),
        num_warps=4,
    )


def select_tiles(
    scores: torch.Tensor,
    tile_segments: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    segment_starts: torch.Tensor,
    segment_keep: torch.Tensor,
    *,
    query_tile_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose every query tile's key tiles from its pooled scores.

    ``scores`` is the contiguous FP32 ``[heads, query tiles, tiles]`` (query
    tile, key tile) matrix and the tables are the int32 per-tile vectors of
    ``uniserve.nn.attention.vsa.Segments``, one entry per tile of the
    sequence. Query tile ``q`` of the scores is tile ``query_tile_offset + q``
    of the sequence. A dense query tile keeps every live key tile; a video
    query tile keeps every live dense tile and, of each segment ``r``, the
    ``segment_keep[r]`` tiles of that segment with the highest scores, equal
    scores in ascending tile order, NaN above every number and -0 equal to
    +0; an empty query tile keeps none.

    Returns ``[heads, query tiles, tiles]`` int32 key-tile indices, each query
    tile's kept tiles first in ascending order followed by the unkept tiles
    in ascending order, and ``[heads, query tiles]`` int32 counts.

    Raises:
        RuntimeError: Triton cannot launch on the scores' device.
        ValueError: The scores or tables do not describe one tile set, the
            query tiles leave the sequence, or there are more tiles than the
            packed key's 13 tile bits hold.
    """
    if not launchable(scores.device):
        raise RuntimeError("tile selection requires Triton")
    heads, query_tiles, tiles = (int(size) for size in scores.shape)
    tables = (tile_segments, variable_block_sizes, segment_starts, segment_keep)
    if (
        scores.ndim != 3
        or scores.dtype != torch.float32
        or not scores.is_contiguous()
        or any(
            table.shape != (tiles,)
            or table.dtype != torch.int32
            or table.device != scores.device
            for table in tables
        )
        or not 0 <= query_tile_offset <= tiles - query_tiles
        or tiles > 1 << _TILE_BITS
    ):
        raise ValueError(
            "tile selection needs [heads, query tiles, tiles] FP32 scores "
            "over a window of the tables' tiles, int32 tables of the same "
            f"tiles and at most {1 << _TILE_BITS} tiles"
        )
    indices = torch.empty(
        (heads, query_tiles, tiles), dtype=torch.int32, device=scores.device
    )
    counts = torch.empty(
        (heads, query_tiles), dtype=torch.int32, device=scores.device
    )
    if heads == 0 or query_tiles == 0:
        return indices, counts
    block = triton.next_power_of_2(tiles)
    # Each program returns its segment choice to tile order through one row.
    kept = torch.empty(
        (heads * query_tiles, block), dtype=torch.int8, device=scores.device
    )
    _select_tiles_kernel[(query_tiles, heads)](
        scores,
        tile_segments,
        variable_block_sizes,
        segment_starts,
        segment_keep,
        kept,
        indices,
        counts,
        query_tiles,
        tiles,
        query_tile_offset,
        block,
        _TILE_BITS,
        num_warps=8 if block >= 1024 else 4,
    )
    return indices, counts


__all__ = [
    "add_gated_tiles",
    "compose_gated_rows",
    "prepare_tiles",
    "select_tiles",
    "zero_tile_padding",
]
