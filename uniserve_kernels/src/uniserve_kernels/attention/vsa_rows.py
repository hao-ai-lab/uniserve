"""Packed input layouts and row-wise compression epilogues for VSA.

The VSA layer (``uniserve.nn.attention.vsa``) receives projected Q/K/V as
tile-aligned row intervals that may arrive out of order.
``pack_sparse_input_rows`` and ``prepare_sparse_input_rows`` publish each
interval into one caller-owned ``packed`` buffer holding all three components;
``compose_attention`` adds the gated per-tile compression to the provider's
attention output and routes the composed rows into row-owner destinations.
The runtime VSA operators under ``uniserve.runtime.backends.attention.vsa``
(``_Rows.prepare`` and ``_flashinfer.prepare_rows``) read the layer's
``packed`` buffer (FlashInfer's BSR path receives a head-major copy of it), or
pack their own inputs when given none, and compose their attention output with
``compose_attention``.

``packed`` is ``[3, rows, heads, width]`` (row-major, used by the row
producers of the Triton, CuTe and SM100 operators and by the SM120 FlashInfer
kernel) or ``[3, heads, rows, width]`` (head-major, used by FlashInfer's
head-flattened BSR path). K and V keep packed row order, with rows past their
tile's valid size stored as zeros. Q uses owner-interval order: the
``rows // owners`` rows of each owner are cut into ``chunk_rows`` segments,
and segment ``s`` of every owner is stored contiguously, owner after owner,
starting at row ``s * chunk_rows * owners``. One exchange interval of all
owners is therefore one contiguous slice of the Q component:
``_flashinfer.prepare_rows`` narrows Q to that slice, and ``_Rows.prepare``
slices the output of its whole-domain attention launch in the same order. In
head-major storage each segment is itself head-major. With one owner and a
single segment the Q order equals the row order.

Shapes and strides specialize the kernels. Input preparation receives the
interval's row offset at runtime, so intervals of the same numerical shape
reuse a compiled kernel while each graph node retains its actual offset.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_TILE = 64


if triton is not None:

    @triton.jit(do_not_specialize=["row_start"])
    def _pack_masked_qkv_kernel(
        query,
        key,
        value,
        valid_sizes,
        packed,
        query_stride_row: tl.constexpr,
        query_stride_head: tl.constexpr,
        key_stride_row: tl.constexpr,
        key_stride_head: tl.constexpr,
        value_stride_row: tl.constexpr,
        value_stride_head: tl.constexpr,
        rows: tl.constexpr,
        input_rows: tl.constexpr,
        row_start,
        heads: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
        owners: tl.constexpr,
        chunk_rows: tl.constexpr,
        row_major: tl.constexpr,
    ):
        """Pack one block of interval rows for one head into ``packed``.

        Grid: ``(cdiv(input_rows, block_rows), heads)``. ``query``, ``key``
        and ``value`` are the interval's ``[input_rows, heads, width]`` views
        with unit column stride. Interval row ``i`` is packed row
        ``row_start + i``, which also selects its entry in ``valid_sizes``.
        """
        input_offsets = (
            tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        ).to(tl.int64)
        row_offsets = row_start + input_offsets
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        row_mask = input_offsets[:, None] < input_rows

        # K/V rows past a tile's valid size load as zeros and are stored as
        # zeros. The providers also mask those keys; zero payload keeps
        # padding values, NaN included, out of their matrix multiplications.
        valid_rows = tl.load(
            valid_sizes + row_offsets // tile_rows,
            mask=input_offsets < input_rows,
            other=0,
        )
        key_mask = row_mask & (
            (row_offsets % tile_rows)[:, None] < valid_rows[:, None]
        )

        if row_major:
            destination = (
                row_offsets[:, None] * heads * width
                + head * width
                + columns[None, :]
            )
        else:
            destination = (
                head * rows * width
                + row_offsets[:, None] * width
                + columns[None, :]
            )

        query_values = tl.load(
            query
            + input_offsets[:, None] * query_stride_row
            + head * query_stride_head
            + columns[None, :],
            mask=row_mask,
            other=0.0,
        )
        key_values = tl.load(
            key
            + input_offsets[:, None] * key_stride_row
            + head * key_stride_head
            + columns[None, :],
            mask=key_mask,
            other=0.0,
        )
        value_values = tl.load(
            value
            + input_offsets[:, None] * value_stride_row
            + head * value_stride_head
            + columns[None, :],
            mask=key_mask,
            other=0.0,
        )

        # Queries are reordered into per-segment intervals that group every
        # owner's rows of one segment together (see the module docstring);
        # K/V keep packed row order. ``count`` is the segment's row count,
        # shorter for an owner's last segment when ``chunk_rows`` does not
        # divide ``owner_rows``.
        component_size = heads * rows * width
        owner_rows = rows // owners
        owner = row_offsets // owner_rows
        local_row = row_offsets % owner_rows
        segment = local_row // chunk_rows
        count = tl.minimum(chunk_rows, owner_rows - segment * chunk_rows)
        interval_offset = segment * chunk_rows * owners * heads * width
        interval_row = owner * count + local_row % chunk_rows

        if row_major:
            query_destination = (
                interval_offset + interval_row * heads * width + head * width
            )
        else:
            query_destination = (
                interval_offset
                + head * owners * count * width
                + interval_row * width
            )

        tl.store(
            packed + query_destination[:, None] + columns,
            query_values,
            mask=row_mask,
        )
        tl.store(
            packed + component_size + destination, key_values, mask=row_mask
        )
        tl.store(
            packed + 2 * component_size + destination,
            value_values,
            mask=row_mask,
        )

    @triton.jit(do_not_specialize=["row_start"])
    def _prepare_masked_qkv_kernel(
        query,
        key,
        value,
        gate,
        query_weight,
        key_weight,
        cosine,
        sine,
        valid_sizes,
        packed,
        packed_gate,
        pooled_query,
        pooled_key,
        pooled_value,
        query_stride_row: tl.constexpr,
        query_stride_head: tl.constexpr,
        key_stride_row: tl.constexpr,
        key_stride_head: tl.constexpr,
        value_stride_row: tl.constexpr,
        value_stride_head: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_head: tl.constexpr,
        rotary_stride_row: tl.constexpr,
        pooled_stride_tile: tl.constexpr,
        pooled_stride_head: tl.constexpr,
        rows: tl.constexpr,
        row_start,
        heads: tl.constexpr,
        width: tl.constexpr,
        half_rotary: tl.constexpr,
        eps: tl.constexpr,
        tile_rows: tl.constexpr,
        slab_rows: tl.constexpr,
        owners: tl.constexpr,
        chunk_rows: tl.constexpr,
    ):
        """Normalize, rotate, pool and pack one 64-row tile of one head.

        Grid: one program per (interval tile, head). The first packed tile
        comes from ``row_start // tile_rows``; adding the interval tile
        indexes ``valid_sizes`` and the pooled outputs.

        Q and K are RMS-normalized in fp32 over the full head and their even
        rotary prefix is rotated split-half with compact factors, the formula
        ``uniserve_kernels.rope.qk_norm_rope`` evaluates for one full-head
        domain over one partially rotated axis. The squared-row sum here adds
        two half-width partial sums, a different fp32 order, so a rounded
        element can differ from that kernel's by one ulp. Each head row is
        read once as two half-width column sets: ``x`` holds the first rotary
        half followed by the first half of the unrotated tail, ``y`` the
        partner column of each ``x`` column, so a lane owns a rotation pair
        and tail columns ride along with unit factors. The packed rows hold
        the rotated values rounded to the source dtype; the pooled means
        average those rounded rows over the tile's valid rows in fp32.
        Queries are packed row-major in owner-interval order, including rows
        past the valid size, and K/V in packed row order with those rows
        zeroed, the row-major layout of the packing kernel; the gate rows are
        copied unmasked in packed row order.
        """
        tile = tl.program_id(0)
        head = tl.program_id(1)
        # Public preparation accepts whole tiles. Keep that alignment while
        # sharing code across offsets instead of compiling each interval.
        row_start = tl.multiple_of(row_start, tile_rows)
        tile_offset = row_start // tile_rows
        half_width: tl.constexpr = width // 2
        rotary_dim: tl.constexpr = 2 * half_rotary
        tail_half: tl.constexpr = half_width - half_rotary
        columns = tl.arange(0, half_width)
        full_columns = tl.arange(0, width)
        rotated = columns < half_rotary
        x_columns = tl.where(
            rotated, columns, rotary_dim + (columns - half_rotary)
        )
        y_columns = x_columns + tl.where(rotated, half_rotary, tail_half)
        valid_rows = tl.load(valid_sizes + tile_offset + tile)

        # Learned normalization weights, gathered per column set.
        query_x_weight = tl.load(query_weight + x_columns)[None, :].to(
            tl.float32
        )
        query_y_weight = tl.load(query_weight + y_columns)[None, :].to(
            tl.float32
        )
        key_x_weight = tl.load(key_weight + x_columns)[None, :].to(tl.float32)
        key_y_weight = tl.load(key_weight + y_columns)[None, :].to(tl.float32)

        component_size = heads * rows * width
        owner_rows = rows // owners
        query_x_sum = tl.zeros((half_width,), dtype=tl.float32)
        query_y_sum = tl.zeros((half_width,), dtype=tl.float32)
        key_x_sum = tl.zeros((half_width,), dtype=tl.float32)
        key_y_sum = tl.zeros((half_width,), dtype=tl.float32)
        value_sum = tl.zeros((width,), dtype=tl.float32)

        for slab in tl.static_range(tile_rows // slab_rows):
            local_rows = slab * slab_rows + tl.arange(0, slab_rows)
            input_offsets = (tile * tile_rows + local_rows).to(tl.int64)
            row_offsets = row_start + input_offsets
            valid_mask = (local_rows < valid_rows)[:, None]

            query_base = (
                input_offsets[:, None] * query_stride_row
                + head * query_stride_head
            )
            key_base = (
                input_offsets[:, None] * key_stride_row + head * key_stride_head
            )
            query_x = tl.load(query + query_base + x_columns[None, :]).to(
                tl.float32
            )
            query_y = tl.load(query + query_base + y_columns[None, :]).to(
                tl.float32
            )
            key_x = tl.load(key + key_base + x_columns[None, :]).to(tl.float32)
            key_y = tl.load(key + key_base + y_columns[None, :]).to(tl.float32)

            # Normalization spans the complete head even though only the
            # rotary prefix consumes sine and cosine factors.
            query_rstd = tl.rsqrt(
                (
                    tl.sum(query_x * query_x, axis=1)
                    + tl.sum(query_y * query_y, axis=1)
                )
                / width
                + eps
            )
            key_rstd = tl.rsqrt(
                (tl.sum(key_x * key_x, axis=1) + tl.sum(key_y * key_y, axis=1))
                / width
                + eps
            )
            query_x = query_x * query_rstd[:, None] * query_x_weight
            query_y = query_y * query_rstd[:, None] * query_y_weight
            key_x = key_x * key_rstd[:, None] * key_x_weight
            key_y = key_y * key_rstd[:, None] * key_y_weight

            # Tail columns load unit factors and pass through the rotation.
            # The always-true row term only broadcasts the column mask to the
            # slab shape.
            factor_mask = rotated[None, :] & (local_rows[:, None] >= 0)
            factor_offsets = (
                input_offsets[:, None] * rotary_stride_row + columns[None, :]
            )
            cosine_values = tl.load(
                cosine + factor_offsets, mask=factor_mask, other=1.0
            ).to(tl.float32)
            sine_values = tl.load(
                sine + factor_offsets, mask=factor_mask, other=0.0
            ).to(tl.float32)
            query_x_out = (query_x * cosine_values - query_y * sine_values).to(
                query.dtype.element_ty
            )
            query_y_out = (query_y * cosine_values + query_x * sine_values).to(
                query.dtype.element_ty
            )
            key_x_out = tl.where(
                valid_mask, key_x * cosine_values - key_y * sine_values, 0.0
            ).to(key.dtype.element_ty)
            key_y_out = tl.where(
                valid_mask, key_y * cosine_values + key_x * sine_values, 0.0
            ).to(key.dtype.element_ty)
            value_values = tl.load(
                value
                + input_offsets[:, None] * value_stride_row
                + head * value_stride_head
                + full_columns[None, :],
                mask=valid_mask,
                other=0.0,
            )

            # Same owner-interval query order as the packing kernel; K/V keep
            # packed row order.
            owner = row_offsets // owner_rows
            local_row = row_offsets % owner_rows
            segment = local_row // chunk_rows
            count = tl.minimum(chunk_rows, owner_rows - segment * chunk_rows)
            interval_row = owner * count + local_row % chunk_rows
            query_destination = (
                packed
                + (
                    segment * chunk_rows * owners * heads * width
                    + interval_row * heads * width
                    + head * width
                )[:, None]
            )
            destination = (row_offsets * heads * width + head * width)[:, None]
            key_destination = packed + component_size + destination
            tl.store(query_destination + x_columns[None, :], query_x_out)
            tl.store(query_destination + y_columns[None, :], query_y_out)
            tl.store(key_destination + x_columns[None, :], key_x_out)
            tl.store(key_destination + y_columns[None, :], key_y_out)
            tl.store(
                packed
                + 2 * component_size
                + destination
                + full_columns[None, :],
                value_values,
            )

            gate_values = tl.load(
                gate
                + input_offsets[:, None] * gate_stride_row
                + head * gate_stride_head
                + full_columns[None, :]
            )
            tl.store(
                packed_gate + destination + full_columns[None, :], gate_values
            )

            # K and V rows past the valid size are already zero, so only the
            # query sums need the validity mask.
            query_x_sum += tl.sum(
                tl.where(valid_mask, query_x_out.to(tl.float32), 0.0), axis=0
            )
            query_y_sum += tl.sum(
                tl.where(valid_mask, query_y_out.to(tl.float32), 0.0), axis=0
            )
            key_x_sum += tl.sum(key_x_out.to(tl.float32), axis=0)
            key_y_sum += tl.sum(key_y_out.to(tl.float32), axis=0)
            value_sum += tl.sum(value_values.to(tl.float32), axis=0)

        # Pooled means, with the clamped divisor keeping empty tiles finite.
        divisor = tl.maximum(valid_rows, 1)
        pooled = (
            tile_offset + tile
        ) * pooled_stride_tile + head * pooled_stride_head
        tl.store(pooled_query + pooled + x_columns, query_x_sum / divisor)
        tl.store(pooled_query + pooled + y_columns, query_y_sum / divisor)
        tl.store(pooled_key + pooled + x_columns, key_x_sum / divisor)
        tl.store(pooled_key + pooled + y_columns, key_y_sum / divisor)
        tl.store(pooled_value + pooled + full_columns, value_sum / divisor)

    @triton.jit
    def _compose_rows_kernel(
        attended,
        gate,
        compressed,
        output,
        attended_stride_head: tl.constexpr,
        attended_stride_row: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_head: tl.constexpr,
        rows: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Compose one block of rows for one head into a single output.

        ``attended`` is the provider output addressed as ``[1, heads, rows,
        width]`` through its head and row strides, ``gate`` is ``[rows, heads,
        width]`` and ``compressed`` is contiguous ``[heads, rows / tile_rows,
        width]``. ``output`` must be contiguous ``[rows, heads, width]`` with
        ``heads`` equal to the grid's second dimension, which supplies its
        row stride.
        """
        row_offsets = (
            tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        ).to(tl.int64)
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        mask = row_offsets[:, None] < rows
        attended_values = tl.load(
            attended
            + head * attended_stride_head
            + row_offsets[:, None] * attended_stride_row
            + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.load(
            gate
            + row_offsets[:, None] * gate_stride_row
            + head * gate_stride_head
            + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        compressed_values = tl.load(
            compressed
            + head * (rows // tile_rows) * width
            + (row_offsets[:, None] // tile_rows) * width
            + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        values = attended_values + gate_values * compressed_values
        tl.store(
            output
            + row_offsets[:, None] * tl.num_programs(1) * width
            + head * width
            + columns,
            values,
            mask=mask,
        )

    @triton.jit
    def _compose_shards_kernel(
        attended,
        gate,
        compressed,
        outputs,
        attended_stride_head: tl.constexpr,
        attended_stride_row: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_head: tl.constexpr,
        rows: tl.constexpr,
        local_rows: tl.constexpr,
        local_heads: tl.constexpr,
        global_heads: tl.constexpr,
        source_rank: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
        owner_rows: tl.constexpr,
        start_row: tl.constexpr,
        global_rows: tl.constexpr,
    ):
        """Compose attended rows and route them into row-owner shards.

        ``attended`` holds ``rows`` rows as ``len(outputs)`` consecutive
        segments of ``local_rows``, one per destination; segment ``d`` row
        ``i`` reads global row ``d * owner_rows + start_row + i`` of ``gate``
        (``[global_rows, heads, width]``) and that row's tile of the
        contiguous ``compressed`` (``[heads, global_rows / tile_rows,
        width]``). Each output is
        contiguous ``[local_rows, global_heads, width]``, and this call's
        ``local_heads`` land at head offset ``source_rank * local_heads``.
        """
        row_offsets = (
            tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        ).to(tl.int64)

        # row_offsets enumerate (destination shard, local row) pairs; gate and
        # compressed rows are indexed by the owning shard's global row.
        local_row_offsets = row_offsets % local_rows
        global_row_offsets = (
            row_offsets // local_rows * owner_rows
            + start_row
            + local_row_offsets
        )

        head = tl.program_id(1)
        columns = tl.arange(0, width)
        mask = row_offsets[:, None] < rows
        attended_values = tl.load(
            attended
            + head * attended_stride_head
            + row_offsets[:, None] * attended_stride_row
            + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.load(
            gate
            + global_row_offsets[:, None] * gate_stride_row
            + head * gate_stride_head
            + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        compressed_values = tl.load(
            compressed
            + head * (global_rows // tile_rows) * width
            + (global_row_offsets[:, None] // tile_rows) * width
            + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        values = attended_values + gate_values * compressed_values

        # Local heads land at their global head offset inside each row-owner
        # shard; only the shard matching the row's destination rank is written.
        destination = (
            local_row_offsets[:, None] * global_heads * width
            + (source_rank * local_heads + head) * width
            + columns
        )
        destination_rank = row_offsets[:, None] // local_rows
        for destination_index in tl.static_range(len(outputs)):
            tl.store(
                outputs[destination_index] + destination,
                values,
                mask=mask & (destination_rank == destination_index),
            )


def pack_sparse_input_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_sizes: torch.Tensor,
    *,
    owners: int = 1,
    chunk_rows: int | None = None,
    packed: torch.Tensor | None = None,
    row_start: int = 0,
    row_major: bool = False,
) -> torch.Tensor:
    """Publish a Q/K/V row interval into the provider's complete input layout.

    Caller-owned storage supports out-of-order disjoint intervals produced by
    a distributed projection. Every row must be published before attention
    consumes the buffer. ``row_start`` and ``valid_sizes`` address rows and
    tiles of ``packed``'s row axis, which under context parallelism is the
    rank's local query domain rather than the whole sequence. Storage is
    ``[3, rows, heads, width]`` with ``row_major`` and
    ``[3, heads, rows, width]`` otherwise (see the module docstring for which
    providers read each). Queries use owner-interval order within their
    component; K/V retain packed row order.

    Args:
        query: Interval queries, ``[input_rows, heads, width]`` with unit
            column stride; ``key`` and ``value`` share its shape.
        valid_sizes: Valid rows per 64-row tile of the packed row axis. Its
            length is not checked; it must cover every tile the interval
            touches.
        owners: Number of row owners; must divide ``rows``.
        chunk_rows: Owner-local segment length; ``None`` means one segment
            per owner covering all of its ``rows // owners`` rows.
        packed: Destination storage. ``None`` allocates storage whose row
            count equals the interval's, so ``row_start`` must then be 0.
        row_start: Packed row of the interval's first row.

    Returns:
        ``packed``, or the newly allocated storage.

    Raises:
        ValueError: If shapes, dtype, device, contiguity, or the owner,
            segment and interval bounds are invalid for ``packed``.
    """
    assert triton is not None
    input_rows, heads, width = (int(size) for size in query.shape)

    if packed is None:
        shape = (
            (3, input_rows, heads, width)
            if row_major
            else (3, heads, input_rows, width)
        )
        packed = torch.empty(shape, dtype=query.dtype, device=query.device)

    rows = packed.shape[1 if row_major else 2]
    expected_shape = (
        (3, rows, heads, width) if row_major else (3, heads, rows, width)
    )
    if (
        query.shape != key.shape
        or query.shape != value.shape
        or packed.shape != expected_shape
        or packed.dtype != query.dtype
        or packed.device != query.device
        or not packed.is_contiguous()
        or row_start < 0
        or row_start + input_rows > rows
        or owners < 1
        or rows % owners
        or (chunk_rows is not None and chunk_rows < 1)
    ):
        raise ValueError(
            "sparse input rows must fit matching packed owner storage"
        )

    block_rows = 8
    _pack_masked_qkv_kernel[(triton.cdiv(input_rows, block_rows), heads)](
        query,
        key,
        value,
        valid_sizes,
        packed,
        int(query.stride(0)),
        int(query.stride(1)),
        int(key.stride(0)),
        int(key.stride(1)),
        int(value.stride(0)),
        int(value.stride(1)),
        rows,
        input_rows,
        row_start,
        heads,
        width,
        _TILE,
        block_rows,
        owners,
        rows if chunk_rows is None else chunk_rows,
        row_major,
        num_warps=4,
        num_stages=1,
    )
    return packed


def prepare_sparse_input_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    valid_sizes: torch.Tensor,
    *,
    eps: float,
    packed: torch.Tensor,
    packed_gate: torch.Tensor,
    pooled_query: torch.Tensor,
    pooled_key: torch.Tensor,
    pooled_value: torch.Tensor,
    owners: int,
    chunk_rows: int,
    row_start: int,
) -> None:
    """Normalize, rotate, pool and pack a projected Q/K/V/gate row interval.

    One read of the projections yields the packed row-major input layout of
    ``pack_sparse_input_rows`` with Q and K normalized and rotated as
    ``uniserve.nn.functional.qk_norm_rope`` computes them for one normalized
    domain with a partially rotated prefix, the gate rows copied into
    ``packed_gate`` at their packed rows, and the per-tile means
    ``vsa_tiles.pool_qkv_means`` computes from stored projections, written
    at the interval's tiles. ``cosine`` and ``sine`` hold the interval's
    compact half-width factors, one row per interval row; the interval
    starts at ``row_start`` and covers whole tiles. The head width must be a
    power of two. Pooled buffers are fp32 with unit column stride and are
    indexed by packed tile, so they must reach the interval's last tile; all
    three are addressed with ``pooled_query``'s tile and head strides.

    Raises:
        ValueError: If tensor shapes, dtypes or contiguity, the rotary or
            head width, tile alignment, or the owner and pooled extents are
            invalid.
    """
    assert triton is not None
    input_rows, heads, width = (int(size) for size in query.shape)
    rows = int(packed.shape[1])
    rotary_dim = int(cosine.shape[-1]) * 2
    if (
        query.shape != key.shape
        or query.shape != value.shape
        or query.shape != gate.shape
        or packed.shape != (3, rows, heads, width)
        or packed_gate.shape != (rows, heads, width)
        or packed_gate.dtype != gate.dtype
        or not packed_gate.is_contiguous()
        or packed.dtype != query.dtype
        or not packed.is_contiguous()
        or query_weight.shape != (width,)
        or key_weight.shape != (width,)
        or cosine.shape != (input_rows, rotary_dim // 2)
        or sine.shape != cosine.shape
        or not 0 < rotary_dim <= width
        or rotary_dim % 2
        or width & (width - 1)
        or row_start < 0
        or row_start % _TILE
        or input_rows % _TILE
        or row_start + input_rows > rows
        or owners < 1
        or rows % owners
        or chunk_rows < 1
        or pooled_query.shape != pooled_key.shape
        or pooled_key.shape != pooled_value.shape
        or pooled_query.shape[1:] != (heads, width)
        or (row_start + input_rows) // _TILE > pooled_query.shape[0]
        or any(
            tensor.dtype != torch.float32 or tensor.stride(-1) != 1
            for tensor in (pooled_query, pooled_key, pooled_value)
        )
    ):
        raise ValueError(
            "sparse input preparation requires whole tiles in matching packed "
            "and pooled storage"
        )

    # One program prepares one 64-row tile of one head in 32-row slabs with
    # two warps, the shape that measured fastest for 128-wide heads.
    half_rotary = rotary_dim // 2
    _prepare_masked_qkv_kernel[(input_rows // _TILE, heads)](
        query,
        key,
        value,
        gate,
        query_weight,
        key_weight,
        cosine,
        sine,
        valid_sizes,
        packed,
        packed_gate,
        pooled_query,
        pooled_key,
        pooled_value,
        int(query.stride(0)),
        int(query.stride(1)),
        int(key.stride(0)),
        int(key.stride(1)),
        int(value.stride(0)),
        int(value.stride(1)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        int(cosine.stride(0)),
        int(pooled_query.stride(0)),
        int(pooled_query.stride(1)),
        rows,
        row_start,
        heads,
        width,
        half_rotary,
        float(eps),
        _TILE,
        32,
        owners,
        chunk_rows,
        num_warps=2,
        num_stages=1,
    )


def compose_attention(
    attended: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    outputs: list[torch.Tensor],
    source_rank: int,
    *,
    owner_rows: int | None = None,
    start_row: int = 0,
) -> None:
    """Add gated per-tile compression to attention output and route its rows.

    Writes ``attended + gate * compressed[tile]`` per row, head and feature,
    computed in FP32 and stored in the destination dtype. Numerical providers
    own key validity and softmax normalization; this epilogue only reads
    their output.

    Args:
        attended: Provider output addressed as ``[1, heads, rows, width]``
            (typically a transposed view of row-major storage).
        gate: Row-major ``[gate_rows, heads, width]`` gate over the packed
            row axis.
        compressed: Contiguous ``[heads, gate_rows / 64, width]`` compressed
            tile values.
        outputs: Destinations, which must be contiguous; their shapes are not
            checked here. One output without ``owner_rows`` receives all
            ``rows`` rows as ``[rows, heads, width]``; ``attended`` row ``i``
            then reads gate row ``i``, and ``compressed`` is addressed as
            ``[heads, rows / 64, width]``. Otherwise each output receives
            ``rows / len(outputs)`` consecutive ``attended`` rows.
        source_rank: On the routed path, the head block of this call inside
            each destination's head axis: heads land at
            ``source_rank * heads``. The single-destination path ignores it.
        owner_rows: Gate-row stride between consecutive destinations;
            ``None`` means ``rows / len(outputs)``.
        start_row: Gate-row offset of each destination's first row, which
            for destination ``d`` is gate row ``d * owner_rows + start_row``.
            The single-destination path ignores it.
    """
    assert triton is not None
    heads, rows, width = (int(size) for size in attended.shape[1:])
    block_rows = 8
    grid = (triton.cdiv(rows, block_rows), heads)

    # A single unrouted destination needs no gate-row remapping.
    if len(outputs) == 1 and owner_rows is None:
        _compose_rows_kernel[grid](
            attended,
            gate,
            compressed,
            outputs[0],
            int(attended.stride(1)),
            int(attended.stride(2)),
            int(gate.stride(0)),
            int(gate.stride(1)),
            rows,
            width,
            _TILE,
            block_rows,
            num_warps=4,
            num_stages=1,
        )
        return

    _compose_shards_kernel[grid](
        attended,
        gate,
        compressed,
        tuple(outputs),
        int(attended.stride(1)),
        int(attended.stride(2)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        rows,
        rows // len(outputs),
        heads,
        int(outputs[0].shape[1]),
        source_rank,
        width,
        _TILE,
        block_rows,
        rows // len(outputs) if owner_rows is None else owner_rows,
        start_row,
        int(gate.shape[0]),
        num_warps=4,
        num_stages=1,
    )
