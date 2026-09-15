"""Shared sparse input layouts and row-wise compression epilogues."""

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

    @triton.jit
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
        row_start: tl.constexpr,
        heads: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
        owners: tl.constexpr,
        chunk_rows: tl.constexpr,
        row_major: tl.constexpr,
    ):
        """Pack interval queries and full masked K/V.

        Pack interval queries and full masked K/V in the provider's
        physical layout.
        """
        input_offsets = (
            tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        ).to(tl.int64)
        row_offsets = row_start + input_offsets
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        row_mask = input_offsets[:, None] < input_rows

        # K/V rows past a tile's valid size are masked out, leaving zeros in
        # the packed buffer for the attention provider to ignore.
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
        # Queries are reordered into per-chunk intervals that group every
        # owner's rows together, so each owner receives one contiguous span.
        # K/V keep global row order; only the query component is rearranged.
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
        """Fuse the attention result with trained compression."""
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
        """Compose sparse outputs and route local heads.

        Compose sparse outputs and route local heads into row-owner shards.
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
    consumes the buffer. Validity and destination offsets use global rows.
    Storage is (3, rows, heads, width) for the native provider and
    (3, heads, rows, width) for flattened BSR. Queries use owner-interval
    order within their component; K/V retain global row order.
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
    """Fuse trained compression into masked attention.

    Fuse trained compression into masked attention and route complete head
    rows.

    Numerical providers own key validity and softmax normalization. Composition
    accumulates in FP32 before writing the destination dtype.
    """
    assert triton is not None
    heads, rows, width = (int(size) for size in attended.shape[1:])
    block_rows = 8
    grid = (triton.cdiv(rows, block_rows), heads)

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
