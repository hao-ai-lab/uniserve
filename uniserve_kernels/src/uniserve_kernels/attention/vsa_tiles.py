"""Sparse-video QKV packing, tile selection, and output composition.

The kernels transform row-major Q/K/V projections into head-major sparse
attention inputs, pool fixed-size token tiles, select score-threshold
candidates, and combine attended values with learned compression tokens.
Composition can write one local result or route rank-local heads directly
into exchange shards.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

if triton is not None:
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def _pool_qkv_means_kernel(
        query,
        key,
        value,
        valid_sizes,
        pooled_query,
        pooled_key,
        pooled_value,
        query_stride_row: tl.constexpr,
        query_stride_head: tl.constexpr,
        key_stride_row: tl.constexpr,
        key_stride_head: tl.constexpr,
        value_stride_row: tl.constexpr,
        value_stride_head: tl.constexpr,
        output_stride_tile: tl.constexpr,
        output_stride_head: tl.constexpr,
        heads: tl.constexpr,
        query_tiles: tl.constexpr,
        query_tile_offset: tl.constexpr,
        key_tile_offset: tl.constexpr,
        tile_rows: tl.constexpr,
        width: tl.constexpr,
    ):
        """Average valid Q/K/V rows for one tile and attention head."""
        # Sources are [rows, heads, width]; pooled outputs are
        # [tiles, heads, width]. Rows beyond ``valid_sizes`` are masked out,
        # and the clamped divisor keeps empty tiles finite.
        tile = tl.program_id(0)
        head = tl.program_id(1)
        row_offsets = (tile * tile_rows + tl.arange(0, tile_rows)).to(tl.int64)
        columns = tl.arange(0, width)
        valid_rows = tl.load(valid_sizes + key_tile_offset + tile)
        mask = tl.arange(0, tile_rows)[:, None] < valid_rows
        output_offsets = (
            tile * output_stride_tile + head * output_stride_head + columns
        )

        # Query and key owners may cover different global tile intervals.
        if tile < query_tiles:
            query_valid_rows = tl.load(valid_sizes + query_tile_offset + tile)
            query_values = tl.load(
                query
                + row_offsets[:, None] * query_stride_row
                + head * query_stride_head
                + columns[None, :],
                mask=tl.arange(0, tile_rows)[:, None] < query_valid_rows,
                other=0.0,
            ).to(tl.float32)

            query_mean = tl.sum(query_values, axis=0) / tl.maximum(
                query_valid_rows, 1
            )
            tl.store(pooled_query + output_offsets, query_mean)

        key_values = tl.load(
            key
            + row_offsets[:, None] * key_stride_row
            + head * key_stride_head
            + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        key_mean = tl.sum(key_values, axis=0) / tl.maximum(valid_rows, 1)
        tl.store(pooled_key + output_offsets, key_mean)

        value_values = tl.load(
            value
            + row_offsets[:, None] * value_stride_row
            + head * value_stride_head
            + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        value_mean = tl.sum(value_values, axis=0) / tl.maximum(valid_rows, 1)
        tl.store(pooled_value + output_offsets, value_mean)

    @triton.jit
    def _select_block_map_kernel(
        scores,
        prefix_key_indices,
        dense_key_indices,
        block_indices,
        block_counts,
        score_stride_head: tl.constexpr,
        score_stride_row: tl.constexpr,
        index_stride_head: tl.constexpr,
        index_stride_row: tl.constexpr,
        count_stride_head: tl.constexpr,
        tiles: tl.constexpr,
        local_prefix: tl.constexpr,
        local_video: tl.constexpr,
        prefix_tiles: tl.constexpr,
        valid_tiles: tl.constexpr,
        columns: tl.constexpr,
        selected: tl.constexpr,
        block: tl.constexpr,
        dense_block: tl.constexpr,
        iterations: tl.constexpr,
    ):
        """Write one query tile's key-tile list and count for one head.

        A prefix query tile attends densely to every valid key tile. A video
        query tile attends to the dense prefix, then to the ``selected`` video
        tiles whose pooled scores clear a threshold found by interpolation
        search; prefix ranks give threshold ties a deterministic order. A
        padding tile attends to one tile so its count stays positive. Entries
        past a row's count are left as they were; no consumer reads them.
        """
        row = tl.program_id(0)
        head = row // tiles
        tile = row % tiles
        index_row = (
            block_indices + head * index_stride_head + tile * index_stride_row
        )
        count_row = block_counts + head * count_stride_head + tile

        if tile < local_prefix:
            offsets = tl.arange(0, dense_block)
            dense = tl.load(
                dense_key_indices + offsets, mask=offsets < valid_tiles, other=0
            )
            tl.store(index_row + offsets, dense, mask=offsets < valid_tiles)
            tl.store(count_row, valid_tiles)
        elif tile < local_video:
            prefix_offsets = tl.arange(0, dense_block)
            prefix = tl.load(
                prefix_key_indices + prefix_offsets,
                mask=prefix_offsets < prefix_tiles,
                other=0,
            )
            tl.store(
                index_row + prefix_offsets,
                prefix,
                mask=prefix_offsets < prefix_tiles,
            )

            offsets = tl.arange(0, block)
            valid = offsets < columns
            values = tl.load(
                scores
                + head * score_stride_head
                + (tile - local_prefix) * score_stride_row
                + offsets,
                mask=valid,
                other=-float("inf"),
            ).to(tl.float32)

            # Maintain score bounds and the number of candidates at each
            # bound. Interpolation converges toward a threshold with at least
            # ``selected`` values while avoiding a full per-row sort.
            lower = tl.min(tl.where(valid, values, float("inf")))
            upper = tl.max(tl.where(valid, values, -float("inf"))) + 1.0
            lower_count = tl.sum(valid.to(tl.int32), axis=0).to(tl.float32)
            upper_count = 0.0
            for _ in tl.static_range(iterations):
                # Clamping the step keeps each iteration inside the bracket.
                denominator = lower_count - upper_count
                fraction = (lower_count - selected) / tl.where(
                    denominator > 0.5,
                    denominator,
                    1.0,
                )
                fraction = tl.minimum(tl.maximum(fraction, 0.05), 0.95)

                threshold = lower + (upper - lower) * fraction
                count = tl.sum(
                    ((values >= threshold) & valid).to(tl.int32),
                    axis=0,
                ).to(tl.float32)

                enough = count >= selected
                lower = tl.where(enough, threshold, lower)
                lower_count = tl.where(enough, count, lower_count)
                upper = tl.where(enough, upper, threshold)
                upper_count = tl.where(enough, upper_count, count)

            # Selected score columns are video key tiles offset by the prefix,
            # stored after the prefix entries and capped at the selection.
            chosen = (values >= lower) & valid
            positions = tl.cumsum(chosen.to(tl.int32), axis=0) - 1
            tl.store(
                index_row + prefix_tiles + positions,
                (offsets + prefix_tiles).to(tl.int32),
                mask=chosen & (positions < selected),
            )
            tl.store(count_row, prefix_tiles + selected)
        else:
            tl.store(index_row, 0)
            tl.store(count_row, 1)

    @triton.jit
    def _tile_softmax_kernel(
        scores,
        valid_sizes,
        score_stride_head: tl.constexpr,
        score_stride_row: tl.constexpr,
        columns: tl.constexpr,
        block: tl.constexpr,
    ):
        """Softmax one query tile's scores over the valid key tiles in place.

        Key tiles without valid rows are excluded as if their score were
        negative infinity; a row with no valid key tile becomes zeros.
        """
        row = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, block)
        mask = offsets < columns
        base = scores + head * score_stride_head + row * score_stride_row
        values = tl.load(base + offsets, mask=mask, other=float("-inf"))
        valid = tl.load(valid_sizes + offsets, mask=mask, other=0)
        values = tl.where(valid > 0, values, float("-inf"))

        peak = tl.max(values, axis=0)
        weights = tl.where(
            values > float("-inf"), libdevice.exp(values - peak), 0.0
        )
        total = tl.sum(weights, axis=0)
        tl.store(base + offsets, weights / tl.maximum(total, 1e-38), mask=mask)

    @triton.jit
    def _pack_qkv_kernel(
        query,
        key,
        value,
        packed,
        query_stride_0: tl.constexpr,
        query_stride_1: tl.constexpr,
        query_stride_2: tl.constexpr,
        key_stride_0: tl.constexpr,
        key_stride_1: tl.constexpr,
        key_stride_2: tl.constexpr,
        value_stride_0: tl.constexpr,
        value_stride_1: tl.constexpr,
        value_stride_2: tl.constexpr,
        rows: tl.constexpr,
        heads: tl.constexpr,
        width: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Pack row-major Q/K/V into contiguous storage.

        Pack row-major Q/K/V into contiguous ``[3, heads, rows, width]``
        storage.
        """
        row_offsets = (
            tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        ).to(tl.int64)
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        mask = row_offsets[:, None] < rows

        destination = (
            head * rows * width
            + row_offsets[:, None] * width
            + columns[None, :]
        )

        # Each component uses its own source strides and a component-sized
        # offset into the shared destination allocation.
        query_values = tl.load(
            query
            + row_offsets[:, None] * query_stride_0
            + head * query_stride_1
            + columns[None, :] * query_stride_2,
            mask=mask,
            other=0.0,
        )
        tl.store(packed + destination, query_values, mask=mask)

        key_values = tl.load(
            key
            + row_offsets[:, None] * key_stride_0
            + head * key_stride_1
            + columns[None, :] * key_stride_2,
            mask=mask,
            other=0.0,
        )
        tl.store(
            packed + heads * rows * width + destination, key_values, mask=mask
        )

        value_values = tl.load(
            value
            + row_offsets[:, None] * value_stride_0
            + head * value_stride_1
            + columns[None, :] * value_stride_2,
            mask=mask,
            other=0.0,
        )
        tl.store(
            packed + 2 * heads * rows * width + destination,
            value_values,
            mask=mask,
        )

    @triton.jit
    def _unpack_add_compression_kernel(
        attended,
        gate,
        compressed,
        output,
        attended_stride_head: tl.constexpr,
        attended_stride_row: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_head: tl.constexpr,
        compressed_stride_head: tl.constexpr,
        compressed_stride_tile: tl.constexpr,
        output_stride_row: tl.constexpr,
        output_stride_head: tl.constexpr,
        rows: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Compose head-major attention output with compression.

        Compose head-major attention output with per-tile compression
        values.
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
            + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        compressed_values = tl.load(
            compressed
            + head * compressed_stride_head
            + (row_offsets[:, None] // tile_rows) * compressed_stride_tile
            + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # Compression is shared by all rows in a fixed 64-row tile and gated
        # independently for every row, head, and feature.
        tl.store(
            output
            + row_offsets[:, None] * output_stride_row
            + head * output_stride_head
            + columns[None, :],
            attended_values + gate_values * compressed_values,
            mask=mask,
        )

    @triton.jit
    def _compose_to_head_shards_kernel(
        attended,
        gate,
        compressed,
        outputs,
        attended_stride_head: tl.constexpr,
        attended_stride_row: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_head: tl.constexpr,
        compressed_stride_head: tl.constexpr,
        compressed_stride_tile: tl.constexpr,
        rows: tl.constexpr,
        local_rows: tl.constexpr,
        local_heads: tl.constexpr,
        global_heads: tl.constexpr,
        source_rank: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Compose values and route them into destination-rank head shards."""
        row_offsets = (
            tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        ).to(tl.int64)
        local_row_offsets = row_offsets % local_rows
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        valid = row_offsets[:, None] < rows

        attended_values = tl.load(
            attended
            + head * attended_stride_head
            + row_offsets[:, None] * attended_stride_row
            + columns[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.load(
            gate
            + row_offsets[:, None] * gate_stride_row
            + head * gate_stride_head
            + columns[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        compressed_values = tl.load(
            compressed
            + head * compressed_stride_head
            + (row_offsets[:, None] // tile_rows) * compressed_stride_tile
            + columns[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)

        values = attended_values + gate_values * compressed_values

        # Consecutive ``local_rows`` source ranges target distinct destination
        # ranks. ``source_rank`` selects this rank's global head interval inside
        # every destination tensor.
        destination = (
            local_row_offsets[:, None] * global_heads * width
            + (source_rank * local_heads + head) * width
            + columns[None, :]
        )
        destination_rank = row_offsets[:, None] // local_rows
        for destination_index in tl.static_range(len(outputs)):
            tl.store(
                outputs[destination_index] + destination,
                values,
                mask=valid & (destination_rank == destination_index),
            )


def _pack_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Allocate and return packed ``[3, heads, rows, width]`` Q/K/V storage."""
    if not launchable(query.device):
        raise RuntimeError("sparse QKV packing requires Triton")

    rows, heads, width = (int(size) for size in query.shape)
    packed = torch.empty(
        (3, heads, rows, width),
        dtype=query.dtype,
        device=query.device,
    )

    # The output orders components before heads and rows, matching the sparse
    # attention kernel while avoiding an intermediate permuted tensor.
    block_rows = 8

    _pack_qkv_kernel[(triton.cdiv(rows, block_rows), heads)](
        query,
        key,
        value,
        packed,
        int(query.stride(0)),
        int(query.stride(1)),
        int(query.stride(2)),
        int(key.stride(0)),
        int(key.stride(1)),
        int(key.stride(2)),
        int(value.stride(0)),
        int(value.stride(1)),
        int(value.stride(2)),
        rows,
        heads,
        width,
        block_rows,
        num_warps=4,
        num_stages=1,
    )
    return packed


def _pool_qkv_means(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_sizes: torch.Tensor,
    pooled_query: torch.Tensor,
    pooled_key: torch.Tensor,
    pooled_value: torch.Tensor,
    query_tile_offset: int = 0,
    key_tile_offset: int = 0,
) -> None:
    """Average each Q/K/V head across fixed 64-row tiles into output buffers."""
    if not launchable(query.device):
        raise RuntimeError("sparse tile pooling requires Triton")

    rows, heads, width = (int(size) for size in query.shape)
    tiles = key.shape[0] // 64
    query_tiles = rows // 64

    if (
        rows % 64
        or key.shape[0] % 64
        or query_tiles > tiles
        or key.shape != value.shape
        or key.shape[1:] != query.shape[1:]
        or query_tile_offset < 0
        or key_tile_offset < 0
        or query_tile_offset + query_tiles > valid_sizes.numel()
        or key_tile_offset + tiles > valid_sizes.numel()
    ):
        raise ValueError(
            "sparse pooling metadata does not cover Q/K tile intervals"
        )

    # One program owns a tile/head pair and reduces only the live prefix given
    # by ``valid_sizes``; the final tile may therefore be partially occupied.
    _pool_qkv_means_kernel[(tiles, heads)](
        query,
        key,
        value,
        valid_sizes,
        pooled_query,
        pooled_key,
        pooled_value,
        int(query.stride(0)),
        int(query.stride(1)),
        int(key.stride(0)),
        int(key.stride(1)),
        int(value.stride(0)),
        int(value.stride(1)),
        int(pooled_query.stride(0)),
        int(pooled_query.stride(1)),
        heads,
        query_tiles,
        query_tile_offset,
        key_tile_offset,
        64,
        width,
        num_warps=1,
        num_stages=1,
    )


@torch.library.custom_op(
    "uniserve::video_sparse_pool_qkv_means",
    mutates_args=("pooled_query", "pooled_key", "pooled_value"),
)
def _pool_qkv_means_custom(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_sizes: torch.Tensor,
    pooled_query: torch.Tensor,
    pooled_key: torch.Tensor,
    pooled_value: torch.Tensor,
    query_tile_offset: int = 0,
    key_tile_offset: int = 0,
) -> None:
    """Expose in-place pooled outputs through the registered custom operator."""
    _pool_qkv_means(
        query,
        key,
        value,
        valid_sizes,
        pooled_query,
        pooled_key,
        pooled_value,
        query_tile_offset,
        key_tile_offset,
    )


@_pool_qkv_means_custom.register_fake
def _pool_qkv_means_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_sizes: torch.Tensor,
    pooled_query: torch.Tensor,
    pooled_key: torch.Tensor,
    pooled_value: torch.Tensor,
    query_tile_offset: int = 0,
    key_tile_offset: int = 0,
) -> None:
    """Declare fake-tensor mutation for the pooled custom operator."""
    del query, key, value, valid_sizes, pooled_query, pooled_key, pooled_value
    del query_tile_offset, key_tile_offset


def _write_block_map(
    scores: torch.Tensor,
    prefix_key_indices: torch.Tensor,
    dense_key_indices: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    local_prefix: int,
    local_video: int,
    prefix_tiles: int,
    valid_tiles: int,
    selected: int,
) -> None:
    """Write every query tile's key-tile list and count in one pass."""
    if not launchable(scores.device):
        raise RuntimeError("sparse block-map selection requires Triton")

    heads, tiles = (int(size) for size in block_counts.shape)
    columns = int(scores.shape[-1])
    if (
        scores.ndim != 3
        or scores.shape[0] != heads
        or scores.shape[1] != local_video - local_prefix
        or block_indices.shape[:2] != (heads, tiles)
        or block_indices.shape[2] < valid_tiles
        or block_indices.shape[2] < prefix_tiles + selected
        or not 0 <= local_prefix <= local_video <= tiles
        or (local_video > local_prefix and columns < selected)
    ):
        raise ValueError("sparse block map does not match its score domain")

    # Each program writes one head/query row: a dense prefix row, a video row
    # searched over a power-of-two score tile, or a padding row.
    _select_block_map_kernel[(heads * tiles,)](
        scores,
        prefix_key_indices,
        dense_key_indices,
        block_indices,
        block_counts,
        int(scores.stride(0)),
        int(scores.stride(1)),
        int(block_indices.stride(0)),
        int(block_indices.stride(1)),
        int(block_counts.stride(0)),
        tiles,
        local_prefix,
        local_video,
        prefix_tiles,
        valid_tiles,
        columns,
        selected,
        triton.next_power_of_2(max(columns, 1)),
        triton.next_power_of_2(max(valid_tiles, 1)),
        32,
        num_warps=2,
        num_stages=1,
    )


@torch.library.custom_op(
    "uniserve::video_sparse_block_map",
    mutates_args=("block_indices", "block_counts"),
)
def _write_block_map_custom(
    scores: torch.Tensor,
    prefix_key_indices: torch.Tensor,
    dense_key_indices: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    local_prefix: int,
    local_video: int,
    prefix_tiles: int,
    valid_tiles: int,
    selected: int,
) -> None:
    """Expose block-map selection as a map-mutating custom operator."""
    _write_block_map(
        scores,
        prefix_key_indices,
        dense_key_indices,
        block_indices,
        block_counts,
        local_prefix,
        local_video,
        prefix_tiles,
        valid_tiles,
        selected,
    )


@_write_block_map_custom.register_fake
def _write_block_map_fake(
    scores: torch.Tensor,
    prefix_key_indices: torch.Tensor,
    dense_key_indices: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    local_prefix: int,
    local_video: int,
    prefix_tiles: int,
    valid_tiles: int,
    selected: int,
) -> None:
    """Declare fake-tensor mutation for the block-map custom operator."""
    del scores, prefix_key_indices, dense_key_indices
    del block_indices, block_counts
    del local_prefix, local_video, prefix_tiles, valid_tiles, selected


def _tile_softmax(scores: torch.Tensor, valid_sizes: torch.Tensor) -> None:
    """Normalize each score row over the key tiles that hold valid rows."""
    assert triton is not None
    heads, rows, columns = (int(size) for size in scores.shape)
    if (
        scores.dtype != torch.float32
        or scores.stride(-1) != 1
        or valid_sizes.dtype != torch.int32
        or valid_sizes.shape != (columns,)
        or not valid_sizes.is_contiguous()
    ):
        raise ValueError(
            "tile softmax requires fp32 score rows and one valid size per "
            "key tile"
        )

    # One program of two warps per head and query tile row, the fastest of
    # the measured warp counts; the row fits one block.
    _tile_softmax_kernel[(rows, heads)](
        scores,
        valid_sizes,
        int(scores.stride(0)),
        int(scores.stride(1)),
        columns,
        triton.next_power_of_2(columns),
        num_warps=2,
        num_stages=1,
    )


@torch.library.custom_op(
    "uniserve::video_sparse_tile_softmax", mutates_args=("scores",)
)
def _tile_softmax_custom(
    scores: torch.Tensor, valid_sizes: torch.Tensor
) -> None:
    """Expose the masked tile softmax as a score-mutating custom operator."""
    _tile_softmax(scores, valid_sizes)


@_tile_softmax_custom.register_fake
def _tile_softmax_fake(scores: torch.Tensor, valid_sizes: torch.Tensor) -> None:
    """Declare fake-tensor mutation for the tile softmax custom operator."""
    del scores, valid_sizes


def _unpack_add_compression(
    attended: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Write gated compression composition in row-major layout.

    Write ``attended + gate * compressed_tile`` in row-major output layout.
    """
    if not launchable(attended.device):
        raise RuntimeError("sparse output composition requires Triton")

    rows, heads, width = (int(size) for size in gate.shape)

    # Attention and compression are head-major; gate and destination are
    # row-major. The kernel composes values while converting between layouts.
    block_rows = 8

    _unpack_add_compression_kernel[(triton.cdiv(rows, block_rows), heads)](
        attended,
        gate,
        compressed,
        output,
        int(attended.stride(1)),
        int(attended.stride(2)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        int(compressed.stride(0)),
        int(compressed.stride(1)),
        int(output.stride(0)),
        int(output.stride(1)),
        rows,
        width,
        64,
        block_rows,
        num_warps=4,
        num_stages=1,
    )


def _compose_to_head_shards(
    attended: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    source_rank: int,
) -> None:
    """Compose values and scatter this rank's heads into exchange shards."""
    if not launchable(attended.device):
        raise RuntimeError("sparse head-shard composition requires Triton")

    rows, local_heads, width = (int(size) for size in gate.shape)
    local_rows = int(outputs[0].shape[0])
    global_heads = int(outputs[0].shape[1])

    # The source row axis concatenates destination-rank segments. Each
    # output receives one segment in the global head range owned by source_rank.
    block_rows = 8

    _compose_to_head_shards_kernel[
        (triton.cdiv(rows, block_rows), local_heads)
    ](
        attended,
        gate,
        compressed,
        outputs,
        int(attended.stride(1)),
        int(attended.stride(2)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        int(compressed.stride(0)),
        int(compressed.stride(1)),
        rows,
        local_rows,
        local_heads,
        global_heads,
        int(source_rank),
        width,
        64,
        block_rows,
        num_warps=4,
        num_stages=1,
    )


def pack_qkv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Return Q/K/V packed into component-head order.

    Return Q/K/V packed from ``[rows, heads, width]`` to component-head
    order.
    """
    return _pack_qkv(query, key, value)


def pool_qkv_means(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_sizes: torch.Tensor,
    pooled_query: torch.Tensor,
    pooled_key: torch.Tensor,
    pooled_value: torch.Tensor,
    query_tile_offset: int = 0,
    key_tile_offset: int = 0,
) -> None:
    """Fill per-tile Q/K/V means using ``valid_sizes`` for partial tiles."""
    _pool_qkv_means_custom(
        query,
        key,
        value,
        valid_sizes,
        pooled_query,
        pooled_key,
        pooled_value,
        query_tile_offset,
        key_tile_offset,
    )


def write_block_map(
    scores: torch.Tensor,
    prefix_key_indices: torch.Tensor,
    dense_key_indices: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    local_prefix: int,
    local_video: int,
    prefix_tiles: int,
    valid_tiles: int,
    selected: int,
) -> None:
    """Fill the block map of every query tile of one call.

    ``scores`` holds the video query rows' scores over the video key tiles,
    ``[heads, local_video - local_prefix, video_tiles]``. Query tiles before
    ``local_prefix`` attend to the ``valid_tiles`` dense key tiles, tiles
    before ``local_video`` to the prefix plus ``selected`` video tiles, and
    later tiles to one tile.
    """
    _write_block_map_custom(
        scores,
        prefix_key_indices,
        dense_key_indices,
        block_indices,
        block_counts,
        int(local_prefix),
        int(local_video),
        int(prefix_tiles),
        int(valid_tiles),
        int(selected),
    )


def tile_softmax(scores: torch.Tensor, valid_sizes: torch.Tensor) -> None:
    """Softmax ``scores`` ``[heads, query_tiles, key_tiles]`` in place.

    Each row is normalized over the key tiles whose ``valid_sizes`` entry is
    positive, the others weigh zero; a row without any valid key tile is
    written as zeros.
    """
    _tile_softmax_custom(scores, valid_sizes)


def unpack_add_compression(
    attended: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Fill row-major output with gated compression.

    Fill row-major output with gated per-tile compression added to
    attention.
    """
    _unpack_add_compression(attended, gate, compressed, output)


def compose_to_head_shards(
    attended: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    source_rank: int,
) -> None:
    """Compose local heads into destination shards.

    Compose local heads into destination shards at ``source_rank``'s head
    range.
    """
    _compose_to_head_shards(
        attended, gate, compressed, outputs, int(source_rank)
    )


__all__ = [
    "compose_to_head_shards",
    "pack_qkv",
    "pool_qkv_means",
    "tile_softmax",
    "unpack_add_compression",
    "write_block_map",
]
