"""Sparse-video QKV packing, tile selection, and output composition.

The kernels transform row-major Q/K/V projections into head-major sparse
attention inputs, pool fixed-size token tiles, select score-threshold candidates,
and combine attended values with learned compression tokens. Composition can
write one local result or route rank-local heads directly into exchange shards.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from ..backends.triton import triton_available
from .core import Dispatcher, Operator

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

if triton is not None:

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

        tile = tl.program_id(0)
        head = tl.program_id(1)
        row_offsets = (tile * tile_rows + tl.arange(0, tile_rows)).to(tl.int64)
        columns = tl.arange(0, width)
        valid_rows = tl.load(valid_sizes + key_tile_offset + tile)
        mask = tl.arange(0, tile_rows)[:, None] < valid_rows
        output_offsets = tile * output_stride_tile + head * output_stride_head + columns

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
            query_mean = tl.sum(query_values, axis=0) / tl.maximum(query_valid_rows, 1)
            tl.store(pooled_query + output_offsets, query_mean)

        key_values = tl.load(
            key + row_offsets[:, None] * key_stride_row + head * key_stride_head + columns[None, :],
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
    def _threshold_topk_kernel(
        scores,
        output,
        score_stride_head: tl.constexpr,
        score_stride_row: tl.constexpr,
        output_stride_head: tl.constexpr,
        output_stride_row: tl.constexpr,
        rows_per_head: tl.constexpr,
        columns: tl.constexpr,
        selected: tl.constexpr,
        block: tl.constexpr,
        iterations: tl.constexpr,
    ):
        """Select up to ``selected`` score columns by iterative threshold search."""

        row = tl.program_id(0)
        head = row // rows_per_head
        query_row = row % rows_per_head
        offsets = tl.arange(0, block)
        valid = offsets < columns
        values = tl.load(
            scores + head * score_stride_head + query_row * score_stride_row + offsets,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)

        # Maintain score bounds and the number of candidates at each bound.
        # Interpolation converges toward a threshold with at least ``selected``
        # values while avoiding a full per-row sort.
        lower = tl.min(tl.where(valid, values, float("inf")))
        upper = tl.max(tl.where(valid, values, -float("inf"))) + 1.0
        lower_count = tl.sum(valid.to(tl.int32), axis=0).to(tl.float32)
        upper_count = 0.0
        for _ in tl.static_range(iterations):
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

        # Prefix ranks give threshold ties a deterministic column order and
        # cap stores at the requested output width.
        chosen = (values >= lower) & valid
        positions = tl.cumsum(chosen.to(tl.int32), axis=0) - 1
        tl.store(
            output + head * output_stride_head + query_row * output_stride_row + positions,
            offsets.to(tl.int32),
            mask=chosen & (positions < selected),
        )

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
        """Pack row-major Q/K/V into contiguous ``[3, heads, rows, width]`` storage."""

        row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        mask = row_offsets[:, None] < rows
        destination = head * rows * width + row_offsets[:, None] * width + columns[None, :]

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
        tl.store(packed + heads * rows * width + destination, key_values, mask=mask)
        value_values = tl.load(
            value
            + row_offsets[:, None] * value_stride_0
            + head * value_stride_1
            + columns[None, :] * value_stride_2,
            mask=mask,
            other=0.0,
        )
        tl.store(packed + 2 * heads * rows * width + destination, value_values, mask=mask)

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
        """Compose head-major attention output with per-tile compression values."""

        row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
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

        row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
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

    if triton is None or not triton_available(query.device):
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

    if triton is None or not triton_available(query.device):
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
        raise ValueError("sparse pooling metadata does not cover Q/K tile intervals")

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
    "uniserve_worker::video_sparse_pool_qkv_means",
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
    """Declare the pooled custom operator's fake-tensor mutation contract."""

    del query, key, value, valid_sizes, pooled_query, pooled_key, pooled_value
    del query_tile_offset, key_tile_offset


def _threshold_topk_indices(scores: torch.Tensor, output: torch.Tensor) -> None:
    """Fill ``output`` with threshold-selected indices for every score row."""

    if triton is None or not triton_available(scores.device):
        raise RuntimeError("sparse threshold selection requires Triton")

    heads, rows, columns = (int(size) for size in scores.shape)
    selected = int(output.shape[-1])

    # Each program searches one head/query row over a power-of-two score tile
    # and writes at most the caller-provided selection width.
    _threshold_topk_kernel[(heads * rows,)](
        scores,
        output,
        int(scores.stride(0)),
        int(scores.stride(1)),
        int(output.stride(0)),
        int(output.stride(1)),
        rows,
        columns,
        selected,
        triton.next_power_of_2(columns),
        32,
        num_warps=2,
        num_stages=1,
    )


@torch.library.custom_op(
    "uniserve_worker::video_sparse_threshold_topk",
    mutates_args=("output",),
)
def _threshold_topk_indices_custom(scores: torch.Tensor, output: torch.Tensor) -> None:
    """Expose threshold selection as an output-mutating custom operator."""

    _threshold_topk_indices(scores, output)


@_threshold_topk_indices_custom.register_fake
def _threshold_topk_indices_fake(scores: torch.Tensor, output: torch.Tensor) -> None:
    """Declare the threshold custom operator's fake-tensor mutation contract."""

    del scores, output


def _unpack_add_compression(
    attended: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Write ``attended + gate * compressed_tile`` in row-major output layout."""

    if triton is None or not triton_available(attended.device):
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

    if triton is None or not triton_available(attended.device):
        raise RuntimeError("sparse head-shard composition requires Triton")

    rows, local_heads, width = (int(size) for size in gate.shape)
    local_rows = int(outputs[0].shape[0])
    global_heads = int(outputs[0].shape[1])

    # The source row axis concatenates destination-rank segments. Each
    # output receives one segment in the global head range owned by source_rank.
    block_rows = 8
    _compose_to_head_shards_kernel[(triton.cdiv(rows, block_rows), local_heads)](
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


@dataclass(frozen=True, slots=True)
class PackQKVReq:
    """Q/K/V projections to pack into newly allocated head-major storage."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class PoolQKVMeansReq:
    """Q/K/V projections, tile occupancy, and caller-owned pooled outputs."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    valid_sizes: torch.Tensor
    pooled_query: torch.Tensor
    pooled_key: torch.Tensor
    pooled_value: torch.Tensor
    query_tile_offset: int = 0
    key_tile_offset: int = 0


@dataclass(frozen=True, slots=True)
class ThresholdTopKReq:
    """Tile scores and caller-owned selected-index storage."""

    scores: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True, slots=True)
class AddCompressionReq:
    """Attention, gate, compression, and output tensors for gated composition."""

    attended: torch.Tensor
    gate: torch.Tensor
    compressed: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True, slots=True)
class ComposeHeadShardsReq:
    """Rank-local attention values and destination shards for global-head composition."""

    attended: torch.Tensor
    gate: torch.Tensor
    compressed: torch.Tensor
    outputs: tuple[torch.Tensor, ...]
    source_rank: int


class _TritonPackQKV(Operator):
    """Packs query, key, and value tensors through the Triton sparse-video kernel."""

    def __init__(self) -> None:
        """Register the Triton QKV-packing provider identity."""

        super().__init__("triton", "video_sparse_pack_qkv")

    def can_run(self, req: PackQKVReq) -> bool:
        """Return whether Triton can pack tensors on the query device."""

        return triton is not None and triton_available(req.query.device)

    def run(self, req: PackQKVReq) -> torch.Tensor:
        """Allocate and return head-major packed Q/K/V storage."""

        return _pack_qkv(req.query, req.key, req.value)


class _TritonPoolQKVMeans(Operator):
    """Computes per-tile Q/K/V means through the Triton sparse-video kernel."""

    def __init__(self) -> None:
        """Register the Triton tile-pooling provider identity."""

        super().__init__("triton", "video_sparse_pool_qkv_means")

    def can_run(self, req: PoolQKVMeansReq) -> bool:
        """Return whether Triton can pool tensors on the query device."""

        return triton is not None and triton_available(req.query.device)

    def run(self, req: PoolQKVMeansReq) -> None:
        """Fill caller-owned pooled Q/K/V buffers through the custom operator."""

        _pool_qkv_means_custom(
            req.query,
            req.key,
            req.value,
            req.valid_sizes,
            req.pooled_query,
            req.pooled_key,
            req.pooled_value,
            req.query_tile_offset,
            req.key_tile_offset,
        )


class _TritonThresholdTopK(Operator):
    """Selects thresholded top-k video tiles through the Triton kernel."""

    def __init__(self) -> None:
        """Register the Triton threshold-selection provider identity."""

        super().__init__("triton", "video_sparse_threshold_topk")

    def can_run(self, req: ThresholdTopKReq) -> bool:
        """Return whether Triton can select scores on their current device."""

        return triton is not None and triton_available(req.scores.device)

    def run(self, req: ThresholdTopKReq) -> None:
        """Fill the request's selected-index output buffer."""

        _threshold_topk_indices_custom(req.scores, req.output)


class _TritonAddCompression(Operator):
    """Adds trained compression tokens through the Triton sparse-video kernel."""

    def __init__(self) -> None:
        """Register the Triton compression-composition provider identity."""

        super().__init__("triton", "video_sparse_add_compression")

    def can_run(self, req: AddCompressionReq) -> bool:
        """Return whether Triton can compose on the attention output device."""

        return triton is not None and triton_available(req.attended.device)

    def run(self, req: AddCompressionReq) -> None:
        """Fill row-major composed output storage."""

        _unpack_add_compression(req.attended, req.gate, req.compressed, req.output)


class _TritonComposeHeadShards(Operator):
    """Composes rank-local attention heads through the Triton sparse-video kernel."""

    def __init__(self) -> None:
        """Register the Triton head-shard composition provider identity."""

        super().__init__("triton", "video_sparse_compose_head_shards")

    def can_run(self, req: ComposeHeadShardsReq) -> bool:
        """Return whether Triton and the destination layout are available."""

        return (
            triton is not None and triton_available(req.attended.device) and len(req.outputs) >= 1
        )

    def run(self, req: ComposeHeadShardsReq) -> None:
        """Compose local heads directly into destination-rank output shards."""

        _compose_to_head_shards(
            req.attended,
            req.gate,
            req.compressed,
            req.outputs,
            req.source_rank,
        )


@lru_cache(maxsize=1)
def pack_qkv_dispatcher() -> Dispatcher[PackQKVReq, torch.Tensor]:
    """Return the process-wide sparse QKV-packing dispatcher."""

    return Dispatcher("video_sparse_pack_qkv", [_TritonPackQKV()])


@lru_cache(maxsize=1)
def pool_qkv_means_dispatcher() -> Dispatcher[PoolQKVMeansReq, None]:
    """Return the process-wide sparse tile-pooling dispatcher."""

    return Dispatcher("video_sparse_pool_qkv_means", [_TritonPoolQKVMeans()])


@lru_cache(maxsize=1)
def threshold_topk_dispatcher() -> Dispatcher[ThresholdTopKReq, None]:
    """Return the process-wide sparse threshold-selection dispatcher."""

    return Dispatcher("video_sparse_threshold_topk", [_TritonThresholdTopK()])


@lru_cache(maxsize=1)
def add_compression_dispatcher() -> Dispatcher[AddCompressionReq, None]:
    """Return the process-wide compression-composition dispatcher."""

    return Dispatcher("video_sparse_add_compression", [_TritonAddCompression()])


@lru_cache(maxsize=1)
def compose_head_shards_dispatcher() -> Dispatcher[ComposeHeadShardsReq, None]:
    """Return the process-wide head-shard composition dispatcher."""

    return Dispatcher("video_sparse_compose_head_shards", [_TritonComposeHeadShards()])


def pack_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Return Q/K/V packed from ``[rows, heads, width]`` to component-head order."""

    return pack_qkv_dispatcher().run(PackQKVReq(query, key, value))


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

    pool_qkv_means_dispatcher().run(
        PoolQKVMeansReq(
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
    )


def threshold_topk_indices(scores: torch.Tensor, output: torch.Tensor) -> None:
    """Fill selected score-column indices, using ``output`` width as selection count."""

    threshold_topk_dispatcher().run(ThresholdTopKReq(scores, output))


def unpack_add_compression(
    attended: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Fill row-major output with gated per-tile compression added to attention."""

    add_compression_dispatcher().run(AddCompressionReq(attended, gate, compressed, output))


def compose_to_head_shards(
    attended: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    source_rank: int,
) -> None:
    """Compose local heads into destination shards at ``source_rank``'s head range."""

    compose_head_shards_dispatcher().run(
        ComposeHeadShardsReq(attended, gate, compressed, outputs, int(source_rank))
    )


__all__ = [
    "AddCompressionReq",
    "ComposeHeadShardsReq",
    "PackQKVReq",
    "PoolQKVMeansReq",
    "ThresholdTopKReq",
    "add_compression_dispatcher",
    "compose_head_shards_dispatcher",
    "compose_to_head_shards",
    "pack_qkv",
    "pack_qkv_dispatcher",
    "pool_qkv_means",
    "pool_qkv_means_dispatcher",
    "threshold_topk_dispatcher",
    "threshold_topk_indices",
    "unpack_add_compression",
]
