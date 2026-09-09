"""FlashInfer provider for mutable head-wise block-64 video sparse attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...nn.mesh import RowChunkProducer
from ...nn.parallel_attention import AttentionOutputTargets
from ..triton import triton_available

try:  # pragma: no cover - worker_config-only CUDA provider.
    import flashinfer as _flashinfer
except BaseException as error:  # pragma: no cover
    _FLASHINFER_IMPORT_ERROR: BaseException | None = error
    _BlockSparseAttentionWrapper = None
else:  # pragma: no cover
    _FLASHINFER_IMPORT_ERROR = None
    _BlockSparseAttentionWrapper = getattr(_flashinfer, "BlockSparseAttentionWrapper", None)

try:  # pragma: no cover - worker_config-only CUDA provider.
    import triton
    import triton.language as tl
except BaseException as error:  # pragma: no cover
    _TRITON_IMPORT_ERROR: BaseException | None = error
    triton = None
    tl = None
else:  # pragma: no cover
    _TRITON_IMPORT_ERROR = None

_TILE = 64
_HEAD_DIM = 128
_FLOAT_WORKSPACE_BYTES = 128 * 1024 * 1024
_INDEX_BLOCK = 256
_FLOAT_WORKSPACES: dict[tuple[str, int | None], torch.Tensor] = {}


@dataclass(frozen=True, slots=True)
class _SparsePlan:
    """Own one geometry-specific FlashInfer plan and its mutable device index buffer."""

    wrapper: Any
    indices: torch.Tensor
    query_tiles: int
    key_tiles: int
    owner_tiles: int
    interval_tiles: int
    start_tile: int


_PLAN_CACHE: dict[tuple[Any, ...], _SparsePlan] = {}


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
    ):
        """Pack interval/head-major queries and full head-major masked K/V."""

        input_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        row_offsets = row_start + input_offsets
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        row_mask = input_offsets[:, None] < input_rows
        valid_rows = tl.load(
            valid_sizes + row_offsets // tile_rows, mask=input_offsets < input_rows, other=0
        )
        key_mask = row_mask & ((row_offsets % tile_rows)[:, None] < valid_rows[:, None])
        destination = head * rows * width + row_offsets[:, None] * width + columns[None, :]

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
        component_size = heads * rows * width
        owner_rows = rows // owners
        owner = row_offsets // owner_rows
        local_row = row_offsets % owner_rows
        segment = local_row // chunk_rows
        count = tl.minimum(chunk_rows, owner_rows - segment * chunk_rows)
        query_destination = (
            segment * chunk_rows * owners * heads * width
            + head * owners * count * width
            + (owner * count + local_row % chunk_rows) * width
        )
        tl.store(packed + query_destination[:, None] + columns, query_values, mask=row_mask)
        tl.store(packed + component_size + destination, key_values, mask=row_mask)
        tl.store(packed + 2 * component_size + destination, value_values, mask=row_mask)

    @triton.jit
    def _fill_flattened_bsr_kernel(
        source_indices,
        valid_sizes,
        indptr,
        destination_indices,
        invalid_counts,
        source_stride_head: tl.constexpr,
        source_stride_tile: tl.constexpr,
        query_tiles: tl.constexpr,
        key_tiles: tl.constexpr,
        prefix_tiles: tl.constexpr,
        valid_tiles: tl.constexpr,
        selected_video_tiles: tl.constexpr,
        source_width: tl.constexpr,
        index_block: tl.constexpr,
        tile_rows: tl.constexpr,
        owner_tiles: tl.constexpr,
        interval_tiles: tl.constexpr,
        start_tile: tl.constexpr,
    ):
        """Flatten per-head block maps and accumulate padded key rows per query tile."""

        row = tl.program_id(0)
        head = row // query_tiles
        local_tile = row % query_tiles
        query_tile = (
            (local_tile // interval_tiles) * owner_tiles + start_tile + local_tile % interval_tiles
        )
        selected_count = tl.where(
            query_tile < prefix_tiles,
            valid_tiles,
            tl.where(
                query_tile < valid_tiles,
                prefix_tiles + selected_video_tiles,
                1,
            ),
        )
        destination = tl.load(indptr + row)
        invalid = 0
        for chunk in tl.static_range((source_width + index_block - 1) // index_block):
            offsets = chunk * index_block + tl.arange(0, index_block)
            mask = offsets < selected_count
            blocks = tl.load(
                source_indices
                + head * source_stride_head
                + query_tile * source_stride_tile
                + offsets,
                mask=mask,
                other=0,
            )
            tl.store(
                destination_indices + destination + offsets,
                blocks + head * key_tiles,
                mask=mask,
            )
            valid = tl.load(valid_sizes + blocks, mask=mask, other=tile_rows)
            invalid += tl.sum(tl.where(mask, tile_rows - valid, 0), axis=0)
        tl.store(invalid_counts + row, invalid)

    @triton.jit
    def _compose_corrected_kernel(
        attended,
        lse,
        invalid_counts,
        gate,
        compressed,
        output,
        attended_stride_head: tl.constexpr,
        attended_stride_row: tl.constexpr,
        lse_stride_head: tl.constexpr,
        lse_stride_row: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_head: tl.constexpr,
        rows: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Remove padded-key softmax mass and fuse trained compression."""

        row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        mask = row_offsets[:, None] < rows
        invalid = tl.load(
            invalid_counts + head * (rows // tile_rows) + row_offsets // tile_rows,
            mask=row_offsets < rows,
            other=0,
        ).to(tl.float32)
        logsumexp = tl.load(
            lse + head * lse_stride_head + row_offsets * lse_stride_row,
            mask=row_offsets < rows,
            other=0.0,
        )
        retained_mass = 1.0 - invalid * tl.exp2(-logsumexp)
        attended_values = tl.load(
            attended
            + head * attended_stride_head
            + row_offsets[:, None] * attended_stride_row
            + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.load(
            gate + row_offsets[:, None] * gate_stride_row + head * gate_stride_head + columns,
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
        values = attended_values / retained_mass[:, None] + gate_values * compressed_values
        tl.store(
            output + row_offsets[:, None] * tl.num_programs(1) * width + head * width + columns,
            values,
            mask=mask,
        )

    @triton.jit
    def _compose_corrected_shards_kernel(
        attended,
        lse,
        invalid_counts,
        gate,
        compressed,
        outputs,
        attended_stride_head: tl.constexpr,
        attended_stride_row: tl.constexpr,
        lse_stride_head: tl.constexpr,
        lse_stride_row: tl.constexpr,
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
        """Correct sparse outputs and route rank-local heads into row-owner shards."""

        row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        local_row_offsets = row_offsets % local_rows
        global_row_offsets = row_offsets // local_rows * owner_rows + start_row + local_row_offsets
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        mask = row_offsets[:, None] < rows
        invalid = tl.load(
            invalid_counts + head * (rows // tile_rows) + row_offsets // tile_rows,
            mask=row_offsets < rows,
            other=0,
        ).to(tl.float32)
        logsumexp = tl.load(
            lse + head * lse_stride_head + row_offsets * lse_stride_row,
            mask=row_offsets < rows,
            other=0.0,
        )
        retained_mass = 1.0 - invalid * tl.exp2(-logsumexp)
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
        values = attended_values / retained_mass[:, None] + gate_values * compressed_values
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


def available(device: torch.device | None = None) -> bool:
    """Return whether FlashInfer and Triton can execute on the selected CUDA device."""

    if _BlockSparseAttentionWrapper is None or triton is None or not torch.cuda.is_available():
        return False
    selected = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    return triton_available(selected)


def import_error() -> BaseException | None:
    """Return the dependency error that disabled this provider, if any."""

    return _FLASHINFER_IMPORT_ERROR or _TRITON_IMPORT_ERROR


def _planned_counts(
    *,
    query_tiles: int,
    prefix_tiles: int,
    valid_tiles: int,
    owner_tiles: int,
    interval_tiles: int,
    start_tile: int,
) -> torch.Tensor:
    """Construct the immutable per-head CSR row lengths for one H3 geometry."""

    video_tiles = valid_tiles - prefix_tiles
    selected_video_tiles = max(1, (video_tiles + 9) // 10)
    local = torch.arange(query_tiles)
    global_tiles = local // interval_tiles * owner_tiles + start_tile + local % interval_tiles
    counts = torch.ones(query_tiles, dtype=torch.int32)
    counts[global_tiles < prefix_tiles] = valid_tiles
    counts[(global_tiles >= prefix_tiles) & (global_tiles < valid_tiles)] = (
        prefix_tiles + selected_video_tiles
    )
    return counts


def _plan_for(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    prefix_tiles: int,
    valid_tiles: int,
    owners: int = 1,
    row_start: int = 0,
    row_count: int | None = None,
) -> _SparsePlan:
    """Return a cached wrapper planned for head-flattened H3 sparsity."""

    rows, heads, width = (int(size) for size in query.shape)
    owner_rows = rows // owners
    row_count = owner_rows if row_count is None else row_count
    rows = owners * row_count
    key_rows = int(key.shape[0])
    query_tiles, key_tiles = rows // _TILE, key_rows // _TILE
    cache_key = (
        query.device.type,
        query.device.index,
        rows,
        key_rows,
        heads,
        width,
        prefix_tiles,
        valid_tiles,
        query.dtype,
        owner_rows,
        row_start,
        row_count,
    )
    cached = _PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if _BlockSparseAttentionWrapper is None:
        raise RuntimeError("FlashInfer block-sparse attention is unavailable") from import_error()

    counts = _planned_counts(
        query_tiles=query_tiles,
        prefix_tiles=prefix_tiles,
        valid_tiles=valid_tiles,
        owner_tiles=owner_rows // _TILE,
        interval_tiles=row_count // _TILE,
        start_tile=row_start // _TILE,
    ).repeat(heads)
    indptr_host = torch.empty(counts.numel() + 1, dtype=torch.int32)
    indptr_host[0] = 0
    torch.cumsum(counts, dim=0, out=indptr_host[1:])
    indptr = indptr_host.to(query.device)
    indices = torch.zeros(int(indptr_host[-1]), dtype=torch.int32, device=query.device)
    workspace_key = (query.device.type, query.device.index)
    float_workspace = _FLOAT_WORKSPACES.get(workspace_key)
    if float_workspace is None:
        float_workspace = torch.empty(
            _FLOAT_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=query.device,
        )
        _FLOAT_WORKSPACES[workspace_key] = float_workspace
    wrapper = _BlockSparseAttentionWrapper(float_workspace, backend="auto")
    wrapper.plan(
        indptr,
        indices,
        heads * rows,
        heads * key_rows,
        _TILE,
        _TILE,
        1,
        1,
        width,
        q_data_type=query.dtype,
        kv_data_type=key.dtype,
        o_data_type=query.dtype,
    )
    bound_indices = getattr(wrapper, "_paged_kv_indices_buf", None)
    if bound_indices is None or bound_indices.numel() != indices.numel():
        raise RuntimeError("FlashInfer sparse plan did not retain its CSR index buffer")
    plan = _SparsePlan(
        wrapper,
        bound_indices,
        query_tiles,
        key_tiles,
        owner_rows // _TILE,
        row_count // _TILE,
        row_start // _TILE,
    )
    _PLAN_CACHE[cache_key] = plan
    return plan


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
) -> torch.Tensor:
    """Publish a Q/K/V row interval into the provider's complete input layout.

    Caller-owned storage supports out-of-order disjoint intervals produced by
    a distributed projection. Every row must be published before attention
    consumes the buffer. Validity and destination offsets use global rows.
    """

    assert triton is not None
    input_rows, heads, width = (int(size) for size in query.shape)
    if packed is None:
        packed = torch.empty((3, heads, input_rows, width), dtype=query.dtype, device=query.device)
    rows = packed.shape[2]
    if (
        query.shape != key.shape
        or query.shape != value.shape
        or packed.shape != (3, heads, rows, width)
        or packed.dtype != query.dtype
        or packed.device != query.device
        or not packed.is_contiguous()
        or row_start < 0
        or row_start + input_rows > rows
        or owners < 1
        or rows % owners
        or (chunk_rows is not None and chunk_rows < 1)
    ):
        raise ValueError("sparse input rows must fit matching packed owner storage")
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
        num_warps=4,
        num_stages=1,
    )
    return packed


def _fill_flattened_bsr(
    plan: _SparsePlan,
    source_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
    invalid_counts: torch.Tensor,
    *,
    prefix_tiles: int,
    valid_tiles: int,
) -> None:
    """Populate a cached plan's indices from the current device-resident head maps."""

    assert triton is not None
    heads = int(source_indices.shape[0])
    video_tiles = valid_tiles - prefix_tiles
    selected_video_tiles = max(1, (video_tiles + 9) // 10)
    wrapper_indptr = getattr(plan.wrapper, "_paged_kv_indptr_buf", None)
    if wrapper_indptr is None:
        raise RuntimeError("FlashInfer sparse plan has no CSR row pointer buffer")
    _fill_flattened_bsr_kernel[(heads * plan.query_tiles,)](
        source_indices,
        valid_sizes,
        wrapper_indptr,
        plan.indices,
        invalid_counts,
        int(source_indices.stride(0)),
        int(source_indices.stride(1)),
        plan.query_tiles,
        plan.key_tiles,
        prefix_tiles,
        valid_tiles,
        selected_video_tiles,
        int(source_indices.shape[2]),
        _INDEX_BLOCK,
        _TILE,
        plan.owner_tiles,
        plan.interval_tiles,
        plan.start_tile,
        num_warps=4,
        num_stages=1,
    )


def _compose_corrected(
    attended: torch.Tensor,
    lse: torch.Tensor,
    invalid_counts: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    outputs: list[torch.Tensor],
    source_rank: int,
    *,
    owner_rows: int | None = None,
    start_row: int = 0,
) -> None:
    """Correct padded softmax mass and compose into caller-owned destinations."""

    assert triton is not None
    heads, rows, width = (int(size) for size in attended.shape[1:])
    lse = lse.view(heads, rows)
    block_rows = 8
    grid = (triton.cdiv(rows, block_rows), heads)
    if len(outputs) == 1 and owner_rows is None:
        _compose_corrected_kernel[grid](
            attended,
            lse,
            invalid_counts,
            gate,
            compressed,
            outputs[0],
            int(attended.stride(1)),
            int(attended.stride(2)),
            int(lse.stride(0)),
            int(lse.stride(1)),
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
    _compose_corrected_shards_kernel[grid](
        attended,
        lse,
        invalid_counts,
        gate,
        compressed,
        tuple(outputs),
        int(attended.stride(1)),
        int(attended.stride(2)),
        int(lse.stride(0)),
        int(lse.stride(1)),
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


@torch.library.custom_op(
    "uniserve_worker::video_sparse_attention_flashinfer",
    mutates_args=("attention_output", "outputs"),
)
def _block_sparse_custom(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    attention_output: torch.Tensor,
    outputs: list[torch.Tensor],
    source_rank: int,
    prefix_tiles: int,
) -> None:
    """Run a cached flattened-head BSR plan and fuse the corrected epilogue."""

    rows, heads, width = (int(size) for size in query.shape)
    valid_tiles = int(mask_block_indices.shape[2])
    plan = _plan_for(
        query,
        key,
        prefix_tiles=prefix_tiles,
        valid_tiles=valid_tiles,
    )
    invalid_counts = torch.empty(
        (heads, rows // _TILE),
        dtype=torch.int32,
        device=query.device,
    )
    _fill_flattened_bsr(
        plan,
        mask_block_indices,
        valid_sizes,
        invalid_counts,
        prefix_tiles=prefix_tiles,
        valid_tiles=valid_tiles,
    )
    packed = pack_sparse_input_rows(query, key, value, valid_sizes)
    output = attention_output.view(heads * rows, 1, width)
    lse = torch.empty((heads * rows, 1), dtype=torch.float32, device=query.device)
    plan.wrapper.run(
        packed[0].reshape(heads * rows, 1, width),
        packed[1].reshape(heads * rows, 1, width),
        packed[2].reshape(heads * rows, 1, width),
        out=output,
        lse=lse,
        return_lse=True,
    )
    attended = output.view(1, heads, rows, width)
    _compose_corrected(attended, lse, invalid_counts, gate, compressed, outputs, source_rank)


@_block_sparse_custom.register_fake
def _block_sparse_custom_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    attention_output: torch.Tensor,
    outputs: list[torch.Tensor],
    source_rank: int,
    prefix_tiles: int,
) -> None:
    """Declare output mutations without executing optional CUDA dependencies."""

    del (
        query,
        key,
        value,
        mask_block_indices,
        valid_sizes,
        gate,
        compressed,
        attention_output,
        outputs,
        source_rank,
        prefix_tiles,
    )


def execute_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask_block_count: torch.Tensor,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
    tile_size: int,
    prefix_tiles: int,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    attention_output: torch.Tensor,
    targets: AttentionOutputTargets,
) -> torch.Tensor:
    """Execute head-flattened BSR attention for the complete local H3 sequence."""

    if not available(query.device):
        raise RuntimeError("FlashInfer sparse video attention is unavailable") from import_error()
    from . import video_sparse_triton

    video_sparse_triton._validate_outputs(
        query,
        key,
        value,
        mask_block_count=mask_block_count,
        mask_block_indices=mask_block_indices,
        valid_sizes=valid_sizes,
        tile_size=tile_size,
        gate=gate,
        compressed=compressed,
        attention_output=attention_output,
        targets=targets,
    )
    if query.shape != key.shape:
        return video_sparse_triton.execute_sparse_attention(
            query,
            key,
            value,
            mask_block_count=mask_block_count,
            mask_block_indices=mask_block_indices,
            valid_sizes=valid_sizes,
            tile_size=tile_size,
            prefix_tiles=prefix_tiles,
            gate=gate,
            compressed=compressed,
            attention_output=attention_output,
            targets=targets,
        )
    _block_sparse_custom(
        query,
        key,
        value,
        mask_block_indices,
        valid_sizes,
        gate,
        compressed,
        attention_output,
        list(targets.buffers),
        targets.source_rank,
        prefix_tiles,
    )
    return targets.buffers[targets.source_rank]


def prepare_sparse_attention_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
    prefix_tiles: int,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    attention_output: torch.Tensor,
    owners: int,
    chunk_rows: int,
    packed: torch.Tensor | None = None,
) -> RowChunkProducer:
    """Prepare full K/V once and produce paired owner query intervals on demand.

    All owners share one selected key domain. Queries are packed in transport
    interval order. Globally visible prefixes use dense prefill; video queries
    retain their selected block maps. The producer writes complete local-head
    vectors into contiguous row-owner views; communication remains caller-owned.
    """

    if (
        query.shape != key.shape
        or query.shape != value.shape
        or query.ndim != 3
        or query.shape[2] != _HEAD_DIM
        or owners < 1
        or query.shape[0] % (owners * _TILE)
        or chunk_rows < _TILE
        or chunk_rows % _TILE
    ):
        raise ValueError(
            "sparse row production requires equal QKV and tile-aligned owner intervals"
        )
    rows, heads, width = query.shape
    owner_rows = rows // owners
    if packed is None:
        packed = pack_sparse_input_rows(
            query, key, value, valid_sizes, owners=owners, chunk_rows=chunk_rows
        )
    elif (
        packed.shape != (3, heads, rows, width)
        or packed.dtype != query.dtype
        or packed.device != query.device
        or not packed.is_contiguous()
    ):
        raise ValueError("prepared sparse inputs must match the complete query geometry")
    packed_key = packed[1].view(heads * rows, 1, width)
    packed_value = packed[2].view(heads * rows, 1, width)
    prefix_rows = prefix_tiles * _TILE
    valid_tiles = int(mask_block_indices.shape[2])
    dense_invalid = valid_tiles * _TILE - valid_sizes[:valid_tiles].sum(dtype=torch.int32)

    def produce_sparse(
        packed_query: torch.Tensor,
        outputs: tuple[torch.Tensor, ...],
        start: int,
        count: int,
        members: int,
    ) -> None:
        plan = _plan_for(
            query,
            key,
            prefix_tiles=prefix_tiles,
            valid_tiles=valid_tiles,
            owners=members,
            row_start=start,
            row_count=count,
        )
        invalid_counts = torch.empty(
            (heads, members * count // _TILE), dtype=torch.int32, device=query.device
        )
        _fill_flattened_bsr(
            plan,
            mask_block_indices,
            valid_sizes,
            invalid_counts,
            prefix_tiles=prefix_tiles,
            valid_tiles=valid_tiles,
        )
        elements = heads * members * count * width
        output = attention_output.view(-1)[:elements].view(heads * members * count, 1, width)
        lse = torch.empty((heads * members * count, 1), dtype=torch.float32, device=query.device)
        plan.wrapper.run(
            packed_query.reshape(heads * members * count, 1, width),
            packed_key,
            packed_value,
            out=output,
            lse=lse,
            return_lse=True,
        )
        _compose_corrected(
            output.view(1, heads, members * count, width),
            lse,
            invalid_counts,
            gate,
            compressed,
            list(outputs),
            0,
            owner_rows=rows // members,
            start_row=start,
        )

    def produce(interval: slice, outputs: tuple[torch.Tensor, ...]) -> None:
        start, end = interval.start, interval.stop
        count = end - start
        if (
            start % chunk_rows
            or count != min(chunk_rows, owner_rows - start)
            or len(outputs) != owners
            or any(
                output.shape != (count, heads, width)
                or not output.is_contiguous()
                or output.dtype != query.dtype
                or output.device != query.device
                for output in outputs
            )
        ):
            raise ValueError("sparse row destinations must match the prepared owner interval")
        elements = heads * owners * count * width
        packed_query = packed[0].view(-1).narrow(0, start * owners * heads * width, elements)
        packed_query = packed_query.view(heads, owners * count, width)
        if start >= prefix_rows:
            produce_sparse(packed_query, outputs, start, count, owners)
            return

        # Prefix queries see the complete valid key domain. Their dense kernel
        # avoids one-head page traversal; video queries retain the selected BSR
        # domain. The communication interval and caller-owned destinations stay
        # unchanged, including an interval spanning the prefix/video boundary.
        for owner, destination in enumerate(outputs):
            global_start = owner * owner_rows + start
            dense_rows = max(0, min(count, prefix_rows - global_start))
            owner_query = packed_query[:, owner * count : (owner + 1) * count]
            if dense_rows:
                attended, lse = _flashinfer.single_prefill_with_kv_cache(
                    owner_query[:, :dense_rows].transpose(0, 1),
                    packed[1, :, : valid_tiles * _TILE],
                    packed[2, :, : valid_tiles * _TILE],
                    kv_layout="HND",
                    backend="fa2",
                    return_lse=True,
                )
                _compose_corrected(
                    attended.transpose(0, 1).unsqueeze(0),
                    lse.transpose(0, 1),
                    dense_invalid.expand(heads, dense_rows // _TILE).contiguous(),
                    gate,
                    compressed,
                    [destination[:dense_rows]],
                    0,
                    owner_rows=rows,
                    start_row=global_start,
                )
            if dense_rows < count:
                produce_sparse(
                    owner_query[:, dense_rows:],
                    (destination[dense_rows:],),
                    global_start + dense_rows,
                    count - dense_rows,
                    1,
                )

    return produce


__all__ = ["available", "execute_sparse_attention", "prepare_sparse_attention_rows", "import_error"]
