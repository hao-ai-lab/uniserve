"""FlashInfer provider for mutable head-wise block-64 video sparse attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

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
        heads: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Pack head-major QKV while zeroing padded K/V rows."""

        row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        row_mask = row_offsets[:, None] < rows
        valid_rows = tl.load(valid_sizes + row_offsets // tile_rows, mask=row_offsets < rows, other=0)
        key_mask = row_mask & ((row_offsets % tile_rows)[:, None] < valid_rows[:, None])
        destination = head * rows * width + row_offsets[:, None] * width + columns[None, :]

        query_values = tl.load(
            query
            + row_offsets[:, None] * query_stride_row
            + head * query_stride_head
            + columns[None, :],
            mask=row_mask,
            other=0.0,
        )
        key_values = tl.load(
            key
            + row_offsets[:, None] * key_stride_row
            + head * key_stride_head
            + columns[None, :],
            mask=key_mask,
            other=0.0,
        )
        value_values = tl.load(
            value
            + row_offsets[:, None] * value_stride_row
            + head * value_stride_head
            + columns[None, :],
            mask=key_mask,
            other=0.0,
        )
        component_size = heads * rows * width
        tl.store(packed + destination, query_values, mask=row_mask)
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
    ):
        """Flatten per-head block maps and accumulate padded key rows per query tile."""

        row = tl.program_id(0)
        head = row // query_tiles
        query_tile = row % query_tiles
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
        flat_rows = head * rows + row_offsets
        invalid = tl.load(
            invalid_counts + head * (rows // tile_rows) + row_offsets // tile_rows,
            mask=row_offsets < rows,
            other=0,
        ).to(tl.float32)
        logsumexp = tl.load(lse + flat_rows, mask=row_offsets < rows, other=0.0)
        retained_mass = 1.0 - invalid * tl.exp2(-logsumexp)
        attended_values = tl.load(
            attended + flat_rows[:, None] * width + columns[None, :],
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
            output
            + row_offsets[:, None] * tl.num_programs(1) * width
            + head * width
            + columns,
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
    ):
        """Correct sparse outputs and route rank-local heads into row-owner shards."""

        row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        local_row_offsets = row_offsets % local_rows
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        mask = row_offsets[:, None] < rows
        flat_rows = head * rows + row_offsets
        invalid = tl.load(
            invalid_counts + head * (rows // tile_rows) + row_offsets // tile_rows,
            mask=row_offsets < rows,
            other=0,
        ).to(tl.float32)
        logsumexp = tl.load(lse + flat_rows, mask=row_offsets < rows, other=0.0)
        retained_mass = 1.0 - invalid * tl.exp2(-logsumexp)
        attended_values = tl.load(
            attended + flat_rows[:, None] * width + columns[None, :],
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
) -> torch.Tensor:
    """Construct the immutable per-head CSR row lengths for one H3 geometry."""

    video_tiles = valid_tiles - prefix_tiles
    selected_video_tiles = max(1, (video_tiles + 9) // 10)
    counts = torch.ones(query_tiles, dtype=torch.int32)
    counts[:prefix_tiles] = valid_tiles
    counts[prefix_tiles:valid_tiles] = prefix_tiles + selected_video_tiles
    return counts


def _plan_for(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    prefix_tiles: int,
    valid_tiles: int,
) -> _SparsePlan:
    """Return a cached wrapper planned for head-flattened H3 sparsity."""

    rows, heads, width = (int(size) for size in query.shape)
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
    plan = _SparsePlan(wrapper, bound_indices, query_tiles, key_tiles)
    _PLAN_CACHE[cache_key] = plan
    return plan


def _pack_masked_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_sizes: torch.Tensor,
) -> torch.Tensor:
    """Return contiguous component/head-major QKV with padded K/V rows zeroed."""

    assert triton is not None
    rows, heads, width = (int(size) for size in query.shape)
    packed = torch.empty((3, heads, rows, width), dtype=query.dtype, device=query.device)
    block_rows = 8
    _pack_masked_qkv_kernel[(triton.cdiv(rows, block_rows), heads)](
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
        heads,
        width,
        _TILE,
        block_rows,
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
) -> None:
    """Correct padded softmax mass and compose into caller-owned destinations."""

    assert triton is not None
    heads, rows, width = (int(size) for size in attended.shape[1:])
    block_rows = 8
    grid = (triton.cdiv(rows, block_rows), heads)
    if len(outputs) == 1:
        _compose_corrected_kernel[grid](
            attended,
            lse,
            invalid_counts,
            gate,
            compressed,
            outputs[0],
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
        int(gate.stride(0)),
        int(gate.stride(1)),
        rows,
        rows // len(outputs),
        heads,
        heads * len(outputs),
        source_rank,
        width,
        _TILE,
        block_rows,
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
    packed = _pack_masked_qkv(query, key, value, valid_sizes)
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


__all__ = ["available", "execute_sparse_attention", "import_error"]
