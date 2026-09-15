"""FlashInfer provider for mutable head-wise block-64 video sparse attention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from uniserve.distributed._chunks import _ChunkProducer
from uniserve.nn.attention.vsa.inputs import Pattern
from uniserve.ops.video_sparse_rows import compose_attention, pack_sparse_input_rows
from uniserve.runtime.triton import triton_available

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


@dataclass(frozen=True, slots=True)
class _SparsePlan:
    """Own one shape-specific FlashInfer plan and its mutable device index buffer."""

    wrapper: Any
    indices: torch.Tensor
    packed_mask: torch.Tensor
    query_tiles: int
    key_tiles: int
    owner_tiles: int
    interval_tiles: int
    start_tile: int


@dataclass(frozen=True, slots=True)
class _NativeSparsePlan:
    """Select owner-local query maps into stable native-provider metadata."""

    query_tiles: torch.Tensor
    indices: torch.Tensor
    counts: torch.Tensor


@dataclass
class SparseExecutionState:
    """Own mutable plans and scratch for one serialized sparse execution domain."""

    plans: dict[tuple[Any, ...], _SparsePlan] = field(default_factory=dict)
    native_plans: dict[tuple[Any, ...], _NativeSparsePlan] = field(default_factory=dict)
    workspaces: dict[torch.device, torch.Tensor] = field(default_factory=dict)


def uses_row_major_inputs(device: torch.device) -> bool:
    """Select the SM120/SM121 block-64 provider's contiguous sequence-head layout."""

    return torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))


if triton is not None:

    @triton.jit
    def _fill_flattened_bsr_kernel(
        source_indices,
        valid_sizes,
        indptr,
        destination_indices,
        packed_mask,
        source_stride_head: tl.constexpr,
        source_stride_tile: tl.constexpr,
        query_tiles: tl.constexpr,
        key_tiles: tl.constexpr,
        source_width: tl.constexpr,
        index_block: tl.constexpr,
        tile_rows: tl.constexpr,
        owner_tiles: tl.constexpr,
        interval_tiles: tl.constexpr,
        start_tile: tl.constexpr,
    ):
        """Populate head-flattened indices and exact little-endian key validity bits."""

        row = tl.program_id(0)
        head = row // query_tiles
        local_tile = row % query_tiles
        query_tile = (
            (local_tile // interval_tiles) * owner_tiles + start_tile + local_tile % interval_tiles
        )
        destination = tl.load(indptr + row)
        selected_count = tl.load(indptr + row + 1) - destination
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

        bytes_per_tile_row = tile_rows // 8
        mask_bytes = selected_count * tile_rows * bytes_per_tile_row
        for begin in tl.range(0, mask_bytes, 1024):
            byte_offsets = begin + tl.arange(0, 1024)
            active = byte_offsets < mask_bytes
            selected = (byte_offsets // bytes_per_tile_row) % selected_count
            selected_blocks = tl.load(
                source_indices
                + head * source_stride_head
                + query_tile * source_stride_tile
                + selected,
                mask=active,
                other=0,
            )
            valid_rows = tl.load(valid_sizes + selected_blocks, mask=active, other=0)
            bits = tl.minimum(
                tl.maximum(valid_rows - (byte_offsets % bytes_per_tile_row) * 8, 0), 8
            )
            tl.store(
                packed_mask + destination * tile_rows * bytes_per_tile_row + byte_offsets,
                ((1 << bits) - 1).to(tl.uint8),
                mask=active,
            )

    @triton.jit
    def _pack_key_validity_kernel(valid_sizes, packed_mask, byte_count: tl.constexpr):
        offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
        valid = tl.load(valid_sizes + offsets // 8, mask=offsets < byte_count, other=0)
        bits = tl.minimum(tl.maximum(valid - (offsets % 8) * 8, 0), 8)
        tl.store(packed_mask + offsets, ((1 << bits) - 1).to(tl.uint8), mask=offsets < byte_count)


def available(device: torch.device | None = None) -> bool:
    """Return whether FlashInfer and Triton can execute on the selected CUDA device."""

    if _BlockSparseAttentionWrapper is None or triton is None or not torch.cuda.is_available():
        return False
    selected = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    return triton_available(selected)


def import_error() -> BaseException | None:
    """Return the dependency error that disabled this provider, if any."""

    return _FLASHINFER_IMPORT_ERROR or _TRITON_IMPORT_ERROR


def _plan_for(
    state: SparseExecutionState,
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    pattern: Pattern,
    index_width: int,
    scale: float,
    owners: int = 1,
    row_start: int = 0,
    row_count: int | None = None,
) -> _SparsePlan:
    """Plan head-flattened BSR from the caller's declared row cardinalities."""

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
        pattern,
        index_width,
        query.dtype,
        scale,
        owner_rows,
        row_start,
        row_count,
    )
    cached = state.plans.get(cache_key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare FlashInfer VSA shapes and scales before capture")
    if _BlockSparseAttentionWrapper is None:
        raise RuntimeError("FlashInfer block-sparse attention is unavailable") from import_error()

    interval_tiles = row_count // _TILE
    local_tiles = torch.arange(owners * interval_tiles)
    selected = local_tiles // interval_tiles * (owner_rows // _TILE)
    selected += row_start // _TILE + local_tiles % interval_tiles
    counts = pattern.counts(
        num_heads=heads,
        query_tiles=query.shape[0] // _TILE,
        index_width=index_width,
        out=torch.empty((heads, query.shape[0] // _TILE), device="cpu", dtype=torch.int32),
    )
    counts = counts.index_select(1, selected).reshape(-1)
    indptr_host = torch.empty(counts.numel() + 1, dtype=torch.int32)
    indptr_host[0] = 0
    torch.cumsum(counts, dim=0, out=indptr_host[1:])
    indptr = indptr_host.to(query.device)
    indices = torch.zeros(int(indptr_host[-1]), dtype=torch.int32, device=query.device)
    workspace_key = query.device
    float_workspace = state.workspaces.get(workspace_key)
    if float_workspace is None:
        float_workspace = torch.empty(
            _FLOAT_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=query.device,
        )
        state.workspaces[workspace_key] = float_workspace
    packed_mask = torch.empty(
        indices.numel() * _TILE * (_TILE // 8), dtype=torch.uint8, device=query.device
    )
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
        packed_mask=packed_mask,
        sm_scale=scale,
    )
    bound_indices = getattr(wrapper, "_paged_kv_indices_buf", None)
    if bound_indices is None or bound_indices.numel() != indices.numel():
        raise RuntimeError("FlashInfer sparse plan did not retain its CSR index buffer")
    bound_mask = wrapper._packed_mask_buf
    mask_offsets = wrapper._mask_indptr_buf
    if bound_mask is None or mask_offsets is None:
        raise RuntimeError("FlashInfer sparse plan did not retain its packed key mask")
    # The execution interface indexes packed-mask bytes. Own these offsets
    # alongside the mutable bits, deriving each query tile from its exact CSR extent.
    mask_offsets.copy_((indptr_host * (_TILE * _TILE // 8)).to(query.device))
    plan = _SparsePlan(
        wrapper,
        bound_indices,
        bound_mask,
        query_tiles,
        key_tiles,
        owner_rows // _TILE,
        row_count // _TILE,
        row_start // _TILE,
    )
    state.plans[cache_key] = plan
    return plan


def _native_plan_for(
    state: SparseExecutionState,
    source_indices: torch.Tensor,
    *,
    owners: int,
    row_start: int,
    row_count: int,
) -> _NativeSparsePlan:
    """Cache query mapping and storage; counts and selected keys remain device mutable."""

    heads, total_tiles, index_width = source_indices.shape
    cache_key = (
        source_indices.device,
        heads,
        total_tiles,
        index_width,
        owners,
        row_start,
        row_count,
    )
    cached = state.native_plans.get(cache_key)
    if cached is not None:
        return cached
    owner_tiles = total_tiles // owners
    interval_tiles = row_count // _TILE
    local_tiles = torch.arange(owners * interval_tiles)
    query_tiles = (
        local_tiles // interval_tiles * owner_tiles
        + row_start // _TILE
        + local_tiles % interval_tiles
    )
    plan = _NativeSparsePlan(
        query_tiles.to(source_indices.device),
        torch.empty(
            (1, heads, owners * interval_tiles, index_width),
            dtype=torch.int32,
            device=source_indices.device,
        ),
        torch.empty(
            (1, heads, owners * interval_tiles), dtype=torch.int32, device=source_indices.device
        ),
    )
    state.native_plans[cache_key] = plan
    return plan


def _fill_flattened_bsr(
    plan: _SparsePlan,
    source_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
) -> None:
    """Populate a cached plan's indices from the current device-resident head maps."""

    assert triton is not None
    heads = int(source_indices.shape[0])
    wrapper_indptr = getattr(plan.wrapper, "_paged_kv_indptr_buf", None)
    if wrapper_indptr is None:
        raise RuntimeError("FlashInfer sparse plan has no CSR row pointer buffer")
    _fill_flattened_bsr_kernel[(heads * plan.query_tiles,)](
        source_indices,
        valid_sizes,
        wrapper_indptr,
        plan.indices,
        plan.packed_mask,
        int(source_indices.stride(0)),
        int(source_indices.stride(1)),
        plan.query_tiles,
        plan.key_tiles,
        int(source_indices.shape[2]),
        _INDEX_BLOCK,
        _TILE,
        plan.owner_tiles,
        plan.interval_tiles,
        plan.start_tile,
        num_warps=4,
        num_stages=1,
    )


def prepare_rows(
    state: SparseExecutionState,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask_block_indices: torch.Tensor,
    mask_block_count: torch.Tensor,
    valid_sizes: torch.Tensor,
    pattern: Pattern,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    attention_output: torch.Tensor,
    owners: int,
    chunk_rows: int,
    packed: torch.Tensor | None = None,
    scale: float,
) -> _ChunkProducer:
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
    native_rows = uses_row_major_inputs(query.device)
    if native_rows:
        from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_fwd

    packed_shape = (3, rows, heads, width) if native_rows else (3, heads, rows, width)
    if packed is None:
        packed = pack_sparse_input_rows(
            query,
            key,
            value,
            valid_sizes,
            owners=owners,
            chunk_rows=chunk_rows,
            row_major=native_rows,
        )
    elif (
        packed.shape != packed_shape
        or packed.dtype != query.dtype
        or packed.device != query.device
        or not packed.is_contiguous()
    ):
        raise ValueError("prepared sparse inputs must match the complete query shape")
    packed_key = packed[1].transpose(0, 1) if native_rows else packed[1]
    packed_value = packed[2].transpose(0, 1) if native_rows else packed[2]
    prefix_rows = pattern.dense_prefix_tiles * _TILE
    valid_tiles = pattern.dense_key_tiles
    key_mask = None
    if valid_tiles:
        key_mask = torch.empty(valid_tiles * (_TILE // 8), dtype=torch.uint8, device=query.device)
        _pack_key_validity_kernel[(triton.cdiv(key_mask.numel(), 256),)](
            valid_sizes, key_mask, key_mask.numel()
        )

    def produce_sparse(
        packed_query: torch.Tensor,
        outputs: tuple[torch.Tensor, ...],
        start: int,
        count: int,
        members: int,
    ) -> None:
        elements = heads * members * count * width
        if native_rows:
            native_plan = _native_plan_for(
                state,
                mask_block_indices,
                owners=members,
                row_start=start,
                row_count=count,
            )
            selected_tiles = native_plan.indices.shape[3]
            torch.index_select(
                mask_block_indices[:, :, :selected_tiles],
                1,
                native_plan.query_tiles,
                out=native_plan.indices[0],
            )
            torch.index_select(
                mask_block_count, 1, native_plan.query_tiles, out=native_plan.counts[0]
            )
            output = attention_output.view(-1)[:elements].view(1, members * count, heads, width)
            bsa_attn_sm120_blk64_fwd(
                packed_query.transpose(0, 1).unsqueeze(0),
                packed[1].unsqueeze(0),
                packed[2].unsqueeze(0),
                native_plan.indices,
                selected_tiles,
                block_sizes=valid_sizes,
                q2k_block_nums=native_plan.counts,
                out=output,
            )
            compose_attention(
                output.transpose(1, 2),
                gate,
                compressed,
                list(outputs),
                0,
                owner_rows=rows // members,
                start_row=start,
            )
            return
        plan = _plan_for(
            state,
            query,
            key,
            pattern=pattern,
            index_width=mask_block_indices.shape[-1],
            scale=scale,
            owners=members,
            row_start=start,
            row_count=count,
        )
        _fill_flattened_bsr(plan, mask_block_indices, valid_sizes)
        output = attention_output.view(-1)[:elements].view(heads * members * count, 1, width)
        plan.wrapper.run(
            packed_query.reshape(heads * members * count, 1, width),
            packed_key.view(heads * rows, 1, width),
            packed_value.view(heads * rows, 1, width),
            out=output,
        )
        compose_attention(
            output.view(1, heads, members * count, width),
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
        packed_query = (
            packed_query.view(owners * count, heads, width).transpose(0, 1)
            if native_rows
            else packed_query.view(heads, owners * count, width)
        )
        if start >= prefix_rows:
            produce_sparse(packed_query, outputs, start, count, owners)
            return

        # Prefix queries see the complete valid key domain. Their dense kernel
        # avoids one-head page traversal; video queries retain their selected key
        # domain. The communication interval and caller-owned destinations stay
        # unchanged, including an interval spanning the prefix/video boundary.
        for owner, destination in enumerate(outputs):
            global_start = owner * owner_rows + start
            dense_rows = max(0, min(count, prefix_rows - global_start))
            owner_query = packed_query[:, owner * count : (owner + 1) * count]
            if dense_rows:
                assert key_mask is not None
                attended = _flashinfer.single_prefill_with_kv_cache(
                    owner_query[:, :dense_rows].transpose(0, 1),
                    packed_key[:, : valid_tiles * _TILE],
                    packed_value[:, : valid_tiles * _TILE],
                    kv_layout="HND",
                    backend="fa2",
                    sm_scale=scale,
                    packed_custom_mask=key_mask.repeat(dense_rows),
                )
                compose_attention(
                    attended.transpose(0, 1).unsqueeze(0),
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
