"""FlashInfer provider for mutable head-wise block-64 video sparse attention.

On Hopper, sparse rows use FlashInfer's FA3 prefill with the checkpoint's
live per-head block selection. Each selected 64-row block expands into
single-token KV pages holding only its valid keys, so partial and empty
blocks need no score mask. Dense-prefix query rows attend their complete
valid key domain through FlashAttention-4's SM90 kernel, whose device
predicate excludes each tile's padding. Query packing preserves owner and
interval order, including a final short interval.

SM120/SM121 use a native kernel that reads the block maps directly. Other
devices use FlashInfer's FA2 block-sparse prefill over a head-flattened BSR
in which each selected 64x64 block carries 512 bytes of packed
key-validity bits, excluding the padding keys of partial and empty tiles.

FA3's scheduler narrows single-token page offsets, and FA2 its packed-mask
byte offsets, to signed 32 bits. A query domain exceeding either splits into
independent windows, each retaining its queries' complete selected keys.

Startup graphs capture the CSR updates together with attention. Serialized
calls borrow the execution context's mutable CSR backing, and layers with
the same numerical signature share their shape-specific plans, which bounds
graph residency across transformer depth and captured layouts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from uniserve_kernels.attention.vsa_rows import (
    compose_attention,
    pack_sparse_input_rows,
)
from uniserve_kernels.triton import launchable

from uniserve.distributed.chunks import ChunkProducer
from uniserve.nn.attention.vsa.inputs import Pattern
from uniserve.tensors import BufferConfig

try:  # pragma: no cover - worker_config-only CUDA provider.
    import flashinfer as _flashinfer
except BaseException as error:  # pragma: no cover
    _FLASHINFER_IMPORT_ERROR: BaseException | None = error
    _BlockSparseAttentionWrapper = None
else:  # pragma: no cover
    _FLASHINFER_IMPORT_ERROR = None
    _BlockSparseAttentionWrapper = getattr(
        _flashinfer, "BlockSparseAttentionWrapper", None
    )

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
# Packed key-validity bytes of one selected 64x64 BSR block.
_MASK_BYTES = _TILE * _TILE // 8
# Largest offset one FA3 or FA2 launch can address.
_MAX_OFFSET = torch.iinfo(torch.int32).max


@dataclass(frozen=True, slots=True)
class _SparsePlan:
    """Own one shape-specific FlashInfer plan.

    Own one shape-specific FlashInfer plan and its mutable device index
    buffer.
    """

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


@dataclass(frozen=True, slots=True)
class _TokenPlan:
    """FlashInfer FA3 CSR over the valid tokens of selected tile-64 blocks.

    Single-token pages express partial and empty tiles without a custom
    score mask. The plan schedules each row's declared capacity; device CSR
    pointers and indices carry its live key domain on every graph replay.
    """

    wrapper: Any
    indices: torch.Tensor
    indptr: torch.Tensor
    counts: torch.Tensor
    work_batches: torch.Tensor
    work_starts: torch.Tensor
    work_counts: torch.Tensor
    query_tiles: int
    key_tiles: int
    owner_tiles: int
    interval_tiles: int
    start_tile: int


@dataclass(frozen=True, slots=True)
class _Queries:
    """Independent query windows over the same complete key domain."""

    plans: tuple[_SparsePlan | _TokenPlan, ...]
    heads: int
    owners: int
    rows: int
    start_tile: int


@dataclass
class SparseExecutionState:
    """Own mutable plans and scratch.

    Own mutable plans and scratch for one serialized sparse execution
    domain.
    """

    plans: dict[tuple[Any, ...], _SparsePlan | _TokenPlan | _Queries] = field(
        default_factory=dict
    )
    native_plans: dict[tuple[Any, ...], _NativeSparsePlan] = field(
        default_factory=dict
    )
    workspaces: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    # Lends the owning context's shared per-call work areas (see
    # ``ExecutionContext.scratch``), or ``None`` to allocate them per plan.
    transient: Callable[..., Mapping[str, torch.Tensor]] | None = None


def uses_row_major_inputs(device: torch.device) -> bool:
    """Select the SM120/SM121 block-64 provider's layout.

    Select the SM120/SM121 block-64 provider's contiguous sequence-head
    layout.
    """
    return torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))


if triton is not None:

    @triton.jit
    def _refresh_sparse_work_kernel(
        batches,
        indptr,
        counts,
        starts,
        lengths,
        work_count: tl.constexpr,
        block: tl.constexpr,
    ):
        work = tl.program_id(0) * block + tl.arange(0, block)
        active = work < work_count
        batch = tl.load(batches + work, mask=active, other=0)
        start = tl.load(indptr + batch, mask=active, other=0)
        count = tl.load(counts + batch, mask=active, other=0)
        tl.store(starts + work, start, mask=active)
        tl.store(lengths + work, count, mask=active)

    @triton.jit
    def _count_sparse_tokens_kernel(
        source_indices,
        source_counts,
        valid_sizes,
        counts,
        index_stride_head: tl.constexpr,
        index_stride_tile: tl.constexpr,
        count_stride_head: tl.constexpr,
        query_tiles: tl.constexpr,
        owner_tiles: tl.constexpr,
        interval_tiles: tl.constexpr,
        start_tile: tl.constexpr,
        source_width: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = row // query_tiles
        local_tile = row % query_tiles
        tile = (
            local_tile // interval_tiles * owner_tiles
            + start_tile
            + local_tile % interval_tiles
        )
        live_count = tl.load(source_counts + head * count_stride_head + tile)
        selected = tl.arange(0, block)
        active = (selected < source_width) & (selected < live_count)
        keys = tl.load(
            source_indices
            + head * index_stride_head
            + tile * index_stride_tile
            + selected,
            mask=active,
            other=0,
        )
        sizes = tl.load(valid_sizes + keys, mask=active, other=0)
        tl.store(counts + row, tl.sum(sizes, 0))

    @triton.jit
    def _fill_sparse_tokens_kernel(
        source_indices,
        source_counts,
        valid_sizes,
        indptr,
        indices,
        index_stride_head: tl.constexpr,
        index_stride_tile: tl.constexpr,
        count_stride_head: tl.constexpr,
        query_tiles: tl.constexpr,
        key_tiles: tl.constexpr,
        owner_tiles: tl.constexpr,
        interval_tiles: tl.constexpr,
        start_tile: tl.constexpr,
        source_width: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = row // query_tiles
        local_tile = row % query_tiles
        tile = (
            local_tile // interval_tiles * owner_tiles
            + start_tile
            + local_tile % interval_tiles
        )
        live_count = tl.load(source_counts + head * count_stride_head + tile)
        destination = tl.load(indptr + row)
        columns = tl.arange(0, 16)
        tokens = tl.arange(0, 64)
        # Expand a bounded group at a time rather than materializing a
        # source_width x 64 register tile for each query block.
        for begin in range(tl.cdiv(source_width, 16)):
            selected = begin * 16 + columns
            active = (selected < source_width) & (selected < live_count)
            keys = tl.load(
                source_indices
                + head * index_stride_head
                + tile * index_stride_tile
                + selected,
                mask=active,
                other=0,
            )
            sizes = tl.load(valid_sizes + keys, mask=active, other=0)
            offsets = tl.cumsum(sizes, 0) - sizes
            tl.store(
                indices + destination + offsets[:, None] + tokens[None, :],
                (head * key_tiles + keys[:, None]) * 64 + tokens[None, :],
                mask=active[:, None] & (tokens[None, :] < sizes[:, None]),
            )
            destination += tl.sum(sizes, 0)

    @triton.jit
    def _fill_flattened_bsr_kernel(
        source_indices,
        source_counts,
        valid_sizes,
        indptr,
        destination_indices,
        packed_mask,
        source_stride_head: tl.constexpr,
        source_stride_tile: tl.constexpr,
        count_stride_head: tl.constexpr,
        query_tiles: tl.constexpr,
        key_tiles: tl.constexpr,
        source_width: tl.constexpr,
        index_block: tl.constexpr,
        tile_rows: tl.constexpr,
        owner_tiles: tl.constexpr,
        interval_tiles: tl.constexpr,
        start_tile: tl.constexpr,
    ):
        """Populate head-flattened indices and key validity bits.

        Populate head-flattened indices and exact little-endian key
        validity bits. The plan reserves each row's declared count bound;
        entries past the row's live count in ``source_counts`` address key
        tile 0 with every validity bit clear, so they contribute nothing.
        """
        # Each program owns one (head, local query tile) row of the
        # flattened BSR.
        row = tl.program_id(0)
        head = row // query_tiles
        local_tile = row % query_tiles
        query_tile = (
            (local_tile // interval_tiles) * owner_tiles
            + start_tile
            + local_tile % interval_tiles
        )
        destination = tl.load(indptr + row)
        selected_count = tl.load(indptr + row + 1) - destination
        live_count = tl.load(
            source_counts + head * count_stride_head + query_tile
        )

        # Copy the selected key-tile IDs, rebasing them into this head's slice
        # of the head-flattened key domain.
        for chunk in tl.static_range(
            (source_width + index_block - 1) // index_block
        ):
            offsets = chunk * index_block + tl.arange(0, index_block)
            mask = offsets < selected_count
            blocks = tl.load(
                source_indices
                + head * source_stride_head
                + query_tile * source_stride_tile
                + offsets,
                mask=mask & (offsets < live_count),
                other=0,
            )
            tl.store(
                destination_indices + destination + offsets,
                blocks + head * key_tiles,
                mask=mask,
            )

        # Pack each key tile's valid rows as bits, 8 rows per byte, so the
        # FlashInfer BSR mask marks exactly valid_sizes[tile] keys per tile.
        # Every BSR entry owns tile_rows * bytes_per_tile_row mask bytes, so
        # the row's byte offset is formed in 64 bits rather than from the
        # int32 entry offset.
        bytes_per_tile_row = tile_rows // 8
        mask_bytes = selected_count * tile_rows * bytes_per_tile_row
        row_mask = packed_mask + destination.to(tl.int64) * (
            tile_rows * bytes_per_tile_row
        )
        for begin in tl.range(0, mask_bytes, 1024):
            byte_offsets = begin + tl.arange(0, 1024)
            active = byte_offsets < mask_bytes
            selected = (byte_offsets // bytes_per_tile_row) % selected_count
            live = active & (selected < live_count)
            selected_blocks = tl.load(
                source_indices
                + head * source_stride_head
                + query_tile * source_stride_tile
                + selected,
                mask=live,
                other=0,
            )
            valid_rows = tl.load(
                valid_sizes + selected_blocks, mask=live, other=0
            )
            bits = tl.minimum(
                tl.maximum(
                    valid_rows - (byte_offsets % bytes_per_tile_row) * 8, 0
                ),
                8,
            )
            tl.store(
                row_mask + byte_offsets,
                ((1 << bits) - 1).to(tl.uint8),
                mask=active,
            )

    @triton.jit
    def _pack_key_validity_kernel(
        valid_sizes, packed_mask, byte_count: tl.constexpr
    ):
        # Same 8-rows-per-byte packing as above, for one dense key prefix.
        offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
        valid = tl.load(
            valid_sizes + offsets // 8, mask=offsets < byte_count, other=0
        )
        bits = tl.minimum(tl.maximum(valid - (offsets % 8) * 8, 0), 8)
        tl.store(
            packed_mask + offsets,
            ((1 << bits) - 1).to(tl.uint8),
            mask=offsets < byte_count,
        )


def available(device: torch.device | None = None) -> bool:
    """Return whether FlashInfer and Triton can execute.

    Return whether FlashInfer and Triton can execute on the selected CUDA
    device.
    """
    if (
        _BlockSparseAttentionWrapper is None
        or triton is None
        or not torch.cuda.is_available()
    ):
        return False
    selected = (
        device
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    return launchable(selected)


def import_error() -> BaseException | None:
    """Return the dependency error that disabled this provider, if any."""
    return _FLASHINFER_IMPORT_ERROR or _TRITON_IMPORT_ERROR


def _csr_buffers(
    entries: int, rows: int, *, hopper: bool
) -> tuple[str, dict[str, BufferConfig]]:
    """Name and size the mutable CSR one plan rewrites before each launch.

    ``rows`` counts the plan's head-flattened query tiles and ``entries``
    their declared key tiles. No extent decreases as either grows, so
    requirements sized for the most rows and entries of several plans
    contain each plan's own.
    """
    if hopper:
        # FA3 lists one single-token page per key row of each selected tile.
        return "vsa_sparse_tokens", {
            "indices": BufferConfig((entries * _TILE,), torch.int32),
            "indptr": BufferConfig((rows + 1,), torch.int32),
            "counts": BufferConfig((rows,), torch.int32),
        }
    return "vsa_bsr_rows", {
        "indices": BufferConfig((entries,), torch.int32),
        "mask": BufferConfig((entries * _MASK_BYTES,), torch.uint8),
    }


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
) -> _SparsePlan | _TokenPlan | _Queries:
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
        raise RuntimeError(
            "prepare FlashInfer VSA shapes and scales before capture"
        )
    if _BlockSparseAttentionWrapper is None:
        raise RuntimeError(
            "FlashInfer block-sparse attention is unavailable"
        ) from import_error()

    # Map each owner's packed local tiles back to its own query tiles in the
    # full pattern: owner stride owner_rows, offset row_start.
    interval_tiles = row_count // _TILE
    local_tiles = torch.arange(owners * interval_tiles)
    selected = local_tiles // interval_tiles * (owner_rows // _TILE)
    selected += row_start // _TILE + local_tiles % interval_tiles

    counts = pattern.counts(
        num_heads=heads,
        query_tiles=query.shape[0] // _TILE,
        index_width=index_width,
        out=torch.empty(
            (heads, query.shape[0] // _TILE), device="cpu", dtype=torch.int32
        ),
    )
    counts = counts.index_select(1, selected).reshape(-1)
    hopper = torch.cuda.get_device_capability(query.device)[0] == 9
    # Each declared key tile occupies 64 single-token pages of FA3's CSR or
    # 512 bytes of FA2's packed mask. FA3's WorkTileInfo narrows KV offsets
    # to signed int even when the public CSR uses int64, and FA2 locates
    # each query tile's mask bits through int32 byte offsets.
    entry_size = _TILE if hopper else _MASK_BYTES
    if int(counts.sum(dtype=torch.int64)) * entry_size > _MAX_OFFSET:
        # Independent query windows preserve every selected key while
        # bounding each launch's offsets.
        costs = (
            counts.view(heads, owners, interval_tiles).sum(
                dim=(0, 1), dtype=torch.int64
            )
            * entry_size
        )
        windows = []
        begin, capacity = 0, 0
        for tile, cost in enumerate(costs.tolist()):
            if cost > _MAX_OFFSET:
                raise ValueError(
                    "one VSA query tile exceeds the 32-bit attention offsets"
                )
            if capacity + cost > _MAX_OFFSET:
                windows.append((begin, tile))
                begin, capacity = tile, 0
            capacity += cost
        windows.append((begin, interval_tiles))

        if state.transient is not None:
            # Windows run one after another and borrow one shared backing,
            # but a window larger than every backing allocates another and
            # earlier ones stay bound to their plans. Greedy windows all
            # approach the offset limit, so borrow the largest extents
            # before planning any of them.
            role, requirements = _csr_buffers(
                max(int(costs[begin:end].sum()) for begin, end in windows)
                // entry_size,
                heads * owners * max(end - begin for begin, end in windows),
                hopper=hopper,
            )
            state.transient(role, requirements, query.device)

        plans = tuple(
            _plan_for(
                state,
                query,
                key,
                pattern=pattern,
                index_width=index_width,
                scale=scale,
                owners=owners,
                row_start=row_start + begin * _TILE,
                row_count=(end - begin) * _TILE,
            )
            for begin, end in windows
        )
        plan = _Queries(plans, heads, owners, row_count, row_start // _TILE)
        state.plans[cache_key] = plan
        return plan

    indptr_host = torch.empty(counts.numel() + 1, dtype=torch.int32)
    indptr_host[0] = 0
    torch.cumsum(counts, dim=0, out=indptr_host[1:])
    if hopper:
        plan = _token_plan(
            state,
            query,
            indptr_host,
            rows=rows,
            key_rows=key_rows,
            owner_rows=owner_rows,
            row_start=row_start,
            row_count=row_count,
            scale=scale,
        )
        state.plans[cache_key] = plan
        return plan
    indptr = indptr_host.to(query.device)
    role, requirements = _csr_buffers(
        int(indptr_host[-1]), counts.numel(), hopper=False
    )
    # Every launch rewrites its complete index list and key-validity mask.
    # Serialized layers and layouts can borrow the same mutable backing;
    # retaining a mask per layer would multiply quadratic storage by depth.
    maps = (
        {
            name: torch.empty(
                config.shape, dtype=config.dtype, device=query.device
            )
            for name, config in requirements.items()
        }
        if state.transient is None
        else state.transient(role, requirements, query.device)
    )
    indices = maps["indices"]
    indices.zero_()
    packed_mask = maps["mask"]

    workspace_key = query.device
    float_workspace = state.workspaces.get(workspace_key)
    if float_workspace is None:
        float_workspace = torch.empty(
            _FLOAT_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=query.device,
        )
        state.workspaces[workspace_key] = float_workspace

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
        raise RuntimeError(
            "FlashInfer sparse plan did not retain its CSR index buffer"
        )
    bound_mask = wrapper._packed_mask_buf
    mask_offsets = wrapper._mask_indptr_buf
    if bound_mask is None or mask_offsets is None:
        raise RuntimeError(
            "FlashInfer sparse plan did not retain its packed key mask"
        )
    # The execution interface indexes packed-mask bytes. Own these offsets
    # alongside the mutable bits, deriving each query tile from its exact
    # CSR extent. They are formed in 64 bits; query windows keep every one
    # within FlashInfer's int32 mask offsets.
    mask_offsets.copy_(indptr_host.to(torch.int64) * _MASK_BYTES)

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


def _token_plan(
    state: SparseExecutionState,
    query: torch.Tensor,
    indptr_host: torch.Tensor,
    *,
    rows: int,
    key_rows: int,
    owner_rows: int,
    row_start: int,
    row_count: int,
    scale: float,
) -> _TokenPlan:
    """Plan the declared sparse capacity with graph-stable token CSR views.

    This is the single-token page representation used by FlashInfer's
    variable-block sparse attention. Keeping the row capacities on the host
    and generating their live CSR on the device permits dynamic block
    selection and exact padding exclusion without replanning during serving.
    """
    device = query.device
    batch_size = indptr_host.numel() - 1
    index_dtype = torch.int32
    token_indptr = indptr_host * _TILE
    role, requirements = _csr_buffers(
        int(indptr_host[-1]), batch_size, hopper=True
    )
    maps = (
        {
            name: torch.empty(config.shape, dtype=config.dtype, device=device)
            for name, config in requirements.items()
        }
        if state.transient is None
        else state.transient(role, requirements, device)
    )
    indices = maps["indices"]
    indices.zero_()
    qo_indptr = torch.arange(batch_size + 1, dtype=index_dtype) * _TILE
    last_page_len = torch.ones(batch_size, dtype=index_dtype)
    workspace = state.workspaces.get(device)
    if workspace is None:
        workspace = torch.empty(
            _FLOAT_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        state.workspaces[device] = workspace

    wrapper = _flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr.to(device),
        paged_kv_indptr_buf=maps["indptr"],
        paged_kv_indices_buf=indices,
        paged_kv_last_page_len_buf=last_page_len.to(device),
        backend="fa3",
    )
    wrapper.plan(
        qo_indptr,
        token_indptr,
        indices,
        last_page_len,
        1,
        1,
        query.shape[-1],
        1,
        q_data_type=query.dtype,
        kv_data_type=query.dtype,
        o_data_type=query.dtype,
        sm_scale=scale,
    )
    # FA3's persistent scheduler caches key offsets and lengths inside its
    # integer workspace, rather than reading the supplied CSR on run. Its
    # PrefillPlanSM90Info serializes nine byte offsets/flags. Each of our
    # 64-row, one-head requests schedules exactly one 128-row query tile;
    # keep that immutable work order and refresh only its live key domain.
    info = wrapper._plan_info
    if len(info) != 9:
        raise RuntimeError("FlashInfer returned an unsupported FA3 plan layout")

    def work_field(index: int) -> torch.Tensor:
        return wrapper._int_workspace_buffer.narrow(
            0, info[index], batch_size * index_dtype.itemsize
        ).view(index_dtype)

    return _TokenPlan(
        wrapper,
        indices,
        maps["indptr"],
        maps["counts"],
        work_field(7),
        work_field(2),
        work_field(4),
        rows // _TILE,
        key_rows // _TILE,
        owner_rows // _TILE,
        row_count // _TILE,
        row_start // _TILE,
    )


def _native_plan_for(
    state: SparseExecutionState,
    source_indices: torch.Tensor,
    *,
    owners: int,
    row_start: int,
    row_count: int,
) -> _NativeSparsePlan:
    """Cache query mapping and storage.

    Cache query mapping and storage; counts and selected keys remain
    device mutable.
    """
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

    # Same owner-local tile mapping as _plan_for, kept on device for
    # index_select into the live head-wise block maps.
    owner_tiles = total_tiles // owners
    interval_tiles = row_count // _TILE
    local_tiles = torch.arange(owners * interval_tiles)
    query_tiles = (
        local_tiles // interval_tiles * owner_tiles
        + row_start // _TILE
        + local_tiles % interval_tiles
    )

    # Each call selects its live maps into these right before its launch,
    # which the stream orders before any later call rewrites them, so every
    # plan of a context can share them.
    requirements = {
        "indices": BufferConfig(
            (1, heads, owners * interval_tiles, index_width), torch.int32
        ),
        "counts": BufferConfig(
            (1, heads, owners * interval_tiles), torch.int32
        ),
    }
    device = source_indices.device
    maps = (
        {
            name: torch.empty(config.shape, dtype=config.dtype, device=device)
            for name, config in requirements.items()
        }
        if state.transient is None
        else state.transient("vsa_native_rows", requirements, device)
    )
    plan = _NativeSparsePlan(
        query_tiles.to(device), maps["indices"], maps["counts"]
    )
    state.native_plans[cache_key] = plan
    return plan


def _fill_flattened_bsr(
    plan: _SparsePlan | _TokenPlan,
    source_indices: torch.Tensor,
    source_counts: torch.Tensor,
    valid_sizes: torch.Tensor,
) -> None:
    """Populate a cached plan's indices.

    Populate a cached plan's indices from the current device-resident head
    maps.
    """
    assert triton is not None
    heads = int(source_indices.shape[0])
    if isinstance(plan, _TokenPlan):
        arguments = (
            int(source_indices.stride(0)),
            int(source_indices.stride(1)),
            int(source_counts.stride(0)),
            plan.query_tiles,
        )
        mapping = (
            plan.owner_tiles,
            plan.interval_tiles,
            plan.start_tile,
            int(source_indices.shape[2]),
        )
        _count_sparse_tokens_kernel[(heads * plan.query_tiles,)](
            source_indices,
            source_counts,
            valid_sizes,
            plan.counts,
            *arguments,
            *mapping,
            triton.next_power_of_2(source_indices.shape[2]),
            num_warps=4,
        )
        plan.indptr[:1].zero_()
        torch.cumsum(
            plan.counts, 0, dtype=plan.indptr.dtype, out=plan.indptr[1:]
        )
        _fill_sparse_tokens_kernel[(heads * plan.query_tiles,)](
            source_indices,
            source_counts,
            valid_sizes,
            plan.indptr,
            plan.indices,
            *arguments,
            plan.key_tiles,
            *mapping,
            num_warps=4,
        )
        _refresh_sparse_work_kernel[(triton.cdiv(plan.counts.numel(), 256),)](
            plan.work_batches,
            plan.indptr,
            plan.counts,
            plan.work_starts,
            plan.work_counts,
            plan.counts.numel(),
            256,
            num_warps=4,
        )
        return
    wrapper_indptr = getattr(plan.wrapper, "_paged_kv_indptr_buf", None)
    if wrapper_indptr is None:
        raise RuntimeError(
            "FlashInfer sparse plan has no CSR row pointer buffer"
        )
    _fill_flattened_bsr_kernel[(heads * plan.query_tiles,)](
        source_indices,
        source_counts,
        valid_sizes,
        wrapper_indptr,
        plan.indices,
        plan.packed_mask,
        int(source_indices.stride(0)),
        int(source_indices.stride(1)),
        int(source_counts.stride(0)),
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


def run_sparse(
    plan: _SparsePlan | _TokenPlan | _Queries,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    valid_sizes: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a prepared sparse plan with its live device key domain."""
    if isinstance(plan, _Queries):
        output = torch.empty_like(query) if out is None else out
        width = query.shape[-1]
        queries = query.view(plan.heads, plan.owners, plan.rows, width)
        outputs = output.view(plan.heads, plan.owners, plan.rows, width)
        for window in plan.plans:
            begin = (window.start_tile - plan.start_tile) * _TILE
            count = window.interval_tiles * _TILE
            packed = queries[:, :, begin : begin + count].contiguous()
            # Plans borrow the same mutable CSR. Populate and consume each
            # window on the serialized stream before the next rewrites it.
            result = run_sparse(
                window,
                packed.view(-1, 1, width),
                key,
                value,
                block_indices=block_indices,
                block_counts=block_counts,
                valid_sizes=valid_sizes,
            )
            outputs[:, :, begin : begin + count].copy_(
                result.view(plan.heads, plan.owners, count, width)
            )
        return output

    _fill_flattened_bsr(plan, block_indices, block_counts, valid_sizes)
    if isinstance(plan, _TokenPlan):
        return plan.wrapper.run(
            query,
            (
                key.view(-1, 1, 1, key.shape[-1]),
                value.view(-1, 1, 1, value.shape[-1]),
            ),
            out=out,
        )
    return plan.wrapper.run(query, key, value, out=out)


def dense_prefix(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_sizes: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Attend Hopper prefix rows to every valid key, returning row-major rows.

    Inputs have head-first layouts [heads, rows, width]. Each key tile has
    64 rows; its borrowed int32 validity count remains live on graph replay.
    The score predicate preserves partial and empty tiles while allowing
    FlashAttention-4's SM90 TMA/WGMMA implementation.
    """
    from uniserve_kernels.attention.block_validity import block_validity

    from ..flash_attn_4 import _flash_attn_forward

    # Preserve the vector's inner stride and alignment when CuTe converts
    # the auxiliary tensor. This changes metadata, not the borrowed counts.
    valid_sizes.__leading_dim__ = 0
    valid_sizes.__assumed_align__ = 4
    return _flash_attn_forward()(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        softmax_scale=scale,
        causal=False,
        mask_mod=block_validity,
        aux_tensors=[valid_sizes],
    )[0][0]


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
) -> ChunkProducer:
    """Prepare full K/V once and produce paired owner query intervals on demand.

    All owners share one selected key domain. Queries are packed in transport
    interval order. On SM120/SM121 the native block-sparse kernel evaluates
    every query tile from its block map, whose dense prefix tiles list the
    complete valid key domain. Elsewhere globally visible prefixes use dense
    prefill and video queries retain their selected block maps. The producer
    writes complete local-head vectors into contiguous row-owner views;
    communication remains caller-owned.
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
            "sparse row production requires equal QKV and tile-aligned "
            "owner intervals"
        )

    rows, heads, width = query.shape
    owner_rows = rows // owners

    # SM120's native kernel consumes row-major [rows, heads, dim] inputs;
    # every other device goes through the head-flattened BSR path.
    native_rows = uses_row_major_inputs(query.device)
    if native_rows:
        from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
            bsa_attn_sm120_blk64_fwd,
        )

    packed_shape = (
        (3, rows, heads, width) if native_rows else (3, heads, rows, width)
    )
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
        raise ValueError(
            "prepared sparse inputs must match the complete query shape"
        )

    packed_key = packed[1].transpose(0, 1) if native_rows else packed[1]
    packed_value = packed[2].transpose(0, 1) if native_rows else packed[2]

    prefix_rows = pattern.dense_prefix_tiles * _TILE
    valid_tiles = pattern.dense_key_tiles
    key_mask = None
    hopper = torch.cuda.get_device_capability(query.device)[0] == 9
    if valid_tiles and not native_rows and not hopper:
        key_mask = torch.empty(
            valid_tiles * (_TILE // 8),
            dtype=torch.uint8,
            device=query.device,
        )
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
                mask_block_count,
                1,
                native_plan.query_tiles,
                out=native_plan.counts[0],
            )

            output = attention_output.view(-1)[:elements].view(
                1, members * count, heads, width
            )
            bsa_attn_sm120_blk64_fwd(
                packed_query.transpose(0, 1).unsqueeze(0),
                packed[1].unsqueeze(0),
                packed[2].unsqueeze(0),
                native_plan.indices,
                selected_tiles,
                block_sizes=valid_sizes,
                q2k_block_nums=native_plan.counts,
                softmax_scale=scale,
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
        output = attention_output.view(-1)[:elements].view(
            heads * members * count, 1, width
        )
        run_sparse(
            plan,
            packed_query.reshape(heads * members * count, 1, width),
            packed_key.view(heads * rows, 1, width),
            packed_value.view(heads * rows, 1, width),
            block_indices=mask_block_indices,
            block_counts=mask_block_count,
            valid_sizes=valid_sizes,
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
            raise ValueError(
                "sparse row destinations must match the prepared owner interval"
            )

        elements = heads * owners * count * width
        packed_query = (
            packed[0]
            .view(-1)
            .narrow(0, start * owners * heads * width, elements)
        )
        packed_query = (
            packed_query.view(owners * count, heads, width).transpose(0, 1)
            if native_rows
            else packed_query.view(heads, owners * count, width)
        )

        # The native kernel reads each prefix tile's complete key list from
        # the block map and masks partial key tiles by their valid sizes. On
        # an RTX PRO 6000 at the 10 s / 10K H3 layout (7 heads per rank) it
        # attends the prefix rows in 10.5 ms, against 32.9 ms for the dense
        # prefill under the key-validity mask, so it serves every interval.
        if native_rows or start >= prefix_rows:
            produce_sparse(packed_query, outputs, start, count, owners)
            return

        # Prefix queries see the complete valid key domain. Their dense
        # kernel avoids one-head page traversal; video queries retain their
        # selected key domain. The communication interval and caller-owned
        # destinations stay unchanged, including an interval spanning the
        # prefix/video boundary.
        for owner, destination in enumerate(outputs):
            global_start = owner * owner_rows + start
            dense_rows = max(0, min(count, prefix_rows - global_start))
            owner_query = packed_query[:, owner * count : (owner + 1) * count]

            if dense_rows:
                if hopper:
                    attended = dense_prefix(
                        owner_query[:, :dense_rows],
                        packed_key[:, : valid_tiles * _TILE],
                        packed_value[:, : valid_tiles * _TILE],
                        valid_sizes,
                        scale=scale,
                    )
                else:
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
