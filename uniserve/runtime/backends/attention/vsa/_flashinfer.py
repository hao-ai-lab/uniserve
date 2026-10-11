"""FlashInfer provider for mutable head-wise block-64 video sparse attention.

Each query tile of every head selects its key tiles live: the plan flattens
the heads into one block-sparse (BSR) matrix whose block indices and
per-key validity mask are rewritten on the device at every launch, so
partial and empty tiles attend only their valid keys.

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
from uniserve_kernels.triton import launchable

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
    start_tile: int


@dataclass
class SparseExecutionState:
    """Own mutable plans and scratch.

    Own mutable plans and scratch for one serialized sparse execution
    domain.
    """

    plans: dict[tuple[Any, ...], _SparsePlan] = field(default_factory=dict)
    workspaces: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    # Lends the owning context's shared per-call work areas (see
    # ``ExecutionContext.scratch``), or ``None`` to allocate them per plan.
    transient: Callable[..., Mapping[str, torch.Tensor]] | None = None


if triton is not None:

    @triton.jit
    def _fill_flattened_bsr_kernel(
        source_indices,
        source_counts,
        variable_block_sizes,
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
        start_tile: tl.constexpr,
    ):
        """Populate head-flattened indices and key validity bits.

        Populate head-flattened indices and exact little-endian key
        validity bits. The plan reserves each row's declared count bound;
        entries past the row's live count in ``source_counts`` address key
        tile 0 with every validity bit clear, so they contribute nothing.
        """
        # Each program owns one (head, query tile) row of the flattened BSR.
        row = tl.program_id(0)
        head = row // query_tiles
        query_tile = start_tile + row % query_tiles
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
        # FlashInfer BSR mask marks exactly variable_block_sizes[tile] keys
        # per tile.
        bytes_per_tile_row = tile_rows // 8
        mask_bytes = selected_count * tile_rows * bytes_per_tile_row
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
                variable_block_sizes + selected_blocks, mask=live, other=0
            )
            bits = tl.minimum(
                tl.maximum(
                    valid_rows - (byte_offsets % bytes_per_tile_row) * 8, 0
                ),
                8,
            )
            tl.store(
                packed_mask
                + destination * tile_rows * bytes_per_tile_row
                + byte_offsets,
                ((1 << bits) - 1).to(tl.uint8),
                mask=active,
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
    # FIXME: the head-flattened BSR path is validated on SM100 only, where
    # it runs under test; other architectures resolve to another provider.
    if torch.cuda.get_device_capability(selected)[0] != 10:
        return False
    return launchable(selected)


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
) -> _SparsePlan:
    """Plan head-flattened BSR from the caller's declared row cardinalities."""
    rows, heads, width = (int(size) for size in query.shape)
    row_count = rows
    key_rows = int(key.shape[0])
    query_tiles, key_tiles = row_count // _TILE, key_rows // _TILE

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

    counts = pattern.counts(
        num_heads=heads,
        query_tiles=rows // _TILE,
        index_width=index_width,
        out=torch.empty(
            (heads, rows // _TILE), device="cpu", dtype=torch.int32
        ),
    )
    counts = counts.reshape(-1)
    indptr_host = torch.empty(counts.numel() + 1, dtype=torch.int32)
    indptr_host[0] = 0
    torch.cumsum(counts, dim=0, out=indptr_host[1:])
    indptr = indptr_host.to(query.device)
    index_count = int(indptr_host[-1])
    requirements = {
        "indices": BufferConfig((index_count,), torch.int32),
        "mask": BufferConfig(
            (index_count * _TILE * (_TILE // 8),), torch.uint8
        ),
    }
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
        else state.transient("vsa_bsr_rows", requirements, query.device)
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
        heads * row_count,
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
    # CSR extent.
    mask_offsets.copy_((indptr_host * (_TILE * _TILE // 8)).to(query.device))

    plan = _SparsePlan(
        wrapper, bound_indices, bound_mask, query_tiles, key_tiles, 0
    )
    state.plans[cache_key] = plan
    return plan


def _fill_flattened_bsr(
    plan: _SparsePlan,
    source_indices: torch.Tensor,
    source_counts: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> None:
    """Populate a cached plan's indices.

    Populate a cached plan's indices from the current device-resident head
    maps.
    """
    assert triton is not None
    heads = int(source_indices.shape[0])
    wrapper_indptr = getattr(plan.wrapper, "_paged_kv_indptr_buf", None)
    if wrapper_indptr is None:
        raise RuntimeError(
            "FlashInfer sparse plan has no CSR row pointer buffer"
        )
    _fill_flattened_bsr_kernel[(heads * plan.query_tiles,)](
        source_indices,
        source_counts,
        variable_block_sizes,
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
        plan.start_tile,
        num_warps=4,
        num_stages=1,
    )


def run_sparse(
    plan: _SparsePlan,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a prepared sparse plan with its live device key domain."""
    _fill_flattened_bsr(plan, q2k_index, q2k_num, variable_block_sizes)
    return plan.wrapper.run(query, key, value, out=out)
