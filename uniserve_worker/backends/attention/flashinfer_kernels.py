"""FlashInfer paged-KV plan-tensor kernels and index gathering."""
from __future__ import annotations

from typing import Any

import torch

from ...contracts.forward_context import get_forward_context
from ...foundation.env import env_optional_flag
from ...foundation.triton_compat import triton_device_supported, triton_fused_layers_enabled
from ..paged_kv_math import decode_write_locations, paged_kv_write
from .flashinfer_plan import _DecodePlanWorkspace, _PrefillPlanWorkspace

try:  # pragma: no cover - optional Triton runtime.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

_TRITON_DECODE_INDICES_ENV = "UNISERVE_FLASHINFER_TRITON_DECODE_INDICES"
# Triton decode-indices kernel block tile (kernel-ABI tuning constant); this is
# unrelated to the paged-KV block size in ...core.sizing.
_TRITON_DECODE_INDICES_BLOCK = 256


if triton is not None:

    @triton.jit
    def _fill_paged_decode_indices_kernel(
        block_table: torch.Tensor,
        page_counts: torch.Tensor,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        stride_row: tl.constexpr,
        stride_col: tl.constexpr,
        block_size: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        page_count = tl.load(page_counts + row)
        base = tl.load(indptr + row)
        offsets = tl.arange(0, block_size)
        num_loop = tl.cdiv(page_count, block_size)
        for i in range(num_loop):
            columns = i * block_size + offsets
            mask = columns < page_count
            values = tl.load(
                block_table + row * stride_row + columns * stride_col,
                mask=mask,
                other=0,
            )
            tl.store(indices + base + columns, values, mask=mask)


def _paged_decode_indices(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_counts = torch.div(
        seq_lens + int(page_size) - 1,
        int(page_size),
        rounding_mode="floor",
    ).to(torch.int32)
    indptr = torch.empty((int(seq_lens.shape[0]) + 1,), dtype=torch.int32, device=seq_lens.device)
    indptr[0] = 0
    indptr[1:] = torch.cumsum(page_counts, dim=0)
    columns = torch.arange(int(block_table.shape[1]), device=block_table.device)
    mask = columns.unsqueeze(0) < page_counts.to(torch.int64).unsqueeze(1)
    indices = block_table[mask].to(dtype=torch.int32).contiguous()
    last_page_len = (torch.remainder(seq_lens - 1, int(page_size)) + 1).to(torch.int32).contiguous()
    return indptr, indices, last_page_len


def _fill_paged_kv_plan(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    *,
    batch_size: int,
    kv_indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    page_counts: torch.Tensor,
) -> None:
    """Fill the shared paged-KV plan tensors in place from ``seq_lens``.

    Computes per-row ``page_counts = cdiv(seq_lens, page_size)``, the KV
    ``indptr`` via cumsum, the ``last_page_len`` remainder, and launches the
    index-gather kernel. ``kv_indptr``/``page_counts``/``last_page_len`` are the
    full workspace buffers, sliced to ``batch_size`` here; ``indices`` is the
    full workspace buffer passed straight to the kernel. The decode and prefill
    wrappers share this core; only the prefill wrapper additionally copies
    ``cu_seqlens_q`` into its ``qo_indptr``.
    """
    page_size = max(1, int(page_size))
    page_counts = page_counts[:batch_size]
    page_counts.copy_(seq_lens, non_blocking=True)
    torch.add(page_counts, page_size - 1, out=page_counts)
    torch.div(page_counts, page_size, rounding_mode="floor", out=page_counts)

    indptr = kv_indptr[: batch_size + 1]
    indptr[:1].zero_()
    torch.cumsum(page_counts, dim=0, out=indptr[1:])

    last_page_len = last_page_len[:batch_size]
    last_page_len.copy_(seq_lens, non_blocking=True)
    last_page_len.sub_(1)
    last_page_len.remainder_(page_size)
    last_page_len.add_(1)

    assert triton is not None
    assert _fill_paged_decode_indices_kernel is not None
    grid = (batch_size,)
    _fill_paged_decode_indices_kernel[grid](
        block_table,
        page_counts,
        indptr,
        indices,
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        block_size=_TRITON_DECODE_INDICES_BLOCK,
    )


def _fill_paged_decode_plan_tensors(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    workspace: _DecodePlanWorkspace,
) -> bool:
    if not _triton_decode_indices_enabled(block_table.device):
        return False
    batch_size = int(seq_lens.shape[0])
    width = int(block_table.shape[1])
    if batch_size <= 0 or width <= 0:
        return False
    if int(workspace.indptr.numel()) < batch_size + 1:
        return False
    if int(workspace.indices.numel()) < int(block_table.numel()):
        return False
    if int(workspace.last_page_len.numel()) < batch_size:
        return False
    if int(workspace.page_counts.numel()) < batch_size:
        return False

    _fill_paged_kv_plan(
        block_table,
        seq_lens,
        page_size,
        batch_size=batch_size,
        kv_indptr=workspace.indptr,
        indices=workspace.indices,
        last_page_len=workspace.last_page_len,
        page_counts=workspace.page_counts,
    )
    return True


def _fill_paged_prefill_plan_tensors(
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_seqlens: torch.Tensor,
    page_size: int,
    workspace: _PrefillPlanWorkspace,
) -> bool:
    if not _triton_decode_indices_enabled(block_table.device):
        return False
    batch_size = int(kv_seqlens.shape[0])
    width = int(block_table.shape[1])
    if batch_size <= 0 or width <= 0:
        return False
    if int(workspace.qo_indptr.numel()) < batch_size + 1:
        return False
    if int(workspace.kv_indptr.numel()) < batch_size + 1:
        return False
    if int(workspace.indices.numel()) < int(block_table.numel()):
        return False
    if int(workspace.last_page_len.numel()) < batch_size:
        return False
    if int(workspace.page_counts.numel()) < batch_size:
        return False

    workspace.qo_indptr[: batch_size + 1].copy_(cu_seqlens_q[: batch_size + 1], non_blocking=True)
    _fill_paged_kv_plan(
        block_table,
        kv_seqlens,
        page_size,
        batch_size=batch_size,
        kv_indptr=workspace.kv_indptr,
        indices=workspace.indices,
        last_page_len=workspace.last_page_len,
        page_counts=workspace.page_counts,
    )
    return True


def _triton_decode_indices_enabled(device: torch.device | str) -> bool:
    if triton is None:
        return False
    if env_optional_flag(_TRITON_DECODE_INDICES_ENV) is False:
        return False
    return triton_fused_layers_enabled() and triton_device_supported(device)


_USE_FORWARD_CONTEXT_PLAN = object()


def _write_decode_token(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    plan: Any = _USE_FORWARD_CONTEXT_PLAN,
) -> None:
    page_ids, offsets = _decode_write_locations_from_context(
        block_table,
        cache_seqlens,
        int(k_cache.shape[1]),
        batch_size=int(k_current.shape[0]),
        device=k_cache.device,
        plan=plan,
    )
    paged_kv_write(k_cache, v_cache, page_ids, offsets, k_current, v_current, cast=True)


def _decode_effective_seqlens(
    cache_seqlens: torch.Tensor,
    current_tokens: int,
    plan: Any,
) -> torch.Tensor:
    """Return once-per-step post-append lengths for paged decode."""

    current_tokens = int(current_tokens)
    if current_tokens == 0:
        return cache_seqlens
    shared = getattr(plan, "kv_seqlens", None)
    if (
        current_tokens == 1
        and isinstance(shared, torch.Tensor)
        and shared.shape == cache_seqlens.shape
        and shared.device == cache_seqlens.device
        and shared.dtype == cache_seqlens.dtype
        and shared.is_contiguous()
    ):
        return shared
    return cache_seqlens + current_tokens


def _decode_write_locations_from_context(
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    *,
    batch_size: int,
    device: torch.device,
    plan: Any = _USE_FORWARD_CONTEXT_PLAN,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Callers on the forward path pass ``plan`` explicitly; the sentinel
    # default falls back to the active forward context so direct callers keep
    # the original global-reach behavior.
    if plan is _USE_FORWARD_CONTEXT_PLAN:
        plan = getattr(get_forward_context(), "attention_plan", None)
    page_ids = getattr(plan, "decode_page_ids", None)
    offsets = getattr(plan, "decode_page_offsets", None)
    if (
        isinstance(page_ids, torch.Tensor)
        and isinstance(offsets, torch.Tensor)
        and int(page_ids.shape[0]) >= int(batch_size)
        and int(offsets.shape[0]) >= int(batch_size)
        and page_ids.device == device
        and offsets.device == device
    ):
        return (
            page_ids[:batch_size].to(dtype=torch.int64),
            offsets[:batch_size].to(dtype=torch.int64),
        )
    return decode_write_locations(block_table, cache_seqlens, page_size)
