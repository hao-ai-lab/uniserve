"""Shared capability probes for paged denoise attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

import uniserve_worker.ops as ops

from ..contracts.forward_context import get_forward_context
from ..nn.attention import RadixAttention
from ..runtime.paged_text_cache import BatchedPagedTextCache, PagedTextCache

__all__ = [
    "PagedDenoiseBranchSet",
    "batched_paged_denoise_cache",
    "can_run_paged_denoise_attention",
]


def _branch_key(branch: Any) -> str:
    return str(getattr(branch, "value", branch))


def batched_paged_denoise_cache(
    caches: Sequence[PagedTextCache],
) -> BatchedPagedTextCache:
    """Construct the system-owned batched view for compatible denoise rows."""

    return BatchedPagedTextCache(list(caches))


@dataclass
class PagedDenoiseBranchSet:
    """Scratch-paged CFG branch prefixes plus batched-row cache reuse."""

    caches: Mapping[str, PagedTextCache]
    positions: Mapping[str, int]
    _batched: dict[tuple[str, ...], BatchedPagedTextCache] = field(default_factory=dict)

    def has_all(self, branches: Sequence[Any]) -> bool:
        return all(_branch_key(branch) in self.caches for branch in branches)

    def batched_cache(self, branches: Sequence[Any]) -> BatchedPagedTextCache:
        key = tuple(_branch_key(branch) for branch in branches)
        batched = self._batched.get(key)
        if batched is None:
            batched = batched_paged_denoise_cache([self.caches[branch] for branch in key])
            self._batched[key] = batched
        return batched

    def positions_tensor(
        self,
        branches: Sequence[Any],
        *,
        device: torch.device | str,
        width: int,
    ) -> torch.Tensor:
        key = tuple(_branch_key(branch) for branch in branches)
        return (
            torch.tensor(
                [int(self.positions[branch]) for branch in key],
                dtype=torch.long,
                device=device,
            )
            .unsqueeze(1)
            .expand(len(key), int(width))
        )

    def release(self, residency: Any) -> None:
        seen: set[int] = set()
        for cache in self.caches.values():
            cache_id = id(cache)
            if cache_id in seen:
                continue
            seen.add(cache_id)
            residency.release_scratch_cache(cache)
        self._batched.clear()


def can_run_paged_denoise_attention(
    cache: Any,
    *,
    prototype: torch.Tensor,
    query_width: int | None = None,
    query_tokens: int | None = None,
    batch_size: int | None = None,
    attention_backend: str | None = None,
) -> bool:
    """Return whether the active backend can run transient paged-varlen denoise.

    ``ops.can_run_attention`` is a pure capability probe. Unsupported
    configurations return ``False``; provider exceptions are intentionally not
    swallowed because they indicate a backend bug, not an expected fallback.
    """

    if prototype.device.type != "cuda" or not prototype.is_cuda or prototype.ndim < 3:
        return False
    pool = getattr(cache, "pool", None)
    block_size = int(getattr(pool, "block_size", 0) or 0)
    if pool is None or block_size <= 0:
        return False
    if not bool(getattr(pool, "supports_paged_attention_storage", True)):
        return False
    pool_k = getattr(pool, "k", None)
    if not isinstance(pool_k, torch.Tensor) or not pool_k.is_cuda:
        return False

    request_cache = getattr(cache, "request_cache_for_transient", None)
    if not callable(request_cache):
        return False
    n_tokens = int(query_tokens if query_tokens is not None else prototype.shape[-2])
    if n_tokens <= 1:
        return False
    view = request_cache(0, n_tokens)
    batch = int(batch_size if batch_size is not None else prototype.shape[0])
    if batch <= 0:
        return False
    width = int(
        query_width
        if query_width is not None
        else getattr(pool, "head_dim", 0) or prototype.shape[-1]
    )
    q_shape_probe = prototype.new_empty((batch, 1, n_tokens, width))
    metadata = RadixAttention._transient_varlen_metadata(view, q_shape_probe)
    if metadata is None:
        return False
    _query_lens, block_table, _cache_seqlens, cu_q, cu_k, max_q, max_k = metadata

    ctx = get_forward_context()
    preferred = ctx.attention_backend_name or attention_backend or "auto"
    probe = prototype.new_empty((1, 1, width))
    return ops.can_run_attention(
        probe,
        probe,
        probe,
        regime=ops.AttentionRegime.EXTEND,
        causal=True,
        scale=1.0,
        ctx=ctx,
        kv_cache=view,
        block_table=block_table,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        override=preferred,
    )
