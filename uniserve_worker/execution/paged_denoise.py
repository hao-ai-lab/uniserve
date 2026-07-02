"""Shared capability probes for paged denoise attention."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

import uniserve_worker.ops as ops
from ..contracts.forward_context import get_forward_context
from ..runtime.paged_text_cache import BatchedPagedTextCache, PagedTextCache

__all__ = ["PagedDenoiseBranchSet", "can_run_paged_denoise_attention"]


@dataclass
class PagedDenoiseBranchSet:
    """Scratch-paged CFG branch prefixes plus batched-row cache reuse."""

    caches: Mapping[str, PagedTextCache]
    positions: Mapping[str, int]
    _batched: dict[tuple[str, ...], BatchedPagedTextCache] = field(default_factory=dict)

    def has_all(self, branches: Sequence[str]) -> bool:
        return all(str(branch) in self.caches for branch in branches)

    def batched_cache(self, branches: Sequence[str]) -> BatchedPagedTextCache:
        key = tuple(str(branch) for branch in branches)
        batched = self._batched.get(key)
        if batched is None:
            batched = BatchedPagedTextCache([self.caches[branch] for branch in key])
            self._batched[key] = batched
        return batched

    def positions_tensor(
        self,
        branches: Sequence[str],
        *,
        device: torch.device | str,
        width: int,
    ) -> torch.Tensor:
        key = tuple(str(branch) for branch in branches)
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
    attention_backend: str | None = None,
) -> bool:
    """Return whether the active backend can run decode-shaped paged denoise.

    ``ops.can_run_attention`` is a pure capability probe. Unsupported
    configurations return ``False``; provider exceptions are intentionally not
    swallowed because they indicate a backend bug, not an expected fallback.
    """

    if prototype.device.type != "cuda" or not prototype.is_cuda:
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

    ctx = get_forward_context()
    preferred = ctx.attention_backend_name or attention_backend or "auto"
    width = int(query_width if query_width is not None else prototype.shape[-1])
    probe = prototype.new_empty((1, 1, 1, width))
    return ops.can_run_attention(
        probe,
        probe,
        probe,
        regime=ops.AttentionRegime.DECODE,
        causal=True,
        scale=1.0,
        ctx=ctx,
        kv_cache=cache,
        block_table=prototype.new_empty((1, 1), dtype=torch.int32),
        cache_seqlens=prototype.new_empty((1,), dtype=torch.int32),
        override=preferred,
    )
