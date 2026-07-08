"""Text KV residency and attention-plan construction."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from ..backends.paged_kv_math import decode_write_locations
from ..contracts.forward_context import AttentionCache, TextAttentionMetadata
from ..contracts.forward_mode import ForwardMode
from .kv_pool import PagedKVPool
from .paged_text_cache import BatchedPagedRequestCache
from .request_session import PreparedTextRow

__all__ = [
    "TextAttentionPlan",
    "TextKvResidency",
]


@dataclass
class TextAttentionPlan:
    """Per-forward text attention residency snapshot."""

    cache: BatchedPagedRequestCache
    mode: ForwardMode
    device: torch.device
    query_lens_cpu: tuple[int, ...]
    max_context_len: int = 0
    stager: Any | None = None
    block_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None
    cache_seqlens_cpu: tuple[int, ...] = ()
    kv_seqlens_cpu: tuple[int, ...] = ()
    query_lens: torch.Tensor | None = None
    kv_seqlens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    decode_page_ids: torch.Tensor | None = None
    decode_page_offsets: torch.Tensor | None = None

    @classmethod
    def from_cache(
        cls,
        cache: BatchedPagedRequestCache,
        *,
        mode: ForwardMode,
        device: torch.device | str,
        query_lens_cpu: Sequence[int],
        stager: Any | None = None,
        max_context_len: int = 0,
    ) -> "TextAttentionPlan":
        target = torch.device(device)
        query_lens_values = tuple(int(length) for length in query_lens_cpu)
        cache_seqlens_cpu = tuple(int(length) for length in cache.base_lens)
        cache_seqlens = cache.cache_seqlens(device=target, stager=stager)
        block_table = cache.block_table(device=target, stager=stager)
        kv_lens_values = tuple(
            int(base_len) + int(query_len)
            for base_len, query_len in zip(cache.base_lens, query_lens_values)
        )
        plan = cls(
            cache=cache,
            mode=mode,
            device=target,
            query_lens_cpu=query_lens_values,
            max_context_len=max(0, int(max_context_len)),
            stager=stager,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cache_seqlens_cpu=cache_seqlens_cpu,
            kv_seqlens_cpu=kv_lens_values,
        )
        if mode == ForwardMode.DECODE:
            page_ids, page_offsets = decode_write_locations(
                block_table,
                cache_seqlens,
                cache.pool.block_size,
            )
            plan.decode_page_ids = page_ids
            plan.decode_page_offsets = page_offsets
            return plan
        query_lens = torch.tensor(query_lens_values, dtype=torch.int32, device=target)
        kv_seqlens = torch.tensor(kv_lens_values, dtype=torch.int32, device=target)
        zero = torch.zeros(1, dtype=torch.int32, device=target)
        plan.query_lens = query_lens
        plan.kv_seqlens = kv_seqlens
        plan.cu_seqlens_q = torch.cat([zero, torch.cumsum(query_lens, dim=0).to(torch.int32)])
        plan.cu_seqlens_k = torch.cat([zero, torch.cumsum(kv_seqlens, dim=0).to(torch.int32)])
        return plan

    def to_metadata(self) -> TextAttentionMetadata:
        return TextAttentionMetadata(
            cache=cast(AttentionCache, self.cache),
            block_table=self.block_table,
            cache_seqlens=self.cache_seqlens,
            cache_seqlens_cpu=self.cache_seqlens_cpu,
            query_lens=self.query_lens,
            query_lens_cpu=self.query_lens_cpu,
            kv_seqlens=self.kv_seqlens,
            kv_seqlens_cpu=self.kv_seqlens_cpu,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k=self.cu_seqlens_k,
            decode_page_ids=self.decode_page_ids,
            decode_page_offsets=self.decode_page_offsets,
            max_seqlen_q=max(self.query_lens_cpu, default=0),
            max_seqlen_k=max(self.kv_seqlens_cpu, default=0),
            max_context_len=int(self.max_context_len),
            mode=self.mode,
        )

    def attach_to(self, batch: Any) -> TextAttentionMetadata:
        metadata = self.to_metadata()
        batch.attn_metadata = metadata
        batch.block_table = metadata.block_table
        batch.cache_seqlens = metadata.cache_seqlens
        return metadata


class TextKvResidency:
    """Builds batched and single-row text KV attention plans."""

    def __init__(self, kv_pool: PagedKVPool, *, max_context_len: int = 0) -> None:
        self.kv_pool = kv_pool
        self.max_context_len = max(0, int(max_context_len))

    def open_batch(
        self,
        rows: Sequence[PreparedTextRow],
        mode: ForwardMode,
        device: torch.device | str,
        stager: Any | None = None,
    ) -> TextAttentionPlan:
        cache = BatchedPagedRequestCache(
            self.kv_pool,
            [row.block_ids for row in rows],
            [row.base_len for row in rows],
        )
        return TextAttentionPlan.from_cache(
            cache,
            mode=mode,
            device=device,
            query_lens_cpu=[row.query_len for row in rows],
            stager=stager,
            max_context_len=self.max_context_len,
        )

    def open_from_block_ids(
        self,
        block_ids_by_row: Sequence[Sequence[int]],
        base_lens: Sequence[int],
        query_lens: Sequence[int],
        *,
        mode: ForwardMode,
        device: torch.device | str,
        stager: Any | None = None,
    ) -> TextAttentionPlan:
        cache = BatchedPagedRequestCache(
            self.kv_pool,
            [list(ids) for ids in block_ids_by_row],
            [int(length) for length in base_lens],
        )
        return TextAttentionPlan.from_cache(
            cache,
            mode=mode,
            device=device,
            query_lens_cpu=[int(length) for length in query_lens],
            stager=stager,
            max_context_len=self.max_context_len,
        )

    def cache_from_block_ids(
        self,
        block_ids_by_row: Sequence[Sequence[int]],
        base_lens: Sequence[int],
    ) -> BatchedPagedRequestCache:
        return BatchedPagedRequestCache(
            self.kv_pool,
            [list(ids) for ids in block_ids_by_row],
            [int(length) for length in base_lens],
        )
