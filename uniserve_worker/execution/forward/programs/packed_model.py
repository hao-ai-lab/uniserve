"""Model-owner support for packed visible text and denoise execution."""

from __future__ import annotations

from typing import Any

import torch

from ....contracts.forward_mode import ForwardMode
from ....foundation.sizing import DEFAULT_BLOCK_SIZE, ceil_div
from ....runtime.kv_pool import PagedKVPool
from ....runtime.paged_text_cache import (
    PagedTextCache,
    PagedTextCacheSpanCopy,
    copy_paged_text_cache_span,
)
from ...denoise_driver import TextImageDenoiseStep, text_image_branches
from ..stream import ForwardPagedKVSegment, ForwardStreamBuilder

__all__ = ["PackedVisibleModelMixin"]


class PackedVisibleModelMixin:
    """Shared cache staging and segment construction for packed models."""

    def _extend_cache_blocks(self, cache: Any, op: dict[str, Any]) -> None:
        self._text_driver().extend_cache_blocks(cache, op)

    def _ensure_host_cache(self, cache: Any) -> None:
        self._text_driver().ensure_host_cache(cache)

    @staticmethod
    def _same_kv_pool(pool: Any, first_pool: Any) -> bool:
        return first_pool is None or pool is first_pool

    def _stage_text_cache_for_forward(
        self,
        source: PagedTextCache,
        *,
        target_pool: PagedKVPool,
        end_len: int,
        pending_prefix_copies: list[PagedTextCacheSpanCopy] | None = None,
    ) -> PagedTextCache:
        scratch_pools = (
            getattr(self, "scratch_pool", None),
            getattr(self, "gen_scratch_pool", None),
        )
        if not any(target_pool is pool for pool in scratch_pools if pool is not None):
            raise RuntimeError("forward text staging target must be a scratch KV pool")
        allocator = self.residency.require_allocator_for_pool(
            target_pool,
            label="forward scratch KV pool",
        )
        staged_by_pool = getattr(source, "_uniserve_forward_staging", None)
        if not isinstance(staged_by_pool, dict):
            staged_by_pool = {}
            setattr(source, "_uniserve_forward_staging", staged_by_pool)
        key = id(target_pool)
        staged = staged_by_pool.get(key)
        source_len = int(source.length)
        source_prefix = self._forward_staging_source_prefix(source, source_len)
        if (
            staged is not None
            and getattr(staged, "pool", None) is target_pool
            and int(getattr(staged, "length", -1)) <= source_len
            and getattr(staged, "_uniserve_forward_staging_source_prefix", ())
            == self._forward_staging_source_prefix(source, int(getattr(staged, "length", 0)))
        ):
            staged.ensure_capacity(int(end_len))
            copied_len = int(staged.length)
            if copied_len < source_len:
                if pending_prefix_copies is None:
                    copy_paged_text_cache_span(
                        source,
                        staged,
                        start=copied_len,
                        length=source_len - copied_len,
                        num_layers=self.num_layers,
                        missing_message="cannot extend packed forward staging without a paged source cache",
                    )
                    setattr(staged, "_uniserve_forward_staging_source_prefix", source_prefix)
                else:
                    pending_prefix_copies.append(
                        PagedTextCacheSpanCopy(
                            source=source,
                            target=staged,
                            start=copied_len,
                            length=source_len - copied_len,
                        )
                    )
                staged.length = source_len
            if copied_len >= source_len:
                setattr(staged, "_uniserve_forward_staging_source_prefix", source_prefix)
            return staged
        if staged is not None:
            self.residency.release_scratch_cache(staged)
        staged = PagedTextCache(
            target_pool,
            [],
            num_layers=self.num_layers,
            length=0,
            allocate_blocks=allocator,
        )
        staged.ensure_capacity(int(end_len))
        if source_len > 0:
            if pending_prefix_copies is None:
                copy_paged_text_cache_span(
                    source,
                    staged,
                    start=0,
                    length=source_len,
                    num_layers=self.num_layers,
                    missing_message="cannot stage packed forward prefix without a paged source cache",
                )
                setattr(staged, "_uniserve_forward_staging_source_prefix", source_prefix)
            else:
                pending_prefix_copies.append(
                    PagedTextCacheSpanCopy(
                        source=source,
                        target=staged,
                        start=0,
                        length=source_len,
                    )
                )
        elif pending_prefix_copies is None:
            setattr(staged, "_uniserve_forward_staging_source_prefix", source_prefix)
        staged.length = source_len
        if source_len <= 0:
            setattr(staged, "_uniserve_forward_staging_source_prefix", source_prefix)
        staged_by_pool[key] = staged
        return staged

    def _mark_forward_staging_advanced(
        self,
        staged: PagedTextCache,
        source: PagedTextCache,
        new_len: int,
    ) -> None:
        staged.length = int(new_len)
        setattr(
            staged,
            "_uniserve_forward_staging_source_prefix",
            self._forward_staging_source_prefix(source, int(new_len)),
        )

    def _release_forward_staging_for_cache(self, cache: Any) -> None:
        staged_by_pool = getattr(cache, "_uniserve_forward_staging", None)
        if not isinstance(staged_by_pool, dict):
            return
        seen: set[int] = set()
        for staged in staged_by_pool.values():
            staged_id = id(staged)
            if staged_id in seen:
                continue
            seen.add(staged_id)
            self.residency.release_scratch_cache(staged)
        staged_by_pool.clear()

    @staticmethod
    def _forward_staging_source_prefix(
        cache: PagedTextCache,
        length: int,
    ) -> tuple[int, ...]:
        length = int(length)
        if length <= 0:
            return ()
        block_size = int(
            getattr(cache.pool, "block_size", DEFAULT_BLOCK_SIZE) or DEFAULT_BLOCK_SIZE
        )
        block_count = ceil_div(length, block_size)
        return tuple(int(block_id) for block_id in list(cache.block_ids)[:block_count])

    def _forward_target_pool(
        self,
        denoise_steps: list[tuple[int, TextImageDenoiseStep]],
    ) -> PagedKVPool | None:
        target_pool = None
        for _row_index, step in denoise_steps:
            image = step.extra["img"]
            for branch in text_image_branches(step):
                _indexes, cache = self._denoise_branch_inputs(image, branch)
                pool = getattr(cache, "pool", None)
                if pool is None:
                    return None
                if target_pool is None:
                    target_pool = pool
                elif pool is not target_pool:
                    return None
        return target_pool

    def _add_text_forward_segment(
        self,
        *,
        builder: ForwardStreamBuilder,
        kv_segments: list[ForwardPagedKVSegment],
        row_index: int,
        req_id: int,
        op: dict[str, Any],
        mode: ForwardMode,
        cache: Any,
        q_len: int,
        start_pos: int,
        device: torch.device,
    ) -> None:
        del device
        builder.add_segment(
            op_index=row_index,
            req_id=req_id,
            kind=str(op["kind"]),
            mode=mode,
            modality="und",
            segment_class="decode" if mode is ForwardMode.DECODE else "extend",
            q_len=q_len,
            prefix_len=int(cache.past.length),
            visible_policy="causal",
            index_start=int(start_pos),
        )
        kv_segments.append(
            ForwardPagedKVSegment(
                block_ids=tuple(cache.past.block_ids),
                base_len=int(cache.past.length),
                q_len=q_len,
                write_kv=True,
            )
        )

    def _add_denoise_forward_segment(
        self,
        *,
        builder: ForwardStreamBuilder,
        kv_segments: list[ForwardPagedKVSegment],
        row_index: int,
        req_id: int,
        op: dict[str, Any],
        cache: Any,
        indexes: torch.Tensor,
        q_len: int,
        branch_index: int,
        device: torch.device,
    ) -> None:
        builder.add_segment(
            op_index=row_index,
            req_id=req_id,
            kind=str(op["kind"]),
            mode=ForwardMode.DENOISE,
            modality="gen",
            segment_class="denoise",
            q_len=q_len,
            prefix_len=int(cache.length),
            branch_id=branch_index,
            visible_policy="bidirectional",
            indexes=indexes.to(device=device),
        )
        kv_segments.append(
            ForwardPagedKVSegment(
                block_ids=tuple(cache.block_ids),
                base_len=int(cache.length),
                q_len=q_len,
                write_kv=True,
                persist_kv=False,
                branch_id=branch_index,
            )
        )

    def _wait_gen_cache_ready(self, cache: Any) -> None:
        del cache

    def packed_denoise_indicators(
        self,
        step: TextImageDenoiseStep,
        q_len: int,
    ) -> torch.Tensor | None:
        del step, q_len
        return None
