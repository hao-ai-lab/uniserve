"""Paged text KV helpers for native decoder integrations.

The worker-owned ``PagedKVPool`` is the authoritative storage.  ``PagedTextCache``
coordinates one request's logical length and exposes per-layer page views for
attention; the hot path attends through ``PagedRequestCache`` and block tables.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import ceil_div
from .cache_protocols import BufferStager
from .host_staging import (
    copy_cpu_to_device,
    cpu_int_staging_buffer,
)
from .host_staging import (
    fill_cpu_ints as _fill_cpu_int,
)
from .host_staging import (
    is_pinned as _is_pinned,
)
from .kv_pool import PagedKVPool, PagedRequestCache

__all__ = [
    'PagedTransformerLayer',
    'PagedTextCache',
    'BatchedPagedRequestCache',
    'BatchedPagedTextCache',
    'PagedTextCacheSpanCopy',
    'copy_paged_text_cache_span',
    'copy_paged_text_cache_spans',
    'stage_paged_text_cache_prefix',
]


@dataclass
class _VarlenAppendPlan:
    key: tuple[Any, ...]
    page_ids: torch.Tensor
    offsets: torch.Tensor


@dataclass(frozen=True)
class PagedTextCacheSpanCopy:
    source: "PagedTextCache"
    target: "PagedTextCache"
    start: int
    length: int


class PagedTransformerLayer:
    """One logical decoder layer backed by ``PagedKVPool``."""

    def __init__(self, cache: "PagedTextCache", layer_idx: int) -> None:
        self.cache = cache
        # Constructed only with range() indices, so layer_idx is already int.
        self.layer_idx = layer_idx
        self.device = cache.pool.k.device
        self.dtype = cache.pool.dtype

    @property
    def keys(self) -> torch.Tensor | None:
        return self._read(self.cache.length)[0]

    @property
    def values(self) -> torch.Tensor | None:
        return self._read(self.cache.length)[1]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape != value_states.shape:
            raise invalid_descriptor("paged text cache key/value shapes must match")
        if key_states.ndim != 4 or key_states.shape[0] != 1:
            raise invalid_descriptor(
                "PagedTextCache supports batch size 1 tensors shaped [1, heads, tokens, dim]"
            )
        n = int(key_states.shape[-2])
        with self.cache.span_update(self.layer_idx, n) as start:
            self.cache.ensure_capacity(start + n)
            k = key_states[0].transpose(0, 1).contiguous()
            v = value_states[0].transpose(0, 1).contiguous()
            self.cache.pool.write(self.layer_idx, self.cache.block_ids, start=start, k=k, v=v)
            end = start + n
        keys, values = self._read(end)
        if keys is None or values is None:
            empty = self._empty()
            return empty, empty
        return keys, values

    def get_seq_length(self) -> int:
        return int(self.cache.length)

    def _read(self, length: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if length <= 0:
            return None, None
        k, v = self.cache.pool.read(self.layer_idx, self.cache.block_ids, start=0, length=length)
        if k is None or v is None:
            return None, None
        return (
            k.transpose(0, 1).unsqueeze(0).contiguous(),
            v.transpose(0, 1).unsqueeze(0).contiguous(),
        )

    def _empty(self) -> torch.Tensor:
        return torch.empty(
            (1, self.cache.pool.n_kv, 0, self.cache.pool.head_dim),
            device=self.cache.pool.k.device,
            dtype=self.cache.pool.dtype,
        )


class PagedTextCache:
    """Native request-length coordinator backed by ``PagedKVPool``.

    Decoder layers append once per layer for the same token span. The cache
    writes every layer at the same logical start slot and only advances the
    public sequence length after all layers have written the span.
    """

    def __init__(
        self,
        pool: PagedKVPool,
        block_ids: list[int] | tuple[int, ...] | None,
        *,
        num_layers: int,
        length: int = 0,
        allocate_blocks: Callable[[int], list[int]] | None = None,
    ) -> None:
        self.pool = pool
        self.block_ids = pool.validate_block_ids(block_ids or [])
        self.length = int(length)
        self.allocate_blocks = allocate_blocks
        self.layers = [PagedTransformerLayer(self, idx) for idx in range(int(num_layers))]
        self._active_start: int | None = None
        self._active_count: int | None = None
        self._updated_layers: set[int] = set()
        self._update_view_cache: tuple[tuple[Any, ...], Any] | None = None
        self._transient_view_cache: tuple[tuple[Any, ...], Any] | None = None
        if self.length:
            self.ensure_capacity(self.length)

    def set_blocks(self, block_ids: list[int] | tuple[int, ...]) -> None:
        self.block_ids = self.pool.validate_block_ids(block_ids)
        self._update_view_cache = None
        self._transient_view_cache = None
        self.ensure_capacity(self.length)

    def ensure_capacity(self, end: int) -> None:
        end = int(end)
        missing_tokens = end - len(self.block_ids) * self.pool.block_size
        if missing_tokens <= 0:
            return
        if self.allocate_blocks is None:
            raise invalid_descriptor(
                "paged text cache does not have enough logical blocks "
                f"for {end} tokens"
            )
        needed = ceil_div(missing_tokens, self.pool.block_size)
        self.block_ids.extend(self.pool.validate_block_ids(self.allocate_blocks(needed)))
        self._update_view_cache = None
        self._transient_view_cache = None
        # ``allocate_blocks`` is only required to extend the contiguous tail; a
        # short return (fewer blocks than requested) would otherwise leave the
        # cache silently under-provisioned and surface as a confusing span error
        # deep inside ``pool.write``.  Fail fast with the same descriptor error
        # ``ensure_capacity`` raises when no allocator is wired.
        if len(self.block_ids) * self.pool.block_size < end:
            raise invalid_descriptor(
                "paged text cache allocator did not provide enough logical blocks "
                f"for {end} tokens"
            )

    def request_cache_for_update(self, layer_idx: int, n_tokens: int):
        start = self.begin_layer_update(layer_idx, n_tokens)
        self.ensure_capacity(start + int(n_tokens))
        key = (tuple(int(block_id) for block_id in self.block_ids), int(start))
        cached = self._update_view_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        view = self.pool.view(self.block_ids, start)
        self._update_view_cache = (key, view)
        return view

    def request_cache_for_transient(self, layer_idx: int, n_tokens: int):
        """Return a page view for temporary tokens without advancing length.

        Image denoise tokens are rewritten every denoise step and should not
        become part of the persistent text/image context until commit.  They
        still need to sit in paged KV storage so attention can use the same
        block-table kernel as decode.
        """

        del layer_idx
        start = int(self.length)
        self.ensure_capacity(start + int(n_tokens))
        key = (tuple(int(block_id) for block_id in self.block_ids), start)
        cached = self._transient_view_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        view = self.pool.view(self.block_ids, start)
        self._transient_view_cache = (key, view)
        return view

    def cancel_layer_update(self, layer_idx: int) -> None:
        self._updated_layers.discard(int(layer_idx))
        if not self._updated_layers:
            self._active_start = None
            self._active_count = None
            self._update_view_cache = None

    @contextmanager
    def span_update(self, layer_idx: int, n_tokens: int) -> Iterator[int]:
        """Transaction for one layer's write into the shared token span.

        Yields the span start slot. A normal exit records the layer and advances
        the public length once every layer has written; an exception rolls back
        via ``cancel_layer_update``.
        """
        start = self.begin_layer_update(layer_idx, n_tokens)
        try:
            yield start
        except BaseException:
            self.cancel_layer_update(layer_idx)
            raise
        else:
            self.finish_layer_update(layer_idx, n_tokens)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[int(layer_idx)].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.length)

    def begin_layer_update(self, layer_idx: int, n_tokens: int) -> int:
        layer_idx = int(layer_idx)
        n_tokens = int(n_tokens)
        if self._active_start is None or layer_idx in self._updated_layers:
            # self.length is maintained as a Python int (assigned only via int()).
            self._active_start = self.length
            self._active_count = n_tokens
            self._updated_layers.clear()
            self._update_view_cache = None
        elif self._active_count != n_tokens:
            raise invalid_descriptor(
                "all layers in one paged cache update must append the same token count"
            )
        return self._active_start

    def finish_layer_update(self, layer_idx: int, n_tokens: int) -> None:
        self._updated_layers.add(int(layer_idx))
        if len(self._updated_layers) >= len(self.layers):
            self.length = (self._active_start or 0) + int(n_tokens)
            self._active_start = None
            self._active_count = None
            self._updated_layers.clear()
            self._update_view_cache = None

    def restore_committed(self, length: int, block_ids: Sequence[int]) -> None:
        """Reset the cache to a committed length and block table.

        The step transaction calls this on rollback: the logical length, the
        block table, and any partially applied layer span return to the
        committed state. K/V already written past the committed length is
        inert scratch that the next span write at those slots overwrites.
        """
        self.block_ids = self.pool.validate_block_ids(block_ids)
        self.length = int(length)
        self._active_start = None
        self._active_count = None
        self._updated_layers.clear()
        self._update_view_cache = None
        self._transient_view_cache = None


def stage_paged_text_cache_prefix(
    source: PagedTextCache,
    *,
    target_pool: PagedKVPool,
    allocate_blocks: Callable[[int], list[int]],
    num_layers: int,
    end_len: int,
) -> PagedTextCache:
    """Create a writable cache in ``target_pool`` with ``source`` prefix copied."""

    staged = PagedTextCache(
        target_pool,
        [],
        num_layers=int(num_layers),
        length=int(source.length),
        allocate_blocks=allocate_blocks,
    )
    staged.ensure_capacity(int(end_len))
    copy_paged_text_cache_span(
        source,
        staged,
        start=0,
        length=int(source.length),
        num_layers=int(num_layers),
        missing_message="cannot stage mixed forward prefix without a paged source cache",
    )
    return staged


def copy_paged_text_cache_span(
    source: PagedTextCache | PagedRequestCache,
    target: PagedTextCache,
    *,
    start: int,
    length: int,
    num_layers: int,
    missing_message: str = "paged text K/V span is missing",
) -> None:
    """Copy a logical KV span between paged text caches."""

    start = int(start)
    length = int(length)
    if length <= 0:
        return
    source_pool = getattr(source, "pool", None)
    source_blocks = list(getattr(source, "block_ids", []) or [])
    if source_pool is None or not source_blocks:
        raise RuntimeError(missing_message)
    target.ensure_capacity(start + length)
    for layer_idx in range(int(num_layers)):
        k, v = source_pool.read(layer_idx, source_blocks, start=start, length=length)
        if k is None or v is None:
            raise RuntimeError(missing_message)
        target.pool.write(
            layer_idx,
            target.block_ids,
            start=start,
            k=k.to(target.pool.k.device),
            v=v.to(target.pool.v.device),
        )


def copy_paged_text_cache_spans(
    spans: Sequence[PagedTextCacheSpanCopy],
    *,
    num_layers: int,
    missing_message: str = "paged text K/V span is missing",
) -> None:
    """Copy multiple logical KV spans between paged text caches."""

    normalized: list[PagedTextCacheSpanCopy] = []
    for span in spans:
        start = int(span.start)
        length = int(span.length)
        if length <= 0:
            continue
        source_pool = getattr(span.source, "pool", None)
        target_pool = getattr(span.target, "pool", None)
        source_blocks = list(getattr(span.source, "block_ids", []) or [])
        if source_pool is None or target_pool is None or not source_blocks:
            raise RuntimeError(missing_message)
        span.target.ensure_capacity(start + length)
        normalized.append(
            PagedTextCacheSpanCopy(
                source=span.source,
                target=span.target,
                start=start,
                length=length,
            )
        )
    if not normalized:
        return
    if _can_copy_paged_text_cache_spans_direct(normalized, num_layers=int(num_layers)):
        _copy_paged_text_cache_spans_direct(normalized, num_layers=int(num_layers))
        return
    for span in normalized:
        copy_paged_text_cache_span(
            span.source,
            span.target,
            start=span.start,
            length=span.length,
            num_layers=int(num_layers),
            missing_message=missing_message,
        )


def _can_copy_paged_text_cache_spans_direct(
    spans: Sequence[PagedTextCacheSpanCopy],
    *,
    num_layers: int,
) -> bool:
    for span in spans:
        source_pool = span.source.pool
        target_pool = span.target.pool
        if bool(getattr(source_pool, "is_quantized", False)) or bool(
            getattr(target_pool, "is_quantized", False)
        ):
            return False
        if (
            source_pool.k.device != source_pool.v.device
            or target_pool.k.device != target_pool.v.device
            or source_pool.k.device != target_pool.k.device
        ):
            return False
        if source_pool.num_layers < int(num_layers) or target_pool.num_layers < int(num_layers):
            return False
        if source_pool.n_kv != target_pool.n_kv or source_pool.head_dim != target_pool.head_dim:
            return False
    return True


def _copy_paged_text_cache_spans_direct(
    spans: Sequence[PagedTextCacheSpanCopy],
    *,
    num_layers: int,
) -> None:
    groups: dict[tuple[int, int], list[PagedTextCacheSpanCopy]] = {}
    for span in spans:
        groups.setdefault((id(span.source.pool), id(span.target.pool)), []).append(span)
    for group in groups.values():
        source_pool = group[0].source.pool
        target_pool = group[0].target.pool
        source_index = _span_positions(
            source_pool,
            [(span.source.block_ids, int(span.start), int(span.length)) for span in group],
        )
        target_index = _span_positions(
            target_pool,
            [(span.target.block_ids, int(span.start), int(span.length)) for span in group],
        )
        if source_index.numel() <= 0:
            continue
        layer_count = int(num_layers)
        source_k = source_pool.k[:layer_count].reshape(
            layer_count, -1, source_pool.n_kv, source_pool.head_dim
        )
        source_v = source_pool.v[:layer_count].reshape(
            layer_count, -1, source_pool.n_kv, source_pool.head_dim
        )
        target_k = target_pool.k[:layer_count].reshape(
            layer_count, -1, target_pool.n_kv, target_pool.head_dim
        )
        target_v = target_pool.v[:layer_count].reshape(
            layer_count, -1, target_pool.n_kv, target_pool.head_dim
        )
        tokens_per_source_layer = int(source_k.shape[1])
        tokens_per_target_layer = int(target_k.shape[1])
        layer_offsets = torch.arange(layer_count, device=source_index.device, dtype=torch.long)
        flat_source_index = (
            source_index.reshape(1, -1)
            + (layer_offsets * tokens_per_source_layer).reshape(-1, 1)
        ).reshape(-1)
        flat_target_index = (
            target_index.reshape(1, -1)
            + (layer_offsets * tokens_per_target_layer).reshape(-1, 1)
        ).reshape(-1)
        source_k_flat = source_k.reshape(-1, source_pool.n_kv, source_pool.head_dim)
        source_v_flat = source_v.reshape(-1, source_pool.n_kv, source_pool.head_dim)
        target_k_flat = target_k.reshape(-1, target_pool.n_kv, target_pool.head_dim)
        target_v_flat = target_v.reshape(-1, target_pool.n_kv, target_pool.head_dim)
        target_k_flat.index_copy_(
            0,
            flat_target_index,
            source_k_flat.index_select(0, flat_source_index).to(dtype=target_k_flat.dtype),
        )
        target_v_flat.index_copy_(
            0,
            flat_target_index,
            source_v_flat.index_select(0, flat_source_index).to(dtype=target_v_flat.dtype),
        )


def _span_positions(
    pool: PagedKVPool,
    spans: Sequence[tuple[Sequence[int], int, int]],
) -> torch.Tensor:
    positions: list[int] = []
    block_size = int(pool.block_size)
    for block_ids, start, length in spans:
        for block_id, offset, count in pool.spans(list(block_ids), int(start), int(length)):
            base = int(block_id) * block_size + int(offset)
            positions.extend(range(base, base + int(count)))
    return torch.tensor(positions, device=pool.k.device, dtype=torch.long)


class BatchedPagedRequestCache:
    """Batched transient page view over compatible request caches."""

    def __init__(
        self,
        pool: PagedKVPool,
        block_ids_by_row: Sequence[Sequence[int]],
        base_lens: Sequence[int],
        *,
        block_table_width: int | None = None,
    ) -> None:
        if not block_ids_by_row:
            raise invalid_descriptor("batched paged request cache requires at least one row")
        if len(block_ids_by_row) != len(base_lens):
            raise invalid_descriptor("batched paged request cache rows and lengths mismatch")
        self.pool = pool
        self.block_ids_by_row = [pool.validate_block_ids(ids) for ids in block_ids_by_row]
        self.base_lens = [int(length) for length in base_lens]
        if any(length < 0 for length in self.base_lens):
            raise invalid_descriptor("batched paged request cache lengths must be non-negative")
        self.base_len = max(self.base_lens, default=0)
        minimum_width = max(len(ids) for ids in self.block_ids_by_row)
        self._block_table_width = int(block_table_width or minimum_width)
        if self._block_table_width < minimum_width:
            raise invalid_descriptor("batched paged request cache block-table width is too small")
        self._fixed_block_table_width = block_table_width is not None
        self._append_plan: _VarlenAppendPlan | None = None
        self._block_table_cache: dict[torch.device, torch.Tensor] = {}
        self._cache_seqlens_cache: dict[torch.device, torch.Tensor] = {}

    def reset_rows(
        self,
        block_ids_by_row: Sequence[Sequence[int]],
        base_lens: Sequence[int],
    ) -> None:
        if len(block_ids_by_row) != len(self.block_ids_by_row) or len(base_lens) != len(self.base_lens):
            raise invalid_descriptor("batched paged request cache reset shape mismatch")
        new_lens = [int(length) for length in base_lens]
        if any(length < 0 for length in new_lens):
            raise invalid_descriptor("batched paged request cache lengths must be non-negative")
        self.block_ids_by_row = [self.pool.validate_block_ids(ids) for ids in block_ids_by_row]
        self.base_lens = new_lens
        self.base_len = max(self.base_lens, default=0)
        minimum_width = max(len(ids) for ids in self.block_ids_by_row)
        if self._fixed_block_table_width and minimum_width > self._block_table_width:
            raise invalid_descriptor("batched paged request cache block-table width exceeded")
        if not self._fixed_block_table_width:
            self._block_table_width = minimum_width
        self._append_plan = None
        self._block_table_cache.clear()
        self._cache_seqlens_cache.clear()

    def refresh_rows(
        self,
        block_ids_by_row: Sequence[Sequence[int]],
        base_lens: Sequence[int],
    ) -> None:
        """Refresh graph inputs without changing their device addresses."""

        if len(block_ids_by_row) != len(self.block_ids_by_row) or len(base_lens) != len(self.base_lens):
            raise invalid_descriptor("batched paged request cache refresh shape mismatch")
        rows = [self.pool.validate_block_ids(ids) for ids in block_ids_by_row]
        lengths = [int(length) for length in base_lens]
        if any(length < 0 for length in lengths):
            raise invalid_descriptor("batched paged request cache lengths must be non-negative")
        if max(len(ids) for ids in rows) > self._block_table_width:
            raise invalid_descriptor("batched paged request cache block-table width exceeded")
        self.block_ids_by_row = rows
        self.base_lens = lengths
        self.base_len = max(lengths, default=0)

        row_count = len(rows)
        for target, out in self._block_table_cache.items():
            cpu = _cpu_int_buffer(
                row_count * self._block_table_width,
                pin=target.type == "cuda",
                name="paged_block_table_refresh",
            )
            _fill_block_table(cpu, rows, self._block_table_width)
            out.copy_(
                cpu.view(row_count, self._block_table_width),
                non_blocking=target.type == "cuda" and _is_pinned(cpu),
            )
        for target, out in self._cache_seqlens_cache.items():
            cpu = _cpu_int_buffer(
                row_count,
                pin=target.type == "cuda",
                name="paged_cache_seqlens_refresh",
            )
            _fill_cpu_int(cpu, lengths)
            out.copy_(cpu, non_blocking=target.type == "cuda" and _is_pinned(cpu))

        cached = getattr(self, "_transient_varlen_metadata", None)
        if cached is not None:
            _old_key, metadata = cached
            (
                query_lens_cpu,
                block_table,
                cache_seqlens,
                cu_seqlens_q,
                cu_seqlens_k,
                max_q,
                _max_k,
            ) = metadata
            kv_lens = [
                int(base_len) + int(query_len)
                for base_len, query_len in zip(lengths, query_lens_cpu, strict=True)
            ]
            cu_k_values = [0]
            for length in kv_lens:
                cu_k_values.append(cu_k_values[-1] + length)
            cu_k_cpu = _cpu_int_buffer(
                len(cu_k_values),
                pin=cu_seqlens_k.device.type == "cuda",
                name="paged_cu_seqlens_k_refresh",
            )
            _fill_cpu_int(cu_k_cpu, cu_k_values)
            cu_seqlens_k.copy_(
                cu_k_cpu,
                non_blocking=cu_seqlens_k.device.type == "cuda" and _is_pinned(cu_k_cpu),
            )
            key = (str(block_table.device), tuple(lengths), int(query_lens_cpu[0]))
            self._transient_varlen_metadata = (
                key,
                (
                    query_lens_cpu,
                    block_table,
                    cache_seqlens,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_q,
                    max(kv_lens, default=0),
                ),
            )

    def invalidate_append_plan(self) -> None:
        self._append_plan = None

    def block_table(
        self,
        *,
        device: torch.device | str | None = None,
        stager: BufferStager | None = None,
    ) -> torch.Tensor:
        max_blocks = self._block_table_width
        row_count = len(self.block_ids_by_row)
        target = torch.device(device if device is not None else self.pool.k.device)
        if stager is None:
            cached = self._block_table_cache.get(target)
            if cached is not None:
                return cached
        cpu = _cpu_int_buffer(
            row_count * max_blocks,
            pin=target.type == "cuda",
            slot=stager,
            name="paged_block_table",
        )
        _fill_block_table(cpu, self.block_ids_by_row, max_blocks)
        non_blocking = target.type == "cuda" and _is_pinned(cpu)
        out = _copy_cpu_int_to_device(
            cpu,
            device=target,
            non_blocking=non_blocking,
            slot=stager,
            name="paged_block_table",
        ).view(row_count, max_blocks)
        if stager is None:
            self._block_table_cache[target] = out
        return out

    def cache_seqlens(
        self,
        *,
        device: torch.device | str | None = None,
        stager: BufferStager | None = None,
    ) -> torch.Tensor:
        target = torch.device(device if device is not None else self.pool.k.device)
        if stager is None:
            cached = self._cache_seqlens_cache.get(target)
            if cached is not None:
                return cached
        cpu = _cpu_int_buffer(
            len(self.base_lens),
            pin=target.type == "cuda",
            slot=stager,
            name="paged_cache_seqlens",
        )
        _fill_cpu_int(cpu, self.base_lens)
        non_blocking = target.type == "cuda" and _is_pinned(cpu)
        out = _copy_cpu_int_to_device(
            cpu,
            device=target,
            non_blocking=non_blocking,
            slot=stager,
            name="paged_cache_seqlens",
        )
        if stager is None:
            self._cache_seqlens_cache[target] = out
        return out

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape:
            raise invalid_descriptor("batched KV append key/value shapes must match")
        if k.ndim != 4:
            raise invalid_descriptor("batched KV append expects [batch, tokens, heads, dim]")
        if k.shape[0] != len(self.block_ids_by_row):
            raise invalid_descriptor("batched KV append batch size does not match cache rows")
        for row, block_ids in enumerate(self.block_ids_by_row):
            self.pool.write(
                layer,
                block_ids,
                start=self.base_lens[row],
                k=k[row],
                v=v[row],
            )

    def append_varlen(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        query_lens: Sequence[int],
        *,
        block_table: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> None:
        if k.shape != v.shape:
            raise invalid_descriptor("ragged KV append key/value shapes must match")
        if k.ndim != 3:
            raise invalid_descriptor("ragged KV append expects [tokens, heads, dim]")
        if len(query_lens) != len(self.block_ids_by_row):
            raise invalid_descriptor("ragged KV append query lengths do not match cache rows")
        query_lens = [int(length) for length in query_lens]
        if any(length < 0 for length in query_lens):
            raise invalid_descriptor("ragged KV append lengths must be non-negative")
        total = sum(query_lens)
        if total != int(k.shape[0]):
            raise invalid_descriptor("ragged KV append token count does not match query lengths")
        if self._append_varlen_indexed(
            int(layer),
            k,
            v,
            total=total,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        ):
            return
        offset = 0
        for row, (block_ids, length) in enumerate(zip(self.block_ids_by_row, query_lens)):
            if length:
                self.pool.write(
                    layer,
                    block_ids,
                    start=self.base_lens[row],
                    k=k[offset:offset + length],
                    v=v[offset:offset + length],
                )
            offset += length

    def _append_varlen_indexed(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        total: int,
        block_table: torch.Tensor | None,
        cache_seqlens: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor | None,
    ) -> bool:
        if total <= 0:
            return True
        if block_table is None or cache_seqlens is None or cu_seqlens_q is None:
            return False
        if self.pool.is_quantized or not bool(getattr(self.pool, "supports_paged_attention_storage", True)):
            return False
        if not (k.is_cuda and v.is_cuda and self.pool.k.is_cuda and self.pool.v.is_cuda):
            return False
        if k.device != self.pool.k.device or v.device != self.pool.v.device:
            return False
        row_count = len(self.block_ids_by_row)
        if int(block_table.shape[0]) != row_count or int(cache_seqlens.shape[0]) != row_count:
            return False
        if int(cu_seqlens_q.numel()) != row_count + 1:
            return False
        if block_table.device != k.device or cache_seqlens.device != k.device or cu_seqlens_q.device != k.device:
            return False
        if k.shape[1:] != (self.pool.n_kv, self.pool.head_dim):
            return False
        plan = self._varlen_append_plan(
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            total=total,
        )
        self.pool.k[layer, plan.page_ids, plan.offsets] = k.to(dtype=self.pool.k.dtype)
        self.pool.v[layer, plan.page_ids, plan.offsets] = v.to(dtype=self.pool.v.dtype)
        return True

    def _varlen_append_plan(
        self,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        total: int,
    ) -> _VarlenAppendPlan:
        key = (
            int(block_table.data_ptr()),
            int(cache_seqlens.data_ptr()),
            int(cu_seqlens_q.data_ptr()),
            tuple(int(dim) for dim in block_table.shape),
            tuple(int(dim) for dim in cache_seqlens.shape),
            tuple(int(dim) for dim in cu_seqlens_q.shape),
            int(total),
        )
        cached = self._append_plan
        if cached is not None and cached.key == key:
            return cached
        device = block_table.device
        token_offsets = torch.arange(int(total), device=device, dtype=torch.int64)
        if len(self.block_ids_by_row) == 1:
            positions = cache_seqlens[0].to(dtype=torch.int64) + token_offsets
            block_slots = torch.div(positions, int(self.pool.block_size), rounding_mode="floor")
            page_ids = block_table[0].to(dtype=torch.int64).index_select(0, block_slots).contiguous()
        else:
            cu = cu_seqlens_q.to(dtype=torch.int64)
            row_ids = torch.bucketize(token_offsets, cu[1:].contiguous(), right=True)
            positions = cache_seqlens.to(dtype=torch.int64)[row_ids] + token_offsets - cu[row_ids]
            block_slots = torch.div(positions, int(self.pool.block_size), rounding_mode="floor")
            page_ids = block_table.to(dtype=torch.int64)[row_ids, block_slots].contiguous()
        offsets = torch.remainder(positions, int(self.pool.block_size)).contiguous()
        plan = _VarlenAppendPlan(key=key, page_ids=page_ids, offsets=offsets)
        self._append_plan = plan
        return plan


class BatchedPagedTextCache:
    """Transient batched wrapper over compatible ``PagedTextCache`` rows."""

    supports_batched_paged = True

    def __init__(
        self,
        caches: Sequence[PagedTextCache],
        *,
        block_table_width: int | None = None,
    ) -> None:
        if not caches:
            raise invalid_descriptor("BatchedPagedTextCache requires at least one cache")
        first = caches[0]
        pool = first.pool
        num_layers = len(first.layers)
        for cache in caches:
            if cache.pool is not pool:
                raise invalid_descriptor("batched paged caches must share one KV pool")
            if len(cache.layers) != num_layers:
                raise invalid_descriptor("batched paged caches must have the same layer count")
        self.caches = list(caches)
        self.pool = pool
        self.block_table_width = block_table_width
        self._transient_view_cache: tuple[tuple[Any, ...], BatchedPagedRequestCache] | None = None

    def refresh_caches(self, caches: Sequence[PagedTextCache], n_tokens: int) -> None:
        """Rebind a graph-owned batched view to same-geometry live caches."""

        if len(caches) != len(self.caches):
            raise invalid_descriptor("batched paged cache refresh row count mismatch")
        n_tokens = int(n_tokens)
        for cache in caches:
            if cache.pool is not self.pool or len(cache.layers) != len(self.caches[0].layers):
                raise invalid_descriptor("batched paged cache refresh geometry mismatch")
            cache.ensure_capacity(int(cache.length) + n_tokens)
        cached = self._transient_view_cache
        if cached is None:
            raise invalid_descriptor("batched paged cache graph view is not initialized")
        view = cached[1]
        view.refresh_rows(
            [cache.block_ids for cache in caches],
            [int(cache.length) for cache in caches],
        )
        self.caches = list(caches)
        key = tuple(
            (tuple(int(block_id) for block_id in cache.block_ids), int(cache.length))
            for cache in caches
        )
        self._transient_view_cache = (key, view)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return max((int(cache.length) for cache in self.caches), default=0)

    def request_cache_for_transient(self, layer_idx: int, n_tokens: int) -> BatchedPagedRequestCache:
        del layer_idx
        n_tokens = int(n_tokens)
        if n_tokens <= 0:
            raise invalid_descriptor("batched transient cache requires positive token count")
        for cache in self.caches:
            cache.ensure_capacity(int(cache.length) + n_tokens)
        key = tuple(
            (
                tuple(int(block_id) for block_id in cache.block_ids),
                int(cache.length),
            )
            for cache in self.caches
        )
        cached = self._transient_view_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        view = BatchedPagedRequestCache(
            self.pool,
            [cache.block_ids for cache in self.caches],
            [int(cache.length) for cache in self.caches],
            block_table_width=self.block_table_width,
        )
        self._transient_view_cache = (key, view)
        return view


def _cpu_int_buffer(
    numel: int,
    *,
    pin: bool,
    slot: BufferStager | None = None,
    name: str = "buffer",
) -> torch.Tensor:
    return cpu_int_staging_buffer(numel, dtype=torch.int32, pin=pin, slot=slot, name=name)


def _fill_block_table(
    cpu: torch.Tensor,
    rows: Sequence[Sequence[int]],
    width: int,
) -> None:
    offset = 0
    width = int(width)
    for block_ids in rows:
        n = len(block_ids)
        _fill_cpu_int(cpu[offset:offset + n], block_ids)
        if n < width:
            cpu[offset + n: offset + width].zero_()
        offset += width


_copy_cpu_int_to_device = copy_cpu_to_device
