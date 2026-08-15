"""Fixed physical KV storage indexed by scheduler-assigned page IDs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import torch

from ..backends.paged_kv_math import paged_kv_write
from ..foundation.device import (
    copy_cpu_to_device,
    cpu_int_staging_buffer,
    fill_cpu_ints,
    is_pinned,
)
from ..foundation.errors import capability_mismatch, compute_error, invalid_descriptor
from ..foundation.sizing import bucketed_page_count
from ..nn.quant.kv_cache import (
    dequantize_fp8_block,
    fp8_quantize,
    is_fp8_kv_dtype,
    resolve_kv_store_dtype,
    scale_for_fp8_block,
)

__all__ = ["CacheBatchView", "CacheExtents", "CachePool", "CacheRow"]


@dataclass(frozen=True, slots=True)
class CacheExtents:
    reserved: int
    initialized: int
    visible: int
    committed: int
    published: int

    def __post_init__(self) -> None:
        if not (
            0
            <= self.published
            <= self.committed
            <= self.visible
            <= self.initialized
            <= self.reserved
        ):
            raise invalid_descriptor("KV extents are not monotonically contained")


@dataclass(slots=True)
class CacheRow:
    """One operation-local cursor over an explicit scheduler block table."""

    block_table: tuple[int, ...]
    length: int
    capacity: int
    group_id: int = 0
    initialized_length: int | None = None
    committed_length: int | None = None
    published_length: int = 0

    def __post_init__(self) -> None:
        initialized = self.length if self.initialized_length is None else self.initialized_length
        committed = self.length if self.committed_length is None else self.committed_length
        self.initialized_length = int(initialized)
        self.committed_length = int(committed)
        if not (
            0
            <= self.published_length
            <= self.committed_length
            <= self.length
            <= self.initialized_length
            <= self.capacity
        ):
            raise invalid_descriptor("KV row lengths are outside the scheduler placement")
        if self.group_id < 0 or (not self.block_table and self.capacity != 0):
            raise invalid_descriptor("KV row has no scheduler placement")

    def advance(self, tokens: int) -> None:
        resulting = self.length + int(tokens)
        if resulting < self.length or resulting > self.capacity:
            raise invalid_descriptor("KV row advance exceeds scheduler placement")
        self.length = resulting
        self.initialized_length = max(cast(int, self.initialized_length), resulting)

    def initialize(self, tokens: int) -> int:
        resulting = self.length + int(tokens)
        if resulting < self.length or resulting > self.capacity:
            raise invalid_descriptor("KV row initialization exceeds scheduler placement")
        self.initialized_length = max(cast(int, self.initialized_length), resulting)
        return resulting

    def select(self, length: int) -> None:
        selected = int(length)
        if selected < cast(int, self.committed_length) or selected > cast(
            int, self.initialized_length
        ):
            raise invalid_descriptor("KV row selection is outside initialized state")
        self.length = selected

    def extents(self) -> CacheExtents:
        return CacheExtents(
            reserved=self.capacity,
            initialized=cast(int, self.initialized_length),
            visible=self.length,
            committed=cast(int, self.committed_length),
            published=self.published_length,
        )


class CachePool:
    """Startup-sized layer-major KV tensors with no allocation authority."""

    def __init__(
        self,
        *,
        num_layers: int,
        request_pages: int,
        scratch_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
        store_dtype: torch.dtype | str | None = None,
        group_ranges: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        self.num_layers = int(num_layers)
        self.request_pages = int(request_pages)
        self.scratch_pages = int(scratch_pages)
        self.scratch_page_offset = self.request_pages
        self.num_pages = self.request_pages + self.scratch_pages
        self.num_blocks = self.num_pages
        self.block_size = int(page_size)
        self.n_kv = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.group_ranges = self._group_ranges(group_ranges)
        self.group_count = len(self.group_ranges)
        self.dtype = dtype
        self.store_dtype = resolve_kv_store_dtype(dtype, store_dtype)
        self.is_quantized = is_fp8_kv_dtype(self.store_dtype)
        self.supports_paged_attention_storage = not self.is_quantized
        if (
            self.num_layers < 1
            or self.request_pages < 1
            or self.scratch_pages < 0
            or self.block_size < 1
            or self.n_kv < 1
            or self.head_dim < 1
        ):
            raise invalid_descriptor("CachePool geometry is invalid")
        shape = (
            self.num_layers,
            self.num_pages,
            self.block_size,
            self.n_kv,
            self.head_dim,
        )
        self.k = torch.zeros(shape, device=device, dtype=self.store_dtype)
        self.v = torch.zeros(shape, device=device, dtype=self.store_dtype)
        self._page_ids = torch.arange(self.num_pages, dtype=torch.long, device=self.k.device)
        scale_shape = (self.num_layers, self.num_pages, 1, 1, 1)
        self.k_scale = (
            torch.ones(scale_shape, device=device, dtype=torch.float32)
            if self.is_quantized
            else None
        )
        self.v_scale = (
            torch.ones(scale_shape, device=device, dtype=torch.float32)
            if self.is_quantized
            else None
        )
        scale_flags = (self.num_layers, self.num_pages)
        self.k_scale_set = (
            torch.zeros(scale_flags, device=device, dtype=torch.bool) if self.is_quantized else None
        )
        self.v_scale_set = (
            torch.zeros(scale_flags, device=device, dtype=torch.bool) if self.is_quantized else None
        )
        self._validated_page_tuples: dict[
            tuple[tuple[int, ...], bool, bool | None, int | None], tuple[int, ...]
        ] = {}

    def _group_ranges(
        self,
        declared: Sequence[tuple[int, int]] | None,
    ) -> tuple[tuple[int, int], ...]:
        ranges = (
            ((0, self.request_pages),)
            if declared is None
            else tuple((int(offset), int(count)) for offset, count in declared)
        )
        if not ranges:
            raise invalid_descriptor("CachePool declares no KV groups")
        covered = [False] * self.request_pages
        for group, (offset, count) in enumerate(ranges):
            end = offset + count
            if offset < 0 or count < 1 or end > self.request_pages:
                raise invalid_descriptor(f"KV group {group} has invalid physical page bounds")
            for page in range(offset, end):
                if covered[page]:
                    raise invalid_descriptor("KV group physical page ranges overlap")
                covered[page] = True
        if not all(covered):
            raise invalid_descriptor("KV group physical page ranges do not cover the request pool")
        return ranges

    def validate_group(self, group: int) -> int:
        value = int(group)
        if value < 0 or value >= self.group_count:
            raise invalid_descriptor(
                f"KV group {value} outside pool group count {self.group_count}"
            )
        return value

    def request_page_ids(self, group: int) -> range:
        group_id = self.validate_group(group)
        offset, count = self.group_ranges[group_id]
        return range(max(1, offset), offset + count)

    def validate_pages(
        self,
        page_ids: Iterable[int],
        *,
        allow_sentinel: bool = False,
        scratch: bool | None = None,
        group: int | None = None,
    ) -> tuple[int, ...]:
        pages = tuple(int(page) for page in page_ids)
        key = (pages, bool(allow_sentinel), scratch, group)
        cached = self._validated_page_tuples.get(key)
        if cached is not None:
            return cached
        validated = self._validate_page_tuple(pages, allow_sentinel, scratch, group)
        if len(self._validated_page_tuples) >= 16_384:
            self._validated_page_tuples.clear()
        self._validated_page_tuples[key] = validated
        return validated

    def _validate_page_tuple(
        self,
        pages: tuple[int, ...],
        allow_sentinel: bool,
        scratch: bool | None,
        group: int | None,
    ) -> tuple[int, ...]:
        real_pages = tuple(page for page in pages if page != 0)
        if len(set(real_pages)) != len(real_pages):
            raise invalid_descriptor("KV placement repeats a physical page")
        lower = 0 if allow_sentinel else 1
        upper = self.num_pages
        if pages and (min(pages) < lower or max(pages) >= upper):
            raise invalid_descriptor("KV placement exceeds the fixed physical pool")
        if scratch is False and any(page >= self.scratch_page_offset for page in pages):
            raise invalid_descriptor("request KV placement addresses generation scratch storage")
        if scratch is True and any(page < self.scratch_page_offset for page in pages):
            raise invalid_descriptor("generation KV placement addresses request storage")
        if group is not None:
            group_id = self.validate_group(group)
            offset, count = self.group_ranges[group_id]
            end = offset + count
            request = tuple(page for page in real_pages if page < self.scratch_page_offset)
            if any(page < offset or page >= end for page in request):
                raise invalid_descriptor("KV placement addresses another cache group")
            if any(page >= self.scratch_page_offset for page in real_pages) and group_id != 0:
                raise invalid_descriptor("generation KV scratch belongs to cache group zero")
        return pages

    def zero_pages(self, group: int, page_ids: Iterable[int]) -> None:
        pages = self.validate_pages(page_ids, group=group)
        if not pages:
            return
        indices = self._device_page_indices(pages)
        self.k.index_fill_(1, indices, 0)
        self.v.index_fill_(1, indices, 0)
        if self.k_scale is not None and self.v_scale is not None:
            self.k_scale.index_fill_(1, indices, 1)
            self.v_scale.index_fill_(1, indices, 1)
        if self.k_scale_set is not None and self.v_scale_set is not None:
            self.k_scale_set.index_fill_(1, indices, False)
            self.v_scale_set.index_fill_(1, indices, False)

    def copy_pages(
        self,
        group: int,
        source_pages: Sequence[int],
        target_pages: Sequence[int],
    ) -> None:
        if len(source_pages) != len(target_pages):
            raise invalid_descriptor("KV page copy requires aligned source and destination pages")
        source_ids = self.validate_pages(source_pages, group=group)
        target_ids = self.validate_pages(target_pages, group=group)
        if not source_ids:
            return
        source = self._device_page_indices(source_ids)
        target = self._device_page_indices(target_ids)
        for store in (
            self.k,
            self.v,
            self.k_scale,
            self.v_scale,
            self.k_scale_set,
            self.v_scale_set,
        ):
            if store is not None:
                store.index_copy_(1, target, store.index_select(1, source))

    def field(self, group: int, layer: int, name: str) -> torch.Tensor:
        self.validate_group(group)
        layer_id = self._validate_layer(layer)
        fields = {
            "key": self.k,
            "value": self.v,
            "key_scale": self.k_scale,
            "value_scale": self.v_scale,
            "key_scale_set": self.k_scale_set,
            "value_scale_set": self.v_scale_set,
        }
        value = fields.get(str(name))
        if value is None:
            raise invalid_descriptor(f"KV field {name!r} is unavailable")
        return value[layer_id]

    def page_view(
        self,
        group: int,
        page_ids: Iterable[int],
    ) -> tuple[torch.Tensor, ...]:
        pages = self.validate_pages(page_ids, group=group)
        index = self._device_page_indices(pages)
        values: list[torch.Tensor] = [
            self.k.index_select(1, index),
            self.v.index_select(1, index),
        ]
        for store in (self.k_scale, self.v_scale, self.k_scale_set, self.v_scale_set):
            if store is not None:
                values.append(store.index_select(1, index))
        return tuple(values)

    def restore_pages(
        self,
        group: int,
        page_ids: Sequence[int],
        tensors: Sequence[torch.Tensor],
    ) -> None:
        pages = self.validate_pages(page_ids, group=group)
        stores = tuple(
            store
            for store in (
                self.k,
                self.v,
                self.k_scale,
                self.v_scale,
                self.k_scale_set,
                self.v_scale_set,
            )
            if store is not None
        )
        if len(tensors) != len(stores):
            raise invalid_descriptor("KV page payload does not match pool fields")
        index = self._device_page_indices(pages)
        for store, tensor in zip(stores, tensors, strict=True):
            expected = (self.num_layers, len(pages), *store.shape[2:])
            if tuple(tensor.shape) != expected:
                raise invalid_descriptor("KV page payload shape does not match pool geometry")
            store.index_copy_(
                1,
                index,
                tensor.to(device=store.device, dtype=store.dtype),
            )

    def _device_page_indices(self, pages: tuple[int, ...]) -> torch.Tensor:
        if not pages:
            return self._page_ids[:0]
        first = pages[0]
        if pages == tuple(range(first, first + len(pages))):
            return self._page_ids[first : first + len(pages)]
        return torch.stack(tuple(self._page_ids[page] for page in pages))

    def layer_cache(self, layer: int, group: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate_group(group)
        layer_id = self._validate_layer(layer)
        if self.is_quantized:
            raise capability_mismatch(
                "paged attention cannot consume quantized KV storage without scale-aware kernels"
            )
        return self.k[layer_id], self.v[layer_id]

    def read(
        self,
        layer: int,
        page_ids: Sequence[int],
        *,
        group: int = 0,
        start: int,
        length: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        layer_id = self._validate_layer(layer)
        pages = self.validate_pages(page_ids, allow_sentinel=True, group=group)
        if length <= 0:
            return None, None
        spans = self._spans(pages, int(start), int(length))
        keys = [
            self._read_span(self.k, self.k_scale, layer_id, page, offset, count)
            for page, offset, count in spans
        ]
        values = [
            self._read_span(self.v, self.v_scale, layer_id, page, offset, count)
            for page, offset, count in spans
        ]
        if len(keys) == 1:
            return keys[0], values[0]
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def write(
        self,
        layer: int,
        page_ids: Sequence[int],
        *,
        group: int = 0,
        start: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        layer_id = self._validate_layer(layer)
        pages = self.validate_pages(page_ids, group=group)
        if k.shape != v.shape:
            raise invalid_descriptor("KV write key/value tensors do not align")
        if k.ndim != 3 or tuple(k.shape[1:]) != (self.n_kv, self.head_dim):
            raise invalid_descriptor("KV write tensors must have shape [tokens, heads, dim]")
        written = 0
        for page, offset, count in self._spans(pages, int(start), int(k.shape[0])):
            self._write_span(
                self.k,
                self.k_scale,
                self.k_scale_set,
                layer_id,
                page,
                offset,
                k[written : written + count],
            )
            self._write_span(
                self.v,
                self.v_scale,
                self.v_scale_set,
                layer_id,
                page,
                offset,
                v[written : written + count],
            )
            written += count

    def _validate_layer(self, layer: int) -> int:
        value = int(layer)
        if value < 0 or value >= self.num_layers:
            raise invalid_descriptor(f"layer {value} outside KV pool layers {self.num_layers}")
        return value

    def _spans(
        self,
        page_ids: Sequence[int],
        start: int,
        length: int,
    ) -> tuple[tuple[int, int, int], ...]:
        if start < 0 or length < 0 or start + length > len(page_ids) * self.block_size:
            raise invalid_descriptor("KV token range exceeds its scheduler block table")
        spans: list[tuple[int, int, int]] = []
        cursor = start
        remaining = length
        while remaining:
            page_slot = cursor // self.block_size
            offset = cursor % self.block_size
            count = min(remaining, self.block_size - offset)
            spans.append((page_ids[page_slot], offset, count))
            cursor += count
            remaining -= count
        return tuple(spans)

    def _read_span(
        self,
        store: torch.Tensor,
        scales: torch.Tensor | None,
        layer: int,
        page: int,
        offset: int,
        count: int,
    ) -> torch.Tensor:
        span = store[layer, page, offset : offset + count]
        if not self.is_quantized:
            return span
        if scales is None:
            raise compute_error("quantized KV storage has no scale table", phase="kv_read")
        return dequantize_fp8_block(span, scales[layer, page], dtype=self.dtype)

    def _write_span(
        self,
        store: torch.Tensor,
        scales: torch.Tensor | None,
        scale_set: torch.Tensor | None,
        layer: int,
        page: int,
        offset: int,
        values: torch.Tensor,
    ) -> None:
        count = int(values.shape[0])
        if not self.is_quantized:
            store[layer, page, offset : offset + count] = values.to(
                device=store.device,
                dtype=store.dtype,
            )
            return
        if scales is None or scale_set is None:
            raise compute_error("quantized KV storage has no scale state", phase="kv_write")
        values_f32 = values.to(device=store.device, dtype=torch.float32)
        candidate = scale_for_fp8_block(values_f32).to(device=store.device)
        scale = torch.where(
            scale_set[layer, page],
            scales[layer, page].to(device=store.device),
            candidate,
        )
        scales[layer, page] = scale.to(device=scales.device)
        scale_set[layer, page] = True
        store[layer, page, offset : offset + count] = fp8_quantize(values_f32, scale)


class CacheBatchView:
    """Ephemeral model-facing view over operation-local cache rows."""

    def __init__(
        self,
        pool: CachePool,
        rows: Sequence[CacheRow],
        query_lengths: Sequence[int] | None = None,
        write_rows: Sequence[bool] | None = None,
    ) -> None:
        if not rows:
            raise invalid_descriptor("KV batch view requires at least one row")
        self._pool = pool
        self._rows = tuple(rows)
        self._block_ids = tuple(
            pool.validate_pages(
                row.block_table,
                allow_sentinel=True,
                group=row.group_id,
            )
            for row in rows
        )
        self._base_lens = tuple(int(row.length) for row in rows)
        self._query_lens = (
            None if query_lengths is None else tuple(int(value) for value in query_lengths)
        )
        if self._query_lens is not None and len(self._query_lens) != len(rows):
            raise invalid_descriptor("KV query lengths do not align with rows")
        self._write_rows = (
            (True,) * len(rows)
            if write_rows is None
            else tuple(bool(value) for value in write_rows)
        )
        if len(self._write_rows) != len(rows):
            raise invalid_descriptor("KV write predicates do not align with rows")
        for row, query, write in zip(
            rows,
            self._query_lens or (0,) * len(rows),
            self._write_rows,
            strict=True,
        ):
            required = row.length + query if write else row.length
            if required > row.capacity:
                raise invalid_descriptor("KV row exceeds scheduler capacity")
        self._block_table_width = bucketed_page_count(max(map(len, self._block_ids)))
        self._block_tables: dict[torch.device, torch.Tensor] = {}
        self._cache_lengths: dict[torch.device, torch.Tensor] = {}

    @classmethod
    def from_validated_rows(
        cls,
        pool: CachePool,
        rows: Sequence[CacheRow],
        query_lengths: Sequence[int] | None = None,
        write_rows: Sequence[bool] | None = None,
    ) -> CacheBatchView:
        """Build a model view from rows validated at partition registration."""

        if not rows:
            raise invalid_descriptor("KV batch view requires at least one row")
        view = cls.__new__(cls)
        view._pool = pool
        view._rows = tuple(rows)
        view._block_ids = tuple(row.block_table for row in rows)
        view._base_lens = tuple(int(row.length) for row in rows)
        view._query_lens = (
            None if query_lengths is None else tuple(int(value) for value in query_lengths)
        )
        if view._query_lens is not None and len(view._query_lens) != len(rows):
            raise invalid_descriptor("KV query lengths do not align with rows")
        view._write_rows = (
            (True,) * len(rows)
            if write_rows is None
            else tuple(bool(value) for value in write_rows)
        )
        if len(view._write_rows) != len(rows):
            raise invalid_descriptor("KV write predicates do not align with rows")
        for row, query, write in zip(
            rows,
            view._query_lens or (0,) * len(rows),
            view._write_rows,
            strict=True,
        ):
            required = row.length + query if write else row.length
            if required > row.capacity:
                raise invalid_descriptor("KV row exceeds scheduler capacity")
        view._block_table_width = bucketed_page_count(max(map(len, view._block_ids)))
        view._block_tables = {}
        view._cache_lengths = {}
        return view

    @property
    def block_size(self) -> int:
        return self._pool.block_size

    @property
    def pool(self) -> CachePool:
        return self._pool

    @property
    def supports_paged_attention_storage(self) -> bool:
        return self._pool.supports_paged_attention_storage

    @property
    def base_lens(self) -> tuple[int, ...]:
        return self._base_lens

    @property
    def query_lens(self) -> tuple[int, ...]:
        if self._query_lens is None:
            raise invalid_descriptor("KV view has no packed query lengths")
        return self._query_lens

    def with_synthetic_row(
        self,
        block_ids: Sequence[int],
        *,
        base_len: int,
        query_len: int,
    ) -> CacheBatchView:
        return self.with_synthetic_rows(
            block_ids,
            count=1,
            base_len=base_len,
            query_len=query_len,
        )

    def with_synthetic_rows(
        self,
        block_ids: Sequence[int],
        *,
        count: int,
        base_len: int,
        query_len: int,
    ) -> CacheBatchView:
        """Extend a validated view with repeated bounded padding rows."""

        if self._query_lens is None:
            raise invalid_descriptor("synthetic KV rows require declared query lengths")
        row_count = int(count)
        if row_count < 0:
            raise invalid_descriptor("synthetic KV row count must not be negative")
        if row_count == 0:
            return self
        pages = self._pool.validate_pages(
            block_ids,
            allow_sentinel=True,
            group=self._rows[0].group_id,
        )
        rows = tuple(
            CacheRow(
                pages,
                int(base_len),
                len(pages) * self.block_size,
                group_id=self._rows[0].group_id,
            )
            for _ in range(row_count)
        )
        return CacheBatchView.from_validated_rows(
            self._pool,
            (*self._rows, *rows),
            (*self._query_lens, *((int(query_len),) * row_count)),
            (*self._write_rows, *((True,) * row_count)),
        )

    def layer_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        groups = {row.group_id for row in self._rows}
        if len(groups) != 1:
            raise invalid_descriptor("one attention call cannot mix KV groups")
        return self._pool.layer_cache(layer, groups.pop())

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape != value.shape or key.ndim != 4 or int(key.shape[0]) != len(self._rows):
            raise invalid_descriptor("batched KV append tensors do not align with rows")
        for index, row in enumerate(self._rows):
            if self._write_rows[index]:
                self._pool.write(
                    layer,
                    row.block_table,
                    group=row.group_id,
                    start=row.length,
                    k=key[index],
                    v=value[index],
                )

    def append_varlen(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        row_lengths: Sequence[int],
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        query_offsets: torch.Tensor,
    ) -> None:
        lengths = tuple(int(item) for item in row_lengths)
        if key.shape != value.shape or key.ndim != 3 or sum(lengths) != int(key.shape[0]):
            raise invalid_descriptor("ragged KV append tensors do not align with rows")
        if not self._pool.is_quantized and all(self._write_rows):
            row_count = len(lengths)
            token_count = int(key.shape[0])
            if (
                block_table.ndim != 2
                or int(block_table.shape[0]) != row_count
                or tuple(cache_seqlens.shape) != (row_count,)
                or tuple(query_offsets.shape) != (row_count + 1,)
            ):
                raise invalid_descriptor("ragged KV write metadata does not align with rows")
            token_indices = torch.arange(token_count, device=key.device, dtype=torch.int64)
            offsets = query_offsets.to(device=key.device, dtype=torch.int64)
            row_indices = torch.searchsorted(offsets[1:], token_indices, right=True)
            positions = cache_seqlens.to(device=key.device, dtype=torch.int64).index_select(
                0, row_indices
            )
            positions += token_indices - offsets.index_select(0, row_indices)
            page_slots = torch.div(positions, self.block_size, rounding_mode="floor")
            page_ids = block_table.to(device=key.device).to(dtype=torch.int64)[
                row_indices, page_slots
            ]
            key_cache, value_cache = self.layer_kv(layer)
            paged_kv_write(
                key_cache,
                value_cache,
                page_ids,
                torch.remainder(positions, self.block_size),
                key,
                value,
                cast=key.dtype != key_cache.dtype or value.dtype != value_cache.dtype,
            )
            return
        offset = 0
        for row, length, write in zip(self._rows, lengths, self._write_rows, strict=True):
            if write and length:
                self._pool.write(
                    layer,
                    row.block_table,
                    group=row.group_id,
                    start=row.length,
                    k=key[offset : offset + length],
                    v=value[offset : offset + length],
                )
            offset += length

    def append_packed(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> None:
        if key.shape != value.shape or key.ndim != 3:
            raise invalid_descriptor("packed KV values must align as [tokens, heads, dim]")
        written = sum(
            query for query, write in zip(self.query_lens, self._write_rows, strict=True) if write
        )
        if not all(
            tuple(tensor.shape) == (written,) for tensor in (page_ids, page_offsets, token_indices)
        ):
            raise invalid_descriptor("packed KV write plan does not match writable tokens")
        if not self._pool.is_quantized:
            key_cache, value_cache = self.layer_kv(layer)
            source_key = key.index_select(0, token_indices)
            source_value = value.index_select(0, token_indices)
            paged_kv_write(
                key_cache,
                value_cache,
                page_ids,
                page_offsets,
                source_key,
                source_value,
                cast=source_key.dtype != key_cache.dtype or source_value.dtype != value_cache.dtype,
            )
            return
        offset = 0
        for row, query, write in zip(self._rows, self.query_lens, self._write_rows, strict=True):
            end = offset + query
            if write:
                self._pool.write(
                    layer,
                    row.block_table,
                    group=row.group_id,
                    start=row.length,
                    k=key[offset:end],
                    v=value[offset:end],
                )
            offset = end

    def block_table(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        target = torch.device(device)
        cached = self._block_tables.get(target)
        if cached is not None:
            return cached
        cpu = cpu_int_staging_buffer(
            len(self._rows) * self._block_table_width,
            dtype=torch.int32,
            pin=target.type == "cuda",
        )
        offset = 0
        for pages in self._block_ids:
            count = len(pages)
            fill_cpu_ints(cpu[offset : offset + count], pages)
            cpu[offset + count : offset + self._block_table_width].zero_()
            offset += self._block_table_width
        result = copy_cpu_to_device(
            cpu,
            device=target,
            non_blocking=target.type == "cuda" and is_pinned(cpu),
        ).view(len(self._rows), self._block_table_width)
        self._block_tables[target] = result
        return result

    def cache_seqlens(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        target = torch.device(device)
        cached = self._cache_lengths.get(target)
        if cached is not None:
            return cached
        cpu = cpu_int_staging_buffer(
            len(self._base_lens),
            dtype=torch.int32,
            pin=target.type == "cuda",
        )
        fill_cpu_ints(cpu, self._base_lens)
        result = copy_cpu_to_device(
            cpu,
            device=target,
            non_blocking=target.type == "cuda" and is_pinned(cpu),
        )
        self._cache_lengths[target] = result
        return result

    def seqused_k(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [row.length + query for row, query in zip(self._rows, self.query_lens, strict=True)],
            dtype=torch.int32,
            device=device,
        )

    def write_plan(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pages: list[int] = []
        offsets: list[int] = []
        selected: list[int] = []
        flat = 0
        for row, query, write in zip(self._rows, self.query_lens, self._write_rows, strict=True):
            if write:
                for local in range(query):
                    position = row.length + local
                    pages.append(row.block_table[position // self.block_size])
                    offsets.append(position % self.block_size)
                    selected.append(flat + local)
            flat += query
        return (
            torch.tensor(pages, dtype=torch.int64, device=device),
            torch.tensor(offsets, dtype=torch.int64, device=device),
            torch.tensor(selected, dtype=torch.long, device=device),
        )
