"""Fixed physical KV storage indexed by scheduler-assigned page IDs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from ..backends.paged_kv_math import paged_kv_write
from ..foundation.errors import capability_mismatch, compute_error, invalid_descriptor
from ..nn.quant.kv_cache import (
    dequantize_fp8_block,
    fp8_quantize,
    is_fp8_kv_dtype,
    resolve_kv_store_dtype,
    scale_for_fp8_block,
)

__all__ = ["CachePool"]


class CachePool:
    """Startup-sized layer-major KV tensors with no allocation authority."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
        store_dtype: torch.dtype | str | None = None,
        group_ranges: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_pages = int(num_pages)
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
            or self.num_pages < 1
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
            tuple[tuple[int, ...], bool, int | None], tuple[int, ...]
        ] = {}

    def _group_ranges(
        self,
        declared: Sequence[tuple[int, int]] | None,
    ) -> tuple[tuple[int, int], ...]:
        ranges = (
            ((0, self.num_pages),)
            if declared is None
            else tuple((int(offset), int(count)) for offset, count in declared)
        )
        if not ranges:
            raise invalid_descriptor("CachePool declares no KV groups")
        covered = [False] * self.num_pages
        for group, (offset, count) in enumerate(ranges):
            end = offset + count
            if offset < 0 or count < 1 or end > self.num_pages:
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

    def page_ids(self, group: int) -> range:
        group_id = self.validate_group(group)
        offset, count = self.group_ranges[group_id]
        return range(max(1, offset), offset + count)

    def validate_pages(
        self,
        page_ids: Iterable[int],
        *,
        allow_sentinel: bool = False,
        group: int | None = None,
    ) -> tuple[int, ...]:
        pages = tuple(int(page) for page in page_ids)
        key = (pages, bool(allow_sentinel), group)
        cached = self._validated_page_tuples.get(key)
        if cached is not None:
            return cached
        validated = self._validate_page_tuple(pages, allow_sentinel, group)
        if len(self._validated_page_tuples) >= 16_384:
            self._validated_page_tuples.clear()
        self._validated_page_tuples[key] = validated
        return validated

    def _validate_page_tuple(
        self,
        pages: tuple[int, ...],
        allow_sentinel: bool,
        group: int | None,
    ) -> tuple[int, ...]:
        real_pages = tuple(page for page in pages if page != 0)
        if len(set(real_pages)) != len(real_pages):
            raise invalid_descriptor("KV placement repeats a physical page")
        lower = 0 if allow_sentinel else 1
        upper = self.num_pages
        if pages and (min(pages) < lower or max(pages) >= upper):
            raise invalid_descriptor("KV placement exceeds the fixed physical pool")
        if group is not None:
            group_id = self.validate_group(group)
            offset, count = self.group_ranges[group_id]
            end = offset + count
            if any(page < offset or page >= end for page in real_pages):
                raise invalid_descriptor("KV placement addresses another cache group")
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

    def write_locations(
        self,
        layer: int,
        locations: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Persist selected packed tokens at encoded physical token locations."""

        layer_id = self._validate_layer(layer)
        flat_locations = locations.reshape(-1).to(device=k.device, dtype=torch.int64)
        if k.shape != v.shape or k.ndim != 3 or int(k.shape[0]) != int(flat_locations.numel()):
            raise invalid_descriptor("KV output locations do not align with current K/V")
        if not self.is_quantized:
            persists = flat_locations > 0
            pages = torch.where(
                persists,
                torch.div(flat_locations, self.block_size, rounding_mode="floor"),
                -1,
            )
            offsets = torch.remainder(flat_locations, self.block_size)
            paged_kv_write(
                self.k[layer_id],
                self.v[layer_id],
                pages,
                offsets,
                k,
                v,
                cast=k.dtype != self.k.dtype or v.dtype != self.v.dtype,
            )
            return
        selected = torch.nonzero(flat_locations > 0, as_tuple=False).reshape(-1)
        if int(selected.numel()) == 0:
            return
        encoded = flat_locations.index_select(0, selected)
        pages = torch.div(encoded, self.block_size, rounding_mode="floor")
        offsets = torch.remainder(encoded, self.block_size)
        selected_k = k.index_select(0, selected)
        selected_v = v.index_select(0, selected)
        for index in range(int(selected.numel())):
            page = int(pages[index].item())
            offset = int(offsets[index].item())
            self._write_span(
                self.k,
                self.k_scale,
                self.k_scale_set,
                layer_id,
                page,
                offset,
                selected_k[index : index + 1],
            )
            self._write_span(
                self.v,
                self.v_scale,
                self.v_scale_set,
                layer_id,
                page,
                offset,
                selected_v[index : index + 1],
            )

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
