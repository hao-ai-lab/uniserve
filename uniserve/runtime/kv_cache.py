"""Numerical KV backing and borrowed layer views, independent of requests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from uniserve.nn.quant.kv_cache import (
    FP8_MAX,
    SCALE_EPS,
    dequantize_fp8_block,
    fp8_quantize,
    scale_for_fp8_block,
)
from uniserve.runtime.paged_kv_math import paged_kv_write

__all__ = ["KVCacheConfig", "KVLayer", "KVCache"]


@dataclass(frozen=True, slots=True)
class KVCacheConfig:
    """Local layer/head partition and numerical representation of cached K/V."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    total_layers: int
    total_kv_heads: int
    layer_offset: int = 0
    kv_head_offset: int = 0
    store_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"KV {name} must be a positive integer")
        for offset, count, total in (
            (self.layer_offset, self.num_layers, self.total_layers),
            (self.kv_head_offset, self.num_kv_heads, self.total_kv_heads),
        ):
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or not isinstance(total, int)
                or isinstance(total, bool)
                or offset < 0
                or offset + count > total
            ):
                raise ValueError("KV partition lies outside its logical layer/head extent")
        if self.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("unsupported KV computation dtype")
        if self.store_dtype is not None and self.store_dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float8_e4m3fn,
        }:
            raise ValueError("unsupported KV storage dtype")
        if self.store_dtype is torch.float8_e4m3fn and (
            self.total_kv_heads % self.num_kv_heads or self.kv_head_offset % self.num_kv_heads
        ):
            raise ValueError("FP8 KV heads must form aligned, uniform scale groups")


def page_spans(
    page_ids: Sequence[int], start: int, length: int, page_size: int
) -> tuple[tuple[int, int, int], ...]:
    """Split a logical token interval into physical page, offset, and count triples."""

    if start < 0 or length < 0 or start + length > len(page_ids) * page_size:
        raise ValueError("KV token interval exceeds its page table")
    spans = []
    while length:
        slot, offset = divmod(start, page_size)
        count = min(length, page_size - offset)
        spans.append((page_ids[slot], offset, count))
        start += count
        length -= count
    return tuple(spans)


@dataclass(frozen=True, slots=True)
class KVLayer:
    """Borrowed [pages, tokens, heads, dim] K/V and optional FP8 scale tensors.

    The allocator owns all backing. Callers authorize writes and keep these
    views alive until kernels and asynchronous readers finish. Scale flags are
    borrowed CPU tensors; first-write scale selection never rereads GPU state.
    """

    k: torch.Tensor
    v: torch.Tensor
    dtype: torch.dtype
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None
    k_initialized: torch.Tensor | None = None
    v_initialized: torch.Tensor | None = None

    @property
    def is_quantized(self) -> bool:
        return self.k.dtype is torch.float8_e4m3fn

    def _write_span(
        self,
        store: torch.Tensor,
        scales: torch.Tensor | None,
        initialized: torch.Tensor | None,
        page: int,
        offset: int,
        values: torch.Tensor,
    ) -> None:
        count = values.shape[0]
        if not self.is_quantized:
            store[page, offset : offset + count].copy_(values)
            return
        if scales is None or initialized is None:
            raise RuntimeError("quantized KV storage has no scale state")
        values = values.to(device=store.device, dtype=torch.float32)
        scale = scales[page]
        if not initialized[page]:
            scale.copy_(scale_for_fp8_block(values))
            initialized[page] = True
        store[page, offset : offset + count].copy_(fp8_quantize(values, scale))

    def read(
        self, page_ids: Sequence[int], *, start: int, length: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Gather and, for FP8 storage, dequantize a numerical token interval."""

        if not length:
            return None, None
        spans = page_spans(page_ids, start, length, self.k.shape[1])
        outputs = []
        for store, scales in ((self.k, self.k_scale), (self.v, self.v_scale)):
            values = []
            for page, offset, count in spans:
                if not 0 <= page < store.shape[0]:
                    raise ValueError("KV page index is outside storage")
                value = store[page, offset : offset + count]
                if self.is_quantized:
                    if scales is None:
                        raise RuntimeError("quantized KV storage has no scale table")
                    value = dequantize_fp8_block(value, scales[page], dtype=self.dtype)
                values.append(value)
            outputs.append(values[0] if len(values) == 1 else torch.cat(values, dim=0))
        return outputs[0], outputs[1]

    def write(
        self, page_ids: Sequence[int], *, start: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Scatter K/V into caller-authorized pages, preserving existing FP8 scales."""

        if k.shape != v.shape or k.ndim != 3 or k.shape[1:] != self.k.shape[2:]:
            raise ValueError("KV write tensors must have matching [tokens, heads, dim] shapes")
        written = 0
        for page, offset, count in page_spans(page_ids, start, k.shape[0], self.k.shape[1]):
            if not 0 <= page < self.k.shape[0]:
                raise ValueError("KV page index is outside storage")
            self._write_span(
                self.k, self.k_scale, self.k_initialized, page, offset, k[written : written + count]
            )
            self._write_span(
                self.v, self.v_scale, self.v_initialized, page, offset, v[written : written + count]
            )
            written += count

    def write_locations(self, locations: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter packed rows at encoded token slots; nonpositive slots do not write."""

        locations = locations.reshape(-1).to(device=k.device, dtype=torch.int64)
        if k.shape != v.shape or k.ndim != 3 or k.shape[0] != locations.numel():
            raise ValueError("KV output locations do not align with current K/V")
        if not self.is_quantized:
            paged_kv_write(
                self.k,
                self.v,
                locations,
                None,
                k,
                v,
                cast=k.dtype != self.k.dtype or v.dtype != self.v.dtype,
            )
            return
        selected = torch.nonzero(locations > 0, as_tuple=False).reshape(-1)
        encoded = locations.index_select(0, selected)
        page_size = self.k.shape[1]
        pages = torch.div(encoded, page_size, rounding_mode="floor")
        offsets = torch.remainder(encoded, page_size)
        selected_k, selected_v = k.index_select(0, selected), v.index_select(0, selected)
        for index in range(selected.numel()):
            page, offset = int(pages[index].item()), int(offsets[index].item())
            self._write_span(
                self.k,
                self.k_scale,
                self.k_initialized,
                page,
                offset,
                selected_k[index : index + 1],
            )
            self._write_span(
                self.v,
                self.v_scale,
                self.v_initialized,
                page,
                offset,
                selected_v[index : index + 1],
            )


class KVCache:
    """Own numerical K/V backing; requests, page assignment, and transfers live outside."""

    def __init__(
        self, config: KVCacheConfig, *, num_pages: int, page_size: int, device: torch.device | str
    ) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (num_pages, page_size)
        ):
            raise ValueError("KV page count and page size must be positive integers")
        self.config = config
        self.num_pages = num_pages
        self.page_size = page_size
        shape = (config.num_layers, num_pages, page_size, config.num_kv_heads, config.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=config.store_dtype or config.dtype)
        self.v = torch.zeros_like(self.k)
        self._scales = (
            torch.ones(
                (2, config.num_layers, num_pages, 1, 1, 1), dtype=torch.float32, device=device
            )
            if self.is_quantized
            else None
        )
        self._initialized = (
            torch.zeros((2, config.num_layers, num_pages), dtype=torch.bool, device="cpu")
            if self.is_quantized
            else None
        )
        self._layers = tuple(
            KVLayer(
                self.k[index],
                self.v[index],
                config.dtype,
                None if self._scales is None else self._scales[0, index],
                None if self._scales is None else self._scales[1, index],
                None if self._initialized is None else self._initialized[0, index],
                None if self._initialized is None else self._initialized[1, index],
            )
            for index in range(config.num_layers)
        )

    @property
    def is_quantized(self) -> bool:
        return self.k.dtype is torch.float8_e4m3fn

    def layer(self, index: int) -> KVLayer:
        """Borrow one local numerical layer for attention or explicit cache operations."""

        if not 0 <= index < len(self._layers):
            raise ValueError("KV layer index is outside storage")
        return self._layers[index]

    def zero_pages(self, pages: Iterable[int]) -> None:
        """Clear caller-owned pages and their first-write scale flags."""

        ranges: list[tuple[int, int]] = []
        for page in sorted(set(pages)):
            if not 0 <= page < self.num_pages:
                raise ValueError("KV page index is outside storage")
            if ranges and ranges[-1][1] == page:
                ranges[-1] = (ranges[-1][0], page + 1)
            else:
                ranges.append((page, page + 1))
        for start, end in ranges:
            for store in (self.k, self.v):
                values = store.view(torch.uint8) if self.is_quantized else store
                values[:, start:end].zero_()
            if self._scales is not None:
                self._scales[:, :, start:end].fill_(1)
                assert self._initialized is not None
                self._initialized[:, :, start:end].fill_(False)

    def mark_initialized(self, pages: Iterable[int]) -> None:
        """Record that externally supplied page scales are numerically initialized."""

        if self._initialized is not None:
            for page in pages:
                self._initialized[:, :, page].fill_(True)

    def transfer_views(
        self, page_ids: Sequence[int], *, start: int, length: int
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Borrow discontiguous K/V intervals and scales without copying or registration.

        K/V views are [tokens, layers, heads, dim]; scale views are
        [pages, 2, layers, 1]. The caller retains backing until all readers retire.
        """

        spans = page_spans(page_ids, start, length, self.page_size)
        if not spans:
            return ()
        if any(not 0 <= page < self.num_pages for page, _, _ in spans):
            raise ValueError("KV page index is outside storage")
        fields = tuple(
            tuple(
                store[:, page, offset : offset + count].permute(1, 0, 2, 3)
                for page, offset, count in spans
            )
            for store in (self.k, self.v)
        )
        if self._scales is not None:
            fields += (
                tuple(
                    self._scales[:, :, page, 0, 0, 0].unsqueeze(0).unsqueeze(-1)
                    for page, _, _ in spans
                ),
            )
        return fields

    def close(self) -> None:
        """Release owned references after the caller retires all borrowed views."""

        self._layers = ()
        self._scales = self._initialized = None
        self.k = self.k.new_empty(0)
        self.v = self.v.new_empty(0)

    def copy_page(
        self,
        source: torch.Tensor,
        *,
        field: int,
        page: int,
        offset: int,
        source_offset: int,
        source_page_size: int,
        source_head_size: int,
        source_compute_dtype: torch.dtype,
        source_scales: torch.Tensor,
        values_buffer: torch.Tensor,
        raw_buffer: torch.Tensor,
        scale_buffer: torch.Tensor,
    ) -> None:
        """Copy one page interval across numerical representations and head partitions.

        Scratch tensors are borrowed. FP8 sources round through their computation
        dtype before destination quantization; initialized destination scales stay fixed.
        """

        count = int(source.shape[0])
        store = self.k if field == 0 else self.v
        destination = store[:, page, offset : offset + count]
        source_quantized = source.dtype is torch.float8_e4m3fn
        if not source_quantized and not self.is_quantized:
            destination.copy_(source.permute(1, 0, 2, 3))
            return
        values = values_buffer[:count]
        values.copy_(source)
        elements = int(values.numel())
        if source_quantized:
            position = 0
            scale_index = 0
            page_size = source_page_size
            head_size = source_head_size
            while position < count:
                length = min(page_size - source_offset, count - position)
                head = 0
                group = 0
                while head < self.config.num_kv_heads:
                    heads = min(
                        head_size - (self.config.kv_head_offset + head) % head_size,
                        self.config.num_kv_heads - head,
                    )
                    scale = source_scales[scale_index, field, :, group].reshape(
                        1, self.config.num_layers, 1, 1
                    )
                    values[position : position + length, :, head : head + heads].mul_(scale)
                    head += heads
                    group += 1
                position += length
                scale_index += 1
                source_offset = 0
            dtype = source_compute_dtype
            if dtype in {torch.float16, torch.bfloat16}:
                rounded = raw_buffer[field, : elements * 2].view(dtype).reshape_as(values)
                rounded.copy_(values)
                values.copy_(rounded)
        if not self.is_quantized:
            destination.copy_(values.permute(1, 0, 2, 3))
            return
        scales = None if self._scales is None else self._scales[field]
        flags = None if self._initialized is None else self._initialized[field]
        assert scales is not None and flags is not None
        pending = tuple(layer for layer in range(self.config.num_layers) if not flags[layer, page])
        if pending:
            absolute = raw_buffer[field, : elements * 4].view(torch.float32).reshape_as(values)
            torch.abs(values, out=absolute)
            torch.amax(absolute, dim=(0, 2, 3), out=scale_buffer)
            scale_buffer.clamp_min_(SCALE_EPS).div_(FP8_MAX)
            for layer in pending:
                scales[layer, page].copy_(scale_buffer[layer])
                flags[layer, page] = 1
        values.div_(scales[:, page].reshape(1, self.config.num_layers, 1, 1))
        values.clamp_(-FP8_MAX, FP8_MAX)
        destination.copy_(values.permute(1, 0, 2, 3))
