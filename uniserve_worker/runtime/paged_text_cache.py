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
from .kv_pool import PagedKVPool

__all__ = [
    'PagedTransformerLayer',
    'PagedTextCache',
    'BatchedPagedRequestCache',
    'BatchedPagedTextCache',
]


@dataclass
class _VarlenAppendPlan:
    key: tuple[Any, ...]
    page_ids: torch.Tensor
    offsets: torch.Tensor


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
        if self.length:
            self.ensure_capacity(self.length)

    def set_blocks(self, block_ids: list[int] | tuple[int, ...]) -> None:
        self.block_ids = self.pool.validate_block_ids(block_ids)
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
        return self.pool.view(self.block_ids, start)

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
        return self.pool.view(self.block_ids, start)

    def cancel_layer_update(self, layer_idx: int) -> None:
        self._updated_layers.discard(int(layer_idx))
        if not self._updated_layers:
            self._active_start = None
            self._active_count = None

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


class BatchedPagedRequestCache:
    """Batched transient page view over compatible request caches."""

    def __init__(
        self,
        pool: PagedKVPool,
        block_ids_by_row: Sequence[Sequence[int]],
        base_lens: Sequence[int],
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
        self._append_plan: _VarlenAppendPlan | None = None

    def block_table(
        self,
        *,
        device: torch.device | str | None = None,
        stager: BufferStager | None = None,
    ) -> torch.Tensor:
        max_blocks = max(len(ids) for ids in self.block_ids_by_row)
        row_count = len(self.block_ids_by_row)
        target = torch.device(device if device is not None else self.pool.k.device)
        cpu = _cpu_int_buffer(
            row_count * max_blocks,
            pin=target.type == "cuda",
            slot=stager,
            name="paged_block_table",
        )
        _fill_block_table(cpu, self.block_ids_by_row, max_blocks)
        non_blocking = target.type == "cuda" and _is_pinned(cpu)
        return _copy_cpu_int_to_device(
            cpu,
            device=target,
            non_blocking=non_blocking,
            slot=stager,
            name="paged_block_table",
        ).view(row_count, max_blocks)

    def cache_seqlens(
        self,
        *,
        device: torch.device | str | None = None,
        stager: BufferStager | None = None,
    ) -> torch.Tensor:
        target = torch.device(device if device is not None else self.pool.k.device)
        cpu = _cpu_int_buffer(
            len(self.base_lens),
            pin=target.type == "cuda",
            slot=stager,
            name="paged_cache_seqlens",
        )
        _fill_cpu_int(cpu, self.base_lens)
        non_blocking = target.type == "cuda" and _is_pinned(cpu)
        return _copy_cpu_int_to_device(
            cpu,
            device=target,
            non_blocking=non_blocking,
            slot=stager,
            name="paged_cache_seqlens",
        )

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

    def __init__(self, caches: Sequence[PagedTextCache]) -> None:
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
        return BatchedPagedRequestCache(
            self.pool,
            [cache.block_ids for cache in self.caches],
            [int(cache.length) for cache in self.caches],
        )


def _cpu_int_buffer(
    numel: int,
    *,
    pin: bool,
    slot: BufferStager | None = None,
    name: str = "buffer",
) -> torch.Tensor:
    # ``int_buffer`` is structurally guaranteed by BufferStager, so a non-None
    # stager always provides it (no callable() probe needed).
    if slot is not None:
        return slot.int_buffer(name, numel, pin=pin)
    if pin:
        try:
            return torch.empty(int(numel), dtype=torch.int32, pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(int(numel), dtype=torch.int32)


def _fill_cpu_int(cpu: torch.Tensor, values: Sequence[int]) -> None:
    for idx, value in enumerate(values):
        cpu[idx] = int(value)


def _fill_block_table(
    cpu: torch.Tensor,
    rows: Sequence[Sequence[int]],
    width: int,
) -> None:
    offset = 0
    width = int(width)
    for block_ids in rows:
        n = len(block_ids)
        for col, block_id in enumerate(block_ids):
            cpu[offset + col] = int(block_id)
        if n < width:
            cpu[offset + n: offset + width].zero_()
        offset += width


def _copy_cpu_int_to_device(
    cpu: torch.Tensor,
    *,
    device: torch.device,
    non_blocking: bool,
    slot: BufferStager | None,
    name: str,
) -> torch.Tensor:
    device = _canonical_device(device)
    # ``device_buffer`` is structurally guaranteed by BufferStager; the device
    # staging path is only taken for a CUDA target with a stager present.
    if slot is None or device.type != "cuda":
        return cpu.to(device=device, non_blocking=non_blocking)
    out = slot.device_buffer(name, int(cpu.numel()), dtype=cpu.dtype, device=device)
    out.copy_(cpu, non_blocking=non_blocking)
    return out


def _canonical_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _is_pinned(tensor: torch.Tensor) -> bool:
    return bool(getattr(tensor, "is_pinned", lambda: False)())
