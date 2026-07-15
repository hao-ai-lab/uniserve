"""Physical paged KV pool shared by runner-backed models.

The host owns logical block ids; the worker owns the resident tensors.  This
pool translates `(block_id, layer, slot)` into page-first K/V storage and
exposes a per-request cache view with the small `get`/`append` interface used by
model attention code.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from ..foundation.errors import capability_mismatch, invalid_descriptor, model_execution_error
from ..nn.quant.kv_cache import (
    dequantize_fp8_block,
    fp8_quantize,
    is_fp8_kv_dtype,
    resolve_kv_store_dtype,
    scale_for_fp8_block,
)

__all__ = [
    'PagedKVPool',
    'PagedRequestCache',
]


@dataclass
class _VarlenAppendPlan:
    key: tuple[object, ...]
    page_ids: torch.Tensor
    offsets: torch.Tensor


class PagedKVPool:
    """Layer-major paged KV storage indexed by host-issued block ids."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        store_dtype: torch.dtype | str | None = None,
        tower_coord: int | None = None,
        reserved_tail_blocks: int = 0,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.n_kv = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.store_dtype = resolve_kv_store_dtype(dtype, store_dtype)
        self.reserved_tail_blocks = int(reserved_tail_blocks)
        # The tower coordinate this pool's storage is Pinned to (``None`` == the
        # primary/shared coordinate). The KV cache as a placed tensor: a gen-tower
        # scratch pool records ``tower_coord=gen`` so the snapshot reshard knows
        # the destination coordinate. Carries no behavior on the single-device path.
        self.tower_coord = None if tower_coord is None else int(tower_coord)
        # FP8 store is a STORAGE-COMPRESSION-ONLY feature: K/V are quantized on
        # write and dequantized on read with a single per-(layer, block) scale
        # that is established ONCE (on the block's first write) and reused for
        # every subsequent append to that block, so appending a token does not
        # dequantize+rescale+requantize the whole block.  The paged attention
        # kernel cannot consume the quantized pages (see ``layer_cache``); only
        # the dequantized request-cache read path is supported by design.
        self.is_quantized = is_fp8_kv_dtype(self.store_dtype)
        self.supports_paged_attention_storage = not self.is_quantized
        if self.num_layers <= 0 or self.num_blocks <= 0 or self.block_size <= 0:
            raise invalid_descriptor("PagedKVPool dimensions must be positive")
        if self.reserved_tail_blocks < 0 or self.reserved_tail_blocks >= self.num_blocks:
            raise invalid_descriptor(
                "PagedKVPool reserved tail blocks must leave at least one schedulable block"
            )
        # Layer-major storage makes a single layer's page table contiguous for
        # flash-attn's paged-kv kernel: [num_blocks, page, kv_heads, head_dim].
        shape = (self.num_layers, self.num_blocks, self.block_size, self.n_kv, self.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=self.store_dtype)
        self.v = torch.zeros(shape, device=device, dtype=self.store_dtype)
        scale_shape = (self.num_layers, self.num_blocks, 1, 1, 1)
        self.k_scale = (
            torch.ones(scale_shape, device=device, dtype=torch.float32) if self.is_quantized else None
        )
        self.v_scale = (
            torch.ones(scale_shape, device=device, dtype=torch.float32) if self.is_quantized else None
        )
        # Tracks whether each (layer, block) scale has been established by a
        # first write; once set, the scale is frozen and reused for appends.
        self.k_scale_set = (
            torch.zeros(
                (self.num_layers, self.num_blocks), device=device, dtype=torch.bool
            )
            if self.is_quantized
            else None
        )
        self.v_scale_set = (
            torch.zeros(
                (self.num_layers, self.num_blocks), device=device, dtype=torch.bool
            )
            if self.is_quantized
            else None
        )

    @property
    def schedulable_num_blocks(self) -> int:
        return self.num_blocks - self.reserved_tail_blocks

    @property
    def reserved_block_ids(self) -> tuple[int, ...]:
        return tuple(range(self.schedulable_num_blocks, self.num_blocks))

    def view(self, block_ids: Iterable[int], base_len: int) -> "PagedRequestCache":
        ids = self.validate_block_ids(block_ids)
        return PagedRequestCache(self, ids, int(base_len))

    def validate_block_ids(self, block_ids: Iterable[int]) -> list[int]:
        ids = [int(block_id) for block_id in block_ids]
        # Bound the whole list with two C-level scans instead of a per-element
        # Python branch on the hot view()/append() path; only fall back to the
        # per-element loop (cold) to surface the offending id in the error.
        if ids and (min(ids) < 0 or max(ids) >= self.num_blocks):
            for block_id in ids:
                if block_id < 0 or block_id >= self.num_blocks:
                    raise invalid_descriptor(
                        f"block id {block_id} outside pool capacity {self.num_blocks}"
                    )
        return ids

    def spans(
        self,
        block_ids: list[int],
        start: int,
        n: int,
    ) -> list[tuple[int, int, int]]:
        if n <= 0:
            return []
        if start < 0:
            raise invalid_descriptor("paged KV range start must be non-negative")
        end = int(start) + int(n)
        if end > len(block_ids) * self.block_size:
            raise invalid_descriptor(
                "paged KV view does not contain enough logical blocks "
                f"for range [{start}, {end})"
            )
        bs = self.block_size
        out: list[tuple[int, int, int]] = []
        while n > 0:
            bidx = start // bs
            off = start % bs
            cnt = min(n, bs - off)
            out.append((block_ids[bidx], off, cnt))
            start += cnt
            n -= cnt
        return out

    def read(
        self,
        layer: int,
        block_ids: list[int],
        *,
        start: int,
        length: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Read (k, v) for a token range. **The result must be treated read-only.**

        This is the contract every caller relies on: the returned tensors MAY
        alias the live pool storage (when the range fits one unquantized block,
        the common decode case, the slice is returned directly to avoid a
        ``torch.cat`` copy) or MAY be a fresh tensor (multi-block spans require
        ``torch.cat``; the quantized store dequantizes into a new tensor). Which
        one you get is data-dependent, so a caller that mutated the result in
        place would corrupt resident KV only sometimes — a heisenbug. Callers
        must consume the result read-only (the sole consumer immediately
        re-concatenates with freshly computed tokens and never writes into it).
        """
        layer = int(layer)
        if layer < 0 or layer >= self.num_layers:
            raise invalid_descriptor(f"layer {layer} outside KV pool layers {self.num_layers}")
        if length <= 0:
            return None, None
        spans = self.spans(block_ids, int(start), int(length))
        if len(spans) == 1:
            blk, off, cnt = spans[0]
            return (
                self._read_span(self.k, self.k_scale, layer, blk, off, cnt),
                self._read_span(self.v, self.v_scale, layer, blk, off, cnt),
            )
        ks, vs = [], []
        for blk, off, cnt in spans:
            ks.append(self._read_span(self.k, self.k_scale, layer, blk, off, cnt))
            vs.append(self._read_span(self.v, self.v_scale, layer, blk, off, cnt))
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)

    def write(
        self,
        layer: int,
        block_ids: list[int],
        *,
        start: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        layer = int(layer)
        if layer < 0 or layer >= self.num_layers:
            raise invalid_descriptor(f"layer {layer} outside KV pool layers {self.num_layers}")
        if k.shape != v.shape:
            raise invalid_descriptor("KV write key/value tensors must have matching shape")
        if k.ndim != 3 or k.shape[1] != self.n_kv or k.shape[2] != self.head_dim:
            raise invalid_descriptor(
                "KV write tensors must have shape [tokens, num_kv_heads, head_dim]"
            )
        n = int(k.shape[0])
        written = 0
        for blk, off, cnt in self.spans(block_ids, int(start), n):
            self._write_span(self.k, self.k_scale, self.k_scale_set, layer, blk, off, k[written:written + cnt])
            self._write_span(self.v, self.v_scale, self.v_scale_set, layer, blk, off, v[written:written + cnt])
            written += cnt

    def layer_cache(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer = int(layer)
        if layer < 0 or layer >= self.num_layers:
            raise invalid_descriptor(f"layer {layer} outside KV pool layers {self.num_layers}")
        if self.is_quantized:
            raise capability_mismatch(
                "paged attention backends cannot consume quantized KV storage yet; "
                "use a dequantized request-cache read path or add a scale-aware paged "
                "attention backend"
            )
        return self.k[layer], self.v[layer]

    def _read_span(
        self,
        store: torch.Tensor,
        scale_table: torch.Tensor | None,
        layer: int,
        block_id: int,
        offset: int,
        count: int,
    ) -> torch.Tensor:
        span = store[layer, block_id, offset:offset + count]
        if not self.is_quantized:
            return span
        if scale_table is None:
            raise model_execution_error("quantized KV storage is missing its scale table")
        scale = scale_table[layer, block_id]
        return dequantize_fp8_block(span, scale, dtype=self.dtype)

    def _write_span(
        self,
        store: torch.Tensor,
        scale_table: torch.Tensor | None,
        scale_set: torch.Tensor | None,
        layer: int,
        block_id: int,
        offset: int,
        values: torch.Tensor,
    ) -> None:
        if not self.is_quantized:
            store[layer, block_id, offset:offset + int(values.shape[0])] = values.to(
                device=store.device,
                dtype=store.dtype,
            )
            return
        if scale_table is None or scale_set is None:
            raise model_execution_error("quantized KV storage is missing its scale table")
        # Storage-compression only: the block's quantization scale is set ONCE
        # from the first values written into the block and frozen thereafter.
        # Subsequent appends quantize against that established scale and write
        # only their own slots, so an append no longer dequantizes, rescales and
        # requantizes the whole block.
        count = int(values.shape[0])
        values_f32 = values.to(device=store.device, dtype=torch.float32)
        if not bool(scale_set[layer, block_id]):
            scale = scale_for_fp8_block(values_f32).to(device=store.device)
            scale_table[layer, block_id] = scale.to(device=scale_table.device)
            scale_set[layer, block_id] = True
        else:
            scale = scale_table[layer, block_id].to(device=store.device)
        store[layer, block_id, offset:offset + count] = fp8_quantize(values_f32, scale)


class PagedRequestCache:
    """Per-request view over pool blocks.

    ``base_len`` is the persistent KV length before the current op's new tokens.
    Appends write at that offset; the runner/model wrapper advances the logical
    length after the op succeeds.
    """

    def __init__(self, pool: PagedKVPool, block_ids: list[int], base_len: int) -> None:
        if base_len < 0:
            raise invalid_descriptor("PagedRequestCache.base_len must be non-negative")
        self.pool = pool
        self.block_ids = block_ids
        # base_len is already coerced at the sole construction site (view()).
        self.base_len = base_len
        self._read_spans = self._spans(0, self.base_len)
        self._block_table_cache: dict[torch.device, torch.Tensor] = {}
        self._cache_seqlens_cache: dict[torch.device, torch.Tensor] = {}
        self._append_plan: _VarlenAppendPlan | None = None

    @property
    def base_lens(self) -> tuple[int]:
        return (int(self.base_len),)

    def _target_device(self, device: torch.device | str | None = None) -> torch.device:
        return torch.device(device if device is not None else self.pool.k.device)

    def _spans(self, start: int, n: int) -> list[tuple[int, int, int]]:
        return self.pool.spans(self.block_ids, start, n)

    def length(self) -> int:
        return self.base_len

    def get(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.pool.read(layer, self.block_ids, start=0, length=self.base_len)

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.pool.write(layer, self.block_ids, start=self.base_len, k=k, v=v)

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
            raise invalid_descriptor("paged KV append key/value shapes must match")
        if k.ndim != 3:
            raise invalid_descriptor("paged KV append expects [tokens, heads, dim]")
        query_lens = tuple(int(length) for length in query_lens)
        if query_lens != (int(k.shape[0]),):
            raise invalid_descriptor("single-row paged KV append requires one full-row query length")
        if self._append_varlen_indexed(
            int(layer),
            k,
            v,
            total=int(k.shape[0]),
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        ):
            return
        self.append(layer, k, v)

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
        if int(block_table.shape[0]) != 1 or int(cache_seqlens.shape[0]) != 1:
            return False
        if int(cu_seqlens_q.numel()) != 2:
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
        token_offsets = torch.arange(int(total), device=block_table.device, dtype=torch.int64)
        positions = cache_seqlens[0].to(dtype=torch.int64) + token_offsets
        block_slots = torch.div(positions, int(self.pool.block_size), rounding_mode="floor")
        page_ids = block_table[0].to(dtype=torch.int64).index_select(0, block_slots).contiguous()
        offsets = torch.remainder(positions, int(self.pool.block_size)).contiguous()
        plan = _VarlenAppendPlan(key=key, page_ids=page_ids, offsets=offsets)
        self._append_plan = plan
        return plan

    def block_table(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        target = self._target_device(device)
        cached = self._block_table_cache.get(target)
        if cached is not None:
            return cached
        out = torch.tensor(
            self.block_ids,
            dtype=torch.int32,
            device=target,
        ).unsqueeze(0)
        self._block_table_cache[target] = out
        return out

    def cache_seqlens(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        target = self._target_device(device)
        cached = self._cache_seqlens_cache.get(target)
        if cached is not None:
            return cached
        out = torch.tensor(
            [self.base_len],
            dtype=torch.int32,
            device=target,
        )
        self._cache_seqlens_cache[target] = out
        return out
