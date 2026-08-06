"""Physical paged KV storage owned and bounded by ``KvStore``."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from ..foundation.errors import capability_mismatch, compute_error, invalid_descriptor
from ..nn.quant.kv_cache import (
    dequantize_fp8_block,
    fp8_quantize,
    is_fp8_kv_dtype,
    resolve_kv_store_dtype,
    scale_for_fp8_block,
)
from .block_allocator import BlockFreeList

__all__ = [
    "PagedKVPool",
]


class PagedKVPool:
    """Layer-major paged KV storage with worker-owned physical page allocation.

    One pool holds every span attention can address in a single forward. Its
    block ids partition into three ranges: the worker allocates ``[0,
    leasable_num_blocks)`` to registered session leases, allocates branch blocks
    for transaction-scoped spans immediately above that, and reserves the tail
    for padded graph rows. A batch that mixes a session's committed
    span with a branch span therefore builds one page table over one storage
    tensor, so no span has to be relocated to be attended to alongside another.
    """

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
        branch_blocks: int = 0,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.n_kv = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.store_dtype = resolve_kv_store_dtype(dtype, store_dtype)
        self.reserved_tail_blocks = int(reserved_tail_blocks)
        self.branch_num_blocks = int(branch_blocks)
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
        if self.branch_num_blocks < 0:
            raise invalid_descriptor("PagedKVPool branch blocks must not be negative")
        if self.leasable_num_blocks < 1:
            raise invalid_descriptor("PagedKVPool must leave at least one leasable block")
        self._branch_free = BlockFreeList(self.branch_num_blocks)
        self._session_free = BlockFreeList(self.leasable_num_blocks)
        # Layer-major storage makes a single layer's page table contiguous for
        # flash-attn's paged-kv kernel: [num_blocks, page, kv_heads, head_dim].
        shape = (self.num_layers, self.num_blocks, self.block_size, self.n_kv, self.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=self.store_dtype)
        self.v = torch.zeros(shape, device=device, dtype=self.store_dtype)
        scale_shape = (self.num_layers, self.num_blocks, 1, 1, 1)
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
        # Tracks whether each (layer, block) scale has been established by a
        # first write; once set, the scale is frozen and reused for appends.
        self.k_scale_set = (
            torch.zeros((self.num_layers, self.num_blocks), device=device, dtype=torch.bool)
            if self.is_quantized
            else None
        )
        self.v_scale_set = (
            torch.zeros((self.num_layers, self.num_blocks), device=device, dtype=torch.bool)
            if self.is_quantized
            else None
        )

    @property
    def leasable_num_blocks(self) -> int:
        """Physical pages available for worker session mappings."""

        return self.num_blocks - self.reserved_tail_blocks - self.branch_num_blocks

    @property
    def reserved_block_ids(self) -> tuple[int, ...]:
        """Tail block ids dedicated to graph-padding rows."""

        start = self.leasable_num_blocks + self.branch_num_blocks
        return tuple(range(start, self.num_blocks))

    @property
    def branch_blocks_available(self) -> int:
        return self._branch_free.available

    @property
    def session_blocks_available(self) -> int:
        return self._session_free.available

    def allocate_session_blocks(self, count: int) -> list[int]:
        """Select physical pages for newly observed logical session blocks."""

        return self._session_free.allocate(int(count), label="session KV pages")

    def release_session_blocks(self, block_ids: Iterable[int]) -> None:
        values = [int(value) for value in block_ids]
        if any(value < 0 or value >= self.leasable_num_blocks for value in values):
            raise invalid_descriptor("session KV page is outside this pool's session range")
        self._session_free.release(values)

    def allocate_branch_blocks(self, count: int) -> list[int]:
        """Take ``count`` transaction-branch blocks from this pool's own range."""

        base = self.leasable_num_blocks
        return [
            base + value
            for value in self._branch_free.allocate(int(count), label="branch KV blocks")
        ]

    def release_branch_blocks(self, block_ids: Iterable[int]) -> None:
        base = self.leasable_num_blocks
        values = [int(value) for value in block_ids]
        if any(value < base or value >= base + self.branch_num_blocks for value in values):
            raise invalid_descriptor("branch KV block id is outside this pool's branch range")
        self._branch_free.release(value - base for value in values)

    def copy_pages(
        self,
        source_blocks: Sequence[int],
        target_blocks: Sequence[int],
    ) -> None:
        """Duplicate whole pages within this pool, one gather-scatter per store.

        Page granularity keeps the transfer independent of how many tokens the
        span holds within its last page, and it carries any per-page quantization
        scale along with the page it describes.
        """

        if len(source_blocks) != len(target_blocks):
            raise invalid_descriptor("paged KV page copy needs one target page per source page")
        if not source_blocks:
            return
        device = self.k.device
        source = torch.tensor(
            self.validate_block_ids(source_blocks), dtype=torch.long, device=device
        )
        target = torch.tensor(
            self.validate_block_ids(target_blocks), dtype=torch.long, device=device
        )
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
                f"paged KV view does not contain enough logical blocks for range [{start}, {end})"
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
            self._write_span(
                self.k, self.k_scale, self.k_scale_set, layer, blk, off, k[written : written + cnt]
            )
            self._write_span(
                self.v, self.v_scale, self.v_scale_set, layer, blk, off, v[written : written + cnt]
            )
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
        span = store[layer, block_id, offset : offset + count]
        if not self.is_quantized:
            return span
        if scale_table is None:
            raise compute_error(
                "quantized KV storage is missing its scale table",
                phase="kv_read",
            )
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
            store[layer, block_id, offset : offset + int(values.shape[0])] = values.to(
                device=store.device,
                dtype=store.dtype,
            )
            return
        if scale_table is None or scale_set is None:
            raise compute_error(
                "quantized KV storage is missing its scale table",
                phase="kv_write",
            )
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
        store[layer, block_id, offset : offset + count] = fp8_quantize(values_f32, scale)
