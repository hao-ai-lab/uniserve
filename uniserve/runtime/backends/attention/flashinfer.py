"""FlashInfer attention with host planning and graph-stable metadata.

FlashInfer attention with explicit host planning and graph-stable page
metadata.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch
import triton
import triton.language as tl

from uniserve.nn.attention.inputs import (
    DenseInput,
    PagedInput,
    SegmentedInput,
    VarlenInput,
    VisibleInput,
)
from uniserve.quantization import QuantizedTensor
from uniserve.tensors import BufferConfig

from . import Backend as _Backend
from . import Operator as _Operator

__all__ = ["Backend", "Config"]


@contextmanager
def _plan_workspace(wrapper: Any) -> Iterator[None]:
    """Give a native plan an immutable pinned upload generation.

    FlashInfer writes this host workspace and enqueues its DMA directly. Its
    next plan may run before the previous upload has reached the device. A fresh
    allocation plus allocator stream tracking protects both reuse and teardown
    while allowing CPU planning to continue asynchronously.
    """
    from uniserve_kernels.peer_storage import record_host_usage

    source = torch.empty_like(
        wrapper._pin_memory_int_workspace_buffer, pin_memory=True
    )
    wrapper._pin_memory_int_workspace_buffer = source
    try:
        yield
    finally:
        record_host_usage(source, torch.cuda.current_stream(wrapper.device))


@dataclass(frozen=True)
class Config:
    """FlashInfer kernel and planning configuration.

    FlashInfer workspace, kernel families, split-KV policy and host planning.
    """

    workspace_size: int = 512 * 1024 * 1024
    use_tensor_core: bool | None = None
    decode_backend: str = "fa2"
    prefill_backend: str = "auto"
    decode_split_tile_size: int | None = None
    prefill_split_tile_size: int | None = None
    disable_split_kv: bool = False

    def __post_init__(self):
        if self.workspace_size < 1:
            raise ValueError("FlashInfer workspace size must be positive")
        for size in (self.decode_split_tile_size, self.prefill_split_tile_size):
            if size is not None and size < 1:
                raise ValueError("FlashInfer split tile sizes must be positive")


def _tensor_cores(config, num_heads, num_kv_heads):
    if config.use_tensor_core is not None:
        return config.use_tensor_core
    # Half-precision decode uses tensor cores from GQA ratio four onward.
    # This is the worker's numerical-provider policy; it does not depend on
    # optional private helpers exported by a particular FlashInfer build.
    return num_heads // num_kv_heads >= 4


@triton.jit
def _page_indices(
    table,
    counts,
    indptr,
    indices,
    row_stride: tl.constexpr,
    column_stride: tl.constexpr,
):
    # Copy one sequence's live page IDs into the compact CSR `indices` array
    # that native wrappers consume; `indptr` delimits each sequence's segment.
    row = tl.program_id(0)
    count = tl.load(counts + row)
    start = tl.load(indptr + row)
    columns = tl.arange(0, 256)

    for tile in range(tl.cdiv(count, 256)):
        offset = tile * 256 + columns
        values = tl.load(
            table + row * row_stride + offset * column_stride,
            offset < count,
            other=0,
        )
        tl.store(indices + start + offset, values, offset < count)


def _host(values):
    """Build a pinned int32 CPU tensor.

    Build a pinned int32 CPU tensor for a non-blocking plan metadata upload.
    """
    return torch.tensor(values, dtype=torch.int32, pin_memory=True)


class _PagePlan:
    """One native wrapper with fixed addresses and independently mutable plans.

    Host page counts come from declared lengths. Physical page IDs remain on
    the GPU and are refreshed by a captured kernel, so updating a block table
    never depends on Python object identity or requires a device-to-host copy.
    """

    def __init__(self, owner, *, queries, table, decode, custom):
        import flashinfer

        self.decode = decode
        self.device = owner.workspace["scratch"].device
        count = len(queries)
        self.capacity = table.indices.numel()

        with torch.inference_mode(False):
            self.query_offsets = torch.empty(
                count + 1, dtype=torch.int32, device=self.device
            )
            self.indptr = torch.empty_like(self.query_offsets)
            self.indices = torch.empty(
                max(1, self.capacity), dtype=torch.int32, device=self.device
            )
            self.last = torch.empty(
                count, dtype=torch.int32, device=self.device
            )
            self.counts = torch.empty_like(self.last)

            # Custom-mask metadata exists together, only for custom masks.
            self.mask: torch.Tensor | None
            self.mask_offsets: torch.Tensor | None
            self.key_offsets: torch.Tensor | None
            self.causal: torch.Tensor | None
            if custom:
                # Table capacity bounds any key distribution across these
                # query rows, including byte padding between sequences.
                bits = sum(queries) * table.indices.shape[1] * table.block_size
                self.mask = torch.empty(
                    (bits + 7) // 8 + count,
                    dtype=torch.uint8,
                    device=self.device,
                )
                self.mask_offsets = torch.empty_like(self.query_offsets)
                self.key_offsets = torch.empty_like(self.query_offsets)
                self.causal = torch.empty(
                    count, dtype=torch.int32, device=self.device
                )
            else:
                self.mask = self.mask_offsets = self.key_offsets = (
                    self.causal
                ) = None
            self.wrapper: (
                flashinfer.BatchDecodeWithPagedKVCacheWrapper
                | flashinfer.BatchPrefillWithPagedKVCacheWrapper
            )
            if decode:
                self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    owner.workspace["scratch"],
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=_tensor_cores(
                        owner.config, owner.num_heads, owner.num_kv_heads
                    ),
                    backend=owner.config.decode_backend,
                    paged_kv_indptr_buffer=self.indptr,
                    paged_kv_indices_buffer=self.indices,
                    paged_kv_last_page_len_buffer=self.last,
                )
            else:
                self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    owner.workspace["scratch"],
                    "NHD",
                    use_cuda_graph=True,
                    backend=owner.config.prefill_backend,
                    qo_indptr_buf=self.query_offsets,
                    paged_kv_indptr_buf=self.indptr,
                    paged_kv_indices_buf=self.indices,
                    paged_kv_last_page_len_buf=self.last,
                    custom_mask_buf=self.mask,
                    mask_indptr_buf=self.mask_offsets,
                )

    def bind(self, owner, queries, keys, table, *, causal):
        # Host page counts and last-page lengths come from declared lengths;
        # physical page IDs are refreshed on device by fill().
        counts = tuple(
            (length + table.block_size - 1) // table.block_size
            for length in keys
        )
        indptr = _host(tuple(accumulate(counts, initial=0)))
        last = _host(
            tuple(
                (length - 1) % table.block_size + 1 if length else 0
                for length in keys
            )
        )

        self.counts.copy_(_host(counts), non_blocking=True)
        self.indptr.copy_(indptr, non_blocking=True)
        self.last.copy_(last, non_blocking=True)
        self.fill(table)

        options = {
            "q_data_type": owner.dtype,
            "kv_data_type": owner.dtype,
            "non_blocking": True,
            "fixed_split_size": (
                owner.config.decode_split_tile_size
                if self.decode
                else owner.config.prefill_split_tile_size
            ),
            "disable_split_kv": owner.config.disable_split_kv,
        }
        with _plan_workspace(self.wrapper):
            if self.decode:
                self.wrapper.plan(
                    indptr,
                    self.indices[: sum(counts)],
                    last,
                    owner.num_heads,
                    owner.num_kv_heads,
                    owner.head_dim,
                    table.block_size,
                    **options,
                )
            else:
                self.wrapper.plan(
                    _host(tuple(accumulate(queries, initial=0))),
                    indptr,
                    self.indices[: sum(counts)],
                    last,
                    owner.num_heads,
                    owner.num_kv_heads,
                    owner.head_dim,
                    table.block_size,
                    causal=all(causal) if self.mask is None else False,
                    packed_custom_mask=self.mask,
                    seq_lens=_host(keys),
                    **options,
                )

        if self.mask is not None:
            assert (
                self.mask_offsets is not None
                and self.key_offsets is not None
                and self.causal is not None
            )
            byte_offsets = tuple(
                accumulate(
                    (
                        (q * k + 7) // 8
                        for q, k in zip(queries, keys, strict=True)
                    ),
                    initial=0,
                )
            )
            self.mask_offsets.copy_(_host(byte_offsets), non_blocking=True)
            self.key_offsets.copy_(
                _host(tuple(accumulate(keys, initial=0))), non_blocking=True
            )
            self.causal.copy_(_host(causal), non_blocking=True)

    def fill(self, table):
        """Refresh device page IDs from the current block table."""
        if self.counts.numel():
            _page_indices[(self.counts.numel(),)](
                table.indices,
                self.counts,
                self.indptr,
                self.indices,
                table.indices.stride(0),
                table.indices.stride(1),
            )

    def run(
        self,
        query,
        key,
        value,
        table,
        *,
        scale,
        out=None,
        lse=False,
        visible=None,
    ):
        self.fill(table)

        if self.mask is not None:
            _ragged_mask[(triton.cdiv(self.mask.numel(), 128),)](
                self.query_offsets,
                self.key_offsets,
                self.mask_offsets,
                self.causal,
                self.mask if visible is None else visible,
                self.mask,
                self.counts.numel(),
                0 if visible is None else visible.stride(0),
                0 if visible is None else visible.stride(1),
                visible is not None,
            )
        # The native run consumes this scalar; all shape-dependent planning is
        # performed by bind, outside capture. The caller owns its score scale.
        self.wrapper._sm_scale = scale
        return self.wrapper.run(
            query.contiguous(), (key, value), out=out, return_lse=lse
        )


@triton.jit
def _ragged_mask(
    query_offsets,
    key_offsets,
    mask_offsets,
    causal,
    visible,
    packed,
    count: tl.constexpr,
    visible_rows: tl.constexpr,
    visible_columns: tl.constexpr,
    use_visible: tl.constexpr,
):
    # Each sequence owns a byte-aligned little-endian mask. Sequence lengths
    # and visibility are loaded at replay, including a changed row partition.
    byte = tl.program_id(0) * 128 + tl.arange(0, 128)
    total = tl.load(mask_offsets + count)
    if tl.program_id(0) * 128 < total:
        # Binary-search the sequence owning each mask byte; offsets are
        # monotone.
        low = tl.full((128,), 0, tl.int32)
        high = tl.full((128,), count, tl.int32)
        for _ in range((count + 1).bit_length()):
            middle = (low + high) // 2
            end = tl.load(
                mask_offsets + middle + 1, middle < count, other=0x7FFFFFFF
            )
            advance = byte >= end
            low = tl.where(advance, middle + 1, low)
            high = tl.where(advance, high, middle)
        row = tl.minimum(low, count - 1)

        begin = tl.load(mask_offsets + row)
        qlen = tl.load(query_offsets + row + 1) - tl.load(query_offsets + row)
        klen = tl.load(key_offsets + row + 1) - tl.load(key_offsets + row)

        # Each byte packs eight consecutive keys of one query row,
        # little-endian.
        bit = (byte[:, None] - begin[:, None]) * 8 + tl.arange(0, 8)[None, :]
        query = bit // tl.maximum(klen[:, None], 1)
        key = bit % tl.maximum(klen[:, None], 1)
        valid = (
            (byte[:, None] < total)
            & (query < qlen[:, None])
            & (klen[:, None] > 0)
        )

        if use_visible:
            limit = tl.load(
                visible + row[:, None] * visible_rows + query * visible_columns,
                valid,
                other=0,
            )
            allowed = valid & (key < limit)
        else:
            is_causal = tl.load(causal + row)
            allowed = valid & (
                (is_causal[:, None] == 0)
                | (key <= query + klen[:, None] - qlen[:, None])
            )

        value = tl.sum(allowed.to(tl.int32) << tl.arange(0, 8)[None, :], axis=1)
        tl.store(packed + byte, value.to(tl.uint8), byte < total)


class _RaggedPlan:
    """Native batched prefill with stable indptr and replayed mask addresses."""

    def __init__(self, owner, batch, *, custom):
        import flashinfer

        self.count = len(batch.queries.host)
        device = owner.workspace["scratch"].device
        with torch.inference_mode(False):
            self.query_offsets = torch.empty(
                self.count + 1, dtype=torch.int32, device=device
            )
            self.key_offsets = torch.empty_like(self.query_offsets)
            self.mask_offsets = (
                torch.empty_like(self.query_offsets) if custom else None
            )
            # Preserve capacity when the same total Q/K rows are repartitioned.
            capacity = (
                batch.queries.num_tokens * batch.keys.num_tokens + 7
            ) // 8 + self.count
            self.mask = (
                torch.empty(capacity, dtype=torch.uint8, device=device)
                if custom
                else None
            )
            self.causal = (
                torch.empty(self.count, dtype=torch.int32, device=device)
                if custom
                else None
            )
        self.wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            owner.workspace["scratch"],
            use_cuda_graph=True,
            qo_indptr_buf=self.query_offsets,
            kv_indptr_buf=self.key_offsets,
            custom_mask_buf=self.mask,
            mask_indptr_buf=self.mask_offsets,
            backend=owner.config.prefill_backend,
        )

    def bind(self, owner, batch):
        queries, keys = batch.queries.host, batch.keys.host
        causal = (
            batch.causal
            if isinstance(batch, VarlenInput)
            else (False,) * self.count
        )
        with _plan_workspace(self.wrapper):
            self.wrapper.plan(
                _host(tuple(accumulate(queries, initial=0))),
                _host(tuple(accumulate(keys, initial=0))),
                owner.num_heads,
                owner.num_kv_heads,
                owner.head_dim,
                packed_custom_mask=self.mask,
                causal=all(causal) if self.mask is None else False,
                q_data_type=owner.dtype,
                kv_data_type=owner.dtype,
                non_blocking=True,
                fixed_split_size=owner.config.prefill_split_tile_size,
                disable_split_kv=owner.config.disable_split_kv,
            )
        if self.mask is not None:
            # The native packed-mask plan derives bit offsets from Q/K indptr.
            # Its kernel consumes byte offsets, with each sequence padded to a
            # whole byte; install those explicit offsets after native planning.
            assert self.mask_offsets is not None and self.causal is not None
            offsets = tuple(
                accumulate(
                    (
                        (q * k + 7) // 8
                        for q, k in zip(queries, keys, strict=True)
                    ),
                    initial=0,
                )
            )
            self.mask_offsets.copy_(_host(offsets), non_blocking=True)
            self.causal.copy_(_host(causal), non_blocking=True)

    def run(self, q, k, v, batch, *, scale, out=None, lse=False):
        if self.mask is not None:
            visible = (
                batch.visible_end if isinstance(batch, VisibleInput) else None
            )
            _ragged_mask[(triton.cdiv(self.mask.numel(), 128),)](
                self.query_offsets,
                self.key_offsets,
                self.mask_offsets,
                self.causal,
                self.mask if visible is None else visible,
                self.mask,
                self.count,
                0 if visible is None else visible.stride(0),
                0 if visible is None else visible.stride(1),
                visible is not None,
            )
        self.wrapper._sm_scale = scale
        return self.wrapper.run(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out=out,
            return_lse=lse,
        )


class _FlashInfer(_Operator):
    def __init__(self, *, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        import flashinfer

        if self.dtype not in {
            torch.float16,
            torch.bfloat16,
        } or self.head_dim not in {
            64,
            128,
            256,
            512,
        }:
            raise ValueError(
                "FlashInfer requires FP16/BF16 queries with head "
                "dimensions 64, 128, 256 or 512"
            )
        if self.cache is not None and isinstance(
            self.cache.key, QuantizedTensor
        ):
            raise ValueError(
                "FlashInfer does not consume per-block FP8 prefix scales"
            )
        self._single = flashinfer.single_prefill_with_kv_cache
        self._merge = flashinfer.merge_state
        self._plans = {}
        self._paged = self._ragged = self._current = None

    def _page_plan(self, queries, keys, table, *, causal, custom=False):
        # Single-token queries without a custom mask take the decode wrapper.
        decode = all(count == 1 for count in queries) and not custom

        # The cache key pins layout, table identity, and mode so captured
        # plans are only reused for replay-compatible invocations.
        signature = (
            "paged",
            len(queries),
            sum(queries),
            tuple(table.indices.shape),
            table.indices.data_ptr(),
            table.block_size,
            decode,
            custom,
            False if custom else all(causal),
        )
        plan = self._plans.get(signature)
        if plan is None:
            plan = _PagePlan(
                self, queries=queries, table=table, decode=decode, custom=custom
            )
            self._plans[signature] = plan
        plan.bind(self, queries, keys, table, causal=causal)
        return plan

    def _ragged_plan(self, batch):
        custom = (
            len(set(batch.causal)) > 1
            if isinstance(batch, VarlenInput)
            else not batch.fully_visible
        )
        signature = (
            "ragged",
            len(batch.queries.host),
            batch.queries.num_tokens,
            batch.keys.num_tokens,
            custom,
            batch.causal[0]
            if isinstance(batch, VarlenInput) and not custom
            else False,
        )
        plan = self._plans.get(signature)
        if plan is None:
            plan = _RaggedPlan(self, batch, custom=custom)
            self._plans[signature] = plan
        plan.bind(self, batch)
        return plan

    def bind(self, batch):
        super().bind(batch)
        self._paged = self._ragged = self._current = None

        if not isinstance(batch, DenseInput) and not batch.queries.num_tokens:
            return

        if isinstance(batch, VarlenInput) or (
            isinstance(batch, VisibleInput) and batch.block_table is None
        ):
            self._ragged = self._ragged_plan(batch)
        elif isinstance(batch, PagedInput):
            # The base binding has required exact host lengths above.
            prefixes, queries = batch.prefixes.host, batch.queries.host
            assert prefixes is not None and queries is not None
            lengths = tuple(
                a + b for a, b in zip(prefixes, queries, strict=True)
            )
            self._paged = self._page_plan(
                queries,
                lengths,
                batch.block_table,
                causal=batch.causal,
                custom=len(set(batch.causal)) > 1,
            )
        elif isinstance(batch, VisibleInput):
            self._paged = self._page_plan(
                batch.queries.host,
                batch.keys.host,
                batch.block_table,
                causal=(False,) * batch.queries.batch_size,
                custom=not batch.fully_visible,
            )
        elif isinstance(batch, SegmentedInput):
            if self.cache is None:
                raise RuntimeError(
                    "segmented attention requires bound prefix state"
                )
            self._paged = self._page_plan(
                batch.queries.host,
                batch.prefixes.host,
                batch.block_table,
                causal=(False,) * batch.queries.batch_size,
            )
            self._current = VisibleInput(
                batch.queries,
                batch.queries,
                batch.visible_current_end,
                None,
                True,
                batch.fully_visible_current,
            )
            self._ragged = self._ragged_plan(self._current)

    def _dense(self, q, k, v, *, scale, causal=False, mask=None, lse=False):
        if causal and mask is not None:
            # Fold causality into the custom mask; single_prefill accepts only
            # one of the two.
            rows = (
                torch.arange(q.shape[0], device=q.device)
                + k.shape[0]
                - q.shape[0]
            )
            columns = torch.arange(k.shape[0], device=q.device)
            mask = mask & (columns[None] <= rows[:, None])
            causal = False

        if not k.shape[0]:
            result = torch.zeros_like(q)
            return (
                (result, torch.full(q.shape[:2], -torch.inf, device=q.device))
                if lse
                else result
            )

        return self._single(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            causal=causal,
            custom_mask=mask,
            sm_scale=scale,
            return_lse=lse,
            backend=self.config.prefill_backend,
        )

    def __call__(self, q, k, v, batch, *, scale, out):
        self._validate(q, k, v, batch, out)

        if (
            isinstance(batch, (PagedInput, SegmentedInput))
            and batch.write_indices is not None
        ):
            self.update_cache(k, v, indices=batch.write_indices)

        if not q.numel():
            return out

        if isinstance(batch, DenseInput):
            if batch.mask is not None and batch.mask.dtype != torch.bool:
                raise ValueError("FlashInfer custom masks must be boolean")
            if q.ndim == 3:
                out.copy_(
                    self._dense(
                        q,
                        k,
                        v,
                        scale=scale,
                        causal=batch.causal,
                        mask=batch.mask,
                    )
                )
            else:
                for row in range(q.shape[0]):
                    mask = batch.mask
                    if mask is not None and mask.ndim > 2:
                        mask = mask[row].squeeze(0)
                    out[row].copy_(
                        self._dense(
                            q[row].transpose(0, 1),
                            k[row].transpose(0, 1),
                            v[row].transpose(0, 1),
                            scale=scale,
                            causal=batch.causal,
                            mask=mask,
                        ).transpose(0, 1)
                    )
            return out
        if isinstance(batch, SegmentedInput):
            if self._ragged is None or self._paged is None:
                raise RuntimeError(
                    "segmented attention requires a bound input plan"
                )
            # Merge the current-window and prefix partial states with
            # online softmax.
            current = self._ragged.run(
                q, k, v, self._current, scale=scale, lse=True
            )
            prefix = self._paged.run(
                q,
                self.cache.key,
                self.cache.value,
                batch.block_table,
                scale=scale,
                lse=True,
            )
            result, _ = self._merge(*current, *prefix)
            out.copy_(result)
            return out

        if self._ragged is not None:
            return self._ragged.run(q, k, v, batch, scale=scale, out=out)

        key, value = (
            (k, v) if self.cache is None else (self.cache.key, self.cache.value)
        )
        visible = (
            batch.visible_end
            if isinstance(batch, VisibleInput) and not batch.fully_visible
            else None
        )
        if self._paged is None:
            raise RuntimeError("paged attention requires a bound input plan")
        return self._paged.run(
            q,
            key,
            value,
            batch.block_table,
            scale=scale,
            out=out,
            visible=visible,
        )

    def close(self):
        self._paged = self._ragged = self._current = None
        self._plans.clear()
        super().close()


class Backend(_Backend):
    name = "flashinfer"

    operator_class = _FlashInfer

    def __init__(self, config: Config = Config()):
        self.config = config

    def workspace_buffers(self, **kwargs):
        return {
            "scratch": BufferConfig((self.config.workspace_size,), torch.uint8)
        }

    def prepare(self, **kwargs):
        return self.operator_class(config=self.config, **kwargs)
