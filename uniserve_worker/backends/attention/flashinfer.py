"""FlashInfer attention backend."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

__all__ = [
    "WrapperKey",
    "FlashInferAttentionBackend",
]

from ...forward import ForwardContext
from ...foundation.runtime_config import FlashInferTuningConfig
from .base import AttentionCapabilities
from .flashinfer_kernels import (
    _decode_effective_seqlens,
    _fill_paged_decode_plan_tensors,
    _fill_paged_prefill_plan_tensors,
    _paged_decode_indices,
    _write_decode_token,
)
from .flashinfer_plan import (
    _cpu_last_page_len,
    _cpu_paged_indptr,
    _decode_plan_key,
    _decode_plan_key_from_shape,
    _DecodePlanTensors,
    _indptr_last,
    _PlanCache,
    _PlanStats,
    _prefill_plan_key,
    _PrefillPlanTensors,
    _record_decode_plan_stats,
    _record_prefill_plan_stats,
    _weakref_or_none,
)
from .flashinfer_pool import WrapperKey, _WrapperPool
from .layout import QKVLayout, normalize_kv, normalize_to

_flashinfer: Any | None
try:  # pragma: no cover - depends on optional CUDA package availability.
    import flashinfer as _flashinfer_module
except Exception:  # pragma: no cover
    _flashinfer = None
else:  # pragma: no cover
    _flashinfer = _flashinfer_module

_BatchDecodeWithPagedKVCacheWrapper = (
    getattr(_flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
    if _flashinfer is not None
    else None
)
_BatchPrefillWithPagedKVCacheWrapper = (
    getattr(_flashinfer, "BatchPrefillWithPagedKVCacheWrapper", None)
    if _flashinfer is not None
    else None
)
_fast_decode_plan = (
    getattr(_flashinfer, "fast_decode_plan", None) if _flashinfer is not None else None
)


class _PagedDecodeInputs(NamedTuple):
    q_bhd: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    restore: Any


class _VarlenPrefillInputs(NamedTuple):
    q: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    batch_size: int


class _DecodeGraphPlanInputs(NamedTuple):
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    effective_seqlens: torch.Tensor
    cpu_indptr: torch.Tensor
    cpu_last_page_len: torch.Tensor
    wrapper_key: WrapperKey
    wrapper: Any


class FlashInferAttentionBackend(_WrapperPool):
    """Paged decode and varlen prefill via FlashInfer wrappers."""

    name = "flashinfer"

    def __init__(self, *, tuning: FlashInferTuningConfig) -> None:
        super().__init__(tuning=tuning)
        self._decode_plan_cache = _PlanCache(
            lambda stats, *, planned, graph, rows, indices: _record_decode_plan_stats(
                stats, planned=planned, graph=graph, rows=rows, indices=indices
            )
        )
        self._prefill_plan_cache = _PlanCache(
            lambda stats, *, planned, graph, rows, indices: _record_prefill_plan_stats(
                stats, planned=planned, rows=rows, indices=indices
            )
        )

    def capabilities(self) -> AttentionCapabilities:
        has_paged_decode = _BatchDecodeWithPagedKVCacheWrapper is not None
        has_paged_prefill = _BatchPrefillWithPagedKVCacheWrapper is not None
        return AttentionCapabilities(
            available=_flashinfer is not None,
            segment_batched_cfg=True,
            mixed_mode=False,
            paged_kv=has_paged_decode,
            varlen_attention=has_paged_prefill,
            varlen_paged_kv=has_paged_prefill,
            requires_paged_varlen=True,
            paged_block_size_multiple=1,
            min_head_dim=64,
            paged_decode_only=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float,
        attn_mask: torch.Tensor | None = None,
        context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del context
        if _flashinfer is None:
            raise RuntimeError("flashinfer backend is not available")
        if attn_mask is not None:
            raise RuntimeError("flashinfer backend does not accept explicit dense masks")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("flashinfer backend expects q/k/v in [L, H, D] layout")
        return _flashinfer.single_prefill_with_kv_cache(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            causal=causal,
            sm_scale=scale,
        )

    def forward_paged(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        causal: bool,
        scale: float,
        context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del causal
        if _BatchDecodeWithPagedKVCacheWrapper is None:
            raise RuntimeError("flashinfer paged decode wrapper is not available")
        inputs = self._prepare_paged_decode_inputs(q, k_cache, v_cache, block_table, cache_seqlens)
        q_bhd = inputs.q_bhd
        plan = None if context is None else context.attention
        binding = getattr(plan, "binding", None)
        current_tokens = self._maybe_write_decode_token(
            k_cache,
            v_cache,
            inputs.block_table,
            inputs.cache_seqlens,
            q_bhd,
            k,
            v,
            plan,
        )
        effective_seqlens = _decode_effective_seqlens(
            inputs.cache_seqlens,
            current_tokens,
            plan,
        )
        if int(q_bhd.shape[0]) != int(effective_seqlens.shape[0]):
            raise ValueError("cache lengths must have one entry per decode row")

        wrapper_key, wrapper = self._decode_wrapper_for(q_bhd, k_cache, binding)
        plan_key = _decode_plan_key(
            binding,
            inputs.block_table,
            inputs.cache_seqlens,
            q_bhd,
            k_cache,
            scale,
            current_tokens,
            wrapper_key,
        )

        def build() -> int:
            return self._build_decode_plan(
                wrapper_key,
                wrapper,
                q_bhd,
                k_cache,
                inputs.block_table,
                effective_seqlens,
                plan,
                scale,
            )

        self._decode_plan_cache.plan_or_reuse(
            wrapper_key=wrapper_key,
            plan_key=plan_key,
            binding=binding,
            stats=None,
            rows=int(q_bhd.shape[0]),
            build=build,
        )

        out = wrapper.run(q_bhd.contiguous(), (k_cache, v_cache))
        return inputs.restore.apply(out)

    def _prepare_paged_decode_inputs(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> _PagedDecodeInputs:
        q_bhd, restore = normalize_to(q, QKVLayout.BHD)
        if q_bhd.shape[0] <= 0:
            raise ValueError("flashinfer paged decode requires a non-empty batch")
        if k_cache.shape != v_cache.shape or k_cache.ndim != 4:
            raise ValueError("flashinfer paged cache expects k/v [pages, page, heads, dim]")
        if q_bhd.shape[-1] != k_cache.shape[-1]:
            raise ValueError("query head dim does not match paged KV cache")
        return _PagedDecodeInputs(
            q_bhd=q_bhd,
            block_table=block_table.to(device=q_bhd.device, dtype=torch.int32).contiguous(),
            cache_seqlens=cache_seqlens.to(device=q_bhd.device, dtype=torch.int32).contiguous(),
            restore=restore,
        )

    def _maybe_write_decode_token(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        q_bhd: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        plan: Any,
    ) -> int:
        if k is None and v is None:
            return 0
        if k is None or v is None:
            raise ValueError("flashinfer paged update requires both k and v")
        k_bhd = normalize_kv(k, QKVLayout.BHD, contiguous=False)
        v_bhd = normalize_kv(v, QKVLayout.BHD, contiguous=False)
        if k_bhd.shape != v_bhd.shape:
            raise ValueError("current paged K/V tensors must have matching shapes")
        if k_bhd.shape[0] != q_bhd.shape[0]:
            raise ValueError("current K/V batch size must match q batch size")
        if k_bhd.shape[1:] != k_cache.shape[2:]:
            raise ValueError("current K/V head geometry does not match paged cache")
        _write_decode_token(k_cache, v_cache, block_table, cache_seqlens, k_bhd, v_bhd, plan)
        return 1

    def _decode_wrapper_for(
        self,
        q_bhd: torch.Tensor,
        k_cache: torch.Tensor,
        binding: Any,
    ) -> tuple[WrapperKey, Any]:
        graph_wrapper = self._decode_graph_wrapper_for_binding(binding)
        if graph_wrapper is not None:
            return graph_wrapper
        return self._decode_wrapper(
            q_bhd.device,
            int(q_bhd.shape[1]),
            int(k_cache.shape[2]),
            k_cache.dtype,
        )

    def _build_decode_plan(
        self,
        wrapper_key: WrapperKey,
        wrapper: Any,
        q_bhd: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        effective_seqlens: torch.Tensor,
        plan: Any,
        scale: float,
    ) -> int:
        cpu_indptr = _cpu_paged_indptr(
            plan,
            int(q_bhd.shape[0]),
            int(k_cache.shape[1]),
        )
        cpu_last_page_len = _cpu_last_page_len(
            plan,
            int(q_bhd.shape[0]),
            int(k_cache.shape[1]),
        )
        plan_tensors = self._decode_plan_tensors(
            wrapper_key,
            block_table,
            effective_seqlens,
            int(k_cache.shape[1]),
            index_count=_indptr_last(cpu_indptr),
        )
        self._plan_decode(
            wrapper_key,
            wrapper,
            plan_tensors.indptr,
            plan_tensors.indices,
            plan_tensors.last_page_len,
            int(q_bhd.shape[1]),
            int(k_cache.shape[2]),
            int(q_bhd.shape[2]),
            int(k_cache.shape[1]),
            pos_encoding_mode="NONE",
            q_data_type=q_bhd.dtype,
            kv_data_type=k_cache.dtype,
            data_type=k_cache.dtype,
            sm_scale=scale,
            block_tables=block_table,
            seq_lens=effective_seqlens,
            global_override_indptr_cpu=cpu_indptr,
            global_override_last_page_len_cpu=cpu_last_page_len,
        )
        return plan_tensors.index_count

    def forward_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        scale: float,
        block_table: torch.Tensor | None = None,
        context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del max_seqlen_q, max_seqlen_k
        if _BatchPrefillWithPagedKVCacheWrapper is None:
            raise RuntimeError("flashinfer paged prefill wrapper is not available")
        inputs = self._prepare_varlen_prefill_inputs(
            q, k, v, cu_seqlens_q, cu_seqlens_k, block_table
        )

        plan = None if context is None else context.attention
        binding = getattr(plan, "binding", None)
        # Binding-identity routing: a forward whose context carries a graph
        # binding runs on that graph's exclusive wrapper (so the
        # capture warmup plans it and the capture bakes only its ``run``); all
        # other forwards keep the shared prefill wrapper.
        graph_wrapper = self._prefill_graph_wrapper_for_binding(binding)
        if graph_wrapper is not None:
            wrapper_key, wrapper = graph_wrapper
        else:
            wrapper_key, wrapper = self._prefill_wrapper(inputs.q.device)
        plan_key = _prefill_plan_key(
            binding,
            inputs.block_table,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            inputs.q,
            k,
            causal,
            scale,
            wrapper_key,
        )

        def build() -> int:
            return self._build_prefill_plan(
                wrapper_key,
                wrapper,
                inputs.q,
                k,
                inputs.block_table,
                inputs.cu_seqlens_q,
                inputs.cu_seqlens_k,
                plan,
                causal,
                scale,
            )

        self._prefill_plan_cache.plan_or_reuse(
            wrapper_key=wrapper_key,
            plan_key=plan_key,
            binding=binding,
            stats=None,
            rows=inputs.batch_size,
            build=build,
        )

        return wrapper.run(inputs.q, (k, v))

    def _prepare_varlen_prefill_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor | None,
    ) -> _VarlenPrefillInputs:
        if block_table is None:
            raise RuntimeError("flashinfer varlen path requires a paged KV block table")
        if q.ndim != 3:
            raise ValueError("flashinfer paged varlen expects q in [total, heads, dim] layout")
        if k.shape != v.shape or k.ndim != 4:
            raise ValueError(
                "flashinfer paged varlen expects k/v caches in [pages, page, heads, dim] layout"
            )
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("query head dim does not match paged KV cache")
        q = q.contiguous()
        block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
        cu_seqlens_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous()
        cu_seqlens_k = cu_seqlens_k.to(device=q.device, dtype=torch.int32).contiguous()
        batch_size = self._validate_varlen_prefill_shape(block_table, cu_seqlens_q, cu_seqlens_k)
        return _VarlenPrefillInputs(q, block_table, cu_seqlens_q, cu_seqlens_k, batch_size)

    def _validate_varlen_prefill_shape(
        self,
        block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> int:
        if int(cu_seqlens_q.numel()) != int(cu_seqlens_k.numel()):
            raise ValueError("q and k cu_seqlens must describe the same batch")
        batch_size = int(cu_seqlens_q.numel()) - 1
        if batch_size <= 0:
            raise ValueError("flashinfer paged varlen requires a non-empty batch")
        if int(block_table.shape[0]) != batch_size:
            raise ValueError("block table rows must match cu_seqlens batch size")
        return batch_size

    def _build_prefill_plan(
        self,
        wrapper_key: WrapperKey,
        wrapper: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        plan: Any,
        causal: bool,
        scale: float,
    ) -> int:
        kv_seqlens = getattr(plan, "kv_seqlens", None)
        if not isinstance(kv_seqlens, torch.Tensor) or tuple(kv_seqlens.shape) != (
            int(cu_seqlens_k.numel()) - 1,
        ):
            kv_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        kv_seqlens = kv_seqlens.to(device=cu_seqlens_k.device, dtype=torch.int32).contiguous()
        query_lens = getattr(plan, "query_lens", None)
        if not isinstance(query_lens, torch.Tensor) or tuple(query_lens.shape) != (
            int(cu_seqlens_q.numel()) - 1,
        ):
            query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        query_lens = query_lens.to(device=cu_seqlens_q.device, dtype=torch.int32).contiguous()
        plan_tensors = self._prefill_plan_tensors(
            wrapper_key,
            block_table,
            cu_seqlens_q,
            kv_seqlens,
            int(k.shape[1]),
        )
        self._plan_prefill_wrapper(
            wrapper_key,
            wrapper,
            plan_tensors,
            block_table=block_table,
            kv_seqlens=kv_seqlens,
            query_lens=query_lens,
            num_q_heads=int(q.shape[1]),
            num_kv_heads=int(k.shape[2]),
            head_dim=int(q.shape[2]),
            page_size=int(k.shape[1]),
            q_dtype=q.dtype,
            kv_dtype=k.dtype,
            causal=causal,
            scale=scale,
        )
        return plan_tensors.index_count

    def _plan_prefill_wrapper(
        self,
        wrapper_key: WrapperKey,
        wrapper: Any,
        plan: _PrefillPlanTensors,
        *,
        block_table: torch.Tensor,
        kv_seqlens: torch.Tensor,
        query_lens: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        causal: bool,
        scale: float | None,
    ) -> None:
        scale_value = None if scale is None else float(scale)
        self.plan_prefill(
            (
                tuple(wrapper_key),
                int(num_q_heads),
                int(num_kv_heads),
                int(head_dim),
                int(page_size),
                bool(causal),
                None if scale_value is None else float(scale_value),
                int(plan.qo_indptr.numel()),
                int(plan.indices.numel()),
                self._tuning.prefill_split_tile_size,
                self._tuning.disable_split_kv,
            ),
            workspace=self._workspace(block_table.device),
            wrapper=wrapper,
        )
        wrapper.plan(
            plan.qo_indptr,
            plan.kv_indptr,
            plan.indices,
            plan.last_page_len,
            int(num_q_heads),
            int(num_kv_heads),
            int(head_dim),
            int(page_size),
            causal=causal,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=q_dtype,
            sm_scale=scale_value,
            non_blocking=True,
            seq_lens=kv_seqlens,
            seq_lens_q=query_lens,
            block_tables=block_table,
            fixed_split_size=self._tuning.prefill_split_tile_size,
            disable_split_kv=self._tuning.disable_split_kv,
        )

    def prepare_paged_prefill_cuda_graph(
        self,
        binding: Any,
        plan: Any,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        causal: bool,
        scale: float | None = None,
        stats: _PlanStats | None = None,
    ) -> None:
        """Refresh a graph-scoped paged-prefill wrapper from live side-table tensors."""

        if _BatchPrefillWithPagedKVCacheWrapper is None:
            raise RuntimeError("flashinfer paged prefill wrapper is not available")
        bound = self._prefill_graph_wrapper_for_binding(binding)
        if bound is None:
            raise RuntimeError("no graph-scoped paged prefill wrapper is bound to binding")
        wrapper_key, wrapper = bound
        block_table = getattr(plan, "block_table", None)
        cu_seqlens_q = getattr(plan, "cu_seqlens_q", None)
        cu_seqlens_k = getattr(plan, "cu_seqlens_k", None)
        if not (
            isinstance(block_table, torch.Tensor)
            and isinstance(cu_seqlens_q, torch.Tensor)
            and isinstance(cu_seqlens_k, torch.Tensor)
        ):
            raise RuntimeError("paged prefill graph plan is missing tensors")
        if int(cu_seqlens_q.numel()) != int(cu_seqlens_k.numel()):
            raise RuntimeError("paged prefill graph q/k sequence tables must have the same length")
        batch_size = int(cu_seqlens_q.numel()) - 1
        if batch_size <= 0 or int(block_table.shape[0]) != batch_size:
            raise RuntimeError("paged prefill graph block table row count mismatch")
        kv_seqlens = getattr(plan, "kv_seqlens", None)
        if not isinstance(kv_seqlens, torch.Tensor) or tuple(kv_seqlens.shape) != (batch_size,):
            kv_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        kv_seqlens = kv_seqlens.to(device=cu_seqlens_k.device, dtype=torch.int32).contiguous()
        query_lens = getattr(plan, "query_lens", None)
        if not isinstance(query_lens, torch.Tensor) or tuple(query_lens.shape) != (batch_size,):
            query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        query_lens = query_lens.to(device=cu_seqlens_q.device, dtype=torch.int32).contiguous()
        plan_tensors = self._prefill_plan_tensors(
            wrapper_key,
            block_table,
            cu_seqlens_q,
            kv_seqlens,
            int(page_size),
        )
        self._plan_prefill_wrapper(
            wrapper_key,
            wrapper,
            plan_tensors,
            block_table=block_table,
            kv_seqlens=kv_seqlens,
            query_lens=query_lens,
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            page_size=int(page_size),
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            causal=bool(causal),
            scale=scale,
        )
        _record_prefill_plan_stats(
            stats,
            planned=True,
            rows=batch_size,
            indices=plan_tensors.index_count,
        )

    def prepare_paged_decode_cuda_graph(
        self,
        binding: Any,
        plan: Any,
        *,
        batch_size: int,
        max_indices: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        scale: float | None = None,
        stats: _PlanStats | None = None,
    ) -> None:
        if _BatchDecodeWithPagedKVCacheWrapper is None:
            raise RuntimeError("flashinfer paged decode wrapper is not available")
        if binding is None:
            raise ValueError("decode graph preparation requires a binding identity")
        inputs = self._decode_graph_plan_inputs(
            plan,
            batch_size=int(batch_size),
            max_indices=int(max_indices),
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            page_size=int(page_size),
            kv_dtype=kv_dtype,
        )
        plan_tensors = self._prepare_decode_graph_plan_tensors(
            inputs.wrapper_key,
            inputs.block_table,
            inputs.effective_seqlens,
            page_size=int(page_size),
            batch_size=int(batch_size),
            cpu_indptr=inputs.cpu_indptr,
            stats=stats,
        )
        plan_key = self._decode_graph_plan_key(
            binding,
            inputs.block_table,
            inputs.cache_seqlens,
            batch_size=int(batch_size),
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            page_size=int(page_size),
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            scale=scale,
            wrapper_key=inputs.wrapper_key,
        )
        self._plan_decode_graph(
            inputs.wrapper_key,
            inputs.wrapper,
            plan_tensors.indptr,
            plan_tensors.indices,
            plan_tensors.last_page_len,
            int(num_q_heads),
            int(num_kv_heads),
            int(head_dim),
            int(page_size),
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            scale=scale,
            block_table=inputs.block_table,
            effective_seqlens=inputs.effective_seqlens,
            cpu_indptr=inputs.cpu_indptr,
            cpu_last_page_len=inputs.cpu_last_page_len,
        )
        self._decode_plan_cache.remember(inputs.wrapper_key, plan_key, binding)
        self._binding_graph_wrappers[id(binding)] = (inputs.wrapper_key, _weakref_or_none(binding))
        self.bind_graph((id(binding), "decode"), inputs.wrapper_key)

    def bind_paged_prefill_graph_wrapper(
        self,
        binding: Any,
        plan: Any,
        *,
        device: torch.device | str,
    ) -> None:
        """Bind ``binding`` to a graph-scoped *exclusive* prefill wrapper.

        A captured prefill ``wrapper.run`` bakes the wrapper's plan (its host
        ``_plan_info`` scalars plus the device int-workspace contents written by
        ``plan``) into the CUDA graph. The shared prefill wrapper is re-planned
        by every other varlen forward in the process, so a graph that captured
        it would replay against a foreign plan. Binding allocates a fresh
        ``scope`` nonce and routes every ``forward_varlen`` whose forward
        context carries ``binding`` to the exclusive wrapper; the
        capture warmup then builds its plan once through the normal path and
        nothing else can ever invalidate it. Release with
        :meth:`release_paged_prefill_graph_wrapper` when the graph is freed.
        """

        if _BatchPrefillWithPagedKVCacheWrapper is None:
            raise RuntimeError("flashinfer paged prefill wrapper is not available")
        if binding is None:
            raise ValueError("graph prefill wrapper binding requires an identity")
        block_table = getattr(plan, "block_table", None)
        cu_seqlens_q = getattr(plan, "cu_seqlens_q", None)
        if not isinstance(block_table, torch.Tensor) or not isinstance(cu_seqlens_q, torch.Tensor):
            raise ValueError("graph prefill wrapper binding requires paged side-table tensors")
        batch_size = int(cu_seqlens_q.numel()) - 1
        if batch_size <= 0 or int(block_table.shape[0]) != batch_size:
            raise ValueError("graph prefill wrapper binding side-table geometry mismatch")
        from .flashinfer_pool import _PREFILL_GRAPH_SCOPES

        scope = next(_PREFILL_GRAPH_SCOPES)
        key, _wrapper = self._prefill_graph_wrapper(
            torch.device(device),
            scope=scope,
            batch_size=batch_size,
            max_indices=max(1, int(block_table.numel())),
        )
        self._binding_prefill_graph_wrappers[id(binding)] = (key, _weakref_or_none(binding))
        self.bind_graph((id(binding), "prefill"), key)

    def paged_prefill_graph_wrapper_planned(self, binding: Any) -> bool:
        """Whether the wrapper bound to ``binding`` has been planned.

        The denoise-step graph runner asserts this after capture: if the
        dispatcher routed the captured attention to a different backend, the
        exclusive wrapper never planned and the capture must be discarded
        rather than replayed against undefined plan state.
        """

        bound = self._prefill_graph_wrapper_for_binding(binding)
        if bound is None:
            return False
        _key, wrapper = bound
        return getattr(wrapper, "_plan_info", None) is not None

    def release_paged_prefill_graph_wrapper(self, binding: Any) -> None:
        """Drop the exclusive prefill wrapper (and its caches) bound to ``binding``."""

        entry = self._binding_prefill_graph_wrappers.pop(id(binding), None)
        if entry is None:
            return
        wrapper_key, _binding_ref = entry
        self._prefill_wrappers.pop(wrapper_key, None)
        self._prefill_plan_workspaces.pop(wrapper_key, None)
        self._prefill_plan_cache.forget(wrapper_key)

    def release_paged_decode_graph_binding(self, binding: Any) -> None:
        """Release one binding while retaining shape-shared decode buffers."""

        self._binding_graph_wrappers.pop(id(binding), None)

    def _decode_graph_plan_inputs(
        self,
        plan: Any,
        *,
        batch_size: int,
        max_indices: int,
        num_q_heads: int,
        num_kv_heads: int,
        page_size: int,
        kv_dtype: torch.dtype,
    ) -> _DecodeGraphPlanInputs:
        block_table = plan.block_table.to(dtype=torch.int32).contiguous()
        cache_seqlens = plan.cache_seqlens.to(dtype=torch.int32).contiguous()
        wrapper_key, wrapper = self._decode_cuda_graph_wrapper(
            block_table.device,
            batch_size=int(batch_size),
            max_indices=int(max_indices),
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            kv_dtype=kv_dtype,
        )
        cpu_indptr = _cpu_paged_indptr(plan, int(batch_size), int(page_size))
        cpu_last_page_len = _cpu_last_page_len(plan, int(batch_size), int(page_size))
        if cpu_indptr is None or cpu_last_page_len is None:
            raise ValueError("decode graph planning requires complete CPU KV lengths")
        effective_seqlens = getattr(plan, "kv_seqlens", None)
        if not isinstance(effective_seqlens, torch.Tensor) or tuple(effective_seqlens.shape) != (
            int(batch_size),
        ):
            effective_seqlens = cache_seqlens + 1
        effective_seqlens = effective_seqlens.to(
            device=cache_seqlens.device,
            dtype=torch.int32,
        ).contiguous()
        return _DecodeGraphPlanInputs(
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            effective_seqlens=effective_seqlens,
            cpu_indptr=cpu_indptr,
            cpu_last_page_len=cpu_last_page_len,
            wrapper_key=wrapper_key,
            wrapper=wrapper,
        )

    def _prepare_decode_graph_plan_tensors(
        self,
        wrapper_key: WrapperKey,
        block_table: torch.Tensor,
        effective_seqlens: torch.Tensor,
        *,
        page_size: int,
        batch_size: int,
        cpu_indptr: torch.Tensor,
        stats: _PlanStats | None,
    ) -> _DecodePlanTensors:
        plan = self._decode_plan_tensors(
            wrapper_key,
            block_table,
            effective_seqlens,
            int(page_size),
            index_count=_indptr_last(cpu_indptr),
        )
        _record_decode_plan_stats(
            stats,
            planned=True,
            graph=True,
            rows=int(batch_size),
            indices=plan.index_count,
        )
        return plan

    def _decode_graph_plan_key(
        self,
        binding: Any,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        *,
        batch_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        scale: float | None,
        wrapper_key: WrapperKey,
    ) -> tuple[Any, ...]:
        return _decode_plan_key_from_shape(
            binding,
            block_table,
            cache_seqlens,
            batch_size=int(batch_size),
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            page_size=int(page_size),
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            scale=scale,
            current_tokens=1,
            wrapper_key=wrapper_key,
        )

    def _plan_decode_graph(
        self,
        wrapper_key: WrapperKey,
        wrapper: Any,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        scale: float | None,
        block_table: torch.Tensor,
        effective_seqlens: torch.Tensor,
        cpu_indptr: torch.Tensor,
        cpu_last_page_len: torch.Tensor,
    ) -> None:
        self._plan_decode(
            wrapper_key,
            wrapper,
            indptr,
            indices,
            last_page_len,
            int(num_q_heads),
            int(num_kv_heads),
            int(head_dim),
            int(page_size),
            pos_encoding_mode="NONE",
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            data_type=kv_dtype,
            sm_scale=scale,
            block_tables=block_table,
            seq_lens=effective_seqlens,
            global_override_indptr_cpu=cpu_indptr,
            global_override_last_page_len_cpu=cpu_last_page_len,
        )

    def _decode_plan_tensors(
        self,
        wrapper_key: WrapperKey,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        page_size: int,
        *,
        index_count: int | None,
    ) -> _DecodePlanTensors:
        batch_size = int(seq_lens.shape[0])
        max_indices = max(1, int(block_table.numel()))
        graph_buffers = self._decode_graph_buffers.get(wrapper_key)
        workspace = self._decode_plan_workspace(
            wrapper_key,
            block_table.device,
            batch_size=batch_size,
            max_indices=max_indices,
            graph_buffers=graph_buffers,
        )
        if _fill_paged_decode_plan_tensors(
            block_table,
            seq_lens,
            int(page_size),
            workspace,
        ):
            count = int(index_count) if index_count is not None else max_indices
            count = max(1, min(count, int(workspace.indices.numel())))
            return _DecodePlanTensors(
                workspace.indptr[: batch_size + 1],
                workspace.indices[:count],
                workspace.last_page_len[:batch_size],
                count,
            )
        indptr, indices, last_page_len = _paged_decode_indices(
            block_table,
            seq_lens,
            int(page_size),
        )
        return _DecodePlanTensors(indptr, indices, last_page_len, int(indices.numel()))

    def _prefill_plan_tensors(
        self,
        wrapper_key: WrapperKey,
        block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        kv_seqlens: torch.Tensor,
        page_size: int,
    ) -> _PrefillPlanTensors:
        batch_size = int(kv_seqlens.shape[0])
        max_indices = max(1, int(block_table.numel()))
        workspace = self._prefill_plan_workspace(
            wrapper_key,
            block_table.device,
            batch_size=batch_size,
            max_indices=max_indices,
        )
        if _fill_paged_prefill_plan_tensors(
            block_table,
            cu_seqlens_q,
            kv_seqlens,
            int(page_size),
            workspace,
        ):
            count = max(1, min(max_indices, int(workspace.kv_indptr[batch_size].item())))
            return _PrefillPlanTensors(
                workspace.qo_indptr[: batch_size + 1],
                workspace.kv_indptr[: batch_size + 1],
                workspace.indices[:count],
                workspace.last_page_len[:batch_size],
                count,
            )
        kv_indptr, indices, last_page_len = _paged_decode_indices(
            block_table, kv_seqlens, int(page_size)
        )
        return _PrefillPlanTensors(
            cu_seqlens_q[: batch_size + 1],
            kv_indptr,
            indices,
            last_page_len,
            int(indices.numel()),
        )
