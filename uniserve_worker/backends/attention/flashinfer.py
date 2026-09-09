"""FlashInfer attention backend."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

__all__ = [
    "WrapperKey",
    "FlashInferAttentionBackend",
]

from ...execution.forward_batch import AttentionMode, ForwardBatch
from .base import AttentionBackend
from .flashinfer_kernels import (
    _decode_effective_seqlens,
    _fill_paged_decode_plan_tensors,
    _fill_paged_prefill_plan_tensors,
    _paged_decode_indices,
    _write_decode_token,
)
from .flashinfer_plan import (
    _binding_identity,
    _cpu_last_page_len,
    _cpu_paged_indptr,
    _decode_plan_key,
    _DecodePlanTensors,
    _indptr_last,
    _PlanCache,
    _PrefillHostPlan,
    _prefill_host_plan,
    _prefill_plan_key,
    _plan_workspace,
    _PrefillPlanTensors,
    _weakref_or_none,
)
from .flashinfer_pool import WrapperKey, _WrapperPool
from .layout import QKVLayout, normalize_kv, normalize_to
from .tuning import FlashInferTuningConfig

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
_single_prefill_return_lse = (
    getattr(_flashinfer, "single_prefill_with_kv_cache_return_lse", None)
    if _flashinfer is not None
    else None
)
_merge_state = getattr(_flashinfer, "merge_state", None) if _flashinfer is not None else None


class _PagedDecodeInputs(NamedTuple):
    """Groups normalized queries, page tables, sequence lengths, and layout restoration for paged decode."""

    q_bhd: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    restore: Any


class _VarlenPrefillInputs(NamedTuple):
    """Groups packed queries and cumulative sequence offsets for paged variable-length prefill."""

    q: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    batch_size: int


class _DecodeGraphPlanInputs(NamedTuple):
    """Holds live decode metadata and wrapper identity used to refresh a captured graph plan."""

    block_table: torch.Tensor
    effective_seqlens: torch.Tensor
    cpu_indptr: torch.Tensor
    cpu_last_page_len: torch.Tensor
    wrapper_key: WrapperKey
    wrapper: Any


class FlashInferAttentionBackend(_WrapperPool, AttentionBackend):
    """Paged decode and varlen prefill via FlashInfer wrappers."""

    name = "flashinfer"
    available = _flashinfer is not None
    paged_varlen = _BatchPrefillWithPagedKVCacheWrapper is not None
    paged_varlen_only = True
    min_head_dim = 64
    single_ar_decode = True
    cuda_only = True
    dense_ranks = frozenset({3})

    def supports(self, mode: AttentionMode, *, cuda_graph: bool = False) -> bool:
        """Accept FlashInfer dense, paged, packed, and variable-length paths supported by the active wrapper."""

        if mode is AttentionMode.PAGED_DECODE and _BatchDecodeWithPagedKVCacheWrapper is None:
            return False
        if mode is AttentionMode.PAGED_VARLEN and _BatchPrefillWithPagedKVCacheWrapper is None:
            return False
        if mode is AttentionMode.PACKED and (
            _BatchPrefillWithPagedKVCacheWrapper is None
            or _single_prefill_return_lse is None
            or _merge_state is None
        ):
            return False
        return super().supports(mode, cuda_graph=cuda_graph)

    def __init__(self, *, tuning: FlashInferTuningConfig) -> None:
        """Initialize decode and prefill plan caches under one tuning policy."""

        super().__init__(tuning=tuning)
        self._decode_plan_cache = _PlanCache()
        self._prefill_plan_cache = _PlanCache()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float,
        attn_mask: torch.Tensor | None = None,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor:
        """Compute dense attention through FlashInfer’s single-request prefill operator."""

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
        context: ForwardBatch | None = None,
    ) -> torch.Tensor:
        """Plan and execute FlashInfer paged decode, including an optional current-token cache write."""

        del causal
        if _BatchDecodeWithPagedKVCacheWrapper is None:
            raise RuntimeError("flashinfer paged decode wrapper is not available")
        inputs = self._prepare_paged_decode_inputs(q, k_cache, v_cache, block_table, cache_seqlens)
        q_bhd = inputs.q_bhd
        plan = context
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

        graph_wrapper = self._decode_graph_wrapper_for_binding(binding)
        if graph_wrapper is not None:
            # Replay preparation owns the mutable page plan. Capturing another
            # planner upload here would overwrite it with capture-time metadata.
            _wrapper_key, wrapper = graph_wrapper
            wrapper._sm_scale = float(scale)
            out = wrapper.run(q_bhd.contiguous(), (k_cache, v_cache))
            return inputs.restore.apply(out)

        wrapper_key, wrapper = self._decode_wrapper(
            q_bhd.device,
            int(q_bhd.shape[1]),
            int(k_cache.shape[2]),
            k_cache.dtype,
        )
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
            """Populate decode plan tensors for the selected wrapper and return index capacity."""

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
        """Normalize decode query and cache tensors to FlashInfer's batch-head-dimension layout."""

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
        """Write an optional decode token into the cache and return the effective sequence lengths."""

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
        """Populate decode page metadata and plan the selected FlashInfer wrapper."""

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
        context: ForwardBatch | None = None,
    ) -> torch.Tensor:
        """Plan and execute FlashInfer paged prefill for packed variable-length queries."""

        del max_seqlen_q, max_seqlen_k
        if _BatchPrefillWithPagedKVCacheWrapper is None:
            raise RuntimeError("flashinfer paged prefill wrapper is not available")
        inputs = self._prepare_varlen_prefill_inputs(
            q, k, v, cu_seqlens_q, cu_seqlens_k, block_table
        )

        plan = context
        binding = getattr(plan, "binding", None)
        # Binding-identity routing: a forward whose context carries a graph
        # binding runs on that graph's exclusive wrapper (so the
        # capture warmup plans it and the capture bakes only its ``run``); all
        # other forwards keep the shared prefill wrapper.
        graph_wrapper = self._prefill_graph_wrapper_for_binding(binding)
        if graph_wrapper is not None:
            _wrapper_key, wrapper = graph_wrapper
            return wrapper.run(inputs.q, (k, v))
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
            """Populate the packed-prefill plan and return its page-index capacity."""

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
            build=build,
        )

        return wrapper.run(inputs.q, (k, v))

    def forward_segmented(
        self,
        q: torch.Tensor,
        current_k: torch.Tensor,
        current_v: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        *,
        page_table: torch.Tensor,
        prefix_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        visible_current_end: torch.Tensor,
        scale: float,
        fully_visible_current: bool,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor:
        """Evaluate live and cached KV segments with FlashInfer and merge their attention states."""

        if (
            _BatchPrefillWithPagedKVCacheWrapper is None
            or _single_prefill_return_lse is None
            or _merge_state is None
        ):
            raise RuntimeError("flashinfer segmented prefill is not available")
        plan = context
        query_lens = tuple(int(value) for value in getattr(plan, "query_lens_cpu", ()) or ())
        causal_rows = tuple(bool(value) for value in getattr(plan, "causal_rows_cpu", ()) or ())
        prefix_lens_cpu = tuple(int(value) for value in getattr(plan, "seq_lens_cpu", ()) or ())
        if (
            not query_lens
            or len(query_lens) != len(prefix_lens_cpu)
            or len(query_lens) != len(causal_rows)
            or sum(query_lens) != int(q.shape[0])
        ):
            raise ValueError("flashinfer segmented rows require host-known lengths")
        offsets = [0]
        for length in query_lens:
            offsets.append(offsets[-1] + length)
        current_outputs: list[torch.Tensor] = []
        current_lses: list[torch.Tensor] = []
        for row, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            row_query = q[begin:end].contiguous()
            row_key = current_k[begin:end].contiguous()
            row_value = current_v[begin:end].contiguous()
            causal = causal_rows[row]
            if fully_visible_current and causal:
                raise RuntimeError("flashinfer segmented visibility metadata is inconsistent")
            output, lse = _single_prefill_return_lse(
                row_query,
                row_key,
                row_value,
                causal=causal,
                sm_scale=scale,
            )
            current_outputs.append(output)
            current_lses.append(lse)
        current_output = torch.cat(current_outputs, dim=0)
        current_lse = torch.cat(current_lses, dim=0)

        cu_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous()
        prefix_lengths = prefix_lens.to(device=q.device, dtype=torch.int32).contiguous()
        cu_prefix = torch.cat((prefix_lengths.new_zeros(1), prefix_lengths.cumsum(0)))
        pages = page_table.to(device=q.device, dtype=torch.int32).contiguous()
        wrapper_key, wrapper = self._prefill_wrapper(q.device)
        self._build_prefill_plan(
            wrapper_key,
            wrapper,
            q,
            prefix_k,
            pages,
            cu_q,
            cu_prefix,
            plan,
            False,
            scale,
        )
        prefix_output, prefix_lse = wrapper.forward_return_lse(
            q.contiguous(),
            (prefix_k, prefix_v),
            causal=False,
            sm_scale=scale,
        )
        output, _ = _merge_state(
            current_output,
            current_lse,
            prefix_output,
            prefix_lse,
        )
        return output

    def _prepare_varlen_prefill_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor | None,
    ) -> _VarlenPrefillInputs:
        """Validate packed prefill boundaries and normalize query, key, and value layouts."""

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
        """Validate packed query and KV boundaries against the paged block table."""

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
        """Populate packed prefill page metadata and plan the selected wrapper."""

        kv_seqlens = getattr(plan, "kv_lens", None)
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
        host_plan = _prefill_host_plan(plan, int(kv_seqlens.shape[0]), int(k.shape[1]))
        plan_tensors = self._prefill_plan_tensors(
            wrapper_key,
            block_table,
            cu_seqlens_q,
            kv_seqlens,
            int(k.shape[1]),
            index_count=(
                _indptr_last(_cpu_paged_indptr(plan, int(kv_seqlens.shape[0]), int(k.shape[1])))
                if host_plan is None
                else host_plan.index_count
            ),
        )
        self._plan_prefill_wrapper(
            wrapper_key,
            wrapper,
            plan_tensors,
            host_plan=host_plan,
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
        host_plan: _PrefillHostPlan | None,
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
        """Bind bounded plan tensors and attention geometry to a prefill wrapper."""

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
        # The planner needs CPU lengths even when its kernels consume GPU
        # metadata. Passing the existing host values avoids draining the
        # execution stream for small D2H copies before every Graph replay.
        device_lengths = self._tuning.prefill_backend == "cudnn"
        with _plan_workspace(wrapper):
            wrapper.plan(
                plan.qo_indptr if host_plan is None else host_plan.qo_indptr,
                plan.kv_indptr if host_plan is None else host_plan.kv_indptr,
                plan.indices,
                plan.last_page_len if host_plan is None else host_plan.last_page_len,
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
                seq_lens=kv_seqlens if host_plan is None or device_lengths else host_plan.kv_lens,
                seq_lens_q=query_lens,
                max_token_per_sequence=None if host_plan is None else host_plan.max_query_rows,
                max_sequence_kv=(
                    host_plan.max_kv_rows if host_plan is not None and device_lengths else None
                ),
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
        kv_seqlens = getattr(plan, "kv_lens", None)
        if not isinstance(kv_seqlens, torch.Tensor) or tuple(kv_seqlens.shape) != (batch_size,):
            kv_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        kv_seqlens = kv_seqlens.to(device=cu_seqlens_k.device, dtype=torch.int32).contiguous()
        query_lens = getattr(plan, "query_lens", None)
        if not isinstance(query_lens, torch.Tensor) or tuple(query_lens.shape) != (batch_size,):
            query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        query_lens = query_lens.to(device=cu_seqlens_q.device, dtype=torch.int32).contiguous()
        host_plan = _prefill_host_plan(plan, batch_size, int(page_size))
        plan_tensors = self._prefill_plan_tensors(
            wrapper_key,
            block_table,
            cu_seqlens_q,
            kv_seqlens,
            int(page_size),
            index_count=(
                _indptr_last(_cpu_paged_indptr(plan, batch_size, int(page_size)))
                if host_plan is None
                else host_plan.index_count
            ),
        )
        self._plan_prefill_wrapper(
            wrapper_key,
            wrapper,
            plan_tensors,
            host_plan=host_plan,
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
    ) -> None:
        """Refresh graph-scoped decode plan tensors and plan the exclusive wrapper before replay."""

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
        binding_key = _binding_identity(binding)
        if binding_key is None:
            raise RuntimeError("paged decode graph preparation requires a binding token")
        self._binding_graph_wrappers[binding_key] = (
            inputs.wrapper_key,
            _weakref_or_none(binding),
        )
        self.bind_graph((binding_key, "decode"), inputs.wrapper_key)

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
        binding_key = _binding_identity(binding)
        if binding_key is None:
            raise RuntimeError("paged prefill graph binding requires a binding token")
        self._binding_prefill_graph_wrappers[binding_key] = (key, _weakref_or_none(binding))
        self.bind_graph((binding_key, "prefill"), key)

    def release_paged_prefill_graph_wrapper(self, binding: Any) -> None:
        """Drop the exclusive prefill wrapper (and its caches) bound to ``binding``."""

        entry = self._binding_prefill_graph_wrappers.pop(_binding_identity(binding), None)
        if entry is None:
            return
        wrapper_key, _binding_ref = entry
        self._prefill_wrappers.pop(wrapper_key, None)
        self._prefill_plan_workspaces.pop(wrapper_key, None)
        self._prefill_plan_cache.forget(wrapper_key)

    def release_paged_decode_graph_binding(self, binding: Any) -> None:
        """Release one binding while retaining shape-shared decode buffers."""

        self._binding_graph_wrappers.pop(_binding_identity(binding), None)

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
        """Allocate or reuse fixed-shape page metadata for decode graph preparation."""

        block_table = plan.block_table.to(dtype=torch.int32).contiguous()
        cache_seqlens = plan.kv_lens.to(dtype=torch.int32).contiguous()
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
        effective_seqlens = getattr(plan, "kv_lens", None)
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
    ) -> _DecodePlanTensors:
        """Fill fixed graph plan buffers and derive CPU indptr metadata."""

        plan = self._decode_plan_tensors(
            wrapper_key,
            block_table,
            effective_seqlens,
            int(page_size),
            index_count=_indptr_last(cpu_indptr),
        )
        return plan

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
        """Plan a graph-bound decode wrapper from caller-owned page metadata buffers."""

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
        """Materialize bounded decode indptr, page-index, and last-page-length tensors."""

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
        *,
        index_count: int | None,
    ) -> _PrefillPlanTensors:
        """Materialize bounded prefill query and KV indptr plus page-index tensors."""

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
            count = int(index_count) if index_count is not None else max_indices
            count = max(1, min(count, int(workspace.indices.numel())))
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
