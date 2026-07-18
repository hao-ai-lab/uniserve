"""General packed-forward runtime shared by segment-oriented model families.

Family-owned machinery shared by BAGEL and SenseNova for packed forward
execution: the ``PackedForwardModelMixin`` model base, the packed batch
adapter that lowers any supported op group into one segment stream, and the
packed forward CUDA-graph runner built on ``execution.graph`` primitives.
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import torch

import uniserve_worker.ops as ops
from uniserve_worker.contracts.attention_plan import GraphBinding, PagedVarlenPlan
from uniserve_worker.contracts.batches import UniForwardBatch
from uniserve_worker.contracts.forward_batch import (
    DenoiseBranchKey,
    DenoisePostprocessEntry,
    ForwardPlan,
    ForwardResult,
    TextPostprocessEntry,
)
from uniserve_worker.contracts.forward_context import get_forward_context, use_forward_context
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.contracts.outputs import TextTokenOutput
from uniserve_worker.execution.engine import (
    DeferredDecodeBurstSeqResult,
    DeferredTerminalDecodeBurstSeqResult,
    DeferredTextSeqResult,
    DenoiseDriver,
    TextDecodeRelay,
    TextImageDenoiseStep,
    sample_logits_result,
    text_image_branches,
    text_image_cfg_plan,
)
from uniserve_worker.execution.graph.bucket import padding_blocks
from uniserve_worker.execution.graph.capture import Event, record
from uniserve_worker.execution.graph.capture import Runner as Capture
from uniserve_worker.execution.graph.executor import backend_name
from uniserve_worker.foundation.env import env_flag
from uniserve_worker.foundation.errors import capability_mismatch, invalid_descriptor
from uniserve_worker.foundation.profiling import profile_range
from uniserve_worker.foundation.runtime_config import DEFAULT_DECODE_GRAPH_BATCH_SIZES
from uniserve_worker.foundation.sizing import DEFAULT_BLOCK_SIZE, ceil_div
from uniserve_worker.models.interleaved_text import hydrate_cached_prefix_from_op
from uniserve_worker.nn.diffusion import euler_step
from uniserve_worker.nn.diffusion.cfg import Branch, CfgPlan
from uniserve_worker.nn.sampler import (
    apply_sampling_batched_with_device_tokens,
    finalize_sampling_result,
    is_deferred_sampling_result,
)
from uniserve_worker.runtime.forward_batch_builder import state_block_ids_for_op
from uniserve_worker.runtime.forward_stream import (
    ForwardGraphPagedKVView,
    ForwardGraphStreamState,
    ForwardPagedKVSegment,
    ForwardPagedKVView,
    ForwardStream,
    ForwardStreamBuilder,
)
from uniserve_worker.runtime.host_staging import fill_cpu_ints, is_pinned
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import (
    PagedTextCache,
    PagedTextCacheSpanCopy,
    copy_paged_text_cache_span,
    copy_paged_text_cache_spans,
)
from uniserve_worker.runtime.tensor_staging import TextTensorStager, TextTensorStagingSlot

# ---------------------
# Packed forward graph runner
# ---------------------

logger = logging.getLogger(__name__)


_RUNNER_ATTR = "_packed_graph_runner"
_MAX_FAILURES = 2
_MAX_RESIDENT_CAPACITIES = 8


class _GraphBackendUnplanned(RuntimeError):
    """Capture completed without planning the graph-scoped attention backend."""


@dataclass
class PackedGraphState:
    key: tuple[Any, ...]
    topology_id: str
    graph: torch.cuda.CUDAGraph
    packed_embeds: torch.Tensor
    indicators: torch.Tensor
    stream_state: ForwardGraphStreamState
    kv_view: ForwardGraphPagedKVView
    plan: PagedVarlenPlan
    graph_binding: GraphBinding
    backend: Any
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    scale: float
    release_backend: Any = None
    logits: torch.Tensor | None = None
    promotion_source_pool: Any = None
    promotion_target_pool: Any = None
    promotion_source_index: torch.Tensor | None = None
    promotion_target_index: torch.Tensor | None = None
    promotion_stager: TextTensorStager = field(
        default_factory=lambda: TextTensorStager(ring_depth=3)
    )


class PackedGraphRunner(Capture):
    """Own the active-topology CUDA graph for packed forward decoder forwards."""

    def __init__(
        self,
        *,
        name: str = "packed_forward",
        default_enabled: bool | None = None,
        logger: Any = logger,
    ) -> None:
        self.name = str(name)
        self.default_enabled = True if default_enabled is None else bool(default_enabled)
        self.default_warmup = False
        self.metric_prefix = "packed_forward_"
        self.logger = logger
        self.states: dict[tuple[Any, ...], PackedGraphState] = {}
        self.disabled: set[tuple[Any, ...]] = set()
        self._capture_pool: Any = None
        self._graph_input_buffer_pool: dict[tuple[str, str, str], torch.Tensor] = {}
        self._failures = 0
        self._hard_disabled = False
        self._backend_ineligible = False
        self._replays = 0
        self._resident_order: OrderedDict[tuple[Any, ...], None] = OrderedDict()
        self.last_miss_reason: str | None = None

    def enabled(self) -> bool:
        return self.default_enabled and not self._hard_disabled and not self._backend_ineligible

    def capture_pool(self) -> Any:
        # Capacity buckets are captured lazily. The resident topology owns
        # its activation pool so its static allocations cannot alias another
        # graph executable.
        return None

    def _admit_capacity(
        self,
        key: tuple[Any, ...],
        *,
        device: torch.device | str,
    ) -> None:
        if key in self.states:
            self._touch_capacity(key)
            return
        while len(self.states) >= _MAX_RESIDENT_CAPACITIES:
            victim = next(
                (candidate for candidate in self._resident_order if candidate in self.states),
                next(iter(self.states)),
            )
            self._resident_order.pop(victim, None)
            self._retire_graph_states((victim,), device=device)

    def _touch_capacity(self, key: tuple[Any, ...]) -> None:
        self._resident_order.pop(key, None)
        self._resident_order[key] = None

    def maybe_run(
        self,
        owner: Any,
        packed_embeds: torch.Tensor,
        *,
        route_indicators: torch.Tensor,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
        text_kv_promotions: tuple[PagedTextCacheSpanCopy, ...] = (),
        text_kv_promotion_capacity: int | None = None,
    ) -> torch.Tensor | None:
        self.last_miss_reason = None
        if not self.enabled():
            self.last_miss_reason = "runner_disabled"
            return None
        if not torch.cuda.is_available():
            self.last_miss_reason = "cuda_unavailable"
            return None
        if packed_embeds.device.type != "cuda":
            self.last_miss_reason = "inputs_not_cuda"
            return None
        ctx = get_forward_context()
        backend = self._resolve_graph_backend(ctx, owner, packed_embeds, forward_stream, kv_view)
        if backend is None:
            self.last_miss_reason = "backend_ineligible"
            self._backend_ineligible = True
            self._record(ctx, Event.MISS, int(packed_embeds.shape[0]))
            if self.logger is not None:
                self.logger.warning(
                    "%s CUDA graph unavailable: no graph-capable paged-varlen attention backend",
                    self.name,
                )
            return None
        promotions = (
            tuple(text_kv_promotions)
            if packed_graph_promotions_supported(text_kv_promotions)
            else ()
        )
        key = self._graph_key(
            owner,
            packed_embeds,
            route_indicators,
            forward_stream,
            kv_view,
            backend,
            promotions,
            text_kv_promotion_capacity,
        )
        if key is None or key in self.disabled:
            self.last_miss_reason = "shape_ineligible" if key is None else "shape_disabled"
            self._record(ctx, Event.MISS, int(packed_embeds.shape[0]))
            return None
        if key not in self.states and not ctx.allow_capture:
            self.last_miss_reason = "capture_disabled"
            self._record(ctx, Event.MISS, int(packed_embeds.shape[0]))
            return None
        topology_id = _graph_topology_id(forward_stream)
        self._admit_capacity(
            key,
            device=packed_embeds.device,
        )
        out = self._capture_or_replay(
            key=key,
            device=packed_embeds.device,
            ctx=ctx,
            capture=lambda: self._capture(
                owner,
                packed_embeds,
                route_indicators=route_indicators,
                forward_stream=forward_stream,
                kv_view=kv_view,
                text_kv_promotions=promotions,
                text_kv_promotion_capacity=text_kv_promotion_capacity,
                topology_id=topology_id,
                key=key,
                ctx=ctx,
                backend=backend,
            ),
            copy_inputs=lambda state: self._copy_inputs(
                state,
                packed_embeds,
                route_indicators=route_indicators,
                forward_stream=forward_stream,
                kv_view=kv_view,
                text_kv_promotions=promotions,
            ),
            replay=self._replay,
            record=lambda event: self._record(ctx, event, int(packed_embeds.shape[0])),
            disable=lambda exc: self._disable(key, exc),
            capture_metric=f"{self.metric_prefix}graph_capture",
            input_copy_metric=f"{self.metric_prefix}graph_input_copy",
            replay_metric=f"{self.metric_prefix}graph_replay_launch",
            after_copy=self._prepare_backend,
            after_copy_metric=f"{self.metric_prefix}graph_attention_prepare",
        )
        if out is None:
            self.last_miss_reason = "capture_or_replay_failed"
            return None
        self._touch_capacity(key)
        self._replays += 1
        if self._replays == 1 and self.logger is not None:
            self.logger.info(
                "%s CUDA graph active: captured packed forward decoder (tokens=%d, rows=%d)",
                self.name,
                int(packed_embeds.shape[0]),
                len(forward_stream.segments),
            )
        return out

    def _capture(
        self,
        owner: Any,
        packed_embeds: torch.Tensor,
        *,
        route_indicators: torch.Tensor,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
        key: tuple[Any, ...],
        ctx: Any,
        backend: Any,
        text_kv_promotions: tuple[PagedTextCacheSpanCopy, ...] = (),
        text_kv_promotion_capacity: int | None = None,
        topology_id: str,
    ) -> PackedGraphState:
        first_attn = _first_attention(owner)
        block_width_capacity = _graph_block_width_capacity(kv_view)
        graph_kv_view = ForwardGraphPagedKVView(
            kv_view.pool,
            kv_view.segments,
            block_width_capacity=block_width_capacity,
        )
        max_context_len = graph_kv_view.max_seqlen_k()
        plan = PagedVarlenPlan(
            residency_cache=graph_kv_view,
            block_table=graph_kv_view.block_table(device=packed_embeds.device),
            cache_seqlens=graph_kv_view.cache_seqlens_after(device=packed_embeds.device),
            cu_seqlens_q=forward_stream.cu_seqlens_q.detach().clone(),
            cu_seqlens_k=graph_kv_view.cu_seqlens_after(device=packed_embeds.device),
            max_seqlen_q=int(forward_stream.visible_end.shape[1]),
            max_seqlen_k=max_context_len,
            max_context_len=max_context_len,
            mode=ForwardMode.MIXED,
        )
        graph_binding = GraphBinding()
        state = PackedGraphState(
            key=key,
            topology_id=str(topology_id),
            graph=torch.cuda.CUDAGraph(),
            packed_embeds=packed_embeds.detach().clone(),
            indicators=route_indicators.detach().clone(),
            stream_state=ForwardGraphStreamState.from_stream(forward_stream),
            kv_view=graph_kv_view,
            plan=plan,
            graph_binding=graph_binding,
            backend=backend,
            num_q_heads=int(getattr(first_attn, "num_heads")),
            num_kv_heads=int(getattr(first_attn, "num_kv_heads")),
            head_dim=int(getattr(first_attn, "head_dim")),
            page_size=int(kv_view.pool.block_size),
            scale=_attention_scale(first_attn),
        )
        if text_kv_promotions:
            source_pool, target_pool, source_index, target_index = _promotion_index_tensors(
                text_kv_promotions,
                capacity=text_kv_promotion_capacity,
            )
            state.promotion_source_pool = source_pool
            state.promotion_target_pool = target_pool
            state.promotion_source_index = source_index
            state.promotion_target_index = target_index
        bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
        release = getattr(backend, "release_paged_prefill_graph_wrapper", None)
        if callable(bind) and callable(release):
            bind(graph_binding, plan, device=packed_embeds.device)
            state.release_backend = lambda: release(graph_binding)
        graph_ctx = replace(
            ctx,
            attention_backend=backend,
            attention_plan=plan,
            graph_binding=graph_binding,
            stats=None,
        )

        def run() -> torch.Tensor:
            # _copy_inputs rebuilds state.plan before each warmup and capture run;
            # publish that snapshot so capture records the same plan
            # _prepare_backend plans from.
            with use_forward_context(replace(graph_ctx, attention_plan=state.plan)):
                hidden = owner.packed_decoder_forward(
                    state.packed_embeds,
                    route_indicators=state.indicators,
                    indexes=state.stream_state.stream.indexes,
                    forward_stream=state.stream_state.stream,
                    kv_view=state.kv_view,
                )
                self._copy_promotions_in_graph(state)
                return hidden

        try:
            self._capture_graph_state(
                device=packed_embeds.device,
                state=state,
                run=run,
                copy_inputs=lambda capture_state: self._copy_inputs(
                    capture_state,
                    packed_embeds,
                    route_indicators=route_indicators,
                    forward_stream=forward_stream,
                    kv_view=kv_view,
                    text_kv_promotions=text_kv_promotions,
                ),
                before_run=self._prepare_backend,
            )
            planned = getattr(backend, "paged_prefill_graph_wrapper_planned", None)
            if callable(planned) and not planned(graph_binding):
                raise _GraphBackendUnplanned(
                    "captured packed forward forward did not plan the graph-scoped prefill wrapper"
                )
        except BaseException:
            release_backend = state.release_backend
            state.release_backend = None
            if callable(release_backend):
                release_backend()
            raise
        return state

    @staticmethod
    def _copy_inputs(
        state: PackedGraphState,
        packed_embeds: torch.Tensor,
        *,
        route_indicators: torch.Tensor,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
        text_kv_promotions: tuple[PagedTextCacheSpanCopy, ...] = (),
    ) -> None:
        state.packed_embeds.copy_(packed_embeds, non_blocking=True)
        state.indicators.copy_(route_indicators, non_blocking=True)
        state.stream_state.refresh(forward_stream)
        state.kv_view.refresh(kv_view.segments)
        # Publish a fresh frozen plan reusing the kv-view's stable device tensors
        # with the refreshed geometry. The graph reads those buffers (stable
        # addresses); wrapper identity stays on the separate graph binding.
        state.plan = replace(
            state.plan,
            block_table=state.kv_view.block_table(device=packed_embeds.device),
            cache_seqlens=state.kv_view.cache_seqlens_after(device=packed_embeds.device),
            cu_seqlens_q=state.stream_state.stream.cu_seqlens_q,
            cu_seqlens_k=state.kv_view.cu_seqlens_after(device=packed_embeds.device),
            max_context_len=state.kv_view.max_seqlen_k(),
        )
        if state.promotion_source_index is not None and text_kv_promotions:
            PackedGraphRunner._copy_promotion_indices(state, text_kv_promotions)

    @staticmethod
    def _copy_promotion_indices(
        state: PackedGraphState,
        text_kv_promotions: tuple[PagedTextCacheSpanCopy, ...],
    ) -> None:
        source_index = state.promotion_source_index
        target_index = state.promotion_target_index
        if source_index is None or target_index is None:
            return
        source_pool, target_pool, source_positions, target_positions = _promotion_index_values(
            text_kv_promotions,
            capacity=int(source_index.numel()),
        )
        if (
            source_pool is not state.promotion_source_pool
            or target_pool is not state.promotion_target_pool
        ):
            raise invalid_descriptor("packed forward graph promotion pool changed")
        slot = state.promotion_stager.acquire_slot(device=source_index.device)
        try:
            _copy_long_values_to_tensor(
                source_index,
                "promotion_source_index",
                source_positions,
                slot=slot,
            )
            _copy_long_values_to_tensor(
                target_index,
                "promotion_target_index",
                target_positions,
                slot=slot,
            )
        finally:
            state.promotion_stager.mark_slot_submitted(slot, device=source_index.device)

    @staticmethod
    def _copy_promotions_in_graph(state: PackedGraphState) -> None:
        source_index = state.promotion_source_index
        target_index = state.promotion_target_index
        source_pool = state.promotion_source_pool
        target_pool = state.promotion_target_pool
        if source_index is None or target_index is None or source_pool is None or target_pool is None:
            return
        layer_count = int(source_pool.num_layers)
        source_k = source_pool.k.reshape(layer_count, -1, source_pool.n_kv, source_pool.head_dim)
        source_v = source_pool.v.reshape(layer_count, -1, source_pool.n_kv, source_pool.head_dim)
        target_k = target_pool.k.reshape(layer_count, -1, target_pool.n_kv, target_pool.head_dim)
        target_v = target_pool.v.reshape(layer_count, -1, target_pool.n_kv, target_pool.head_dim)
        target_k.index_copy_(1, target_index, source_k.index_select(1, source_index))
        target_v.index_copy_(1, target_index, source_v.index_select(1, source_index))

    @staticmethod
    def _prepare_backend(state: PackedGraphState) -> None:
        prepare = getattr(state.backend, "prepare_paged_prefill_cuda_graph", None)
        if not callable(prepare):
            return
        prepare(
            state.graph_binding,
            state.plan,
            num_q_heads=state.num_q_heads,
            num_kv_heads=state.num_kv_heads,
            head_dim=state.head_dim,
            page_size=state.page_size,
            q_dtype=state.packed_embeds.dtype,
            kv_dtype=state.kv_view.pool.k.dtype,
            causal=False,
            scale=state.scale,
        )

    @staticmethod
    def _replay(state: PackedGraphState) -> torch.Tensor:
        state.graph.replay()
        assert state.logits is not None
        return state.logits

    def _record(self, ctx: Any, event: Event, tokens: int) -> None:
        record(
            ctx,
            event,
            unpadded_tokens=int(tokens),
            padded_tokens=int(tokens),
        )

    def _disable(self, key: tuple[Any, ...], exc: BaseException) -> None:
        self.disabled.add(key)
        self._resident_order.pop(key, None)
        state = self.states.pop(key, None)
        if state is not None and callable(state.release_backend):
            try:
                state.release_backend()
            except Exception:  # pragma: no cover - defensive release
                pass
        self._failures += 1
        if self._failures >= _MAX_FAILURES:
            self._hard_disabled = True
        if self.logger is not None:
            self.logger.warning(
                "disabling %s CUDA graph (%s failure(s)%s): %s",
                self.name,
                self._failures,
                "; runner hard-disabled" if self._hard_disabled else "",
                exc,
            )

    def _resolve_graph_backend(
        self,
        ctx: Any,
        owner: Any,
        packed_embeds: torch.Tensor,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
    ) -> Any | None:
        try:
            first_attn = _first_attention(owner)
            preferred = getattr(ctx, "attention_preference", None)
            explicit_backend = backend_name(preferred)
            q_probe = packed_embeds.new_empty(
                (
                    int(packed_embeds.shape[0]),
                    int(getattr(first_attn, "num_heads")),
                    int(getattr(first_attn, "head_dim")),
                )
            )
            k_cache, v_cache = kv_view.pool.layer_cache(0)
            req = ops.AttentionReq(
                q=q_probe,
                k=k_cache,
                v=v_cache,
                regime=ops.AttentionRegime.VISIBLE_END,
                causal=False,
                scale=_attention_scale(first_attn),
                ctx=ctx,
                visible_end=forward_stream.visible_end,
                cu_seqlens_q=forward_stream.cu_seqlens_q,
                page_table=kv_view.block_table(device=packed_embeds.device),
                seqused_k=kv_view.cache_seqlens_after(device=packed_embeds.device),
                max_seqlen_q=int(forward_stream.visible_end.shape[1]),
                max_seqlen_k=_max_context_len(kv_view),
                use_prefix_bounds=True,
                fully_visible=bool(forward_stream.fully_visible),
            )
            for provider in ops.attention_dispatcher().ordered(preferred):
                try:
                    if not provider.can_run(req):
                        continue
                except Exception:
                    continue
                backend = getattr(provider, "backend", None)
                if backend is None:
                    backend = getattr(req, "backend", None) or getattr(ctx, "attention_backend", None)
                if backend is not None and _backend_can_host_graph(backend):
                    return backend
                if explicit_backend is not None and str(getattr(provider, "name", "")).lower() == explicit_backend:
                    return None
                if explicit_backend is not None and str(getattr(backend, "name", "")).lower() == explicit_backend:
                    return None
        except Exception:
            if self.logger is not None:
                self.logger.debug("%s graph backend probe failed", self.name, exc_info=True)
        return None

    @staticmethod
    def _graph_key(
        owner: Any,
        packed_embeds: torch.Tensor,
        route_indicators: torch.Tensor,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
        backend: Any,
        text_kv_promotions: tuple[PagedTextCacheSpanCopy, ...] = (),
        text_kv_promotion_capacity: int | None = None,
    ) -> tuple[Any, ...] | None:
        if tuple(route_indicators.shape) != (int(packed_embeds.shape[0]),):
            return None
        block_width_capacity = _graph_block_width_capacity(kv_view)
        if block_width_capacity <= 0:
            return None
        return (
            id(owner),
            id(kv_view.pool),
            str(getattr(backend, "name", type(backend).__name__)),
            str(packed_embeds.device),
            str(packed_embeds.dtype),
            tuple(int(dim) for dim in packed_embeds.shape),
            str(route_indicators.dtype),
            tuple(int(dim) for dim in route_indicators.shape),
            _stream_capacity(forward_stream),
            _route_index_capacity(forward_stream),
            _kv_capacity(kv_view),
            int(block_width_capacity),
            int(kv_view.pool.block_size),
            int(block_width_capacity * int(kv_view.pool.block_size)),
            _promotion_geometry(
                text_kv_promotions,
                capacity=text_kv_promotion_capacity,
            ),
        )


def _first_attention(owner: Any) -> Any:
    packed_attention = getattr(owner, "packed_graph_attention", None)
    if not callable(packed_attention):
        raise RuntimeError("packed forward graph requires an explicit attention binding")
    return packed_attention()


def _graph_topology_id(forward_stream: ForwardStream) -> str:
    return "fully_visible" if forward_stream.fully_visible else "visible_end"


def _attention_scale(attention: Any) -> float:
    scale = getattr(attention, "scaling", None)
    if scale is None:
        scale = getattr(attention, "scale", None)
    if scale is None:
        raise RuntimeError("packed forward graph attention does not expose a scale")
    return float(scale)


def _stream_capacity(forward_stream: ForwardStream) -> tuple[Any, ...]:
    return (
        len(forward_stream.segments),
        sum(int(segment.q_len) for segment in forward_stream.segments),
        tuple(int(dim) for dim in forward_stream.cu_seqlens_q.shape),
        tuple(int(dim) for dim in forward_stream.visible_end.shape),
        tuple(int(dim) for dim in forward_stream.indexes.shape),
        bool(forward_stream.fully_visible),
    )


def _kv_capacity(kv_view: ForwardPagedKVView) -> tuple[int, int, int, bool]:
    return (
        len(kv_view.segments),
        sum(int(segment.q_len) for segment in kv_view.segments),
        sum(int(segment.q_len) for segment in kv_view.segments if segment.write_kv),
        all(bool(segment.write_kv) for segment in kv_view.segments),
    )


def _route_index_capacity(forward_stream: ForwardStream) -> tuple[Any, ...]:
    def capacity(indices: torch.Tensor | None) -> tuple[Any, ...] | None:
        if indices is None:
            return None
        return (tuple(int(dim) for dim in indices.shape), str(indices.dtype))

    return (
        capacity(forward_stream.und_indices),
        capacity(forward_stream.gen_indices),
    )


def packed_graph_promotions_supported(
    promotions: tuple[PagedTextCacheSpanCopy, ...] | list[PagedTextCacheSpanCopy],
) -> bool:
    if not promotions:
        return False
    first_source = promotions[0].source.pool
    first_target = promotions[0].target.pool
    total = 0
    for promotion in promotions:
        source_pool = promotion.source.pool
        target_pool = promotion.target.pool
        if source_pool is not first_source or target_pool is not first_target:
            return False
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
        if source_pool.k.dtype != target_pool.k.dtype or source_pool.v.dtype != target_pool.v.dtype:
            return False
        if source_pool.num_layers != target_pool.num_layers:
            return False
        if source_pool.n_kv != target_pool.n_kv or source_pool.head_dim != target_pool.head_dim:
            return False
        total += max(0, int(promotion.length))
    return total > 0


def _promotion_geometry(
    promotions: tuple[PagedTextCacheSpanCopy, ...],
    *,
    capacity: int | None = None,
) -> tuple[Any, ...]:
    if not promotions:
        return ()
    source_pool = promotions[0].source.pool
    target_pool = promotions[0].target.pool
    return (
        id(source_pool),
        id(target_pool),
        str(source_pool.k.device),
        str(source_pool.k.dtype),
        str(target_pool.k.dtype),
        max(
            sum(max(0, int(promotion.length)) for promotion in promotions),
            0 if capacity is None else int(capacity),
        ),
    )


def _promotion_index_tensors(
    promotions: tuple[PagedTextCacheSpanCopy, ...],
    *,
    capacity: int | None = None,
) -> tuple[Any, Any, torch.Tensor, torch.Tensor]:
    source_pool, target_pool, source_positions, target_positions = _promotion_index_values(
        promotions,
        capacity=capacity,
    )
    device = source_pool.k.device
    return (
        source_pool,
        target_pool,
        torch.tensor(source_positions, device=device, dtype=torch.long),
        torch.tensor(target_positions, device=device, dtype=torch.long),
    )


def _promotion_index_values(
    promotions: tuple[PagedTextCacheSpanCopy, ...],
    *,
    capacity: int | None = None,
) -> tuple[Any, Any, list[int], list[int]]:
    source_pool = promotions[0].source.pool
    target_pool = promotions[0].target.pool
    source_positions: list[int] = []
    target_positions: list[int] = []
    for promotion in promotions:
        source_positions.extend(
            _cache_positions(
                promotion.source.pool,
                promotion.source.block_ids,
                promotion.start,
                promotion.length,
            )
        )
        target_positions.extend(
            _cache_positions(
                promotion.target.pool,
                promotion.target.block_ids,
                promotion.start,
                promotion.length,
            )
        )
    resolved_capacity = (
        len(source_positions)
        if capacity is None
        else max(
            len(source_positions),
            int(capacity),
        )
    )
    if resolved_capacity > len(source_positions):
        if not source_positions:
            raise invalid_descriptor("packed forward graph promotion capacity requires an index")
        source_positions.extend([source_positions[0]] * (resolved_capacity - len(source_positions)))
        target_positions.extend([target_positions[0]] * (resolved_capacity - len(target_positions)))
    return source_pool, target_pool, source_positions, target_positions


def _copy_long_values_to_tensor(
    target: torch.Tensor,
    name: str,
    values: list[int],
    *,
    slot: TextTensorStagingSlot,
) -> None:
    if int(target.numel()) != len(values):
        raise invalid_descriptor("packed forward graph promotion index length changed")
    cpu = slot.long_buffer(name, len(values), pin=target.device.type == "cuda")
    fill_cpu_ints(cpu, values)
    target.copy_(cpu, non_blocking=target.device.type == "cuda" and is_pinned(cpu))


def _cache_positions(pool: Any, block_ids: list[int], start: int, length: int) -> list[int]:
    positions: list[int] = []
    block_size = int(pool.block_size)
    for block_id, offset, count in pool.spans(list(block_ids), int(start), int(length)):
        base = int(block_id) * block_size + int(offset)
        positions.extend(range(base, base + int(count)))
    return positions


def _max_context_len(kv_view: ForwardPagedKVView) -> int:
    return max((len(seg.block_ids) * int(kv_view.pool.block_size) for seg in kv_view.segments), default=0)


def _graph_block_width_capacity(kv_view: ForwardPagedKVView) -> int:
    required = max((len(seg.block_ids) for seg in kv_view.segments), default=0)
    if required <= 0:
        return 0
    bucket = 1 << (required - 1).bit_length()
    return min(bucket, int(kv_view.pool.num_blocks))


def _backend_can_host_graph(backend: Any) -> bool:
    try:
        caps = backend.capabilities()
    except Exception:
        return False
    return bool(getattr(caps, "visible_end_cuda_graph", False))


def packed_graph_runner(owner: Any) -> PackedGraphRunner:
    runner = getattr(owner, _RUNNER_ATTR, None)
    if runner is None:
        runner = PackedGraphRunner()
        setattr(owner, _RUNNER_ATTR, runner)
    return runner


def maybe_run_packed_graph(
    owner: Any,
    packed_embeds: torch.Tensor,
    *,
    route_indicators: torch.Tensor,
    forward_stream: ForwardStream,
    kv_view: ForwardPagedKVView,
    text_kv_promotions: tuple[PagedTextCacheSpanCopy, ...] = (),
    text_kv_promotion_capacity: int | None = None,
) -> torch.Tensor | None:
    runner = getattr(owner, _RUNNER_ATTR, None)
    if runner is None:
        runner = packed_graph_runner(owner)
    return runner.maybe_run(
        owner,
        packed_embeds,
        route_indicators=route_indicators,
        forward_stream=forward_stream,
        kv_view=kv_view,
        text_kv_promotions=text_kv_promotions,
        text_kv_promotion_capacity=text_kv_promotion_capacity,
    )


# ---------------------
# Packed model mixin
# ---------------------

class PackedForwardModelMixin:
    """Shared cache staging and segment construction for packed models."""

    def run_segment_graph(
        self,
        plan: ForwardPlan,
        *,
        request_states: Any,
        result_publisher: Any | None = None,
    ) -> ForwardResult | None:
        """Execute any graphable segment-table composition owned by this family."""

        dispatch_batch = plan.runtime_handles.get("dispatch_batch")
        if not isinstance(dispatch_batch, UniForwardBatch):
            dispatch_batch = UniForwardBatch.from_ops(plan.ops)
        if any(
            int(op.get("decode_token_count") or 1) > 1
            or int(op.get("denoise_step_count") or 1) > 1
            for op in dispatch_batch.ops
        ):
            run = getattr(self, "_run_forward_adapter", None)
            if not callable(run):
                return None
            outputs = run(
                dispatch_batch,
                request_states=request_states,
                group=list(enumerate(dispatch_batch.ops)),
                defer_text_cpu_results=bool(
                    plan.runtime_handles.get("defer_text_cpu_results", False)
                ),
            )
            if isinstance(outputs, ForwardResult):
                return outputs
            if not isinstance(outputs, Sequence) or isinstance(
                outputs, (str, bytes, bytearray)
            ):
                raise invalid_descriptor(
                    "forward adapter must return one result per operation"
                )
            if len(outputs) != len(plan.rows):
                raise invalid_descriptor(
                    "forward adapter returned the wrong number of results"
                )
            return ForwardResult(runtime_outputs=tuple(outputs))

        denoise_steps: list[tuple[int, TextImageDenoiseStep]] = []
        for row in plan.rows:
            if row.mode is not ForwardMode.DENOISE:
                continue
            state = request_states.get(int(row.req_id))
            step = self.prepare_denoise(state, dict(row.op))
            denoise_steps.append((int(row.row_index), step))
            extra = getattr(step, "extra", None)
            image = extra.get("img") if isinstance(extra, dict) else None
            residual_state = getattr(image, "residual_cache", None)
            if residual_state is not None:
                residual_state.invalidate()

        result = self.run_packed_forward_result(
            dispatch_batch,
            request_states,
            denoise_steps,
            defer_text_cpu_results=bool(
                plan.runtime_handles.get("defer_text_cpu_results", False)
            ),
            allow_graph=True,
            require_graph=True,
        )
        if result is None or not plan.shape.commit_row_count:
            return result
        if result_publisher is None:
            raise invalid_descriptor("segment result publication is not bound")
        commit_rows = tuple(row for row in plan.rows if row.mode is ForwardMode.COMMIT)
        commit_result = result_publisher.forward_result(
            tuple(
                (int(row.req_id), request_states.get(int(row.req_id)), row.op)
                for row in commit_rows
            ),
            self,
            row_indices=tuple(int(row.row_index) for row in commit_rows),
        )
        if not isinstance(commit_result, ForwardResult) or commit_result.commit_outputs is None:
            raise invalid_descriptor("segment result publisher returned no outputs")
        commit_outputs = dict(result.commit_outputs or {})
        commit_outputs.update(commit_result.commit_outputs)
        return replace(result, commit_outputs=commit_outputs)

    def run_packed_forward_result(
        self,
        batch: UniForwardBatch,
        request_states: Any,
        denoise_steps: "list[tuple[int, TextImageDenoiseStep]]",
        *,
        defer_text_cpu_results: bool = False,
        allow_graph: bool = False,
        require_graph: bool = False,
    ) -> "ForwardResult | None":
        # Family entry the neutral graph program dispatches through; keeps the
        # packed forward forward owned by the family rather than imported upward.
        return run_packed_forward_result(
            self,
            batch,
            request_states,
            denoise_steps,
            defer_text_cpu_results=defer_text_cpu_results,
            allow_graph=allow_graph,
            require_graph=require_graph,
        )

    def _extend_cache_blocks(self, cache: Any, op: dict[str, Any]) -> None:
        self._text_driver().extend_cache_blocks(cache, op)

    def _ensure_host_cache(self, cache: Any) -> None:
        self._text_driver().ensure_host_cache(cache)

    @staticmethod
    def _same_kv_pool(pool: Any, first_pool: Any) -> bool:
        return first_pool is None or pool is first_pool

    def _stage_text_cache_for_forward(
        self,
        source: PagedTextCache,
        *,
        target_pool: PagedKVPool,
        end_len: int,
        pending_prefix_copies: list[PagedTextCacheSpanCopy] | None = None,
    ) -> PagedTextCache:
        scratch_pools = (
            getattr(self, "scratch_pool", None),
            getattr(self, "gen_scratch_pool", None),
        )
        if not any(target_pool is pool for pool in scratch_pools if pool is not None):
            raise RuntimeError("forward text staging target must be a scratch KV pool")
        allocator = self.residency.require_allocator_for_pool(
            target_pool,
            label="forward scratch KV pool",
        )
        staged_by_pool = getattr(source, "_uniserve_forward_staging", None)
        if not isinstance(staged_by_pool, dict):
            staged_by_pool = {}
            setattr(source, "_uniserve_forward_staging", staged_by_pool)
        key = id(target_pool)
        staged = staged_by_pool.get(key)
        source_len = int(source.length)
        source_prefix = self._forward_staging_source_prefix(source, source_len)
        if (
            staged is not None
            and getattr(staged, "pool", None) is target_pool
            and int(getattr(staged, "length", -1)) <= source_len
            and getattr(staged, "_uniserve_forward_staging_source_prefix", ())
            == self._forward_staging_source_prefix(source, int(getattr(staged, "length", 0)))
        ):
            staged.ensure_capacity(int(end_len))
            copied_len = int(staged.length)
            if copied_len < source_len:
                if pending_prefix_copies is None:
                    copy_paged_text_cache_span(
                        source,
                        staged,
                        start=copied_len,
                        length=source_len - copied_len,
                        num_layers=self.num_layers,
                        missing_message="cannot extend packed forward staging without a paged source cache",
                    )
                    setattr(staged, "_uniserve_forward_staging_source_prefix", source_prefix)
                else:
                    pending_prefix_copies.append(
                        PagedTextCacheSpanCopy(
                            source=source,
                            target=staged,
                            start=copied_len,
                            length=source_len - copied_len,
                        )
                    )
                staged.length = source_len
            if copied_len >= source_len:
                setattr(staged, "_uniserve_forward_staging_source_prefix", source_prefix)
            return staged
        if staged is not None:
            self.residency.release_scratch_cache(staged)
        staged = PagedTextCache(
            target_pool,
            [],
            num_layers=self.num_layers,
            length=0,
            allocate_blocks=allocator,
        )
        staged.ensure_capacity(int(end_len))
        if source_len > 0:
            if pending_prefix_copies is None:
                copy_paged_text_cache_span(
                    source,
                    staged,
                    start=0,
                    length=source_len,
                    num_layers=self.num_layers,
                    missing_message="cannot stage packed forward prefix without a paged source cache",
                )
                setattr(staged, "_uniserve_forward_staging_source_prefix", source_prefix)
            else:
                pending_prefix_copies.append(
                    PagedTextCacheSpanCopy(
                        source=source,
                        target=staged,
                        start=0,
                        length=source_len,
                    )
                )
        elif pending_prefix_copies is None:
            setattr(staged, "_uniserve_forward_staging_source_prefix", source_prefix)
        staged.length = source_len
        if source_len <= 0:
            setattr(staged, "_uniserve_forward_staging_source_prefix", source_prefix)
        staged_by_pool[key] = staged
        return staged

    def _mark_forward_staging_advanced(
        self,
        staged: PagedTextCache,
        source: PagedTextCache,
        new_len: int,
    ) -> None:
        staged.length = int(new_len)
        setattr(
            staged,
            "_uniserve_forward_staging_source_prefix",
            self._forward_staging_source_prefix(source, int(new_len)),
        )

    def _release_forward_staging_for_cache(self, cache: Any) -> None:
        staged_by_pool = getattr(cache, "_uniserve_forward_staging", None)
        if not isinstance(staged_by_pool, dict):
            return
        seen: set[int] = set()
        for staged in staged_by_pool.values():
            staged_id = id(staged)
            if staged_id in seen:
                continue
            seen.add(staged_id)
            self.residency.release_scratch_cache(staged)
        staged_by_pool.clear()

    @staticmethod
    def _forward_staging_source_prefix(
        cache: PagedTextCache,
        length: int,
    ) -> tuple[int, ...]:
        length = int(length)
        if length <= 0:
            return ()
        block_size = int(
            getattr(cache.pool, "block_size", DEFAULT_BLOCK_SIZE) or DEFAULT_BLOCK_SIZE
        )
        block_count = ceil_div(length, block_size)
        return tuple(int(block_id) for block_id in list(cache.block_ids)[:block_count])

    def _forward_target_pool(
        self,
        denoise_steps: list[tuple[int, TextImageDenoiseStep]],
    ) -> PagedKVPool | None:
        target_pool = None
        for _row_index, step in denoise_steps:
            image = step.extra["img"]
            for branch in text_image_branches(step):
                _indexes, cache = self._denoise_branch_inputs(image, branch)
                pool = getattr(cache, "pool", None)
                if pool is None:
                    return None
                if target_pool is None:
                    target_pool = pool
                elif pool is not target_pool:
                    return None
        return target_pool

    def _add_text_forward_segment(
        self,
        *,
        builder: ForwardStreamBuilder,
        kv_segments: list[ForwardPagedKVSegment],
        row_index: int,
        req_id: int,
        op: dict[str, Any],
        mode: ForwardMode,
        cache: Any,
        q_len: int,
        start_pos: int,
        device: torch.device,
    ) -> None:
        del device
        builder.add_segment(
            op_index=row_index,
            req_id=req_id,
            kind=str(op["kind"]),
            mode=mode,
            modality="und",
            segment_class="decode" if mode is ForwardMode.DECODE else "extend",
            q_len=q_len,
            prefix_len=int(cache.past.length),
            visible_policy="causal",
            index_start=int(start_pos),
        )
        kv_segments.append(
            ForwardPagedKVSegment(
                block_ids=tuple(cache.past.block_ids),
                base_len=int(cache.past.length),
                q_len=q_len,
                write_kv=True,
            )
        )

    def _add_denoise_forward_segment(
        self,
        *,
        builder: ForwardStreamBuilder,
        kv_segments: list[ForwardPagedKVSegment],
        row_index: int,
        req_id: int,
        op: dict[str, Any],
        cache: Any,
        indexes: torch.Tensor,
        q_len: int,
        branch_index: int,
        device: torch.device,
    ) -> None:
        builder.add_segment(
            op_index=row_index,
            req_id=req_id,
            kind=str(op["kind"]),
            mode=ForwardMode.DENOISE,
            modality="gen",
            segment_class="denoise",
            q_len=q_len,
            prefix_len=int(cache.length),
            branch_id=branch_index,
            visible_policy="bidirectional",
            indexes=indexes.to(device=device),
        )
        kv_segments.append(
            ForwardPagedKVSegment(
                block_ids=tuple(cache.block_ids),
                base_len=int(cache.length),
                q_len=q_len,
                write_kv=True,
                persist_kv=False,
                branch_id=branch_index,
            )
        )

    def _wait_gen_cache_ready(self, cache: Any) -> None:
        del cache

    def packed_denoise_indicators(
        self,
        step: TextImageDenoiseStep,
        q_len: int,
    ) -> torch.Tensor | None:
        del step, q_len
        return None


# ---------------------
# Packed visible forward
# ---------------------

logger = logging.getLogger(__name__)
_DECODE_RELAY = TextDecodeRelay()

_PACKED_FORWARD_TIMING = env_flag("UNISERVE_PACKED_FORWARD_TIMING")
_PACKED_FORWARD_TIMING_SYNC = env_flag("UNISERVE_PACKED_FORWARD_TIMING_SYNC")
_PACKED_FORWARD_CANONICAL_ORDER = env_flag("UNISERVE_PACKED_FORWARD_CANONICAL_ORDER", default=True)

TextResultSlot = tuple[int, int, int, PagedTextCache, PagedTextCache, int, int]
DenoiseResultSlot = tuple[int, TextImageDenoiseStep, int, int, Branch]


@dataclass
class _PendingTextBuildRow:
    row_index: int
    q_len: int
    input_ids: torch.Tensor
    persistent_cache: PagedTextCache
    staged_cache: PagedTextCache
    base_len: int
    last_input_token: int


def _append_decode_graph_padding(
    *,
    builder: ForwardStreamBuilder,
    kv_segments: list[ForwardPagedKVSegment],
    embed_chunks: list[torch.Tensor],
    indicator_chunks: list[tuple[int, bool] | torch.Tensor],
    padding_pool: Any,
    decode_rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> int:
    capacity = max(DEFAULT_DECODE_GRAPH_BATCH_SIZES)
    decode_rows = int(decode_rows)
    if decode_rows <= 0 or decode_rows >= capacity:
        return decode_rows
    padding_rows = capacity - decode_rows
    block_ids = padding_blocks(padding_pool)
    block_size = int(getattr(padding_pool, "block_size", 0) or 0)
    if not block_ids:
        raise invalid_descriptor("packed decode graph padding requires reserved KV blocks")
    if block_size <= 0 or padding_rows > len(block_ids) * block_size:
        raise invalid_descriptor("packed decode graph padding exceeds the reserved KV blocks")
    for offset in range(padding_rows):
        base_len = offset
        builder.add_segment(
            op_index=-1,
            req_id=-1,
            kind="decode_und",
            mode=ForwardMode.DECODE,
            modality="und",
            segment_class="decode",
            q_len=1,
            prefix_len=base_len,
            visible_policy="causal",
            index_start=base_len,
        )
        kv_segments.append(
            ForwardPagedKVSegment(
                block_ids=block_ids,
                base_len=base_len,
                q_len=1,
                write_kv=True,
                persist_kv=True,
            )
        )
    embed_chunks.append(
        torch.zeros(
            (padding_rows, int(hidden_size)),
            dtype=dtype,
            device=device,
        )
    )
    indicator_chunks.append((padding_rows, False))
    return capacity


@dataclass
class PackedForwardPlan:
    batch: UniForwardBatch
    denoise_steps: list[tuple[int, TextImageDenoiseStep]]
    results: list[Any]
    text_result_slots: list[TextResultSlot] = field(default_factory=list)
    denoise_result_slots: list[DenoiseResultSlot] = field(default_factory=list)
    denoise_cfg_plans: dict[int, CfgPlan] = field(default_factory=dict)

    def denoise_step_for_row(self, row_index: int) -> TextImageDenoiseStep:
        for result_index, step in self.denoise_steps:
            if int(result_index) == int(row_index):
                return step
        raise invalid_descriptor(f"no denoise step prepared for row {int(row_index)}")

    def add_text_slot(
        self,
        *,
        row_index: int,
        segment_start: int,
        q_len: int,
        persistent_cache: PagedTextCache,
        staged_cache: PagedTextCache,
        base_len: int,
        last_input_token: int,
    ) -> None:
        self.text_result_slots.append(
            (
                int(row_index),
                int(segment_start),
                int(q_len),
                persistent_cache,
                staged_cache,
                int(base_len),
                int(last_input_token),
            )
        )

    def add_denoise_slot(
        self,
        *,
        row_index: int,
        step: TextImageDenoiseStep,
        segment_start: int,
        q_len: int,
        branch: Branch,
    ) -> None:
        self.denoise_result_slots.append(
            (int(row_index), step, int(segment_start), int(q_len), branch)
        )

    def set_denoise_cfg_plan(self, row_index: int, cfg_plan: CfgPlan) -> None:
        self.denoise_cfg_plans[int(row_index)] = cfg_plan

    def denoise_cfg_plan_for_row(self, row_index: int) -> CfgPlan:
        cfg_plan = self.denoise_cfg_plans.get(int(row_index))
        if cfg_plan is None:
            raise invalid_descriptor(f"no denoise CFG plan prepared for row {int(row_index)}")
        return cfg_plan

    def set_text_result(self, row_index: int, output: Any) -> None:
        self.results[int(row_index)] = output

    def set_denoise_result(self, row_index: int, step: TextImageDenoiseStep) -> None:
        self.results[int(row_index)] = {
            "req_id": step.req_id,
            "denoise_done": step.step_index + 1 >= step.total_steps,
            "num_steps_done": step.step_index + 1,
        }


class PackedForward:
    """Owns packed forward-forward row slots, cache writeback, and output order."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def execute(
        self,
        batch: UniForwardBatch,
        request_states: Any,
        denoise_steps: list[tuple[int, TextImageDenoiseStep]],
        results: list[Any],
        *,
        defer_text_cpu_results: bool = False,
        allow_graph: bool = True,
        require_graph: bool = False,
    ) -> bool:
        plan = PackedForwardPlan(batch=batch, denoise_steps=denoise_steps, results=results)
        result = _run_packed_forward_impl(
            self.owner,
            plan,
            request_states,
            defer_text_cpu_results=defer_text_cpu_results,
            allow_graph=allow_graph,
            require_graph=require_graph,
        )
        return bool(result)

    def execute_forward_result(
        self,
        batch: UniForwardBatch,
        request_states: Any,
        denoise_steps: list[tuple[int, TextImageDenoiseStep]],
        *,
        defer_text_cpu_results: bool = False,
        allow_graph: bool = False,
        require_graph: bool = False,
    ) -> ForwardResult | None:
        plan = PackedForwardPlan(
            batch=batch,
            denoise_steps=denoise_steps,
            results=[None] * len(batch.ops),
        )
        result = _run_packed_forward_impl(
            self.owner,
            plan,
            request_states,
            defer_text_cpu_results=defer_text_cpu_results,
            return_forward_result=True,
            allow_graph=allow_graph,
            require_graph=require_graph,
        )
        return result if isinstance(result, ForwardResult) else None


def run_packed_forward(
    owner,
    batch: UniForwardBatch,
    request_states: Any,
    denoise_steps: list[tuple[int, TextImageDenoiseStep]],
    results: list[Any],
    *,
    defer_text_cpu_results: bool = False,
    allow_graph: bool = True,
    require_graph: bool = False,
) -> bool:
    return PackedForward(owner).execute(
        batch,
        request_states,
        denoise_steps,
        results,
        defer_text_cpu_results=defer_text_cpu_results,
        allow_graph=allow_graph,
        require_graph=require_graph,
    )


def run_packed_forward_result(
    owner,
    batch: UniForwardBatch,
    request_states: Any,
    denoise_steps: list[tuple[int, TextImageDenoiseStep]],
    *,
    defer_text_cpu_results: bool = False,
    allow_graph: bool = False,
    require_graph: bool = False,
) -> ForwardResult | None:
    return PackedForward(owner).execute_forward_result(
        batch,
        request_states,
        denoise_steps,
        defer_text_cpu_results=defer_text_cpu_results,
        allow_graph=allow_graph,
        require_graph=require_graph,
    )


def _run_packed_forward_impl(
    owner,
    plan: PackedForwardPlan,
    request_states: Any,
    *,
    defer_text_cpu_results: bool = False,
    return_forward_result: bool = False,
    allow_graph: bool = True,
    require_graph: bool = False,
) -> bool | ForwardResult:
    if owner.model is None:
        if require_graph:
            raise capability_mismatch("packed forward graph requires a loaded model")
        return False
    batch = plan.batch
    builder = ForwardStreamBuilder()
    kv_segments: list[ForwardPagedKVSegment] = []
    embed_chunks: list[torch.Tensor] = []
    indicator_chunks: list[tuple[int, bool] | torch.Tensor] = []
    first_pool = owner._forward_target_pool(plan.denoise_steps)
    device = torch.device(str(owner.device))
    current_context: dict[str, Any] | None = None
    ctx = get_forward_context()
    timing = _PackedForwardTiming(device)
    total_start = timing.start()

    try:
        build_stats_start = ctx.component_timer_start()
        build_start = timing.start()
        text_stage_prefix_copies: list[PagedTextCacheSpanCopy] = []
        pending_text_rows: list[_PendingTextBuildRow] = []
        text_kv_promotion_capacity: int | None = None
        decode_boundary_flushed = False

        def flush_text_rows(*, pad_decode_rows: bool = False) -> int | None:
            if not pending_text_rows:
                return None
            total_q = sum(int(row.q_len) for row in pending_text_rows)
            with profile_range("uniserve.packed_forward.text_embed"):
                input_ids = torch.cat(
                    [row.input_ids.reshape(-1) for row in pending_text_rows], dim=0
                )
                if int(input_ids.numel()) != int(total_q):
                    raise invalid_descriptor(
                        "packed forward text input id count does not match row lengths"
                    )
                text_embeds = owner.packed_text_embeddings(input_ids).reshape(total_q, -1)
            segment_base = _append_packed_chunk(
                embed_chunks,
                indicator_chunks,
                text_embeds,
                image_tokens=False,
                indicators=None,
                device=device,
            )
            offset = 0
            for row in pending_text_rows:
                plan.add_text_slot(
                    row_index=row.row_index,
                    segment_start=segment_base + offset,
                    q_len=row.q_len,
                    persistent_cache=row.persistent_cache,
                    staged_cache=row.staged_cache,
                    base_len=row.base_len,
                    last_input_token=row.last_input_token,
                )
                offset += int(row.q_len)
            capacity = None
            if pad_decode_rows:
                first_row = pending_text_rows[0]
                capacity = _append_decode_graph_padding(
                    builder=builder,
                    kv_segments=kv_segments,
                    embed_chunks=embed_chunks,
                    indicator_chunks=indicator_chunks,
                    padding_pool=first_row.staged_cache.pool,
                    decode_rows=len(pending_text_rows),
                    hidden_size=int(text_embeds.shape[1]),
                    dtype=text_embeds.dtype,
                    device=device,
                )
            pending_text_rows.clear()
            return capacity

        with profile_range("uniserve.packed_forward.build"):
            for row_index in _packed_row_order(batch):
                op = batch.ops[row_index]
                mode = batch.op_modes[row_index]
                if require_graph and mode is not ForwardMode.DECODE and not decode_boundary_flushed:
                    text_kv_promotion_capacity = flush_text_rows(pad_decode_rows=True)
                    decode_boundary_flushed = True
                req_id = int(op["req_id"])
                current_context = {
                    "row_index": row_index,
                    "req_id": req_id,
                    "kind": op.get("kind"),
                    "mode": mode.value,
                    "pos_range": op.get("pos_range"),
                    "new_block_ids_len": len(op.get("new_block_ids") or []),
                }
                if mode in {ForwardMode.EXTEND, ForwardMode.DECODE}:
                    state = owner.interleaved_image_state(req_id)
                    cache = state.cond
                    owner._extend_cache_blocks(cache, dict(op))
                    owner._ensure_host_cache(cache)
                    if cache.past is None:
                        if require_graph:
                            raise capability_mismatch(
                                "packed forward graph requires a paged text cache",
                                details=current_context,
                            )
                        return False
                    current_context.update(
                        {
                            "cache_blocks_before_sync": len(getattr(cache, "block_ids", []) or []),
                            "past_blocks_before_sync": len(
                                getattr(cache.past, "block_ids", []) or []
                            ),
                            "past_length_before_sync": int(getattr(cache.past, "length", 0)),
                            "cache_t_index_before_sync": int(getattr(cache, "t_index", -1)),
                        }
                    )
                    _sync_host_cache_blocks(owner, cache, request_states, req_id, dict(op))
                    current_context.update(
                        {
                            "cache_blocks_after_sync": len(getattr(cache, "block_ids", []) or []),
                            "past_blocks_after_sync": len(
                                getattr(cache.past, "block_ids", []) or []
                            ),
                            "past_length_after_sync": int(getattr(cache.past, "length", 0)),
                            "cache_t_index_after_sync": int(getattr(cache, "t_index", -1)),
                        }
                    )
                    hydrate_cached_prefix_from_op(cache, op)
                    current_context.update(
                        {
                            "past_blocks_after_hydrate": len(
                                getattr(cache.past, "block_ids", []) or []
                            ),
                            "past_length_after_hydrate": int(getattr(cache.past, "length", 0)),
                            "cache_t_index_after_hydrate": int(getattr(cache, "t_index", -1)),
                        }
                    )
                    tokens = list(op.get("token_ids") or [])
                    if not tokens:
                        tokens = [int(owner.eos_id or 0)]
                    pos = op.get("pos_range") or [
                        cache.t_index + 1,
                        cache.t_index + 1 + len(tokens),
                    ]
                    start = int(pos[0])
                    q_len = len(tokens)
                    cache.past.ensure_capacity(cache.past.length + q_len)
                    persistent_cache = cache.past
                    staged_cache = persistent_cache
                    if first_pool is not None and persistent_cache.pool is not first_pool:
                        with profile_range("uniserve.packed_forward.text_cache_stage"):
                            staged_cache = owner._stage_text_cache_for_forward(
                                persistent_cache,
                                target_pool=first_pool,
                                end_len=persistent_cache.length + q_len,
                                pending_prefix_copies=text_stage_prefix_copies,
                            )
                    pool = staged_cache.pool
                    if not owner._same_kv_pool(pool, first_pool):
                        if require_graph:
                            raise capability_mismatch(
                                "packed forward graph requires one KV pool",
                                details={
                                    **current_context,
                                    "target_pool": type(first_pool).__name__,
                                    "row_pool": type(pool).__name__,
                                },
                            )
                        return False
                    first_pool = pool if first_pool is None else first_pool
                    ids = _forward_text_input_ids(
                        op,
                        req_id=req_id,
                        tokens=tokens,
                        request_states=request_states,
                        device=device,
                    )
                    last_input_token = int(tokens[-1])
                    if str(op.get("token_source") or "wire") == "last_sampled":
                        relay = getattr(request_states.get(req_id), "decode_relay", None)
                        relay_token = getattr(relay, "token_id", None)
                        if relay_token is not None:
                            last_input_token = int(relay_token)
                    with profile_range("uniserve.packed_forward.text_segment"):
                        owner._add_text_forward_segment(
                            builder=builder,
                            kv_segments=kv_segments,
                            row_index=row_index,
                            req_id=req_id,
                            op=dict(op),
                            mode=mode,
                            cache=SimpleNamespace(past=staged_cache),
                            q_len=q_len,
                            start_pos=start,
                            device=device,
                        )
                    pending_text_rows.append(
                        _PendingTextBuildRow(
                            row_index=row_index,
                            q_len=q_len,
                            input_ids=ids,
                            persistent_cache=persistent_cache,
                            staged_cache=staged_cache,
                            base_len=int(persistent_cache.length),
                            last_input_token=last_input_token,
                        )
                    )
                elif mode is ForwardMode.DENOISE:
                    flush_text_rows()
                    step = plan.denoise_step_for_row(row_index)
                    cfg_plan = text_image_cfg_plan(step)
                    plan.set_denoise_cfg_plan(row_index, cfg_plan)
                    for branch_index, branch in enumerate(cfg_plan.branches):
                        img = step.extra["img"]
                        with profile_range("uniserve.packed_forward.denoise_branch_inputs"):
                            indexes, cache = owner._denoise_branch_inputs(img, branch)
                        if cache is None or getattr(cache, "pool", None) is None:
                            if require_graph:
                                raise capability_mismatch(
                                    "packed forward graph requires a paged denoise cache",
                                    details={**current_context, "branch": str(branch)},
                                )
                            return False
                        owner._wait_gen_cache_ready(cache)
                        pool = cache.pool
                        if not owner._same_kv_pool(pool, first_pool):
                            if require_graph:
                                raise capability_mismatch(
                                    "packed forward graph requires one KV pool",
                                    details={
                                        **current_context,
                                        "branch": str(branch),
                                        "target_pool": type(first_pool).__name__,
                                        "row_pool": type(pool).__name__,
                                    },
                                )
                            return False
                        first_pool = pool if first_pool is None else first_pool
                        q_len = int(step.extra["image_embeds"].shape[1])
                        ensure_capacity = getattr(cache, "ensure_capacity", None)
                        if callable(ensure_capacity):
                            ensure_capacity(int(cache.length) + q_len)
                        if indexes is None or tuple(indexes.shape) != (3, q_len):
                            if require_graph:
                                raise capability_mismatch(
                                    "packed forward graph denoise indexes do not match the token geometry",
                                    details={
                                        **current_context,
                                        "branch": str(branch),
                                        "expected_shape": [3, q_len],
                                        "actual_shape": None
                                        if indexes is None
                                        else list(indexes.shape),
                                    },
                                )
                            return False
                        segment_start = _append_packed_chunk(
                            embed_chunks,
                            indicator_chunks,
                            step.extra["image_embeds"].reshape(q_len, -1),
                            image_tokens=True,
                            indicators=_packed_denoise_indicators(owner, step, q_len),
                            device=device,
                        )
                        with profile_range("uniserve.packed_forward.denoise_segment"):
                            owner._add_denoise_forward_segment(
                                builder=builder,
                                kv_segments=kv_segments,
                                row_index=row_index,
                                req_id=req_id,
                                op=dict(op),
                                cache=cache,
                                indexes=indexes,
                                q_len=q_len,
                                branch_index=branch_index,
                                device=device,
                            )
                        plan.add_denoise_slot(
                            row_index=row_index,
                            step=step,
                            segment_start=segment_start,
                            q_len=q_len,
                            branch=branch,
                        )
                elif mode is ForwardMode.COMMIT:
                    # Publication-only rows contribute no hidden-state segment; the
                    # model hook publishes them after the packed forward updates state.
                    pass
                else:
                    if require_graph:
                        raise capability_mismatch(
                            "packed forward graph received an unsupported mode",
                            details=current_context,
                        )
                    return False
                current_context = None
            if require_graph and not decode_boundary_flushed:
                text_kv_promotion_capacity = flush_text_rows(pad_decode_rows=True)
            else:
                flush_text_rows()
        if text_stage_prefix_copies:
            with profile_range("uniserve.packed_forward.text_prefix_stage"):
                copy_paged_text_cache_spans(
                    text_stage_prefix_copies,
                    num_layers=owner.num_layers,
                    missing_message="cannot stage a packed-forward prefix without a paged source cache",
                )
            for span in text_stage_prefix_copies:
                owner._mark_forward_staging_advanced(
                    span.target,
                    span.source,
                    int(span.start) + int(span.length),
                )
        timing.stop("build_ms", build_start)
        ctx.record_component_elapsed("packed_forward_build", build_stats_start)
        if first_pool is None or not embed_chunks:
            if require_graph:
                raise capability_mismatch(
                    "packed forward graph has no decoder segments",
                    details={
                        "pool_available": first_pool is not None,
                        "embed_chunk_count": len(embed_chunks),
                        "op_modes": [mode.value for mode in batch.op_modes],
                    },
                )
            return False
        stream_stats_start = ctx.component_timer_start()
        stream_start = timing.start()
        with profile_range("uniserve.packed_forward.stream_build"):
            forward_stream = builder.build(device=device)
            route_indices = getattr(owner, "packed_route_indices", None)
            if callable(route_indices):
                und_indices, gen_indices = route_indices(
                    forward_stream,
                    device=device,
                )
                token_count = int(forward_stream.indexes.shape[1])
                und_indices = _validate_route_indices(
                    und_indices,
                    token_count=token_count,
                    device=device,
                    name="text",
                )
                gen_indices = _validate_route_indices(
                    gen_indices,
                    token_count=token_count,
                    device=device,
                    name="generation",
                )
                if int(und_indices.numel()) + int(gen_indices.numel()) != token_count:
                    raise invalid_descriptor(
                        "packed route indices must partition the token stream"
                    )
                forward_stream = replace(
                    forward_stream,
                    und_indices=und_indices,
                    gen_indices=gen_indices,
                )
            kv_view = ForwardPagedKVView(first_pool, kv_segments)
        timing.stop("stream_build_ms", stream_start)
        ctx.record_component_elapsed("packed_forward_stream_build", stream_stats_start)
        text_kv_promotions = [
            PagedTextCacheSpanCopy(
                source=staged_cache,
                target=persistent_cache,
                start=base_len,
                length=q_len,
            )
            for (
                _row_index,
                _start,
                q_len,
                persistent_cache,
                staged_cache,
                base_len,
                _last_input_token,
            ) in plan.text_result_slots
            if staged_cache is not persistent_cache
        ]
        graph_text_kv_promotions = (
            tuple(text_kv_promotions)
            if packed_graph_promotions_supported(text_kv_promotions)
            else ()
        )
        decoder_stats_start = ctx.component_timer_start()
        decoder_start = timing.start()
        decoder_component_start = timing.component_snapshot(ctx)
        with profile_range("uniserve.packed_forward.decoder_input_pack"):
            packed_embeds = torch.cat(embed_chunks, dim=0)
            packed_indicators = _packed_indicator_tensor(
                owner,
                indicator_chunks,
                device=device,
            )
        hidden = None
        if allow_graph:
            graph_kwargs = (
                {"text_kv_promotion_capacity": text_kv_promotion_capacity}
                if text_kv_promotion_capacity is not None
                else {}
            )
            hidden = maybe_run_packed_graph(
                owner,
                packed_embeds,
                route_indicators=packed_indicators,
                forward_stream=forward_stream,
                kv_view=kv_view,
                text_kv_promotions=graph_text_kv_promotions,
                **graph_kwargs,
            )
        graph_promoted_text_kv = hidden is not None and bool(graph_text_kv_promotions)
        if hidden is None:
            if require_graph:
                graph_runner = getattr(owner, "_packed_graph_runner", None)
                raise capability_mismatch(
                    "packed forward CUDA graph did not produce hidden states",
                    details={
                        "graph_miss_reason": getattr(graph_runner, "last_miss_reason", None),
                        "packed_tokens": int(packed_embeds.shape[0]),
                        "segments": len(forward_stream.segments),
                        "block_width": max(
                            (len(segment.block_ids) for segment in kv_view.segments),
                            default=0,
                        ),
                        "promotion_count": len(graph_text_kv_promotions),
                    },
                )
            hidden = owner.packed_decoder_forward(
                packed_embeds,
                route_indicators=packed_indicators,
                indexes=forward_stream.indexes,
                forward_stream=forward_stream,
                kv_view=kv_view,
            )
        timing.add_component_deltas(
            ctx,
            decoder_component_start,
            (
                "packed_decoder_input_norm",
                "packed_decoder_qkv",
                "packed_decoder_attention",
                "packed_decoder_o_proj",
                "packed_decoder_attn_block",
                "packed_decoder_mlp",
            ),
        )
        timing.stop("decoder_ms", decoder_start)
        ctx.record_component_elapsed("packed_forward_decoder", decoder_stats_start)
        text_stats_start = ctx.component_timer_start()
        text_start = timing.start()
        text_logits_by_row: dict[int, torch.Tensor] = {}
        text_outputs_by_row: dict[int, Any] = {}
        text_device_tokens_by_row: dict[int, torch.Tensor] = {}
        text_sample_indices_by_row: dict[int, int] = {}
        deferred_text_sampling: Any | None = None
        text_logits_for_result: torch.Tensor | None = None
        text_postprocess_entries: list[TextPostprocessEntry] = []
        if plan.text_result_slots:
            with profile_range("uniserve.packed_forward.text_last_hidden"):
                text_hidden = torch.stack(
                    [
                        hidden[int(start) + int(q_len) - 1]
                        for _row_index, start, q_len, *_ in plan.text_result_slots
                    ],
                    dim=0,
                )
            with profile_range("uniserve.packed_forward.text_logits"):
                text_logits = owner.packed_text_logits(text_hidden.unsqueeze(0)).squeeze(0)
            with profile_range("uniserve.packed_forward.text_logits_scatter"):
                for offset, (row_index, *_rest) in enumerate(plan.text_result_slots):
                    text_logits_by_row[int(row_index)] = text_logits[offset : offset + 1].unsqueeze(
                        0
                    )
            if return_forward_result:
                text_logits_for_result = text_logits.reshape(len(plan.text_result_slots), -1)
            else:
                sample_logits: list[torch.Tensor] = []
                sampling_params: list[dict[str, Any]] = []
                sampling_generators: list[torch.Generator] = []
                with profile_range("uniserve.packed_forward.text_sampling_inputs"):
                    for row_index, *_rest in plan.text_result_slots:
                        req_id = int(batch.ops[row_index]["req_id"])
                        state = request_states.get(req_id)
                        sampling_params.append(dict(state.sampling or {}))
                        sampling_generators.append(
                            state.device_rng(
                                text_logits.device,
                                stream="text_sampling",
                            )
                        )
                        sample_logits.append(
                            text_logits_by_row[int(row_index)].reshape(-1, text_logits.shape[-1])[
                                -1
                            ]
                        )
                can_defer_text_cpu = bool(defer_text_cpu_results)
                with profile_range("uniserve.packed_forward.text_sampling"):
                    sampled = apply_sampling_batched_with_device_tokens(
                        torch.stack(sample_logits, dim=0),
                        sampling_params,
                        [[] for _ in sample_logits],
                        [None for _ in sample_logits],
                        [None for _ in sample_logits],
                        generators=sampling_generators,
                        defer_cpu=can_defer_text_cpu,
                    )
                if is_deferred_sampling_result(sampled) and can_defer_text_cpu:
                    deferred_text_sampling = sampled
                    for sample_index, (row_index, *_rest) in enumerate(plan.text_result_slots):
                        text_sample_indices_by_row[int(row_index)] = sample_index
                        text_device_tokens_by_row[int(row_index)] = sampled.device_tokens[
                            sample_index : sample_index + 1
                        ]
                else:
                    immediate_sampling = finalize_sampling_result(sampled)
                    for sample_index, (row_index, *_rest) in enumerate(plan.text_result_slots):
                        req_id = int(batch.ops[row_index]["req_id"])
                        sample = immediate_sampling.samples[sample_index]
                        top_logprobs = (
                            [
                                (int(item[0]), float(item[1]), int(item[2]))
                                for item in sample.top_logprobs
                            ]
                            if sample.top_logprobs is not None
                            else None
                        )
                        text_outputs_by_row[int(row_index)] = TextTokenOutput(
                            req_id=req_id,
                            sampled_token_id=int(sample.token_id),
                            sampled_logprob=sample.logprob,
                            top_logprobs=top_logprobs,
                        )
                        text_device_tokens_by_row[int(row_index)] = sampled.device_tokens[
                            sample_index : sample_index + 1
                        ]
        with profile_range("uniserve.packed_forward.burst_position_stage"):
            burst_position_tensors_by_row = _forward_burst_position_tensors(
                owner,
                plan,
                device=device,
            )
        if text_kv_promotions and not graph_promoted_text_kv and not return_forward_result:
            with profile_range("uniserve.packed_forward.text_kv_promote"):
                copy_paged_text_cache_spans(
                    text_kv_promotions,
                    num_layers=owner.num_layers,
                    missing_message="forward text K/V span is missing from staged cache",
                )
        with profile_range("uniserve.packed_forward.text_result_publish"):
            for logits_index, (
                row_index,
                start,
                q_len,
                persistent_cache,
                staged_cache,
                base_len,
                last_input_token,
            ) in enumerate(plan.text_result_slots):
                op = batch.ops[row_index]
                req_id = int(op["req_id"])
                logits = text_logits_by_row[int(row_index)]
                state = owner.interleaved_image_state(req_id)
                position_id = int((op.get("pos_range") or [0, state.cond.t_index + q_len])[1])
                new_len = int(base_len) + int(q_len)
                if return_forward_result:
                    promotion = (
                        PagedTextCacheSpanCopy(
                            source=staged_cache,
                            target=persistent_cache,
                            start=base_len,
                            length=q_len,
                        )
                        if staged_cache is not persistent_cache and not graph_promoted_text_kv
                        else None
                    )
                    text_postprocess_entries.append(
                        TextPostprocessEntry(
                            row_index=int(row_index),
                            req_id=req_id,
                            logits_index=int(logits_index),
                            position_id=position_id,
                            kv_new_length=new_len,
                            last_input_token=int(last_input_token),
                            interleaved_state=state,
                            persistent_cache=persistent_cache,
                            staged_cache=staged_cache,
                            kv_promotion=promotion,
                            num_layers=int(owner.num_layers),
                            mark_staging_advanced=owner._mark_forward_staging_advanced,
                        )
                    )
                    continue
                state.cond.t_index = position_id - 1
                state.cond.last_logits = logits
                state.cond.last_token_id = int(last_input_token)
                persistent_cache.length = new_len
                if staged_cache is not persistent_cache:
                    owner._mark_forward_staging_advanced(staged_cache, persistent_cache, new_len)
                req_state = request_states.get(req_id)
                output = text_outputs_by_row.get(int(row_index))
                if output is None:
                    if deferred_text_sampling is None:
                        raise invalid_descriptor("packed forward text sampling result is missing")
                    sample_index = text_sample_indices_by_row[int(row_index)]
                    _store_forward_sampled_token_relay(
                        req_state,
                        token_id=None,
                        device=device,
                        position_id=position_id,
                        token_tensor=text_device_tokens_by_row[int(row_index)],
                        position_tensor=burst_position_tensors_by_row.get(int(row_index)),
                    )
                    output = DeferredTextSeqResult(
                        req_id=req_id,
                        row=sample_index,
                        state=req_state,
                        sampling_result=deferred_text_sampling,
                        relay_token_tensor=req_state.decode_relay.token_tensor,
                    )
                    text_outputs_by_row[int(row_index)] = output
                else:
                    _store_forward_sampled_token_relay(
                        req_state,
                        token_id=int(output.sampled_token_id),
                        device=device,
                        position_id=position_id,
                        token_tensor=text_device_tokens_by_row.get(int(row_index)),
                        position_tensor=burst_position_tensors_by_row.get(int(row_index)),
                    )
                plan.set_text_result(row_index, output)
        timing.stop("text_post_ms", text_start)
        ctx.record_component_elapsed("packed_forward_text_post", text_stats_start)
        velocity_stats_start = ctx.component_timer_start()
        velocity_start = timing.start()
        branch_velocities: dict[int, dict[str, torch.Tensor]] = {}
        denoise_velocities: dict[DenoiseBranchKey, torch.Tensor] = {}
        denoise_branch_counts: dict[int, int] = {}
        with profile_range("uniserve.packed_forward.velocity"):
            for row_index, step, start, q_len, branch in plan.denoise_result_slots:
                img = step.extra["img"]
                with profile_range("uniserve.packed_forward.velocity_branch"):
                    velocity = owner.packed_hidden_to_velocity(
                        hidden[start : start + q_len].unsqueeze(0),
                        step.t,
                        step.latent,
                        image_token_num=img.token_h * img.token_w,
                        image_size=(img.width, img.height),
                    )
                branch_velocities.setdefault(row_index, {})[branch] = velocity
                branch_id = denoise_branch_counts.get(int(row_index), 0)
                denoise_velocities[DenoiseBranchKey(int(row_index), branch_id)] = velocity
                denoise_branch_counts[int(row_index)] = branch_id + 1
        timing.stop("velocity_ms", velocity_start)
        ctx.record_component_elapsed("packed_forward_velocity", velocity_stats_start)
        if return_forward_result:
            denoise_updates: dict[int, DenoisePostprocessEntry] = {}
            for result_index, step in plan.denoise_steps:
                cfg_plan = plan.denoise_cfg_plan_for_row(result_index)

                def combine_velocity(
                    values: Mapping[Any, torch.Tensor],
                    current_plan: CfgPlan = cfg_plan,
                ) -> torch.Tensor:
                    return current_plan.combine(values)

                def accept_update(
                    updated: torch.Tensor,
                    current_owner: Any = owner,
                    current_step: TextImageDenoiseStep = step,
                ) -> None:
                    current_owner.accept_denoise_update(current_step, updated)

                denoise_updates[int(result_index)] = DenoisePostprocessEntry(
                    row_index=int(result_index),
                    req_id=int(step.req_id),
                    step_index=int(step.step_index),
                    total_steps=int(step.total_steps),
                    branch_names=tuple(cfg_plan.branches),
                    latent=step.latent,
                    t=step.t,
                    t_next=step.t_next,
                    combine_velocity=combine_velocity,
                    accept_update=accept_update,
                )
            timing.stop("total_ms", total_start)
            timing.log(
                batch=batch,
                forward_stream=forward_stream,
                embed_chunks=embed_chunks,
                text_result_slots=plan.text_result_slots,
                denoise_result_slots=plan.denoise_result_slots,
                kv_segments=kv_segments,
            )
            return ForwardResult(
                text_logits=text_logits_for_result,
                text_postprocess=tuple(text_postprocess_entries),
                denoise_velocities=denoise_velocities,
                denoise_updates=denoise_updates,
            )
        update_stats_start = ctx.component_timer_start()
        update_start = timing.start()
        with profile_range("uniserve.packed_forward.denoise_update"):
            for result_index, step in plan.denoise_steps:
                velocities = branch_velocities.get(result_index)
                if not velocities:
                    if require_graph:
                        raise capability_mismatch(
                            "packed forward graph produced no denoise velocities",
                            details={"row_index": int(result_index)},
                        )
                    return False
                velocity = plan.denoise_cfg_plan_for_row(result_index).combine(velocities)
                updated = euler_step(step.latent, velocity, step.t, step.t_next)
                owner.accept_denoise_update(step, updated)
                plan.set_denoise_result(result_index, step)
        timing.stop("denoise_update_ms", update_start)
        ctx.record_component_elapsed("packed_forward_denoise_update", update_stats_start)
        timing.stop("total_ms", total_start)
        timing.log(
            batch=batch,
            forward_stream=forward_stream,
            embed_chunks=embed_chunks,
            text_result_slots=plan.text_result_slots,
            denoise_result_slots=plan.denoise_result_slots,
            kv_segments=kv_segments,
        )
        return True
    except Exception as exc:
        if current_context:
            logger.exception(
                "packed forward failed for an admitted segment group: context=%s",
                current_context,
            )
        else:
            logger.exception("packed forward failed for an admitted segment group")
        raise capability_mismatch(
            "packed forward failed for an admitted segment group",
            details={"cause_type": type(exc).__name__, "cause": str(exc)[:500]},
        ) from exc


class _PackedForwardTiming:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.enabled = _PACKED_FORWARD_TIMING
        self.sync = _PACKED_FORWARD_TIMING_SYNC
        self.values: dict[str, float] = {}

    def start(self) -> int:
        if not self.enabled:
            return 0
        self._sync()
        return time.perf_counter_ns()

    def stop(self, key: str, start_ns: int) -> None:
        if not self.enabled or not start_ns:
            return
        self._sync()
        self.values[key] = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    def component_snapshot(self, ctx: Any) -> dict[str, int]:
        if not self.enabled:
            return {}
        stats = getattr(ctx, "stats", None)
        component_ns = getattr(stats, "component_ns", None)
        return dict(component_ns) if isinstance(component_ns, dict) else {}

    def add_component_deltas(
        self,
        ctx: Any,
        start: Mapping[str, int],
        keys: Sequence[str],
    ) -> None:
        if not self.enabled:
            return
        stats = getattr(ctx, "stats", None)
        component_ns = getattr(stats, "component_ns", None)
        if not isinstance(component_ns, dict):
            return
        for key in keys:
            before = int(start.get(key, 0))
            after = int(component_ns.get(key, 0))
            if after > before:
                self.values[f"{key}_ms"] = (after - before) / 1_000_000.0

    def log(
        self,
        *,
        batch: UniForwardBatch,
        forward_stream: Any,
        embed_chunks: Sequence[torch.Tensor],
        text_result_slots: Sequence[tuple[Any, ...]],
        denoise_result_slots: Sequence[tuple[Any, ...]],
        kv_segments: Sequence[Any],
    ) -> None:
        if not self.enabled:
            return
        mode_counts: dict[str, int] = {}
        for mode in batch.op_modes:
            key = mode.value
            mode_counts[key] = mode_counts.get(key, 0) + 1
        text_tokens = sum(int(slot[2]) for slot in text_result_slots)
        image_tokens = sum(int(slot[3]) for slot in denoise_result_slots)
        payload = {
            "rows": len(batch.ops),
            "mode_counts": mode_counts,
            "tokens": sum(int(chunk.shape[0]) for chunk in embed_chunks),
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "text_rows": len(text_result_slots),
            "denoise_rows": len({int(slot[0]) for slot in denoise_result_slots}),
            "denoise_segments": len(denoise_result_slots),
            "kv_segments": len(kv_segments),
            "fully_visible": bool(getattr(forward_stream, "fully_visible", False)),
            **self.values,
        }
        logger.info("packed_forward_timing %s", json.dumps(payload, sort_keys=True))

    def _sync(self) -> None:
        if self.sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _sync_host_cache_blocks(
    owner: Any,
    cache: Any,
    request_states: Any,
    req_id: int,
    op: Mapping[str, Any] | None = None,
) -> None:
    past = getattr(cache, "past", None)
    if past is None or getattr(past, "pool", None) is not getattr(owner, "kv_pool", None):
        return
    get_state = getattr(request_states, "get", None)
    if not callable(get_state):
        return
    state = get_state(int(req_id))
    if op is not None:
        block_ids = state_block_ids_for_op(op, state)
    else:
        block_ids = [int(block_id) for block_id in (getattr(state, "block_ids", []) or [])]
    if not block_ids:
        return
    current = [int(block_id) for block_id in (getattr(cache, "block_ids", []) or [])]
    if len(block_ids) < len(current):
        return
    cache.block_ids = block_ids
    past.set_blocks(cache.block_ids)


def _append_packed_chunk(
    embed_chunks: list[torch.Tensor],
    indicator_chunks: list[tuple[int, bool] | torch.Tensor],
    embeds: torch.Tensor,
    *,
    image_tokens: bool,
    indicators: torch.Tensor | None,
    device: torch.device,
) -> int:
    del device
    start = sum(chunk.shape[0] for chunk in embed_chunks)
    q_len = int(embeds.shape[0])
    embed_chunks.append(embeds)
    if indicators is None:
        indicator_chunks.append((q_len, bool(image_tokens)))
    else:
        flat = indicators.reshape(-1)
        if int(flat.numel()) != q_len:
            raise invalid_descriptor("packed forward route mask does not match chunk length")
        indicator_chunks.append(flat)
    return start


def _packed_denoise_indicators(
    owner: Any,
    step: TextImageDenoiseStep,
    q_len: int,
) -> torch.Tensor | None:
    build = getattr(owner, "packed_denoise_indicators", None)
    if not callable(build):
        return None
    indicators = build(step, int(q_len))
    if indicators is not None and not isinstance(indicators, torch.Tensor):
        raise invalid_descriptor("packed route indicators must be a tensor")
    return indicators


def _packed_indicator_tensor(
    owner: Any,
    chunks: Sequence[tuple[int, bool] | torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    total = sum(
        int(chunk.numel()) if isinstance(chunk, torch.Tensor) else int(chunk[0])
        for chunk in chunks
    )
    if total <= 0:
        raise invalid_descriptor("packed forward indicators must not be empty")
    if device.type != "cuda" or any(isinstance(chunk, torch.Tensor) for chunk in chunks):
        tensors: list[torch.Tensor] = []
        for chunk in chunks:
            if isinstance(chunk, torch.Tensor):
                if int(chunk.numel()) > 0:
                    tensors.append(chunk.reshape(-1).to(device=device, dtype=torch.bool))
                continue
            length, value = chunk
            if int(length) > 0:
                tensors.append(
                    torch.full(
                        (int(length),),
                        bool(value),
                        dtype=torch.bool,
                        device=device,
                    )
                )
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=0)
    stager = getattr(owner, "_packed_forward_indicator_stager", None)
    if not isinstance(stager, TextTensorStager):
        stager = TextTensorStager(ring_depth=3)
        setattr(owner, "_packed_forward_indicator_stager", stager)
    slot = stager.next_slot()
    cpu = slot.bool_buffer("packed_forward_indicators", total, pin=True)
    offset = 0
    for chunk in chunks:
        if isinstance(chunk, torch.Tensor):
            raise invalid_descriptor("CUDA tensor route chunks must use tensor packing")
        length, value = chunk
        length = int(length)
        if length <= 0:
            continue
        cpu[offset : offset + length].fill_(bool(value))
        offset += length
    indicators = slot.device_buffer(
        "packed_forward_indicators",
        total,
        dtype=torch.bool,
        device=device,
    )
    indicators.copy_(cpu, non_blocking=is_pinned(cpu))
    return indicators


def _validate_route_indices(
    indices: torch.Tensor,
    *,
    token_count: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if not isinstance(indices, torch.Tensor):
        raise invalid_descriptor(f"packed {name} route indices must be a tensor")
    resolved = indices.reshape(-1).to(device=device, dtype=torch.long)
    if int(resolved.numel()) > int(token_count):
        raise invalid_descriptor(f"packed {name} route indices exceed the token count")
    return resolved


def _packed_row_order(batch: UniForwardBatch) -> list[int]:
    if not _PACKED_FORWARD_CANONICAL_ORDER:
        return list(range(len(batch.ops)))
    decode_rows: list[int] = []
    extend_rows: list[int] = []
    denoise_rows: list[int] = []
    commit_rows: list[int] = []
    other_rows: list[int] = []
    for row_index, mode in enumerate(batch.op_modes):
        if mode is ForwardMode.DECODE:
            decode_rows.append(row_index)
        elif mode is ForwardMode.EXTEND:
            extend_rows.append(row_index)
        elif mode is ForwardMode.DENOISE:
            denoise_rows.append(row_index)
        elif mode is ForwardMode.COMMIT:
            commit_rows.append(row_index)
        else:
            other_rows.append(row_index)
    return decode_rows + extend_rows + denoise_rows + commit_rows + other_rows


def _store_forward_sampled_token_relay(
    state: Any,
    *,
    token_id: int | None,
    device: torch.device,
    position_id: int | None = None,
    token_tensor: torch.Tensor | None = None,
    position_tensor: torch.Tensor | None = None,
) -> None:
    relay = getattr(state, "decode_relay", None)
    if relay is None:
        return
    if token_tensor is None:
        if token_id is None:
            raise invalid_descriptor("packed forward relay publish requires a token tensor")
        token_tensor = torch.tensor([int(token_id)], dtype=torch.long, device=device)
    else:
        token_tensor = token_tensor.reshape(1).to(device=device, dtype=torch.long)
    _DECODE_RELAY.publish_sample(
        state,
        token_id=None if token_id is None else int(token_id),
        token_tensor=token_tensor,
    )
    if position_id is not None:
        if position_tensor is None:
            position_tensor = torch.tensor([int(position_id)], dtype=torch.long, device=device)
        else:
            position_tensor = position_tensor.reshape(1).to(device=device, dtype=torch.long)
        _DECODE_RELAY.publish_position(
            state,
            position_id=int(position_id),
            position_tensor=position_tensor,
        )


def _forward_burst_position_tensors(
    owner: Any,
    plan: PackedForwardPlan,
    *,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    rows: list[int] = []
    position_ids: list[int] = []
    for (
        row_index,
        _start,
        q_len,
        _persistent,
        _staged,
        base_len,
        _last_token,
    ) in plan.text_result_slots:
        op = plan.batch.ops[int(row_index)]
        if plan.batch.op_modes[int(row_index)] is not ForwardMode.DECODE:
            continue
        if int(op.get("decode_token_count") or 1) <= 1:
            continue
        pos = op.get("pos_range") or [int(base_len), int(base_len) + int(q_len)]
        if not isinstance(pos, Sequence) or len(pos) != 2:
            raise invalid_descriptor("decode burst op.pos_range must be [start, end]")
        rows.append(int(row_index))
        position_ids.append(int(pos[1]))
    if not rows:
        return {}
    if device.type != "cuda":
        positions = torch.tensor(position_ids, dtype=torch.long, device=device)
    else:
        stager = getattr(owner, "_packed_forward_burst_position_stager", None)
        if not isinstance(stager, TextTensorStager):
            stager = TextTensorStager(ring_depth=3)
            setattr(owner, "_packed_forward_burst_position_stager", stager)
        slot = stager.next_slot()
        cpu = slot.long_buffer("packed_forward_burst_positions", len(position_ids), pin=True)
        fill_cpu_ints(cpu, position_ids)
        positions = slot.device_buffer(
            "packed_forward_burst_positions",
            len(position_ids),
            dtype=torch.long,
            device=device,
        )
        positions.copy_(cpu, non_blocking=is_pinned(cpu))
    return {row: positions[offset : offset + 1] for offset, row in enumerate(rows)}


def _forward_text_input_ids(
    op: Mapping[str, Any],
    *,
    req_id: int,
    tokens: Sequence[int],
    request_states: Any,
    device: torch.device,
) -> torch.Tensor:
    source = str(op.get("token_source") or "wire")
    if source not in {"wire", "last_sampled"}:
        raise invalid_descriptor(f"unsupported text token_source {source!r}")
    if source == "wire":
        return torch.tensor(list(tokens), dtype=torch.long, device=device)
    if len(tokens) != 1:
        raise invalid_descriptor(
            "decode op requested token_source='last_sampled' but does not have exactly one token"
        )
    state = request_states.get(int(req_id))
    token = _DECODE_RELAY.consume_token(
        state,
        expected_token_id=None,
        device=device,
        token_source=source,
        require=True,
    )
    assert token is not None
    return token.reshape(1)


# ---------------------
# Packed batch adapter
# ---------------------

_DECODE_RELAY = TextDecodeRelay()
_RELAY_PLACEHOLDER_TOKEN_ID = -1


class PackedForwardExecutor:
    """Execute one admitted segment group without mode splitting."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def execute(
        self,
        batch: UniForwardBatch,
        *,
        request_states: Any,
        defer_text_cpu_results: bool = False,
    ) -> Any:
        results: list[Any] = [None] * len(batch.ops)
        denoise_steps: list[tuple[int, TextImageDenoiseStep]] = []
        commit_rows: list[tuple[int, int, dict[str, Any]]] = []
        has_burst_rows = False
        with profile_range("uniserve.packed_forward.prepare_ops"):
            for row_index, op in enumerate(batch.ops):
                mode = batch.op_modes[row_index]
                req_id = int(op["req_id"])
                if (
                    int(op.get("decode_token_count") or 1) > 1
                    or int(op.get("denoise_step_count") or 1) > 1
                ):
                    has_burst_rows = True
                if mode is ForwardMode.DENOISE:
                    step = self.owner.prepare_denoise(
                        request_states.get(req_id),
                        dict(op),
                    )
                    denoise_steps.append((row_index, step))
                elif mode is ForwardMode.COMMIT:
                    commit_rows.append((row_index, req_id, dict(op)))
                elif mode not in {ForwardMode.EXTEND, ForwardMode.DECODE}:
                    raise capability_mismatch(
                        "packed forward group contains an unsupported mode",
                        details={"mode": mode.value},
                    )
        for _row_index, step in denoise_steps:
            extra = getattr(step, "extra", None)
            image = extra.get("img") if isinstance(extra, dict) else None
            residual_state = getattr(image, "residual_cache", None)
            if residual_state is not None:
                residual_state.invalidate()
        if not commit_rows and not has_burst_rows:
            result = run_packed_forward_result(
                self.owner,
                batch,
                request_states,
                denoise_steps,
                defer_text_cpu_results=defer_text_cpu_results,
                allow_graph=True,
                require_graph=True,
            )
            if result is not None:
                return result
        if not run_packed_forward(
            self.owner,
            batch,
            request_states,
            denoise_steps,
            results,
            defer_text_cpu_results=defer_text_cpu_results,
            require_graph=True,
        ):
            raise capability_mismatch("admitted segment group did not execute as one packed graph")
        self._complete_decode_bursts(
            batch,
            request_states,
            results,
            defer_final_cpu_results=defer_text_cpu_results,
        )
        self._complete_denoise_bursts(batch, request_states, results)
        for row_index, req_id, op in commit_rows:
            state = request_states.get(req_id)
            decoded = self.owner.decode_image(
                getattr(state, "latent", None),
                req_id=req_id,
                state=state,
                op=op,
            )
            out = dict(decoded)
            logits = out.pop("logits", None)
            if logits is not None:
                sampled = sample_logits_result(
                    req_id=req_id,
                    state=state,
                    logits=logits,
                    op=op,
                )
                sampled.pop("req_id", None)
                out.update(sampled)
            results[row_index] = out
        return results

    def _complete_decode_bursts(
        self,
        batch: UniForwardBatch,
        request_states: Any,
        results: list[Any],
        *,
        defer_final_cpu_results: bool = False,
    ) -> None:
        active: list[dict[str, Any]] = []
        for row_index, op in enumerate(batch.ops):
            if batch.op_modes[row_index] is not ForwardMode.DECODE:
                continue
            requested = int(op.get("decode_token_count") or 1)
            if requested <= 1:
                continue
            stop_ids = {int(token) for token in (op.get("decode_stop_token_ids") or [])}
            terminal_stop = not stop_ids or op.get("decode_stop_terminal") is True
            position = op.get("pos_range") or [0, 0]
            if not isinstance(position, Sequence) or len(position) != 2:
                raise invalid_descriptor("decode burst op.pos_range must be [start, end]")
            active.append(
                {
                    "row_index": row_index,
                    "op": dict(op),
                    "tokens": [],
                    "requested": requested,
                    "launched": 1,
                    "last_op": dict(op),
                    "pending": results[row_index],
                    "pending_tokens": [results[row_index]] if terminal_stop else [],
                    "stop_ids": stop_ids,
                    "terminal_stop": terminal_stop,
                    "done": False,
                }
            )
        while any(not bool(item["done"]) for item in active):
            iter_ops: list[dict[str, Any]] = []
            iter_items: list[dict[str, Any]] = []
            for item in active:
                if bool(item["done"]) or int(item["launched"]) >= int(item["requested"]):
                    continue
                op = dict(item["op"])
                op["new_block_ids"] = []
                state = request_states.get(int(op["req_id"]))
                if _attach_decode_relay_input(op, state):
                    op["token_ids"] = [_RELAY_PLACEHOLDER_TOKEN_ID]
                    op["token_source"] = "last_sampled"
                else:
                    pending = item.get("pending")
                    if pending is None:
                        raise invalid_descriptor("decode burst relay input is missing")
                    token = _sampled_token_id(
                        pending,
                        profile_name="uniserve.packed_burst.relay_input_materialize",
                    )
                    item["tokens"].append(token)
                    item["pending"] = None
                    if token in item["stop_ids"]:
                        item["done"] = True
                        continue
                    op["token_ids"] = [token]
                    op.pop("token_source", None)
                    op.pop("token_tensor", None)
                next_pos = _decode_op_next_pos(item["last_op"])
                op["pos_range"] = [next_pos, next_pos + 1]
                op["decode_token_count"] = None
                op["decode_stop_token_ids"] = []
                item["last_op"] = op
                iter_ops.append(op)
                iter_items.append(item)
            if not iter_ops:
                break
            followup_results = self._run_decode_burst_graph_followup(
                iter_ops,
                request_states,
                defer_cpu_results=True,
            )
            if followup_results is None:
                raise capability_mismatch(
                    "packed decode burst follow-up requires CUDA graph coverage"
                )
            for item, output in zip(iter_items, followup_results, strict=True):
                previous = item.get("pending")
                item["pending"] = output
                item["launched"] = int(item["launched"]) + 1
                if bool(item.get("terminal_stop")):
                    item["pending_tokens"].append(output)
                    continue
                if previous is None:
                    continue
                token = _sampled_token_id(
                    previous,
                    profile_name="uniserve.packed_burst.stop_check",
                )
                item["tokens"].append(token)
                if token in item["stop_ids"]:
                    item["pending"] = None
                    item["done"] = True
        for item in active:
            if bool(item.get("terminal_stop")):
                results[int(item["row_index"])] = _decode_burst_terminal_result(
                    item["op"],
                    item["pending_tokens"],
                    item["stop_ids"],
                    defer_cpu=bool(defer_final_cpu_results),
                )
                item["pending"] = None
                continue
            pending = item.get("pending")
            if pending is not None:
                results[int(item["row_index"])] = _decode_burst_result_with_pending(
                    item["op"],
                    item["tokens"],
                    pending,
                    defer_cpu=bool(defer_final_cpu_results),
                )
                item["pending"] = None
                continue
            results[int(item["row_index"])] = _decode_burst_result(
                item["op"],
                item["tokens"],
            )

    def _run_decode_burst_graph_followup(
        self,
        ops: Sequence[Mapping[str, Any]],
        request_states: Any,
        *,
        defer_cpu_results: bool,
    ) -> list[Any] | None:
        driver = self.owner._text_driver()
        run_graph = getattr(driver, "try_run_decode_graph_logits_batch", None)
        if not callable(run_graph):
            return None
        logits_rows = run_graph(ops)
        if logits_rows is None:
            return None
        if len(logits_rows) != len(ops):
            raise invalid_descriptor("decode graph follow-up logits row count must match ops")
        rows = [_coerce_logits_row(logits) for logits in logits_rows]
        if not rows:
            return []
        logits_batch = torch.stack(rows, dim=0)
        sampling_params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        generators: list[torch.Generator] = []
        for op in ops:
            state = request_states.get(int(op["req_id"]))
            sampling_params.append(dict(state.sampling or {}))
            recent.append(op.get("recent_tokens") or [])
            allowed.append(op.get("allowed_tokens"))
            suppress.append(op.get("suppress_tokens"))
            generators.append(state.device_rng(logits_batch.device, stream="text_sampling"))
        sampled = apply_sampling_batched_with_device_tokens(
            logits_batch,
            sampling_params,
            recent,
            allowed,
            suppress,
            generators=generators,
            defer_cpu=defer_cpu_results,
        )
        device = logits_batch.device
        position_ids, position_tensors = _decode_followup_position_tensors(
            self.owner,
            ops,
            device=device,
        )
        if is_deferred_sampling_result(sampled) and defer_cpu_results:
            deferred_outputs: list[Any] = []
            for row, op in enumerate(ops):
                req_id = int(op["req_id"])
                state = request_states.get(req_id)
                _store_decode_followup_relay(
                    state,
                    token_id=None,
                    device=device,
                    position_id=position_ids[row],
                    token_tensor=sampled.device_tokens[row : row + 1],
                    position_tensor=position_tensors[row],
                )
                deferred_outputs.append(
                    DeferredTextSeqResult(
                        req_id=req_id,
                        row=row,
                        state=state,
                        sampling_result=sampled,
                        relay_token_tensor=sampled.device_tokens[row : row + 1],
                    )
                )
            return deferred_outputs
        sampled = finalize_sampling_result(sampled)
        immediate_outputs: list[Any] = []
        for row, op in enumerate(ops):
            req_id = int(op["req_id"])
            sample = sampled.samples[row]
            token_id = int(sample.token_id)
            _store_decode_followup_relay(
                request_states.get(req_id),
                token_id=token_id,
                device=device,
                position_id=position_ids[row],
                token_tensor=sampled.device_tokens[row : row + 1],
                position_tensor=position_tensors[row],
            )
            top_logprobs = (
                [(int(item[0]), float(item[1]), int(item[2])) for item in sample.top_logprobs]
                if sample.top_logprobs is not None
                else None
            )
            immediate_outputs.append(
                {
                    "req_id": req_id,
                    "sampled_token_id": token_id,
                    "sampled_logprob": sample.logprob,
                    "top_logprobs": top_logprobs,
                }
            )
        return immediate_outputs

    def _complete_denoise_bursts(
        self,
        batch: UniForwardBatch,
        request_states: Any,
        results: list[Any],
    ) -> None:
        items: list[tuple[int, Any, dict[str, Any]]] = []
        row_indexes: list[int] = []
        for row_index, op in enumerate(batch.ops):
            if batch.op_modes[row_index] is not ForwardMode.DENOISE:
                continue
            requested = int(op.get("denoise_step_count") or 1)
            if requested <= 1 or _output_bool(results[row_index], "denoise_done"):
                continue
            remaining = requested - 1
            req_id = int(op["req_id"])
            followup = dict(op)
            followup["timestep_idx"] = _output_int(results[row_index], "num_steps_done")
            followup["denoise_step_count"] = remaining
            row_indexes.append(row_index)
            items.append((req_id, request_states.get(req_id), followup))
        if not items:
            return
        outputs = DenoiseDriver().step_many(items, self.owner, graph_mode="require")
        for row_index, output in zip(row_indexes, outputs, strict=True):
            results[row_index] = output


def _output_value(output: Any, field: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(field)
    if hasattr(output, field):
        return getattr(output, field)
    finalize = getattr(output, "finalize", None)
    if callable(finalize):
        finalized = finalize()
        if isinstance(finalized, Mapping):
            return finalized.get(field)
    to_seq_result = getattr(output, "to_seq_result", None)
    if callable(to_seq_result):
        return dict(to_seq_result()).get(field)
    raise invalid_descriptor(f"packed forward output does not expose {field!r}")


def _output_int(output: Any, field: str) -> int:
    value = _output_value(output, field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_descriptor(f"packed forward output field {field!r} must be an integer")
    return int(value)


def _output_bool(output: Any, field: str) -> bool:
    value = _output_value(output, field)
    if not isinstance(value, bool):
        raise invalid_descriptor(f"packed forward output field {field!r} must be a boolean")
    return bool(value)


def _sampled_token_id(output: Any, *, profile_name: str) -> int:
    materialize = getattr(output, "materialize_sampled_token_id", None)
    with profile_range(profile_name):
        token = (
            int(materialize())
            if callable(materialize)
            else _output_int(output, "sampled_token_id")
        )
    if token < 0:
        raise invalid_descriptor("sampled_token_id must be non-negative")
    return token


def _decode_burst_result(op: Mapping[str, Any], tokens: Sequence[int]) -> dict[str, Any]:
    if not tokens:
        raise invalid_descriptor("decode burst did not produce a sampled token")
    token_ids = [int(token) for token in tokens]
    return {
        "req_id": int(op["req_id"]),
        "sampled_token_id": token_ids[-1],
        "sampled_token_ids": token_ids,
    }


def _decode_burst_result_with_pending(
    op: Mapping[str, Any],
    tokens: Sequence[int],
    pending: Any,
    *,
    defer_cpu: bool,
) -> dict[str, Any] | DeferredDecodeBurstSeqResult:
    materialize = getattr(pending, "materialize_sampled_token_id", None)
    ready = getattr(pending, "ready", None)
    if defer_cpu and callable(materialize) and callable(ready):
        return DeferredDecodeBurstSeqResult(
            req_id=int(op["req_id"]),
            prefix_token_ids=tokens,
            pending=pending,
        )
    token = _sampled_token_id(
        pending,
        profile_name="uniserve.packed_burst.finalize_pending",
    )
    return _decode_burst_result(op, [*tokens, token])


def _decode_burst_terminal_result(
    op: Mapping[str, Any],
    pending_tokens: Sequence[Any],
    stop_ids: set[int],
    *,
    defer_cpu: bool,
) -> dict[str, Any] | DeferredTerminalDecodeBurstSeqResult:
    if not pending_tokens:
        raise invalid_descriptor("decode burst did not produce a sampled token")
    if defer_cpu and all(
        callable(getattr(pending, "materialize_sampled_token_id", None))
        for pending in pending_tokens
    ):
        return DeferredTerminalDecodeBurstSeqResult(
            req_id=int(op["req_id"]),
            pending_tokens=pending_tokens,
            stop_token_ids=stop_ids,
        )
    token_ids: list[int] = []
    for pending in pending_tokens:
        token = _sampled_token_id(
            pending,
            profile_name="uniserve.packed_burst.finalize_terminal",
        )
        token_ids.append(token)
        if token in stop_ids:
            break
    return _decode_burst_result(op, token_ids)


def _attach_decode_relay_input(op: dict[str, Any], state: Any) -> bool:
    relay = getattr(state, "decode_relay", None)
    token_tensor = getattr(relay, "token_tensor", None)
    if not isinstance(token_tensor, torch.Tensor) or token_tensor.dtype != torch.long:
        return False
    op["token_tensor"] = token_tensor
    return True


def _decode_op_next_pos(op: Mapping[str, Any]) -> int:
    position = op.get("pos_range") or [0, 0]
    if not isinstance(position, Sequence) or len(position) != 2:
        raise invalid_descriptor("decode burst op.pos_range must be [start, end]")
    return int(position[1])


def _coerce_logits_row(logits: Any) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise invalid_descriptor("decode graph follow-up must return logits tensors")
    if logits.ndim == 0:
        raise invalid_descriptor("decode graph follow-up logits must have a vocabulary dimension")
    return logits.reshape(-1, logits.shape[-1])[-1]


def _store_decode_followup_relay(
    state: Any,
    *,
    token_id: int | None,
    device: torch.device,
    position_id: int,
    token_tensor: torch.Tensor,
    position_tensor: torch.Tensor | None,
) -> None:
    relay = getattr(state, "decode_relay", None)
    if relay is None:
        return
    _DECODE_RELAY.publish_sample(
        state,
        token_id=None if token_id is None else int(token_id),
        token_tensor=token_tensor.reshape(1).to(device=device, dtype=torch.long),
    )
    position_tensor = (
        torch.tensor([int(position_id)], dtype=torch.long, device=device)
        if position_tensor is None
        else position_tensor.reshape(1).to(device=device, dtype=torch.long)
    )
    _DECODE_RELAY.publish_position(
        state,
        position_id=int(position_id),
        position_tensor=position_tensor,
    )


def _decode_followup_position_tensors(
    owner: Any,
    ops: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> tuple[list[int], list[torch.Tensor]]:
    position_ids = [int((op.get("pos_range") or [0, 0])[1]) for op in ops]
    if not position_ids:
        return [], []
    if device.type != "cuda":
        positions = torch.tensor(position_ids, dtype=torch.long, device=device)
    else:
        stager = getattr(owner, "_decode_followup_position_stager", None)
        if not isinstance(stager, TextTensorStager):
            stager = TextTensorStager(ring_depth=3)
            setattr(owner, "_decode_followup_position_stager", stager)
        slot = stager.next_slot()
        cpu = slot.long_buffer("packed_decode_followup_positions", len(position_ids), pin=True)
        fill_cpu_ints(cpu, position_ids)
        positions = slot.device_buffer(
            "packed_decode_followup_positions",
            len(position_ids),
            dtype=torch.long,
            device=device,
        )
        positions.copy_(cpu, non_blocking=is_pinned(cpu))
    return position_ids, [positions[row : row + 1] for row in range(len(position_ids))]
