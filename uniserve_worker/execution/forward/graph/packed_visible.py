"""CUDA graph capture/replay for packed mixed decoder forwards."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import torch

import uniserve_worker.ops as ops

from ....contracts.forward_context import (
    TextAttentionMetadata,
    get_forward_context,
    use_forward_context,
)
from ....contracts.forward_mode import ForwardMode
from ....foundation.errors import invalid_descriptor
from ....runtime.host_staging import fill_cpu_ints, is_pinned
from ....runtime.paged_text_cache import PagedTextCacheSpanCopy
from ....runtime.tensor_staging import TextTensorStager, TextTensorStagingSlot
from ..stream import (
    ForwardGraphPagedKVView,
    ForwardGraphStreamState,
    ForwardPagedKVView,
    ForwardStream,
)
from .base import GraphEvent, _GraphRunnerBase, record_graph_stats

logger = logging.getLogger(__name__)

__all__ = [
    "PackedMixedGraphRunner",
    "maybe_run_packed_mixed_graph",
    "packed_mixed_graph_promotions_supported",
]

_RUNNER_ATTR = "_packed_mixed_graph_runner"
_MAX_FAILURES = 2


class _MixedGraphBackendUnplanned(RuntimeError):
    """Capture completed without planning the graph-scoped attention backend."""


@dataclass
class PackedMixedGraphState:
    key: tuple[Any, ...]
    graph: torch.cuda.CUDAGraph
    packed_embeds: torch.Tensor
    indicators: torch.Tensor
    stream_state: ForwardGraphStreamState
    kv_view: ForwardGraphPagedKVView
    metadata: TextAttentionMetadata
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


class PackedMixedGraphRunner(_GraphRunnerBase):
    """Own the active-topology CUDA graph for packed mixed decoder forwards."""

    def __init__(
        self,
        *,
        name: str = "packed_mixed",
        default_enabled: bool | None = None,
        logger: Any = logger,
    ) -> None:
        self.name = str(name)
        self.default_enabled = True if default_enabled is None else bool(default_enabled)
        self.default_warmup = False
        self.metric_prefix = "packed_mixed_"
        self.logger = logger
        self.states: dict[tuple[Any, ...], PackedMixedGraphState] = {}
        self.disabled: set[tuple[Any, ...]] = set()
        self._capture_pool: Any = None
        self._graph_input_buffer_pool: dict[tuple[str, str, str], torch.Tensor] = {}
        self._failures = 0
        self._hard_disabled = False
        self._backend_ineligible = False
        self._replays = 0
        self.last_miss_reason: str | None = None

    def enabled(self) -> bool:
        return self.default_enabled and not self._hard_disabled and not self._backend_ineligible

    def capture_pool(self) -> Any:
        # Exact mixed geometries are captured lazily. The resident topology owns
        # its activation pool so its static allocations cannot alias another
        # graph executable.
        return None

    def _retire_inactive_geometry(
        self,
        key: tuple[Any, ...],
        *,
        device: torch.device | str,
    ) -> None:
        inactive = tuple(existing for existing in self.states if existing != key)
        self._retire_graph_states(inactive, device=device, reclaim=True)

    def maybe_run(
        self,
        owner: Any,
        packed_embeds: torch.Tensor,
        *,
        image_gen_indicators: torch.Tensor,
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
            self._record(ctx, GraphEvent.MISS, int(packed_embeds.shape[0]))
            if self.logger is not None:
                self.logger.warning(
                    "%s CUDA graph unavailable: no graph-capable paged-varlen attention backend",
                    self.name,
                )
            return None
        promotions = (
            tuple(text_kv_promotions)
            if packed_mixed_graph_promotions_supported(text_kv_promotions)
            else ()
        )
        key = self._graph_key(
            owner,
            packed_embeds,
            image_gen_indicators,
            forward_stream,
            kv_view,
            backend,
            promotions,
            text_kv_promotion_capacity,
        )
        if key is None or key in self.disabled:
            self.last_miss_reason = "shape_ineligible" if key is None else "shape_disabled"
            self._record(ctx, GraphEvent.MISS, int(packed_embeds.shape[0]))
            return None
        self._retire_inactive_geometry(key, device=packed_embeds.device)
        out = self._capture_or_replay(
            key=key,
            ctx=ctx,
            capture=lambda: self._capture(
                owner,
                packed_embeds,
                image_gen_indicators=image_gen_indicators,
                forward_stream=forward_stream,
                kv_view=kv_view,
                text_kv_promotions=promotions,
                text_kv_promotion_capacity=text_kv_promotion_capacity,
                key=key,
                ctx=ctx,
                backend=backend,
            ),
            copy_inputs=lambda state: self._copy_inputs(
                state,
                packed_embeds,
                image_gen_indicators=image_gen_indicators,
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
        self._replays += 1
        if self._replays == 1 and self.logger is not None:
            self.logger.info(
                "%s CUDA graph active: captured packed mixed decoder (tokens=%d, rows=%d)",
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
        image_gen_indicators: torch.Tensor,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
        key: tuple[Any, ...],
        ctx: Any,
        backend: Any,
        text_kv_promotions: tuple[PagedTextCacheSpanCopy, ...] = (),
        text_kv_promotion_capacity: int | None = None,
    ) -> PackedMixedGraphState:
        first_attn = _first_attention(owner)
        block_width_capacity = _graph_block_width_capacity(kv_view)
        graph_kv_view = ForwardGraphPagedKVView(
            kv_view.pool,
            kv_view.segments,
            block_width_capacity=block_width_capacity,
        )
        max_context_len = graph_kv_view.max_seqlen_k()
        metadata = TextAttentionMetadata(
            cache=graph_kv_view,
            block_table=graph_kv_view.block_table(device=packed_embeds.device),
            cache_seqlens=graph_kv_view.cache_seqlens_after(device=packed_embeds.device),
            cu_seqlens_q=forward_stream.cu_seqlens_q.detach().clone(),
            cu_seqlens_k=graph_kv_view.cu_seqlens_after(device=packed_embeds.device),
            max_seqlen_q=int(forward_stream.visible_end.shape[1]),
            max_seqlen_k=max_context_len,
            max_context_len=max_context_len,
            mode=ForwardMode.MIXED,
        )
        state = PackedMixedGraphState(
            key=key,
            graph=torch.cuda.CUDAGraph(),
            packed_embeds=packed_embeds.detach().clone(),
            indicators=image_gen_indicators.detach().clone(),
            stream_state=ForwardGraphStreamState.from_stream(forward_stream),
            kv_view=graph_kv_view,
            metadata=metadata,
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
            bind(metadata, device=packed_embeds.device)
            state.release_backend = lambda: release(metadata)
        graph_ctx = replace(ctx, attention_backend=backend, attention_metadata=metadata, stats=None)

        def run() -> torch.Tensor:
            with use_forward_context(graph_ctx):
                hidden = owner.packed_decoder_forward(
                    state.packed_embeds,
                    image_gen_indicators=state.indicators,
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
                    image_gen_indicators=image_gen_indicators,
                    forward_stream=forward_stream,
                    kv_view=kv_view,
                    text_kv_promotions=text_kv_promotions,
                ),
                before_run=self._prepare_backend,
            )
            planned = getattr(backend, "paged_prefill_graph_wrapper_planned", None)
            if callable(planned) and not planned(metadata):
                raise _MixedGraphBackendUnplanned(
                    "captured packed mixed forward did not plan the graph-scoped prefill wrapper"
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
        state: PackedMixedGraphState,
        packed_embeds: torch.Tensor,
        *,
        image_gen_indicators: torch.Tensor,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
        text_kv_promotions: tuple[PagedTextCacheSpanCopy, ...] = (),
    ) -> None:
        state.packed_embeds.copy_(packed_embeds, non_blocking=True)
        state.indicators.copy_(image_gen_indicators, non_blocking=True)
        state.stream_state.refresh(forward_stream)
        state.kv_view.refresh(kv_view.segments)
        state.metadata.block_table = state.kv_view.block_table(device=packed_embeds.device)
        state.metadata.cache_seqlens = state.kv_view.cache_seqlens_after(device=packed_embeds.device)
        state.metadata.cu_seqlens_q = state.stream_state.stream.cu_seqlens_q
        state.metadata.cu_seqlens_k = state.kv_view.cu_seqlens_after(device=packed_embeds.device)
        state.metadata.max_context_len = state.kv_view.max_seqlen_k()
        if state.promotion_source_index is not None and text_kv_promotions:
            PackedMixedGraphRunner._copy_promotion_indices(state, text_kv_promotions)

    @staticmethod
    def _copy_promotion_indices(
        state: PackedMixedGraphState,
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
            raise invalid_descriptor("packed mixed graph promotion pool changed")
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
    def _copy_promotions_in_graph(state: PackedMixedGraphState) -> None:
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
    def _prepare_backend(state: PackedMixedGraphState) -> None:
        prepare = getattr(state.backend, "prepare_paged_prefill_cuda_graph", None)
        if not callable(prepare):
            return
        prepare(
            state.metadata,
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
    def _replay(state: PackedMixedGraphState) -> torch.Tensor:
        state.graph.replay()
        assert state.logits is not None
        return state.logits

    def _record(self, ctx: Any, event: GraphEvent, tokens: int) -> None:
        record_graph_stats(
            ctx,
            event,
            mode=ForwardMode.MIXED,
            unpadded_tokens=int(tokens),
            padded_tokens=int(tokens),
        )

    def _disable(self, key: tuple[Any, ...], exc: BaseException) -> None:
        self.disabled.add(key)
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
            preferred = getattr(ctx, "attention_backend_name", None)
            explicit_backend = _explicit_attention_backend_name(preferred)
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
        image_gen_indicators: torch.Tensor,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
        backend: Any,
        text_kv_promotions: tuple[PagedTextCacheSpanCopy, ...] = (),
        text_kv_promotion_capacity: int | None = None,
    ) -> tuple[Any, ...] | None:
        if tuple(image_gen_indicators.shape) != (int(packed_embeds.shape[0]),):
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
            str(image_gen_indicators.dtype),
            tuple(int(dim) for dim in image_gen_indicators.shape),
            _stream_geometry(forward_stream),
            _modality_index_geometry(forward_stream),
            _kv_geometry(kv_view),
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
        raise RuntimeError("packed mixed graph requires an explicit attention binding")
    return packed_attention()


def _attention_scale(attention: Any) -> float:
    scale = getattr(attention, "scaling", None)
    if scale is None:
        scale = getattr(attention, "scale", None)
    if scale is None:
        raise RuntimeError("packed mixed graph attention does not expose a scale")
    return float(scale)


def _stream_geometry(forward_stream: ForwardStream) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            int(seg.q_len),
            str(seg.mode.value),
            seg.modality,
            seg.segment_class,
            seg.visible_policy,
            int(seg.branch_id),
        )
        for seg in forward_stream.segments
    )


def _kv_geometry(kv_view: ForwardPagedKVView) -> tuple[tuple[int, bool, bool, int], ...]:
    return tuple(
        (
            int(seg.q_len),
            bool(seg.write_kv),
            bool(seg.persist_kv),
            int(seg.branch_id),
        )
        for seg in kv_view.segments
    )


def _modality_index_geometry(forward_stream: ForwardStream) -> tuple[Any, ...]:
    def geometry(indices: torch.Tensor | None) -> tuple[Any, ...] | None:
        if indices is None:
            return None
        return (tuple(int(dim) for dim in indices.shape), str(indices.dtype))

    return (geometry(forward_stream.und_indices), geometry(forward_stream.gen_indices))


def packed_mixed_graph_promotions_supported(
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
            raise invalid_descriptor("packed mixed graph promotion capacity requires an index")
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
        raise invalid_descriptor("packed mixed graph promotion index length changed")
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


def _explicit_attention_backend_name(name: str | None) -> str | None:
    from ....backends.attention.registry import normalize_attention_backend_name

    normalized = normalize_attention_backend_name(name)
    if normalized in {"", "auto", "0", "false", "off", "1", "true", "on"}:
        return None
    if normalized == "eager":
        return "torch_sdpa"
    return normalized


def packed_mixed_graph_runner(owner: Any) -> PackedMixedGraphRunner:
    runner = getattr(owner, _RUNNER_ATTR, None)
    if runner is None:
        runner = PackedMixedGraphRunner()
        setattr(owner, _RUNNER_ATTR, runner)
    return runner


def maybe_run_packed_mixed_graph(
    owner: Any,
    packed_embeds: torch.Tensor,
    *,
    image_gen_indicators: torch.Tensor,
    forward_stream: ForwardStream,
    kv_view: ForwardPagedKVView,
    text_kv_promotions: tuple[PagedTextCacheSpanCopy, ...] = (),
    text_kv_promotion_capacity: int | None = None,
) -> torch.Tensor | None:
    runner = getattr(owner, _RUNNER_ATTR, None)
    if runner is None:
        runner = packed_mixed_graph_runner(owner)
    return runner.maybe_run(
        owner,
        packed_embeds,
        image_gen_indicators=image_gen_indicators,
        forward_stream=forward_stream,
        kv_view=kv_view,
        text_kv_promotions=text_kv_promotions,
        text_kv_promotion_capacity=text_kv_promotion_capacity,
    )
