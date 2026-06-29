"""CUDA graph plumbing for text decode (re-exports the shared and prefill surface)."""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch

from ..backends.paged_kv_math import decode_write_locations
from ..contracts.forward_context import ForwardContext, TextAttentionMetadata, use_forward_context
from ..contracts.forward_mode import ForwardMode
from ..foundation.errors import invalid_descriptor
from ..runtime.kv_pool import PagedKVPool
from ..runtime.paged_text_cache import BatchedPagedRequestCache
from .cuda_graph_base import (
    _DEFAULT_DECODE_GRAPH_BATCH_SIZES,
    _DEFAULT_METRIC_PREFIX,
    GraphEvent,
    _GraphRunnerBase,
    _parse_positive_int_csv,
    _share_decode_graph_input_buffer,
    maybe_weak_ref_cuda_graph_tensor,
    record_graph_stats,
)

__all__ = [
    'GraphEvent',
    'record_graph_stats',
    'TextDecodeGraphState',
    'TextInitialPrefillGraphState',
    'DecodeCudaGraphRunner',
    'PrefillCudaGraphRunner',
    'make_text_decode_graph_state',
    'make_text_initial_prefill_graph_state',
    'copy_text_decode_graph_inputs',
    'copy_text_initial_prefill_graph_inputs',
    'maybe_weak_ref_cuda_graph_tensor',
]

@dataclass
class TextDecodeGraphState:
    """Static buffers bound into a captured paged-text decode graph."""

    batch_size: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    graph: torch.cuda.CUDAGraph
    cache: BatchedPagedRequestCache
    metadata: TextAttentionMetadata
    logits: torch.Tensor | None = None


class DecodeCudaGraphRunner(_GraphRunnerBase):
    """Own per-bucket decode CUDA graph state and its capture/replay lifecycle."""

    def __init__(
        self,
        *,
        name: str,
        enabled_env: str,
        warmup_env: str,
        warmup_batches_env: str,
        default_enabled: bool = True,
        default_warmup: bool = True,
        default_warmup_batch_sizes: tuple[int, ...] = _DEFAULT_DECODE_GRAPH_BATCH_SIZES,
        metric_prefix: str = _DEFAULT_METRIC_PREFIX,
        logger: Any = None,
    ) -> None:
        self.name = str(name)
        self.enabled_env = enabled_env
        self.warmup_env = warmup_env
        self.warmup_batches_env = warmup_batches_env
        self.default_enabled = bool(default_enabled)
        self.default_warmup = bool(default_warmup)
        self.default_warmup_batch_sizes = tuple(
            sorted({int(size) for size in default_warmup_batch_sizes if int(size) > 0})
        )
        self.metric_prefix = str(metric_prefix)
        self.logger = logger
        self.states: dict[int, TextDecodeGraphState] = {}
        self.disabled: set[int] = set()
        self._capture_pool: Any = None
        self._graph_input_buffer_pool: dict[tuple[str, str, str], torch.Tensor] = {}

    def warmup_batch_sizes(self) -> tuple[int, ...]:
        raw = os.environ.get(self.warmup_batches_env)
        if raw:
            sizes = _parse_positive_int_csv(raw)
            if sizes:
                return sizes
        return self.default_warmup_batch_sizes

    def warmup_capture_batch_sizes(self) -> tuple[int, ...]:
        return tuple(reversed(self.warmup_batch_sizes()))

    def bucket_batch_size(self, batch_size: int) -> int:
        batch_size = int(batch_size)
        candidates = [
            int(size)
            for size in self.warmup_batch_sizes()
            if int(size) >= batch_size and int(size) not in self.disabled
        ]
        if candidates:
            return min(candidates)
        return batch_size

    def resolve_bucket(self, batch_size: int) -> int:
        """Authoritative decode bucket: prefer an already-captured bucket."""

        batch_size = int(batch_size)
        candidates = [
            int(size)
            for size in self.states
            if int(size) >= batch_size and int(size) not in self.disabled
        ]
        if candidates:
            return min(candidates)
        return self.bucket_batch_size(batch_size)

    def can_use(self, batch_size: int) -> bool:
        return self.enabled() and int(batch_size) > 0 and int(batch_size) not in self.disabled

    def disable(self, batch_size: int, exc: BaseException, *, phase: str) -> None:
        batch_size = int(batch_size)
        self.disabled.add(batch_size)
        self.states.pop(batch_size, None)
        if self.logger is not None:
            self.logger.warning(
                "%s %s decode CUDA graph for batch size %s: %s",
                phase,
                self.name,
                batch_size,
                exc,
            )

    @staticmethod
    def record_stats(
        ctx: Any,
        event: str,
        *,
        batch_size: int,
        graph_batch_size: int | None = None,
    ) -> None:
        batch_size = int(batch_size)
        graph_batch_size = int(graph_batch_size or batch_size)
        record_graph_stats(
            ctx,
            event,
            mode=ForwardMode.DECODE,
            unpadded_tokens=batch_size,
            padded_tokens=graph_batch_size,
        )

    def make_state(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        batch_size: int,
        device: torch.device | str,
    ) -> TextDecodeGraphState:
        return make_text_decode_graph_state(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            batch_size=batch_size,
            device=device,
            buffer_pool=self,
        )

    def capture(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: TextAttentionMetadata,
        ctx: Any,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
    ) -> TextDecodeGraphState:
        """Capture a decode graph bucket bound to the model's forward closure."""

        device = input_ids.device
        state = self.make_state(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            batch_size=batch_size,
            device=device,
        )
        copy_text_decode_graph_inputs(
            state,
            input_ids=input_ids,
            positions=positions,
            attention_metadata=attention_metadata,
        )
        graph_ctx = replace(ctx, attention_metadata=state.metadata, stats=None)

        def run() -> torch.Tensor:
            with use_forward_context(graph_ctx):
                return forward_fn(state)

        def copy_inputs(capture_state: TextDecodeGraphState) -> None:
            copy_text_decode_graph_inputs(
                capture_state,
                input_ids=input_ids,
                positions=positions,
                attention_metadata=attention_metadata,
            )

        def prepare(capture_state: TextDecodeGraphState) -> None:
            if prepare_backend is not None:
                prepare_backend(capture_state, graph_ctx)

        return self._capture_graph_state(
            device=device,
            state=state,
            run=run,
            copy_inputs=copy_inputs,
            before_run=prepare,
        )

    def _record_decode_graph_miss(
        self,
        *,
        ctx: Any,
        batch_size: int,
        graph_batch_size: int,
    ) -> None:
        self.record_stats(
            ctx,
            GraphEvent.MISS,
            batch_size=batch_size,
            graph_batch_size=graph_batch_size,
        )

    def _capture_decode_graph(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        graph_batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: TextAttentionMetadata,
        ctx: Any,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None,
    ) -> TextDecodeGraphState:
        return self.capture(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            batch_size=graph_batch_size,
            input_ids=input_ids,
            positions=positions,
            attention_metadata=attention_metadata,
            ctx=ctx,
            forward_fn=forward_fn,
            prepare_backend=prepare_backend,
        )

    def _prepare_decode_graph(
        self,
        state: TextDecodeGraphState,
        ctx: Any,
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None,
    ) -> None:
        if prepare_backend is not None:
            prepare_backend(state, ctx)

    def _record_decode_graph_event(
        self,
        *,
        ctx: Any,
        event: GraphEvent,
        batch_size: int,
        graph_batch_size: int,
    ) -> None:
        self.record_stats(
            ctx,
            event,
            batch_size=batch_size,
            graph_batch_size=graph_batch_size,
        )

    def maybe_run(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: TextAttentionMetadata,
        ctx: Any,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
    ) -> torch.Tensor | None:
        """Capture-or-replay the decode graph for ``batch_size``; ``None`` on miss."""

        graph_batch_size = self.resolve_bucket(batch_size)
        if not self.can_use(graph_batch_size):
            self._record_decode_graph_miss(
                ctx=ctx,
                batch_size=batch_size,
                graph_batch_size=graph_batch_size,
            )
            return None

        return self._capture_or_replay(
            key=graph_batch_size,
            ctx=ctx,
            capture=lambda: self._capture_decode_graph(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                graph_batch_size=graph_batch_size,
                input_ids=input_ids,
                positions=positions,
                attention_metadata=attention_metadata,
                ctx=ctx,
                forward_fn=forward_fn,
                prepare_backend=prepare_backend,
            ),
            copy_inputs=lambda state: copy_text_decode_graph_inputs(
                state,
                input_ids=input_ids,
                positions=positions,
                attention_metadata=attention_metadata,
            ),
            replay=lambda state: _replay_decode_graph(state, batch_size),
            record=lambda event: self._record_decode_graph_event(
                ctx=ctx,
                event=event,
                batch_size=batch_size,
                graph_batch_size=graph_batch_size,
            ),
            disable=lambda exc: self.disable(graph_batch_size, exc, phase="disabling"),
            capture_metric=f"{self.metric_prefix}decode_graph_capture",
            input_copy_metric=f"{self.metric_prefix}decode_graph_input_copy",
            replay_metric=f"{self.metric_prefix}decode_graph_replay_launch",
            after_copy=lambda state: self._prepare_decode_graph(state, ctx, prepare_backend),
            after_copy_metric=f"{self.metric_prefix}decode_graph_attention_prepare",
        )

    def warmup(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        device: torch.device,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
    ) -> None:
        """Pre-capture decode graph buckets ahead of serving."""

        ctx = ForwardContext(attention_backend_name="auto")
        def should_skip(batch_size: int) -> bool:
            return int(batch_size) <= 0 or int(batch_size) > int(num_blocks)

        def capture_bucket(batch_size: int, capture_ctx: ForwardContext) -> TextDecodeGraphState:
            batch_size = int(batch_size)
            input_ids, positions = _synthetic_decode_inputs(batch_size, device)
            metadata = _synthetic_decode_metadata(
                _synthetic_decode_cache(kv_pool, batch_size),
                batch_size,
                device,
            )
            return self.capture(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                batch_size=batch_size,
                input_ids=input_ids,
                positions=positions,
                attention_metadata=metadata,
                ctx=capture_ctx,
                forward_fn=forward_fn,
                prepare_backend=prepare_backend,
            )

        def copy_inputs(batch_size: int, state: TextDecodeGraphState) -> None:
            batch_size = int(batch_size)
            input_ids, positions = _synthetic_decode_inputs(batch_size, device)
            metadata = _synthetic_decode_metadata(state.cache, batch_size, device)
            copy_text_decode_graph_inputs(
                state,
                input_ids=input_ids,
                positions=positions,
                attention_metadata=metadata,
            )

        self._warmup_capture_buckets(
            device=device,
            ctx=ctx,
            buckets=self.warmup_capture_batch_sizes,
            key_for=int,
            should_skip=should_skip,
            capture_bucket=capture_bucket,
            copy_inputs=copy_inputs,
            replay=lambda state: state.graph.replay(),
            disable=lambda batch_size, exc: self.disable(
                int(batch_size), exc, phase="skipping warmup for"
            ),
        )


def _synthetic_decode_inputs(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (int(batch_size), 1)
    return (
        torch.zeros(shape, dtype=torch.long, device=device),
        torch.zeros(shape, dtype=torch.long, device=device),
    )


def _synthetic_decode_cache(kv_pool: PagedKVPool, batch_size: int) -> BatchedPagedRequestCache:
    batch_size = int(batch_size)
    return BatchedPagedRequestCache(
        kv_pool,
        [[row] for row in range(batch_size)],
        [0 for _ in range(batch_size)],
    )


def _replay_decode_graph(state: TextDecodeGraphState, batch_size: int) -> torch.Tensor:
    state.graph.replay()
    return state.logits[:batch_size]


def _synthetic_decode_metadata(
    cache: BatchedPagedRequestCache,
    batch_size: int,
    device: torch.device,
) -> TextAttentionMetadata:
    batch_size = int(batch_size)
    return TextAttentionMetadata(
        cache=cache,
        block_table=cache.block_table(device=device),
        cache_seqlens=cache.cache_seqlens(device=device),
        cache_seqlens_cpu=tuple(0 for _ in range(batch_size)),
        query_lens=torch.ones(batch_size, dtype=torch.int32, device=device),
        query_lens_cpu=tuple(1 for _ in range(batch_size)),
        kv_seqlens_cpu=tuple(1 for _ in range(batch_size)),
        mode=ForwardMode.DECODE,
    )


def make_text_decode_graph_state(
    *,
    kv_pool: PagedKVPool,
    num_blocks: int,
    batch_size: int,
    device: torch.device | str,
    buffer_pool: _GraphRunnerBase | None = None,
) -> TextDecodeGraphState:
    """Allocate fixed buffers for a paged one-token decode graph bucket."""

    batch_size = int(batch_size)
    device = torch.device(device)
    max_blocks_per_seq = max(1, int(num_blocks))
    if buffer_pool is not None:
        share = buffer_pool.share_graph_input_buffer
    else:
        share = _share_decode_graph_input_buffer
    graph_cache = BatchedPagedRequestCache(
        kv_pool,
        [[] for _ in range(batch_size)],
        [0 for _ in range(batch_size)],
    )
    block_table = share(
        "text_decode.block_table",
        torch.empty((batch_size, max_blocks_per_seq), dtype=torch.int32, device=device),
    )
    cache_seqlens = share(
        "text_decode.cache_seqlens",
        torch.empty(batch_size, dtype=torch.int32, device=device),
    )
    state = TextDecodeGraphState(
        batch_size=batch_size,
        input_ids=share(
            "text_decode.input_ids",
            torch.empty((batch_size, 1), dtype=torch.long, device=device),
        ),
        positions=share(
            "text_decode.positions",
            torch.empty((batch_size, 1), dtype=torch.long, device=device),
        ),
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        graph=torch.cuda.CUDAGraph(),
        cache=graph_cache,
        metadata=TextAttentionMetadata.for_decode_graph(
            cache=graph_cache,
            batch_size=batch_size,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            query_lens=share(
                "text_decode.query_lens",
                torch.ones(batch_size, dtype=torch.int32, device=device),
            ),
            decode_page_ids=share(
                "text_decode.decode_page_ids",
                torch.empty(batch_size, dtype=torch.long, device=device),
            ),
            decode_page_offsets=share(
                "text_decode.decode_page_offsets",
                torch.empty(batch_size, dtype=torch.long, device=device),
            ),
        ),
    )
    return state


def copy_text_decode_graph_inputs(
    state: TextDecodeGraphState,
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    attention_metadata: TextAttentionMetadata,
) -> None:
    """Refresh dynamic tensors feeding a captured paged one-token decode graph."""

    actual_batch = int(input_ids.shape[0])
    if actual_batch <= 0 or actual_batch > state.batch_size or int(input_ids.shape[1]) != 1:
        raise invalid_descriptor("decode CUDA graph input shape mismatch")
    state.input_ids[:actual_batch].copy_(input_ids, non_blocking=True)
    state.positions[:actual_batch].copy_(positions, non_blocking=True)
    if actual_batch < state.batch_size:
        state.input_ids[actual_batch:].zero_()
        state.positions[actual_batch:].zero_()
    block_table = attention_metadata.block_table
    if block_table is None:
        raise invalid_descriptor("decode CUDA graph block table is missing")
    if block_table.shape[0] != actual_batch:
        raise invalid_descriptor("decode CUDA graph block-table batch mismatch")
    if block_table.shape[1] > state.block_table.shape[1]:
        raise invalid_descriptor("decode CUDA graph block-table width exceeded")
    state.block_table[:actual_batch, : block_table.shape[1]].copy_(
        block_table.to(dtype=torch.int32),
        non_blocking=True,
    )
    if block_table.shape[1] < state.block_table.shape[1]:
        state.block_table[:actual_batch, block_table.shape[1]:].zero_()
    if actual_batch < state.batch_size:
        state.block_table[actual_batch:].zero_()
    cache_seqlens = attention_metadata.cache_seqlens
    if cache_seqlens is None:
        raise invalid_descriptor("decode CUDA graph cache lengths are missing")
    if int(cache_seqlens.shape[0]) != actual_batch:
        raise invalid_descriptor("decode CUDA graph cache length batch mismatch")
    state.cache_seqlens[:actual_batch].copy_(cache_seqlens.to(dtype=torch.int32), non_blocking=True)
    if actual_batch < state.batch_size:
        state.cache_seqlens[actual_batch:].zero_()
    decode_page_ids = getattr(state.metadata, "decode_page_ids", None)
    decode_page_offsets = getattr(state.metadata, "decode_page_offsets", None)
    if (
        isinstance(decode_page_ids, torch.Tensor)
        and isinstance(decode_page_offsets, torch.Tensor)
        and isinstance(state.cache, BatchedPagedRequestCache)
    ):
        page_ids, offsets = decode_write_locations(
            state.block_table[:actual_batch],
            state.cache_seqlens[:actual_batch],
            state.cache.pool.block_size,
        )
        decode_page_ids[:actual_batch].copy_(page_ids, non_blocking=True)
        decode_page_offsets[:actual_batch].copy_(offsets, non_blocking=True)
        if actual_batch < state.batch_size:
            decode_page_ids[actual_batch:].zero_()
            decode_page_offsets[actual_batch:].zero_()
    cache_cpu = tuple(int(x) for x in getattr(attention_metadata, "cache_seqlens_cpu", ())[:actual_batch])
    kv_cpu = tuple(int(x) for x in getattr(attention_metadata, "kv_seqlens_cpu", ())[:actual_batch])
    if len(cache_cpu) != actual_batch:
        cache_cpu = ()
    if len(kv_cpu) != actual_batch:
        kv_cpu = ()
    if cache_cpu and actual_batch < state.batch_size:
        cache_cpu = cache_cpu + tuple(0 for _ in range(state.batch_size - actual_batch))
    if kv_cpu and actual_batch < state.batch_size:
        kv_cpu = kv_cpu + tuple(1 for _ in range(state.batch_size - actual_batch))
    # Captured graph closures keep a reference to this metadata object.  Keep
    # its identity stable so backend graph-wrapper lookup and replay planning
    # stay attached to the same object while the dynamic CPU summaries change.
    state.metadata.cache_seqlens_cpu = cache_cpu
    state.metadata.kv_seqlens_cpu = kv_cpu




from .prefill_cuda_graph import (  # noqa: E402
    PrefillCudaGraphRunner,
    TextInitialPrefillGraphState,
    copy_text_initial_prefill_graph_inputs,
    make_text_initial_prefill_graph_state,
)
