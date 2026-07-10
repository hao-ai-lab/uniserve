"""CUDA graph plumbing for text decode (re-exports the shared and prefill surface)."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from ....backends.paged_kv_math import decode_write_locations
from ....contracts.forward_context import ForwardContext, TextAttentionMetadata, use_forward_context
from ....contracts.forward_mode import ForwardMode
from ....foundation.errors import invalid_descriptor
from ....foundation.sizing import ceil_div
from ....runtime.host_staging import cpu_int_staging_buffer, fill_cpu_ints, is_pinned
from ....runtime.kv_pool import PagedKVPool
from ....runtime.paged_text_cache import BatchedPagedRequestCache
from ....runtime.tensor_views import adjacent_one_token_view
from .base import (
    _DEFAULT_DECODE_GRAPH_BATCH_SIZES,
    _DEFAULT_METRIC_PREFIX,
    GraphEvent,
    _GraphRunnerBase,
    _share_decode_graph_input_buffer,
    maybe_weak_ref_cuda_graph_tensor,
    record_graph_stats,
)

__all__ = [
    'GraphEvent',
    'record_graph_stats',
    'TextDecodeGraphHostInputs',
    'TextDecodeGraphState',
    'TextInitialPrefillGraphState',
    'DecodeCudaGraphRunner',
    'PrefillCudaGraphRunner',
    'make_text_decode_graph_state',
    'make_text_initial_prefill_graph_state',
    'copy_text_decode_graph_inputs',
    'copy_text_decode_graph_host_inputs',
    'copy_text_initial_prefill_graph_inputs',
    'maybe_weak_ref_cuda_graph_tensor',
    'resolve_paged_decode_graph_backend',
    'prepare_paged_decode_graph_backend',
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
    long_inputs: torch.Tensor | None = None
    block_table_rows: tuple[tuple[int, ...], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TextDecodeGraphHostInputs:
    """Host-resident dynamic inputs for a one-token decode graph replay."""

    input_ids: Sequence[int]
    positions: Sequence[int]
    block_ids_by_row: Sequence[Sequence[int]]
    cache_seqlens_cpu: Sequence[int]
    kv_seqlens_cpu: Sequence[int]
    token_replacements: Sequence[tuple[int, torch.Tensor]] = field(default_factory=tuple)
    max_context_len: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.input_ids)


class DecodeCudaGraphRunner(_GraphRunnerBase):
    """Own per-bucket decode CUDA graph state and its capture/replay lifecycle."""

    def __init__(
        self,
        *,
        name: str,
        default_enabled: bool = True,
        default_warmup: bool = True,
        default_warmup_batch_sizes: tuple[int, ...] = _DEFAULT_DECODE_GRAPH_BATCH_SIZES,
        metric_prefix: str = _DEFAULT_METRIC_PREFIX,
        logger: Any = None,
    ) -> None:
        self.name = str(name)
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
        max_context_len: int = 0,
    ) -> TextDecodeGraphState:
        return make_text_decode_graph_state(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            batch_size=batch_size,
            device=device,
            buffer_pool=self,
            max_context_len=max_context_len,
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
            max_context_len=int(getattr(attention_metadata, "max_context_len", 0) or 0),
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

    def capture_host_inputs(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        batch_size: int,
        device: torch.device | str,
        host_inputs: TextDecodeGraphHostInputs,
        ctx: Any,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
        staging_slot: Any | None = None,
    ) -> TextDecodeGraphState:
        """Capture a decode graph bucket using host-staged dynamic inputs."""

        state = self.make_state(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            batch_size=batch_size,
            device=device,
            max_context_len=int(host_inputs.max_context_len),
        )
        copy_text_decode_graph_host_inputs(state, host_inputs, staging_slot=staging_slot)
        graph_ctx = replace(ctx, attention_metadata=state.metadata, stats=None)

        def run() -> torch.Tensor:
            with use_forward_context(graph_ctx):
                return forward_fn(state)

        def copy_inputs(capture_state: TextDecodeGraphState) -> None:
            copy_text_decode_graph_host_inputs(capture_state, host_inputs, staging_slot=staging_slot)

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

    def maybe_run_host_inputs(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        device: torch.device | str,
        host_inputs: TextDecodeGraphHostInputs,
        ctx: Any,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
        staging_slot: Any | None = None,
    ) -> torch.Tensor | None:
        """Capture-or-replay the decode graph from host-staged inputs."""

        batch_size = int(host_inputs.batch_size)
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
            capture=lambda: self.capture_host_inputs(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                batch_size=graph_batch_size,
                device=device,
                host_inputs=host_inputs,
                ctx=ctx,
                forward_fn=forward_fn,
                prepare_backend=prepare_backend,
                staging_slot=staging_slot,
            ),
            copy_inputs=lambda state: copy_text_decode_graph_host_inputs(
                state,
                host_inputs,
                staging_slot=staging_slot,
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
        max_context_len: int = 0,
        attention_backend_name: str | None = "auto",
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
    ) -> None:
        """Pre-capture decode graph buckets ahead of serving."""

        ctx = ForwardContext(attention_backend_name=attention_backend_name or "auto")
        def should_skip(batch_size: int) -> bool:
            return int(batch_size) <= 0 or int(batch_size) > int(num_blocks)

        def capture_bucket(batch_size: int, capture_ctx: ForwardContext) -> TextDecodeGraphState:
            batch_size = int(batch_size)
            input_ids, positions = _synthetic_decode_inputs(batch_size, device)
            metadata = _synthetic_decode_metadata(
                _synthetic_decode_cache(kv_pool, batch_size),
                batch_size,
                device,
                max_context_len=max_context_len,
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
            metadata = _synthetic_decode_metadata(
                state.cache,
                batch_size,
                device,
                max_context_len=max_context_len,
            )
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
    if state.logits is None:
        raise RuntimeError("captured decode graph has no logits buffer")
    return state.logits[:batch_size]


def _synthetic_decode_metadata(
    cache: BatchedPagedRequestCache,
    batch_size: int,
    device: torch.device,
    *,
    max_context_len: int = 0,
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
        max_context_len=int(max_context_len),
        mode=ForwardMode.DECODE,
    )


def make_text_decode_graph_state(
    *,
    kv_pool: PagedKVPool,
    num_blocks: int,
    batch_size: int,
    device: torch.device | str,
    buffer_pool: _GraphRunnerBase | None = None,
    max_context_len: int = 0,
) -> TextDecodeGraphState:
    """Allocate fixed buffers for a paged one-token decode graph bucket."""

    batch_size = int(batch_size)
    device = torch.device(device)
    max_blocks_per_seq = max(1, int(num_blocks))
    context_len = max(0, int(max_context_len))
    if context_len > 0:
        max_blocks_per_seq = max(1, min(max_blocks_per_seq, ceil_div(context_len, kv_pool.block_size)))
    if buffer_pool is not None:
        share = buffer_pool.share_graph_input_buffer
    else:
        share = _share_decode_graph_input_buffer
    graph_cache = BatchedPagedRequestCache(
        kv_pool,
        [[] for _ in range(batch_size)],
        [0 for _ in range(batch_size)],
    )
    long_inputs = share(
        "text_decode.long_inputs",
        torch.empty(4 * batch_size, dtype=torch.long, device=device),
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
        input_ids=long_inputs[:batch_size].view(batch_size, 1),
        positions=long_inputs[batch_size : 2 * batch_size].view(batch_size, 1),
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
            decode_page_ids=long_inputs[2 * batch_size : 3 * batch_size],
            decode_page_offsets=long_inputs[3 * batch_size : 4 * batch_size],
            max_context_len=max_context_len,
        ),
        long_inputs=long_inputs,
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
    source_cache = getattr(attention_metadata, "cache", None)
    if isinstance(state.cache, BatchedPagedRequestCache) and isinstance(source_cache, BatchedPagedRequestCache):
        if len(source_cache.block_ids_by_row) != actual_batch or len(source_cache.base_lens) != actual_batch:
            raise invalid_descriptor("decode CUDA graph cache row batch mismatch")
        graph_block_rows = [list(row) for row in source_cache.block_ids_by_row]
        graph_base_lens = [int(length) for length in source_cache.base_lens]
        if actual_batch < state.batch_size:
            graph_block_rows.extend([] for _ in range(state.batch_size - actual_batch))
            graph_base_lens.extend(0 for _ in range(state.batch_size - actual_batch))
        state.cache.reset_rows(graph_block_rows, graph_base_lens)
    block_table = attention_metadata.block_table
    if block_table is None:
        raise invalid_descriptor("decode CUDA graph block table is missing")
    if block_table.shape[0] != actual_batch:
        raise invalid_descriptor("decode CUDA graph block-table batch mismatch")
    if block_table.shape[1] > state.block_table.shape[1]:
        raise invalid_descriptor("decode CUDA graph block-table width exceeded")
    block_rows_key: tuple[tuple[int, ...], ...] | None = None
    if isinstance(source_cache, BatchedPagedRequestCache):
        block_rows_key = _block_rows_key(source_cache.block_ids_by_row[:actual_batch])
    if not _block_table_rows_match(state, block_rows_key):
        state.block_table[:actual_batch, : block_table.shape[1]].copy_(
            block_table.to(dtype=torch.int32),
            non_blocking=True,
        )
        if block_table.shape[1] < state.block_table.shape[1]:
            state.block_table[:actual_batch, block_table.shape[1]:].zero_()
        if actual_batch < state.batch_size:
            state.block_table[actual_batch:].zero_()
        state.block_table_rows = block_rows_key or ()
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
    state.metadata.max_context_len = int(
        getattr(attention_metadata, "max_context_len", 0) or state.metadata.max_context_len
    )


def copy_text_decode_graph_host_inputs(
    state: TextDecodeGraphState,
    host_inputs: TextDecodeGraphHostInputs,
    *,
    staging_slot: Any | None = None,
) -> None:
    """Refresh decode graph inputs directly from host row descriptors."""

    rows = _normalize_text_decode_graph_host_inputs(state, host_inputs)
    actual_batch = len(rows["input_ids"])
    block_rows = rows["block_ids_by_row"]
    max_blocks = max(len(row) for row in block_rows)
    if max_blocks > int(state.block_table.shape[1]):
        raise invalid_descriptor("decode CUDA graph block-table width exceeded")

    cache_lens = rows["cache_seqlens_cpu"]
    decode_page_ids = getattr(state.metadata, "decode_page_ids", None)
    decode_page_offsets = getattr(state.metadata, "decode_page_offsets", None)
    page_ids: list[int] = []
    offsets: list[int] = []
    if isinstance(decode_page_ids, torch.Tensor) and isinstance(decode_page_offsets, torch.Tensor):
        page_ids, offsets = _host_decode_write_locations(
            block_rows,
            cache_lens,
            int(state.cache.pool.block_size),
        )

    dense_replacements = _dense_token_replacements(
        state,
        host_inputs.token_replacements,
        actual_batch=actual_batch,
    )
    copied_long = _copy_fused_long_host_inputs(
        state,
        input_ids=rows["input_ids"],
        positions=rows["positions"],
        page_ids=page_ids,
        page_offsets=offsets,
        include_input_ids=dense_replacements is None,
        actual_batch=actual_batch,
        slot=staging_slot,
    )
    if not copied_long:
        _copy_host_ints_to_device(
            rows["input_ids"],
            state.input_ids[:actual_batch],
            dtype=torch.long,
            slot=staging_slot,
            name="text_decode.input_ids",
            view_shape=(actual_batch, 1),
        )
        _copy_host_ints_to_device(
            rows["positions"],
            state.positions[:actual_batch],
            dtype=torch.long,
            slot=staging_slot,
            name="text_decode.positions",
            view_shape=(actual_batch, 1),
        )
        if isinstance(decode_page_ids, torch.Tensor) and isinstance(decode_page_offsets, torch.Tensor):
            _copy_host_ints_to_device(
                page_ids,
                decode_page_ids[:actual_batch],
                dtype=torch.long,
                slot=staging_slot,
                name="text_decode.decode_page_ids",
            )
            _copy_host_ints_to_device(
                offsets,
                decode_page_offsets[:actual_batch],
                dtype=torch.long,
                slot=staging_slot,
                name="text_decode.decode_page_offsets",
            )
        if actual_batch < state.batch_size:
            state.input_ids[actual_batch:].zero_()
            state.positions[actual_batch:].zero_()
            if isinstance(decode_page_ids, torch.Tensor) and isinstance(decode_page_offsets, torch.Tensor):
                decode_page_ids[actual_batch:].zero_()
                decode_page_offsets[actual_batch:].zero_()
    if dense_replacements is None:
        _copy_token_replacements(
            state,
            host_inputs.token_replacements,
            actual_batch=actual_batch,
        )
    else:
        _copy_dense_token_replacements(state, dense_replacements)

    block_rows_key = _block_rows_key(block_rows)
    if not _block_table_rows_match(state, block_rows_key):
        block_values: list[int] = []
        for row in block_rows:
            block_values.extend(row)
            block_values.extend(0 for _ in range(max_blocks - len(row)))
        _copy_host_ints_to_device(
            block_values,
            state.block_table[:actual_batch, :max_blocks],
            dtype=torch.int32,
            slot=staging_slot,
            name="text_decode.block_table",
            view_shape=(actual_batch, max_blocks),
        )
        if max_blocks < int(state.block_table.shape[1]):
            state.block_table[:actual_batch, max_blocks:].zero_()
        if actual_batch < state.batch_size:
            state.block_table[actual_batch:].zero_()
        state.block_table_rows = block_rows_key

    _copy_host_ints_to_device(
        cache_lens,
        state.cache_seqlens[:actual_batch],
        dtype=torch.int32,
        slot=staging_slot,
        name="text_decode.cache_seqlens",
    )
    if actual_batch < state.batch_size:
        state.cache_seqlens[actual_batch:].zero_()

    if isinstance(state.cache, BatchedPagedRequestCache):
        graph_block_rows = [list(row) for row in block_rows]
        graph_base_lens = [int(length) for length in cache_lens]
        if actual_batch < state.batch_size:
            graph_block_rows.extend([] for _ in range(state.batch_size - actual_batch))
            graph_base_lens.extend(0 for _ in range(state.batch_size - actual_batch))
        state.cache.reset_rows(graph_block_rows, graph_base_lens)

    cache_cpu = tuple(int(x) for x in rows["cache_seqlens_cpu"])
    kv_cpu = tuple(int(x) for x in rows["kv_seqlens_cpu"])
    if actual_batch < state.batch_size:
        cache_cpu = cache_cpu + tuple(0 for _ in range(state.batch_size - actual_batch))
        kv_cpu = kv_cpu + tuple(1 for _ in range(state.batch_size - actual_batch))
    state.metadata.cache_seqlens_cpu = cache_cpu
    state.metadata.kv_seqlens_cpu = kv_cpu
    state.metadata.max_context_len = int(host_inputs.max_context_len or state.metadata.max_context_len)


def _normalize_text_decode_graph_host_inputs(
    state: TextDecodeGraphState,
    host_inputs: TextDecodeGraphHostInputs,
) -> dict[str, Any]:
    actual_batch = int(host_inputs.batch_size)
    if actual_batch <= 0 or actual_batch > int(state.batch_size):
        raise invalid_descriptor("decode CUDA graph input shape mismatch")
    if (
        len(host_inputs.positions) != actual_batch
        or len(host_inputs.block_ids_by_row) != actual_batch
        or len(host_inputs.cache_seqlens_cpu) != actual_batch
        or len(host_inputs.kv_seqlens_cpu) != actual_batch
    ):
        raise invalid_descriptor("decode CUDA graph host input row mismatch")
    block_rows = [[int(block_id) for block_id in row] for row in host_inputs.block_ids_by_row]
    if any(not row for row in block_rows):
        raise invalid_descriptor("decode CUDA graph host block rows must be non-empty")
    cache_lens = [int(length) for length in host_inputs.cache_seqlens_cpu]
    kv_lens = [int(length) for length in host_inputs.kv_seqlens_cpu]
    if any(length < 0 for length in cache_lens) or any(length <= 0 for length in kv_lens):
        raise invalid_descriptor("decode CUDA graph host sequence lengths are invalid")
    return {
        "input_ids": [int(token) for token in host_inputs.input_ids],
        "positions": [int(pos) for pos in host_inputs.positions],
        "block_ids_by_row": block_rows,
        "cache_seqlens_cpu": cache_lens,
        "kv_seqlens_cpu": kv_lens,
    }


def _copy_host_ints_to_device(
    values: Sequence[int],
    target: torch.Tensor,
    *,
    dtype: torch.dtype,
    slot: Any | None,
    name: str,
    view_shape: tuple[int, ...] | None = None,
) -> None:
    cpu = cpu_int_staging_buffer(
        len(values),
        dtype=dtype,
        pin=target.device.type == "cuda",
        slot=slot,
        name=name,
    )
    fill_cpu_ints(cpu, [int(value) for value in values])
    source = cpu if view_shape is None else cpu.view(*view_shape)
    target.copy_(source, non_blocking=target.device.type == "cuda" and is_pinned(cpu))


def _block_rows_key(rows: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(block_id) for block_id in row) for row in rows)


def _block_table_rows_match(
    state: TextDecodeGraphState,
    rows: tuple[tuple[int, ...], ...] | None,
) -> bool:
    return rows is not None and bool(rows) and state.block_table_rows == rows


def _copy_fused_long_host_inputs(
    state: TextDecodeGraphState,
    *,
    input_ids: Sequence[int],
    positions: Sequence[int],
    page_ids: Sequence[int],
    page_offsets: Sequence[int],
    include_input_ids: bool,
    actual_batch: int,
    slot: Any | None,
) -> bool:
    if not _long_inputs_match_state(state):
        return False
    actual_batch = int(actual_batch)
    batch = int(state.batch_size)
    if len(page_ids) > 0 and len(page_ids) != actual_batch:
        raise invalid_descriptor("decode CUDA graph page-id batch mismatch")
    if len(page_offsets) > 0 and len(page_offsets) != actual_batch:
        raise invalid_descriptor("decode CUDA graph page-offset batch mismatch")
    row_values: list[int] = []
    if include_input_ids:
        row_values.extend(int(value) for value in input_ids)
        row_values.extend(0 for _ in range(batch - actual_batch))
    row_values.extend(int(value) for value in positions)
    row_values.extend(0 for _ in range(batch - actual_batch))
    if len(page_ids) > 0:
        row_values.extend(int(value) for value in page_ids)
        row_values.extend(0 for _ in range(batch - actual_batch))
    if len(page_offsets) > 0:
        row_values.extend(int(value) for value in page_offsets)
        row_values.extend(0 for _ in range(batch - actual_batch))
    if not row_values:
        return True
    start = 0 if include_input_ids else batch
    if state.long_inputs is None:
        raise RuntimeError("captured decode graph has no host-input buffer")
    target = state.long_inputs[start : start + len(row_values)]
    _copy_host_ints_to_device(
        row_values,
        target,
        dtype=torch.long,
        slot=slot,
        name="text_decode.long_inputs",
    )
    return True


def _long_inputs_match_state(state: TextDecodeGraphState) -> bool:
    long_inputs = state.long_inputs
    if not isinstance(long_inputs, torch.Tensor):
        return False
    batch = int(state.batch_size)
    if int(long_inputs.numel()) < 4 * batch or long_inputs.dtype != torch.long:
        return False
    decode_page_ids = getattr(state.metadata, "decode_page_ids", None)
    decode_page_offsets = getattr(state.metadata, "decode_page_offsets", None)
    if not isinstance(decode_page_ids, torch.Tensor) or not isinstance(decode_page_offsets, torch.Tensor):
        return False
    return (
        state.input_ids.data_ptr() == long_inputs[:batch].data_ptr()
        and state.positions.data_ptr() == long_inputs[batch : 2 * batch].data_ptr()
        and decode_page_ids.data_ptr() == long_inputs[2 * batch : 3 * batch].data_ptr()
        and decode_page_offsets.data_ptr() == long_inputs[3 * batch : 4 * batch].data_ptr()
    )


def _copy_token_replacements(
    state: TextDecodeGraphState,
    replacements: Sequence[tuple[int, torch.Tensor]],
    *,
    actual_batch: int,
) -> None:
    for row_idx, token in replacements:
        row = int(row_idx)
        if row < 0 or row >= actual_batch:
            raise invalid_descriptor("decode CUDA graph token replacement row is out of range")
        if token.dtype != torch.long or int(token.numel()) != 1:
            raise invalid_descriptor("decode CUDA graph token replacement must be one int64 token")
        if torch.device(token.device) != torch.device(state.input_ids.device):
            raise invalid_descriptor("decode CUDA graph token replacement device mismatch")
        state.input_ids[row, 0:1].copy_(token.reshape(1), non_blocking=True)


def _dense_token_replacements(
    state: TextDecodeGraphState,
    replacements: Sequence[tuple[int, torch.Tensor]],
    *,
    actual_batch: int,
) -> torch.Tensor | list[torch.Tensor] | None:
    if len(replacements) != int(actual_batch):
        return None
    tokens: list[torch.Tensor | None] = [None for _ in range(int(actual_batch))]
    for row_idx, token in replacements:
        row = int(row_idx)
        if row < 0 or row >= int(actual_batch):
            raise invalid_descriptor("decode CUDA graph token replacement row is out of range")
        if tokens[row] is not None:
            return None
        if token.dtype != torch.long or int(token.numel()) != 1:
            raise invalid_descriptor("decode CUDA graph token replacement must be one int64 token")
        if torch.device(token.device) != torch.device(state.input_ids.device):
            raise invalid_descriptor("decode CUDA graph token replacement device mismatch")
        tokens[row] = token.reshape(1)
    if any(token is None for token in tokens):
        return None
    dense = [token for token in tokens if token is not None]
    view = adjacent_one_token_view(dense)
    return view if view is not None else dense


def _copy_dense_token_replacements(
    state: TextDecodeGraphState,
    replacements: torch.Tensor | Sequence[torch.Tensor],
) -> None:
    if isinstance(replacements, torch.Tensor):
        source = replacements.reshape(-1)
        count = int(source.numel())
        if count <= 0:
            return
        state.input_ids[:count, 0].copy_(source, non_blocking=True)
        if count < int(state.batch_size):
            state.input_ids[count:].zero_()
        return
    if not replacements:
        return
    target = state.input_ids[: len(replacements), 0]
    if len(replacements) == 1:
        target.copy_(replacements[0], non_blocking=True)
        if len(replacements) < int(state.batch_size):
            state.input_ids[len(replacements) :].zero_()
        return
    torch.cat(tuple(replacements), out=target)
    if len(replacements) < int(state.batch_size):
        state.input_ids[len(replacements) :].zero_()


def _host_decode_write_locations(
    block_ids_by_row: Sequence[Sequence[int]],
    cache_lens: Sequence[int],
    page_size: int,
) -> tuple[list[int], list[int]]:
    page_size = max(1, int(page_size))
    page_ids: list[int] = []
    offsets: list[int] = []
    for row, length in zip(block_ids_by_row, cache_lens, strict=True):
        page_slot = int(length) // page_size
        if page_slot < 0 or page_slot >= len(row):
            raise invalid_descriptor("decode CUDA graph host row lacks write page")
        page_ids.append(int(row[page_slot]))
        offsets.append(int(length) % page_size)
    return page_ids, offsets


def resolve_paged_decode_graph_backend(attention_backend_name: str | None) -> Any | None:
    """Return the attention backend that can host a *captured* paged-decode graph.

    A paged-decode graph is only correct when its backend refills the page-index /
    length plan buffers before every replay, or when the backend has no wrapper
    plan state to bake into the graph. FlashInfer's wrapper path uses
    ``prepare_paged_decode_cuda_graph``; direct paged-decode backends consume the
    live block-table and sequence-length tensors directly, so a no-op prepare is
    enough. Other backends stay eager rather than capture a stale paged plan.
    """

    from ....backends.attention import (
        get_attention_backend,
        has_attention_backend,
        normalize_attention_backend_name,
    )

    normalized = normalize_attention_backend_name(attention_backend_name)
    if normalized == "auto" and has_attention_backend("trtllm_mha"):
        backend = get_attention_backend("trtllm_mha")
        if bool(getattr(backend.capabilities(), "available", True)):
            return backend
    if normalized == "trtllm_mha" and has_attention_backend("trtllm_mha"):
        backend = get_attention_backend("trtllm_mha")
        if bool(getattr(backend.capabilities(), "available", True)):
            return backend
        return None
    if normalized not in ("auto", "flashinfer"):
        if normalized != "fa4_cute":
            return None
        if not has_attention_backend("fa4_cute"):
            return None
        backend = get_attention_backend("fa4_cute")
        caps = backend.capabilities()
        if not bool(getattr(caps, "available", True)) or not bool(getattr(caps, "paged_kv", False)):
            return None
        if not hasattr(backend, "forward_paged"):
            return None
        return backend
    if has_attention_backend("flashinfer"):
        backend = get_attention_backend("flashinfer")
        if hasattr(backend, "prepare_paged_decode_cuda_graph"):
            return backend
    if normalized != "auto" or not has_attention_backend("fa4_cute"):
        return None
    backend = get_attention_backend("fa4_cute")
    caps = backend.capabilities()
    if not bool(getattr(caps, "available", True)) or not bool(getattr(caps, "paged_kv", False)):
        return None
    if not hasattr(backend, "forward_paged"):
        return None
    return backend


def resolve_paged_decode_graph_prepare(
    *,
    owner: Any,
    kv_pool: PagedKVPool,
    num_blocks: int,
    attention_backend_name: str | None,
    before: Callable[[TextDecodeGraphState, Any], None] | None = None,
) -> Callable[[TextDecodeGraphState, Any], None] | None:
    """Build the per-replay decode-graph prepare hook, or ``None`` to stay eager.

    The single assembly point for paged-decode graph preparation shared by the
    thin :class:`~uniserve_worker.execution.forward.graph.text.TextGraphRunner`
    and the interleaved decode adapter: KV-side geometry comes off the shared
    pool, query-side geometry from the owner's
    ``text_decode_graph_query_geometry`` hook, and the backend must expose
    graph-aware planning (otherwise the capture-time plan would be baked in and
    the caller must stay eager). ``before`` runs first on every capture/replay
    for caller-specific static state (e.g. the interleaved indexes sidecar).
    """

    geometry_hook = getattr(owner, "text_decode_graph_query_geometry", None)
    if not callable(geometry_hook):
        return None
    backend = resolve_paged_decode_graph_backend(attention_backend_name)
    if backend is None:
        return None
    caps = backend.capabilities()
    multiple = int(getattr(caps, "paged_block_size_multiple", 1) or 1)
    if int(kv_pool.block_size) % max(1, multiple) != 0:
        return None
    num_q_heads, scale, q_dtype = geometry_hook()
    num_blocks = int(num_blocks)

    def prepare(state: TextDecodeGraphState, ctx: Any) -> None:
        if before is not None:
            before(state, ctx)
        if hasattr(backend, "prepare_paged_decode_cuda_graph"):
            prepare_paged_decode_graph_backend(
                state,
                backend=backend,
                num_q_heads=int(num_q_heads),
                num_kv_heads=int(kv_pool.n_kv),
                head_dim=int(kv_pool.head_dim),
                page_size=int(kv_pool.block_size),
                q_dtype=q_dtype,
                kv_dtype=kv_pool.k.dtype,
                scale=scale,
                max_indices=num_blocks * int(state.batch_size),
            )

    return prepare


def prepare_paged_decode_graph_backend(
    state: TextDecodeGraphState,
    *,
    backend: Any,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    scale: float | None,
    max_indices: int,
) -> None:
    """Refill ``backend``'s graph-decode plan buffers for the pending replay.

    Called from the runner's ``prepare_backend`` hook (outside the captured
    region) after ``copy_text_decode_graph_inputs`` has refreshed ``state``'s
    static block table + cache lengths. It re-plans the graph decode wrapper bound
    to ``state.metadata`` from those tensors so the captured ``wrapper.run`` reads
    the current pages/lengths — the mechanism that makes one capture correct across
    growth and across requests.
    """

    backend.prepare_paged_decode_cuda_graph(
        state.metadata,
        batch_size=int(state.batch_size),
        max_indices=int(max_indices),
        num_q_heads=int(num_q_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        page_size=int(page_size),
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        scale=scale,
    )


from .text_prefill import (  # noqa: E402
    PrefillCudaGraphRunner,
    TextInitialPrefillGraphState,
    copy_text_initial_prefill_graph_inputs,
    make_text_initial_prefill_graph_state,
)
