"""CUDA graph plumbing for text initial prefill."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch

from ....contracts.forward_context import ForwardContext, TextAttentionMetadata, use_forward_context
from ....contracts.forward_mode import ForwardMode
from ....foundation.errors import invalid_descriptor
from ....foundation.sizing import ceil_div
from ....runtime.kv_pool import PagedKVPool
from ....runtime.paged_text_cache import BatchedPagedRequestCache
from .base import (
    _DEFAULT_METRIC_PREFIX,
    _DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
    GraphEvent,
    _GraphRunnerBase,
    record_graph_stats,
)

_DEFAULT_PREFILL_GRAPH_BATCH_SIZES = (1, 2, 4, 8)


def _reset_append_plan(state: "TextInitialPrefillGraphState") -> None:
    """Drop the cache's staged append plan so capture/replay re-derives it."""

    state.cache._append_plan = None


@dataclass
class TextInitialPrefillGraphState:
    """Static buffers bound into a text prefill graph bucket."""

    num_tokens: int
    max_kv_tokens: int
    batch_size: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    query_lens: torch.Tensor
    kv_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    last_token_indices: torch.Tensor
    graph: torch.cuda.CUDAGraph
    cache: BatchedPagedRequestCache
    metadata: TextAttentionMetadata
    logits: torch.Tensor | None = None
    release_backend: Callable[[], None] | None = None


class PrefillCudaGraphRunner(_GraphRunnerBase):
    """Own per-token-bucket initial-prefill graph state and its lifecycle."""

    def __init__(
        self,
        *,
        name: str,
        default_enabled: bool = False,
        default_warmup: bool = False,
        default_warmup_token_buckets: tuple[int, ...] = _DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
        default_warmup_batch_sizes: tuple[int, ...] = _DEFAULT_PREFILL_GRAPH_BATCH_SIZES,
        token_bucket_parser: Callable[[str], tuple[int, ...]] | None = None,
        metric_prefix: str = _DEFAULT_METRIC_PREFIX,
        logger: Any = None,
    ) -> None:
        self.name = str(name)
        self.default_enabled = bool(default_enabled)
        self.default_warmup = bool(default_warmup)
        self.default_warmup_token_buckets = tuple(
            sorted({int(size) for size in default_warmup_token_buckets if int(size) > 0})
        )
        self.default_warmup_batch_sizes = tuple(
            sorted({int(size) for size in default_warmup_batch_sizes if int(size) > 0})
        )
        self.token_bucket_parser = token_bucket_parser
        self.metric_prefix = str(metric_prefix)
        self.logger = logger
        self.states: dict[tuple[int, int, int], TextInitialPrefillGraphState] = {}
        self.disabled: set[tuple[int, int, int]] = set()
        self._capture_pool: Any = None
        self._graph_input_buffer_pool: dict[tuple[str, str, str], torch.Tensor] = {}

    def warmup_token_buckets(self) -> tuple[int, ...]:
        return self.default_warmup_token_buckets

    def warmup_capture_token_buckets(self) -> tuple[int, ...]:
        return tuple(reversed(self.warmup_token_buckets()))

    def warmup_batch_sizes(self) -> tuple[int, ...]:
        return self.default_warmup_batch_sizes

    def warmup_capture_batch_sizes(self) -> tuple[int, ...]:
        return tuple(reversed(self.warmup_batch_sizes()))

    def state_key(self, num_tokens: int, batch_size: int, max_kv_tokens: int) -> tuple[int, int, int]:
        return (int(num_tokens), int(batch_size), int(max_kv_tokens))

    def has_state(self, num_tokens: int, batch_size: int, max_kv_tokens: int) -> bool:
        return self.state_key(num_tokens, batch_size, max_kv_tokens) in self.states

    def bucket_num_tokens(self, num_tokens: int) -> int:
        num_tokens = int(num_tokens)
        candidates = [
            int(size)
            for size in self.warmup_token_buckets()
            if int(size) >= num_tokens
        ]
        if candidates:
            return min(candidates)
        return num_tokens

    def bucket_batch_size(self, batch_size: int, *, num_tokens: int | None = None) -> int:
        batch_size = int(batch_size)
        prefer_reusable = num_tokens is not None
        candidates = [
            int(size)
            for size in self.warmup_batch_sizes()
            if int(size) >= batch_size
        ]
        if candidates:
            return max(candidates) if prefer_reusable else min(candidates)
        return batch_size

    def resolve_batch_size(self, num_tokens: int, batch_size: int, max_kv_tokens: int) -> int:
        batch_size = int(batch_size)
        candidates = [
            int(state_batch_size)
            for state_tokens, state_batch_size, state_kv_tokens in self.states
            if int(state_tokens) == int(num_tokens)
            and int(state_batch_size) >= batch_size
            and int(state_kv_tokens) == int(max_kv_tokens)
            and (int(state_tokens), int(state_batch_size), int(state_kv_tokens)) not in self.disabled
        ]
        if candidates:
            return min(candidates)
        return self.bucket_batch_size(batch_size, num_tokens=num_tokens)

    def bucket_kv_tokens(self, max_kv_tokens: int, *, max_context_len: int = 0) -> int:
        max_kv_tokens = int(max_kv_tokens)
        context_len = int(max_context_len)
        if context_len > 0 and max_kv_tokens <= context_len:
            return context_len
        return self.bucket_num_tokens(max_kv_tokens)

    def can_use(self, num_tokens: int, batch_size: int = 1, max_kv_tokens: int | None = None) -> bool:
        max_kv_tokens = int(max_kv_tokens if max_kv_tokens is not None else num_tokens)
        key = self.state_key(num_tokens, batch_size, max_kv_tokens)
        return (
            self.enabled()
            and key[0] > 0
            and key[1] > 0
            and key[2] >= key[0]
            and key not in self.disabled
        )

    def disable(
        self,
        num_tokens: int,
        exc: BaseException,
        *,
        phase: str,
        batch_size: int = 1,
        max_kv_tokens: int | None = None,
    ) -> None:
        max_kv_tokens = int(max_kv_tokens if max_kv_tokens is not None else num_tokens)
        key = self.state_key(num_tokens, batch_size, max_kv_tokens)
        self.disabled.add(key)
        state = self.states.pop(key, None)
        if state is not None and state.release_backend is not None:
            state.release_backend()
        if self.logger is not None:
            self.logger.warning(
                "%s %s prefill CUDA graph for %s query tokens x %s rows x %s kv tokens: %s",
                phase,
                self.name,
                key[0],
                key[1],
                key[2],
                exc,
            )

    @staticmethod
    def record_stats(
        ctx: Any,
        event: str,
        *,
        mode: ForwardMode,
        raw_tokens: int,
        padded_tokens: int,
    ) -> None:
        record_graph_stats(
            ctx,
            event,
            mode=mode,
            unpadded_tokens=raw_tokens,
            padded_tokens=padded_tokens,
        )

    def capture(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        num_tokens: int,
        max_kv_tokens: int,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: TextAttentionMetadata,
        last_token_indices: torch.Tensor,
        raw_num_tokens: int,
        ctx: Any,
        forward_fn: Callable[[TextInitialPrefillGraphState], torch.Tensor],
        prepare_backend: Callable[[TextInitialPrefillGraphState, Any], None] | None = None,
    ) -> TextInitialPrefillGraphState:
        """Capture an initial-prefill token bucket bound to the model forward."""

        device = input_ids.device
        state = make_text_initial_prefill_graph_state(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            num_tokens=int(num_tokens),
            max_kv_tokens=int(max_kv_tokens),
            batch_size=int(batch_size),
            device=device,
            max_context_len=int(getattr(attention_metadata, "max_context_len", 0) or 0),
        )
        copy_text_initial_prefill_graph_inputs(
            state,
            input_ids=input_ids,
            positions=positions,
            attention_metadata=attention_metadata,
            raw_num_tokens=raw_num_tokens,
            last_token_indices=last_token_indices,
        )
        graph_ctx = replace(ctx, attention_metadata=state.metadata, stats=None)
        bind_paged_prefill_graph_wrapper(state, graph_ctx)

        def run() -> torch.Tensor:
            with use_forward_context(graph_ctx):
                return forward_fn(state)

        def copy_inputs(capture_state: TextInitialPrefillGraphState) -> None:
            copy_text_initial_prefill_graph_inputs(
                capture_state,
                input_ids=input_ids,
                positions=positions,
                attention_metadata=attention_metadata,
                raw_num_tokens=raw_num_tokens,
                last_token_indices=last_token_indices,
            )

        def prepare(capture_state: TextInitialPrefillGraphState) -> None:
            _reset_append_plan(capture_state)
            if prepare_backend is not None:
                prepare_backend(capture_state, graph_ctx)

        try:
            captured = self._capture_graph_state(
                device=device,
                state=state,
                run=run,
                copy_inputs=copy_inputs,
                before_run=prepare,
            )
            assert_paged_prefill_graph_wrapper_planned(captured, graph_ctx)
            return captured
        except BaseException:
            release = state.release_backend
            state.release_backend = None
            if callable(release):
                release()
            raise

    def _prepare_prefill_graph(
        self,
        state: TextInitialPrefillGraphState,
        ctx: Any,
        prepare_backend: Callable[[TextInitialPrefillGraphState, Any], None] | None,
    ) -> None:
        _reset_append_plan(state)
        if prepare_backend is not None:
            prepare_backend(state, ctx)

    def maybe_run(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        num_tokens: int,
        max_kv_tokens: int,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: TextAttentionMetadata,
        last_token_indices: torch.Tensor,
        raw_num_tokens: int,
        ctx: Any,
        forward_fn: Callable[[TextInitialPrefillGraphState], torch.Tensor],
        prepare_backend: Callable[[TextInitialPrefillGraphState, Any], None] | None = None,
    ) -> torch.Tensor | None:
        """Capture-or-replay an initial-prefill bucket; ``None`` on miss/fallback."""

        graph_batch_size = self.resolve_batch_size(num_tokens, batch_size, max_kv_tokens)
        graph_key = self.state_key(num_tokens, graph_batch_size, max_kv_tokens)
        if graph_key not in self.states and self.warmup_enabled():
            self.record_stats(
                ctx,
                GraphEvent.MISS,
                mode=ForwardMode.EXTEND,
                raw_tokens=raw_num_tokens,
                padded_tokens=num_tokens,
            )
            return None

        def capture() -> TextInitialPrefillGraphState:
            return self.capture(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                num_tokens=num_tokens,
                max_kv_tokens=max_kv_tokens,
                batch_size=graph_batch_size,
                input_ids=input_ids,
                positions=positions,
                attention_metadata=attention_metadata,
                last_token_indices=last_token_indices,
                raw_num_tokens=raw_num_tokens,
                ctx=ctx,
                forward_fn=forward_fn,
                prepare_backend=prepare_backend,
            )

        def copy_inputs(state: TextInitialPrefillGraphState) -> None:
            copy_text_initial_prefill_graph_inputs(
                state,
                input_ids=input_ids,
                positions=positions,
                attention_metadata=attention_metadata,
                raw_num_tokens=raw_num_tokens,
                last_token_indices=last_token_indices,
            )

        def replay(state: TextInitialPrefillGraphState) -> torch.Tensor:
            state.graph.replay()
            return _slice_prefill_graph_logits(state.logits, batch_size)

        def record(event: GraphEvent) -> None:
            self.record_stats(
                ctx,
                event,
                mode=ForwardMode.EXTEND,
                raw_tokens=raw_num_tokens,
                padded_tokens=num_tokens,
            )

        return self._capture_or_replay(
            key=graph_key,
            ctx=ctx,
            capture=capture,
            copy_inputs=copy_inputs,
            replay=replay,
            record=record,
            disable=lambda exc: self.disable(
                num_tokens,
                exc,
                phase="disabling",
                batch_size=graph_batch_size,
                max_kv_tokens=max_kv_tokens,
            ),
            capture_metric=f"{self.metric_prefix}prefill_graph_capture",
            input_copy_metric=f"{self.metric_prefix}prefill_graph_input_copy",
            replay_metric=f"{self.metric_prefix}prefill_graph_replay_launch",
            after_copy=lambda state: self._prepare_prefill_graph(state, ctx, prepare_backend),
            after_copy_metric=f"{self.metric_prefix}prefill_graph_attention_prepare",
        )

    def warmup(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        max_context_len: int = 0,
        attention_backend_name: str | None = "auto",
        forward_fn: Callable[[TextInitialPrefillGraphState], torch.Tensor],
        prepare_backend: Callable[[TextInitialPrefillGraphState, Any], None] | None = None,
    ) -> None:
        """Pre-capture initial-prefill token/batch buckets ahead of serving."""

        max_tokens = int(num_blocks) * int(block_size)
        ctx = ForwardContext(attention_backend_name=attention_backend_name or "auto")

        def max_kv_bucket() -> int:
            context_len = int(max_context_len)
            if context_len > 0:
                return min(max_tokens, context_len)
            return 0

        def key_for(bucket: tuple[int, int]) -> tuple[int, int, int]:
            num_tokens, batch_size = bucket
            kv_tokens = max_kv_bucket() or int(num_tokens)
            return self.state_key(int(num_tokens), int(batch_size), int(kv_tokens))

        def should_skip(bucket: tuple[int, int]) -> bool:
            num_tokens, batch_size = bucket
            num_tokens = int(num_tokens)
            batch_size = int(batch_size)
            if num_tokens <= 0 or batch_size <= 0 or num_tokens > max_tokens:
                return True
            return ceil_div(num_tokens, block_size) > int(num_blocks)

        def capture_bucket(
            bucket: tuple[int, int],
            capture_ctx: ForwardContext,
        ) -> TextInitialPrefillGraphState:
            num_tokens, batch_size = bucket
            num_tokens = int(num_tokens)
            batch_size = int(batch_size)
            kv_tokens = max_kv_bucket() or num_tokens
            inputs = _synthetic_initial_prefill_inputs(
                kv_pool=kv_pool,
                num_tokens=num_tokens,
                max_kv_tokens=kv_tokens,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
                max_context_len=max_context_len,
            )
            return self.capture(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                num_tokens=num_tokens,
                max_kv_tokens=kv_tokens,
                batch_size=batch_size,
                input_ids=inputs.input_ids,
                positions=inputs.positions,
                attention_metadata=inputs.metadata,
                last_token_indices=inputs.last_token_indices,
                raw_num_tokens=num_tokens,
                ctx=capture_ctx,
                forward_fn=forward_fn,
                prepare_backend=prepare_backend,
            )

        def copy_inputs(bucket: tuple[int, int], state: TextInitialPrefillGraphState) -> None:
            num_tokens = int(bucket[0])
            inputs = _synthetic_initial_prefill_inputs(
                kv_pool=kv_pool,
                num_tokens=num_tokens,
                max_kv_tokens=state.max_kv_tokens,
                batch_size=state.batch_size,
                block_size=block_size,
                device=device,
                max_context_len=max_context_len,
            )
            copy_text_initial_prefill_graph_inputs(
                state,
                input_ids=inputs.input_ids,
                positions=inputs.positions,
                attention_metadata=inputs.metadata,
                raw_num_tokens=num_tokens,
                last_token_indices=inputs.last_token_indices,
            )

        def replay_state(state: TextInitialPrefillGraphState) -> None:
            self._prepare_prefill_graph(state, ctx, prepare_backend)
            state.graph.replay()

        self._warmup_capture_buckets(
            device=device,
            ctx=ctx,
            buckets=self.warmup_capture_buckets,
            key_for=key_for,
            should_skip=should_skip,
            capture_bucket=capture_bucket,
            copy_inputs=copy_inputs,
            replay=replay_state,
            disable=lambda bucket, exc: self.disable(
                int(bucket[0]),
                exc,
                phase="skipping warmup for",
                batch_size=int(bucket[1]),
                max_kv_tokens=max_kv_bucket() or int(bucket[0]),
            ),
        )

    def warmup_capture_buckets(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(num_tokens), self.bucket_batch_size(1, num_tokens=int(num_tokens)))
            for num_tokens in self.warmup_capture_token_buckets()
        )


@dataclass(frozen=True)
class _SyntheticInitialPrefillInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    metadata: TextAttentionMetadata
    last_token_indices: torch.Tensor


def _synthetic_initial_prefill_inputs(
    *,
    kv_pool: PagedKVPool,
    num_tokens: int,
    max_kv_tokens: int | None = None,
    batch_size: int = 1,
    block_size: int,
    device: torch.device,
    max_context_len: int = 0,
) -> _SyntheticInitialPrefillInputs:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise invalid_descriptor("synthetic prefill graph batch size must be positive")
    query_lens = _synthetic_query_lens(int(num_tokens), batch_size)
    block_ids_by_row: list[list[int]] = []
    next_block = 0
    for query_len in query_lens:
        block_count = ceil_div(query_len, block_size)
        block_ids_by_row.append(list(range(next_block, next_block + block_count)))
        next_block += block_count
    cache = BatchedPagedRequestCache(kv_pool, block_ids_by_row, [0] * batch_size)
    positions = torch.cat(
        [torch.arange(length, dtype=torch.long, device=device) for length in query_lens],
        dim=0,
    )
    last_indices = []
    offset = 0
    for length in query_lens:
        last_indices.append(offset + int(length) - 1 if int(length) > 0 else 0)
        offset += int(length)
    return _SyntheticInitialPrefillInputs(
        input_ids=torch.zeros(num_tokens, dtype=torch.long, device=device),
        positions=positions,
        metadata=_synthetic_initial_prefill_metadata(
            cache,
            num_tokens,
            batch_size,
            device,
            max_context_len=max_context_len,
        ),
        last_token_indices=torch.tensor(last_indices, dtype=torch.long, device=device),
    )


def _synthetic_initial_prefill_metadata(
    cache: BatchedPagedRequestCache,
    num_tokens: int,
    batch_size: int,
    device: torch.device,
    max_context_len: int = 0,
) -> TextAttentionMetadata:
    query_lens_cpu = _synthetic_query_lens(int(num_tokens), int(batch_size))
    query_lens = torch.tensor(query_lens_cpu, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(query_lens, dim=0).to(torch.int32)]
    )
    return TextAttentionMetadata(
        cache=cache,
        block_table=cache.block_table(device=device),
        cache_seqlens=cache.cache_seqlens(device=device),
        cache_seqlens_cpu=tuple(0 for _ in range(int(batch_size))),
        query_lens=query_lens,
        query_lens_cpu=query_lens_cpu,
        kv_seqlens=query_lens,
        kv_seqlens_cpu=query_lens_cpu,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(query_lens_cpu, default=0),
        max_seqlen_k=max(query_lens_cpu, default=0),
        max_context_len=int(max_context_len),
        mode=ForwardMode.EXTEND,
    )


def _synthetic_query_lens(num_tokens: int, batch_size: int) -> tuple[int, ...]:
    num_tokens = int(num_tokens)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise invalid_descriptor("synthetic prefill graph batch size must be positive")
    if num_tokens <= 0:
        raise invalid_descriptor("synthetic prefill graph token count must be positive")
    return (num_tokens,) + tuple(0 for _ in range(batch_size - 1))


def make_text_initial_prefill_graph_state(
    *,
    kv_pool: PagedKVPool,
    num_blocks: int,
    num_tokens: int,
    max_kv_tokens: int | None = None,
    batch_size: int = 1,
    device: torch.device | str,
    max_context_len: int = 0,
) -> TextInitialPrefillGraphState:
    """Allocate fixed buffers for a text prefill graph bucket."""

    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        raise invalid_descriptor("prefill graph token bucket must be positive")
    max_kv_tokens = int(max_kv_tokens if max_kv_tokens is not None else num_tokens)
    if max_kv_tokens < num_tokens:
        raise invalid_descriptor("prefill graph KV bucket must cover the query bucket")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise invalid_descriptor("prefill graph batch size must be positive")
    device = torch.device(device)
    max_blocks_per_seq = max(
        1,
        min(int(num_blocks), ceil_div(max_kv_tokens, kv_pool.block_size)),
    )
    graph_cache = BatchedPagedRequestCache(kv_pool, [[] for _ in range(batch_size)], [0] * batch_size)
    query_lens_cpu = (num_tokens,) + tuple(0 for _ in range(batch_size - 1))
    state = TextInitialPrefillGraphState(
        num_tokens=num_tokens,
        max_kv_tokens=max_kv_tokens,
        batch_size=batch_size,
        input_ids=torch.empty(num_tokens, dtype=torch.long, device=device),
        positions=torch.empty(num_tokens, dtype=torch.long, device=device),
        block_table=torch.empty((batch_size, max_blocks_per_seq), dtype=torch.int32, device=device),
        cache_seqlens=torch.zeros(batch_size, dtype=torch.int32, device=device),
        query_lens=torch.zeros(batch_size, dtype=torch.int32, device=device),
        kv_seqlens=torch.zeros(batch_size, dtype=torch.int32, device=device),
        cu_seqlens_q=torch.zeros(batch_size + 1, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.zeros(batch_size + 1, dtype=torch.int32, device=device),
        last_token_indices=torch.zeros(batch_size, dtype=torch.long, device=device),
        graph=torch.cuda.CUDAGraph(),
        cache=graph_cache,
        metadata=TextAttentionMetadata(
            cache=graph_cache,
            block_table=None,
            cache_seqlens=None,
            cache_seqlens_cpu=tuple(0 for _ in range(batch_size)),
            query_lens=None,
            query_lens_cpu=query_lens_cpu,
            kv_seqlens=None,
            kv_seqlens_cpu=query_lens_cpu,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=num_tokens,
            max_seqlen_k=max_kv_tokens,
            max_context_len=int(max_context_len),
            mode=ForwardMode.EXTEND,
        ),
    )
    state.metadata = replace(
        state.metadata,
        block_table=state.block_table,
        cache_seqlens=state.cache_seqlens,
        query_lens=state.query_lens,
        kv_seqlens=state.kv_seqlens,
        cu_seqlens_q=state.cu_seqlens_q,
        cu_seqlens_k=state.cu_seqlens_k,
    )
    state.cache_seqlens.zero_()
    state.query_lens.zero_()
    state.query_lens[0] = num_tokens
    state.kv_seqlens.zero_()
    state.kv_seqlens[0] = num_tokens
    state.cu_seqlens_q.zero_()
    state.cu_seqlens_q[1:] = num_tokens
    state.cu_seqlens_k.zero_()
    state.cu_seqlens_k[1:] = num_tokens
    return state


def copy_text_initial_prefill_graph_inputs(
    state: TextInitialPrefillGraphState,
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    attention_metadata: TextAttentionMetadata,
    raw_num_tokens: int,
    last_token_indices: torch.Tensor | None = None,
) -> None:
    """Refresh dynamic tensors feeding a captured prefill graph."""

    raw_num_tokens = int(raw_num_tokens)
    if raw_num_tokens <= 0 or raw_num_tokens > state.num_tokens:
        raise invalid_descriptor("prefill graph raw token count mismatch")
    if int(input_ids.numel()) > state.num_tokens or int(positions.numel()) > state.num_tokens:
        raise invalid_descriptor("prefill graph input token bucket exceeded")
    flat_ids = input_ids.reshape(-1)
    flat_positions = positions.reshape(-1)
    state.input_ids[: flat_ids.numel()].copy_(flat_ids, non_blocking=True)
    state.positions[: flat_positions.numel()].copy_(flat_positions, non_blocking=True)
    if int(flat_ids.numel()) < state.num_tokens:
        state.input_ids[flat_ids.numel() :].zero_()
    if int(flat_positions.numel()) < state.num_tokens:
        state.positions[flat_positions.numel() :].zero_()
    block_table = attention_metadata.block_table
    if block_table is None:
        raise invalid_descriptor("prefill CUDA graph block table is missing")
    real_rows = int(block_table.shape[0])
    if real_rows <= 0 or real_rows > state.batch_size:
        raise invalid_descriptor("prefill CUDA graph row count mismatch")
    if int(block_table.shape[1]) > int(state.block_table.shape[1]):
        raise invalid_descriptor("prefill CUDA graph block-table width exceeded")
    state.block_table[:real_rows, : block_table.shape[1]].copy_(
        block_table.to(dtype=torch.int32),
        non_blocking=True,
    )
    if int(block_table.shape[1]) < int(state.block_table.shape[1]):
        state.block_table[:real_rows, block_table.shape[1] :].zero_()
    if real_rows < state.batch_size:
        state.block_table[real_rows:].zero_()
    cache_seqlens = attention_metadata.cache_seqlens
    if not isinstance(cache_seqlens, torch.Tensor):
        raise invalid_descriptor("prefill CUDA graph cache-seqlens tensor is missing")
    if int(cache_seqlens.numel()) != real_rows:
        raise invalid_descriptor("prefill CUDA graph cache-seqlens row count mismatch")
    state.cache_seqlens[:real_rows].copy_(cache_seqlens.to(dtype=torch.int32), non_blocking=True)
    if real_rows < state.batch_size:
        state.cache_seqlens[real_rows:].zero_()
    cache_lens_cpu = tuple(int(length) for length in getattr(attention_metadata, "cache_seqlens_cpu", ()) or ())
    if len(cache_lens_cpu) != real_rows:
        raise invalid_descriptor("prefill CUDA graph cache length count mismatch")
    graph_cache_lens_cpu = cache_lens_cpu + tuple(0 for _ in range(state.batch_size - real_rows))
    source_cache = getattr(attention_metadata, "cache", None)
    block_ids_by_row = getattr(source_cache, "block_ids_by_row", None)
    if block_ids_by_row is not None:
        graph_block_ids = [list(row) for row in block_ids_by_row] + [[] for _ in range(state.batch_size - real_rows)]
        state.cache.reset_rows(graph_block_ids, graph_cache_lens_cpu)
    raw_lens = tuple(int(length) for length in getattr(attention_metadata, "query_lens_cpu", ()) or ())
    if len(raw_lens) != real_rows:
        raise invalid_descriptor("prefill CUDA graph query lengths mismatch")
    if sum(raw_lens) != raw_num_tokens:
        raise invalid_descriptor("prefill CUDA graph raw token count does not match query lengths")
    graph_lens = list(raw_lens) + [0 for _ in range(state.batch_size - real_rows)]
    graph_lens[real_rows - 1] += int(state.num_tokens) - raw_num_tokens
    graph_lens_tuple = tuple(int(length) for length in graph_lens)
    kv_lens_tuple = tuple(int(base) + int(query) for base, query in zip(graph_cache_lens_cpu, graph_lens_tuple, strict=True))
    if max(kv_lens_tuple, default=0) > int(state.max_kv_tokens):
        raise invalid_descriptor("prefill CUDA graph KV bucket exceeded")
    query_lens = attention_metadata.query_lens
    if not isinstance(query_lens, torch.Tensor):
        raise invalid_descriptor("prefill CUDA graph query lens tensor is missing")
    if int(query_lens.numel()) != real_rows:
        raise invalid_descriptor("prefill CUDA graph query lens tensor count mismatch")
    state.query_lens[:real_rows].copy_(query_lens.to(dtype=torch.int32), non_blocking=True)
    if real_rows < state.batch_size:
        state.query_lens[real_rows:].zero_()
    if state.num_tokens > raw_num_tokens:
        state.query_lens[real_rows - 1].add_(int(state.num_tokens) - raw_num_tokens)
    state.kv_seqlens.copy_(state.cache_seqlens, non_blocking=True)
    state.kv_seqlens.add_(state.query_lens)
    state.cu_seqlens_q[:1].zero_()
    torch.cumsum(state.query_lens, dim=0, out=state.cu_seqlens_q[1:])
    state.cu_seqlens_k[:1].zero_()
    torch.cumsum(state.kv_seqlens, dim=0, out=state.cu_seqlens_k[1:])
    state.metadata.cache_seqlens_cpu = graph_cache_lens_cpu
    state.metadata.query_lens_cpu = graph_lens_tuple
    state.metadata.kv_seqlens_cpu = kv_lens_tuple
    state.metadata.max_seqlen_q = max(graph_lens_tuple, default=state.num_tokens)
    state.metadata.max_seqlen_k = state.max_kv_tokens
    state.metadata.max_context_len = int(
        getattr(attention_metadata, "max_context_len", 0) or state.metadata.max_context_len
    )
    if last_token_indices is None:
        if state.batch_size != 1:
            raise invalid_descriptor("multi-row prefill CUDA graph requires last-token indices")
        state.last_token_indices[:1].fill_(raw_num_tokens - 1)
    else:
        flat_indices = last_token_indices.reshape(-1)
        if int(flat_indices.numel()) != real_rows:
            raise invalid_descriptor("prefill CUDA graph last-token index count mismatch")
        state.last_token_indices[:real_rows].copy_(flat_indices.to(dtype=torch.long), non_blocking=True)
        if real_rows < state.batch_size:
            state.last_token_indices[real_rows:].zero_()


def resolve_paged_prefill_graph_prepare(
    *,
    owner: Any,
    kv_pool: PagedKVPool,
    attention_backend_name: str | None,
    before: Callable[[TextInitialPrefillGraphState, Any], None] | None = None,
) -> Callable[[TextInitialPrefillGraphState, Any], None] | None:
    """Build the per-replay prefill-graph prepare hook, or ``None`` to stay eager."""

    backend = _resolve_graph_prefill_backend(ForwardContext(attention_backend_name=attention_backend_name))
    if backend is None:
        return None
    prepare = getattr(backend, "prepare_paged_prefill_cuda_graph", None)
    if callable(prepare):
        geometry_hook = getattr(owner, "text_decode_graph_query_geometry", None)
        if not callable(geometry_hook):
            return None
        num_q_heads, scale, q_dtype = geometry_hook()

        def prepare_with_backend(state: TextInitialPrefillGraphState, ctx: Any) -> None:
            if before is not None:
                before(state, ctx)
            bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
            release = getattr(backend, "release_paged_prefill_graph_wrapper", None)
            if state.release_backend is None and callable(bind) and callable(release):
                bind(state.metadata, device=state.input_ids.device)
                state.release_backend = lambda: release(state.metadata)
            prepare_paged_prefill_graph_backend(
                state,
                backend=backend,
                num_q_heads=int(num_q_heads),
                num_kv_heads=int(kv_pool.n_kv),
                head_dim=int(kv_pool.head_dim),
                page_size=int(kv_pool.block_size),
                q_dtype=q_dtype,
                kv_dtype=kv_pool.k.dtype,
                causal=True,
                scale=scale,
            )

        return prepare_with_backend
    try:
        caps = backend.capabilities()
    except Exception:
        return None
    if not bool(getattr(caps, "paged_varlen_cuda_graph", False)):
        return None

    def prepare_direct_graph_backend(state: TextInitialPrefillGraphState, ctx: Any) -> None:
        if before is not None:
            before(state, ctx)

    return prepare_direct_graph_backend


def prepare_paged_prefill_graph_backend(
    state: TextInitialPrefillGraphState,
    *,
    backend: Any,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    causal: bool,
    scale: float | None,
) -> None:
    """Refresh ``backend``'s graph-prefill plan buffers for the pending replay."""

    backend.prepare_paged_prefill_cuda_graph(
        state.metadata,
        num_q_heads=int(num_q_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        page_size=int(page_size),
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        causal=bool(causal),
        scale=scale,
    )


def bind_paged_prefill_graph_wrapper(state: TextInitialPrefillGraphState, ctx: Any) -> None:
    """Bind a graph-scoped paged-prefill wrapper when the selected backend owns one."""

    backend = _resolve_graph_prefill_backend(ctx)
    bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
    release = getattr(backend, "release_paged_prefill_graph_wrapper", None)
    if not callable(bind) or not callable(release):
        return
    bind(state.metadata, device=state.input_ids.device)
    state.release_backend = lambda: release(state.metadata)


def assert_paged_prefill_graph_wrapper_planned(state: TextInitialPrefillGraphState, ctx: Any) -> None:
    if state.release_backend is None:
        return
    backend = _resolve_graph_prefill_backend(ctx)
    planned = getattr(backend, "paged_prefill_graph_wrapper_planned", None)
    if callable(planned) and not planned(state.metadata):
        raise invalid_descriptor("captured prefill forward did not plan the graph-scoped prefill backend")


def _resolve_graph_prefill_backend(ctx: Any) -> Any:
    backend = getattr(ctx, "attention_backend", None)
    if backend is not None:
        return backend
    try:
        from ....backends.attention import (
            get_attention_backend,
            has_attention_backend,
            normalize_attention_backend_name,
        )

        name = normalize_attention_backend_name(getattr(ctx, "attention_backend_name", None))
        if name == "auto" and has_attention_backend("trtllm_mha"):
            backend = get_attention_backend("trtllm_mha")
            if bool(getattr(backend.capabilities(), "available", True)):
                return backend
        if name == "trtllm_mha" and has_attention_backend("trtllm_mha"):
            backend = get_attention_backend("trtllm_mha")
            if bool(getattr(backend.capabilities(), "available", True)):
                return backend
            return None
        if name == "flashinfer" or (name == "auto" and has_attention_backend("flashinfer")):
            return get_attention_backend("flashinfer")
    except Exception:
        return None
    return None


def _slice_prefill_graph_logits(logits: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    if logits is None:
        return None
    batch_size = int(batch_size)
    if batch_size <= 0 or int(logits.shape[0]) == batch_size:
        return logits
    return logits[:batch_size]
