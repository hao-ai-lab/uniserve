"""CUDA graph plumbing for text initial prefill."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch

from ..contracts.forward_context import ForwardContext, TextAttentionMetadata, use_forward_context
from ..contracts.forward_mode import ForwardMode
from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import ceil_div
from ..runtime.kv_pool import PagedKVPool
from ..runtime.paged_text_cache import BatchedPagedRequestCache
from .cuda_graph_base import (
    _DEFAULT_METRIC_PREFIX,
    _DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
    GraphEvent,
    _GraphRunnerBase,
    _normalize_token_buckets,
    _parse_positive_int_csv,
    record_graph_stats,
)


@dataclass
class TextInitialPrefillGraphState:
    """Static buffers bound into an initial-prefill graph bucket."""

    num_tokens: int
    batch_size: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    query_lens: torch.Tensor
    kv_seqlens: torch.Tensor
    cu_seqlens: torch.Tensor
    last_token_indices: torch.Tensor
    graph: torch.cuda.CUDAGraph
    cache: BatchedPagedRequestCache
    metadata: TextAttentionMetadata
    logits: torch.Tensor | None = None


class PrefillCudaGraphRunner(_GraphRunnerBase):
    """Own per-token-bucket initial-prefill graph state and its lifecycle."""

    def __init__(
        self,
        *,
        name: str,
        default_enabled: bool = False,
        default_warmup: bool = False,
        default_warmup_token_buckets: tuple[int, ...] = _DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
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
        self.token_bucket_parser = token_bucket_parser
        self.metric_prefix = str(metric_prefix)
        self.logger = logger
        self.states: dict[tuple[int, int], TextInitialPrefillGraphState] = {}
        self.disabled: set[tuple[int, int]] = set()
        self._capture_pool: Any = None
        self._graph_input_buffer_pool: dict[tuple[str, str, str], torch.Tensor] = {}

    def warmup_token_buckets(self) -> tuple[int, ...]:
        return self.default_warmup_token_buckets

    def warmup_capture_token_buckets(self) -> tuple[int, ...]:
        return tuple(reversed(self.warmup_token_buckets()))

    def state_key(self, num_tokens: int, batch_size: int) -> tuple[int, int]:
        return (int(num_tokens), int(batch_size))

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

    def can_use(self, num_tokens: int, batch_size: int = 1) -> bool:
        key = self.state_key(num_tokens, batch_size)
        return self.enabled() and key[0] > 0 and key[1] > 0 and key not in self.disabled

    def disable(self, num_tokens: int, exc: BaseException, *, phase: str, batch_size: int = 1) -> None:
        key = self.state_key(num_tokens, batch_size)
        self.disabled.add(key)
        self.states.pop(key, None)
        if self.logger is not None:
            self.logger.warning(
                "%s %s initial-prefill CUDA graph for %s tokens x %s rows: %s",
                phase,
                self.name,
                key[0],
                key[1],
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
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: TextAttentionMetadata,
        last_token_indices: torch.Tensor,
        raw_num_tokens: int,
        ctx: Any,
        forward_fn: Callable[[TextInitialPrefillGraphState], torch.Tensor],
    ) -> TextInitialPrefillGraphState:
        """Capture an initial-prefill token bucket bound to the model forward."""

        device = input_ids.device
        state = make_text_initial_prefill_graph_state(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            num_tokens=int(num_tokens),
            batch_size=int(batch_size),
            device=device,
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

        def reset_append_plan(capture_state: TextInitialPrefillGraphState) -> None:
            capture_state.cache._append_plan = None

        return self._capture_graph_state(
            device=device,
            state=state,
            run=run,
            copy_inputs=copy_inputs,
            before_run=reset_append_plan,
        )

    def maybe_run(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        num_tokens: int,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: TextAttentionMetadata,
        last_token_indices: torch.Tensor,
        raw_num_tokens: int,
        ctx: Any,
        forward_fn: Callable[[TextInitialPrefillGraphState], torch.Tensor],
    ) -> torch.Tensor | None:
        """Capture-or-replay an initial-prefill bucket; ``None`` on miss/fallback."""

        graph_key = self.state_key(num_tokens, batch_size)

        def capture() -> TextInitialPrefillGraphState:
            return self.capture(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                num_tokens=num_tokens,
                batch_size=batch_size,
                input_ids=input_ids,
                positions=positions,
                attention_metadata=attention_metadata,
                last_token_indices=last_token_indices,
                raw_num_tokens=raw_num_tokens,
                ctx=ctx,
                forward_fn=forward_fn,
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
            return state.logits

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
            disable=lambda exc: self.disable(num_tokens, exc, phase="disabling", batch_size=batch_size),
            capture_metric=f"{self.metric_prefix}prefill_graph_capture",
            input_copy_metric=f"{self.metric_prefix}prefill_graph_input_copy",
            replay_metric=f"{self.metric_prefix}prefill_graph_replay_launch",
        )

    def warmup(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        forward_fn: Callable[[TextInitialPrefillGraphState], torch.Tensor],
    ) -> None:
        """Pre-capture single-row initial-prefill token buckets ahead of serving."""

        max_tokens = int(num_blocks) * int(block_size)
        ctx = ForwardContext(attention_backend_name="auto")

        def key_for(num_tokens: int) -> tuple[int, int]:
            return self.state_key(int(num_tokens), 1)

        def should_skip(num_tokens: int) -> bool:
            num_tokens = int(num_tokens)
            if num_tokens <= 0 or num_tokens > max_tokens:
                return True
            return ceil_div(num_tokens, block_size) > int(num_blocks)

        def capture_bucket(
            num_tokens: int,
            capture_ctx: ForwardContext,
        ) -> TextInitialPrefillGraphState:
            num_tokens = int(num_tokens)
            inputs = _synthetic_initial_prefill_inputs(
                kv_pool=kv_pool,
                num_tokens=num_tokens,
                block_size=block_size,
                device=device,
            )
            return self.capture(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                num_tokens=num_tokens,
                batch_size=1,
                input_ids=inputs.input_ids,
                positions=inputs.positions,
                attention_metadata=inputs.metadata,
                last_token_indices=inputs.last_token_indices,
                raw_num_tokens=num_tokens,
                ctx=capture_ctx,
                forward_fn=forward_fn,
            )

        def copy_inputs(num_tokens: int, state: TextInitialPrefillGraphState) -> None:
            num_tokens = int(num_tokens)
            metadata = _synthetic_initial_prefill_metadata(state.cache, num_tokens, device)
            copy_text_initial_prefill_graph_inputs(
                state,
                input_ids=torch.zeros(num_tokens, dtype=torch.long, device=device),
                positions=torch.arange(num_tokens, dtype=torch.long, device=device),
                attention_metadata=metadata,
                raw_num_tokens=num_tokens,
                last_token_indices=torch.tensor([num_tokens - 1], dtype=torch.long, device=device),
            )

        self._warmup_capture_buckets(
            device=device,
            ctx=ctx,
            buckets=self.warmup_capture_token_buckets,
            key_for=key_for,
            should_skip=should_skip,
            capture_bucket=capture_bucket,
            copy_inputs=copy_inputs,
            replay=lambda state: state.graph.replay(),
            disable=lambda num_tokens, exc: self.disable(
                int(num_tokens), exc, phase="skipping warmup for", batch_size=1
            ),
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
    block_size: int,
    device: torch.device,
) -> _SyntheticInitialPrefillInputs:
    block_count = ceil_div(num_tokens, block_size)
    cache = BatchedPagedRequestCache(kv_pool, [list(range(block_count))], [0])
    return _SyntheticInitialPrefillInputs(
        input_ids=torch.zeros(num_tokens, dtype=torch.long, device=device),
        positions=torch.arange(num_tokens, dtype=torch.long, device=device),
        metadata=_synthetic_initial_prefill_metadata(cache, num_tokens, device),
        last_token_indices=torch.tensor([num_tokens - 1], dtype=torch.long, device=device),
    )


def _synthetic_initial_prefill_metadata(
    cache: BatchedPagedRequestCache,
    num_tokens: int,
    device: torch.device,
) -> TextAttentionMetadata:
    query_lens = torch.full((1,), num_tokens, dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    return TextAttentionMetadata(
        cache=cache,
        block_table=cache.block_table(device=device),
        cache_seqlens=cache.cache_seqlens(device=device),
        cache_seqlens_cpu=(0,),
        query_lens=query_lens,
        query_lens_cpu=(num_tokens,),
        kv_seqlens=query_lens,
        kv_seqlens_cpu=(num_tokens,),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=num_tokens,
        mode=ForwardMode.EXTEND,
    )


def make_text_initial_prefill_graph_state(
    *,
    kv_pool: PagedKVPool,
    num_blocks: int,
    num_tokens: int,
    batch_size: int = 1,
    device: torch.device | str,
) -> TextInitialPrefillGraphState:
    """Allocate fixed buffers for an initial-prefill token bucket."""

    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        raise invalid_descriptor("initial-prefill graph token bucket must be positive")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise invalid_descriptor("initial-prefill graph batch size must be positive")
    device = torch.device(device)
    max_blocks_per_seq = max(
        1,
        min(int(num_blocks), ceil_div(num_tokens, kv_pool.block_size)),
    )
    graph_cache = BatchedPagedRequestCache(kv_pool, [[] for _ in range(batch_size)], [0] * batch_size)
    query_lens_cpu = (num_tokens,) + tuple(0 for _ in range(batch_size - 1))
    state = TextInitialPrefillGraphState(
        num_tokens=num_tokens,
        batch_size=batch_size,
        input_ids=torch.empty(num_tokens, dtype=torch.long, device=device),
        positions=torch.empty(num_tokens, dtype=torch.long, device=device),
        block_table=torch.empty((batch_size, max_blocks_per_seq), dtype=torch.int32, device=device),
        cache_seqlens=torch.zeros(batch_size, dtype=torch.int32, device=device),
        query_lens=torch.zeros(batch_size, dtype=torch.int32, device=device),
        kv_seqlens=torch.zeros(batch_size, dtype=torch.int32, device=device),
        cu_seqlens=torch.zeros(batch_size + 1, dtype=torch.int32, device=device),
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
            max_seqlen_k=num_tokens,
            mode=ForwardMode.EXTEND,
        ),
    )
    state.metadata = replace(
        state.metadata,
        block_table=state.block_table,
        cache_seqlens=state.cache_seqlens,
        query_lens=state.query_lens,
        kv_seqlens=state.kv_seqlens,
        cu_seqlens_q=state.cu_seqlens,
        cu_seqlens_k=state.cu_seqlens,
    )
    state.cache_seqlens.zero_()
    state.query_lens.zero_()
    state.query_lens[0] = num_tokens
    state.kv_seqlens.zero_()
    state.kv_seqlens[0] = num_tokens
    state.cu_seqlens.zero_()
    state.cu_seqlens[1:] = num_tokens
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
    """Refresh dynamic tensors feeding a captured initial-prefill graph."""

    raw_num_tokens = int(raw_num_tokens)
    if raw_num_tokens <= 0 or raw_num_tokens > state.num_tokens:
        raise invalid_descriptor("initial-prefill graph raw token count mismatch")
    if int(input_ids.numel()) > state.num_tokens or int(positions.numel()) > state.num_tokens:
        raise invalid_descriptor("initial-prefill graph input token bucket exceeded")
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
        raise invalid_descriptor("initial-prefill CUDA graph block table is missing")
    if int(block_table.shape[0]) != state.batch_size:
        raise invalid_descriptor("initial-prefill CUDA graph row count mismatch")
    if int(block_table.shape[1]) > int(state.block_table.shape[1]):
        raise invalid_descriptor("initial-prefill CUDA graph block-table width exceeded")
    state.block_table[:, : block_table.shape[1]].copy_(
        block_table.to(dtype=torch.int32),
        non_blocking=True,
    )
    if int(block_table.shape[1]) < int(state.block_table.shape[1]):
        state.block_table[:, block_table.shape[1] :].zero_()
    state.cache_seqlens.zero_()
    raw_lens = tuple(int(length) for length in getattr(attention_metadata, "query_lens_cpu", ()) or ())
    if len(raw_lens) != state.batch_size:
        raise invalid_descriptor("initial-prefill CUDA graph query lengths mismatch")
    if sum(raw_lens) != raw_num_tokens:
        raise invalid_descriptor("initial-prefill CUDA graph raw token count does not match query lengths")
    graph_lens = list(raw_lens)
    graph_lens[-1] += int(state.num_tokens) - raw_num_tokens
    graph_lens_tuple = tuple(int(length) for length in graph_lens)
    query_lens = attention_metadata.query_lens
    if not isinstance(query_lens, torch.Tensor):
        raise invalid_descriptor("initial-prefill CUDA graph query lens tensor is missing")
    if int(query_lens.numel()) != state.batch_size:
        raise invalid_descriptor("initial-prefill CUDA graph query lens tensor count mismatch")
    state.query_lens.copy_(query_lens.to(dtype=torch.int32), non_blocking=True)
    if state.num_tokens > raw_num_tokens:
        state.query_lens[-1].add_(int(state.num_tokens) - raw_num_tokens)
    state.kv_seqlens.copy_(state.query_lens, non_blocking=True)
    state.cu_seqlens[0] = 0
    torch.cumsum(state.query_lens, dim=0, out=state.cu_seqlens[1:])
    state.metadata.cache_seqlens_cpu = tuple(0 for _ in range(state.batch_size))
    state.metadata.query_lens_cpu = graph_lens_tuple
    state.metadata.kv_seqlens_cpu = graph_lens_tuple
    if last_token_indices is None:
        if state.batch_size != 1:
            raise invalid_descriptor("multi-row initial-prefill CUDA graph requires last-token indices")
        state.last_token_indices[0] = raw_num_tokens - 1
    else:
        flat_indices = last_token_indices.reshape(-1)
        if int(flat_indices.numel()) != state.batch_size:
            raise invalid_descriptor("initial-prefill CUDA graph last-token index count mismatch")
        state.last_token_indices.copy_(flat_indices.to(dtype=torch.long), non_blocking=True)
