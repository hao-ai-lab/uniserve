"""Executor over the step and span graph implementations."""

from __future__ import annotations

import logging
from typing import Any

import torch

from uniserve_worker.contracts.forward_batch import ForwardBatch
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.foundation.runtime_config import get_execution_config
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import BatchedPagedRequestCache

from .span import Runner as Span
from .span import State as SpanState
from .span import resolve_prepare as resolve_span
from .step import Runner as Step
from .step import State as StepState
from .step import resolve_prepare as resolve_step

# ---------------------
# Graph executor
# ---------------------

logger = logging.getLogger(__name__)


class Executor:
    """Owns the single-step and variable-length graphs keyed on the system pool."""

    def __init__(
        self,
        *,
        kv_pool: "PagedKVPool",
        num_blocks: int,
        block_size: int,
        device: "torch.device",
        attention_preference: str | None = None,
        max_context_len: int = 0,
    ) -> None:
        self.kv_pool = kv_pool
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.device = device
        self.max_context_len = max(0, int(max_context_len))
        # Startup (warmup) has no forward context to read the backend name from;
        # per-forward calls prefer the context's resolved name.
        self.attention_preference = attention_preference
        runtime = get_execution_config()
        self._step = Step(
            name="step",
            default_enabled=runtime.cuda_graph,
            default_warmup=runtime.cuda_graph_warmup,
            default_warmup_batch_sizes=runtime.cuda_graph_warmup_batches,
            metric_prefix="",
            logger=logger,
        )
        self._span = Span(
            name="span",
            default_enabled=runtime.prefill_cuda_graph,
            default_warmup=runtime.prefill_cuda_graph_warmup,
            default_warmup_token_buckets=runtime.prefill_cuda_graph_warmup_tokens,
            metric_prefix="",
            logger=logger,
        )

    # ---- capture-or-replay --------------------------------------------------

    def maybe_run(
        self,
        model: Any,
        input_ids: "torch.Tensor",
        positions: "torch.Tensor",
        fb: ForwardBatch,
        ctx: Any,
    ) -> "torch.Tensor | None":
        """Replay (or capture) the graph for this forward; ``None`` to fall back."""

        plan = fb.attn_plan
        if not isinstance(getattr(plan, "residency_cache", None), BatchedPagedRequestCache):
            return None
        if getattr(input_ids, "device", None) is None or input_ids.device.type != "cuda":
            return None
        if fb.forward_mode == ForwardMode.DECODE:
            return self._maybe_step(model, input_ids, positions, fb, plan, ctx)
        if fb.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            # A mixed extend+decode group is shape-identical to a cached-prefix
            # extend group (flat varlen rows, per-row context lengths, per-row
            # last-token sampling), so it replays the same prefill buckets.
            return self._maybe_span(model, input_ids, positions, fb, plan, ctx)
        return None

    def _maybe_step(self, model, input_ids, positions, fb, plan, ctx):
        if not self._step.enabled():
            return None
        if input_ids.ndim != 2 or int(input_ids.shape[1]) != 1:
            return None
        batch_size = int(input_ids.shape[0])
        prepare_backend = self._step_prepare(
            model,
            attention_preference=getattr(ctx, "attention_preference", None)
            or self.attention_preference,
        )
        if prepare_backend is None:
            # A captured paged-step graph is only correct when the backend can
            # refill its plan buffers before every replay; without that, the
            # capture-time plan is baked in and later steps silently read stale
            # pages. Stay eager rather than capture a wrong graph.
            return None
        return self._step.maybe_run(
            kv_pool=self.kv_pool,
            num_blocks=self.num_blocks,
            batch_size=batch_size,
            input_ids=input_ids,
            positions=positions,
            attention_plan=plan,
            ctx=ctx,
            forward_fn=lambda state: self._step_forward(model, state),
            prepare_backend=prepare_backend,
        )

    def _step_prepare(
        self,
        model: Any,
        *,
        attention_preference: str | None,
    ) -> Any | None:
        return resolve_step(
            owner=model,
            kv_pool=self.kv_pool,
            num_blocks=self.num_blocks,
            attention_preference=attention_preference,
        )

    def _maybe_span(self, model, input_ids, positions, fb, plan, ctx):
        if not self._span.enabled():
            return None
        batch_size = int(fb.batch_size)
        if batch_size <= 0 or fb.last_token_indices is None:
            return None
        if any(fb.spec_token_ids):
            return None
        raw_tokens = int(fb.num_token_non_padded)
        padded_tokens = int(input_ids.numel())
        if raw_tokens <= 0 or padded_tokens < raw_tokens:
            return None
        max_kv_tokens = self._span.bucket_kv_tokens(
            _padded_max_tokens(
                plan,
                padded_tokens=padded_tokens,
                raw_tokens=raw_tokens,
                batch_size=batch_size,
            ),
            max_context_len=self.max_context_len,
        )
        if not self._span.can_use(
            padded_tokens,
            batch_size=batch_size,
            max_kv_tokens=max_kv_tokens,
        ):
            return None
        prepare_backend = self._span_prepare(
            model,
            attention_preference=getattr(ctx, "attention_preference", None)
            or getattr(self, "attention_preference", None),
        )
        if prepare_backend is None:
            return None
        return self._span.maybe_run(
            kv_pool=self.kv_pool,
            num_blocks=self.num_blocks,
            num_tokens=padded_tokens,
            max_kv_tokens=max_kv_tokens,
            batch_size=batch_size,
            input_ids=input_ids,
            positions=positions,
            attention_plan=plan,
            last_token_indices=fb.last_token_indices,
            raw_num_tokens=raw_tokens,
            ctx=ctx,
            forward_fn=lambda state: self._span_forward(model, state),
            prepare_backend=prepare_backend,
        )

    def _span_prepare(
        self,
        model: Any,
        *,
        attention_preference: str | None,
    ) -> Any | None:
        return resolve_span(
            owner=model,
            kv_pool=self.kv_pool,
            attention_preference=attention_preference,
        )

    # ---- the graph-unaware model forward ------------------------------------

    def _step_forward(self, model: Any, state: StepState) -> "torch.Tensor":
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            req_ids=tuple(range(int(state.batch_size))),
            input_ids=state.input_ids,
            positions=state.positions,
            attn_plan=state.plan,
        )
        return model.forward(state.input_ids, state.positions, fb)

    def _span_forward(self, model: Any, state: SpanState) -> "torch.Tensor":
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            req_ids=tuple(range(int(state.batch_size))),
            input_ids=state.input_ids,
            positions=state.positions,
            last_token_indices=state.last_token_indices,
            attn_plan=state.plan,
        )
        return model.forward(state.input_ids, state.positions, fb)

    # ---- prefill bucket padding + warmup ------------------------------------

    def reorder_mixed_for_padding(self, batch: Any) -> Any:
        """Keep row order stable; graph token padding has an isolated row."""

        return batch

    def padded_num_tokens(self, batch: Any, *, attention_preference: str | None) -> int | None:
        """Pad an extend group up to a captured prefill bucket, else ``None``."""

        del attention_preference
        if not self._span.enabled():
            return None
        if batch.mode not in (ForwardMode.EXTEND, ForwardMode.MIXED):
            return None
        if any(batch.spec_token_ids):
            return None
        lengths = [len(tokens) for tokens in batch.token_ids]
        if not lengths or any(length <= 0 for length in lengths):
            return None
        raw_tokens = sum(int(length) for length in lengths)
        bucket = self._span.bucket_num_tokens(raw_tokens)
        if bucket <= raw_tokens:
            return None
        graph_batch_size = self._span.padding_batch_size(
            len(lengths),
            num_tokens=int(bucket),
        )
        if graph_batch_size <= len(lengths):
            return None
        return bucket

    def warmup(self, model: Any) -> None:
        if self.device.type != "cuda":
            return
        if self._step.enabled() and self._step.warmup_enabled():
            prepare_backend = self._step_prepare(
                model, attention_preference=self.attention_preference
            )
            if prepare_backend is None:
                logger.info(
                    "skipping step graph warmup: no graph-capable paged-decode "
                    "backend or model geometry hook; step stays eager"
                )
            else:
                self._step.warmup(
                    kv_pool=self.kv_pool,
                    num_blocks=self.num_blocks,
                    device=self.device,
                    max_context_len=self.max_context_len,
                    attention_preference=self.attention_preference,
                    forward_fn=lambda state: self._step_forward(model, state),
                    prepare_backend=prepare_backend,
                )
        if self._span.enabled() and self._span.warmup_enabled():
            prepare_backend = self._span_prepare(
                model, attention_preference=self.attention_preference
            )
            if prepare_backend is None:
                logger.info(
                    "skipping span graph warmup: no graph-capable paged-prefill "
                    "backend or model geometry hook; span stays eager"
                )
            else:
                self._span.warmup(
                    kv_pool=self.kv_pool,
                    num_blocks=self.num_blocks,
                    block_size=self.block_size,
                    device=self.device,
                    max_context_len=self.max_context_len,
                    attention_preference=self.attention_preference,
                    forward_fn=lambda state: self._span_forward(model, state),
                    prepare_backend=prepare_backend,
                )


def _padded_max_tokens(
    plan: Any,
    *,
    padded_tokens: int,
    raw_tokens: int,
    batch_size: int,
) -> int:
    pad = max(0, int(padded_tokens) - int(raw_tokens))
    fallback = max(int(getattr(plan, "max_seqlen_k", 0) or 0), pad)
    cache_lens = tuple(int(length) for length in getattr(plan, "cache_seqlens_cpu", ()) or ())
    query_lens = tuple(int(length) for length in getattr(plan, "query_lens_cpu", ()) or ())
    if len(cache_lens) != int(batch_size) or len(query_lens) != int(batch_size) or not query_lens:
        return max(1, fallback)
    padded_max = max(
        (int(base) + int(query) for base, query in zip(cache_lens, query_lens, strict=True)),
        default=fallback,
    )
    return max(1, padded_max, pad)


def backend_name(name: str | None) -> str | None:
    """Resolve an explicit attention-backend request to its registry name.

    Auto/boolean sentinels mean "no explicit choice"; ``eager`` maps to the
    dense SDPA backend. Family graph runners consult this instead of touching
    the backend registry themselves.
    """
    from uniserve_worker.backends.attention.registry import normalize_attention_backend_name

    normalized = normalize_attention_backend_name(name)
    if normalized in {"", "auto", "0", "false", "off", "1", "true", "on"}:
        return None
    if normalized == "eager":
        return "torch_sdpa"
    return normalized
