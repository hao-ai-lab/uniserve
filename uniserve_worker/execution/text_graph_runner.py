"""System-owned CUDA-graph runner for the text forward.

Owns decode and text-prefill CUDA graphs keyed on the system KV pool. The
graph-unaware model is captured/replayed around: the runner builds a static
:class:`ForwardBatch` wrapping the captured graph state's attention plan and calls
the same thin ``model.forward(input_ids, positions, forward_batch)`` the eager path
uses, so the model never knows it is being graphed.

Graph settings are model-neutral and come from the worker runtime config.
"""
from __future__ import annotations

import logging
from dataclasses import replace as _dc_replace
from typing import TYPE_CHECKING, Any

from ..contracts.forward_batch import ForwardBatch
from ..contracts.forward_mode import ForwardMode
from ..foundation.runtime_config import get_worker_config
from ..runtime.paged_text_cache import BatchedPagedRequestCache
from .decode_cuda_graph import (
    DecodeCudaGraphRunner,
    TextDecodeGraphState,
    resolve_paged_decode_graph_prepare,
)
from .prefill_cuda_graph import (
    PrefillCudaGraphRunner,
    TextInitialPrefillGraphState,
    resolve_paged_prefill_graph_prepare,
)

if TYPE_CHECKING:
    import torch

    from ..runtime.kv_pool import PagedKVPool

logger = logging.getLogger(__name__)

__all__ = ["TextGraphRunner"]

class TextGraphRunner:
    """Owns the decode + text-prefill CUDA graphs, keyed on the system pool."""

    def __init__(
        self,
        *,
        kv_pool: "PagedKVPool",
        num_blocks: int,
        block_size: int,
        device: "torch.device",
        attention_backend_name: str | None = None,
        max_context_len: int = 0,
    ) -> None:
        self.kv_pool = kv_pool
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.device = device
        self.max_context_len = max(0, int(max_context_len))
        # Startup (warmup) has no forward context to read the backend name from;
        # per-forward calls prefer the context's resolved name.
        self.attention_backend_name = attention_backend_name
        runtime = get_worker_config()
        self._decode = DecodeCudaGraphRunner(
            name="text",
            default_enabled=runtime.cuda_graph,
            default_warmup=runtime.cuda_graph_warmup,
            default_warmup_batch_sizes=runtime.cuda_graph_warmup_batches,
            metric_prefix="text_",
            logger=logger,
        )
        self._prefill = PrefillCudaGraphRunner(
            name="text",
            default_enabled=runtime.prefill_cuda_graph,
            default_warmup=runtime.prefill_cuda_graph_warmup,
            default_warmup_token_buckets=runtime.prefill_cuda_graph_warmup_tokens,
            metric_prefix="text_",
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

        metadata = fb.attn_metadata
        if not isinstance(getattr(metadata, "cache", None), BatchedPagedRequestCache):
            return None
        if getattr(input_ids, "device", None) is None or input_ids.device.type != "cuda":
            return None
        if fb.forward_mode == ForwardMode.DECODE:
            return self._maybe_decode(model, input_ids, positions, fb, metadata, ctx)
        if fb.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            # A mixed extend+decode group is shape-identical to a cached-prefix
            # extend group (flat varlen rows, per-row context lengths, per-row
            # last-token sampling), so it replays the same prefill buckets.
            return self._maybe_prefill(model, input_ids, positions, fb, metadata, ctx)
        return None

    def _maybe_decode(self, model, input_ids, positions, fb, metadata, ctx):
        if not self._decode.enabled():
            return None
        if input_ids.ndim != 2 or int(input_ids.shape[1]) != 1:
            return None
        batch_size = int(input_ids.shape[0])
        prepare_backend = self._decode_prepare_backend(
            model,
            attention_backend_name=getattr(ctx, "attention_backend_name", None)
            or self.attention_backend_name,
        )
        if prepare_backend is None:
            # A captured paged-decode graph is only correct when the backend can
            # refill its plan buffers before every replay; without that, the
            # capture-time plan is baked in and later steps silently read stale
            # pages. Stay eager rather than capture a wrong graph.
            return None
        return self._decode.maybe_run(
            kv_pool=self.kv_pool,
            num_blocks=self.num_blocks,
            batch_size=batch_size,
            input_ids=input_ids,
            positions=positions,
            attention_metadata=metadata,
            ctx=ctx,
            forward_fn=lambda state: self._decode_forward(model, state),
            prepare_backend=prepare_backend,
        )

    def _decode_prepare_backend(
        self,
        model: Any,
        *,
        attention_backend_name: str | None,
    ) -> Any | None:
        return resolve_paged_decode_graph_prepare(
            owner=model,
            kv_pool=self.kv_pool,
            num_blocks=self.num_blocks,
            attention_backend_name=attention_backend_name,
        )

    def _maybe_prefill(self, model, input_ids, positions, fb, metadata, ctx):
        if not self._prefill.enabled():
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
        max_kv_tokens = self._prefill.bucket_kv_tokens(
            _padded_prefill_max_kv_tokens(
                metadata,
                padded_tokens=padded_tokens,
                raw_tokens=raw_tokens,
                batch_size=batch_size,
            ),
            max_context_len=self.max_context_len,
        )
        if not self._prefill.can_use(
            padded_tokens,
            batch_size=batch_size,
            max_kv_tokens=max_kv_tokens,
        ):
            return None
        prepare_backend = self._prefill_prepare_backend(
            model,
            attention_backend_name=getattr(ctx, "attention_backend_name", None)
            or getattr(self, "attention_backend_name", None),
        )
        if prepare_backend is None:
            return None
        return self._prefill.maybe_run(
            kv_pool=self.kv_pool,
            num_blocks=self.num_blocks,
            num_tokens=padded_tokens,
            max_kv_tokens=max_kv_tokens,
            batch_size=batch_size,
            input_ids=input_ids,
            positions=positions,
            attention_metadata=metadata,
            last_token_indices=fb.last_token_indices,
            raw_num_tokens=raw_tokens,
            ctx=ctx,
            forward_fn=lambda state: self._prefill_forward(model, state),
            prepare_backend=prepare_backend,
        )

    def _prefill_prepare_backend(
        self,
        model: Any,
        *,
        attention_backend_name: str | None,
    ) -> Any | None:
        return resolve_paged_prefill_graph_prepare(
            owner=model,
            kv_pool=self.kv_pool,
            attention_backend_name=attention_backend_name,
        )

    # ---- the graph-unaware model forward ------------------------------------

    def _decode_forward(self, model: Any, state: TextDecodeGraphState) -> "torch.Tensor":
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            req_ids=tuple(range(int(state.batch_size))),
            input_ids=state.input_ids,
            positions=state.positions,
            attn_metadata=state.metadata,
        )
        return model.forward(state.input_ids, state.positions, fb)

    def _prefill_forward(self, model: Any, state: TextInitialPrefillGraphState) -> "torch.Tensor":
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            req_ids=tuple(range(int(state.batch_size))),
            input_ids=state.input_ids,
            positions=state.positions,
            last_token_indices=state.last_token_indices,
            attn_metadata=state.metadata,
        )
        return model.forward(state.input_ids, state.positions, fb)

    # ---- prefill bucket padding + warmup ------------------------------------

    def reorder_mixed_for_padding(self, text: Any) -> Any:
        """Reorder a MIXED group so bucket padding can extend its final row.

        Token-bucket padding grows the last row's query span, which is only
        legal while the pad stays inside that row's current KV block. When the
        natural last row sits at (or too near) a block boundary, swap in any
        row with enough tail room; row order is otherwise semantically free
        (results are keyed per op).
        """

        if getattr(text, "mode", None) != ForwardMode.MIXED or not self._prefill.enabled():
            return text
        if any(text.spec_token_ids):
            return text
        lengths = [len(tokens) for tokens in text.token_ids]
        if not lengths or any(length <= 0 for length in lengths):
            return text
        raw_tokens = sum(lengths)
        pad = self._prefill.bucket_num_tokens(raw_tokens) - raw_tokens
        if pad <= 0:
            return text

        def room_ok(row: int) -> bool:
            end = int(text.pos_ranges[row][1])
            return _blocks_for_tokens(end + pad, self.block_size) <= _blocks_for_tokens(
                end, self.block_size
            )

        count = len(lengths)
        if room_ok(count - 1):
            return text
        swap = next((row for row in range(count - 1) if room_ok(row)), None)
        if swap is None:
            return text
        order = list(range(count))
        order[swap], order[-1] = order[-1], order[swap]

        def pick(seq: Any) -> tuple:
            return tuple(seq[row] for row in order)

        return _dc_replace(
            text,
            req_ids=pick(text.req_ids),
            token_ids=pick(text.token_ids),
            spec_token_ids=pick(text.spec_token_ids),
            pos_ranges=pick(text.pos_ranges),
            ops=pick(text.ops),
        )

    def padded_num_tokens(self, text: Any, *, attention_backend_name: str | None) -> int | None:
        """Pad an extend group up to a captured prefill bucket, else ``None``."""

        del attention_backend_name
        if not self._prefill.enabled():
            return None
        if text.mode not in (ForwardMode.EXTEND, ForwardMode.MIXED):
            return None
        if any(text.spec_token_ids):
            return None
        lengths = [len(tokens) for tokens in text.token_ids]
        if not lengths or any(length <= 0 for length in lengths):
            return None
        raw_tokens = sum(int(length) for length in lengths)
        bucket = self._prefill.bucket_num_tokens(raw_tokens)
        if bucket <= raw_tokens:
            return None
        pad = int(bucket) - int(raw_tokens)
        base, end = text.pos_ranges[-1]
        raw_last = int(end) - int(base)
        padded_last = raw_last + pad
        if _blocks_for_tokens(int(base) + padded_last, self.block_size) > _blocks_for_tokens(int(end), self.block_size):
            return None
        return bucket

    def warmup(self, model: Any) -> None:
        if self.device.type != "cuda":
            return
        if self._decode.enabled() and self._decode.warmup_enabled():
            prepare_backend = self._decode_prepare_backend(
                model, attention_backend_name=self.attention_backend_name
            )
            if prepare_backend is None:
                logger.info(
                    "skipping decode graph warmup: no graph-capable paged-decode "
                    "backend or model geometry hook; decode stays eager"
                )
            else:
                self._decode.warmup(
                    kv_pool=self.kv_pool,
                    num_blocks=self.num_blocks,
                    device=self.device,
                    max_context_len=self.max_context_len,
                    attention_backend_name=self.attention_backend_name,
                    forward_fn=lambda state: self._decode_forward(model, state),
                    prepare_backend=prepare_backend,
                )
        if self._prefill.enabled() and self._prefill.warmup_enabled():
            prepare_backend = self._prefill_prepare_backend(
                model, attention_backend_name=self.attention_backend_name
            )
            if prepare_backend is None:
                logger.info(
                    "skipping prefill graph warmup: no graph-capable paged-prefill "
                    "backend or model geometry hook; prefill stays eager"
                )
            else:
                self._prefill.warmup(
                    kv_pool=self.kv_pool,
                    num_blocks=self.num_blocks,
                    block_size=self.block_size,
                    device=self.device,
                    max_context_len=self.max_context_len,
                    attention_backend_name=self.attention_backend_name,
                    forward_fn=lambda state: self._prefill_forward(model, state),
                    prepare_backend=prepare_backend,
                )


def _blocks_for_tokens(tokens: int, block_size: int) -> int:
    tokens = max(0, int(tokens))
    block_size = max(1, int(block_size))
    return (tokens + block_size - 1) // block_size


def _padded_prefill_max_kv_tokens(
    metadata: Any,
    *,
    padded_tokens: int,
    raw_tokens: int,
    batch_size: int,
) -> int:
    pad = max(0, int(padded_tokens) - int(raw_tokens))
    fallback = int(getattr(metadata, "max_seqlen_k", 0) or 0) + pad
    cache_lens = tuple(int(length) for length in getattr(metadata, "cache_seqlens_cpu", ()) or ())
    query_lens = tuple(int(length) for length in getattr(metadata, "query_lens_cpu", ()) or ())
    if len(cache_lens) != int(batch_size) or len(query_lens) != int(batch_size) or not query_lens:
        return max(1, fallback)
    graph_lens = list(query_lens)
    graph_lens[-1] += pad
    padded_max = max(
        (int(base) + int(query) for base, query in zip(cache_lens, graph_lens, strict=True)),
        default=fallback,
    )
    return max(1, padded_max)
