"""System-owned CUDA-graph runner for the text forward.

Owns decode and initial-prefill CUDA graphs keyed on the system KV pool. The
graph-unaware model is captured/replayed around: the runner builds a static
:class:`ForwardBatch` wrapping the captured graph state's attention plan and calls
the same thin ``model.forward(input_ids, positions, forward_batch)`` the eager path
uses, so the model never knows it is being graphed.

Graph settings are model-neutral and come from the worker runtime config.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..contracts.forward_batch import ForwardBatch
from ..contracts.forward_mode import ForwardMode
from ..foundation.runtime_config import get_worker_config
from ..runtime.paged_text_cache import BatchedPagedRequestCache
from .decode_cuda_graph import (
    DecodeCudaGraphRunner,
    PrefillCudaGraphRunner,
    TextDecodeGraphState,
    TextInitialPrefillGraphState,
    resolve_paged_decode_graph_prepare,
)

if TYPE_CHECKING:
    import torch

    from ..runtime.kv_pool import PagedKVPool

logger = logging.getLogger(__name__)

__all__ = ["TextGraphRunner"]

class TextGraphRunner:
    """Owns the decode + initial-prefill text CUDA graphs, keyed on the system pool."""

    def __init__(
        self,
        *,
        kv_pool: "PagedKVPool",
        num_blocks: int,
        block_size: int,
        device: "torch.device",
        attention_backend_name: str | None = None,
    ) -> None:
        self.kv_pool = kv_pool
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.device = device
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
        if fb.forward_mode == ForwardMode.EXTEND:
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
        cache = metadata.cache
        batch_size = int(fb.batch_size)
        if batch_size <= 0 or fb.last_token_indices is None:
            return None
        if any(int(base) != 0 for base in cache.base_lens):
            return None  # initial-prefill graphs only (no KV history)
        if any(fb.spec_token_ids):
            return None
        raw_tokens = int(fb.num_token_non_padded)
        padded_tokens = int(input_ids.numel())
        if raw_tokens <= 0 or padded_tokens < raw_tokens:
            return None
        if not self._prefill.can_use(padded_tokens, batch_size=batch_size):
            return None
        return self._prefill.maybe_run(
            kv_pool=self.kv_pool,
            num_blocks=self.num_blocks,
            num_tokens=padded_tokens,
            batch_size=batch_size,
            input_ids=input_ids,
            positions=positions,
            attention_metadata=metadata,
            last_token_indices=fb.last_token_indices,
            raw_num_tokens=raw_tokens,
            ctx=ctx,
            forward_fn=lambda state: self._prefill_forward(model, state),
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

    def padded_num_tokens(self, text: Any, *, attention_backend_name: str | None) -> int | None:
        """Pad an initial-extend group up to a captured prefill bucket, else ``None``."""

        del attention_backend_name
        if not self._prefill.enabled():
            return None
        if text.mode != ForwardMode.EXTEND:
            return None
        if any(text.spec_token_ids):
            return None
        if not all(int(pos[0]) == 0 for pos in text.pos_ranges):
            return None
        lengths = [len(tokens) for tokens in text.token_ids]
        if not lengths or any(length <= 0 for length in lengths):
            return None
        raw_tokens = sum(int(length) for length in lengths)
        bucket = self._prefill.bucket_num_tokens(raw_tokens)
        if bucket <= raw_tokens:
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
                    forward_fn=lambda state: self._decode_forward(model, state),
                    prepare_backend=prepare_backend,
                )
        if self._prefill.enabled() and self._prefill.warmup_enabled():
            self._prefill.warmup(
                kv_pool=self.kv_pool,
                num_blocks=self.num_blocks,
                block_size=self.block_size,
                device=self.device,
                forward_fn=lambda state: self._prefill_forward(model, state),
            )
