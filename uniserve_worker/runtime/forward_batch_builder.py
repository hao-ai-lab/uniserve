"""The system ``ForwardBatchBuilder`` — UniServe's ``ForwardBatch.init_new``.

Builds the GPU snapshot for a forward: stages text core tensors, resolves
KV-residency indices from host-leased ``block_ids`` (in ``RequestStateTable``)
against the system-owned ``KvPool``, and builds the per-forward attention plan
(paged request-cache view + block_table / cache_seqlens / cu_seqlens / decode
write-locations). The model receives the finished :class:`ForwardBatch`.

Attention-plan construction (``build_text_attention_plan``) is model-neutral
(pool + block ids + the batch) and preserves numerics unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ..contracts.attention_plan import AttentionPlanBase
from ..contracts.forward_batch import ForwardBatch
from ..contracts.forward_mode import ForwardMode
from ..foundation.errors import invalid_descriptor
from .kv_pool import PagedKVPool
from .paged_text_cache import BatchedPagedRequestCache
from .text_forward_assembler import TextForwardAssembler
from .text_kv import TextAttentionPlan, TextKvResidency

if TYPE_CHECKING:
    from ..contracts.batches import TextBatch
    from .request_state import RequestStateTable

__all__ = [
    "ForwardBatchBuilder",
    "build_text_attention_plan",
    "state_block_ids_for_op",
]

_TEXT_MODES = frozenset(
    {ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.VERIFY_DRAFT}
)


def state_block_ids_for_op(op: Mapping[str, Any], state: Any) -> list[int]:
    """Return the request's full block-id chain, tail-deduping a retried op.

    The runner already extends ``state.block_ids`` with each op's
    ``new_block_ids`` during accounting; this re-asserts the idempotency
    contract (a retried op carrying the same tail is a no-op) and returns the
    resolved chain the system pool is addressed by.
    """

    new_blocks = [int(block_id) for block_id in (op.get("new_block_ids") or [])]
    if new_blocks and list(state.block_ids[-len(new_blocks):]) != new_blocks:
        state.extend_block_ids(new_blocks)
    return [int(block_id) for block_id in state.block_ids]


def build_text_attention_plan(
    batch: ForwardBatch,
    cache: BatchedPagedRequestCache,
    *,
    stager: Any | None = None,
    max_context_len: int = 0,
) -> AttentionPlanBase:
    """Build the per-forward text attention plan from ``ForwardBatch`` indices.

    Relocated from the model: this is the system's per-forward plan (SGLang's
    ``init_forward_metadata``), built from the batch + the system pool, never by
    the model.
    """

    if batch.input_ids is None:
        raise invalid_descriptor("text attention plan requires input ids")
    device = batch.input_ids.device
    query_lens_values = tuple(len(op.get("token_ids") or []) for op in batch.ops)
    return TextAttentionPlan.from_cache(
        cache,
        mode=batch.mode,
        device=device,
        query_lens_cpu=query_lens_values,
        stager=stager,
        max_context_len=max_context_len,
    ).to_plan()


class ForwardBatchBuilder:
    """Stages every mode's GPU tensors and resolves system-owned residency.

    Owns the pinned-buffer staging ring; the text path stages the input/position
    tensors then resolves the KV-residency indices (``block_table`` /
    ``cache_seqlens`` / the paged request-cache view) from the host-leased block
    ids against the system pool. The result is a :class:`ForwardBatch` carrying
    the finished attention plan on ``attn_plan``.
    """

    def __init__(self, *, ring_depth: int = 3, max_context_len: int = 0) -> None:
        self.max_context_len = max(0, int(max_context_len))
        self._assembler = TextForwardAssembler(
            ring_depth=ring_depth,
            max_context_len=self.max_context_len,
        )
        self._stager = self._assembler.stager

    def build_text(
        self,
        text: "TextBatch",
        *,
        device: torch.device | str,
        kv_pool: PagedKVPool,
        request_states: "RequestStateTable",
        input_ids_override: torch.Tensor | None = None,
        positions_override: torch.Tensor | None = None,
        input_ids_replacements: Mapping[int, torch.Tensor] | None = None,
        padded_num_tokens: int | None = None,
        with_metadata: bool = True,
        stage_slot: Any | None = None,
    ) -> ForwardBatch:
        """Stage one text group and attach its system-built attention plan."""

        return self._assembler.build(
            text,
            device=device,
            kv_pool=kv_pool,
            request_states=request_states,
            input_ids_override=input_ids_override,
            positions_override=positions_override,
            input_ids_replacements=input_ids_replacements,
            padded_num_tokens=padded_num_tokens,
            with_metadata=with_metadata,
            stage_slot=stage_slot,
        )

    def build_text_op(
        self,
        *,
        op: Mapping[str, Any],
        token_ids: tuple[int, ...],
        pos_range: tuple[int, int],
        req_id: int,
        mode: "ForwardMode",
        device: torch.device | str,
        kv_pool: PagedKVPool,
        request_states: "RequestStateTable",
        input_ids_override: torch.Tensor | None = None,
    ) -> ForwardBatch:
        """Build a single-request ``ForwardBatch`` for the per-op dense fallback.

        When no batched paged/varlen backend can serve the group (e.g.
        ``torch_sdpa``), the driver runs each op through the same thin
        ``model.forward`` against a single-request :class:`PagedRequestCache`
        (the only cache with the dense ``get``/``append`` read path). The
        attention plan is the minimal metadata the dense path consumes.
        """

        return self._assembler.build_op(
            op=op,
            token_ids=token_ids,
            pos_range=pos_range,
            req_id=req_id,
            mode=mode,
            device=device,
            kv_pool=kv_pool,
            request_states=request_states,
            input_ids_override=input_ids_override,
        )

    def attach_text_metadata(
        self,
        fb: ForwardBatch,
        *,
        kv_pool: PagedKVPool,
        request_states: "RequestStateTable",
        stager: Any | None = None,
    ) -> ForwardBatch:
        """Resolve the paged request-cache view + attention plan onto ``fb``."""

        self._assembler.attach_metadata(
            fb,
            kv_pool=kv_pool,
            request_states=request_states,
            stager=stager,
        )
        return fb

    @staticmethod
    def _batched_cache(
        fb: ForwardBatch,
        *,
        kv_pool: PagedKVPool,
        request_states: "RequestStateTable",
    ) -> BatchedPagedRequestCache:
        base_lens = [int((op.get("pos_range") or (0, 0))[0]) for op in fb.ops]
        block_ids_by_row: list[list[int]] = []
        for op, req_id in zip(fb.ops, fb.req_ids):
            state = request_states.get(int(req_id))
            block_ids_by_row.append(state_block_ids_for_op(op, state))
        return BatchedPagedRequestCache(kv_pool, block_ids_by_row, [int(x) for x in base_lens])

    @staticmethod
    def cache_from_block_ids(
        kv_pool: PagedKVPool,
        block_ids_by_row: Sequence[Sequence[int]],
        base_lens: Sequence[int],
    ) -> BatchedPagedRequestCache:
        """Build a batched paged cache directly (graph warmup / verify paths)."""

        return TextKvResidency(kv_pool).cache_from_block_ids(block_ids_by_row, base_lens)
