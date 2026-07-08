"""The system ``ForwardBatchBuilder`` — UniServe's ``ForwardBatch.init_new``.

Builds the GPU snapshot for a forward: stages text core tensors, resolves
KV-residency indices from host-leased ``block_ids`` (in ``RequestStateTable``)
against the system-owned ``KvPool``, and builds the per-forward attention plan
(paged request-cache view + block_table / cache_seqlens / cu_seqlens / decode
write-locations). The model receives the finished :class:`ForwardBatch`.

Attention-plan construction (``build_text_attention_metadata``) is model-neutral
(pool + block ids + the batch) and preserves numerics unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ..backends.paged_kv_math import decode_write_locations
from ..contracts.forward_batch import ForwardBatch
from ..contracts.forward_context import TextAttentionMetadata
from ..contracts.forward_mode import ForwardMode
from .kv_pool import PagedKVPool
from .paged_text_cache import BatchedPagedRequestCache
from .tensor_staging import TextTensorStager, stage_text_forward_batch

if TYPE_CHECKING:
    from ..contracts.batches import TextBatch
    from .request_state import RequestStateTable

__all__ = [
    "ForwardBatchBuilder",
    "build_text_attention_metadata",
    "state_block_ids_for_op",
]

_TEXT_MODES = frozenset(
    {ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.TARGET_VERIFY}
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


def build_text_attention_metadata(
    batch: ForwardBatch,
    cache: BatchedPagedRequestCache,
    *,
    stager: Any | None = None,
    max_context_len: int = 0,
) -> TextAttentionMetadata:
    """Build the per-forward text attention plan from ``ForwardBatch`` indices.

    Relocated from the model: this is the system's per-forward plan (SGLang's
    ``init_forward_metadata``), built from the batch + the system pool, never by
    the model.
    """

    device = batch.input_ids.device
    query_lens_values = tuple(len(op.get("token_ids") or []) for op in batch.ops)
    cache_seqlens_cpu = tuple(int(length) for length in cache.base_lens)
    cache_seqlens = cache.cache_seqlens(device=device, stager=stager)
    block_table = cache.block_table(device=device, stager=stager)
    kv_lens_values = tuple(
        int(base_len) + int(query_len)
        for base_len, query_len in zip(cache.base_lens, query_lens_values)
    )
    if batch.mode == ForwardMode.DECODE:
        decode_page_ids, decode_page_offsets = decode_write_locations(
            block_table,
            cache_seqlens,
            cache.pool.block_size,
        )
        return TextAttentionMetadata(
            cache=cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cache_seqlens_cpu=cache_seqlens_cpu,
            query_lens_cpu=query_lens_values,
            kv_seqlens_cpu=kv_lens_values,
            decode_page_ids=decode_page_ids,
            decode_page_offsets=decode_page_offsets,
            max_seqlen_q=max(query_lens_values, default=0),
            max_seqlen_k=max(kv_lens_values, default=0),
            max_context_len=int(max_context_len),
            mode=batch.mode,
        )

    query_lens = batch.extend_seq_lens.to(dtype=torch.int32).contiguous()
    kv_seqlens = batch.seq_lens.to(dtype=torch.int32).contiguous()
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.cat([zero, torch.cumsum(query_lens, dim=0).to(torch.int32)])
    cu_seqlens_k = torch.cat([zero, torch.cumsum(kv_seqlens, dim=0).to(torch.int32)])
    return TextAttentionMetadata(
        cache=cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=cache_seqlens_cpu,
        query_lens=query_lens,
        query_lens_cpu=query_lens_values,
        kv_seqlens=kv_seqlens,
        kv_seqlens_cpu=kv_lens_values,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(query_lens_values, default=0),
        max_seqlen_k=max(kv_lens_values, default=0),
        max_context_len=int(max_context_len),
        mode=batch.mode,
    )


class ForwardBatchBuilder:
    """Stages every mode's GPU tensors and resolves system-owned residency.

    Owns the pinned-buffer staging ring; the text path stages the input/position
    tensors then resolves the KV-residency indices (``block_table`` /
    ``cache_seqlens`` / the paged request-cache view) from the host-leased block
    ids against the system pool. The result is a :class:`ForwardBatch` carrying
    the finished attention plan on ``attn_metadata``.
    """

    def __init__(self, *, ring_depth: int = 3, max_context_len: int = 0) -> None:
        self._stager = TextTensorStager(ring_depth=ring_depth)
        self.max_context_len = max(0, int(max_context_len))

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

        device = torch.device(device)
        stager = self._stager if device.type == "cuda" else None
        if stage_slot is None and stager is not None:
            stage_slot = stager.next_slot()
        fb = stage_text_forward_batch(
            text,
            device,
            stager=stager,
            stage_slot=stage_slot,
            input_ids_override=input_ids_override,
            positions_override=positions_override,
            input_ids_replacements=input_ids_replacements,
            padded_num_tokens=padded_num_tokens,
        )
        fb.device = device
        if with_metadata:
            self.attach_text_metadata(
                fb, kv_pool=kv_pool, request_states=request_states, stager=stage_slot
            )
        return fb

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

        device = torch.device(device)
        state = request_states.get(int(req_id))
        block_ids = state_block_ids_for_op(op, state)
        base_len = int(pos_range[0])
        query_len = len(token_ids)
        cache = kv_pool.view(block_ids, base_len)
        if input_ids_override is not None:
            input_ids = input_ids_override.reshape(-1)
        else:
            input_ids = torch.tensor([int(t) for t in token_ids], dtype=torch.long, device=device)
        positions = torch.arange(base_len, base_len + query_len, dtype=torch.long, device=device)
        block_table = cache.block_table(device=device)
        cache_seqlens = torch.tensor([base_len], dtype=torch.int32, device=device)
        metadata = TextAttentionMetadata(
            cache=cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cache_seqlens_cpu=(base_len,),
            query_lens_cpu=(query_len,),
            kv_seqlens_cpu=(base_len + query_len,),
            max_seqlen_q=query_len,
            max_seqlen_k=base_len + query_len,
            max_context_len=self.max_context_len,
            mode=mode,
        )
        last_token = torch.tensor([query_len - 1], dtype=torch.long, device=device)
        return ForwardBatch(
            forward_mode=mode,
            req_ids=(int(req_id),),
            ops=(op,),
            device=device,
            input_ids=input_ids,
            positions=positions,
            last_token_indices=last_token,
            num_token_non_padded=query_len,
            padded_num_tokens=query_len,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            attn_metadata=metadata,
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

        cache = self._batched_cache(fb, kv_pool=kv_pool, request_states=request_states)
        metadata = build_text_attention_metadata(
            fb,
            cache,
            stager=stager,
            max_context_len=self.max_context_len,
        )
        fb.attn_metadata = metadata
        fb.block_table = metadata.block_table
        fb.cache_seqlens = metadata.cache_seqlens
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

        return BatchedPagedRequestCache(
            kv_pool, [list(ids) for ids in block_ids_by_row], [int(x) for x in base_lens]
        )
