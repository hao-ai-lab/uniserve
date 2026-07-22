"""Text forward-batch assembly over request sessions and KV plans."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

import torch

from ..contracts.attention_plan import AttnPlan, KvView, PagedDecodePlan, PagedVarlenPlan
from ..contracts.forward_batch import ForwardBatch
from ..contracts.forward_mode import ForwardMode
from ..foundation.errors import invalid_descriptor
from .kv_pool import PagedKVPool
from .request_session import PreparedTextRow
from .tensor_staging import TextTensorStager, stage_text_forward_batch
from .text_kv import TextAttentionPlan, TextKvResidency

if TYPE_CHECKING:
    from ..contracts.batches import TextBatch
    from .request_state import RequestStateTable

__all__ = [
    "TextForwardAssembler",
]


class TextForwardAssembler:
    """Stages text tensors and attaches a system-built KV attention plan."""

    def __init__(self, *, ring_depth: int = 3, max_context_len: int = 0) -> None:
        self.stager = TextTensorStager(ring_depth=ring_depth)
        self.max_context_len = max(0, int(max_context_len))

    def build(
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
        target = torch.device(device)
        stager = self.stager if target.type == "cuda" else None
        if stage_slot is None and stager is not None:
            stage_slot = stager.next_slot()
        fb = stage_text_forward_batch(
            text,
            target,
            stager=stager,
            stage_slot=stage_slot,
            input_ids_override=input_ids_override,
            positions_override=positions_override,
            input_ids_replacements=input_ids_replacements,
            padded_num_tokens=padded_num_tokens,
        )
        fb.device = target
        if with_metadata:
            self.attach_metadata(
                fb,
                kv_pool=kv_pool,
                request_states=request_states,
                stager=stage_slot,
            )
        return fb

    def build_op(
        self,
        *,
        op: Mapping[str, Any],
        token_ids: tuple[int, ...],
        pos_range: tuple[int, int],
        req_id: int,
        mode: ForwardMode,
        device: torch.device | str,
        kv_pool: PagedKVPool,
        request_states: "RequestStateTable",
        input_ids_override: torch.Tensor | None = None,
    ) -> ForwardBatch:
        target = torch.device(device)
        row = self._resolve_row(request_states, int(req_id), op)
        cache = kv_pool.view(list(row.block_ids), int(pos_range[0]))
        query_len = len(token_ids)
        if input_ids_override is not None:
            input_ids = input_ids_override.reshape(-1)
        else:
            input_ids = torch.tensor([int(t) for t in token_ids], dtype=torch.long, device=target)
        positions = torch.arange(
            int(pos_range[0]),
            int(pos_range[0]) + query_len,
            dtype=torch.long,
            device=target,
        )
        block_table = cache.block_table(device=target)
        cache_seqlens = torch.tensor([int(pos_range[0])], dtype=torch.int32, device=target)
        residency = cast(KvView, cache)
        if mode == ForwardMode.DECODE and query_len == 1:
            kv_seqlens = cache_seqlens + 1
            from ..backends.paged_kv_math import decode_write_locations

            page_ids, page_offsets = decode_write_locations(
                block_table,
                cache_seqlens,
                kv_pool.block_size,
            )
            plan: AttnPlan = PagedDecodePlan(
                residency_cache=residency,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_seqlens_cpu=(int(pos_range[0]),),
                kv_seqlens=kv_seqlens,
                query_lens=torch.ones(1, dtype=torch.int32, device=target),
                query_lens_cpu=(1,),
                kv_seqlens_cpu=(int(pos_range[0]) + 1,),
                decode_page_ids=page_ids,
                decode_page_offsets=page_offsets,
                max_context_len=self.max_context_len,
            )
        else:
            plan = PagedVarlenPlan(
                residency_cache=residency,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_seqlens_cpu=(int(pos_range[0]),),
                query_lens_cpu=(query_len,),
                kv_seqlens_cpu=(int(pos_range[0]) + query_len,),
                cu_seqlens_q=torch.tensor([0, query_len], dtype=torch.int32, device=target),
                cu_seqlens_k=torch.tensor(
                    [0, int(pos_range[0]) + query_len],
                    dtype=torch.int32,
                    device=target,
                ),
                max_seqlen_q=query_len,
                max_seqlen_k=int(pos_range[0]) + query_len,
                max_context_len=self.max_context_len,
                mode=mode,
            )
        last_token = torch.tensor([query_len - 1], dtype=torch.long, device=target)
        return ForwardBatch(
            forward_mode=mode,
            req_ids=(int(req_id),),
            ops=(op,),
            device=target,
            input_ids=input_ids,
            positions=positions,
            last_token_indices=last_token,
            num_token_non_padded=query_len,
            padded_num_tokens=query_len,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            attn_plan=plan,
        )

    def attach_metadata(
        self,
        fb: ForwardBatch,
        *,
        kv_pool: PagedKVPool,
        request_states: "RequestStateTable",
        stager: Any | None = None,
    ) -> AttnPlan:
        rows = [
            self._resolve_row(request_states, int(req_id), op)
            for op, req_id in zip(fb.ops, fb.req_ids)
        ]
        if fb.input_ids is None:
            raise invalid_descriptor("text forward batch is missing input_ids")
        plan = TextKvResidency(kv_pool, max_context_len=self.max_context_len).open_batch(
            rows,
            fb.mode,
            fb.input_ids.device,
            stager=stager,
        )
        return plan.attach_to(fb)

    def plan_from_block_ids(
        self,
        kv_pool: PagedKVPool,
        block_ids_by_row: Sequence[Sequence[int]],
        base_lens: Sequence[int],
        query_lens: Sequence[int],
        *,
        mode: ForwardMode,
        device: torch.device | str,
        stager: Any | None = None,
    ) -> TextAttentionPlan:
        return TextKvResidency(kv_pool, max_context_len=self.max_context_len).open_from_block_ids(
            block_ids_by_row,
            base_lens,
            query_lens,
            mode=mode,
            device=device,
            stager=stager,
        )

    @staticmethod
    def _resolve_row(
        request_states: "RequestStateTable",
        req_id: int,
        op: Mapping[str, Any],
    ) -> PreparedTextRow:
        resolver = getattr(request_states, "resolve_text_row", None)
        if callable(resolver):
            return cast(PreparedTextRow, resolver(int(req_id), op))
        state = request_states.get(int(req_id))
        from .forward_batch_builder import state_block_ids_for_op

        block_ids = state_block_ids_for_op(op, state)
        return PreparedTextRow(
            req_id=int(req_id),
            op=op,
            block_ids=tuple(block_ids),
            base_len=int((op.get("pos_range") or (0, 0))[0]),
            query_len=len(op.get("token_ids") or ()),
        )
