"""Packed mixed text/denoise forward driver."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from ....contracts.batches import UniForwardBatch
from ....contracts.forward_context import get_forward_context
from ....contracts.forward_mode import ForwardMode
from ....contracts.outputs import TextTokenOutput
from ....foundation.env import env_flag
from ....foundation.errors import capability_mismatch, invalid_descriptor
from ....foundation.profiling import profile_range
from ....nn.diffusion import euler_step
from ....nn.diffusion.cfg import Branch, CfgPlan
from ....nn.sampler import (
    apply_sampling_batched_with_device_tokens,
    finalize_sampling_result,
    is_deferred_sampling_result,
)
from ....runtime.forward_batch_builder import state_block_ids_for_op
from ....runtime.host_staging import fill_cpu_ints, is_pinned
from ....runtime.paged_text_cache import (
    PagedTextCache,
    PagedTextCacheSpanCopy,
    copy_paged_text_cache_spans,
)
from ....runtime.tensor_staging import TextTensorStager
from ...denoise_driver import TextImageDenoiseStep, text_image_cfg_plan
from ...interleaved_text_stepper import hydrate_cached_prefix_from_op
from ...text_decode_relay import TextDecodeRelay
from ..deferred_text import DeferredTextSeqResult
from ..graph.packed_visible import (
    maybe_run_packed_mixed_graph,
    packed_mixed_graph_promotions_supported,
)
from ..result import (
    DenoiseBranchKey,
    DenoisePostprocessEntry,
    ForwardResult,
    TextPostprocessEntry,
)
from ..stream import (
    ForwardPagedKVSegment,
    ForwardPagedKVView,
    ForwardStreamBuilder,
)

__all__ = [
    "PackedForwardPlan",
    "PackedMixedForward",
    "run_packed_mixed_forward",
    "run_packed_visible_forward_result",
]

logger = logging.getLogger(__name__)
_DECODE_RELAY = TextDecodeRelay()

_PACKED_MIXED_TIMING = env_flag("UNISERVE_PACKED_MIXED_TIMING")
_PACKED_MIXED_TIMING_SYNC = env_flag("UNISERVE_PACKED_MIXED_TIMING_SYNC")
_PACKED_MIXED_SORT_BY_MODALITY = env_flag("UNISERVE_PACKED_MIXED_SORT_BY_MODALITY", default=True)

TextResultSlot = tuple[int, int, int, PagedTextCache, PagedTextCache, int, int]
DenoiseResultSlot = tuple[int, TextImageDenoiseStep, int, int, Branch]


@dataclass
class _PendingTextBuildRow:
    row_index: int
    q_len: int
    input_ids: torch.Tensor
    persistent_cache: PagedTextCache
    staged_cache: PagedTextCache
    base_len: int
    last_input_token: int


@dataclass
class PackedForwardPlan:
    batch: UniForwardBatch
    denoise_steps: list[tuple[int, TextImageDenoiseStep]]
    results: list[Any]
    text_result_slots: list[TextResultSlot] = field(default_factory=list)
    denoise_result_slots: list[DenoiseResultSlot] = field(default_factory=list)
    denoise_cfg_plans: dict[int, CfgPlan] = field(default_factory=dict)

    def denoise_step_for_row(self, row_index: int) -> TextImageDenoiseStep:
        for result_index, step in self.denoise_steps:
            if int(result_index) == int(row_index):
                return step
        raise invalid_descriptor(f"no denoise step prepared for mixed row {int(row_index)}")

    def add_text_slot(
        self,
        *,
        row_index: int,
        segment_start: int,
        q_len: int,
        persistent_cache: PagedTextCache,
        staged_cache: PagedTextCache,
        base_len: int,
        last_input_token: int,
    ) -> None:
        self.text_result_slots.append(
            (
                int(row_index),
                int(segment_start),
                int(q_len),
                persistent_cache,
                staged_cache,
                int(base_len),
                int(last_input_token),
            )
        )

    def add_denoise_slot(
        self,
        *,
        row_index: int,
        step: TextImageDenoiseStep,
        segment_start: int,
        q_len: int,
        branch: Branch,
    ) -> None:
        self.denoise_result_slots.append(
            (int(row_index), step, int(segment_start), int(q_len), branch)
        )

    def set_denoise_cfg_plan(self, row_index: int, cfg_plan: CfgPlan) -> None:
        self.denoise_cfg_plans[int(row_index)] = cfg_plan

    def denoise_cfg_plan_for_row(self, row_index: int) -> CfgPlan:
        cfg_plan = self.denoise_cfg_plans.get(int(row_index))
        if cfg_plan is None:
            raise invalid_descriptor(f"no denoise CFG plan prepared for mixed row {int(row_index)}")
        return cfg_plan

    def set_text_result(self, row_index: int, output: Any) -> None:
        self.results[int(row_index)] = output

    def set_denoise_result(self, row_index: int, step: TextImageDenoiseStep) -> None:
        self.results[int(row_index)] = {
            "req_id": step.req_id,
            "denoise_done": step.step_index + 1 >= step.total_steps,
            "num_steps_done": step.step_index + 1,
        }


class PackedMixedForward:
    """Owns packed mixed-forward row slots, cache writeback, and output order."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def execute(
        self,
        batch: UniForwardBatch,
        request_states: Any,
        denoise_steps: list[tuple[int, TextImageDenoiseStep]],
        results: list[Any],
        *,
        defer_text_cpu_results: bool = False,
        allow_graph: bool = True,
        require_graph: bool = False,
    ) -> bool:
        plan = PackedForwardPlan(batch=batch, denoise_steps=denoise_steps, results=results)
        result = _run_packed_mixed_forward_impl(
            self.owner,
            plan,
            request_states,
            defer_text_cpu_results=defer_text_cpu_results,
            allow_graph=allow_graph,
            require_graph=require_graph,
        )
        return bool(result)

    def execute_forward_result(
        self,
        batch: UniForwardBatch,
        request_states: Any,
        denoise_steps: list[tuple[int, TextImageDenoiseStep]],
        *,
        defer_text_cpu_results: bool = False,
        allow_graph: bool = False,
        require_graph: bool = False,
    ) -> ForwardResult | None:
        plan = PackedForwardPlan(
            batch=batch,
            denoise_steps=denoise_steps,
            results=[None] * len(batch.ops),
        )
        result = _run_packed_mixed_forward_impl(
            self.owner,
            plan,
            request_states,
            defer_text_cpu_results=defer_text_cpu_results,
            return_forward_result=True,
            allow_graph=allow_graph,
            require_graph=require_graph,
        )
        return result if isinstance(result, ForwardResult) else None


def run_packed_mixed_forward(
    owner,
    batch: UniForwardBatch,
    request_states: Any,
    denoise_steps: list[tuple[int, TextImageDenoiseStep]],
    results: list[Any],
    *,
    defer_text_cpu_results: bool = False,
    allow_graph: bool = True,
    require_graph: bool = False,
) -> bool:
    return PackedMixedForward(owner).execute(
        batch,
        request_states,
        denoise_steps,
        results,
        defer_text_cpu_results=defer_text_cpu_results,
        allow_graph=allow_graph,
        require_graph=require_graph,
    )


def run_packed_visible_forward_result(
    owner,
    batch: UniForwardBatch,
    request_states: Any,
    denoise_steps: list[tuple[int, TextImageDenoiseStep]],
    *,
    defer_text_cpu_results: bool = False,
    allow_graph: bool = False,
    require_graph: bool = False,
) -> ForwardResult | None:
    return PackedMixedForward(owner).execute_forward_result(
        batch,
        request_states,
        denoise_steps,
        defer_text_cpu_results=defer_text_cpu_results,
        allow_graph=allow_graph,
        require_graph=require_graph,
    )


def _run_packed_mixed_forward_impl(
    owner,
    plan: PackedForwardPlan,
    request_states: Any,
    *,
    defer_text_cpu_results: bool = False,
    return_forward_result: bool = False,
    allow_graph: bool = True,
    require_graph: bool = False,
) -> bool | ForwardResult:
    if owner.model is None:
        return False
    batch = plan.batch
    builder = ForwardStreamBuilder()
    kv_segments: list[ForwardPagedKVSegment] = []
    embed_chunks: list[torch.Tensor] = []
    indicator_spans: list[tuple[int, bool]] = []
    first_pool = owner._forward_target_pool(plan.denoise_steps)
    device = torch.device(str(owner.device))
    current_context: dict[str, Any] | None = None
    ctx = get_forward_context()
    timing = _PackedMixedTiming(device)
    total_start = timing.start()

    try:
        build_stats_start = ctx.component_timer_start()
        build_start = timing.start()
        text_stage_prefix_copies: list[PagedTextCacheSpanCopy] = []
        pending_text_rows: list[_PendingTextBuildRow] = []

        def flush_text_rows() -> None:
            if not pending_text_rows:
                return
            total_q = sum(int(row.q_len) for row in pending_text_rows)
            with profile_range("uniserve.packed_mixed.text_embed"):
                input_ids = torch.cat(
                    [row.input_ids.reshape(-1) for row in pending_text_rows], dim=0
                )
                if int(input_ids.numel()) != int(total_q):
                    raise invalid_descriptor(
                        "packed mixed text input id count does not match row lengths"
                    )
                text_embeds = owner.packed_text_embeddings(input_ids).reshape(total_q, -1)
            segment_base = _append_packed_chunk(
                embed_chunks,
                indicator_spans,
                text_embeds,
                image_tokens=False,
                device=device,
            )
            offset = 0
            for row in pending_text_rows:
                plan.add_text_slot(
                    row_index=row.row_index,
                    segment_start=segment_base + offset,
                    q_len=row.q_len,
                    persistent_cache=row.persistent_cache,
                    staged_cache=row.staged_cache,
                    base_len=row.base_len,
                    last_input_token=row.last_input_token,
                )
                offset += int(row.q_len)
            pending_text_rows.clear()

        with profile_range("uniserve.packed_mixed.build"):
            for row_index in _packed_mixed_row_order(batch):
                op = batch.ops[row_index]
                mode = batch.op_modes[row_index]
                req_id = int(op["req_id"])
                current_context = {
                    "row_index": row_index,
                    "req_id": req_id,
                    "kind": op.get("kind"),
                    "mode": mode.value,
                    "pos_range": op.get("pos_range"),
                    "new_block_ids_len": len(op.get("new_block_ids") or []),
                }
                if mode in {ForwardMode.EXTEND, ForwardMode.DECODE}:
                    state = owner.interleaved_image_state(req_id)
                    cache = state.cond
                    owner._extend_cache_blocks(cache, dict(op))
                    owner._ensure_host_cache(cache)
                    if cache.past is None:
                        return False
                    current_context.update(
                        {
                            "cache_blocks_before_sync": len(getattr(cache, "block_ids", []) or []),
                            "past_blocks_before_sync": len(
                                getattr(cache.past, "block_ids", []) or []
                            ),
                            "past_length_before_sync": int(getattr(cache.past, "length", 0)),
                            "cache_t_index_before_sync": int(getattr(cache, "t_index", -1)),
                        }
                    )
                    _sync_host_cache_blocks(owner, cache, request_states, req_id, dict(op))
                    current_context.update(
                        {
                            "cache_blocks_after_sync": len(getattr(cache, "block_ids", []) or []),
                            "past_blocks_after_sync": len(
                                getattr(cache.past, "block_ids", []) or []
                            ),
                            "past_length_after_sync": int(getattr(cache.past, "length", 0)),
                            "cache_t_index_after_sync": int(getattr(cache, "t_index", -1)),
                        }
                    )
                    hydrate_cached_prefix_from_op(cache, op)
                    current_context.update(
                        {
                            "past_blocks_after_hydrate": len(
                                getattr(cache.past, "block_ids", []) or []
                            ),
                            "past_length_after_hydrate": int(getattr(cache.past, "length", 0)),
                            "cache_t_index_after_hydrate": int(getattr(cache, "t_index", -1)),
                        }
                    )
                    tokens = list(op.get("token_ids") or [])
                    if not tokens:
                        tokens = [int(owner.eos_id or 0)]
                    pos = op.get("pos_range") or [
                        cache.t_index + 1,
                        cache.t_index + 1 + len(tokens),
                    ]
                    start = int(pos[0])
                    q_len = len(tokens)
                    cache.past.ensure_capacity(cache.past.length + q_len)
                    persistent_cache = cache.past
                    staged_cache = persistent_cache
                    if first_pool is not None and persistent_cache.pool is not first_pool:
                        with profile_range("uniserve.packed_mixed.text_cache_stage"):
                            staged_cache = owner._stage_text_cache_for_forward(
                                persistent_cache,
                                target_pool=first_pool,
                                end_len=persistent_cache.length + q_len,
                                pending_prefix_copies=text_stage_prefix_copies,
                            )
                    pool = staged_cache.pool
                    if not owner._same_kv_pool(pool, first_pool):
                        return False
                    first_pool = pool if first_pool is None else first_pool
                    ids = _forward_text_input_ids(
                        op,
                        req_id=req_id,
                        tokens=tokens,
                        request_states=request_states,
                        device=device,
                    )
                    last_input_token = int(tokens[-1])
                    if str(op.get("token_source") or "wire") == "last_sampled":
                        relay = getattr(request_states.get(req_id), "decode_relay", None)
                        relay_token = getattr(relay, "token_id", None)
                        if relay_token is not None:
                            last_input_token = int(relay_token)
                    with profile_range("uniserve.packed_mixed.text_segment"):
                        owner._add_text_forward_segment(
                            builder=builder,
                            kv_segments=kv_segments,
                            row_index=row_index,
                            req_id=req_id,
                            op=dict(op),
                            mode=mode,
                            cache=SimpleNamespace(past=staged_cache),
                            q_len=q_len,
                            start_pos=start,
                            device=device,
                        )
                    pending_text_rows.append(
                        _PendingTextBuildRow(
                            row_index=row_index,
                            q_len=q_len,
                            input_ids=ids,
                            persistent_cache=persistent_cache,
                            staged_cache=staged_cache,
                            base_len=int(persistent_cache.length),
                            last_input_token=last_input_token,
                        )
                    )
                elif mode is ForwardMode.DENOISE:
                    flush_text_rows()
                    step = plan.denoise_step_for_row(row_index)
                    cfg_plan = text_image_cfg_plan(step)
                    plan.set_denoise_cfg_plan(row_index, cfg_plan)
                    for branch_index, branch in enumerate(cfg_plan.branches):
                        img = step.extra["img"]
                        with profile_range("uniserve.packed_mixed.denoise_branch_inputs"):
                            indexes, cache = owner._denoise_branch_inputs(img, branch)
                        if cache is None or getattr(cache, "pool", None) is None:
                            return False
                        owner._wait_gen_cache_ready(cache)
                        pool = cache.pool
                        if not owner._same_kv_pool(pool, first_pool):
                            return False
                        first_pool = pool if first_pool is None else first_pool
                        q_len = int(step.extra["image_embeds"].shape[1])
                        ensure_capacity = getattr(cache, "ensure_capacity", None)
                        if callable(ensure_capacity):
                            ensure_capacity(int(cache.length) + q_len)
                        if indexes is None or tuple(indexes.shape) != (3, q_len):
                            return False
                        segment_start = _append_packed_chunk(
                            embed_chunks,
                            indicator_spans,
                            step.extra["image_embeds"].reshape(q_len, -1),
                            image_tokens=True,
                            device=device,
                        )
                        with profile_range("uniserve.packed_mixed.denoise_segment"):
                            owner._add_denoise_forward_segment(
                                builder=builder,
                                kv_segments=kv_segments,
                                row_index=row_index,
                                req_id=req_id,
                                op=dict(op),
                                cache=cache,
                                indexes=indexes,
                                q_len=q_len,
                                branch_index=branch_index,
                                device=device,
                            )
                        plan.add_denoise_slot(
                            row_index=row_index,
                            step=step,
                            segment_start=segment_start,
                            q_len=q_len,
                            branch=branch,
                        )
                elif mode is ForwardMode.COMMIT:
                    # Commit rows are part of the same admitted mixed batch, but they do
                    # not contribute hidden-state segments. The model hook decodes them
                    # after this packed text/denoise forward has updated latent state.
                    pass
                else:
                    return False
                current_context = None
            flush_text_rows()
        if text_stage_prefix_copies:
            with profile_range("uniserve.packed_mixed.text_prefix_stage"):
                copy_paged_text_cache_spans(
                    text_stage_prefix_copies,
                    num_layers=owner.num_layers,
                    missing_message="cannot stage mixed forward prefix without a paged source cache",
                )
            for span in text_stage_prefix_copies:
                owner._mark_forward_staging_advanced(
                    span.target,
                    span.source,
                    int(span.start) + int(span.length),
                )
        timing.stop("build_ms", build_start)
        ctx.record_component_elapsed("packed_mixed_build", build_stats_start)
        if first_pool is None or not embed_chunks:
            return False
        stream_stats_start = ctx.component_timer_start()
        stream_start = timing.start()
        with profile_range("uniserve.packed_mixed.stream_build"):
            forward_stream = builder.build(device=device)
            kv_view = ForwardPagedKVView(first_pool, kv_segments)
        timing.stop("stream_build_ms", stream_start)
        ctx.record_component_elapsed("packed_mixed_stream_build", stream_stats_start)
        text_kv_promotions = [
            PagedTextCacheSpanCopy(
                source=staged_cache,
                target=persistent_cache,
                start=base_len,
                length=q_len,
            )
            for (
                _row_index,
                _start,
                q_len,
                persistent_cache,
                staged_cache,
                base_len,
                _last_input_token,
            ) in plan.text_result_slots
            if staged_cache is not persistent_cache
        ]
        graph_text_kv_promotions = (
            tuple(text_kv_promotions)
            if packed_mixed_graph_promotions_supported(text_kv_promotions)
            else ()
        )
        decoder_stats_start = ctx.component_timer_start()
        decoder_start = timing.start()
        decoder_component_start = timing.component_snapshot(ctx)
        with profile_range("uniserve.packed_mixed.decoder_input_pack"):
            packed_embeds = torch.cat(embed_chunks, dim=0)
            packed_indicators = _packed_indicator_tensor(
                owner,
                indicator_spans,
                device=device,
            )
        hidden = None
        if allow_graph:
            hidden = maybe_run_packed_mixed_graph(
                owner,
                packed_embeds,
                image_gen_indicators=packed_indicators,
                forward_stream=forward_stream,
                kv_view=kv_view,
                text_kv_promotions=graph_text_kv_promotions,
            )
        graph_promoted_text_kv = hidden is not None and bool(graph_text_kv_promotions)
        if hidden is None:
            if require_graph:
                return False
            hidden = owner.packed_decoder_forward(
                packed_embeds,
                image_gen_indicators=packed_indicators,
                indexes=forward_stream.indexes,
                forward_stream=forward_stream,
                kv_view=kv_view,
            )
        timing.add_component_deltas(
            ctx,
            decoder_component_start,
            (
                "packed_decoder_input_norm",
                "packed_decoder_qkv",
                "packed_decoder_attention",
                "packed_decoder_o_proj",
                "packed_decoder_attn_block",
                "packed_decoder_mlp",
            ),
        )
        timing.stop("decoder_ms", decoder_start)
        ctx.record_component_elapsed("packed_mixed_decoder", decoder_stats_start)
        text_stats_start = ctx.component_timer_start()
        text_start = timing.start()
        text_logits_by_row: dict[int, torch.Tensor] = {}
        text_outputs_by_row: dict[int, Any] = {}
        text_device_tokens_by_row: dict[int, torch.Tensor] = {}
        text_sample_indices_by_row: dict[int, int] = {}
        deferred_text_sampling: Any | None = None
        text_logits_for_result: torch.Tensor | None = None
        text_postprocess_entries: list[TextPostprocessEntry] = []
        if plan.text_result_slots:
            with profile_range("uniserve.packed_mixed.text_last_hidden"):
                text_hidden = torch.stack(
                    [
                        hidden[int(start) + int(q_len) - 1]
                        for _row_index, start, q_len, *_ in plan.text_result_slots
                    ],
                    dim=0,
                )
            with profile_range("uniserve.packed_mixed.text_logits"):
                text_logits = owner.packed_text_logits(text_hidden.unsqueeze(0)).squeeze(0)
            with profile_range("uniserve.packed_mixed.text_logits_scatter"):
                for offset, (row_index, *_rest) in enumerate(plan.text_result_slots):
                    text_logits_by_row[int(row_index)] = text_logits[offset : offset + 1].unsqueeze(
                        0
                    )
            if return_forward_result:
                text_logits_for_result = text_logits.reshape(len(plan.text_result_slots), -1)
            else:
                sample_logits: list[torch.Tensor] = []
                sampling_params: list[dict[str, Any]] = []
                sampling_generators: list[torch.Generator] = []
                with profile_range("uniserve.packed_mixed.text_sampling_inputs"):
                    for row_index, *_rest in plan.text_result_slots:
                        req_id = int(batch.ops[row_index]["req_id"])
                        state = request_states.get(req_id)
                        sampling_params.append(dict(state.sampling or {}))
                        sampling_generators.append(
                            state.device_rng(
                                text_logits.device,
                                stream="text_sampling",
                            )
                        )
                        sample_logits.append(
                            text_logits_by_row[int(row_index)].reshape(-1, text_logits.shape[-1])[
                                -1
                            ]
                        )
                can_defer_text_cpu = bool(defer_text_cpu_results)
                with profile_range("uniserve.packed_mixed.text_sampling"):
                    sampled = apply_sampling_batched_with_device_tokens(
                        torch.stack(sample_logits, dim=0),
                        sampling_params,
                        [[] for _ in sample_logits],
                        [None for _ in sample_logits],
                        [None for _ in sample_logits],
                        generators=sampling_generators,
                        defer_cpu=can_defer_text_cpu,
                    )
                if is_deferred_sampling_result(sampled) and can_defer_text_cpu:
                    deferred_text_sampling = sampled
                    for sample_index, (row_index, *_rest) in enumerate(plan.text_result_slots):
                        text_sample_indices_by_row[int(row_index)] = sample_index
                        text_device_tokens_by_row[int(row_index)] = sampled.device_tokens[
                            sample_index : sample_index + 1
                        ]
                else:
                    immediate_sampling = finalize_sampling_result(sampled)
                    for sample_index, (row_index, *_rest) in enumerate(plan.text_result_slots):
                        req_id = int(batch.ops[row_index]["req_id"])
                        sample = immediate_sampling.samples[sample_index]
                        top_logprobs = (
                            [
                                (int(item[0]), float(item[1]), int(item[2]))
                                for item in sample.top_logprobs
                            ]
                            if sample.top_logprobs is not None
                            else None
                        )
                        text_outputs_by_row[int(row_index)] = TextTokenOutput(
                            req_id=req_id,
                            sampled_token_id=int(sample.token_id),
                            sampled_logprob=sample.logprob,
                            top_logprobs=top_logprobs,
                        )
                        text_device_tokens_by_row[int(row_index)] = sampled.device_tokens[
                            sample_index : sample_index + 1
                        ]
        with profile_range("uniserve.packed_mixed.burst_position_stage"):
            burst_position_tensors_by_row = _forward_burst_position_tensors(
                owner,
                plan,
                device=device,
            )
        if text_kv_promotions and not graph_promoted_text_kv and not return_forward_result:
            with profile_range("uniserve.packed_mixed.text_kv_promote"):
                copy_paged_text_cache_spans(
                    text_kv_promotions,
                    num_layers=owner.num_layers,
                    missing_message="forward text K/V span is missing from staged cache",
                )
        with profile_range("uniserve.packed_mixed.text_result_publish"):
            for logits_index, (
                row_index,
                start,
                q_len,
                persistent_cache,
                staged_cache,
                base_len,
                last_input_token,
            ) in enumerate(plan.text_result_slots):
                op = batch.ops[row_index]
                req_id = int(op["req_id"])
                logits = text_logits_by_row[int(row_index)]
                state = owner.interleaved_image_state(req_id)
                position_id = int((op.get("pos_range") or [0, state.cond.t_index + q_len])[1])
                new_len = int(base_len) + int(q_len)
                if return_forward_result:
                    promotion = (
                        PagedTextCacheSpanCopy(
                            source=staged_cache,
                            target=persistent_cache,
                            start=base_len,
                            length=q_len,
                        )
                        if staged_cache is not persistent_cache and not graph_promoted_text_kv
                        else None
                    )
                    text_postprocess_entries.append(
                        TextPostprocessEntry(
                            row_index=int(row_index),
                            req_id=req_id,
                            logits_index=int(logits_index),
                            position_id=position_id,
                            kv_new_length=new_len,
                            last_input_token=int(last_input_token),
                            interleaved_state=state,
                            persistent_cache=persistent_cache,
                            staged_cache=staged_cache,
                            kv_promotion=promotion,
                            num_layers=int(owner.num_layers),
                            mark_staging_advanced=owner._mark_forward_staging_advanced,
                        )
                    )
                    continue
                state.cond.t_index = position_id - 1
                state.cond.last_logits = logits
                state.cond.last_token_id = int(last_input_token)
                persistent_cache.length = new_len
                if staged_cache is not persistent_cache:
                    owner._mark_forward_staging_advanced(staged_cache, persistent_cache, new_len)
                req_state = request_states.get(req_id)
                output = text_outputs_by_row.get(int(row_index))
                if output is None:
                    if deferred_text_sampling is None:
                        raise invalid_descriptor("packed mixed text sampling result is missing")
                    sample_index = text_sample_indices_by_row[int(row_index)]
                    _store_forward_sampled_token_relay(
                        req_state,
                        token_id=None,
                        device=device,
                        position_id=position_id,
                        token_tensor=text_device_tokens_by_row[int(row_index)],
                        position_tensor=burst_position_tensors_by_row.get(int(row_index)),
                    )
                    output = DeferredTextSeqResult(
                        req_id=req_id,
                        row=sample_index,
                        state=req_state,
                        sampling_result=deferred_text_sampling,
                        relay_token_tensor=req_state.decode_relay.token_tensor,
                    )
                    text_outputs_by_row[int(row_index)] = output
                else:
                    _store_forward_sampled_token_relay(
                        req_state,
                        token_id=int(output.sampled_token_id),
                        device=device,
                        position_id=position_id,
                        token_tensor=text_device_tokens_by_row.get(int(row_index)),
                        position_tensor=burst_position_tensors_by_row.get(int(row_index)),
                    )
                plan.set_text_result(row_index, output)
        timing.stop("text_post_ms", text_start)
        ctx.record_component_elapsed("packed_mixed_text_post", text_stats_start)
        velocity_stats_start = ctx.component_timer_start()
        velocity_start = timing.start()
        branch_velocities: dict[int, dict[str, torch.Tensor]] = {}
        denoise_velocities: dict[DenoiseBranchKey, torch.Tensor] = {}
        denoise_branch_counts: dict[int, int] = {}
        with profile_range("uniserve.packed_mixed.velocity"):
            for row_index, step, start, q_len, branch in plan.denoise_result_slots:
                img = step.extra["img"]
                with profile_range("uniserve.packed_mixed.velocity_branch"):
                    velocity = owner.packed_hidden_to_velocity(
                        hidden[start : start + q_len].unsqueeze(0),
                        step.t,
                        step.latent,
                        image_token_num=img.token_h * img.token_w,
                        image_size=(img.width, img.height),
                    )
                branch_velocities.setdefault(row_index, {})[branch] = velocity
                branch_id = denoise_branch_counts.get(int(row_index), 0)
                denoise_velocities[DenoiseBranchKey(int(row_index), branch_id)] = velocity
                denoise_branch_counts[int(row_index)] = branch_id + 1
        timing.stop("velocity_ms", velocity_start)
        ctx.record_component_elapsed("packed_mixed_velocity", velocity_stats_start)
        if return_forward_result:
            denoise_updates: dict[int, DenoisePostprocessEntry] = {}
            for result_index, step in plan.denoise_steps:
                cfg_plan = plan.denoise_cfg_plan_for_row(result_index)

                def combine_velocity(
                    values: Mapping[Any, torch.Tensor],
                    current_plan: CfgPlan = cfg_plan,
                ) -> torch.Tensor:
                    return current_plan.combine(values)

                def accept_update(
                    updated: torch.Tensor,
                    current_owner: Any = owner,
                    current_step: TextImageDenoiseStep = step,
                ) -> None:
                    current_owner.accept_denoise_update(current_step, updated)

                denoise_updates[int(result_index)] = DenoisePostprocessEntry(
                    row_index=int(result_index),
                    req_id=int(step.req_id),
                    step_index=int(step.step_index),
                    total_steps=int(step.total_steps),
                    branch_names=tuple(cfg_plan.branches),
                    latent=step.latent,
                    t=step.t,
                    t_next=step.t_next,
                    combine_velocity=combine_velocity,
                    accept_update=accept_update,
                )
            timing.stop("total_ms", total_start)
            timing.log(
                batch=batch,
                forward_stream=forward_stream,
                embed_chunks=embed_chunks,
                text_result_slots=plan.text_result_slots,
                denoise_result_slots=plan.denoise_result_slots,
                kv_segments=kv_segments,
            )
            return ForwardResult(
                text_logits=text_logits_for_result,
                text_postprocess=tuple(text_postprocess_entries),
                denoise_velocities=denoise_velocities,
                denoise_updates=denoise_updates,
            )
        update_stats_start = ctx.component_timer_start()
        update_start = timing.start()
        with profile_range("uniserve.packed_mixed.denoise_update"):
            for result_index, step in plan.denoise_steps:
                velocities = branch_velocities.get(result_index)
                if not velocities:
                    return False
                velocity = plan.denoise_cfg_plan_for_row(result_index).combine(velocities)
                updated = euler_step(step.latent, velocity, step.t, step.t_next)
                owner.accept_denoise_update(step, updated)
                plan.set_denoise_result(result_index, step)
        timing.stop("denoise_update_ms", update_start)
        ctx.record_component_elapsed("packed_mixed_denoise_update", update_stats_start)
        timing.stop("total_ms", total_start)
        timing.log(
            batch=batch,
            forward_stream=forward_stream,
            embed_chunks=embed_chunks,
            text_result_slots=plan.text_result_slots,
            denoise_result_slots=plan.denoise_result_slots,
            kv_segments=kv_segments,
        )
        return True
    except Exception as exc:
        if current_context:
            logger.exception(
                "packed mixed forward failed for an admitted mixed batch: context=%s",
                current_context,
            )
        else:
            logger.exception("packed mixed forward failed for an admitted mixed batch")
        raise capability_mismatch(
            "packed mixed forward failed for an admitted mixed batch",
            details={"cause_type": type(exc).__name__, "cause": str(exc)[:500]},
        ) from exc


class _PackedMixedTiming:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.enabled = _PACKED_MIXED_TIMING
        self.sync = _PACKED_MIXED_TIMING_SYNC
        self.values: dict[str, float] = {}

    def start(self) -> int:
        if not self.enabled:
            return 0
        self._sync()
        return time.perf_counter_ns()

    def stop(self, key: str, start_ns: int) -> None:
        if not self.enabled or not start_ns:
            return
        self._sync()
        self.values[key] = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    def component_snapshot(self, ctx: Any) -> dict[str, int]:
        if not self.enabled:
            return {}
        stats = getattr(ctx, "stats", None)
        component_ns = getattr(stats, "component_ns", None)
        return dict(component_ns) if isinstance(component_ns, dict) else {}

    def add_component_deltas(
        self,
        ctx: Any,
        start: Mapping[str, int],
        keys: Sequence[str],
    ) -> None:
        if not self.enabled:
            return
        stats = getattr(ctx, "stats", None)
        component_ns = getattr(stats, "component_ns", None)
        if not isinstance(component_ns, dict):
            return
        for key in keys:
            before = int(start.get(key, 0))
            after = int(component_ns.get(key, 0))
            if after > before:
                self.values[f"{key}_ms"] = (after - before) / 1_000_000.0

    def log(
        self,
        *,
        batch: UniForwardBatch,
        forward_stream: Any,
        embed_chunks: Sequence[torch.Tensor],
        text_result_slots: Sequence[tuple[Any, ...]],
        denoise_result_slots: Sequence[tuple[Any, ...]],
        kv_segments: Sequence[Any],
    ) -> None:
        if not self.enabled:
            return
        mode_counts: dict[str, int] = {}
        for mode in batch.op_modes:
            key = mode.value
            mode_counts[key] = mode_counts.get(key, 0) + 1
        text_tokens = sum(int(slot[2]) for slot in text_result_slots)
        image_tokens = sum(int(slot[3]) for slot in denoise_result_slots)
        payload = {
            "rows": len(batch.ops),
            "mode_counts": mode_counts,
            "tokens": sum(int(chunk.shape[0]) for chunk in embed_chunks),
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "text_rows": len(text_result_slots),
            "denoise_rows": len({int(slot[0]) for slot in denoise_result_slots}),
            "denoise_segments": len(denoise_result_slots),
            "kv_segments": len(kv_segments),
            "fully_visible": bool(getattr(forward_stream, "fully_visible", False)),
            **self.values,
        }
        logger.info("packed_mixed_timing %s", json.dumps(payload, sort_keys=True))

    def _sync(self) -> None:
        if self.sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _sync_host_cache_blocks(
    owner: Any,
    cache: Any,
    request_states: Any,
    req_id: int,
    op: Mapping[str, Any] | None = None,
) -> None:
    past = getattr(cache, "past", None)
    if past is None or getattr(past, "pool", None) is not getattr(owner, "kv_pool", None):
        return
    get_state = getattr(request_states, "get", None)
    if not callable(get_state):
        return
    state = get_state(int(req_id))
    if op is not None:
        block_ids = state_block_ids_for_op(op, state)
    else:
        block_ids = [int(block_id) for block_id in (getattr(state, "block_ids", []) or [])]
    if not block_ids:
        return
    current = [int(block_id) for block_id in (getattr(cache, "block_ids", []) or [])]
    if len(block_ids) < len(current):
        return
    cache.block_ids = block_ids
    past.set_blocks(cache.block_ids)


def _append_packed_chunk(
    embed_chunks: list[torch.Tensor],
    indicator_spans: list[tuple[int, bool]],
    embeds: torch.Tensor,
    *,
    image_tokens: bool,
    device: torch.device,
) -> int:
    del device
    start = sum(chunk.shape[0] for chunk in embed_chunks)
    q_len = int(embeds.shape[0])
    embed_chunks.append(embeds)
    indicator_spans.append((q_len, bool(image_tokens)))
    return start


def _packed_indicator_tensor(
    owner: Any,
    spans: Sequence[tuple[int, bool]],
    *,
    device: torch.device,
) -> torch.Tensor:
    total = sum(int(length) for length, _value in spans)
    if total <= 0:
        raise invalid_descriptor("packed mixed indicators must not be empty")
    if device.type != "cuda":
        chunks = [
            torch.full((int(length),), bool(value), dtype=torch.bool, device=device)
            for length, value in spans
            if int(length) > 0
        ]
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=0)
    stager = getattr(owner, "_packed_mixed_indicator_stager", None)
    if not isinstance(stager, TextTensorStager):
        stager = TextTensorStager(ring_depth=3)
        setattr(owner, "_packed_mixed_indicator_stager", stager)
    slot = stager.next_slot()
    cpu = slot.bool_buffer("packed_mixed_indicators", total, pin=True)
    offset = 0
    for length, value in spans:
        length = int(length)
        if length <= 0:
            continue
        cpu[offset : offset + length].fill_(bool(value))
        offset += length
    indicators = slot.device_buffer(
        "packed_mixed_indicators",
        total,
        dtype=torch.bool,
        device=device,
    )
    indicators.copy_(cpu, non_blocking=is_pinned(cpu))
    return indicators


def _packed_mixed_row_order(batch: UniForwardBatch) -> list[int]:
    if not _PACKED_MIXED_SORT_BY_MODALITY:
        return list(range(len(batch.ops)))
    text_rows: list[int] = []
    denoise_rows: list[int] = []
    commit_rows: list[int] = []
    other_rows: list[int] = []
    for row_index, mode in enumerate(batch.op_modes):
        if mode in {ForwardMode.EXTEND, ForwardMode.DECODE}:
            text_rows.append(row_index)
        elif mode is ForwardMode.DENOISE:
            denoise_rows.append(row_index)
        elif mode is ForwardMode.COMMIT:
            commit_rows.append(row_index)
        else:
            other_rows.append(row_index)
    return text_rows + denoise_rows + commit_rows + other_rows


def _store_forward_sampled_token_relay(
    state: Any,
    *,
    token_id: int | None,
    device: torch.device,
    position_id: int | None = None,
    token_tensor: torch.Tensor | None = None,
    position_tensor: torch.Tensor | None = None,
) -> None:
    relay = getattr(state, "decode_relay", None)
    if relay is None:
        return
    if token_tensor is None:
        if token_id is None:
            raise invalid_descriptor("packed mixed relay publish requires a token tensor")
        token_tensor = torch.tensor([int(token_id)], dtype=torch.long, device=device)
    else:
        token_tensor = token_tensor.reshape(1).to(device=device, dtype=torch.long)
    _DECODE_RELAY.publish_sample(
        state,
        token_id=None if token_id is None else int(token_id),
        token_tensor=token_tensor,
    )
    if position_id is not None:
        if position_tensor is None:
            position_tensor = torch.tensor([int(position_id)], dtype=torch.long, device=device)
        else:
            position_tensor = position_tensor.reshape(1).to(device=device, dtype=torch.long)
        _DECODE_RELAY.publish_position(
            state,
            position_id=int(position_id),
            position_tensor=position_tensor,
        )


def _forward_burst_position_tensors(
    owner: Any,
    plan: PackedForwardPlan,
    *,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    rows: list[int] = []
    position_ids: list[int] = []
    for (
        row_index,
        _start,
        q_len,
        _persistent,
        _staged,
        base_len,
        _last_token,
    ) in plan.text_result_slots:
        op = plan.batch.ops[int(row_index)]
        if plan.batch.op_modes[int(row_index)] is not ForwardMode.DECODE:
            continue
        if int(op.get("decode_token_count") or 1) <= 1:
            continue
        pos = op.get("pos_range") or [int(base_len), int(base_len) + int(q_len)]
        if not isinstance(pos, Sequence) or len(pos) != 2:
            raise invalid_descriptor("decode burst op.pos_range must be [start, end]")
        rows.append(int(row_index))
        position_ids.append(int(pos[1]))
    if not rows:
        return {}
    if device.type != "cuda":
        positions = torch.tensor(position_ids, dtype=torch.long, device=device)
    else:
        stager = getattr(owner, "_packed_mixed_burst_position_stager", None)
        if not isinstance(stager, TextTensorStager):
            stager = TextTensorStager(ring_depth=3)
            setattr(owner, "_packed_mixed_burst_position_stager", stager)
        slot = stager.next_slot()
        cpu = slot.long_buffer("packed_mixed_burst_positions", len(position_ids), pin=True)
        fill_cpu_ints(cpu, position_ids)
        positions = slot.device_buffer(
            "packed_mixed_burst_positions",
            len(position_ids),
            dtype=torch.long,
            device=device,
        )
        positions.copy_(cpu, non_blocking=is_pinned(cpu))
    return {row: positions[offset : offset + 1] for offset, row in enumerate(rows)}


def _forward_text_input_ids(
    op: Mapping[str, Any],
    *,
    req_id: int,
    tokens: Sequence[int],
    request_states: Any,
    device: torch.device,
) -> torch.Tensor:
    source = str(op.get("token_source") or "wire")
    if source not in {"wire", "last_sampled"}:
        raise invalid_descriptor(f"unsupported text token_source {source!r}")
    if source == "wire":
        return torch.tensor(list(tokens), dtype=torch.long, device=device)
    if len(tokens) != 1:
        raise invalid_descriptor(
            "decode op requested token_source='last_sampled' but does not have exactly one token"
        )
    state = request_states.get(int(req_id))
    token = _DECODE_RELAY.consume_token(
        state,
        expected_token_id=None,
        device=device,
        token_source=source,
        require=True,
    )
    assert token is not None
    return token.reshape(1)
