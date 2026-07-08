"""Packed mixed text/denoise forward driver."""
from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from ..contracts.batches import UniForwardBatch
from ..contracts.forward_context import get_forward_context
from ..contracts.forward_mode import ForwardMode
from ..contracts.outputs import TextTokenOutput
from ..foundation.env import env_flag
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..nn.sampler import apply_sampling_batched_with_device_tokens
from ..runtime.forward_batch_builder import state_block_ids_for_op
from ..runtime.paged_text_cache import PagedTextCache, copy_paged_text_cache_span
from .denoise_driver import TextImageDenoiseStep, combine_text_image_velocity, text_image_branches
from .forward_stream import (
    ForwardPagedKVSegment,
    ForwardPagedKVView,
    ForwardStreamBuilder,
)
from .interleaved_text_stepper import hydrate_cached_prefix_from_op

__all__ = ["run_packed_mixed_forward"]

logger = logging.getLogger(__name__)

_PACKED_MIXED_TIMING = env_flag("UNISERVE_PACKED_MIXED_TIMING")
_PACKED_MIXED_TIMING_SYNC = env_flag("UNISERVE_PACKED_MIXED_TIMING_SYNC")


def run_packed_mixed_forward(
    owner,
    batch: UniForwardBatch,
    request_states: Any,
    denoise_steps: list[tuple[int, TextImageDenoiseStep]],
    results: list[Any],
) -> bool:
    if owner.model is None:
        return False
    builder = ForwardStreamBuilder()
    kv_segments: list[ForwardPagedKVSegment] = []
    embed_chunks: list[torch.Tensor] = []
    indicators: list[torch.Tensor] = []
    text_result_slots: list[tuple[int, int, int, PagedTextCache, PagedTextCache, int, int]] = []
    denoise_result_slots: list[tuple[int, TextImageDenoiseStep, int, int]] = []
    first_pool = owner._forward_target_pool(denoise_steps)
    device = torch.device(str(owner.device))
    current_context: dict[str, Any] | None = None
    ctx = get_forward_context()
    timing = _PackedMixedTiming(device)
    total_start = timing.start()

    try:
        build_stats_start = ctx.component_timer_start()
        build_start = timing.start()
        for row_index, op in enumerate(batch.ops):
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
                        "past_blocks_before_sync": len(getattr(cache.past, "block_ids", []) or []),
                        "past_length_before_sync": int(getattr(cache.past, "length", 0)),
                        "cache_t_index_before_sync": int(getattr(cache, "t_index", -1)),
                    }
                )
                _sync_host_cache_blocks(owner, cache, request_states, req_id, dict(op))
                current_context.update(
                    {
                        "cache_blocks_after_sync": len(getattr(cache, "block_ids", []) or []),
                        "past_blocks_after_sync": len(getattr(cache.past, "block_ids", []) or []),
                        "past_length_after_sync": int(getattr(cache.past, "length", 0)),
                        "cache_t_index_after_sync": int(getattr(cache, "t_index", -1)),
                    }
                )
                hydrate_cached_prefix_from_op(cache, op)
                current_context.update(
                    {
                        "past_blocks_after_hydrate": len(getattr(cache.past, "block_ids", []) or []),
                        "past_length_after_hydrate": int(getattr(cache.past, "length", 0)),
                        "cache_t_index_after_hydrate": int(getattr(cache, "t_index", -1)),
                    }
                )
                tokens = list(op.get("token_ids") or [])
                if not tokens:
                    tokens = [int(owner.eos_id or 0)]
                pos = op.get("pos_range") or [cache.t_index + 1, cache.t_index + 1 + len(tokens)]
                start = int(pos[0])
                q_len = len(tokens)
                cache.past.ensure_capacity(cache.past.length + q_len)
                persistent_cache = cache.past
                staged_cache = persistent_cache
                if first_pool is not None and persistent_cache.pool is not first_pool:
                    staged_cache = owner._stage_text_cache_for_forward(
                        persistent_cache,
                        target_pool=first_pool,
                        end_len=persistent_cache.length + q_len,
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
                embeds = owner.packed_text_embeddings(ids).reshape(q_len, -1)
                segment_start = _append_packed_chunk(
                    embed_chunks,
                    indicators,
                    embeds,
                    image_tokens=False,
                    device=device,
                )
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
                text_result_slots.append(
                    (
                        row_index,
                        segment_start,
                        q_len,
                        persistent_cache,
                        staged_cache,
                        int(persistent_cache.length),
                        last_input_token,
                    )
                )
            elif mode is ForwardMode.DENOISE:
                step = next(step for result_index, step in denoise_steps if result_index == row_index)
                branches = text_image_branches(step)
                for branch_index, branch in enumerate(branches):
                    img = step.extra["img"]
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
                        indicators,
                        step.extra["image_embeds"].reshape(q_len, -1),
                        image_tokens=True,
                        device=device,
                    )
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
                    denoise_result_slots.append((row_index, step, segment_start, q_len))
            elif mode is ForwardMode.COMMIT:
                # Commit rows are part of the same admitted mixed batch, but they do
                # not contribute hidden-state segments. The model hook decodes them
                # after this packed text/denoise forward has updated latent state.
                pass
            else:
                return False
            current_context = None
        timing.stop("build_ms", build_start)
        ctx.record_component_elapsed("packed_mixed_build", build_stats_start)
        if first_pool is None or not embed_chunks:
            return False
        stream_stats_start = ctx.component_timer_start()
        stream_start = timing.start()
        forward_stream = builder.build(device=device)
        kv_view = ForwardPagedKVView(first_pool, kv_segments)
        timing.stop("stream_build_ms", stream_start)
        ctx.record_component_elapsed("packed_mixed_stream_build", stream_stats_start)
        decoder_stats_start = ctx.component_timer_start()
        decoder_start = timing.start()
        decoder_component_start = timing.component_snapshot(ctx)
        packed_embeds = torch.cat(embed_chunks, dim=0)
        packed_indicators = torch.cat(indicators, dim=0)
        # No CUDA-graph fast path here, deliberately: a mixed und+gen batch only
        # exists under concurrency, where text-row base lengths grow every step
        # and neighbouring requests re-plan the shared FlashInfer prefill
        # wrapper — a captured replay would run against stale/foreign attention
        # plans (observed as cudaGraphLaunch segfaults on both TP ranks). The
        # denoise-step graph (frozen per-image geometry + exclusive wrapper)
        # remains the sanctioned graph path for the denoise loop.
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
        text_outputs_by_row: dict[int, TextTokenOutput] = {}
        text_device_tokens_by_row: dict[int, torch.Tensor] = {}
        if text_result_slots:
            text_hidden = torch.cat(
                [hidden[start:start + q_len] for row_index, start, q_len, *_ in text_result_slots],
                dim=0,
            )
            text_logits = owner.packed_text_logits(text_hidden.unsqueeze(0)).squeeze(0)
            offset = 0
            for row_index, _start, q_len, *_rest in text_result_slots:
                next_offset = offset + int(q_len)
                text_logits_by_row[int(row_index)] = text_logits[offset:next_offset].unsqueeze(0)
                offset = next_offset
            sample_logits: list[torch.Tensor] = []
            sampling_params: list[dict[str, Any]] = []
            for row_index, *_rest in text_result_slots:
                req_id = int(batch.ops[row_index]["req_id"])
                state = request_states.get(req_id)
                sampling_params.append(dict(state.sampling or {}))
                sample_logits.append(text_logits_by_row[int(row_index)].reshape(-1, text_logits.shape[-1])[-1])
            sampled = apply_sampling_batched_with_device_tokens(
                torch.stack(sample_logits, dim=0),
                sampling_params,
                [[] for _ in sample_logits],
                [None for _ in sample_logits],
                [None for _ in sample_logits],
            )
            for sample_index, (row_index, *_rest) in enumerate(text_result_slots):
                req_id = int(batch.ops[row_index]["req_id"])
                sample = sampled.samples[sample_index]
                text_outputs_by_row[int(row_index)] = TextTokenOutput(
                    req_id=req_id,
                    sampled_token_id=int(sample.token_id),
                    sampled_logprob=sample.logprob,
                    top_logprobs=sample.top_logprobs,
                )
                text_device_tokens_by_row[int(row_index)] = sampled.device_tokens[
                    sample_index:sample_index + 1
                ]
        for (
            row_index,
            start,
            q_len,
            persistent_cache,
            staged_cache,
            base_len,
            last_input_token,
        ) in text_result_slots:
            op = batch.ops[row_index]
            req_id = int(op["req_id"])
            logits = text_logits_by_row[int(row_index)]
            results[row_index] = text_outputs_by_row[int(row_index)]
            if staged_cache is not persistent_cache:
                copy_paged_text_cache_span(
                    staged_cache,
                    persistent_cache,
                    start=base_len,
                    length=q_len,
                    num_layers=owner.num_layers,
                    missing_message="forward text K/V span is missing from staged cache",
                )
            state = owner.interleaved_image_state(req_id)
            position_id = int((op.get("pos_range") or [0, state.cond.t_index + q_len])[1])
            state.cond.t_index = position_id - 1
            state.cond.last_logits = logits
            state.cond.last_token_id = int(last_input_token)
            new_len = int(base_len) + int(q_len)
            persistent_cache.length = new_len
            if staged_cache is not persistent_cache:
                owner._mark_forward_staging_advanced(staged_cache, persistent_cache, new_len)
            _store_forward_sampled_token_relay(
                request_states.get(req_id),
                token_id=int(results[row_index].sampled_token_id),
                device=device,
                position_id=position_id,
                token_tensor=text_device_tokens_by_row.get(int(row_index)),
            )
        timing.stop("text_post_ms", text_start)
        ctx.record_component_elapsed("packed_mixed_text_post", text_stats_start)
        velocity_stats_start = ctx.component_timer_start()
        velocity_start = timing.start()
        branch_velocities: dict[int, dict[str, torch.Tensor]] = {}
        for row_index, step, start, q_len in denoise_result_slots:
            img = step.extra["img"]
            branch = text_image_branches(step)[len(branch_velocities.setdefault(row_index, {}))]
            velocity = owner.packed_hidden_to_velocity(
                hidden[start:start + q_len].unsqueeze(0),
                step.t,
                step.latent,
                image_token_num=img.token_h * img.token_w,
                image_size=(img.width, img.height),
            )
            branch_velocities[row_index][branch] = velocity
        timing.stop("velocity_ms", velocity_start)
        ctx.record_component_elapsed("packed_mixed_velocity", velocity_stats_start)
        update_stats_start = ctx.component_timer_start()
        update_start = timing.start()
        for result_index, step in denoise_steps:
            velocities = branch_velocities.get(result_index)
            if not velocities:
                return False
            velocity = combine_text_image_velocity(step, velocities)
            from ..nn.diffusion import euler_step

            updated = euler_step(step.latent, velocity, step.t, step.t_next)
            owner.accept_denoise_update(step, updated)
            results[result_index] = {
                "req_id": step.req_id,
                "denoise_done": step.step_index + 1 >= step.total_steps,
                "num_steps_done": step.step_index + 1,
            }
        timing.stop("denoise_update_ms", update_start)
        ctx.record_component_elapsed("packed_mixed_denoise_update", update_stats_start)
        timing.stop("total_ms", total_start)
        timing.log(
            batch=batch,
            forward_stream=forward_stream,
            embed_chunks=embed_chunks,
            text_result_slots=text_result_slots,
            denoise_result_slots=denoise_result_slots,
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
    indicators: list[torch.Tensor],
    embeds: torch.Tensor,
    *,
    image_tokens: bool,
    device: torch.device,
) -> int:
    start = sum(chunk.shape[0] for chunk in embed_chunks)
    q_len = int(embeds.shape[0])
    embed_chunks.append(embeds)
    indicators.append(torch.full((q_len,), bool(image_tokens), dtype=torch.bool, device=device))
    return start


def _store_forward_sampled_token_relay(
    state: Any,
    *,
    token_id: int,
    device: torch.device,
    position_id: int | None = None,
    token_tensor: torch.Tensor | None = None,
) -> None:
    relay = getattr(state, "decode_relay", None)
    if relay is None:
        return
    if token_tensor is None:
        token_tensor = torch.tensor([int(token_id)], dtype=torch.long, device=device)
    else:
        token_tensor = token_tensor.reshape(1).to(device=device, dtype=torch.long)
    if token_tensor.device.type == "cuda":
        token_tensor.record_stream(torch.cuda.current_stream(token_tensor.device))
    relay.token_id = int(token_id)
    relay.token_tensor = token_tensor
    if position_id is not None:
        position_tensor = torch.tensor([int(position_id)], dtype=torch.long, device=device)
        if position_tensor.device.type == "cuda":
            position_tensor.record_stream(torch.cuda.current_stream(position_tensor.device))
        relay.position_id = int(position_id)
        relay.position_tensor = position_tensor


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
    relay = getattr(getattr(state, "decode_relay", None), "token_tensor", None)
    if not isinstance(relay, torch.Tensor) or relay.dtype != torch.long or relay.device != device:
        raise invalid_descriptor(
            "decode op requested token_source='last_sampled' but the relay tensor is unavailable"
        )
    return relay.reshape(1)
