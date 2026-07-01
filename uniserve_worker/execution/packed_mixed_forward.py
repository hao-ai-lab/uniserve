"""Packed mixed text/denoise forward driver."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from ..contracts.batches import UniForwardBatch
from ..contracts.forward_mode import ForwardMode
from ..foundation.errors import capability_mismatch
from ..runtime.paged_text_cache import PagedTextCache
from .denoise_driver import TextImageDenoiseStep, combine_text_image_velocity, text_image_branches
from .forward_stream import ForwardPagedKVSegment, ForwardPagedKVView, ForwardStreamBuilder

__all__ = ["run_packed_mixed_forward"]


def run_packed_mixed_forward(
    owner,
    batch: UniForwardBatch,
    request_states: Any,
    denoise_steps: list[tuple[int, TextImageDenoiseStep]],
    results: list[Any],
) -> bool:
    if owner.model is None or not denoise_steps:
        return False
    language = owner.model.language_model
    decoder = language.model
    embed = language.get_input_embeddings()
    builder = ForwardStreamBuilder()
    kv_segments: list[ForwardPagedKVSegment] = []
    embed_chunks: list[torch.Tensor] = []
    indicators: list[torch.Tensor] = []
    text_result_slots: list[tuple[int, int, int, PagedTextCache, PagedTextCache, int, int]] = []
    denoise_result_slots: list[tuple[int, TextImageDenoiseStep, int, int]] = []
    first_pool = owner._forward_target_pool(denoise_steps)
    staged_text_caches: list[PagedTextCache] = []
    device = torch.device(str(owner.device))

    try:
        for row_index, op in enumerate(batch.ops):
            mode = batch.op_modes[row_index]
            req_id = int(op["req_id"])
            if mode in {ForwardMode.EXTEND, ForwardMode.DECODE}:
                state = owner.interleaved_image_state(req_id)
                cache = state.cond
                owner._extend_cache_blocks(cache, dict(op))
                owner._ensure_host_cache(cache)
                if cache.past is None:
                    return False
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
                    staged_text_caches.append(staged_cache)
                pool = staged_cache.pool
                if not owner._same_kv_pool(pool, first_pool):
                    return False
                first_pool = pool if first_pool is None else first_pool
                ids = owner._forward_text_input_ids(
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
                embeds = embed(ids).reshape(q_len, -1)
                segment_start = owner._append_packed_chunk(
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
                    if indexes is None or tuple(indexes.shape) != (3, q_len):
                        return False
                    segment_start = owner._append_packed_chunk(
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
            else:
                return False
        if first_pool is None or not embed_chunks:
            return False
        forward_stream = builder.build(device=device)
        kv_view = ForwardPagedKVView(first_pool, kv_segments)
        hidden = decoder.forward_packed_visible(
            torch.cat(embed_chunks, dim=0),
            image_gen_indicators=torch.cat(indicators, dim=0),
            indexes=forward_stream.indexes,
            forward_stream=forward_stream,
            kv_view=kv_view,
        )
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
            logits = language.lm_head(hidden[start:start + q_len].unsqueeze(0))
            results[row_index] = owner._sample_text_logits(req_id, logits, request_states)
            if staged_cache is not persistent_cache:
                owner._copy_cache_span(
                    staged_cache,
                    persistent_cache,
                    start=base_len,
                    length=q_len,
                    num_layers=owner.num_layers,
                )
            state = owner.interleaved_image_state(req_id)
            position_id = int((op.get("pos_range") or [0, state.cond.t_index + q_len])[1])
            state.cond.t_index = position_id - 1
            state.cond.last_logits = logits
            state.cond.last_token_id = int(last_input_token)
            owner._store_forward_sampled_token_relay(
                request_states.get(req_id),
                token_id=int(results[row_index].sampled_token_id),
                device=device,
                position_id=position_id,
            )
            if state.cond.past is not None:
                state.cond.past.length += q_len
        branch_velocities: dict[int, dict[str, torch.Tensor]] = {}
        for row_index, step, start, q_len in denoise_result_slots:
            img = step.extra["img"]
            branch = text_image_branches(step)[len(branch_velocities.setdefault(row_index, {}))]
            velocity = owner.model._t2i_hidden_to_velocity(
                hidden[start:start + q_len].unsqueeze(0),
                step.t,
                step.latent,
                image_token_num=img.token_h * img.token_w,
                image_size=(img.width, img.height),
            )
            branch_velocities[row_index][branch] = velocity
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
        return True
    except Exception as exc:
        raise capability_mismatch("packed mixed forward failed for an admitted mixed batch") from exc
    finally:
        for staged in staged_text_caches:
            owner._release_scratch_cache(staged)
