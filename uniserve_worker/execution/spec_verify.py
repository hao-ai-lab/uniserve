"""System-owned speculative-decode verification.

Relocated from the model: building the verify forward, running it, and applying
the accept rule (greedy device fast-path / sglang target-only / sequential
target-sample) are system policy over the thin ``model.forward``. The model only
returns per-position verify logits for the ``TARGET_VERIFY`` forward; the system
groups the draft rows, runs one rectangular forward per draft length, and decides
acceptance + the next KV length here.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from ..contracts.batches import UniForwardBatch
from ..contracts.forward_context import get_forward_context, use_forward_context
from ..contracts.forward_mode import ForwardMode
from ..contracts.forward_stats import ForwardStats
from ..contracts.op_kinds import TARGET_VERIFY_UND
from ..contracts.outputs import TextTokenOutput
from ..foundation.errors import invalid_descriptor
from ..nn.sampler import apply_sampling_batched_with_device_tokens
from ..spec import speculative_sample_target_only

if TYPE_CHECKING:
    from ..runtime.request_state import RequestStateTable
    from .text_driver import TextDriver

__all__ = ["verify_speculative_tokens"]

_KV_LANE = "text"
_SAMPLED_TOKEN_DEVICE_KEY = "sampled_token_device"
_SAMPLED_POSITION_DEVICE_KEY = "sampled_position_device"


def verify_speculative_tokens(
    driver: "TextDriver",
    model: Any,
    text: Any,
    request_states: "RequestStateTable",
) -> list[TextTokenOutput]:
    """Verify the draft tokens on ``text`` against the target model.

    Groups the per-row draft sequences by length, runs one rectangular
    ``target_verify`` forward per length through the system-built attention plan,
    and applies the accept rule per row — system policy over the thin model.
    """

    if text.mode != ForwardMode.DECODE:
        raise invalid_descriptor("spec_token_ids are only supported on decode text ops")
    if any(len(tokens) != 1 for tokens in text.token_ids):
        raise invalid_descriptor("speculative decode requires one committed input token per op")
    if driver.builder is None or driver.kv_pool is None:
        raise invalid_descriptor("speculative verify requires the system ForwardBatchBuilder and KV pool")
    ctx = get_forward_context()
    device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))

    grouped: dict[int, list[tuple[int, dict[str, Any], tuple[int, ...]]]] = {}
    for idx, (op, spec, tokens, pos_range) in enumerate(
        zip(text.ops, text.spec_token_ids, text.token_ids, text.pos_ranges)
    ):
        length = 1 + len(spec)
        start = int(pos_range[0])
        committed = [int(token) for token in tokens]
        extended = dict(op)
        extended["kind"] = TARGET_VERIFY_UND
        extended["token_ids"] = committed + [int(token) for token in spec]
        extended["pos_range"] = [start, start + length]
        extended.pop("spec_token_ids", None)
        grouped.setdefault(length, []).append((idx, extended, tuple(int(token) for token in spec)))

    results: list[dict[str, Any] | None] = [None] * len(text.ops)
    builder, kv_pool = driver._system_forward_runtime()
    for length, rows in grouped.items():
        extended_ops = [op for _, op, _ in rows]
        verify_text = UniForwardBatch.from_ops(extended_ops).as_text()
        fb = builder.build_text(
            verify_text,
            device=device,
            kv_pool=kv_pool,
            request_states=request_states,
        )
        if fb.input_ids is None or fb.positions is None:
            raise invalid_descriptor("speculative verification batch is missing text inputs")
        input_ids = fb.input_ids.reshape(len(rows), length)
        positions = fb.positions.reshape(len(rows), length)
        with use_forward_context(
            replace(ctx, attention_metadata=fb.attn_metadata, kv_pool=kv_pool)
        ):
            logits = model.forward(input_ids, positions, fb)
        for row, (original_idx, op, spec) in enumerate(rows):
            results[original_idx] = _verify_spec_row(
                logits[row],
                op,
                spec,
                request_states.get(int(op["req_id"])),
                stats=ctx.stats,
            )
    if any(result is None for result in results):
        raise invalid_descriptor("speculative verification missed a result row")

    cleaned: list[TextTokenOutput] = []
    for op, result in zip(text.ops, (r for r in results if r is not None)):
        cleaned.append(_finalize_row(op, dict(result), request_states))
    return cleaned


def _finalize_row(
    op: dict[str, Any],
    result: dict[str, Any],
    request_states: "RequestStateTable",
) -> TextTokenOutput:
    token_tensor = result.pop(_SAMPLED_TOKEN_DEVICE_KEY, None)
    position_tensor = result.pop(_SAMPLED_POSITION_DEVICE_KEY, None)
    req_id = result.get("req_id")
    token_id = result.get("sampled_token_id")
    if isinstance(token_tensor, torch.Tensor) and isinstance(req_id, int) and isinstance(token_id, int):
        state = request_states.get(int(req_id))
        device_token = token_tensor.detach().reshape(1)
        if device_token.device.type == "cuda":
            device_token.record_stream(torch.cuda.current_stream(device_token.device))
        state.decode_relay.token_id = int(token_id)
        state.decode_relay.token_tensor = device_token
        if isinstance(position_tensor, torch.Tensor):
            pos_range = op.get("pos_range") or (0, 0)
            accepted = int(result.get("num_accepted_tokens") or 0)
            device_position = position_tensor.detach().reshape(1)
            if device_position.device.type == "cuda":
                device_position.record_stream(torch.cuda.current_stream(device_position.device))
            state.decode_relay.position_id = int(pos_range[0]) + 1 + accepted
            state.decode_relay.position_tensor = device_position
    return _spec_verify_token_output(result)


def _verify_spec_row(
    logits: torch.Tensor,
    op: dict[str, Any],
    spec: tuple[int, ...],
    state: Any,
    *,
    stats: ForwardStats | None = None,
) -> dict[str, Any]:
    sampling = dict(state.sampling or {})
    recent = list(op.get("recent_tokens") or [])
    allowed = op.get("allowed_tokens")
    suppress = op.get("suppress_tokens")
    n_logprobs = int(sampling.get("n_logprobs", 0) or 0)
    if _can_use_greedy_spec_verify_fast_path(sampling, recent, allowed, suppress):
        return _verify_spec_row_greedy(logits, op, spec, state, stats=stats)
    if _can_use_sglang_target_only_spec_verify(sampling):
        return _verify_spec_row_target_only(
            logits,
            op,
            spec,
            state,
            sampling,
            recent=recent,
            allowed=allowed,
            suppress=suppress,
            stats=stats,
        )
    return _verify_spec_row_sequential(
        logits,
        op,
        spec,
        state,
        sampling,
        recent=recent,
        allowed=allowed,
        suppress=suppress,
        n_logprobs=n_logprobs,
        stats=stats,
    )


def _verify_spec_row_greedy(
    logits: torch.Tensor,
    op: dict[str, Any],
    spec: tuple[int, ...],
    state: Any,
    *,
    stats: ForwardStats | None,
) -> dict[str, Any]:
    chosen = torch.argmax(logits, dim=-1)
    accepted = _accepted_greedy_prefix(chosen, spec)
    sampled_token_tensor = chosen[accepted:accepted + 1]
    sampled_token = int(sampled_token_tensor.detach().to("cpu").item())
    next_pos = _advance_spec_kv(state, op, accepted)
    _record_spec_verify_stats(stats, len(spec), accepted, "greedy_device")
    return {
        "req_id": int(op["req_id"]),
        "sampled_token_id": sampled_token,
        "num_accepted_tokens": int(accepted),
        _SAMPLED_TOKEN_DEVICE_KEY: sampled_token_tensor,
        _SAMPLED_POSITION_DEVICE_KEY: chosen.new_full((1,), next_pos, dtype=torch.long),
    }


def _verify_spec_row_target_only(
    logits: torch.Tensor,
    op: dict[str, Any],
    spec: tuple[int, ...],
    state: Any,
    sampling: dict[str, Any],
    *,
    recent: list[int],
    allowed: Any,
    suppress: Any,
    stats: ForwardStats | None,
) -> dict[str, Any]:
    spec_sample = speculative_sample_target_only(
        logits[: len(spec) + 1].reshape(len(spec) + 1, -1),
        spec,
        sampling,
        recent=recent,
        allowed=allowed,
        suppress=suppress,
    )
    accepted = int(spec_sample.num_accepted_tokens)
    next_pos = _advance_spec_kv(state, op, accepted)
    _record_spec_verify_stats(stats, len(spec), accepted, "sglang_target_only")
    return {
        "req_id": int(op["req_id"]),
        "sampled_token_id": int(spec_sample.sampled_token_id),
        "num_accepted_tokens": int(accepted),
        _SAMPLED_TOKEN_DEVICE_KEY: spec_sample.sampled_token_device,
        _SAMPLED_POSITION_DEVICE_KEY: spec_sample.sampled_token_device.new_full(
            (1,), next_pos, dtype=torch.long
        ),
    }


def _verify_spec_row_sequential(
    logits: torch.Tensor,
    op: dict[str, Any],
    spec: tuple[int, ...],
    state: Any,
    sampling: dict[str, Any],
    *,
    recent: list[int],
    allowed: Any,
    suppress: Any,
    n_logprobs: int,
    stats: ForwardStats | None,
) -> dict[str, Any]:
    accepted = 0
    sampled_token = None
    sampled_token_tensor = None
    sampled_logprob = None
    top_logprobs = None
    for pos in range(len(spec) + 1):
        requested_logprobs = n_logprobs if n_logprobs > 0 else 0
        sampling_for_pos = dict(sampling)
        sampling_for_pos["n_logprobs"] = requested_logprobs
        sampling_result = apply_sampling_batched_with_device_tokens(
            logits[pos].reshape(1, -1),
            [sampling_for_pos],
            [recent],
            [allowed],
            [suppress],
        )
        token, logprob, top = sampling_result.samples[0]
        if pos < len(spec) and int(token) == int(spec[pos]):
            accepted += 1
            recent.append(int(token))
            continue
        sampled_token = int(token)
        sampled_token_tensor = sampling_result.device_tokens[:1]
        sampled_logprob = logprob
        top_logprobs = top
        break
    if sampled_token is None:
        raise invalid_descriptor("speculative verification did not produce a sampled token")
    next_pos = _advance_spec_kv(state, op, accepted)
    _record_spec_verify_stats(stats, len(spec), accepted, "sequential_target_sample")
    result: dict[str, Any] = {
        "req_id": int(op["req_id"]),
        "sampled_token_id": sampled_token,
        "num_accepted_tokens": int(accepted),
    }
    if sampled_logprob is not None:
        result["sampled_logprob"] = sampled_logprob
    if top_logprobs:
        result["top_logprobs"] = top_logprobs
    if sampled_token_tensor is not None:
        result[_SAMPLED_TOKEN_DEVICE_KEY] = sampled_token_tensor
        result[_SAMPLED_POSITION_DEVICE_KEY] = sampled_token_tensor.new_full((1,), next_pos, dtype=torch.long)
    return result


def _accepted_greedy_prefix(chosen: torch.Tensor, spec: tuple[int, ...]) -> int:
    if not spec:
        return 0
    draft = torch.tensor(spec, dtype=chosen.dtype, device=chosen.device)
    mismatch = torch.nonzero(chosen[: len(spec)] != draft, as_tuple=False)
    return int(mismatch[0].item()) if int(mismatch.numel()) > 0 else len(spec)


def _advance_spec_kv(state: Any, op: dict[str, Any], accepted: int) -> int:
    base_len = int((op.get("pos_range") or (0, 0))[0])
    next_pos = base_len + 1 + int(accepted)
    state.set_kv_length(next_pos, lane=_KV_LANE)
    return next_pos


def _spec_verify_token_output(result: dict[str, Any]) -> TextTokenOutput:
    req_id = result.get("req_id")
    token_id = result.get("sampled_token_id")
    if not isinstance(req_id, int) or isinstance(req_id, bool):
        raise invalid_descriptor("speculative verify output req_id must be an integer")
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise invalid_descriptor("speculative verify output sampled_token_id must be an integer")
    num_accepted = result.get("num_accepted_tokens")
    return TextTokenOutput(
        req_id=int(req_id),
        sampled_token_id=int(token_id),
        sampled_logprob=result.get("sampled_logprob"),
        top_logprobs=result.get("top_logprobs") or None,
        num_accepted_tokens=int(num_accepted) if num_accepted is not None else None,
    )


def _can_use_greedy_spec_verify_fast_path(
    sampling: dict[str, Any],
    recent: list[int],
    allowed: Any,
    suppress: Any,
) -> bool:
    if allowed or suppress or sampling.get("logit_bias"):
        return False
    if _generated_logprobs_requested(sampling):
        return False
    if float(sampling.get("temperature", 0.0) or 0.0) > 0.0:
        return False
    if float(sampling.get("min_p", 0.0) or 0.0) > 0.0:
        return False
    if int(sampling.get("top_k", 0) or 0) > 0:
        return False
    if float(sampling.get("top_p", 1.0) or 1.0) < 1.0:
        return False
    repetition = float(sampling.get("repetition_penalty", 1.0) or 1.0)
    frequency = float(sampling.get("frequency_penalty", 0.0) or 0.0)
    presence = float(sampling.get("presence_penalty", 0.0) or 0.0)
    return not recent or (repetition == 1.0 and frequency == 0.0 and presence == 0.0)


def _can_use_sglang_target_only_spec_verify(sampling: dict[str, Any]) -> bool:
    if _generated_logprobs_requested(sampling):
        return False
    return float(sampling.get("temperature", 0.0) or 0.0) > 0.0


def _generated_logprobs_requested(sampling: dict[str, Any]) -> bool:
    return (
        bool(sampling.get("return_logprobs", False))
        or int(sampling.get("n_logprobs", 0) or 0) > 0
        or bool(sampling.get("logprob_token_ids"))
    )


def _record_spec_verify_stats(
    stats: ForwardStats | None,
    draft_tokens: int,
    accepted_tokens: int,
    path: str,
) -> None:
    if stats is None:
        return
    draft = max(0, int(draft_tokens))
    accepted = max(0, min(int(accepted_tokens), draft))
    stats.spec_verify_rows += 1
    stats.spec_verify_draft_tokens += draft
    stats.spec_verify_accepted_tokens += accepted
    stats.spec_verify_rejected_tokens += max(0, draft - accepted)
    stats.spec_verify_committed_tokens += accepted + 1
    stats.record_spec_path(path)
