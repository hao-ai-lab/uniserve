"""Whole-batch packed text and denoise execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ....contracts.batches import UniForwardBatch
from ....contracts.forward_mode import ForwardMode
from ....foundation.errors import capability_mismatch, invalid_descriptor
from ....foundation.profiling import profile_range
from ....nn.sampler import (
    apply_sampling_batched_with_device_tokens,
    finalize_sampling_result,
    is_deferred_sampling_result,
)
from ....runtime.host_staging import fill_cpu_ints, is_pinned
from ....runtime.tensor_staging import TextTensorStager
from ...denoise_driver import DenoiseDriver, TextImageDenoiseStep
from ...text_decode_relay import TextDecodeRelay
from ...text_driver import sample_logits_result
from ..deferred_text import (
    DeferredDecodeBurstSeqResult,
    DeferredTerminalDecodeBurstSeqResult,
    DeferredTextSeqResult,
)
from .packed_visible import run_packed_mixed_forward, run_packed_visible_forward_result

__all__ = ["PackedVisibleBatchAdapter"]

_DECODE_RELAY = TextDecodeRelay()
_RELAY_PLACEHOLDER_TOKEN_ID = -1


class PackedVisibleBatchAdapter:
    """Executes one admitted text/denoise batch without mode splitting."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def execute(
        self,
        batch: UniForwardBatch,
        *,
        request_states: Any,
        defer_text_cpu_results: bool = False,
    ) -> Any:
        results: list[Any] = [None] * len(batch.ops)
        denoise_steps: list[tuple[int, TextImageDenoiseStep]] = []
        commit_rows: list[tuple[int, int, dict[str, Any]]] = []
        has_burst_rows = False
        with profile_range("uniserve.packed_visible.prepare_ops"):
            for row_index, op in enumerate(batch.ops):
                mode = batch.op_modes[row_index]
                req_id = int(op["req_id"])
                if (
                    int(op.get("decode_token_count") or 1) > 1
                    or int(op.get("denoise_step_count") or 1) > 1
                ):
                    has_burst_rows = True
                if mode is ForwardMode.DENOISE:
                    step = self.owner.prepare_denoise(
                        request_states.get(req_id),
                        dict(op),
                    )
                    denoise_steps.append((row_index, step))
                elif mode is ForwardMode.COMMIT:
                    commit_rows.append((row_index, req_id, dict(op)))
                elif mode not in {ForwardMode.EXTEND, ForwardMode.DECODE}:
                    raise capability_mismatch(
                        "packed visible batch contains an unsupported mode",
                        details={"mode": mode.value},
                    )
        for _row_index, step in denoise_steps:
            extra = getattr(step, "extra", None)
            image = extra.get("img") if isinstance(extra, dict) else None
            residual_state = getattr(image, "residual_cache", None)
            if residual_state is not None:
                residual_state.invalidate()
        if not commit_rows and not has_burst_rows:
            result = run_packed_visible_forward_result(
                self.owner,
                batch,
                request_states,
                denoise_steps,
                defer_text_cpu_results=defer_text_cpu_results,
                allow_graph=True,
                require_graph=True,
            )
            if result is not None:
                return result
        if not run_packed_mixed_forward(
            self.owner,
            batch,
            request_states,
            denoise_steps,
            results,
            defer_text_cpu_results=defer_text_cpu_results,
            require_graph=True,
        ):
            raise capability_mismatch("admitted mixed batch did not execute as one packed graph")
        self._complete_decode_bursts(
            batch,
            request_states,
            results,
            defer_final_cpu_results=defer_text_cpu_results,
        )
        self._complete_denoise_bursts(batch, request_states, results)
        for row_index, req_id, op in commit_rows:
            state = request_states.get(req_id)
            decoded = self.owner.decode_image(
                getattr(state, "latent", None),
                req_id=req_id,
                state=state,
                op=op,
            )
            out = dict(decoded)
            logits = out.pop("logits", None)
            if logits is not None:
                sampled = sample_logits_result(
                    req_id=req_id,
                    state=state,
                    logits=logits,
                    op=op,
                )
                sampled.pop("req_id", None)
                out.update(sampled)
            results[row_index] = out
        return results

    def _complete_decode_bursts(
        self,
        batch: UniForwardBatch,
        request_states: Any,
        results: list[Any],
        *,
        defer_final_cpu_results: bool = False,
    ) -> None:
        active: list[dict[str, Any]] = []
        for row_index, op in enumerate(batch.ops):
            if batch.op_modes[row_index] is not ForwardMode.DECODE:
                continue
            requested = int(op.get("decode_token_count") or 1)
            if requested <= 1:
                continue
            stop_ids = {int(token) for token in (op.get("decode_stop_token_ids") or [])}
            terminal_stop = not stop_ids or op.get("decode_stop_terminal") is True
            position = op.get("pos_range") or [0, 0]
            if not isinstance(position, Sequence) or len(position) != 2:
                raise invalid_descriptor("decode burst op.pos_range must be [start, end]")
            active.append(
                {
                    "row_index": row_index,
                    "op": dict(op),
                    "tokens": [],
                    "requested": requested,
                    "launched": 1,
                    "last_op": dict(op),
                    "pending": results[row_index],
                    "pending_tokens": [results[row_index]] if terminal_stop else [],
                    "stop_ids": stop_ids,
                    "terminal_stop": terminal_stop,
                    "done": False,
                }
            )
        while any(not bool(item["done"]) for item in active):
            iter_ops: list[dict[str, Any]] = []
            iter_items: list[dict[str, Any]] = []
            for item in active:
                if bool(item["done"]) or int(item["launched"]) >= int(item["requested"]):
                    continue
                op = dict(item["op"])
                op["new_block_ids"] = []
                state = request_states.get(int(op["req_id"]))
                if _attach_decode_relay_input(op, state):
                    op["token_ids"] = [_RELAY_PLACEHOLDER_TOKEN_ID]
                    op["token_source"] = "last_sampled"
                else:
                    pending = item.get("pending")
                    if pending is None:
                        raise invalid_descriptor("decode burst relay input is missing")
                    token = _sampled_token_id(
                        pending,
                        profile_name="uniserve.packed_burst.relay_input_materialize",
                    )
                    item["tokens"].append(token)
                    item["pending"] = None
                    if token in item["stop_ids"]:
                        item["done"] = True
                        continue
                    op["token_ids"] = [token]
                    op.pop("token_source", None)
                    op.pop("token_tensor", None)
                next_pos = _decode_op_next_pos(item["last_op"])
                op["pos_range"] = [next_pos, next_pos + 1]
                op["decode_token_count"] = None
                op["decode_stop_token_ids"] = []
                item["last_op"] = op
                iter_ops.append(op)
                iter_items.append(item)
            if not iter_ops:
                break
            followup_results = self._run_decode_burst_graph_followup(
                iter_ops,
                request_states,
                defer_cpu_results=True,
            )
            if followup_results is None:
                raise capability_mismatch(
                    "packed decode burst follow-up requires CUDA graph coverage"
                )
            for item, output in zip(iter_items, followup_results, strict=True):
                previous = item.get("pending")
                item["pending"] = output
                item["launched"] = int(item["launched"]) + 1
                if bool(item.get("terminal_stop")):
                    item["pending_tokens"].append(output)
                    continue
                if previous is None:
                    continue
                token = _sampled_token_id(
                    previous,
                    profile_name="uniserve.packed_burst.stop_check",
                )
                item["tokens"].append(token)
                if token in item["stop_ids"]:
                    item["pending"] = None
                    item["done"] = True
        for item in active:
            if bool(item.get("terminal_stop")):
                results[int(item["row_index"])] = _decode_burst_terminal_result(
                    item["op"],
                    item["pending_tokens"],
                    item["stop_ids"],
                    defer_cpu=bool(defer_final_cpu_results),
                )
                item["pending"] = None
                continue
            pending = item.get("pending")
            if pending is not None:
                results[int(item["row_index"])] = _decode_burst_result_with_pending(
                    item["op"],
                    item["tokens"],
                    pending,
                    defer_cpu=bool(defer_final_cpu_results),
                )
                item["pending"] = None
                continue
            results[int(item["row_index"])] = _decode_burst_result(
                item["op"],
                item["tokens"],
            )

    def _run_decode_burst_graph_followup(
        self,
        ops: Sequence[Mapping[str, Any]],
        request_states: Any,
        *,
        defer_cpu_results: bool,
    ) -> list[Any] | None:
        driver = self.owner._text_driver()
        run_graph = getattr(driver, "try_run_decode_graph_logits_batch", None)
        if not callable(run_graph):
            return None
        logits_rows = run_graph(ops)
        if logits_rows is None:
            return None
        if len(logits_rows) != len(ops):
            raise invalid_descriptor("decode graph follow-up logits row count must match ops")
        rows = [_coerce_logits_row(logits) for logits in logits_rows]
        if not rows:
            return []
        logits_batch = torch.stack(rows, dim=0)
        sampling_params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        generators: list[torch.Generator] = []
        for op in ops:
            state = request_states.get(int(op["req_id"]))
            sampling_params.append(dict(state.sampling or {}))
            recent.append(op.get("recent_tokens") or [])
            allowed.append(op.get("allowed_tokens"))
            suppress.append(op.get("suppress_tokens"))
            generators.append(state.device_rng(logits_batch.device, stream="text_sampling"))
        sampled = apply_sampling_batched_with_device_tokens(
            logits_batch,
            sampling_params,
            recent,
            allowed,
            suppress,
            generators=generators,
            defer_cpu=defer_cpu_results,
        )
        device = logits_batch.device
        position_ids, position_tensors = _decode_followup_position_tensors(
            self.owner,
            ops,
            device=device,
        )
        if is_deferred_sampling_result(sampled) and defer_cpu_results:
            deferred_outputs: list[Any] = []
            for row, op in enumerate(ops):
                req_id = int(op["req_id"])
                state = request_states.get(req_id)
                _store_decode_followup_relay(
                    state,
                    token_id=None,
                    device=device,
                    position_id=position_ids[row],
                    token_tensor=sampled.device_tokens[row : row + 1],
                    position_tensor=position_tensors[row],
                )
                deferred_outputs.append(
                    DeferredTextSeqResult(
                        req_id=req_id,
                        row=row,
                        state=state,
                        sampling_result=sampled,
                        relay_token_tensor=sampled.device_tokens[row : row + 1],
                    )
                )
            return deferred_outputs
        sampled = finalize_sampling_result(sampled)
        immediate_outputs: list[Any] = []
        for row, op in enumerate(ops):
            req_id = int(op["req_id"])
            sample = sampled.samples[row]
            token_id = int(sample.token_id)
            _store_decode_followup_relay(
                request_states.get(req_id),
                token_id=token_id,
                device=device,
                position_id=position_ids[row],
                token_tensor=sampled.device_tokens[row : row + 1],
                position_tensor=position_tensors[row],
            )
            top_logprobs = (
                [(int(item[0]), float(item[1]), int(item[2])) for item in sample.top_logprobs]
                if sample.top_logprobs is not None
                else None
            )
            immediate_outputs.append(
                {
                    "req_id": req_id,
                    "sampled_token_id": token_id,
                    "sampled_logprob": sample.logprob,
                    "top_logprobs": top_logprobs,
                }
            )
        return immediate_outputs

    def _complete_denoise_bursts(
        self,
        batch: UniForwardBatch,
        request_states: Any,
        results: list[Any],
    ) -> None:
        items: list[tuple[int, Any, dict[str, Any]]] = []
        row_indexes: list[int] = []
        for row_index, op in enumerate(batch.ops):
            if batch.op_modes[row_index] is not ForwardMode.DENOISE:
                continue
            requested = int(op.get("denoise_step_count") or 1)
            if requested <= 1 or _output_bool(results[row_index], "denoise_done"):
                continue
            remaining = requested - 1
            req_id = int(op["req_id"])
            followup = dict(op)
            followup["timestep_idx"] = _output_int(results[row_index], "num_steps_done")
            followup["denoise_step_count"] = remaining
            row_indexes.append(row_index)
            items.append((req_id, request_states.get(req_id), followup))
        if not items:
            return
        outputs = DenoiseDriver().step_many(items, self.owner, graph_mode="require")
        for row_index, output in zip(row_indexes, outputs, strict=True):
            results[row_index] = output


def _output_value(output: Any, field: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(field)
    if hasattr(output, field):
        return getattr(output, field)
    finalize = getattr(output, "finalize", None)
    if callable(finalize):
        finalized = finalize()
        if isinstance(finalized, Mapping):
            return finalized.get(field)
    to_seq_result = getattr(output, "to_seq_result", None)
    if callable(to_seq_result):
        return dict(to_seq_result()).get(field)
    raise invalid_descriptor(f"packed mixed output does not expose {field!r}")


def _output_int(output: Any, field: str) -> int:
    value = _output_value(output, field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_descriptor(f"packed mixed output field {field!r} must be an integer")
    return int(value)


def _output_bool(output: Any, field: str) -> bool:
    value = _output_value(output, field)
    if not isinstance(value, bool):
        raise invalid_descriptor(f"packed mixed output field {field!r} must be a boolean")
    return bool(value)


def _sampled_token_id(output: Any, *, profile_name: str) -> int:
    materialize = getattr(output, "materialize_sampled_token_id", None)
    with profile_range(profile_name):
        token = (
            int(materialize())
            if callable(materialize)
            else _output_int(output, "sampled_token_id")
        )
    if token < 0:
        raise invalid_descriptor("sampled_token_id must be non-negative")
    return token


def _decode_burst_result(op: Mapping[str, Any], tokens: Sequence[int]) -> dict[str, Any]:
    if not tokens:
        raise invalid_descriptor("decode burst did not produce a sampled token")
    token_ids = [int(token) for token in tokens]
    return {
        "req_id": int(op["req_id"]),
        "sampled_token_id": token_ids[-1],
        "sampled_token_ids": token_ids,
    }


def _decode_burst_result_with_pending(
    op: Mapping[str, Any],
    tokens: Sequence[int],
    pending: Any,
    *,
    defer_cpu: bool,
) -> dict[str, Any] | DeferredDecodeBurstSeqResult:
    materialize = getattr(pending, "materialize_sampled_token_id", None)
    ready = getattr(pending, "ready", None)
    if defer_cpu and callable(materialize) and callable(ready):
        return DeferredDecodeBurstSeqResult(
            req_id=int(op["req_id"]),
            prefix_token_ids=tokens,
            pending=pending,
        )
    token = _sampled_token_id(
        pending,
        profile_name="uniserve.packed_burst.finalize_pending",
    )
    return _decode_burst_result(op, [*tokens, token])


def _decode_burst_terminal_result(
    op: Mapping[str, Any],
    pending_tokens: Sequence[Any],
    stop_ids: set[int],
    *,
    defer_cpu: bool,
) -> dict[str, Any] | DeferredTerminalDecodeBurstSeqResult:
    if not pending_tokens:
        raise invalid_descriptor("decode burst did not produce a sampled token")
    if defer_cpu and all(
        callable(getattr(pending, "materialize_sampled_token_id", None))
        for pending in pending_tokens
    ):
        return DeferredTerminalDecodeBurstSeqResult(
            req_id=int(op["req_id"]),
            pending_tokens=pending_tokens,
            stop_token_ids=stop_ids,
        )
    token_ids: list[int] = []
    for pending in pending_tokens:
        token = _sampled_token_id(
            pending,
            profile_name="uniserve.packed_burst.finalize_terminal",
        )
        token_ids.append(token)
        if token in stop_ids:
            break
    return _decode_burst_result(op, token_ids)


def _attach_decode_relay_input(op: dict[str, Any], state: Any) -> bool:
    relay = getattr(state, "decode_relay", None)
    token_tensor = getattr(relay, "token_tensor", None)
    if not isinstance(token_tensor, torch.Tensor) or token_tensor.dtype != torch.long:
        return False
    op["token_tensor"] = token_tensor
    return True


def _decode_op_next_pos(op: Mapping[str, Any]) -> int:
    position = op.get("pos_range") or [0, 0]
    if not isinstance(position, Sequence) or len(position) != 2:
        raise invalid_descriptor("decode burst op.pos_range must be [start, end]")
    return int(position[1])


def _coerce_logits_row(logits: Any) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise invalid_descriptor("decode graph follow-up must return logits tensors")
    if logits.ndim == 0:
        raise invalid_descriptor("decode graph follow-up logits must have a vocabulary dimension")
    return logits.reshape(-1, logits.shape[-1])[-1]


def _store_decode_followup_relay(
    state: Any,
    *,
    token_id: int | None,
    device: torch.device,
    position_id: int,
    token_tensor: torch.Tensor,
    position_tensor: torch.Tensor | None,
) -> None:
    relay = getattr(state, "decode_relay", None)
    if relay is None:
        return
    _DECODE_RELAY.publish_sample(
        state,
        token_id=None if token_id is None else int(token_id),
        token_tensor=token_tensor.reshape(1).to(device=device, dtype=torch.long),
    )
    position_tensor = (
        torch.tensor([int(position_id)], dtype=torch.long, device=device)
        if position_tensor is None
        else position_tensor.reshape(1).to(device=device, dtype=torch.long)
    )
    _DECODE_RELAY.publish_position(
        state,
        position_id=int(position_id),
        position_tensor=position_tensor,
    )


def _decode_followup_position_tensors(
    owner: Any,
    ops: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> tuple[list[int], list[torch.Tensor]]:
    position_ids = [int((op.get("pos_range") or [0, 0])[1]) for op in ops]
    if not position_ids:
        return [], []
    if device.type != "cuda":
        positions = torch.tensor(position_ids, dtype=torch.long, device=device)
    else:
        stager = getattr(owner, "_decode_followup_position_stager", None)
        if not isinstance(stager, TextTensorStager):
            stager = TextTensorStager(ring_depth=3)
            setattr(owner, "_decode_followup_position_stager", stager)
        slot = stager.next_slot()
        cpu = slot.long_buffer("packed_decode_followup_positions", len(position_ids), pin=True)
        fill_cpu_ints(cpu, position_ids)
        positions = slot.device_buffer(
            "packed_decode_followup_positions",
            len(position_ids),
            dtype=torch.long,
            device=device,
        )
        positions.copy_(cpu, non_blocking=is_pinned(cpu))
    return position_ids, [positions[row : row + 1] for row in range(len(position_ids))]
