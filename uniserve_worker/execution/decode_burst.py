"""Pipelined multi-token decode burst execution."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..contracts.batches import UniForwardBatch
from ..contracts.forward_mode import ForwardMode
from ..contracts.outputs import ForwardOutputBase
from ..foundation.errors import invalid_descriptor

__all__ = [
    "DecodeBurstExecutor",
]


StepOnce = Callable[..., list[Any]]


class DecodeBurstExecutor:
    """Runs one or more decode bursts through relay-backed one-token steps."""

    def __init__(self, step_once: StepOnce, *, relay_placeholder_token_id: int = -1) -> None:
        self._step_once = step_once
        self._relay_placeholder_token_id = int(relay_placeholder_token_id)

    def run(
        self,
        first_ops: list[Mapping[str, Any]],
        request_states: Any,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> list[dict[str, Any]]:
        if len(first_ops) == 1:
            return [
                self._run_one(
                    dict(first_ops[0]),
                    request_states,
                    model,
                    defer_cpu_results=defer_cpu_results,
                )
            ]
        return self._run_many(
            first_ops,
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
        )

    def _run_one(
        self,
        first_op: dict[str, Any],
        request_states: Any,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> dict[str, Any]:
        count = _positive_int(first_op.get("decode_token_count") or 1, "decode_token_count")
        stop_ids = set(_int_list(first_op.get("decode_stop_token_ids") or []))
        terminal_stop = first_op.get("decode_stop_terminal") is True
        tokens: list[int] = []
        last: dict[str, Any] = {}
        op = dict(first_op)
        op["decode_token_count"] = 1
        op["decode_stop_token_ids"] = []

        def resolve(out: Any) -> dict[str, Any]:
            result = _seq_result_dict(out)
            tok = _positive_int(result.get("sampled_token_id"), "sampled_token_id", minimum=0)
            tokens.append(tok)
            return result

        pending: Any = None
        launched = 0
        while launched < count:
            out = self._launch_one(op, request_states, model)
            launched += 1
            if pending is not None:
                last = resolve(pending)
                pending = None
                if not terminal_stop and tokens[-1] in stop_ids:
                    out = None
                    break
            pending = out
            if launched >= count:
                break
            next_pos = _next_decode_position(op)
            op = self._next_relay_op(first_op, next_pos)
        if pending is not None:
            last = resolve(pending)

        if terminal_stop:
            tokens = _truncate_at_stop(tokens, stop_ids)
        result = (
            {"req_id": _positive_int(first_op.get("req_id"), "req_id", minimum=0)}
            if terminal_stop
            else dict(last)
        )
        result["sampled_token_id"] = tokens[-1]
        result["sampled_token_ids"] = tokens
        return result

    def _run_many(
        self,
        first_ops: list[Mapping[str, Any]],
        request_states: Any,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for op in first_ops:
            op_dict = dict(op)
            count = _positive_int(op_dict.get("decode_token_count") or 1, "decode_token_count")
            first = dict(op_dict)
            first["decode_token_count"] = 1
            first["decode_stop_token_ids"] = []
            states.append(
                {
                    "op": op_dict,
                    "last_op": first,
                    "requested": count,
                    "launched": 0,
                    "stop_ids": set(_int_list(op_dict.get("decode_stop_token_ids") or [])),
                    "terminal_stop": op_dict.get("decode_stop_terminal") is True,
                    "tokens": [],
                    "last": None,
                    "pending": None,
                    "done": False,
                }
            )

        while any(not state["done"] for state in states):
            iter_ops: list[dict[str, Any]] = []
            iter_indexes: list[int] = []
            for index, state in enumerate(states):
                if state["done"] or int(state["launched"]) >= int(state["requested"]):
                    continue
                if int(state["launched"]) == 0:
                    op = dict(state["last_op"])
                else:
                    op = self._next_relay_op(state["op"], _next_decode_position(state["last_op"]))
                state["last_op"] = op
                iter_indexes.append(index)
                iter_ops.append(op)
            if not iter_ops:
                break
            iter_outputs = self._launch_many(iter_ops, request_states, model)
            for index, output in zip(iter_indexes, iter_outputs, strict=True):
                state = states[index]
                previous = state["pending"]
                state["pending"] = output
                state["launched"] = int(state["launched"]) + 1
                if previous is None:
                    continue
                result = _seq_result_dict(previous)
                token = _positive_int(result.get("sampled_token_id"), "sampled_token_id", minimum=0)
                state["tokens"].append(token)
                state["last"] = result
                if not state["terminal_stop"] and token in state["stop_ids"]:
                    state["pending"] = None
                    state["done"] = True

        out: list[dict[str, Any]] = []
        for state in states:
            pending = state["pending"]
            if pending is not None:
                result = _seq_result_dict(pending)
                token = _positive_int(result.get("sampled_token_id"), "sampled_token_id", minimum=0)
                state["tokens"].append(token)
                state["last"] = result
                state["pending"] = None
            tokens = [int(token) for token in state["tokens"]]
            if state["terminal_stop"]:
                tokens = _truncate_at_stop(tokens, state["stop_ids"])
            last = dict(state["last"] or {})
            if not tokens:
                raise invalid_descriptor("decode burst did not produce a sampled token")
            if state["terminal_stop"]:
                last = {"req_id": _positive_int(state["op"].get("req_id"), "req_id", minimum=0)}
            last["sampled_token_id"] = tokens[-1]
            if int(state["requested"]) > 1:
                last["sampled_token_ids"] = tokens
            out.append(last)
        return out

    def _launch_one(self, op: dict[str, Any], request_states: Any, model: Any) -> Any:
        return self._launch_many([op], request_states, model)[0]

    def _launch_many(self, ops: list[dict[str, Any]], request_states: Any, model: Any) -> list[Any]:
        text = UniForwardBatch.from_ops(ops).as_text()
        if text.mode != ForwardMode.DECODE:
            raise invalid_descriptor("decode burst can only launch decode ops")
        return self._step_once(
            text,
            ops,
            request_states,
            model,
            defer_cpu_results=True,
            defer_sampling=False,
            tensor_store=None,
        )

    def _next_relay_op(self, first_op: Mapping[str, Any], next_pos: int) -> dict[str, Any]:
        op = dict(first_op)
        op["new_block_ids"] = []
        op["token_ids"] = [self._relay_placeholder_token_id]
        op["token_source"] = "last_sampled"
        op["pos_range"] = [next_pos, next_pos + 1]
        op["decode_token_count"] = 1
        op["decode_stop_token_ids"] = []
        return op


def _seq_result_dict(output: Any) -> dict[str, Any]:
    if hasattr(output, "finalize") and callable(output.finalize):
        finalized = output.finalize()
        if isinstance(finalized, Mapping):
            return dict(finalized)
    if isinstance(output, ForwardOutputBase):
        return dict(output.to_seq_result())
    if isinstance(output, Mapping):
        return dict(output)
    raise invalid_descriptor(f"unsupported text burst output type {type(output).__name__}")


def _positive_int(value: Any, where: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise invalid_descriptor(f"{where} must be an integer >= {minimum}")
    return int(value)


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise invalid_descriptor("decode_stop_token_ids must be a list")
    out: list[int] = []
    for idx, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise invalid_descriptor(f"decode_stop_token_ids[{idx}] must be a non-negative integer")
        out.append(int(item))
    return out


def _truncate_at_stop(tokens: list[int], stop_ids: set[int]) -> list[int]:
    if not stop_ids:
        return tokens
    out: list[int] = []
    for token in tokens:
        out.append(int(token))
        if int(token) in stop_ids:
            break
    return out


def _next_decode_position(op: Mapping[str, Any]) -> int:
    pos = op.get("pos_range") or [0, 0]
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        raise invalid_descriptor("decode burst op.pos_range must be [start, end]")
    return _positive_int(pos[1], "decode burst op.pos_range[1]", minimum=0)
