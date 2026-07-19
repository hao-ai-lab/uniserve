"""Autoregressive preparation and projection internals for ``ModelRunner``."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from uniserve_worker.contracts.batches import TextBatch
from uniserve_worker.contracts.forward_batch import (
    ForwardBatch,
    ForwardExecutionOptions,
    ForwardPlan,
    ForwardResult,
)
from uniserve_worker.contracts.forward_context import (
    component_timer_start,
    get_forward_context,
    record_component_elapsed,
    use_forward_context,
)
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.contracts.forward_stats import ForwardStats
from uniserve_worker.contracts.op_kinds import TARGET_VERIFY_UND
from uniserve_worker.contracts.outputs import (
    ForwardOutputBase,
    TextTokenOutput,
)
from uniserve_worker.foundation.errors import capability_mismatch, invalid_descriptor
from uniserve_worker.foundation.profiling import profile_range
from uniserve_worker.nn.sampler import (
    DeferredBatchedSamplingResult,
    apply_sampling_batched_with_device_tokens,
    finalize_sampling_result,
    is_deferred_sampling_result,
    sample_one_from_logits,
    score_prompt_token_logprobs,
)
from uniserve_worker.runtime.forward_batch_builder import ForwardBatchBuilder
from uniserve_worker.runtime.host_staging import (
    canonical_device,
    copy_cpu_to_device,
    cpu_int_staging_buffer,
    fill_cpu_ints,
    is_pinned,
)
from uniserve_worker.runtime.request_state import RequestState, RequestStateTable
from uniserve_worker.runtime.tensor_views import coalesce_one_token_rows
from uniserve_worker.spec import speculative_sample_target_only

if TYPE_CHECKING:
    from uniserve_worker.backends.attention.text_dispatch import TextBackendGate
    from uniserve_worker.contracts.forward_batch import ForwardBatch
    from uniserve_worker.contracts.forward_stats import ForwardStats
    from uniserve_worker.runtime.kv_pool import PagedKVPool


_TEXT_MODES = frozenset({ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT})


# Decode token/position relays (device-resident sequence feedback)
# ---------------------


class TextDecodeRelay:
    """Publishes and consumes device-resident decode token/position relays."""

    def publish_sample(
        self,
        state: "RequestState",
        *,
        token_id: int | None,
        token_tensor: torch.Tensor,
    ) -> torch.Tensor:
        device_token = token_tensor.detach().reshape(1)
        if device_token.device.type == "cuda":
            device_token.record_stream(torch.cuda.current_stream(device_token.device))
        state.decode_relay.token_id = None if token_id is None else int(token_id)
        state.decode_relay.token_tensor = device_token
        return device_token

    def publish_position(
        self,
        state: "RequestState",
        *,
        position_id: int,
        position_tensor: torch.Tensor,
    ) -> torch.Tensor:
        device_position = position_tensor.detach().reshape(1)
        if device_position.device.type == "cuda":
            device_position.record_stream(torch.cuda.current_stream(device_position.device))
        state.decode_relay.position_id = int(position_id)
        state.decode_relay.position_tensor = device_position
        return device_position

    def publish_positions(
        self,
        states: Sequence["RequestState"],
        *,
        position_ids: Sequence[int],
        device: torch.device | str,
    ) -> torch.Tensor:
        """Publish one contiguous batch of device-resident position relays.

        CUDA scalar construction from Python values performs a blocking host-to-device
        transfer on the current stream. Decode calls this after graph replay, so doing
        that once per row serializes the host behind every forward. Stage the complete
        position vector in pinned memory and enqueue one non-blocking copy instead.
        """

        if len(states) != len(position_ids):
            raise invalid_descriptor("decode position relay states and ids must align")
        resolved_device = canonical_device(device)
        count = len(position_ids)
        if resolved_device.type == "cuda":
            cpu = cpu_int_staging_buffer(
                count,
                dtype=torch.long,
                pin=True,
                name="decode_position_relay",
            )
            fill_cpu_ints(cpu, [int(position_id) for position_id in position_ids])
            positions = copy_cpu_to_device(
                cpu,
                device=resolved_device,
                non_blocking=is_pinned(cpu),
                slot=None,
                name="decode_position_relay",
            )
        else:
            positions = torch.tensor(
                [int(position_id) for position_id in position_ids],
                dtype=torch.long,
                device=resolved_device,
            )
        for row, (state, position_id) in enumerate(zip(states, position_ids, strict=True)):
            self.publish_position(
                state,
                position_id=int(position_id),
                position_tensor=positions[row : row + 1],
            )
        return positions

    def publish_deferred_sample_id_if_current(
        self,
        state: "RequestState",
        *,
        token_id: int,
        relay_token_tensor: torch.Tensor,
    ) -> bool:
        current = state.decode_relay.token_tensor
        if not _same_tensor(current, relay_token_tensor):
            return False
        state.decode_relay.token_id = int(token_id)
        return True

    def consume_token(
        self,
        state: "RequestState",
        *,
        expected_token_id: int | None,
        device: torch.device,
        token_source: str = "wire",
        require: bool = False,
        stats: "ForwardStats | None" = None,
    ) -> torch.Tensor | None:
        source = str(token_source or "wire")
        if source not in {"wire", "last_sampled"}:
            raise invalid_descriptor(f"unsupported text token_source {source!r}")
        from_last_sampled = source == "last_sampled"
        relay = state.decode_relay
        relay_token_id = getattr(relay, "token_id", None)
        relay_token_tensor = getattr(relay, "token_tensor", None)
        if relay_token_id is None and not from_last_sampled:
            _bump_stat(stats, "text_decode_token_relay_misses")
            return self._missing_token(require=require)
        if (
            not from_last_sampled
            and expected_token_id is not None
            and relay_token_id is not None
            and int(relay_token_id) != int(expected_token_id)
        ):
            _bump_stat(stats, "text_decode_token_relay_misses")
            return self._missing_token(require=require)
        if not isinstance(relay_token_tensor, torch.Tensor):
            _bump_stat(stats, "text_decode_token_relay_misses")
            return self._missing_token(require=require)
        if relay_token_tensor.dtype != torch.long or not _same_device(
            relay_token_tensor.device,
            device,
        ):
            _bump_stat(stats, "text_decode_token_relay_misses")
            if from_last_sampled or require:
                raise invalid_descriptor(
                    "decode op requested token_source='last_sampled' but the relay tensor is on the wrong device"
                )
            return None
        return relay_token_tensor.reshape(1)

    def consume_position(
        self,
        state: "RequestState",
        *,
        expected_position_id: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        relay = state.decode_relay
        relay_position_id = getattr(relay, "position_id", None)
        relay_position_tensor = getattr(relay, "position_tensor", None)
        if (
            relay_position_id is None
            or int(relay_position_id) != int(expected_position_id)
            or not isinstance(relay_position_tensor, torch.Tensor)
            or relay_position_tensor.dtype != torch.long
            or not _same_device(relay_position_tensor.device, device)
        ):
            return None
        return relay_position_tensor.reshape(1)

    def replace_inputs(
        self,
        text: "TextBatch",
        request_states: "RequestStateTable",
        device: torch.device,
    ) -> dict[int, torch.Tensor] | None:
        replacements: dict[int, torch.Tensor] = {}
        flat_idx = 0
        for req_id, tokens, op in zip(text.req_ids, text.token_ids, text.ops):
            source = str(op.get("token_source") or "wire")
            if source not in {"wire", "last_sampled"}:
                raise invalid_descriptor(f"unsupported text token_source {source!r}")
            if source == "last_sampled":
                if len(tokens) != 1:
                    raise invalid_descriptor(
                        "decode op requested token_source='last_sampled' but does not have exactly one token"
                    )
                state = request_states.get(int(req_id))
                relay_token = self.consume_token(
                    state,
                    expected_token_id=None,
                    device=device,
                    token_source=source,
                    require=True,
                )
                if relay_token is None:
                    raise invalid_descriptor(
                        "decode op requested token_source='last_sampled' but no relay token is available"
                    )
                replacements[flat_idx] = relay_token.reshape(1)
            flat_idx += len(tokens)
        return replacements or None

    def resolve_decode_batch(
        self,
        text: "TextBatch",
        request_states: "RequestStateTable",
        device: torch.device,
        *,
        stats: "ForwardStats | None" = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        relay_rows: list[torch.Tensor] = []
        position_rows: list[torch.Tensor] = []
        position_complete = True
        for req_id, tokens, pos_range, op in zip(
            text.req_ids,
            text.token_ids,
            text.pos_ranges,
            text.ops,
        ):
            if len(tokens) != 1:
                _bump_stat(stats, "text_decode_token_relay_misses")
                return None, None
            state = request_states.get(int(req_id))
            source = str(op.get("token_source") or "wire")
            token = self.consume_token(
                state,
                expected_token_id=int(tokens[0]),
                device=device,
                token_source=source,
                require=source == "last_sampled",
                stats=stats,
            )
            if token is None:
                return None, None
            relay_rows.append(token.reshape(1))
            position = self.consume_position(
                state,
                expected_position_id=int(pos_range[0]),
                device=device,
            )
            if position is None:
                position_complete = False
                continue
            position_rows.append(position.reshape(1))
        _bump_stat(stats, "text_decode_token_relay_hits", len(relay_rows))
        if position_complete and len(position_rows) == len(relay_rows):
            _bump_stat(stats, "text_decode_position_relay_hits", len(position_rows))
            relay_positions = coalesce_one_token_rows(position_rows)
        else:
            _bump_stat(stats, "text_decode_position_relay_misses")
            relay_positions = None
        return coalesce_one_token_rows(relay_rows), relay_positions

    def attach_last_sampled_to_op(self, op: dict[str, Any], state: "RequestState") -> None:
        relay = state.decode_relay
        tensor = relay.token_tensor
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.long:
            raise invalid_descriptor(
                "decode op requested token_source='last_sampled' but the relay tensor is unavailable"
            )
        op["token_tensor"] = tensor
        if relay.token_id is not None:
            op["token_ids"] = [int(relay.token_id)]

    @staticmethod
    def _missing_token(*, require: bool) -> None:
        if require:
            raise invalid_descriptor(
                "decode op requested token_source='last_sampled' but the relay tensor is unavailable"
            )
        return None


def _same_tensor(lhs: Any, rhs: torch.Tensor) -> bool:
    if not isinstance(lhs, torch.Tensor):
        return False
    if lhs.device != rhs.device or lhs.dtype != rhs.dtype or lhs.shape != rhs.shape:
        return False
    return int(lhs.data_ptr()) == int(rhs.data_ptr())


def _same_device(lhs: torch.device | str, rhs: torch.device | str) -> bool:
    return canonical_device(lhs) == canonical_device(rhs)


def _bump_stat(stats: "ForwardStats | None", attr: str, delta: int = 1) -> None:
    if stats is None:
        return
    setattr(stats, attr, int(getattr(stats, attr)) + int(delta))


# ---------------------
# Decode burst execution
# ---------------------

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
        result: dict[str, Any] = (
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
        text = ForwardBatch.from_ops(ops).as_text()
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


# ---------------------
# Deferred text results
# ---------------------

_DECODE_RELAY = TextDecodeRelay()


class DeferredTextSeqResult:
    """One text seq-result whose CPU token id is finalized at response time."""

    req_id: int
    _row: int
    _state: RequestState
    _sampling_result: DeferredBatchedSamplingResult
    _relay_token_tensor: torch.Tensor
    _finalized: dict[str, Any] | None

    def __init__(
        self,
        *,
        req_id: int,
        row: int,
        state: "RequestState",
        sampling_result: DeferredBatchedSamplingResult,
        relay_token_tensor: torch.Tensor,
    ) -> None:
        object.__setattr__(self, "req_id", int(req_id))
        object.__setattr__(self, "_row", int(row))
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_sampling_result", sampling_result)
        object.__setattr__(self, "_relay_token_tensor", relay_token_tensor)
        object.__setattr__(self, "_finalized", None)

    def to_seq_result(self) -> "DeferredTextSeqResult":
        return self

    def finalize(self) -> dict[str, Any]:
        finalized = self._finalized
        if finalized is None:
            sample = self._sampling_result.finalize().samples[self._row]
            tok, lp, top = sample
            _DECODE_RELAY.publish_deferred_sample_id_if_current(
                self._state,
                token_id=int(tok),
                relay_token_tensor=self._relay_token_tensor,
            )
            result: dict[str, Any] = {
                "req_id": self.req_id,
                "sampled_token_id": int(tok),
            }
            if lp is not None:
                result["sampled_logprob"] = lp
            if top:
                result["top_logprobs"] = top
            object.__setattr__(self, "_finalized", result)
            finalized = result
        return dict(finalized)

    def materialize_sampled_token_id(self) -> int:
        if self._finalized is not None:
            return int(self._finalized["sampled_token_id"])
        token_ids = self._sampling_result.token_ids()
        tok = int(token_ids[self._row])
        _DECODE_RELAY.publish_deferred_sample_id_if_current(
            self._state,
            token_id=tok,
            relay_token_tensor=self._relay_token_tensor,
        )
        return tok

    def ready(self) -> bool:
        if self._finalized is not None:
            return True
        ready = getattr(self._sampling_result, "ready", None)
        return bool(ready()) if callable(ready) else True

    def cuda_ready_group_key(self) -> int:
        return id(self._sampling_result)

    def cuda_ready_elapsed_us(self) -> int | None:
        elapsed = getattr(self._sampling_result, "cuda_ready_elapsed_us", None)
        return elapsed() if callable(elapsed) else None


class DeferredDecodeBurstSeqResult:
    """Decode-burst result whose final sampled token is still event-backed."""

    req_id: int
    _prefix_token_ids: tuple[int, ...]
    _pending: Any
    _finalized: dict[str, Any] | None

    def __init__(
        self,
        *,
        req_id: int,
        prefix_token_ids: Sequence[int],
        pending: Any,
    ) -> None:
        object.__setattr__(self, "req_id", int(req_id))
        object.__setattr__(
            self, "_prefix_token_ids", tuple(int(token) for token in prefix_token_ids)
        )
        object.__setattr__(self, "_pending", pending)
        object.__setattr__(self, "_finalized", None)

    def to_seq_result(self) -> "DeferredDecodeBurstSeqResult":
        return self

    def finalize(self) -> dict[str, Any]:
        finalized = self._finalized
        if finalized is None:
            token_id = self._materialize_pending_token_id()
            token_ids = [*self._prefix_token_ids, token_id]
            finalized = {
                "req_id": self.req_id,
                "sampled_token_id": int(token_ids[-1]),
                "sampled_token_ids": [int(token) for token in token_ids],
            }
            object.__setattr__(self, "_finalized", finalized)
        return dict(finalized)

    def _materialize_pending_token_id(self) -> int:
        return _materialize_pending_token_id(self._pending)

    def ready(self) -> bool:
        if self._finalized is not None:
            return True
        ready = getattr(self._pending, "ready", None)
        return bool(ready()) if callable(ready) else True

    def cuda_ready_group_key(self) -> int:
        key = getattr(self._pending, "cuda_ready_group_key", None)
        return int(key()) if callable(key) else id(self._pending)

    def cuda_ready_elapsed_us(self) -> int | None:
        elapsed = getattr(self._pending, "cuda_ready_elapsed_us", None)
        return elapsed() if callable(elapsed) else None


class DeferredTerminalDecodeBurstSeqResult:
    """Terminal-stop decode-burst result with all sampled tokens deferred."""

    req_id: int
    _pending_tokens: tuple[Any, ...]
    _stop_token_ids: frozenset[int]
    _finalized: dict[str, Any] | None

    def __init__(
        self,
        *,
        req_id: int,
        pending_tokens: Sequence[Any],
        stop_token_ids: Iterable[int],
    ) -> None:
        object.__setattr__(self, "req_id", int(req_id))
        object.__setattr__(self, "_pending_tokens", tuple(pending_tokens))
        object.__setattr__(
            self, "_stop_token_ids", frozenset(int(token) for token in stop_token_ids)
        )
        object.__setattr__(self, "_finalized", None)

    def to_seq_result(self) -> "DeferredTerminalDecodeBurstSeqResult":
        return self

    def finalize(self) -> dict[str, Any]:
        finalized = self._finalized
        if finalized is None:
            token_ids: list[int] = []
            for pending in self._pending_tokens:
                token_id = _materialize_pending_token_id(pending)
                token_ids.append(token_id)
                if token_id in self._stop_token_ids:
                    break
            if not token_ids:
                raise invalid_descriptor("terminal decode burst did not produce a sampled token")
            finalized = {
                "req_id": self.req_id,
                "sampled_token_id": int(token_ids[-1]),
                "sampled_token_ids": [int(token) for token in token_ids],
            }
            object.__setattr__(self, "_finalized", finalized)
        return dict(finalized)

    def ready(self) -> bool:
        if self._finalized is not None:
            return True
        for pending in self._pending_tokens:
            ready = getattr(pending, "ready", None)
            if callable(ready) and not bool(ready()):
                return False
        return True

    def cuda_ready_group_key(self) -> int:
        keys: list[int] = []
        for pending in self._pending_tokens:
            key = getattr(pending, "cuda_ready_group_key", None)
            keys.append(int(key()) if callable(key) else id(pending))
        return hash(tuple(keys))

    def cuda_ready_elapsed_us(self) -> int | None:
        if not self.ready():
            return None
        total = 0
        seen: set[int] = set()
        for pending in self._pending_tokens:
            key = getattr(pending, "cuda_ready_group_key", None)
            group_key = int(key()) if callable(key) else id(pending)
            if group_key in seen:
                continue
            seen.add(group_key)
            elapsed = getattr(pending, "cuda_ready_elapsed_us", None)
            value = elapsed() if callable(elapsed) else None
            if isinstance(value, int) and value > 0:
                total += value
        return total if total > 0 else None


def _materialize_pending_token_id(pending: Any) -> int:
    materialize = getattr(pending, "materialize_sampled_token_id", None)
    if callable(materialize):
        return int(materialize())
    finalize = getattr(pending, "finalize", None)
    finalized = finalize() if callable(finalize) else pending
    if isinstance(finalized, Mapping):
        return int(finalized["sampled_token_id"])
    return int(dict(finalized)["sampled_token_id"])


# ---------------------
# Candidate verification (speculative token acceptance)
# ---------------------

_KV_LANE = "text"
_SAMPLED_TOKEN_DEVICE_KEY = "sampled_token_device"
_SAMPLED_POSITION_DEVICE_KEY = "sampled_position_device"


def verify_speculative_tokens(
    runtime: "_AutoregressiveRuntime",
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
    if runtime.builder is None or runtime.kv_pool is None:
        raise invalid_descriptor(
            "speculative verify requires the system ForwardBatchBuilder and KV pool"
        )
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
    builder, kv_pool = runtime._system_forward_runtime()
    for length, rows in grouped.items():
        extended_ops = [op for _, op, _ in rows]
        verify_text = ForwardBatch.from_ops(extended_ops).as_text()
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
        with use_forward_context(replace(ctx, attention_plan=fb.attn_plan, kv_pool=kv_pool)):
            logits = _text_model_forward(
                model,
                fb,
                input_ids=input_ids,
                positions=positions,
            )
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
    if (
        isinstance(token_tensor, torch.Tensor)
        and isinstance(req_id, int)
        and isinstance(token_id, int)
    ):
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
    sampled_token_tensor = chosen[accepted : accepted + 1]
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
            generators=[state.device_rng(logits.device, stream="text_sampling")],
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
        result[_SAMPLED_POSITION_DEVICE_KEY] = sampled_token_tensor.new_full(
            (1,), next_pos, dtype=torch.long
        )
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


# ---------------------
# Text sequence execution
# ---------------------


# Wire token id carried by pipelined-burst ``last_sampled`` ops whose real token
# still lives only in the device relay tensor. Deliberately invalid: any path
# that embeds the wire token instead of consuming the relay fails loudly.
_RELAY_PLACEHOLDER_TOKEN_ID = -1
_GRAPH_RUNNER_UNSET = object()


class _DecodeBurstGraphMiss(Exception):
    pass


@dataclass(frozen=True)
class TextForwardLogits:
    """Neural text forward result before sampling or request-state mutation."""

    logits: torch.Tensor
    req_ids: tuple[int, ...]
    cuda_ready_start_event: torch.cuda.Event | None = None


def _text_model_forward(
    model: Any,
    batch: ForwardBatch,
    *,
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    forward_text = getattr(model, "forward_text", None)
    if not callable(forward_text):
        raise capability_mismatch("text-capable model must implement forward_text(batch)")
    prepared = (
        replace(batch, input_ids=input_ids, positions=positions)
        if input_ids is not None or positions is not None
        else batch
    )
    result = forward_text(prepared)
    if not isinstance(result, torch.Tensor):
        raise invalid_descriptor("model forward_text must return a logits tensor")
    return result


def text_input_id_replacements_from_relays(
    text: "TextBatch",
    request_states: RequestStateTable,
    device: torch.device,
) -> dict[int, torch.Tensor] | None:
    """Return flat input-token replacements for ``last_sampled`` decode rows.

    The async decode fast path keeps the last sampled token resident on device;
    a mixed extend+decode forward replaces only the decode rows' placeholder
    tokens by flat index (the contiguous-relay override is the pure-decode case).
    """

    return _DECODE_RELAY.replace_inputs(text, request_states, device)


def sample_logits_result(
    *,
    req_id: int,
    state: RequestState,
    logits: torch.Tensor,
    op: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample one token from model-produced logits using the canonical pipeline."""

    if not isinstance(logits, torch.Tensor):
        raise invalid_descriptor("text logits output must contain a tensor")
    if logits.ndim == 0:
        raise invalid_descriptor("text logits tensor must have a vocabulary dimension")
    vocab_logits = logits.float()
    if vocab_logits.ndim > 1:
        vocab_logits = vocab_logits.reshape(-1, vocab_logits.shape[-1])[-1]
    sp = dict(state.sampling or {})
    tok, lp, top = sample_one_from_logits(
        vocab_logits,
        sp,
        recent=op.get("recent_tokens") or [],
        allowed=op.get("allowed_tokens"),
        suppress=op.get("suppress_tokens"),
        n_logprobs=int(sp.get("n_logprobs", 0) or 0),
        generator=state.device_rng(vocab_logits.device, stream="text_sampling"),
    )
    result: dict[str, Any] = {"req_id": int(req_id), "sampled_token_id": tok}
    if lp is not None:
        result["sampled_logprob"] = lp
    if top:
        result["top_logprobs"] = top
    return result


class _AutoregressiveRuntime:
    """Run the system-managed text forward and own the post-model sampler.

    Constructed by the runner with the system collaborators it orchestrates:
    the ``ForwardBatchBuilder`` (GPU snapshot + residency + plan), the
    ``TextBackendGate`` (batched-vs-per-op), and the system-owned ``kv_pool``.
    The optional ``graph_runner`` captures/replays the decode/prefill graphs
    around the graph-unaware model.
    """

    def __init__(
        self,
        *,
        builder: "ForwardBatchBuilder | None" = None,
        gate: "TextBackendGate | None" = None,
        kv_pool: "PagedKVPool | None" = None,
        graph_runner: Any | None = None,
    ) -> None:
        self.builder = builder
        self.gate = gate
        self.kv_pool = kv_pool
        self.graph_runner = graph_runner

    def _system_forward_runtime(self) -> tuple["ForwardBatchBuilder", "PagedKVPool"]:
        if self.builder is None or self.kv_pool is None:
            raise invalid_descriptor("system-managed text forward requires a builder and KV pool")
        return self.builder, self.kv_pool

    def prepare_batch(
        self,
        plan: ForwardPlan,
        request_states: RequestStateTable,
        *,
        device: torch.device,
    ) -> ForwardBatch | None:
        """Build the reusable text snapshot consumed by the public model forward."""

        if self.builder is None or self.kv_pool is None:
            return None
        if not plan.rows or any(row.mode not in _TEXT_MODES for row in plan.rows):
            return None
        text = TextBatch.from_ops(
            plan.forward_mode,
            plan.ops,
            op_modes=plan.op_modes,
            allow_mixed_text=plan.forward_mode is ForwardMode.MIXED,
        )
        relay_input_ids = relay_positions = None
        relay_replacements = None
        if text.mode is ForwardMode.DECODE:
            relay_input_ids, relay_positions = self._decode_relay_tensors(
                text,
                request_states,
                device,
            )
        elif text.mode is ForwardMode.MIXED:
            relay_replacements = text_input_id_replacements_from_relays(
                text,
                request_states,
                device,
            )
        padded = self._graph_padded_num_tokens(text, get_forward_context())
        batch = self.builder.build_text(
            text,
            device=device,
            kv_pool=self.kv_pool,
            request_states=request_states,
            input_ids_override=relay_input_ids,
            positions_override=relay_positions,
            input_ids_replacements=relay_replacements,
            padded_num_tokens=padded,
        )
        batch.op_modes = plan.op_modes
        return batch

    def forward_result(
        self,
        fb: ForwardBatch,
        request_states: RequestStateTable,
        model: Any,
        *,
        options: ForwardExecutionOptions = ForwardExecutionOptions(),
        tensor_store: Any | None = None,
    ) -> ForwardResult:
        """Execute one complete text batch behind the model's public forward boundary."""

        text_result = self.forward_logits(
            fb,
            request_states,
            model,
            defer_cpu_results=options.defer_text_cpu_results,
            defer_sampling=options.defer_sampling,
        )
        if text_result is not None:
            expected_req_ids = tuple(int(op["req_id"]) for op in fb.ops)
            if tuple(int(req_id) for req_id in text_result.req_ids) != expected_req_ids:
                raise invalid_descriptor("text logits result req_ids must align with forward ops")
            return ForwardResult(
                text_logits=text_result.logits,
                text_cuda_ready_start_event=text_result.cuda_ready_start_event,
            )
        outputs = self.step(
            fb,
            request_states,
            model,
            defer_cpu_results=options.defer_text_cpu_results,
            defer_sampling=options.defer_sampling,
            tensor_store=tensor_store,
        )
        return ForwardResult(runtime_outputs=tuple(outputs))

    @torch.inference_mode()
    def step(
        self,
        fb: ForwardBatch,
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
    ) -> list[Any]:
        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(bool(op.get("return_all_logits")) for op in ops):
            with profile_range("uniserve.text.prompt_logprobs"):
                return self._step_prompt_prefill(
                    text,
                    ops,
                    request_states,
                    model,
                    defer_sampling=defer_sampling,
                    tensor_store=tensor_store,
                )
        if any(text.spec_token_ids):
            pass

            with profile_range("uniserve.text.speculative_verify"):
                return verify_speculative_tokens(self, model, text, request_states)
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            with profile_range("uniserve.text.decode_burst"):
                return DecodeBurstExecutor(
                    self._step_once,
                    relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
                ).run(
                    ops,
                    request_states,
                    model,
                    defer_cpu_results=defer_cpu_results,
                )
        if text.mode == ForwardMode.MIXED and self.graph_runner is not None:
            # Compute in an order that keeps graph token-bucket padding legal
            # for the final row, but return results in the wire op order (the
            # response finalizer matches per-seq results to ops positionally).
            reordered = self.graph_runner.reorder_mixed_for_padding(text)
            if reordered is not text:
                row_by_op = {id(op): row for row, op in enumerate(reordered.ops)}
                results = self._step_once(
                    reordered,
                    list(reordered.ops),
                    request_states,
                    model,
                    defer_cpu_results=defer_cpu_results,
                    defer_sampling=defer_sampling,
                    tensor_store=tensor_store,
                )
                return [results[row_by_op[id(op)]] for op in ops]
        return self._step_once(
            text,
            ops,
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
            defer_sampling=defer_sampling,
            tensor_store=tensor_store,
        )

    def forward_logits(
        self,
        fb: ForwardBatch,
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> TextForwardLogits | None:
        """Run text neural execution and leave postprocessing to the forward stack."""

        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(bool(op.get("return_all_logits")) for op in ops):
            return None
        if any(text.spec_token_ids):
            return None
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            return None
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward"):
            prepared = (
                self._forward_prepared(fb, text, model)
                if fb.device is not None and fb.device.type == "cuda" and fb.attn_plan is not None
                else None
            )
            logits_batch, req_ids = (
                prepared
                if prepared is not None
                else self._forward_with_optional_padding_reorder(
                    text,
                    ops,
                    request_states,
                    model,
                    store_position_relays=False,
                )
            )
        return TextForwardLogits(
            logits=logits_batch,
            req_ids=tuple(int(req_id) for req_id in req_ids),
            cuda_ready_start_event=cuda_ready_start_event,
        )

    def _forward_prepared(
        self,
        batch: ForwardBatch,
        text: "TextBatch",
        model: Any,
    ) -> tuple[torch.Tensor, list[int]] | None:
        if self.kv_pool is None:
            return None
        ctx = get_forward_context()
        input_ids, positions = self._reshape_inputs(batch, text)
        with use_forward_context(
            replace(ctx, attention_plan=batch.attn_plan, kv_pool=self.kv_pool)
        ):
            logits = self._run_model_forward(
                model,
                input_ids,
                positions,
                batch,
                ctx,
            )
        if logits is None:
            return None
        return logits, [int(req_id) for req_id in text.req_ids]

    def forward_logits_graph(
        self,
        fb: ForwardBatch,
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> TextForwardLogits | None:
        """Run text neural execution only when a CUDA graph handles the batch."""

        if graph_runner is None and self.builder is not None:
            return None
        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(bool(op.get("return_all_logits")) for op in ops):
            return None
        if any(text.spec_token_ids):
            return None
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            return None
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward_graph"):
            if graph_runner is None:
                graph_result = self._forward(
                    model,
                    text,
                    request_states,
                    store_position_relays=False,
                    require_graph=True,
                )
            else:
                graph_result = self._forward_graph_with_optional_padding_reorder(
                    text,
                    ops,
                    request_states,
                    model,
                    graph_runner=graph_runner,
                    store_position_relays=False,
                )
        if graph_result is None:
            return None
        logits_batch, req_ids = graph_result
        return TextForwardLogits(
            logits=logits_batch,
            req_ids=tuple(int(req_id) for req_id in req_ids),
            cuda_ready_start_event=cuda_ready_start_event,
        )

    def forward_graph_result(
        self,
        fb: ForwardBatch,
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> ForwardResult | None:
        """Run a text batch only when graph-backed execution can cover it."""

        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(text.spec_token_ids):
            return None
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            outputs = self._decode_burst_graph_many(
                ops,
                request_states,
                model,
                graph_runner=graph_runner,
                defer_cpu_results=defer_cpu_results,
            )
            if outputs is None:
                return None
            return ForwardResult(runtime_outputs=tuple(outputs))
        text_result = self.forward_logits_graph(
            fb,
            request_states,
            model,
            graph_runner=graph_runner,
            defer_cpu_results=defer_cpu_results,
            defer_sampling=defer_sampling,
        )
        if text_result is None:
            return None
        expected_req_ids = tuple(int(op["req_id"]) for op in ops)
        req_ids = tuple(int(req_id) for req_id in text_result.req_ids)
        if req_ids != expected_req_ids:
            raise invalid_descriptor("text graph result req_ids must align with forward ops")
        return ForwardResult(
            text_logits=text_result.logits,
            text_cuda_ready_start_event=text_result.cuda_ready_start_event,
        )

    def _step_once(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
    ) -> list[Any]:
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward"):
            forward_result = self._forward(model, text, request_states)
        if forward_result is None:
            raise invalid_descriptor("eager text forward did not produce logits")
        logits_batch, req_ids = forward_result
        # KV-length advance is system-owned now (derived from seq_lens), not the
        # model's job.
        self._advance_kv_lengths(text, request_states)
        if defer_sampling and tensor_store is not None:
            with profile_range("uniserve.text.publish_logits"):
                return self._publish_logits(ops, req_ids, logits_batch, tensor_store)
        start = component_timer_start(stats)
        with profile_range("uniserve.text.sample"):
            return self._sample_logits_batch(
                ops,
                req_ids,
                logits_batch,
                request_states,
                stats,
                start,
                defer_cpu_results=defer_cpu_results,
                cuda_ready_start_event=cuda_ready_start_event,
            )

    def _decode_burst(
        self,
        first_op: dict[str, Any],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> dict[str, Any]:
        return DecodeBurstExecutor(
            self._step_once,
            relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
        ).run(
            [first_op],
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
        )[0]

    def _decode_burst_many(
        self,
        first_ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> list[dict[str, Any]]:
        return DecodeBurstExecutor(
            self._step_once,
            relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
        ).run(
            list(first_ops),
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
        )

    def _decode_burst_graph_many(
        self,
        first_ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
    ) -> list[dict[str, Any]] | None:
        def step_once_graph(
            text: "TextBatch",
            ops: list[Mapping[str, Any]],
            request_states: RequestStateTable,
            model: Any,
            *,
            defer_cpu_results: bool = False,
            defer_sampling: bool = False,
            tensor_store: Any | None = None,
        ) -> list[Any]:
            del tensor_store
            outputs = self._step_once_graph(
                text,
                ops,
                request_states,
                model,
                graph_runner=graph_runner,
                defer_cpu_results=defer_cpu_results,
                defer_sampling=defer_sampling,
            )
            if outputs is None:
                raise _DecodeBurstGraphMiss
            return outputs

        try:
            return DecodeBurstExecutor(
                step_once_graph,
                relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
            ).run(
                list(first_ops),
                request_states,
                model,
                defer_cpu_results=defer_cpu_results,
            )
        except _DecodeBurstGraphMiss:
            return None

    def _step_once_graph(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> list[Any] | None:
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward_graph"):
            graph_result = self._forward(
                model,
                text,
                request_states,
                graph_runner=graph_runner,
                require_graph=True,
            )
        if graph_result is None:
            return None
        logits_batch, req_ids = graph_result
        self._advance_kv_lengths(text, request_states)
        if defer_sampling:
            return None
        start = component_timer_start(stats)
        with profile_range("uniserve.text.sample"):
            return self._sample_logits_batch(
                ops,
                req_ids,
                logits_batch,
                request_states,
                stats,
                start,
                defer_cpu_results=defer_cpu_results,
                cuda_ready_start_event=cuda_ready_start_event,
            )

    # ---- forward ---------------------------------------------------------

    def _forward(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
        *,
        store_position_relays: bool = True,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
        require_graph: bool = False,
    ) -> tuple[torch.Tensor, list[int]] | None:
        if self.builder is None or self.kv_pool is None:
            # Self-managing text models (the HF day-zero fallback and the
            # composed multimodal programs whose KV is intrinsically coupled to
            # their modality FSM) declare no ``kv_cache_spec``; the system owns no
            # pool for them. They expose their own per-op text logits and the
            # driver still owns the post-model sampler.
            if require_graph:
                return self._model_owned_kv_forward_graph(model, text, request_states)
            return self._model_owned_kv_forward(model, text, request_states)
        ctx = get_forward_context()
        device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
        batched = self.gate is not None and self.gate.batched_capable(
            text, attention_preference=ctx.attention_preference
        )
        if batched:
            return self._forward_batched(
                model,
                text,
                request_states,
                ctx,
                device,
                store_position_relays=store_position_relays,
                graph_runner=graph_runner,
                require_graph=require_graph,
            )
        if require_graph:
            return None
        return self._forward_per_op(
            model,
            text,
            request_states,
            ctx,
            device,
            store_position_relays=store_position_relays,
        )

    def _forward_with_optional_padding_reorder(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        store_position_relays: bool,
    ) -> tuple[torch.Tensor, list[int]]:
        if text.mode == ForwardMode.MIXED and self.graph_runner is not None:
            reordered = self.graph_runner.reorder_mixed_for_padding(text)
            if reordered is not text:
                row_by_op = {id(op): row for row, op in enumerate(reordered.ops)}
                forward_result = self._forward(
                    model,
                    reordered,
                    request_states,
                    store_position_relays=store_position_relays,
                )
                if forward_result is None:
                    raise invalid_descriptor("reordered eager text forward did not produce logits")
                logits, req_ids = forward_result
                rows = [row_by_op[id(op)] for op in ops]
                order = torch.tensor(rows, dtype=torch.long, device=logits.device)
                return logits.index_select(0, order), [int(req_ids[row]) for row in rows]
        forward_result = self._forward(
            model,
            text,
            request_states,
            store_position_relays=store_position_relays,
        )
        if forward_result is None:
            raise invalid_descriptor("eager text forward did not produce logits")
        return forward_result

    def _forward_graph_with_optional_padding_reorder(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        store_position_relays: bool,
    ) -> tuple[torch.Tensor, list[int]] | None:
        if text.mode == ForwardMode.MIXED:
            reordered = graph_runner.reorder_mixed_for_padding(text)
            if reordered is not text:
                row_by_op = {id(op): row for row, op in enumerate(reordered.ops)}
                graph_result = self._forward(
                    model,
                    reordered,
                    request_states,
                    store_position_relays=store_position_relays,
                    graph_runner=graph_runner,
                    require_graph=True,
                )
                if graph_result is None:
                    return None
                logits, req_ids = graph_result
                rows = [row_by_op[id(op)] for op in ops]
                order = torch.tensor(rows, dtype=torch.long, device=logits.device)
                return logits.index_select(0, order), [int(req_ids[row]) for row in rows]
        return self._forward(
            model,
            text,
            request_states,
            store_position_relays=store_position_relays,
            graph_runner=graph_runner,
            require_graph=True,
        )

    def _model_owned_kv_forward(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
    ) -> tuple[torch.Tensor, list[int]]:
        """Run a self-managing model's per-op text logits and stack them.

        Accepts a batched ``run_text_logits_batch`` or a per-op
        ``run_text_logits``; both return a raw logits tensor per op, coerced to
        one ``[vocab]`` row. Req ids come from the op order, not the tensors.

        The driver owns the decode-relay lookup: ``last_sampled`` ops get the
        device relay tensor attached as ``op['token_tensor']`` (and the resolved
        id when the CPU copy has landed) so the model side can consume the
        sampled token without a GPU synchronize and without reaching into
        system request state.
        """

        ops = self._model_owned_ops(text, request_states)
        outputs = list(model.run_text_logits_batch(ops))
        rows = [self._coerce_logits_row(out) for out in outputs]
        return torch.stack(rows, dim=0), [int(op["req_id"]) for op in ops]

    def _model_owned_kv_forward_graph(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
    ) -> tuple[torch.Tensor, list[int]] | None:
        graph_logits = getattr(model, "try_run_graph_logits_batch", None)
        if not callable(graph_logits):
            return None
        ops = self._model_owned_ops(text, request_states)
        outputs = graph_logits(ops)
        if outputs is None:
            return None
        rows = [self._coerce_logits_row(out) for out in outputs]
        return torch.stack(rows, dim=0), [int(op["req_id"]) for op in ops]

    @staticmethod
    def _model_owned_ops(
        text: "TextBatch",
        request_states: RequestStateTable,
    ) -> list[dict[str, Any]]:
        ops = [dict(op) for op in text.ops]
        for op in ops:
            if str(op.get("token_source") or "wire") != "last_sampled":
                continue
            _DECODE_RELAY.attach_last_sampled_to_op(op, request_states.get(int(op["req_id"])))
        return ops

    @staticmethod
    def _coerce_logits_row(logits: Any) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            raise invalid_descriptor("self-managing text model must return a logits tensor")
        return logits.reshape(-1, logits.shape[-1])[-1]

    def _forward_batched(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
        ctx: Any,
        device: torch.device,
        *,
        store_position_relays: bool = True,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
        require_graph: bool = False,
    ) -> tuple[torch.Tensor, list[int]] | None:
        stats = ctx.stats
        builder, kv_pool = self._system_forward_runtime()
        active_graph_runner = (
            self.graph_runner if graph_runner is _GRAPH_RUNNER_UNSET else graph_runner
        )
        start = component_timer_start(stats)
        # Pure decode uses the contiguous-relay override (fast); a mixed
        # extend+decode batch replaces only its last_sampled decode rows by index.
        relay_input_ids = relay_positions = None
        relay_replacements = None
        if text.mode == ForwardMode.DECODE:
            relay_input_ids, relay_positions = self._decode_relay_tensors(
                text, request_states, device
            )
        elif text.mode == ForwardMode.MIXED:
            relay_replacements = text_input_id_replacements_from_relays(
                text, request_states, device
            )
        record_component_elapsed(stats, "text_decode_relay", start)
        start = component_timer_start(stats)
        padded = self._graph_padded_num_tokens(text, ctx, graph_runner=active_graph_runner)
        fb = builder.build_text(
            text,
            device=device,
            kv_pool=kv_pool,
            request_states=request_states,
            input_ids_override=relay_input_ids,
            positions_override=relay_positions,
            input_ids_replacements=relay_replacements,
            padded_num_tokens=padded,
        )
        record_component_elapsed(stats, "text_build_batch", start)
        input_ids, positions = self._reshape_inputs(fb, text)
        start = component_timer_start(stats)
        with profile_range("uniserve.text.model_forward"):
            with use_forward_context(replace(ctx, attention_plan=fb.attn_plan, kv_pool=kv_pool)):
                logits = self._run_model_forward(
                    model,
                    input_ids,
                    positions,
                    fb,
                    ctx,
                    graph_runner=active_graph_runner,
                    require_graph=require_graph,
                )
        if logits is None:
            return None
        record_component_elapsed(stats, "text_model_forward", start)
        if store_position_relays:
            start = component_timer_start(stats)
            self._store_decode_position_relays(text, fb, request_states)
            record_component_elapsed(stats, "text_decode_position_store", start)
        return logits, [int(req_id) for req_id in text.req_ids]

    def _forward_per_op(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
        ctx: Any,
        device: torch.device,
        *,
        store_position_relays: bool = True,
    ) -> tuple[torch.Tensor, list[int]]:
        rows: list[torch.Tensor] = []
        next_positions: list[tuple[int, int]] = []
        builder, kv_pool = self._system_forward_runtime()
        for req_id, tokens, pos_range, op in zip(
            text.req_ids, text.token_ids, text.pos_ranges, text.ops
        ):
            state = request_states.get(int(req_id))
            relay = self._per_op_relay_input(op, tokens, state, device)
            fb = builder.build_text_op(
                op=op,
                token_ids=tokens,
                pos_range=pos_range,
                req_id=int(req_id),
                mode=text.mode,
                device=device,
                kv_pool=kv_pool,
                request_states=request_states,
                input_ids_override=relay,
            )
            with use_forward_context(replace(ctx, attention_plan=fb.attn_plan, kv_pool=kv_pool)):
                logits = _text_model_forward(model, fb)
            rows.append(logits.reshape(-1, logits.shape[-1])[-1])
            next_positions.append((int(req_id), int(pos_range[1])))
        if store_position_relays and text.mode == ForwardMode.DECODE:
            for (req_id, position), op_positions in zip(next_positions, text.pos_ranges):
                tensor = torch.tensor([position], dtype=torch.long, device=device)
                self._store_position_relay(
                    request_states.get(int(req_id)),
                    position_id=position,
                    position_tensor=tensor,
                )
        return torch.stack(rows, dim=0), [int(req_id) for req_id in text.req_ids]

    def _run_model_forward(
        self,
        model: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        fb: "ForwardBatch",
        ctx: Any,
        *,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
        require_graph: bool = False,
    ) -> torch.Tensor | None:
        # System-owned CUDA graphs capture/replay around the graph-unaware model;
        # a miss (or graphs disabled) falls through to the eager forward.
        active_graph_runner = (
            self.graph_runner if graph_runner is _GRAPH_RUNNER_UNSET else graph_runner
        )
        if active_graph_runner is not None:
            logits = active_graph_runner.maybe_run(model, input_ids, positions, fb, ctx)
            if logits is not None:
                return logits
        if require_graph:
            return None
        return _text_model_forward(
            model,
            fb,
            input_ids=input_ids,
            positions=positions,
        )

    def _reshape_inputs(
        self, fb: "ForwardBatch", text: "TextBatch"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick the model input geometry (flat varlen vs rectangular) for the mode.

        Decode is rectangular ``[batch, 1]`` (the batched-decode kernels); ragged
        or padded extend is flat ``[total]`` (gathered logits via
        ``last_token_indices``); equal-length extend is rectangular ``[batch, L]``;
        mixed stays flat varlen.
        """

        mode = text.mode
        if fb.input_ids is None or fb.positions is None:
            raise invalid_descriptor("text forward batch is missing input ids or positions")
        if mode == ForwardMode.DECODE:
            return fb.input_ids.reshape(fb.batch_size, 1), fb.positions.reshape(fb.batch_size, 1)
        if mode == ForwardMode.EXTEND:
            return fb.input_ids, fb.positions
        return fb.input_ids, fb.positions

    def _graph_padded_num_tokens(
        self,
        text: "TextBatch",
        ctx: Any,
        *,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
    ) -> int | None:
        active_graph_runner = (
            self.graph_runner if graph_runner is _GRAPH_RUNNER_UNSET else graph_runner
        )
        if active_graph_runner is None:
            return None
        return active_graph_runner.padded_num_tokens(
            text, attention_preference=ctx.attention_preference
        )

    def _advance_kv_lengths(self, text: "TextBatch", request_states: RequestStateTable) -> None:
        for req_id, pos_range in zip(text.req_ids, text.pos_ranges):
            request_states.get(int(req_id)).set_kv_length(int(pos_range[1]), lane=_KV_LANE)

    # ---- deferred sampling -----------------------------------------------

    def _publish_logits(
        self,
        ops: Sequence[Mapping[str, Any]],
        req_ids: Sequence[int],
        logits_batch: torch.Tensor,
        tensor_store: Any,
    ) -> list[dict[str, Any]]:
        if logits_batch.ndim != 2 or int(logits_batch.shape[0]) != len(ops):
            raise invalid_descriptor("deferred-sampler logits must be shaped [ops, vocab]")
        if logits_batch.is_cuda:
            torch.cuda.synchronize(logits_batch.device)
        results: list[dict[str, Any]] = []
        for row, op in enumerate(ops):
            handle = tensor_store.publish(logits_batch[row].contiguous(), "logits")
            result: dict[str, Any] = {"req_id": int(op["req_id"]), "logits_handle": int(handle)}
            locator = tensor_store.locator_of(handle)
            if locator is not None:
                result["locator"] = base64.b64encode(locator).decode("ascii")
            results.append(result)
        return results

    # ---- sampling --------------------------------------------------------

    def _step_prompt_prefill(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_sampling: bool,
        tensor_store: Any | None,
    ) -> list[Any]:
        if any(str(op.get("kind")) != "prefill_und" for op in ops):
            raise invalid_descriptor("prompt scoring is valid only for prefill_und operations")

        final_logits: list[torch.Tensor] = []
        prompt_scores: list[list[list[tuple[int, float, int]]] | None] = []
        for row, (op, req_id, tokens, pos_range) in enumerate(
            zip(ops, text.req_ids, text.token_ids, text.pos_ranges, strict=True)
        ):
            del row
            state = request_states.get(int(req_id))
            logits = self._forward_prompt_op(
                model,
                op,
                tokens,
                pos_range,
                int(req_id),
                request_states,
            )
            rows = logits.reshape(-1, logits.shape[-1])
            if bool(op.get("return_all_logits")) and int(rows.shape[0]) != len(tokens):
                raise invalid_descriptor(
                    "prompt-scoring model output must contain one logits row per input token"
                )
            chunk_scores = self._score_prompt_chunk(state, rows, tokens)
            prompt_scores.append(chunk_scores or None)
            final_logits.append(rows[-1])
            state.set_kv_length(int(pos_range[1]), lane=_KV_LANE)

        logits_batch = torch.stack(final_logits, dim=0)
        if defer_sampling:
            if tensor_store is None:
                raise invalid_descriptor("deferred prompt sampling requires a tensor store")
            results = self._publish_logits(ops, text.req_ids, logits_batch, tensor_store)
            for result, prompt_score in zip(results, prompt_scores, strict=True):
                if prompt_score is not None:
                    result["prompt_logprobs"] = prompt_score
            return results

        outputs: list[TextTokenOutput] = []
        for row, (op, req_id, prompt_score) in enumerate(
            zip(ops, text.req_ids, prompt_scores, strict=True)
        ):
            state = request_states.get(int(req_id))
            sample = sample_one_from_logits(
                logits_batch[row],
                dict(state.sampling or {}),
                recent=op.get("recent_tokens") or [],
                allowed=op.get("allowed_tokens"),
                suppress=op.get("suppress_tokens"),
                n_logprobs=int(state.sampling.get("n_logprobs", 0) or 0),
                generator=state.device_rng(
                    logits_batch.device,
                    stream="text_sampling",
                ),
            )
            self._store_sampled_token_relay(
                state,
                token_id=int(sample.token_id),
                token_tensor=torch.tensor(
                    [int(sample.token_id)], dtype=torch.long, device=logits_batch.device
                ),
            )
            outputs.append(
                TextTokenOutput(
                    req_id=int(req_id),
                    sampled_token_id=int(sample.token_id),
                    sampled_logprob=sample.logprob,
                    top_logprobs=(
                        [
                            (int(item[0]), float(item[1]), int(item[2]))
                            for item in sample.top_logprobs
                        ]
                        if sample.top_logprobs
                        else None
                    ),
                    prompt_logprobs=prompt_score,
                )
            )
        return outputs

    def _forward_prompt_op(
        self,
        model: Any,
        op: Mapping[str, Any],
        tokens: tuple[int, ...],
        pos_range: tuple[int, int],
        req_id: int,
        request_states: RequestStateTable,
    ) -> torch.Tensor:
        if not tokens:
            raise invalid_descriptor("prompt-scoring prefill operation has no token ids")
        if self.builder is None or self.kv_pool is None:
            predecessor = getattr(model, "prompt_predecessor_logits", None)
            if bool(op.get("return_all_logits")) and callable(predecessor):
                previous_logits = predecessor(req_id)
                if isinstance(previous_logits, torch.Tensor) and previous_logits.ndim > 0:
                    request_states.get(req_id).prompt_last_logits = previous_logits.reshape(
                        -1, previous_logits.shape[-1]
                    )[-1].detach()
            logits = model.run_text_logits(dict(op))
        else:
            ctx = get_forward_context()
            device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
            batch = self.builder.build_text_op(
                op=op,
                token_ids=tokens,
                pos_range=pos_range,
                req_id=req_id,
                mode=ForwardMode.EXTEND,
                device=device,
                kv_pool=self.kv_pool,
                request_states=request_states,
            )
            batch.return_all_logits = bool(op.get("return_all_logits"))
            with use_forward_context(
                replace(ctx, attention_plan=batch.attn_plan, kv_pool=self.kv_pool)
            ):
                logits = _text_model_forward(model, batch)
        if not isinstance(logits, torch.Tensor) or logits.ndim == 0:
            raise invalid_descriptor("prompt-scoring model output must be a logits tensor")
        return logits

    @staticmethod
    def _score_prompt_chunk(
        state: RequestState,
        logits: torch.Tensor,
        tokens: tuple[int, ...],
    ) -> list[list[tuple[int, float, int]]]:
        sampling = dict(state.sampling or {})
        if not (
            bool(sampling.get("return_prompt_logprobs"))
            or int(sampling.get("n_prompt_logprobs", 0) or 0) > 0
        ):
            return []
        predictors: list[torch.Tensor] = []
        targets: list[int] = []
        if state.prompt_last_logits is not None:
            predictors.append(state.prompt_last_logits.reshape(1, -1))
            targets.append(int(tokens[0]))
        if len(tokens) > 1:
            predictors.append(logits[:-1])
            targets.extend(int(token_id) for token_id in tokens[1:])
        state.prompt_last_logits = logits[-1].detach()
        if not predictors:
            return []
        return score_prompt_token_logprobs(
            torch.cat(predictors, dim=0),
            targets,
            n_logprobs=int(sampling.get("n_prompt_logprobs", 0) or 0),
            logprob_token_ids=sampling.get("logprob_token_ids") or (),
        )

    def _sample_logits_batch(
        self,
        ops: Sequence[Mapping[str, Any]],
        req_ids: Sequence[int],
        logits_batch: torch.Tensor,
        request_states: RequestStateTable,
        stats: ForwardStats | None,
        start: int,
        *,
        defer_cpu_results: bool = False,
        cuda_ready_start_event: torch.cuda.Event | None = None,
    ) -> list[TextTokenOutput | DeferredTextSeqResult]:
        if logits_batch.ndim != 2:
            raise invalid_descriptor("batched text logits rows must form a [batch, vocab] tensor")
        if int(logits_batch.shape[0]) != len(req_ids):
            raise invalid_descriptor("batched text logits row count must match req_ids")
        params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        generators: list[torch.Generator] = []
        for op, req_id in zip(ops, req_ids):
            state = request_states.get(req_id)
            params.append(dict(state.sampling or {}))
            recent.append(op.get("recent_tokens") or [])
            allowed.append(op.get("allowed_tokens"))
            suppress.append(op.get("suppress_tokens"))
            generators.append(state.device_rng(logits_batch.device, stream="text_sampling"))
        with profile_range("uniserve.text.apply_sampling"):
            sampling_result = apply_sampling_batched_with_device_tokens(
                logits_batch,
                params,
                recent,
                allowed,
                suppress,
                generators=generators,
                defer_cpu=defer_cpu_results,
                enable_cuda_timing=cuda_ready_start_event is not None,
            )
        if is_deferred_sampling_result(sampling_result):
            sampling_result.set_ready_start_event(cuda_ready_start_event)
            out: list[TextTokenOutput | DeferredTextSeqResult] = []
            for row, req_id in enumerate(req_ids):
                state = request_states.get(req_id)
                relay_token_tensor = sampling_result.device_tokens[row : row + 1]
                self._store_sampled_token_relay(
                    state, token_id=None, token_tensor=relay_token_tensor
                )
                out.append(
                    DeferredTextSeqResult(
                        req_id=req_id,
                        row=row,
                        state=state,
                        sampling_result=sampling_result,
                        relay_token_tensor=relay_token_tensor,
                    )
                )
            record_component_elapsed(stats, "text_sample", start)
            return out

        immediate_result = finalize_sampling_result(sampling_result)
        samples = immediate_result.samples
        out = []
        for row, (req_id, (tok, lp, top)) in enumerate(zip(req_ids, samples)):
            self._store_sampled_token_relay(
                request_states.get(req_id),
                token_id=int(tok),
                token_tensor=immediate_result.device_tokens[row : row + 1],
            )
            out.append(
                TextTokenOutput(
                    req_id=int(req_id),
                    sampled_token_id=tok,
                    sampled_logprob=lp,
                    top_logprobs=(
                        [(int(item[0]), float(item[1]), int(item[2])) for item in top]
                        if top
                        else None
                    ),
                )
            )
        record_component_elapsed(stats, "text_sample", start)
        return out

    # ---- decode relays ---------------------------------------------------

    def _per_op_relay_input(
        self,
        op: Mapping[str, Any],
        tokens: tuple[int, ...],
        state: RequestState,
        device: torch.device,
    ) -> torch.Tensor | None:
        source = str(op.get("token_source") or "wire")
        if source not in {"wire", "last_sampled"}:
            raise invalid_descriptor(f"unsupported text token_source {source!r}")
        if source != "last_sampled":
            return None
        if len(tokens) != 1:
            raise invalid_descriptor(
                "decode op requested token_source='last_sampled' but does not have exactly one token"
            )
        return _DECODE_RELAY.consume_token(
            state,
            expected_token_id=None,
            device=device,
            token_source=source,
            require=True,
        )

    def _decode_relay_tensors(
        self,
        text: "TextBatch",
        request_states: RequestStateTable,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if text.mode != ForwardMode.DECODE:
            return None, None
        return _DECODE_RELAY.resolve_decode_batch(
            text,
            request_states,
            device,
            stats=get_forward_context().stats,
        )

    def _store_decode_position_relays(
        self,
        text: "TextBatch",
        fb: "ForwardBatch",
        request_states: RequestStateTable,
    ) -> None:
        if text.mode != ForwardMode.DECODE:
            return
        if any(len(tokens) != 1 for tokens in text.token_ids):
            return
        if fb.positions is None:
            raise invalid_descriptor("decode position relay requires position ids")
        next_positions = fb.positions.reshape(-1) + 1
        for row, (req_id, pos_range) in enumerate(zip(text.req_ids, text.pos_ranges)):
            self._store_position_relay(
                request_states.get(int(req_id)),
                position_id=int(pos_range[1]),
                position_tensor=next_positions[row : row + 1],
            )

    @staticmethod
    def _store_sampled_token_relay(
        state: RequestState,
        *,
        token_id: int | None,
        token_tensor: torch.Tensor,
    ) -> None:
        _DECODE_RELAY.publish_sample(state, token_id=token_id, token_tensor=token_tensor)

    @staticmethod
    def _store_position_relay(
        state: RequestState,
        *,
        position_id: int,
        position_tensor: torch.Tensor,
    ) -> None:
        _DECODE_RELAY.publish_position(
            state,
            position_id=position_id,
            position_tensor=position_tensor,
        )


def _can_decode_burst(
    text: "TextBatch", ops: list[Mapping[str, Any]], *, defer_sampling: bool
) -> bool:
    if defer_sampling or text.mode != ForwardMode.DECODE:
        return False
    if any(text.spec_token_ids):
        return False
    try:
        return any(
            _positive_int(op.get("decode_token_count") or 1, "decode_token_count") > 1 for op in ops
        )
    except Exception:
        raise


def _record_cuda_ready_start_event(
    kv_pool: "PagedKVPool | None",
    *,
    stats: ForwardStats | None,
    defer_cpu_results: bool,
) -> torch.cuda.Event | None:
    if stats is None or not defer_cpu_results:
        return None
    tensor = getattr(kv_pool, "k", None)
    device = getattr(tensor, "device", None)
    if not isinstance(device, torch.device) or device.type != "cuda":
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream(device))
    return event
