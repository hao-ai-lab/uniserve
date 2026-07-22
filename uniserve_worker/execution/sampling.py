"""Shared autoregressive relay, sampling, burst, and speculative-result behavior."""

from __future__ import annotations

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
)
from uniserve_worker.contracts.forward_context import (
    ForwardStats,
    get_forward_context,
    use_forward_context,
)
from uniserve_worker.contracts.forward_mode import ForwardMode
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
)
from uniserve_worker.runtime.host_staging import (
    canonical_device,
    copy_cpu_to_device,
    cpu_int_staging_buffer,
    fill_cpu_ints,
    is_pinned,
)
from uniserve_worker.runtime.request_state import (
    RequestState,
    RequestStateTable,
    sampling_draw_seed,
)
from uniserve_worker.runtime.tensor_views import coalesce_one_token_rows
from uniserve_worker.spec import speculative_sample_target_only

if TYPE_CHECKING:
    from uniserve_worker.contracts.forward_batch import ForwardBatch

    from .autoregressive import _AutoregressiveRuntime


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
    # Counter-based accept/residual coins: one generator seeded from the first
    # drawn position of this verify op, then the ``len(spec)`` accept coins and
    # the final residual/bonus coin drawn in declared order. A retried or
    # re-batched verify op reproduces the same coins.
    first_drawn_position = int((op.get("pos_range") or (0, 0))[0]) + 1
    generator = sampling_draw_generator(state, logits.device, position=first_drawn_position)
    coins = torch.rand(
        (len(spec),),
        dtype=torch.float32,
        device=logits.device,
        generator=generator,
    )
    final_coin = torch.rand(
        (1,),
        dtype=torch.float32,
        device=logits.device,
        generator=generator,
    )
    spec_sample = speculative_sample_target_only(
        logits[: len(spec) + 1].reshape(len(spec) + 1, -1),
        spec,
        sampling,
        recent=recent,
        allowed=allowed,
        suppress=suppress,
        uniform_samples=coins,
        uniform_sample_for_final=final_coin,
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
    base_position = int((op.get("pos_range") or (0, 0))[0])
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
            generators=[
                sampling_draw_generator(
                    state,
                    logits.device,
                    position=base_position + 1 + pos,
                )
            ],
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
# Sampling RNG coordinates
# ---------------------


def sampled_token_position(op: Mapping[str, Any]) -> int:
    """Sequence position of the token an op's sampler row draws.

    ``pos_range`` spans the op's input tokens, so the freshly drawn token sits
    at ``pos_range[1]`` for prefill chunks, decode steps, and burst iterations
    alike. This is the position coordinate of the counter-based draw seed.
    """
    pos = op.get("pos_range") or (0, 0)
    return int(pos[1])


def sampling_draw_generator(
    state: "RequestState",
    device: torch.device,
    *,
    position: int,
) -> torch.Generator | None:
    """Per-draw generator seeded from the draw's semantic coordinates.

    The single owner of sampled-row RNG policy: greedy rows never draw and get
    ``None`` (no per-row seeding cost); sampled rows reuse the request's cached
    device generator reseeded with ``sampling_draw_seed(request seed, position)``
    so a retried or re-batched draw reproduces the same token. ``manual_seed``
    is a host-side state write — it enqueues no device work and never
    synchronizes.
    """
    sampling = state.sampling
    if not sampling or float(sampling.get("temperature", 0.0) or 0.0) <= 0.0:
        return None
    generator = state.device_rng(device, stream="text_sampling")
    generator.manual_seed(
        sampling_draw_seed(0 if state.seed is None else int(state.seed), int(position))
    )
    return generator


def batched_sampling_inputs(
    ops: Sequence[Mapping[str, Any]],
    req_ids: Sequence[int],
    request_states: "RequestStateTable",
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[list[int] | tuple[int, ...]],
    list[list[int] | tuple[int, ...] | None],
    list[list[int] | tuple[int, ...] | None],
    list[torch.Generator | None],
]:
    """Per-row sampler inputs for a batched text draw.

    Computes the sampling params, penalty/mask lists, and counter-seeded
    generators in one place so every batched text path shares one policy.
    """
    params: list[dict[str, Any]] = []
    recent: list[list[int] | tuple[int, ...]] = []
    allowed: list[list[int] | tuple[int, ...] | None] = []
    suppress: list[list[int] | tuple[int, ...] | None] = []
    generators: list[torch.Generator | None] = []
    for op, req_id in zip(ops, req_ids, strict=True):
        state = request_states.get(int(req_id))
        params.append(dict(state.sampling or {}))
        recent.append(op.get("recent_tokens") or [])
        allowed.append(op.get("allowed_tokens"))
        suppress.append(op.get("suppress_tokens"))
        generators.append(
            sampling_draw_generator(state, device, position=sampled_token_position(op))
        )
    return params, recent, allowed, suppress, generators


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
        generator=sampling_draw_generator(
            state,
            vocab_logits.device,
            position=sampled_token_position(op),
        ),
    )
    result: dict[str, Any] = {"req_id": int(req_id), "sampled_token_id": tok}
    if lp is not None:
        result["sampled_logprob"] = lp
    if top:
        result["top_logprobs"] = top
    return result


def sample_text_rows_batched(
    ops: Sequence[Mapping[str, Any]],
    req_ids: Sequence[int],
    logits_batch: torch.Tensor,
    request_states: "RequestStateTable",
    *,
    defer_cpu_results: bool = False,
    cuda_ready_start_event: "torch.cuda.Event | None" = None,
) -> list[TextTokenOutput | DeferredTextSeqResult]:
    """Sample one token per text row and publish the decode token relays.

    The batched text sampler behind every system text path: it derives the
    per-row sampler inputs (including the counter-seeded draw generators),
    runs the canonical batched pipeline, and wraps rows as immediate
    ``TextTokenOutput`` values or, when the CPU copy is event-backed and
    deferral is requested, as ``DeferredTextSeqResult`` handles.
    """
    if logits_batch.ndim != 2:
        raise invalid_descriptor("batched text logits rows must form a [batch, vocab] tensor")
    if int(logits_batch.shape[0]) != len(req_ids):
        raise invalid_descriptor("batched text logits row count must match req_ids")
    params, recent, allowed, suppress, generators = batched_sampling_inputs(
        ops,
        req_ids,
        request_states,
        logits_batch.device,
    )
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
        deferred_outputs: list[TextTokenOutput | DeferredTextSeqResult] = []
        for row, req_id in enumerate(req_ids):
            state = request_states.get(int(req_id))
            relay_token_tensor = sampling_result.device_tokens[row : row + 1]
            _DECODE_RELAY.publish_sample(
                state,
                token_id=None,
                token_tensor=relay_token_tensor,
            )
            deferred_outputs.append(
                DeferredTextSeqResult(
                    req_id=int(req_id),
                    row=row,
                    state=state,
                    sampling_result=sampling_result,
                    relay_token_tensor=relay_token_tensor,
                )
            )
        return deferred_outputs

    immediate_result = finalize_sampling_result(sampling_result)
    outputs: list[TextTokenOutput | DeferredTextSeqResult] = []
    for row, (req_id, (tok, lp, top)) in enumerate(
        zip(req_ids, immediate_result.samples, strict=True)
    ):
        _DECODE_RELAY.publish_sample(
            request_states.get(int(req_id)),
            token_id=int(tok),
            token_tensor=immediate_result.device_tokens[row : row + 1],
        )
        outputs.append(
            TextTokenOutput(
                req_id=int(req_id),
                sampled_token_id=int(tok),
                sampled_logprob=lp,
                top_logprobs=(
                    [(int(item[0]), float(item[1]), int(item[2])) for item in top]
                    if top
                    else None
                ),
            )
        )
    return outputs
