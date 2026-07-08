"""Decode-relay ownership for text generation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..foundation.errors import invalid_descriptor

if TYPE_CHECKING:
    from ..contracts.batches import TextBatch
    from ..contracts.forward_stats import ForwardStats
    from ..runtime.request_state import RequestState, RequestStateTable

__all__ = [
    "TextDecodeRelay",
]


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
        if relay_token_tensor.dtype != torch.long or relay_token_tensor.device != device:
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
            or relay_position_tensor.device != device
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
            relay_positions = _coalesce_relay_rows(position_rows)
        else:
            _bump_stat(stats, "text_decode_position_relay_misses")
            relay_positions = None
        return _coalesce_relay_rows(relay_rows), relay_positions

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


def _coalesce_relay_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    if not rows:
        raise invalid_descriptor("decode relay rows must not be empty")
    if len(rows) == 1:
        return rows[0].reshape(-1)
    first = rows[0].reshape(-1)
    if int(first.numel()) != 1:
        return torch.cat([row.reshape(-1) for row in rows], dim=0)
    elem_size = int(first.element_size())
    base_ptr = int(first.data_ptr())
    for idx, row in enumerate(rows):
        flat = row.reshape(-1)
        if (
            int(flat.numel()) != 1
            or flat.dtype != first.dtype
            or flat.device != first.device
            or int(flat.data_ptr()) != base_ptr + idx * elem_size
        ):
            return torch.cat([candidate.reshape(-1) for candidate in rows], dim=0)
    try:
        return first.as_strided((len(rows),), (1,))
    except RuntimeError:
        return torch.cat([candidate.reshape(-1) for candidate in rows], dim=0)


def _same_tensor(lhs: Any, rhs: torch.Tensor) -> bool:
    if not isinstance(lhs, torch.Tensor):
        return False
    if lhs.device != rhs.device or lhs.dtype != rhs.dtype or lhs.shape != rhs.shape:
        return False
    return int(lhs.data_ptr()) == int(rhs.data_ptr())


def _bump_stat(stats: "ForwardStats | None", attr: str, delta: int = 1) -> None:
    if stats is None:
        return
    setattr(stats, attr, int(getattr(stats, attr)) + int(delta))
