"""Decode-relay ownership for text generation."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from ..foundation.errors import invalid_descriptor
from ..runtime.host_staging import (
    canonical_device,
    copy_cpu_to_device,
    cpu_int_staging_buffer,
    fill_cpu_ints,
    is_pinned,
)
from ..runtime.tensor_views import coalesce_one_token_rows

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
