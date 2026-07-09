"""Deferred text sampling result objects shared by text execution paths."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ...contracts.outputs import ForwardOutputBase
from ...foundation.errors import invalid_descriptor
from ...nn.sampler import DeferredBatchedSamplingResult
from ..text_decode_relay import TextDecodeRelay

if TYPE_CHECKING:
    from ..runtime.request_state import RequestState

__all__ = [
    "DeferredDecodeBurstSeqResult",
    "DeferredTerminalDecodeBurstSeqResult",
    "DeferredTextSeqResult",
]

_DECODE_RELAY = TextDecodeRelay()


class DeferredTextSeqResult(ForwardOutputBase):
    """One text seq-result whose CPU token id is finalized at response time."""

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
        if self._finalized is None:
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
        return dict(self._finalized)

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


class DeferredDecodeBurstSeqResult(ForwardOutputBase):
    """Decode-burst result whose final sampled token is still event-backed."""

    def __init__(
        self,
        *,
        req_id: int,
        prefix_token_ids: Sequence[int],
        pending: Any,
    ) -> None:
        object.__setattr__(self, "req_id", int(req_id))
        object.__setattr__(self, "_prefix_token_ids", tuple(int(token) for token in prefix_token_ids))
        object.__setattr__(self, "_pending", pending)
        object.__setattr__(self, "_finalized", None)

    def to_seq_result(self) -> "DeferredDecodeBurstSeqResult":
        return self

    def finalize(self) -> dict[str, Any]:
        if self._finalized is None:
            token_id = self._materialize_pending_token_id()
            token_ids = [*self._prefix_token_ids, token_id]
            object.__setattr__(
                self,
                "_finalized",
                {
                    "req_id": self.req_id,
                    "sampled_token_id": int(token_ids[-1]),
                    "sampled_token_ids": [int(token) for token in token_ids],
                },
            )
        return dict(self._finalized)

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


class DeferredTerminalDecodeBurstSeqResult(ForwardOutputBase):
    """Terminal-stop decode-burst result with all sampled tokens deferred."""

    def __init__(
        self,
        *,
        req_id: int,
        pending_tokens: Sequence[Any],
        stop_token_ids: Sequence[int],
    ) -> None:
        object.__setattr__(self, "req_id", int(req_id))
        object.__setattr__(self, "_pending_tokens", tuple(pending_tokens))
        object.__setattr__(self, "_stop_token_ids", frozenset(int(token) for token in stop_token_ids))
        object.__setattr__(self, "_finalized", None)

    def to_seq_result(self) -> "DeferredTerminalDecodeBurstSeqResult":
        return self

    def finalize(self) -> dict[str, Any]:
        if self._finalized is None:
            token_ids: list[int] = []
            for pending in self._pending_tokens:
                token_id = _materialize_pending_token_id(pending)
                token_ids.append(token_id)
                if token_id in self._stop_token_ids:
                    break
            if not token_ids:
                raise invalid_descriptor("terminal decode burst did not produce a sampled token")
            object.__setattr__(
                self,
                "_finalized",
                {
                    "req_id": self.req_id,
                    "sampled_token_id": int(token_ids[-1]),
                    "sampled_token_ids": [int(token) for token in token_ids],
                },
            )
        return dict(self._finalized)

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
