"""Request-session ownership for mutable per-request runtime state."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch

from .request_state import (
    RequestLifecycle,
    RequestState,
    append_new_block_ids,
)
from .request_state import (
    RequestStateTable as _RequestStateTable,
)

__all__ = [
    "PreparedTextRow",
    "RequestSession",
    "SessionStore",
    "StepTxn",
]

if TYPE_CHECKING:
    from .resources import ResourceRuntime


@dataclass(frozen=True)
class PreparedTextRow:
    """Resolved text-row residency for one request/op pair."""

    req_id: int
    op: Mapping[str, Any]
    block_ids: tuple[int, ...]
    base_len: int
    query_len: int


class RequestSession:
    """Owns state transitions for one request.

    The backing ``RequestState`` stores the session data, while every mutation
    with lifecycle or ordering invariants is defined here.
    """

    def __init__(self, req_id: int, state: RequestState) -> None:
        self.req_id = int(req_id)
        self.state = state

    def ingest_registered_blocks(self, block_ids: list[int] | tuple[int, ...] | None) -> None:
        incoming = [int(block_id) for block_id in (block_ids or ())]
        if not incoming:
            return
        if not self.state.block_ids:
            self.state.block_ids.extend(incoming)
            return
        if self.state.block_ids == incoming:
            return
        if (
            len(incoming) > len(self.state.block_ids)
            and incoming[: len(self.state.block_ids)] == self.state.block_ids
        ):
            self.state.block_ids[:] = incoming
            return
        if (
            len(self.state.block_ids) >= len(incoming)
            and self.state.block_ids[: len(incoming)] == incoming
        ):
            return
        self.ingest_new_blocks(incoming)

    def ingest_new_blocks(self, block_ids: list[int] | tuple[int, ...] | None) -> bool:
        return append_new_block_ids(self.state.block_ids, block_ids)

    def resolve_text_row(self, op: Mapping[str, Any]) -> PreparedTextRow:
        self.ingest_new_blocks(tuple(int(block_id) for block_id in (op.get("new_block_ids") or ())))
        pos_range = op.get("pos_range") or (0, 0)
        base_len = int(pos_range[0])
        query_len = len(op.get("token_ids") or ())
        return PreparedTextRow(
            req_id=self.req_id,
            op=op,
            block_ids=tuple(int(block_id) for block_id in self.state.block_ids),
            base_len=base_len,
            query_len=query_len,
        )

    def advance_denoise(self, num_steps_done: int | None = None) -> None:
        if num_steps_done is None:
            self.state.schedule_cursor += 1
            return
        self.state.schedule_cursor = int(num_steps_done)

    def activate_image_latent(self) -> None:
        self.state.activate_image_latent()

    def deactivate_image_latent(self) -> None:
        self.state.deactivate_image_latent()

    def activate_scratch(self) -> None:
        self.state.activate_scratch()

    def deactivate_scratch(self) -> None:
        self.state.deactivate_scratch()

    def finish_generation(self, *, committed: bool) -> None:
        self.state.clear_generation_state(reset_cursor=bool(committed))

    def device_rng(
        self,
        device: torch.device | str,
        *,
        stream: str = "model",
    ) -> torch.Generator:
        return self.state.device_rng(device, stream=stream)

    def mark_active(self) -> None:
        self.state.lifecycle = RequestLifecycle.ACTIVE


class SessionStore(_RequestStateTable):
    """System authority for request sessions and their mutation scopes."""

    def session(self, req_id: int) -> RequestSession:
        return RequestSession(int(req_id), self.get(int(req_id)))

    def admit(self, req_id: int, new_req: Mapping[str, Any]) -> RequestSession:
        state = self.create_or_update(int(req_id), dict(new_req))
        return RequestSession(int(req_id), state)

    def resolve_text_row(self, req_id: int, op: Mapping[str, Any]) -> PreparedTextRow:
        return self.session(int(req_id)).resolve_text_row(op)

    def ingest_new_blocks(
        self,
        req_id: int,
        block_ids: list[int] | tuple[int, ...] | None,
    ) -> bool:
        return self.session(int(req_id)).ingest_new_blocks(block_ids)

    def finish_generation(self, req_id: int, *, committed: bool) -> None:
        self.session(int(req_id)).finish_generation(committed=committed)

    def begin_step(
        self,
        step_id: int,
        request_ids: set[int],
        resources: "ResourceRuntime",
    ) -> "StepTxn":
        return StepTxn(self, resources, step_id=int(step_id), request_ids=request_ids)


@dataclass(frozen=True)
class _SessionSnapshot:
    existed: bool
    state: RequestState | None
    rng_state: torch.Tensor | None
    device_rng_states: dict[str, torch.Tensor]


class StepTxn:
    """Atomic mutation scope for system-owned request and resource state."""

    def __init__(
        self,
        sessions: SessionStore,
        resources: "ResourceRuntime",
        *,
        step_id: int,
        request_ids: set[int],
    ) -> None:
        self.sessions = sessions
        self.resources = resources
        self.step_id = int(step_id)
        self.request_ids = {int(request_id) for request_id in request_ids}
        self._snapshots = {
            request_id: self._snapshot(request_id) for request_id in self.request_ids
        }
        self._resource_snapshot = resources.snapshot_requests(self.request_ids)
        self._closed = False

    def commit(self) -> None:
        self._require_open()
        for request_id in self.request_ids:
            state = self.sessions._states.get(request_id)
            if state is not None:
                state.version += 1
                state.last_step_id = self.step_id
        self._closed = True

    def rollback(self) -> None:
        if self._closed:
            return
        for request_id, snapshot in self._snapshots.items():
            if not snapshot.existed:
                self.sessions._states.pop(request_id, None)
                continue
            if snapshot.state is None:
                raise RuntimeError("existing session snapshot is missing state")
            self.sessions._states[request_id] = snapshot.state
            if snapshot.state.rng is not None and snapshot.rng_state is not None:
                snapshot.state.rng.set_state(snapshot.rng_state)
            for key, rng_state in snapshot.device_rng_states.items():
                rng = snapshot.state.device_rngs.get(key)
                if rng is not None:
                    rng.set_state(rng_state)
        self.resources.restore_requests(self.request_ids, self._resource_snapshot)
        self._closed = True

    def _snapshot(self, request_id: int) -> _SessionSnapshot:
        state = self.sessions._states.get(request_id)
        if state is None:
            return _SessionSnapshot(False, None, None, {})
        snapshot = copy.copy(state)
        snapshot.sampling = dict(state.sampling)
        snapshot.image = dict(state.image)
        snapshot.neg_token_ids = list(state.neg_token_ids)
        snapshot.block_ids = list(state.block_ids)
        snapshot.resident_block_ids = set(state.resident_block_ids)
        snapshot.device_rngs = dict(state.device_rngs)
        snapshot.kv_lengths = dict(state.kv_lengths)
        snapshot.residency = copy.copy(state.residency)
        snapshot.decode_relay = copy.copy(state.decode_relay)
        snapshot.cfg_geometry = dict(state.cfg_geometry) if state.cfg_geometry is not None else None
        rng_state = state.rng.get_state().clone() if state.rng is not None else None
        device_rng_states = {
            key: generator.get_state().clone() for key, generator in state.device_rngs.items()
        }
        return _SessionSnapshot(True, snapshot, rng_state, device_rng_states)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("step transaction is already closed")
