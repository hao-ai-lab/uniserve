"""Request-session ownership for mutable per-request runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
    "RequestSessionTable",
]


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

    The backing ``RequestState`` remains available during migration so existing
    drivers and tests can inspect the same object, but state mutations that have
    lifecycle or ordering invariants live here.
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


class RequestSessionTable(_RequestStateTable):
    """Request-state table with session-level mutation APIs."""

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
