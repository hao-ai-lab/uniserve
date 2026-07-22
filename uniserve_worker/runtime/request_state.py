"""Per-request state owned by the shared runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

__all__ = [
    "append_new_block_ids",
    "request_seed",
    "sampling_draw_seed",
    "RequestLifecycle",
    "ResidencyFlags",
    "DecodeRelay",
    "RequestState",
    "RequestStateTable",
]

_U64 = 0xFFFFFFFFFFFFFFFF
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15


def sampling_draw_seed(seed: int, position: int) -> int:
    """Counter-based generator seed for one sampled-token draw.

    The coordinates are semantic: the request seed and the sequence position of
    the token being drawn. The position selects the element of a SplitMix64
    stream and the finalizer decorrelates neighboring coordinates, so the
    randomness of a draw is a pure function of ``(seed, position)`` — thread
    timing, batch composition, draw history, and retry count cannot change it.
    """
    x = (int(seed) + (int(position) + 1) * _SPLITMIX64_GAMMA) & _U64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64
    return (x ^ (x >> 31)) & _U64


def request_seed(request: Mapping[str, Any], req_id: int) -> int:
    sampling = request.get("sampling")
    image = request.get("image")
    candidates = (
        request.get("seed"),
        sampling.get("seed") if isinstance(sampling, Mapping) else None,
        image.get("seed") if isinstance(image, Mapping) else None,
    )
    return next((int(value) for value in candidates if value is not None), int(req_id))


def append_new_block_ids(
    block_ids: list[int],
    new_block_ids: list[int] | tuple[int, ...] | None,
) -> bool:
    """Idempotently ingest a host ``ForwardOp.new_block_ids`` payload.

    This is the single source of truth for the host->worker block-id append
    contract: every consumer of ``new_block_ids`` must route through it so the
    idempotency rule lives in exactly one place. A retried/duplicated op that
    carries the same tail of block ids is a no-op (the tail already matches), so
    appends are not double-applied; any other suffix is appended verbatim.

    Mutates ``block_ids`` in place and returns ``True`` iff blocks were appended.
    """
    new_blocks = [int(block_id) for block_id in (new_block_ids or [])]
    if new_blocks and block_ids[-len(new_blocks) :] != new_blocks:
        block_ids.extend(new_blocks)
        return True
    return False


def _merge_registered_block_ids(block_ids: list[int], registered: list[int]) -> None:
    if not registered:
        return
    if not block_ids:
        block_ids.extend(registered)
        return
    if block_ids == registered:
        return
    if len(registered) > len(block_ids) and registered[: len(block_ids)] == block_ids:
        block_ids[:] = registered
        return
    if len(block_ids) >= len(registered) and block_ids[: len(registered)] == registered:
        return
    append_new_block_ids(block_ids, registered)


class RequestLifecycle(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    GENERATING = "generating"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass
class ResidencyFlags:
    """Transient per-step resource residency flags for runner accounting."""

    image_latent_active: bool = False
    scratch_active: bool = False


@dataclass
class DecodeRelay:
    """On-device relay of the last sampled token/position for the next decode step.

    The text driver keeps the freshly sampled token (and its position) resident
    on-device so the following decode step can consume it without a host round
    trip. ``*_id`` are the host-visible scalars; ``*_tensor`` are the resident
    device copies.
    """

    token_id: int | None = None
    token_tensor: torch.Tensor | None = None
    position_id: int | None = None
    position_tensor: torch.Tensor | None = None


@dataclass
class RequestState:
    """Per-request mutable data owned by the session store."""

    sampling: dict[str, Any] = field(default_factory=dict)
    image: dict[str, Any] = field(default_factory=dict)
    neg_token_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    resident_block_ids: set[int] = field(default_factory=set)
    # Scheduler-declared prefix-cache reuse boundary in tokens: the leading
    # span of ``block_ids`` whose KV is already resident through prefix reuse
    # (0 on a cold admission). Ingested from ``NewRequestData.prefix_len``.
    prefix_len: int = 0
    lora_id: int | None = None
    raw_new_request: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    schedule_cursor: int = 0
    latent: Any = None
    rng: Any = None
    device_rngs: dict[str, torch.Generator] = field(default_factory=dict)
    # Per-lane committed KV lengths (lane -> token count), recorded by the
    # system executor through ``kv_length``/``set_kv_length``. The system-planned
    # text path commits its lane here; interleaved sequence branches commit their
    # lengths on the branch caches owned by the executor ``KvStore``.
    kv_lengths: dict[str, int] = field(default_factory=dict)
    residency: ResidencyFlags = field(default_factory=ResidencyFlags)
    decode_relay: DecodeRelay = field(default_factory=DecodeRelay)
    prompt_last_logits: torch.Tensor | None = None
    cfg_geometry: dict[str, Any] | None = None
    lifecycle: RequestLifecycle = RequestLifecycle.CREATED
    epoch: int = 0
    version: int = 0
    last_op_id: int | None = None
    last_step_id: int | None = None

    def extend_block_ids(self, block_ids: list[int] | tuple[int, ...]) -> None:
        self.block_ids.extend(int(block_id) for block_id in block_ids)

    def append_new_block_ids(self, new_block_ids: list[int] | tuple[int, ...] | None) -> bool:
        """Tail-deduping ingest of a host ``new_block_ids`` payload.

        Thin wrapper over the shared :func:`append_new_block_ids` so the
        idempotency contract is defined exactly once. Returns ``True`` iff
        blocks were appended.
        """
        return append_new_block_ids(self.block_ids, new_block_ids)

    def kv_length(self, lane: str = "default") -> int:
        return int(self.kv_lengths.get(lane, 0))

    def set_kv_length(self, value: int, lane: str = "default") -> None:
        self.kv_lengths[lane] = int(value)

    def device_rng(
        self,
        device: torch.device | str,
        *,
        stream: str = "model",
    ) -> torch.Generator:
        dev = torch.device(device)
        key = f"{stream}:{dev}"
        rng = self.device_rngs.get(key)
        if rng is None:
            seed = int(self.seed if self.seed is not None else 0)
            rng = torch.Generator(device=dev).manual_seed(seed)
            self.device_rngs[key] = rng
        return rng

    def activate_image_latent(self) -> None:
        self.residency.image_latent_active = True
        self.lifecycle = RequestLifecycle.GENERATING

    def deactivate_image_latent(self) -> None:
        self.residency.image_latent_active = False

    def activate_scratch(self) -> None:
        self.residency.scratch_active = True

    def deactivate_scratch(self) -> None:
        self.residency.scratch_active = False

    def clear_generation_state(self, *, reset_cursor: bool) -> None:
        if reset_cursor:
            self.schedule_cursor = 0
        self.latent = None
        self.cfg_geometry = None
        self.prompt_last_logits = None
        self.deactivate_image_latent()
        self.deactivate_scratch()
        self.lifecycle = RequestLifecycle.COMMITTED if reset_cursor else RequestLifecycle.ABORTED


class RequestStateTable:
    """In-memory table of ``RequestState`` keyed by request id."""

    def __init__(self) -> None:
        self._states: dict[int, RequestState] = {}

    def create_or_update(self, req_id: int, new_req: dict[str, Any]) -> RequestState:
        state = self._states.get(req_id)
        existed = state is not None
        if state is None:
            state = RequestState()
            self._states[req_id] = state
            seed = request_seed(new_req, req_id)
            state.seed = seed
            state.rng = torch.Generator(device="cpu")
            state.rng.manual_seed(seed)
        state.sampling = dict(new_req.get("sampling") or {})
        state.lifecycle = RequestLifecycle.ACTIVE
        state.image = dict(new_req.get("image") or {})
        state.neg_token_ids = list(new_req.get("neg_token_ids") or [])
        if "block_ids" in new_req:
            incoming = [int(block_id) for block_id in (new_req.get("block_ids") or [])]
            if existed:
                _merge_registered_block_ids(state.block_ids, incoming)
            else:
                state.block_ids = incoming
        state.prefix_len = int(new_req.get("prefix_len") or 0)
        state.lora_id = new_req.get("lora_id")
        state.raw_new_request = dict(new_req)
        if "epoch" in new_req:
            state.epoch = int(new_req["epoch"])
        cfg = new_req.get("cfg") or state.image.get("cfg")
        if isinstance(cfg, dict):
            state.cfg_geometry = dict(cfg)
        return state

    def get(self, req_id: int) -> RequestState:
        return self._states.setdefault(req_id, RequestState())

    def advance_denoise(self, req_id: int, num_steps_done: int | None = None) -> None:
        state = self.get(req_id)
        if num_steps_done is None:
            state.schedule_cursor += 1
            return
        state.schedule_cursor = int(num_steps_done)

    def commit(self, req_id: int) -> None:
        self.get(req_id).clear_generation_state(reset_cursor=True)

    def abort(self, req_id: int) -> None:
        state = self._states.get(req_id)
        if state is not None:
            state.clear_generation_state(reset_cursor=False)

    def drop(self, req_id: int) -> None:
        self._states.pop(req_id, None)

    def __contains__(self, req_id: int) -> bool:
        return req_id in self._states
