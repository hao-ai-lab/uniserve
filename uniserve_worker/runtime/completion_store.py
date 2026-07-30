"""Bounded query-only host-observation storage for operation completions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

import torch

from ..foundation.errors import ErrorCode, WorkerError, resource_error
from .device_events import DeviceEventPool
from .host_staging import canonical_device

__all__ = ["CompletionArena", "CompletionCapture", "CompletionLease"]

_MAX_GENERATION: Final[int] = (1 << 32) - 1


def _invariant(message: str) -> WorkerError:
    return WorkerError(
        code=ErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


@dataclass(slots=True)
class _CompletionSlot:
    host_tokens: torch.Tensor
    events: dict[str, torch.cuda.Event]
    devices: set[torch.device] | None = None
    generation: int = 0
    owner: int = 0
    rows: int = 0
    observed: set[int] | None = None
    sealed: bool = False
    abandoned: bool = False
    sealed_ns: int = 0
    ready_ns: int = 0


@dataclass(frozen=True, slots=True)
class CompletionCapture:
    """One bounded token range copied into a completion lease's pinned slab."""

    lease: CompletionLease
    offset: int
    count: int

    def ready(self) -> bool:
        return self.lease.ready()

    def values(self) -> tuple[int, ...]:
        return self.lease.read_tokens(self)


class CompletionLease:
    """Exclusive generation-tagged ownership of one completion arena slab."""

    __slots__ = (
        "_arena",
        "_slot_index",
        "_owner",
        "_generation",
        "_rows",
        "_token_cursor",
        "_token_cache",
    )

    def __init__(
        self,
        arena: CompletionArena,
        slot_index: int,
        owner: int,
        generation: int,
        rows: int,
    ) -> None:
        self._arena = arena
        self._slot_index = int(slot_index)
        self._owner = int(owner)
        self._generation = int(generation)
        self._rows = int(rows)
        self._token_cursor = 0
        self._token_cache: dict[tuple[int, int], tuple[int, ...]] = {}

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def row_count(self) -> int:
        return self._rows

    def register_device(self, device: torch.device | str) -> None:
        slot = self._arena._require_slot(self)
        if slot.sealed:
            raise _invariant("completion device was registered after its slot was sealed")
        target = canonical_device(device)
        if target.type != "cuda":
            return
        if target not in self._arena.devices:
            raise _invariant("completion work uses an undeclared CUDA device")
        if slot.devices is None:
            raise _invariant("completion slot has no device readiness set")
        slot.devices.add(target)

    def device_event(self, device: torch.device | str) -> torch.cuda.Event | None:
        """Return this generation's completion fence before it is recorded."""

        slot = self._arena._require_slot(self)
        if slot.sealed:
            raise _invariant("completion event was requested after its slot was sealed")
        target = canonical_device(device)
        if target.type != "cuda":
            return None
        self.register_device(target)
        device_name = str(target)
        event = slot.events.get(device_name)
        if event is None:
            event = self._arena.event_pool.acquire(target)
            self._arena.event_pool.retain(event, target)
            slot.events[device_name] = event
        return event

    def capture(self, tokens: torch.Tensor) -> CompletionCapture:
        flat = tokens.reshape(-1)
        count = int(flat.numel())
        offset = self._token_cursor
        end = offset + count
        slot = self._arena._require_slot(self)
        if slot.sealed:
            raise _invariant("completion capture was registered after its slot was sealed")
        if end > int(slot.host_tokens.numel()):
            raise resource_error("completion token span exceeds its reserved pinned-host capacity")
        host = slot.host_tokens[offset:end]
        if flat.device.type == "cuda":
            if not bool(host.is_pinned()):
                raise _invariant("CUDA completion copy targets pageable host storage")
            if slot.devices is None:
                raise _invariant("completion slot has no device readiness set")
            device = canonical_device(flat.device)
            self.register_device(device)
            host.copy_(flat, non_blocking=True)
        else:
            host.copy_(flat.to(device="cpu"))
        self._token_cursor = end
        return CompletionCapture(self, offset, count)

    def seal(self) -> None:
        slot = self._arena._require_slot(self)
        if slot.sealed:
            return
        for device in slot.devices or ():
            event = slot.events.get(str(device))
            if event is None:
                event = self._arena.event_pool.acquire(device)
                self._arena.event_pool.retain(event, device)
                slot.events[str(device)] = event
            self._arena.event_pool.record(event, device)
        slot.sealed = True
        slot.sealed_ns = time.perf_counter_ns()

    def ready(self) -> bool:
        slot = self._arena._require_slot(self)
        if not slot.sealed:
            return False
        if slot.ready_ns != 0:
            return True
        ready = all(bool(event.query()) for event in slot.events.values())
        if ready and slot.ready_ns == 0:
            slot.ready_ns = time.perf_counter_ns()
        return ready

    def read_tokens(self, capture: CompletionCapture) -> tuple[int, ...]:
        if capture.lease is not self:
            raise _invariant("completion capture belongs to a different arena lease")
        key = (int(capture.offset), int(capture.count))
        cached = self._token_cache.get(key)
        if cached is not None:
            return cached
        if not self.ready():
            raise _invariant("completion storage was observed before its copy event was ready")
        slot = self._arena._require_slot(self)
        end = capture.offset + capture.count
        if capture.offset < 0 or end > self._token_cursor:
            raise _invariant("completion capture range is outside its registered token extent")
        values = tuple(int(value) for value in slot.host_tokens[capture.offset : end].tolist())
        self._token_cache[key] = values
        return values

    def observe(self, row: int, generation: int) -> tuple[int, int]:
        return self._arena._observe(self, row, generation)

    def discard(self, row: int, generation: int) -> None:
        self._arena._discard(self, row, generation)

    def abandon(self) -> None:
        self._arena._abandon(self)


class CompletionArena:
    """Fixed-capacity pinned completion slabs matched to execution pipeline depth.

    A lease never waits for a prior generation. Exhaustion is reported as
    backpressure, and abandoned generations become reusable only after every
    recorded event reports ready through ``query``.
    """

    def __init__(
        self,
        *,
        depth: int,
        token_capacity: int,
        devices: tuple[torch.device | str, ...] = (),
        event_pool: DeviceEventPool | None = None,
    ) -> None:
        self.depth = max(1, int(depth))
        self.token_capacity = max(1, int(token_capacity))
        normalized: list[torch.device] = []
        for value in devices:
            device = canonical_device(value)
            if device.type == "cuda" and device not in normalized:
                normalized.append(device)
        self.devices = tuple(normalized)
        self.event_pool = DeviceEventPool() if event_pool is None else event_pool
        pin = bool(self.devices)
        self._slots = [
            _CompletionSlot(
                host_tokens=torch.empty(
                    self.token_capacity,
                    dtype=torch.long,
                    device="cpu",
                    pin_memory=pin,
                ),
                events={},
                devices=set(),
            )
            for _ in range(self.depth)
        ]
        self._cursor = 0
        self._next_owner = 1

    def reserve(
        self,
        rows: int,
        *,
        devices: tuple[torch.device | str, ...] = (),
    ) -> CompletionLease:
        count = int(rows)
        if count < 1:
            raise ValueError("a completion lease must contain at least one operation row")
        if count > self.token_capacity:
            raise resource_error("completion row count exceeds the arena capacity")
        selected_devices: set[torch.device] = set()
        for value in devices:
            device = canonical_device(value)
            if device.type != "cuda":
                continue
            if device not in self.devices:
                raise _invariant("completion lease names an undeclared CUDA device")
            selected_devices.add(device)
        for offset in range(self.depth):
            index = (self._cursor + offset) % self.depth
            slot = self._slots[index]
            self._reclaim_if_ready(slot)
            if slot.owner != 0:
                continue
            generation = slot.generation + 1
            if generation > _MAX_GENERATION:
                generation = 1
            owner = self._next_owner
            self._next_owner += 1
            slot.generation = generation
            slot.owner = owner
            slot.rows = count
            slot.observed = set()
            slot.devices = selected_devices
            slot.sealed = False
            slot.abandoned = False
            slot.sealed_ns = 0
            slot.ready_ns = 0
            self._cursor = (index + 1) % self.depth
            return CompletionLease(self, index, owner, generation, count)
        raise resource_error("completion arena has no query-ready free generation")

    def _require_slot(self, lease: CompletionLease) -> _CompletionSlot:
        if lease._arena is not self:
            raise _invariant("completion lease belongs to a different arena")
        slot = self._slots[lease._slot_index]
        if slot.owner != lease._owner or slot.generation != lease._generation:
            raise _invariant("stale completion-slot generation")
        return slot

    def _observe(self, lease: CompletionLease, row: int, generation: int) -> tuple[int, int]:
        slot = self._require_slot(lease)
        index = int(row)
        if int(generation) != slot.generation:
            raise _invariant("completion record carries a stale slot generation")
        if index < 0 or index >= slot.rows:
            raise _invariant("completion record row is outside its reserved slot")
        if not lease.ready():
            raise _invariant("completion record was observed before query-ready")
        observed = slot.observed
        if observed is None:
            raise _invariant("completion slot has no observation state")
        observed_ns = time.perf_counter_ns()
        copy_us = max(0, slot.ready_ns - slot.sealed_ns) // 1000
        ready_to_observed_us = max(0, observed_ns - slot.ready_ns) // 1000
        observed.add(index)
        if len(observed) == slot.rows:
            self._release(slot)
        return copy_us, ready_to_observed_us

    def _abandon(self, lease: CompletionLease) -> None:
        try:
            slot = self._require_slot(lease)
        except WorkerError:
            return
        if not slot.sealed:
            lease.seal()
        slot.abandoned = True
        self._reclaim_if_ready(slot)

    def _discard(self, lease: CompletionLease, row: int, generation: int) -> None:
        try:
            slot = self._require_slot(lease)
        except WorkerError:
            return
        index = int(row)
        if int(generation) != slot.generation or index < 0 or index >= slot.rows:
            return
        observed = slot.observed
        if observed is None:
            return
        observed.add(index)
        if len(observed) != slot.rows:
            return
        if slot.sealed and all(bool(event.query()) for event in slot.events.values()):
            self._release(slot)
        else:
            slot.abandoned = True

    def _release(self, slot: _CompletionSlot) -> None:
        for event in slot.events.values():
            self.event_pool.release(event)
        slot.events.clear()
        slot.owner = 0
        slot.rows = 0
        slot.observed = None
        slot.devices = None
        slot.sealed = False
        slot.abandoned = False
        slot.sealed_ns = 0
        slot.ready_ns = 0

    def _reclaim_if_ready(self, slot: _CompletionSlot) -> None:
        if (
            slot.owner != 0
            and slot.abandoned
            and slot.sealed
            and all(bool(event.query()) for event in slot.events.values())
        ):
            self._release(slot)
