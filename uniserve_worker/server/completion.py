"""Bounded completion staging and immutable host materialization."""

from __future__ import annotations

import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final, cast, overload

import torch

from ..batch import (
    CompletionRecord,
    CompletionReport,
    FinishFlags,
    FixedPoint,
    LogicalLengths,
    OpStatus,
    PartitionCompletion,
    TokenSpan,
    VersionRef,
)
from ..batch import ErrorCode as ProtocolErrorCode
from ..foundation.device import canonical_device
from ..foundation.errors import ErrorCode, WorkerError, invalid_descriptor, resource_error
from ..runtime.device_events import DeviceEventPool
from ..transfer.tickets import (
    TRANSFER_DESCRIPTOR_PREFIX,
    Locator,
    Transport,
    encode_transfer_descriptor,
)
from .cpu_tasks import CpuTaskReservation
from .image_codec import uint8_image_to_png_base64_bytes
from .request_state import RequestRuntime

__all__ = [
    "CompletionArena",
    "CompletionByteCapture",
    "CompletionCapture",
    "CompletionLease",
    "completion_report_ready",
    "completion_word_capacity",
    "finalize_completion_report",
    "partition_completion_ready",
]

_MAX_GENERATION: Final[int] = (1 << 32) - 1
_SAMPLING_FIELDS_PER_OPERATION: Final[int] = 4


def completion_word_capacity(max_operations: int, payload_bytes: int) -> int:
    operations = int(max_operations)
    payload = int(payload_bytes)
    if operations < 1 or payload < 1:
        raise ValueError("completion geometry must be positive")
    return _SAMPLING_FIELDS_PER_OPERATION * operations + (payload + 3) // 4


def _invariant(message: str) -> WorkerError:
    return WorkerError(
        code=ErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


@dataclass(slots=True)
class _CompletionSlot:
    host_tokens: torch.Tensor
    start_events: dict[str, torch.cuda.Event]
    producer_events: dict[str, torch.cuda.Event]
    events: dict[str, torch.cuda.Event]
    devices: set[torch.device] | None = None
    generation: int = 0
    owner: int = 0
    rows: int = 0
    observed: set[int] | None = None
    sealed: bool = False
    abandoned: bool = False
    reserved_ns: int = 0
    device_started_ns: int = 0
    copy_started_ns: int = 0
    sealed_ns: int = 0
    ready_ns: int = 0
    token_offset: int = 0
    token_capacity: int = 0


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


@dataclass(frozen=True, slots=True)
class CompletionByteCapture:
    """One shaped byte range copied into a completion generation's pinned slab."""

    lease: CompletionLease
    offset: int
    count: int
    shape: tuple[int, ...]

    def ready(self) -> bool:
        return self.lease.ready()

    def tensor(self) -> torch.Tensor:
        return self.lease.read_bytes(self)

    def numpy(self) -> Any:
        return self.tensor().numpy()


class CompletionLease:
    """Exclusive generation-tagged ownership of one completion arena slab."""

    __slots__ = (
        "_arena",
        "_slot_index",
        "_owner",
        "_generation",
        "_rows",
        "_token_cursor",
        "_byte_cursor",
        "_token_cache",
        "_timing",
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
        self._byte_cursor = 0
        self._token_cache: dict[tuple[int, int], tuple[int, ...]] = {}
        self._timing: tuple[int, int, int, int] | None = None

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

    def begin_device(self, device: torch.device | str) -> None:
        """Record this generation's first device-work boundary."""

        slot = self._arena._require_slot(self)
        if slot.sealed:
            raise _invariant("completion device timing began after its slot was sealed")
        target = canonical_device(device)
        if slot.device_started_ns == 0:
            slot.device_started_ns = time.perf_counter_ns()
        if target.type != "cuda":
            return
        self.register_device(target)
        device_name = str(target)
        if device_name in slot.start_events:
            return
        event = self._arena.event_pool.acquire(target, timing=True)
        self._arena.event_pool.retain(event, target)
        self._arena.event_pool.record(event, target)
        slot.start_events[device_name] = event

    def _mark_copy_started(self, device: torch.device) -> None:
        slot = self._arena._require_slot(self)
        if slot.copy_started_ns == 0:
            slot.copy_started_ns = time.perf_counter_ns()
        device_name = str(device)
        if device_name in slot.producer_events:
            return
        if device_name not in slot.start_events:
            self.begin_device(device)
        event = self._arena.event_pool.acquire(device, timing=True)
        self._arena.event_pool.retain(event, device)
        self._arena.event_pool.record(event, device)
        slot.producer_events[device_name] = event

    def capture(self, tokens: torch.Tensor) -> CompletionCapture:
        flat = tokens.reshape(-1).to(dtype=torch.long)
        count = int(flat.numel())
        offset = self._token_cursor
        end = offset + count
        slot = self._arena._require_slot(self)
        if slot.sealed:
            raise _invariant("completion capture was registered after its slot was sealed")
        if end * int(slot.host_tokens.element_size()) > self._byte_floor(slot):
            raise resource_error("completion token span exceeds its reserved pinned-host capacity")
        host = slot.host_tokens[offset:end]
        if flat.device.type == "cuda":
            if not bool(host.is_pinned()):
                raise _invariant("CUDA completion copy targets pageable host storage")
            if slot.devices is None:
                raise _invariant("completion slot has no device readiness set")
            device = canonical_device(flat.device)
            self.register_device(device)
            self._mark_copy_started(device)
            host.copy_(flat, non_blocking=True)
        else:
            host.copy_(flat.to(device="cpu"))
        self._token_cursor = end
        return CompletionCapture(self, offset, count)

    def capture_bytes(self, value: torch.Tensor) -> CompletionByteCapture:
        """Enqueue one contiguous uint8 D2H copy into this generation's byte tail."""

        if value.dtype is not torch.uint8:
            raise ValueError("completion byte capture requires uint8 storage")
        contiguous = value.detach().contiguous()
        count = int(contiguous.numel())
        if count < 1:
            raise ValueError("completion byte capture must not be empty")
        slot = self._arena._require_slot(self)
        if slot.sealed:
            raise _invariant("completion byte capture was registered after its slot was sealed")
        end = self._byte_floor(slot)
        offset = end - count
        token_bytes = self._token_cursor * int(slot.host_tokens.element_size())
        if offset < token_bytes:
            raise resource_error("completion byte span exceeds its reserved pinned-host capacity")
        host = slot.host_tokens.view(torch.uint8)[offset:end]
        flat = contiguous.reshape(-1)
        if flat.device.type == "cuda":
            if not bool(host.is_pinned()):
                raise _invariant("CUDA completion byte copy targets pageable host storage")
            device = canonical_device(flat.device)
            self.register_device(device)
            self._mark_copy_started(device)
            host.copy_(flat, non_blocking=True)
        else:
            host.copy_(flat.to(device="cpu"))
        self._byte_cursor += count
        return CompletionByteCapture(self, offset, count, tuple(int(v) for v in value.shape))

    def _byte_floor(self, slot: _CompletionSlot) -> int:
        return (
            int(slot.host_tokens.numel()) * int(slot.host_tokens.element_size()) - self._byte_cursor
        )

    def seal(self) -> None:
        slot = self._arena._require_slot(self)
        if slot.sealed:
            return
        for device in slot.devices or ():
            device_name = str(device)
            if device_name not in slot.start_events:
                self.begin_device(device)
            if device_name not in slot.producer_events:
                self._mark_copy_started(device)
            event = slot.events.get(device_name)
            if event is None:
                event = self._arena.event_pool.acquire(device, timing=True)
                self._arena.event_pool.retain(event, device)
                slot.events[device_name] = event
            self._arena.event_pool.record(event, device)
            self._arena.schedule_completion_wake(device, event)
        slot.sealed = True
        slot.sealed_ns = time.perf_counter_ns()

    def ready(self) -> bool:
        slot = self._arena._require_slot(self)
        if not slot.sealed:
            return False
        if slot.ready_ns != 0:
            return True
        for event in slot.events.values():
            if not bool(event.query()):
                return False
        if slot.ready_ns == 0:
            slot.ready_ns = time.perf_counter_ns()
        return True

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

    def read_bytes(self, capture: CompletionByteCapture) -> torch.Tensor:
        if capture.lease is not self:
            raise _invariant("completion byte capture belongs to a different arena lease")
        if not self.ready():
            raise _invariant("completion byte storage was observed before its copy event was ready")
        slot = self._arena._require_slot(self)
        end = int(capture.offset) + int(capture.count)
        total = int(slot.host_tokens.numel()) * int(slot.host_tokens.element_size())
        if capture.offset < 0 or end > total:
            raise _invariant("completion byte range is outside its registered extent")
        return slot.host_tokens.view(torch.uint8)[capture.offset : end].view(capture.shape)

    def observe(self, row: int, generation: int) -> tuple[int, int]:
        timing = self._arena._observe(
            self,
            row,
            generation,
            timing=self._timing,
        )
        if self._timing is None:
            self._timing = timing
        return timing[2], timing[3]

    def timing(self) -> tuple[int, int, int, int]:
        if self._timing is None:
            raise _invariant("completion timing was read before observation")
        return self._timing

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
        total_token_capacity: int | None = None,
        devices: tuple[torch.device | str, ...] = (),
        event_pool: DeviceEventPool | None = None,
    ) -> None:
        self.depth = max(1, int(depth))
        self.token_capacity = max(1, int(token_capacity))
        self.total_token_capacity = (
            self.depth * self.token_capacity
            if total_token_capacity is None
            else int(total_token_capacity)
        )
        if self.total_token_capacity < 1:
            raise ValueError("completion arena total token capacity must be positive")
        normalized: list[torch.device] = []
        for value in devices:
            device = canonical_device(value)
            if device.type == "cuda" and device not in normalized:
                normalized.append(device)
        self.devices = tuple(normalized)
        self._owns_event_pool = event_pool is None
        self.event_pool = DeviceEventPool() if event_pool is None else event_pool
        pin = bool(self.devices)
        self._host_tokens = torch.empty(
            self.total_token_capacity,
            dtype=torch.long,
            device="cpu",
            pin_memory=pin,
        )
        self._slots = [
            _CompletionSlot(
                host_tokens=self._host_tokens[:0],
                start_events={},
                producer_events={},
                events={},
                devices=set(),
            )
            for _ in range(self.depth)
        ]
        self._free_token_ranges = [(0, self.total_token_capacity)]
        self._cursor = 0
        self._next_owner = 1
        self._closed = False
        self._wake_on_stream: Callable[[int], None] | None = None
        self._wake_streams: dict[str, torch.cuda.Stream] = {}

    def set_completion_wake(self, wake_on_stream: Callable[[int], None]) -> None:
        self._wake_on_stream = wake_on_stream

    def schedule_completion_wake(
        self,
        device: torch.device,
        event: torch.cuda.Event,
    ) -> None:
        wake_on_stream = self._wake_on_stream
        if wake_on_stream is None:
            return
        device_name = str(device)
        stream = self._wake_streams.get(device_name)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._wake_streams[device_name] = stream
        stream.wait_event(event)
        wake_on_stream(int(stream.cuda_stream))

    def close(self) -> None:
        if self._closed:
            return
        for stream in self._wake_streams.values():
            stream.synchronize()
        self._wake_streams.clear()
        for slot in self._slots:
            for event in (
                *slot.start_events.values(),
                *slot.producer_events.values(),
                *slot.events.values(),
            ):
                if bool(event.query()):
                    self.event_pool.release(event)
            slot.start_events.clear()
            slot.producer_events.clear()
            slot.events.clear()
        self._slots.clear()
        self._free_token_ranges.clear()
        self._host_tokens = torch.empty(0, dtype=torch.long)
        self._closed = True
        if self._owns_event_pool:
            self.event_pool.close()

    def reserve(
        self,
        rows: int,
        *,
        token_capacity: int | None = None,
        devices: tuple[torch.device | str, ...] = (),
    ) -> CompletionLease:
        if self._closed:
            raise RuntimeError("completion arena is closed")
        count = int(rows)
        requested_tokens = self.token_capacity if token_capacity is None else int(token_capacity)
        if count < 1:
            raise ValueError("a completion lease must contain at least one operation row")
        if requested_tokens < 1:
            raise ValueError("a completion lease token capacity must be positive")
        if count > requested_tokens or requested_tokens > self.token_capacity:
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
            allocation = self._allocate_tokens(requested_tokens)
            if allocation is None:
                break
            token_offset, token_count = allocation
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
            slot.reserved_ns = time.perf_counter_ns()
            slot.device_started_ns = 0
            slot.copy_started_ns = 0
            slot.sealed_ns = 0
            slot.ready_ns = 0
            slot.token_offset = token_offset
            slot.token_capacity = token_count
            slot.host_tokens = self._host_tokens[token_offset : token_offset + token_count]
            self._cursor = (index + 1) % self.depth
            return CompletionLease(self, index, owner, generation, count)
        raise resource_error("completion arena has no query-ready slot and byte capacity")

    def _allocate_tokens(self, count: int) -> tuple[int, int] | None:
        for index, (offset, available) in enumerate(self._free_token_ranges):
            if available < count:
                continue
            if available == count:
                del self._free_token_ranges[index]
            else:
                self._free_token_ranges[index] = (offset + count, available - count)
            return offset, count
        return None

    def _release_tokens(self, offset: int, count: int) -> None:
        if count < 1:
            return
        ranges = sorted((*self._free_token_ranges, (offset, count)))
        merged: list[tuple[int, int]] = []
        for current_offset, current_count in ranges:
            if merged and merged[-1][0] + merged[-1][1] == current_offset:
                prior_offset, prior_count = merged[-1]
                merged[-1] = (prior_offset, prior_count + current_count)
            else:
                merged.append((current_offset, current_count))
        self._free_token_ranges = merged

    def _require_slot(self, lease: CompletionLease) -> _CompletionSlot:
        if lease._arena is not self:
            raise _invariant("completion lease belongs to a different arena")
        if self._closed or lease._slot_index < 0 or lease._slot_index >= len(self._slots):
            raise _invariant("completion lease belongs to a closed arena")
        slot = self._slots[lease._slot_index]
        if slot.owner != lease._owner or slot.generation != lease._generation:
            raise _invariant("stale completion-slot generation")
        return slot

    def _observe(
        self,
        lease: CompletionLease,
        row: int,
        generation: int,
        *,
        timing: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
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
        if timing is None:
            queued_us = (
                max(0, slot.device_started_ns - slot.reserved_ns) // 1000
                if slot.device_started_ns > 0
                else 0
            )
            device_us = 0
            copy_us = 0
            for device_name, end_event in slot.events.items():
                start_event = slot.start_events.get(device_name)
                producer_event = slot.producer_events.get(device_name)
                if start_event is None or producer_event is None:
                    raise _invariant("completion timing events are incomplete")
                device_us = max(
                    device_us,
                    max(0, round(float(start_event.elapsed_time(producer_event)) * 1000.0)),
                )
                copy_us = max(
                    copy_us,
                    max(0, round(float(producer_event.elapsed_time(end_event)) * 1000.0)),
                )
            if not slot.events and slot.device_started_ns > 0:
                copy_started_ns = slot.copy_started_ns or slot.sealed_ns
                device_us = max(0, copy_started_ns - slot.device_started_ns) // 1000
                copy_us = max(0, slot.sealed_ns - copy_started_ns) // 1000
            ready_to_observed_us = max(0, observed_ns - slot.ready_ns) // 1000
            timing = (queued_us, device_us, copy_us, ready_to_observed_us)
        observed.add(index)
        if len(observed) == slot.rows:
            self._release(slot)
        return timing

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
        for event in (
            *slot.start_events.values(),
            *slot.producer_events.values(),
            *slot.events.values(),
        ):
            self.event_pool.release(event)
        slot.start_events.clear()
        slot.producer_events.clear()
        slot.events.clear()
        slot.owner = 0
        slot.rows = 0
        slot.observed = None
        slot.devices = None
        slot.sealed = False
        slot.abandoned = False
        slot.reserved_ns = 0
        slot.device_started_ns = 0
        slot.copy_started_ns = 0
        slot.sealed_ns = 0
        slot.ready_ns = 0
        self._release_tokens(slot.token_offset, slot.token_capacity)
        slot.token_offset = 0
        slot.token_capacity = 0
        slot.host_tokens = self._host_tokens[:0]

    def _reclaim_if_ready(self, slot: _CompletionSlot) -> None:
        if (
            slot.owner != 0
            and slot.abandoned
            and slot.sealed
            and all(bool(event.query()) for event in slot.events.values())
        ):
            self._release(slot)

class _CompletionTokenSpan:
    """One token vector backed exclusively by host-observation storage."""

    __slots__ = ("capture", "count", "_values")

    def __init__(self, capture: CompletionCapture) -> None:
        self.capture = capture
        self.count = int(capture.count)
        self._values: tuple[int, ...] | None = None

    def ready(self) -> bool:
        return self._values is not None or self.capture.ready()

    def finalize(self) -> tuple[int, ...]:
        if self._values is None:
            self._values = self.capture.values()
        return self._values


class _CompletionToken:
    """A protocol integer finalized only when the worker serializes its result."""

    __slots__ = ("span", "index")

    def __init__(self, span: _CompletionTokenSpan, index: int) -> None:
        self.span = span
        self.index = int(index)

    def ready(self) -> bool:
        return self.span.ready()

    def finalize(self) -> int:
        return self.span.finalize()[self.index]

    def __int__(self) -> int:
        return self.finalize()

    def __index__(self) -> int:
        return self.finalize()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CompletionToken):
            return self.finalize() == other.finalize()
        if isinstance(other, int):
            return self.finalize() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.finalize())


class _InvalidSamplingDistribution(RuntimeError):
    pass


class _PredicatedOperation(RuntimeError):
    pass


class _CompletionSampleSpan:
    """Selected tokens, row validity, predicates, and accepted counts."""

    __slots__ = ("capture", "count", "_values")

    def __init__(
        self,
        capture: CompletionCapture | None,
        count: int,
        values: tuple[int, ...] | None = None,
    ) -> None:
        self.capture = capture
        self.count = int(count)
        self._values = values

    def ready(self) -> bool:
        return self._values is not None or (self.capture is not None and self.capture.ready())

    def finalize(self) -> tuple[int, ...]:
        if self._values is None:
            if self.capture is None:
                raise RuntimeError("sampling completion metadata has no capture")
            values = self.capture.values()
            if len(values) != self.count * 4:
                raise RuntimeError("sampling completion metadata has an invalid extent")
            self._values = values
        return self._values

    def token(self, index: int) -> int:
        values = self.finalize()
        if not bool(values[self.count + index]):
            raise _PredicatedOperation("operation predicate selected no state")
        if not bool(values[index]):
            raise _InvalidSamplingDistribution("sampling policy produced an invalid distribution")
        return values[self.count * 2 + index]

    def accepted(self, index: int) -> int:
        values = self.finalize()
        if not bool(values[self.count + index]):
            raise _PredicatedOperation("operation predicate selected no state")
        if not bool(values[index]):
            raise _InvalidSamplingDistribution("sampling policy produced an invalid distribution")
        return values[self.count * 3 + index]


class _CompletionSampleToken(_CompletionToken):
    __slots__ = ("sample_span",)

    def __init__(self, span: _CompletionSampleSpan, index: int) -> None:
        self.sample_span = span
        self.span = cast(_CompletionTokenSpan, span)
        self.index = int(index)

    def ready(self) -> bool:
        return self.sample_span.ready()

    def finalize(self) -> int:
        return self.sample_span.token(self.index)


class _CompletionInteger:
    __slots__ = ("span", "index")

    def __init__(self, span: _CompletionSampleSpan, index: int) -> None:
        self.span = span
        self.index = int(index)

    def ready(self) -> bool:
        return self.span.ready()

    def finalize(self) -> int:
        return self.span.accepted(self.index)

    def __int__(self) -> int:
        return self.finalize()

    def __index__(self) -> int:
        return self.finalize()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CompletionInteger):
            return self.finalize() == other.finalize()
        if isinstance(other, int):
            return self.finalize() == other
        return NotImplemented


class _CompletionDerivedInteger:
    __slots__ = ("source", "offset")

    def __init__(self, source: _CompletionInteger, offset: int) -> None:
        self.source = source
        self.offset = int(offset)

    def ready(self) -> bool:
        return self.source.ready()

    def finalize(self) -> int:
        return int(self.source) + self.offset

    def __int__(self) -> int:
        return self.finalize()

    def __index__(self) -> int:
        return self.finalize()


class _CompletionSpeculativePoint:
    __slots__ = ("accepted", "terminal_prefix")

    def __init__(self, accepted: _CompletionInteger, terminal_prefix: int | None) -> None:
        self.accepted = accepted
        self.terminal_prefix = terminal_prefix

    def ready(self) -> bool:
        return self.accepted.ready()

    def finalize(self) -> int:
        accepted = int(self.accepted)
        if self.terminal_prefix is not None and accepted >= self.terminal_prefix:
            return accepted
        return accepted + 1

    def __int__(self) -> int:
        return self.finalize()

    def __index__(self) -> int:
        return self.finalize()


class _CompletionSpeculativeTokens(Sequence[int]):
    __slots__ = ("draft", "accepted", "continuation", "terminal_prefix", "_value")

    def __init__(
        self,
        draft: tuple[int, ...],
        accepted: _CompletionInteger,
        continuation: int | _CompletionToken,
        terminal_prefix: int | None,
    ) -> None:
        self.draft = tuple(int(value) for value in draft)
        self.accepted = accepted
        self.continuation = continuation
        self.terminal_prefix = terminal_prefix
        self._value: tuple[int, ...] | None = None

    def ready(self) -> bool:
        continuation = self.continuation
        return self.accepted.ready() and (
            not isinstance(continuation, _CompletionToken) or continuation.ready()
        )

    def finalize(self) -> tuple[int, ...]:
        if self._value is None:
            accepted = int(self.accepted)
            if accepted < 0 or accepted > len(self.draft):
                raise RuntimeError("speculative acceptance count is outside the draft span")
            if self.terminal_prefix is not None and accepted >= self.terminal_prefix:
                self._value = self.draft[:accepted]
            else:
                self._value = (*self.draft[:accepted], int(self.continuation))
        return self._value

    def __len__(self) -> int:
        return len(self.finalize())

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        return self.finalize()[index]


class _CompletionLogprobBatch:
    """Packed query-ready logprob tensors shared by a sampling group."""

    __slots__ = (
        "capture",
        "rows",
        "counts",
        "requested_ids",
        "max_count",
        "max_requested",
        "_details",
    )

    def __init__(
        self,
        capture: CompletionCapture | None,
        rows: tuple[int, ...],
        counts: tuple[int, ...],
        requested_ids: tuple[tuple[int, ...], ...],
        max_count: int,
        max_requested: int,
        values: tuple[int, ...] | None = None,
    ) -> None:
        self.capture = capture
        self.rows = rows
        self.counts = counts
        self.requested_ids = requested_ids
        self.max_count = int(max_count)
        self.max_requested = int(max_requested)
        self._details: dict[int, tuple[float, tuple[tuple[int, float, int], ...]]] | None = None
        if values is not None:
            self._details = self._decode(values)

    def ready(self) -> bool:
        return self._details is not None or (self.capture is not None and self.capture.ready())

    @staticmethod
    def _float(value: int) -> float:
        return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]

    def finalize(self) -> dict[int, tuple[float, tuple[tuple[int, float, int], ...]]]:
        if self._details is not None:
            return self._details
        if self.capture is None:
            raise RuntimeError("logprob completion metadata has no capture")
        self._details = self._decode(self.capture.values())
        return self._details

    def _decode(
        self,
        values: tuple[int, ...],
    ) -> dict[int, tuple[float, tuple[tuple[int, float, int], ...]]]:
        row_count = len(self.rows)
        cursor = 0

        def vector(width: int) -> tuple[tuple[int, ...], ...]:
            nonlocal cursor
            total = row_count * width
            part = values[cursor : cursor + total]
            if len(part) != total:
                raise RuntimeError("logprob completion metadata is truncated")
            cursor += total
            return tuple(tuple(part[row * width : (row + 1) * width]) for row in range(row_count))

        selected_tokens = vector(1)
        selected_values = vector(1)
        selected_ranks = vector(1)
        top_indexes = vector(self.max_count)
        top_values = vector(self.max_count)
        top_ranks = vector(self.max_count)
        candidate_values = vector(self.max_requested)
        candidate_ranks = vector(self.max_requested)
        if cursor != len(values):
            raise RuntimeError("logprob completion metadata has trailing values")
        details: dict[int, tuple[float, tuple[tuple[int, float, int], ...]]] = {}
        for local, result_index in enumerate(self.rows):
            selected = selected_tokens[local][0]
            selected_value = self._float(selected_values[local][0])
            entries: list[tuple[int, float, int]] = [
                (selected, selected_value, selected_ranks[local][0])
            ]
            seen = {selected}
            for index in range(self.counts[local]):
                candidate = top_indexes[local][index]
                if candidate not in seen:
                    entries.append(
                        (
                            candidate,
                            self._float(top_values[local][index]),
                            top_ranks[local][index],
                        )
                    )
                    seen.add(candidate)
            for index, candidate in enumerate(self.requested_ids[local]):
                if candidate not in seen:
                    entries.append(
                        (
                            candidate,
                            self._float(candidate_values[local][index]),
                            candidate_ranks[local][index],
                        )
                    )
                    seen.add(candidate)
            details[result_index] = (selected_value, tuple(entries))
        return details


class _CompletionLogprobValue:
    __slots__ = ("batch", "index")

    def __init__(self, batch: _CompletionLogprobBatch, index: int) -> None:
        self.batch = batch
        self.index = int(index)

    def ready(self) -> bool:
        return self.batch.ready()

    def finalize(self) -> float:
        return self.batch.finalize()[self.index][0]

    def __float__(self) -> float:
        return self.finalize()


class _CompletionTopLogprobs:
    __slots__ = ("batch", "index")

    def __init__(self, batch: _CompletionLogprobBatch, index: int) -> None:
        self.batch = batch
        self.index = int(index)

    def ready(self) -> bool:
        return self.batch.ready()

    def finalize(self) -> tuple[tuple[int, float, int], ...]:
        return self.batch.finalize()[self.index][1]

    def max_entries(self) -> int:
        local = self.batch.rows.index(self.index)
        return 1 + int(self.batch.counts[local]) + len(self.batch.requested_ids[local])


class _CompletionLogprobPayload:
    __slots__ = ("logprob", "top_logprobs", "prompt_logprobs", "_value")

    def __init__(
        self,
        logprob: float | _CompletionLogprobValue | None,
        top_logprobs: tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs | None,
        prompt_logprobs: tuple[
            tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs,
            ...,
        ] = (),
    ) -> None:
        self.logprob = logprob
        self.top_logprobs = top_logprobs
        self.prompt_logprobs = prompt_logprobs
        self._value: bytes | None = None

    def ready(self) -> bool:
        if self._value is not None:
            return True
        return (
            (not isinstance(self.logprob, _CompletionLogprobValue) or self.logprob.ready())
            and (
                not isinstance(self.top_logprobs, _CompletionTopLogprobs)
                or self.top_logprobs.ready()
            )
            and all(
                not isinstance(position, _CompletionTopLogprobs) or position.ready()
                for position in self.prompt_logprobs
            )
        )

    def max_encoded_bytes(self) -> int:
        def entry_bound(
            entries: tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs | None,
        ) -> int:
            if isinstance(entries, _CompletionTopLogprobs):
                return entries.max_entries()
            return len(entries or ())

        return (
            (5 if self.logprob is not None else 1)
            + 4
            + 12 * entry_bound(self.top_logprobs)
            + 4
            + sum(4 + 12 * entry_bound(position) for position in self.prompt_logprobs)
        )

    def finalize(self) -> bytes:
        if self._value is not None:
            return self._value
        if not self.ready():
            raise RuntimeError("logprob payload was observed before query-ready")
        logprob = None if self.logprob is None else float(self.logprob)
        top = (
            self.top_logprobs.finalize()
            if isinstance(self.top_logprobs, _CompletionTopLogprobs)
            else self.top_logprobs or ()
        )
        out = bytearray(b"\x00" if logprob is None else b"\x01" + struct.pack("<f", logprob))
        out += struct.pack("<I", len(top))
        for token_id, value, rank in top:
            out += struct.pack("<IfI", int(token_id), float(value), int(rank))
        out += struct.pack("<I", len(self.prompt_logprobs))
        for position in self.prompt_logprobs:
            entries = (
                position.finalize() if isinstance(position, _CompletionTopLogprobs) else position
            )
            out += struct.pack("<I", len(entries))
            for token_id, value, rank in entries:
                out += struct.pack("<IfI", int(token_id), float(value), int(rank))
        self._value = bytes(out)
        return self._value

    def __bytes__(self) -> bytes:
        return self.finalize()


class _CompletionTransferPayload:
    __slots__ = (
        "kind",
        "descriptor_value",
        "locators",
        "producer_plan_digest",
        "transport",
        "_value",
    )

    def __init__(
        self,
        kind: str,
        descriptor_value: dict[str, object],
        locators: tuple[Locator, ...],
        producer_plan_digest: str,
        transport: Transport,
    ) -> None:
        self.kind = kind
        self.descriptor_value = descriptor_value
        self.locators = locators
        self.producer_plan_digest = producer_plan_digest
        self.transport = transport
        self._value: bytes | None = None

    def ready(self) -> bool:
        return self._value is not None or all(
            self.transport.ready(locator) for locator in self.locators
        )

    def max_encoded_bytes(self) -> int:
        return len(
            encode_transfer_descriptor(
                self.kind,
                self.descriptor_value,
                self.producer_plan_digest,
            )
        )

    def finalize(self) -> bytes:
        if self._value is None:
            if not self.ready():
                raise RuntimeError("transport descriptor was observed before producer readiness")
            self._value = encode_transfer_descriptor(
                self.kind,
                self.descriptor_value,
                self.producer_plan_digest,
            )
        return self._value

    def __bytes__(self) -> bytes:
        return self.finalize()


class _CompletionImagePayload:
    """Pinned D2H image capture followed by bounded asynchronous PNG encoding."""

    __slots__ = (
        "capture",
        "reservation",
        "max_bytes",
        "_future",
        "_value",
        "_submission_error",
    )

    def __init__(
        self,
        capture: CompletionByteCapture,
        reservation: CpuTaskReservation,
        max_bytes: int,
    ) -> None:
        self.capture = capture
        self.reservation = reservation
        self.max_bytes = int(max_bytes)
        self._future: Any | None = None
        self._value: bytes | None = None
        self._submission_error: Exception | None = None

    def ready(self) -> bool:
        if self._value is not None or self._submission_error is not None:
            return True
        if self._future is None:
            if not self.capture.ready():
                return False
            try:
                self._future = self.reservation.submit(
                    uint8_image_to_png_base64_bytes,
                    self.capture.tensor(),
                )
            except Exception as error:
                self._submission_error = error
                return True
        return bool(self._future.done())

    def max_encoded_bytes(self) -> int:
        return self.max_bytes

    def finalize(self) -> bytes:
        if self._value is not None:
            return self._value
        if not self.ready():
            raise RuntimeError("image payload was observed before CPU encoding was ready")
        if self._submission_error is not None:
            raise self._submission_error
        if self._future is None:
            raise RuntimeError("image encoding task lost its CPU future")
        value = self._future.result(timeout=0)
        if not isinstance(value, bytes) or not value:
            raise RuntimeError("image encoding task produced an invalid payload")
        if len(value) > self.max_bytes:
            raise RuntimeError("encoded image exceeds its registered product byte bound")
        self._value = value
        return self._value

    def __bytes__(self) -> bytes:
        return self.finalize()

    def __del__(self) -> None:
        self.reservation.abandon()


class _PendingDigest:
    """A semantic digest finalized from a query-ready completion generation.

    The digest includes committed tokens copied asynchronously into the pinned
    completion arena. Resolution reads that host storage only after every copy
    event reports ready, validates the physical slot generation, and releases
    the observed row. A device-parent successor may retain its predecessor's
    pending digest, so resolution follows the request lineage while unrelated
    completions remain independently dispatchable.
    """

    __slots__ = (
        "_record",
        "_parent",
        "_plan_digest",
        "_lease",
        "_row",
        "_generation",
        "_completion_timing",
        "_value",
        "_observed",
        "_invalid_sampling",
        "_predicated",
        "_predicated_parent",
        "_selected_point",
        "_selected_runtime",
        "_resolved_callback",
        "_completion_tasks",
        "_completion_error",
    )

    def __init__(
        self,
        record: CompletionRecord,
        parent: object,
        plan_digest: str,
        lease: CompletionLease,
        row: int,
        predicated_parent: Callable[[], tuple[VersionRef, RequestRuntime]],
        resolved_callback: Callable[[CompletionRecord, str, str], None] | None = None,
        completion_tasks: tuple[_CompletionImagePayload | _CompletionLogprobPayload, ...] = (),
    ) -> None:
        self._record = record
        self._parent = parent
        self._plan_digest = plan_digest
        self._lease: CompletionLease | None = lease
        self._row = int(row)
        self._generation = int(record.completion_slot_generation)
        self._completion_timing: tuple[int, int, int, int] | None = None
        self._value: str | None = None
        self._observed = False
        self._invalid_sampling = False
        self._predicated = record.status is OpStatus.PREDICATED
        self._predicated_parent: Callable[[], tuple[VersionRef, RequestRuntime]] | None = (
            predicated_parent
        )
        self._selected_point = record.selected_point
        self._selected_runtime: RequestRuntime | None = None
        self._resolved_callback = resolved_callback
        self._completion_tasks = completion_tasks
        self._completion_error = False

    def ready(self) -> bool:
        if self._value is not None:
            return True
        if isinstance(self._parent, _PendingDigest) and not self._parent.ready():
            return False
        if self._lease is None or not self._lease.ready():
            return False
        for task in self._completion_tasks:
            if not task.ready():
                return False
        return True

    def resolve(self) -> str:
        if self._value is None:
            if not self.ready():
                raise RuntimeError("completion digest was resolved before query-ready")
            parent = (
                self._parent.resolve() if isinstance(self._parent, _PendingDigest) else self._parent
            )
            if self._predicated:
                self._resolve_predicated(cast(str, parent))
            else:
                try:
                    for task in self._completion_tasks:
                        task.finalize()
                except Exception:
                    self._completion_error = True
                    self._value = _completion_error_record(self._record).compute_semantic_digest(
                        parent_semantic=cast(str, parent),
                        plan_digest=self._plan_digest,
                    )
                else:
                    try:
                        # The digest packs each committed token via ``__index__``, which
                        # finalizes a deferred token exactly as ``int(value)`` would, so
                        # the record is hashed in place without a concrete-token copy.
                        digest = self._record.compute_semantic_digest(
                            parent_semantic=cast(str, parent),
                            plan_digest=self._plan_digest,
                        )
                        if self._resolved_callback is not None:
                            self._resolved_callback(self._record, digest, cast(str, parent))
                        self._value = digest
                    except _PredicatedOperation:
                        self._predicated = True
                        self._resolve_predicated(cast(str, parent))
                    except _InvalidSamplingDistribution:
                        self._invalid_sampling = True
                        self._value = _invalid_sampling_record(self._record).compute_semantic_digest(
                            parent_semantic=cast(str, parent),
                            plan_digest=self._plan_digest,
                        )
            lease = self._lease
            if lease is None:
                raise RuntimeError("completion digest lost its arena lease")
            lease.observe(self._row, self._generation)
            self._completion_timing = lease.timing()
            self._observed = True
            self._lease = None
        resolved = self._value
        if resolved is None:
            raise RuntimeError("completion digest resolved without a value")
        return resolved

    def _resolve_predicated(self, parent: str) -> None:
        self._value = parent
        predicated_parent = self._predicated_parent
        if predicated_parent is None:
            raise RuntimeError("predicated completion lost its parent resolver")
        selected, runtime = predicated_parent()
        point = selected.point
        if not isinstance(point, FixedPoint):
            raise RuntimeError("predicated operation selected a non-fixed parent")
        self._selected_point = int(point.point_index)
        self._selected_runtime = runtime

    def __str__(self) -> str:
        return self.resolve()

    def completion_timing(self) -> tuple[int, int, int, int]:
        self.resolve()
        return self._completion_timing or (0, 0, 0, 0)

    @property
    def invalid_sampling(self) -> bool:
        self.resolve()
        return self._invalid_sampling

    @property
    def predicated(self) -> bool:
        self.resolve()
        return self._predicated

    @property
    def completion_error(self) -> bool:
        self.resolve()
        return self._completion_error

    @property
    def selected_point(self) -> int:
        self.resolve()
        return int(self._selected_point)

    @property
    def selected_runtime(self) -> RequestRuntime:
        self.resolve()
        if self._selected_runtime is None:
            raise RuntimeError("predicated operation lost its selected runtime state")
        return self._selected_runtime

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _PendingDigest):
            return self.resolve() == other.resolve()
        if isinstance(other, str):
            return self.resolve() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.resolve())

    def __deepcopy__(self, memo: dict[int, object]) -> _PendingDigest:
        # A committed session snapshot shares ownership of the exact pinned
        # completion generation and its lineage digest.
        memo[id(self)] = self
        return self

    def __del__(self) -> None:
        lease = self._lease
        if lease is not None and not self._observed:
            lease.discard(self._row, self._generation)


class _PendingErrorDigest:
    """An error digest causally chained to an unobserved parent completion."""

    __slots__ = ("_parent", "_plan_digest", "_record", "_value")

    def __init__(
        self,
        parent: _PendingDigest | _PendingErrorDigest,
        record: CompletionRecord,
        plan_digest: str,
    ) -> None:
        self._parent = parent
        self._record = record
        self._plan_digest = plan_digest
        self._value: str | None = None

    def ready(self) -> bool:
        return self._value is not None or self._parent.ready()

    def resolve(self) -> str:
        if self._value is None:
            if not self.ready():
                raise RuntimeError("error digest was resolved before its parent was query-ready")
            self._value = self._record.compute_semantic_digest(
                self._parent.resolve(),
                self._plan_digest,
            )
        return self._value

    def __str__(self) -> str:
        return self.resolve()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (_PendingDigest, _PendingErrorDigest)):
            return self.resolve() == other.resolve()
        if isinstance(other, str):
            return self.resolve() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.resolve())

    def __deepcopy__(self, memo: dict[int, object]) -> _PendingErrorDigest:
        memo[id(self)] = self
        return self


def _record_ready(record: CompletionRecord) -> bool:
    """Whether a completion's deferred token copy and digest chain have landed."""

    digest = record.semantic_digest
    if isinstance(digest, _PendingDigest):
        return digest.ready()
    if isinstance(digest, _PendingErrorDigest):
        return digest.ready()
    for value in cast(tuple[object, ...], record.committed_tokens):
        if isinstance(value, _CompletionToken) and not value.ready():
            return False
    return True


def _finalized_record(record: CompletionRecord) -> CompletionRecord:
    digest = record.semantic_digest
    if isinstance(digest, _PendingErrorDigest):
        return replace(record, semantic_digest=digest.resolve())
    if isinstance(digest, _PendingDigest):
        resolved = digest.resolve()
        queued_us, device_us, copy_us, host_us = digest.completion_timing()
        timing = replace(
            record.timing_counters,
            queued_us=queued_us,
            device_us=device_us,
            copy_us=copy_us,
            host_us=host_us,
        )
        if digest.completion_error:
            return replace(
                _completion_error_record(record),
                semantic_digest=resolved,
                timing_counters=timing,
            )
        if digest.invalid_sampling:
            return replace(
                _invalid_sampling_record(record),
                semantic_digest=resolved,
                timing_counters=timing,
            )
        if digest.predicated:
            return replace(
                _predicated_record(record, digest.selected_point, digest.selected_runtime),
                semantic_digest=resolved,
                timing_counters=timing,
            )
    else:
        resolved = digest
        timing = record.timing_counters
    tokens = tuple(int(value) for value in record.committed_tokens)
    lengths = record.logical_lengths
    span = record.token_span
    return replace(
        record,
        selected_point=int(record.selected_point),
        logical_lengths=LogicalLengths(
            token_len=int(lengths.token_len),
            kv_visible_len=int(lengths.kv_visible_len),
            latent_len=int(lengths.latent_len),
            kv_reserved_len=int(lengths.kv_reserved_len),
            kv_initialized_len=int(lengths.kv_initialized_len),
            kv_committed_len=int(lengths.kv_committed_len),
            kv_published_len=int(lengths.kv_published_len),
        ),
        token_span=TokenSpan(base=int(span.base), len=int(span.len)),
        committed_tokens=tokens,
        semantic_digest=resolved,
        timing_counters=timing,
    )


def _invalid_sampling_record(record: CompletionRecord) -> CompletionRecord:
    return replace(
        record,
        status=OpStatus.ERROR,
        selected_point=max(0, int(record.selected_point) - 1),
        token_span=replace(record.token_span, len=0),
        committed_tokens=(),
        finish_flags=FinishFlags(),
        product_generations=(),
        error_code=ProtocolErrorCode.INVALID_OPERATION,
    )


def _completion_error_record(record: CompletionRecord) -> CompletionRecord:
    return replace(
        record,
        status=OpStatus.ERROR,
        token_span=replace(record.token_span, len=0),
        committed_tokens=(),
        finish_flags=FinishFlags(),
        product_generations=(),
        error_code=ProtocolErrorCode.COMPUTE_ERROR,
    )


def _predicated_record(
    record: CompletionRecord,
    selected_point: int,
    runtime: RequestRuntime,
) -> CompletionRecord:
    return replace(
        record,
        status=OpStatus.PREDICATED,
        selected_point=int(selected_point),
        logical_lengths=LogicalLengths(
            token_len=runtime.logical_position,
            kv_visible_len=runtime.kv_visible_len,
            latent_len=0,
            kv_reserved_len=runtime.kv_reserved_len,
            kv_initialized_len=runtime.kv_initialized_len,
            kv_committed_len=runtime.kv_committed_len,
            kv_published_len=runtime.kv_published_len,
        ),
        token_span=replace(record.token_span, len=0),
        committed_tokens=(),
        finish_flags=FinishFlags(),
        product_generations=(),
        error_code=None,
    )


def completion_report_ready(report: CompletionReport) -> bool:
    """True once every completion's deferred token/digest/artifact can be read
    without a stall."""

    for record in report.completions:
        if not _record_ready(record):
            return False
    for product in report.products:
        if not _completion_payload_ready(product.payload):
            return False
    return True


def partition_completion_ready(partition: PartitionCompletion) -> bool:
    for record in partition.completions:
        if not _record_ready(record):
            return False
    for product in partition.products:
        if not _completion_payload_ready(product.payload):
            return False
    return True


def _completion_payload_ready(payload: object) -> bool:
    return (
        not isinstance(
            payload,
            (_CompletionImagePayload, _CompletionLogprobPayload, _CompletionTransferPayload),
        )
        or payload.ready()
    )


def finalize_completion_report(report: CompletionReport) -> CompletionReport:
    """Materialize query-ready partitions into host-owned protocol values."""

    changed = False
    partitions: list[PartitionCompletion] = []
    for partition in report.partitions:
        nonpublishing_ops = {
            int(record.op_id)
            for record in partition.completions
            if record.status is not OpStatus.OK
        }
        retained_products = tuple(
            product
            for product in partition.products
            if int(product.product.producer_op_id) not in nonpublishing_ops
        )
        products = tuple(
            replace(product, payload=product.payload.finalize())
            if isinstance(
                product.payload,
                (
                    _CompletionImagePayload,
                    _CompletionLogprobPayload,
                    _CompletionTransferPayload,
                ),
            )
            and product.payload.ready()
            else product
            for product in retained_products
        )
        completions = tuple(
            _finalized_record(record) if _record_ready(record) else record
            for record in partition.completions
        )
        for product in products:
            if (
                isinstance(product.payload, bytes)
                and not product.payload.startswith(TRANSFER_DESCRIPTOR_PREFIX)
                and len(product.payload) > int(product.product.max_bytes)
            ):
                raise invalid_descriptor(
                    "completion product exceeds its registered product byte bound"
                )
        if (
            not all(new is old for new, old in zip(completions, partition.completions, strict=True))
            or len(products) != len(partition.products)
            or not all(new is old for new, old in zip(products, partition.products))
        ):
            changed = True
            partition = replace(partition, completions=completions, products=products)
        partitions.append(partition)
    return replace(report, partitions=tuple(partitions)) if changed else report
