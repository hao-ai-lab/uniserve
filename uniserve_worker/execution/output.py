"""Persistent output storage and concrete lane result materialization."""

from __future__ import annotations

import concurrent.futures
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Final, cast

import torch

from ..execution.batch import (
    ArResult,
    Checkpoint,
    CompletionState,
    DiffusionResult,
    EncoderResult,
    ErrorCode,
    FinishFlags,
    FixedCheckpoint,
    LaneResult,
    LogicalLengths,
    MediaOutput,
    ModelOutput,
    OpStatus,
    RequestKey,
    RunKind,
    RunResult,
    TimingCounters,
    TokenSpan,
    TransferHandle,
    TransferResult,
)
from ..foundation.errors import WorkerError, WorkerErrorCode, invalid_descriptor, resource_error
from ..media.codec import uint8_image_to_png_base64_bytes
from ..profiling import profile_range, timing_events_enabled
from ..runtime.cpu import CpuTaskReservation
from ..runtime.device import canonical_device
from ..runtime.device_events import DeviceEventPool
from ..runtime.request import RequestRuntime
from ..transfer.tickets import Locator, Transport, encode_transfer_handle

__all__ = [
    "CpuJob",
    "PendingOutput",
    "ImagePayload",
    "LogprobCapture",
    "LogprobOutputRow",
    "LogprobPayload",
    "SamplingCapture",
    "SamplingOutputRow",
    "TransferPayload",
    "ByteCapture",
    "OutputBuffer",
    "OutputPool",
    "TokenCapture",
    "run_result_ready",
    "finalize_run_result",
    "lane_completion_ready",
]

_SAMPLING_FIELDS_PER_OPERATION: Final[int] = 4
_next_buffer_generation = 1


def _invariant(message: str) -> WorkerError:
    """Construct a classified invariant error for output-buffer misuse."""

    return WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


@dataclass(frozen=True, slots=True)
class TokenCapture:
    """One token range copied into a lane's pinned output buffer."""

    buffer: OutputBuffer
    offset: int
    count: int

    def ready(self) -> bool:
        """Indicate whether the token copy event has completed."""

        return self.buffer.ready()

    def values(self) -> tuple[int, ...]:
        """Read this row's captured tokens after the owning buffer becomes ready."""

        return self.buffer.read_tokens(self)


@dataclass(frozen=True, slots=True)
class ByteCapture:
    """One shaped byte range copied into a lane's pinned output buffer."""

    buffer: OutputBuffer
    offset: int
    count: int
    shape: tuple[int, ...]
    external: torch.Tensor | None = None

    def ready(self) -> bool:
        """Indicate whether the byte-range copy event has completed."""

        return self.buffer.ready()

    def tensor(self) -> torch.Tensor:
        """Expose this capture's shaped CPU byte view after readiness."""

        return self.buffer.read_bytes(self)

    def numpy(self) -> Any:
        """Return the captured byte range as a shaped NumPy view after readiness."""

        if self.external is not None:
            if int(self.external.numel()) != int(self.count):
                raise _invariant("external completion byte storage has an invalid extent")
            return self.external.view(self.shape).numpy()
        return self.tensor().numpy()


class OutputBuffer:
    """Pinned host storage and completion events for one lane commit."""

    __slots__ = (
        "event_pool",
        "devices",
        "_rows",
        "_generation",
        "_host",
        "_token_cursor",
        "_byte_cursor",
        "_token_cache",
        "_start_events",
        "_producer_events",
        "_events",
        "_observed",
        "_sealed",
        "_abandoned",
        "_events_released",
        "_release_pending",
        "_reserved_ns",
        "_device_started_ns",
        "_copy_started_ns",
        "_sealed_ns",
        "_ready_ns",
        "_timing",
        "_timing_events",
        "_retained_until_ready",
        "_release_to_pool",
        "_released_to_pool",
    )

    def __init__(
        self,
        rows: int,
        *,
        token_capacity: int,
        devices: Sequence[torch.device | str] = (),
        event_pool: DeviceEventPool,
        release_to_pool: Callable[[OutputBuffer], None] | None = None,
    ) -> None:
        """Reserve pinned completion rows and generation-tagged CUDA copy state."""

        global _next_buffer_generation
        count = int(rows)
        capacity = int(token_capacity)
        if count < 1:
            raise ValueError("a pinned output buffer must contain at least one operation row")
        if capacity < count:
            raise resource_error("completion row count exceeds pinned output capacity")
        normalized: list[torch.device] = []
        for value in devices:
            device = canonical_device(value)
            if device.type == "cuda" and device not in normalized:
                normalized.append(device)
        self.event_pool = event_pool
        self.devices = tuple(normalized)
        self._rows = count
        self._generation = _next_buffer_generation
        _next_buffer_generation += 1
        self._host = torch.empty(
            capacity,
            dtype=torch.long,
            device="cpu",
            pin_memory=bool(self.devices),
        )
        self._token_cursor = 0
        self._byte_cursor = 0
        self._token_cache: dict[tuple[int, int], tuple[int, ...]] = {}
        self._start_events: dict[str, torch.cuda.Event] = {}
        self._producer_events: dict[str, torch.cuda.Event] = {}
        self._events: dict[str, torch.cuda.Event] = {}
        self._observed: set[int] = set()
        self._sealed = False
        self._abandoned = False
        self._events_released = False
        self._release_pending = False
        self._reserved_ns = time.perf_counter_ns()
        self._device_started_ns = 0
        self._copy_started_ns = 0
        self._sealed_ns = 0
        self._ready_ns = 0
        self._timing: tuple[int, int, int, int] | None = None
        self._timing_events = timing_events_enabled()
        self._retained_until_ready: list[object] = []
        self._release_to_pool = release_to_pool
        self._released_to_pool = False

    def reset(
        self,
        rows: int,
        *,
        token_capacity: int,
        devices: Sequence[torch.device | str],
    ) -> None:
        """Begin a new lease over this persistent pinned allocation."""

        global _next_buffer_generation
        if not self._events_released or self._release_pending:
            raise _invariant("output storage was leased before its prior events were released")
        count = int(rows)
        capacity = int(token_capacity)
        if count < 1 or capacity < count:
            raise resource_error("output lease geometry is invalid")
        normalized: list[torch.device] = []
        for value in devices:
            device = canonical_device(value)
            if device.type == "cuda" and device not in normalized:
                normalized.append(device)
        if capacity > int(self._host.numel()):
            self._host = torch.empty(
                capacity,
                dtype=torch.long,
                device="cpu",
                pin_memory=bool(normalized),
            )
        elif normalized and not bool(self._host.is_pinned()):
            self._host = torch.empty(
                int(self._host.numel()),
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
        self.devices = tuple(normalized)
        self._rows = count
        self._generation = _next_buffer_generation
        _next_buffer_generation += 1
        self._token_cursor = 0
        self._byte_cursor = 0
        self._token_cache.clear()
        self._start_events.clear()
        self._producer_events.clear()
        self._events.clear()
        self._observed.clear()
        self._sealed = False
        self._abandoned = False
        self._events_released = False
        self._release_pending = False
        self._reserved_ns = time.perf_counter_ns()
        self._device_started_ns = 0
        self._copy_started_ns = 0
        self._sealed_ns = 0
        self._ready_ns = 0
        self._timing = None
        self._timing_events = timing_events_enabled()
        self._retained_until_ready.clear()
        self._released_to_pool = False

    @property
    def generation(self) -> int:
        """Identify the lease generation guarding all captures from this buffer use."""

        return self._generation

    @property
    def row_count(self) -> int:
        """Count completion rows reserved by the active buffer lease."""

        return self._rows

    def register_device(self, device: torch.device | str) -> None:
        """Verify that a CUDA producer belongs to the devices declared for this lease."""

        if self._sealed:
            raise _invariant("completion device was registered after its buffer was sealed")
        target = canonical_device(device)
        if target.type == "cuda" and target not in self.devices:
            raise _invariant("completion work uses an undeclared CUDA device")

    def begin_device(self, device: torch.device | str) -> None:
        """Mark device execution start and record its optional timing event."""

        if self._sealed:
            raise _invariant("completion device timing began after its buffer was sealed")
        target = canonical_device(device)
        if self._device_started_ns == 0:
            self._device_started_ns = time.perf_counter_ns()
        if target.type != "cuda":
            return
        self.register_device(target)
        if not self._timing_events:
            return
        name = str(target)
        if name in self._start_events:
            return
        event = self.event_pool.acquire(target, timing=True)
        self.event_pool.retain(event, target)
        self.event_pool.record(event, target)
        self._start_events[name] = event

    def _mark_copy_started(self, device: torch.device) -> None:
        """Record the stream event that protects one device-to-host completion copy."""

        if self._copy_started_ns == 0:
            self._copy_started_ns = time.perf_counter_ns()
        name = str(device)
        if not self._timing_events:
            return
        if name in self._producer_events:
            return
        if name not in self._start_events:
            self.begin_device(device)
        event = self.event_pool.acquire(device, timing=True)
        self.event_pool.retain(event, device)
        self.event_pool.record(event, device)
        self._producer_events[name] = event

    def capture(self, tokens: torch.Tensor) -> TokenCapture:
        """Copy a token tensor into the next bounded span of pinned host storage."""

        if self._sealed:
            raise _invariant("completion capture was registered after its buffer was sealed")
        flat = tokens.reshape(-1).to(dtype=torch.long)
        count = int(flat.numel())
        offset = self._token_cursor
        end = offset + count
        if end * int(self._host.element_size()) > self._byte_floor():
            raise resource_error("completion token span exceeds pinned output capacity")
        host = self._host[offset:end]
        if flat.device.type == "cuda":
            if not bool(host.is_pinned()):
                raise _invariant("CUDA completion copy targets pageable host storage")
            device = canonical_device(flat.device)
            self.register_device(device)
            self._mark_copy_started(device)
            host.copy_(flat, non_blocking=True)
        else:
            host.copy_(flat.to(device="cpu"))
        self._token_cursor = end
        return TokenCapture(self, offset, count)

    def capture_bytes(self, value: torch.Tensor) -> ByteCapture:
        """Copy a contiguous uint8 tensor into the byte region growing from the buffer tail."""

        if value.dtype is not torch.uint8:
            raise ValueError("completion byte capture requires uint8 storage")
        if self._sealed:
            raise _invariant("completion byte capture was registered after its buffer was sealed")
        contiguous = value.detach().contiguous()
        count = int(contiguous.numel())
        if count < 1:
            raise ValueError("completion byte capture must not be empty")
        end = self._byte_floor()
        offset = end - count
        if offset < self._token_cursor * int(self._host.element_size()):
            raise resource_error("completion byte span exceeds pinned output capacity")
        host = self._host.view(torch.uint8)[offset:end]
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
        return ByteCapture(
            self,
            offset,
            count,
            tuple(int(value) for value in contiguous.shape),
        )

    def capture_bytes_into(
        self,
        value: torch.Tensor,
        storage: torch.Tensor,
    ) -> ByteCapture:
        """Copy bytes into caller-owned pinned storage under this buffer's events."""

        if value.dtype is not torch.uint8:
            raise ValueError("completion byte capture requires uint8 storage")
        if self._sealed:
            raise _invariant("completion byte capture was registered after its buffer was sealed")
        if storage.device.type != "cpu" or storage.dtype is not torch.uint8:
            raise ValueError("external completion storage must be a CPU uint8 tensor")
        contiguous = value.detach().contiguous()
        count = int(contiguous.numel())
        if count < 1:
            raise ValueError("completion byte capture must not be empty")
        if int(storage.numel()) < count:
            raise resource_error("external completion byte storage is too small")
        host = storage.reshape(-1)[:count]
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
        return ByteCapture(
            self,
            0,
            count,
            tuple(int(item) for item in contiguous.shape),
            host,
        )

    def _byte_floor(self) -> int:
        """Return the first byte offset not reserved for fixed completion words."""

        return int(self._host.numel()) * int(self._host.element_size()) - self._byte_cursor

    def seal(self) -> None:
        """Record completion events for every producer device and prohibit additional captures."""

        if self._sealed:
            return
        for device in self.devices:
            name = str(device)
            if self._timing_events:
                if name not in self._start_events:
                    self.begin_device(device)
                if name not in self._producer_events:
                    self._mark_copy_started(device)
            event = self.event_pool.acquire(device, timing=self._timing_events)
            self.event_pool.retain(event, device)
            self.event_pool.record(event, device)
            self._events[name] = event
            self.event_pool.schedule_completion_wake(device, event)
        self._sealed = True
        self._sealed_ns = time.perf_counter_ns()

    def ready(self) -> bool:
        """Return whether all sealed device-copy events have completed."""

        if not self._sealed:
            return False
        if self._ready_ns:
            return True
        if any(not bool(event.query()) for event in self._events.values()):
            return False
        self._ready_ns = time.perf_counter_ns()
        self._retained_until_ready.clear()
        return True

    def retain_until_ready(self, owner: object) -> None:
        """Keep an external resource alive until every registered copy completes."""

        if self.ready():
            return
        self._retained_until_ready.append(owner)

    def read_tokens(self, capture: TokenCapture) -> tuple[int, ...]:
        """Read and cache a validated token capture after its copy completes."""

        if capture.buffer is not self:
            raise _invariant("completion capture belongs to a different pinned output buffer")
        key = (int(capture.offset), int(capture.count))
        cached = self._token_cache.get(key)
        if cached is not None:
            return cached
        if not self.ready():
            raise _invariant("completion storage was observed before its copy event was ready")
        end = capture.offset + capture.count
        if capture.offset < 0 or end > self._token_cursor:
            raise _invariant("completion capture range is outside its registered token extent")
        values = tuple(int(value) for value in self._host[capture.offset : end].tolist())
        self._token_cache[key] = values
        return values

    def read_bytes(self, capture: ByteCapture) -> torch.Tensor:
        """Return a validated shaped view of captured bytes after readiness."""

        if capture.buffer is not self:
            raise _invariant("completion byte capture belongs to a different pinned output buffer")
        if not self.ready():
            raise _invariant("completion byte storage was observed before its copy event was ready")
        if capture.external is not None:
            if int(capture.external.numel()) != int(capture.count):
                raise _invariant("external completion byte storage has an invalid extent")
            return capture.external.view(capture.shape)
        end = int(capture.offset) + int(capture.count)
        total = int(self._host.numel()) * int(self._host.element_size())
        if capture.offset < 0 or end > total:
            raise _invariant("completion byte range is outside its registered extent")
        return self._host.view(torch.uint8)[capture.offset : end].view(capture.shape)

    def observe(self, row: int, generation: int) -> tuple[int, int]:
        """Mark one result row observed and return copy and host-observation timing."""

        index = int(row)
        if int(generation) != self._generation:
            raise _invariant("completion record carries a stale buffer generation")
        if index < 0 or index >= self._rows:
            raise _invariant("completion record row is outside its pinned output buffer")
        if not self.ready():
            raise _invariant("completion record was observed before query-ready")
        observed_ns = time.perf_counter_ns()
        if self._timing is None:
            queued_us = (
                max(0, self._device_started_ns - self._reserved_ns) // 1000
                if self._device_started_ns
                else 0
            )
            device_us = 0
            copy_us = 0
            if self._timing_events:
                for name, end_event in self._events.items():
                    start_event = self._start_events.get(name)
                    producer_event = self._producer_events.get(name)
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
            if not self._events and self._device_started_ns:
                copy_started_ns = self._copy_started_ns or self._sealed_ns
                device_us = max(0, copy_started_ns - self._device_started_ns) // 1000
                copy_us = max(0, self._sealed_ns - copy_started_ns) // 1000
            ready_to_observed_us = max(0, observed_ns - self._ready_ns) // 1000
            self._timing = (queued_us, device_us, copy_us, ready_to_observed_us)
        self._observed.add(index)
        if len(self._observed) == self._rows:
            self._release_events()
        return self._timing[2], self._timing[3]

    def timing(self) -> tuple[int, int, int, int]:
        """Expose submit, seal, ready, and observation timestamps for the completed lease."""

        if self._timing is None:
            raise _invariant("completion timing was read before observation")
        return self._timing

    def discard(self, row: int, generation: int) -> None:
        """Retire one unobserved result row while preserving unfinished device copies."""

        index = int(row)
        if int(generation) != self._generation or index < 0 or index >= self._rows:
            return
        self._observed.add(index)
        if len(self._observed) == self._rows:
            if self.ready():
                self._release_events()
            else:
                self._defer_release()

    def abandon(self) -> None:
        """Seal and retire the entire output lease without exposing its rows."""

        if self._abandoned:
            return
        if not self._sealed:
            self.seal()
        self._abandoned = True
        if self.ready():
            self._release_events()
        else:
            self._defer_release()

    def _all_events(self) -> tuple[torch.cuda.Event, ...]:
        """Collect distinct device-copy events currently owned by the buffer."""

        return tuple(
            (
                *self._start_events.values(),
                *self._producer_events.values(),
                *self._events.values(),
            )
        )

    def _release_events(self) -> None:
        """Return all owned device-copy events to the shared event pool."""

        if self._events_released or self._release_pending:
            return
        for event in self._all_events():
            self.event_pool.release(event)
        self._events_released = True
        self._return_to_pool()

    def _defer_release(self) -> None:
        """Defer buffer reuse until every outstanding device-copy event completes."""

        if self._events_released or self._release_pending:
            return
        events = self._all_events()
        if events:
            self._release_pending = True
            self.event_pool.defer_release(events, self)
        else:
            self._events_released = True
            self._return_to_pool()

    def events_released(self) -> None:
        """Receive completion of an event-pool asynchronous release."""

        self._release_pending = False
        self._events_released = True
        self._return_to_pool()

    def _return_to_pool(self) -> None:
        """Return the fully released completion buffer to its owning pool."""

        if self._released_to_pool or self._release_to_pool is None:
            return
        self._released_to_pool = True
        self._release_to_pool(self)


class OutputPool:
    """Bounded owner of reusable pinned lane-output allocations."""

    def __init__(
        self,
        *,
        capacity: int,
        max_words: int,
        event_pool: DeviceEventPool,
    ) -> None:
        """Allocate a bounded set of reusable pinned completion buffers."""

        self.capacity = int(capacity)
        self.max_words = int(max_words)
        if self.capacity < 1 or self.max_words < 1:
            raise ValueError("output-pool bounds must be positive")
        self.event_pool = event_pool
        self._buffers: list[OutputBuffer] = []
        self._free: list[OutputBuffer] = []
        self._lock = RLock()
        self._closed = False

    def acquire(
        self,
        rows: int,
        *,
        token_capacity: int,
        devices: Sequence[torch.device | str] = (),
    ) -> OutputBuffer:
        """Lease a reset or newly allocated output buffer within startup row and byte bounds."""

        words = int(token_capacity)
        if words > self.max_words:
            raise resource_error("lane output exceeds its startup storage bound")
        with self._lock:
            if self._closed:
                raise resource_error("output pool is closed")
            if self._free:
                buffer = self._free.pop()
                buffer.reset(rows, token_capacity=words, devices=devices)
                return buffer
            if len(self._buffers) >= self.capacity:
                raise resource_error("all lane output leases are active")
            buffer = OutputBuffer(
                rows,
                token_capacity=words,
                devices=devices,
                event_pool=self.event_pool,
                release_to_pool=self._release,
            )
            self._buffers.append(buffer)
            return buffer

    def _release(self, buffer: OutputBuffer) -> None:
        """Accept a released completion buffer back into the bounded free list."""

        with self._lock:
            if self._closed:
                return
            if buffer not in self._buffers or buffer in self._free:
                raise _invariant("output pool received an invalid lease return")
            self._free.append(buffer)

    def close(self) -> None:
        """Prevent new leases and release references to every pooled output buffer."""

        with self._lock:
            self._closed = True
            self._free.clear()
            self._buffers.clear()


class _InvalidSamplingDistribution(RuntimeError):
    """Marks a sampling row whose filtered probability mass is unusable."""

    pass


class _PredicatedOperation(RuntimeError):
    """Marks an operation suppressed by its resolved device predicate."""

    pass


class SamplingCapture:
    """Contiguous validity, activity, token, and acceptance metadata."""

    __slots__ = ("capture", "count", "_values")

    def __init__(
        self,
        capture: TokenCapture | None,
        count: int,
        values: tuple[int, ...] | None = None,
    ) -> None:
        """Bind packed sampling metadata to row layouts or predecoded values."""

        self.capture = capture
        self.count = int(count)
        self._values = values

    def ready(self) -> bool:
        """Indicate whether the packed sampling metadata can be decoded without blocking."""

        return self._values is not None or (self.capture is not None and self.capture.ready())

    def finalize(self) -> tuple[int, ...]:
        """Decode and cache validity, activity, token, and acceptance vectors from pinned output."""

        if self._values is None:
            if self.capture is None:
                raise RuntimeError("sampling completion metadata has no capture")
            values = self.capture.values()
            if len(values) != self.count * 4:
                raise RuntimeError("sampling completion metadata has an invalid extent")
            self._values = values
        return self._values

    def token(self, index: int) -> int:
        """Return the selected token after enforcing predicate and distribution validity."""

        values = self.finalize()
        if not bool(values[self.count + index]):
            raise _PredicatedOperation("operation predicate selected no state")
        if not bool(values[index]):
            raise _InvalidSamplingDistribution("sampling policy produced an invalid distribution")
        return values[self.count * 2 + index]

    def accepted(self, index: int) -> int:
        """Return the accepted speculative-prefix length for one valid active row."""

        values = self.finalize()
        if not bool(values[self.count + index]):
            raise _PredicatedOperation("operation predicate selected no state")
        if not bool(values[index]):
            raise _InvalidSamplingDistribution("sampling policy produced an invalid distribution")
        return values[self.count * 3 + index]


@dataclass(frozen=True, slots=True)
class SamplingOutputRow:
    """One operation's view of a contiguous sampling metadata capture."""

    capture: SamplingCapture
    index: int
    draft_tokens: tuple[int, ...] = ()
    terminal_prefix: int | None = None
    logical_base: int | None = None
    kv_base: int | None = None

    def ready(self) -> bool:
        """Indicate whether this row's speculative sampling capture is host-visible."""

        return self.capture.ready()

    def materialize(self) -> tuple[tuple[int, ...], int, int]:
        """Combine accepted draft tokens with the sampled continuation and return its committed extent."""

        token = self.capture.token(int(self.index))
        accepted = self.capture.accepted(int(self.index))
        if not self.draft_tokens:
            return (token,), 1, accepted
        if accepted < 0 or accepted > len(self.draft_tokens):
            raise RuntimeError("speculative acceptance count is outside the draft span")
        if self.terminal_prefix is not None and accepted >= self.terminal_prefix:
            tokens = self.draft_tokens[:accepted]
        else:
            tokens = (*self.draft_tokens[:accepted], token)
        return tokens, len(tokens), accepted


class LogprobCapture:
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
        capture: TokenCapture | None,
        rows: tuple[int, ...],
        counts: tuple[int, ...],
        requested_ids: tuple[tuple[int, ...], ...],
        max_count: int,
        max_requested: int,
        values: tuple[int, ...] | None = None,
    ) -> None:
        """Bind packed log-probability metadata to its requested row schemas."""

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
        """Indicate whether all selected and requested log-probability entries are host-visible."""

        return self._details is not None or (self.capture is not None and self.capture.ready())

    @staticmethod
    def _float(value: int) -> float:
        """Decode a float32 value from its unsigned integer bit pattern."""

        return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]

    def finalize(self) -> dict[int, tuple[float, tuple[tuple[int, float, int], ...]]]:
        """Decode and cache selected, top-k, and explicitly requested token log probabilities."""

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
        """Decode flattened completion fields into token, chosen, and requested log probabilities."""

        row_count = len(self.rows)
        cursor = 0

        def vector(width: int) -> tuple[tuple[int, ...], ...]:
            """Consume one row-major field of fixed width from the packed capture."""

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


@dataclass(frozen=True, slots=True)
class LogprobOutputRow:
    """One operation or prompt position in a packed logprob capture."""

    capture: LogprobCapture
    index: int

    def ready(self) -> bool:
        """Indicate whether this row's shared log-probability capture is host-visible."""

        return self.capture.ready()

    def finalize(self) -> tuple[float, tuple[tuple[int, float, int], ...]]:
        """Select this row's decoded log-probability record from the shared capture."""

        return self.capture.finalize()[int(self.index)]

    def max_entries(self) -> int:
        """Return the maximum unique log-probability entries this row can encode."""

        local = self.capture.rows.index(int(self.index))
        return (
            1
            + int(self.capture.counts[local])
            + len(self.capture.requested_ids[local])
        )


class CpuJob:
    """A bounded CPU action gated by dependencies and optional CUDA readiness."""

    __slots__ = (
        "capture",
        "reservation",
        "dependencies",
        "action",
        "_future",
        "promise",
        "_submission_error",
        "profile_name",
        "ready_event",
        "_release",
        "_defer_release",
        "_resource_released",
    )

    def __init__(
        self,
        reservation: CpuTaskReservation,
        action: Callable[[], object],
        *,
        capture: ByteCapture | None = None,
        dependencies: tuple[concurrent.futures.Future[object], ...] = (),
        profile_name: str,
        release: Callable[[], None] | None = None,
        defer_release: Callable[[ByteCapture], None] | None = None,
    ) -> None:
        """Retain a bounded CPU reservation and lazily submitted host operation."""

        self.capture = capture
        self.reservation = reservation
        self.dependencies = dependencies
        self.action = action
        self._future: concurrent.futures.Future[object] | None = None
        self.promise: concurrent.futures.Future[object] = concurrent.futures.Future()
        self._submission_error: BaseException | None = None
        self.profile_name = profile_name
        self.ready_event: torch.cuda.Event | None = None
        self._release = release
        self._defer_release = defer_release
        self._resource_released = False

    def _release_now(self) -> None:
        """Release the CPU task reservation unless submission already transferred ownership."""

        if self._resource_released or self._release is None:
            return
        self._resource_released = True
        self._release()

    def _release_after_capture(self) -> None:
        """Release the task reservation after captured output ownership is established."""

        if self._resource_released or self._release is None:
            return
        self._resource_released = True
        if self.capture is not None and self._defer_release is not None:
            self._defer_release(self.capture)
        else:
            self._release()

    def _run(self) -> object:
        """Execute one host job and capture its value or exception exactly once."""

        try:
            if self.ready_event is not None:
                self.ready_event.synchronize()
            for dependency in self.dependencies:
                dependency.result()
            with profile_range(self.profile_name):
                value = self.action()
        except BaseException as error:
            self.promise.set_exception(error)
            raise
        else:
            self.promise.set_result(value)
        finally:
            self._release_now()
        return value

    def start(self, ready_event: torch.cuda.Event | None = None) -> None:
        """Submit the bounded host action, optionally gated by a CUDA readiness event."""

        if self._submission_error is not None:
            return
        if self._future is not None:
            raise RuntimeError("output CPU job was submitted more than once")
        self.ready_event = ready_event
        try:
            self._future = self.reservation.submit(self._run)
        except BaseException as error:
            self._submission_error = error
            self._release_now()
            if not self.promise.done():
                self.promise.set_exception(error)

    def ready(self) -> bool:
        """Start eligible work lazily and report whether the host action has completed."""

        if self._submission_error is not None:
            return True
        if self._future is None:
            if self.capture is not None and not self.capture.ready():
                return False
            self.start()
        return self._future is None or bool(self._future.done())

    def finalize(self) -> object:
        """Return the completed host result or raise its captured failure without blocking."""

        if not self.ready():
            raise RuntimeError("output CPU job was observed before it was ready")
        if self._submission_error is not None:
            raise self._submission_error
        if self._future is None:
            raise RuntimeError("output CPU job lost its submitted future")
        return self._future.result(timeout=0)

    def __del__(self) -> None:
        """Release an unsubmitted CPU reservation during finalization."""

        self.reservation.abandon()
        if self.capture is None or self.capture.ready():
            self._release_now()
        else:
            self._release_after_capture()


class LogprobPayload:
    """Owns asynchronously copied log-probability entries until wire serialization."""

    __slots__ = ("selected", "prompt", "_value")

    def __init__(
        self,
        selected: LogprobOutputRow | None,
        prompt: tuple[LogprobOutputRow, ...] = (),
    ) -> None:
        """Collect selected and prompt log-probability rows for bounded serialization."""

        self.selected = selected
        self.prompt = prompt
        self._value: TransferHandle | None = None

    def ready(self) -> bool:
        """Return whether every selected and prompt log-probability capture is query-ready."""

        if self._value is not None:
            return True
        return (self.selected is None or self.selected.ready()) and all(
            position.ready() for position in self.prompt
        )

    def max_encoded_bytes(self) -> int:
        """Calculate the exact upper bound for the binary log-probability payload."""

        return (
            (5 if self.selected is not None else 1)
            + 4
            + 12 * (0 if self.selected is None else self.selected.max_entries())
            + 4
            + sum(4 + 12 * position.max_entries() for position in self.prompt)
        )

    def finalize(self) -> bytes:
        """Serialize selected and prompt log probabilities into the bounded binary wire format."""

        if self._value is not None:
            return self._value
        if not self.ready():
            raise RuntimeError("logprob payload was observed before query-ready")
        selected = None if self.selected is None else self.selected.finalize()
        logprob = None if selected is None else selected[0]
        top = () if selected is None else selected[1]
        out = bytearray(b"\x00" if logprob is None else b"\x01" + struct.pack("<f", logprob))
        out += struct.pack("<I", len(top))
        for token_id, value, rank in top:
            out += struct.pack("<IfI", int(token_id), float(value), int(rank))
        out += struct.pack("<I", len(self.prompt))
        for position in self.prompt:
            entries = position.finalize()[1]
            out += struct.pack("<I", len(entries))
            for token_id, value, rank in entries:
                out += struct.pack("<IfI", int(token_id), float(value), int(rank))
        self._value = bytes(out)
        return self._value

    def __bytes__(self) -> bytes:
        """Serialize captured log-probability records to their binary payload."""

        return self.finalize()


class TransferPayload:
    """Owns an asynchronous transfer ticket until its encoded handle is ready."""

    __slots__ = (
        "kind",
        "descriptor_value",
        "locators",
        "transport",
        "_value",
    )

    def __init__(
        self,
        kind: str,
        descriptor_value: dict[str, object],
        locators: tuple[Locator, ...],
        transport: Transport,
    ) -> None:
        """Retain transport locators until every producer becomes externally readable."""

        self.kind = kind
        self.descriptor_value = descriptor_value
        self.locators = locators
        self.transport = transport
        self._value: TransferHandle | None = None

    def ready(self) -> bool:
        """Indicate whether every asynchronous transfer descriptor is available."""

        return self._value is not None or all(
            self.transport.ready(locator) for locator in self.locators
        )

    def max_encoded_bytes(self) -> int:
        """Bound the encoded transport handle using its kind and descriptor schema."""

        return encode_transfer_handle(self.kind, self.descriptor_value).encoded_size_bound()

    def finalize(self) -> TransferHandle:
        """Return the encoded transfer handle after its asynchronous ticket completes."""

        if self._value is None:
            if not self.ready():
                raise RuntimeError("transport descriptor was observed before producer readiness")
            self._value = encode_transfer_handle(
                self.kind,
                self.descriptor_value,
            )
        return self._value


class ImagePayload:
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
        capture: ByteCapture,
        reservation: CpuTaskReservation,
        max_bytes: int,
    ) -> None:
        """Own a captured image tensor and deferred bounded host encoding job."""

        self.capture = capture
        self.reservation = reservation
        self.max_bytes = int(max_bytes)
        self._future: Any | None = None
        self._value: bytes | None = None
        self._submission_error: Exception | None = None

    def ready(self) -> bool:
        """Return whether image bytes are encoded or all deferred encoding work is complete."""

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
        """Expose the byte capacity reserved for the encoded image payload."""

        return self.max_bytes

    def finalize(self) -> bytes:
        """Return encoded image bytes, materializing the deferred host result when necessary."""

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
        """Return the encoded image artifact bytes."""

        return self.finalize()

    def __del__(self) -> None:
        """Release an unconsumed encoded image payload during finalization."""

        self.reservation.abandon()


@dataclass(frozen=True, slots=True)
class OutputRecord:
    """One unresolved output row retained outside the public wire model."""

    request_key: RequestKey
    op_id: int
    kind: RunKind
    completion_slot_generation: int
    status: OpStatus
    selected_point: int
    logical_lengths: LogicalLengths
    token_span: TokenSpan
    committed_tokens: tuple[int, ...]
    sampling: SamplingOutputRow | None
    finish_flags: FinishFlags
    product_generations: tuple[int, ...]
    error_code: ErrorCode | None
    next_cursor: int = 0
    done: bool = False


class PendingOutput:
    """A query-ready completion backed by one pinned output-buffer row."""

    __slots__ = (
        "_record",
        "_parent",
        "_buffer",
        "_row",
        "_generation",
        "_completion_timing",
        "_done",
        "_observed",
        "_invalid_sampling",
        "_predicated",
        "_predicated_parent",
        "_selected_point",
        "_selected_runtime",
        "_resolved_callback",
        "_completion_tasks",
        "_completion_error",
        "_media_output",
        "_value",
    )

    def __init__(
        self,
        parent: CompletionState | None,
        buffer: OutputBuffer,
        row: int,
        predicated_parent: Callable[[], tuple[Checkpoint, RequestRuntime]],
        *,
        status: OpStatus,
        selected_point: int,
        resolved_callback: Callable[[ModelOutput], None] | None = None,
        completion_tasks: tuple[CpuJob | ImagePayload | LogprobPayload, ...] = (),
    ) -> None:
        """Bind deferred device, CPU, transport, and media work to one completion record."""

        self._record: OutputRecord | None = None
        self._parent = parent
        self._buffer: OutputBuffer | None = buffer
        self._row = int(row)
        self._generation = int(buffer.generation)
        self._completion_timing: tuple[int, int, int, int] | None = None
        self._done = False
        self._observed = False
        self._invalid_sampling = False
        self._predicated = status is OpStatus.PREDICATED
        self._predicated_parent: Callable[[], tuple[Checkpoint, RequestRuntime]] | None = (
            predicated_parent
        )
        self._selected_point = int(selected_point)
        self._selected_runtime: RequestRuntime | None = None
        self._resolved_callback = resolved_callback
        self._completion_tasks = completion_tasks
        self._completion_error = False
        self._media_output: MediaOutput | None = None
        self._value: ModelOutput | None = None

    def bind_record(self, record: OutputRecord) -> PendingOutput:
        """Bind the one record backed by this pending output row."""

        if self._record is not None:
            raise RuntimeError("completion record was bound more than once")
        if int(record.completion_slot_generation) != self._generation:
            raise RuntimeError("completion record generation does not match its output buffer")
        if (record.status is OpStatus.PREDICATED) != self._predicated:
            raise RuntimeError("completion record status changed during binding")
        if int(record.selected_point) != int(self._selected_point):
            raise RuntimeError("completion selected point changed during binding")
        self._record = record
        return self

    @property
    def request_key(self) -> object:
        """Identify the request generation that owns the bound completion record."""

        if self._record is None:
            raise RuntimeError("completion has no bound record")
        return self._record.request_key

    @property
    def op_id(self) -> int:
        """Identify the operation within the bound request generation."""

        if self._record is None:
            raise RuntimeError("completion has no bound record")
        return self._record.op_id

    @property
    def status(self) -> OpStatus:
        """Expose the status staged by device execution before final host materialization."""

        if self._record is None:
            raise RuntimeError("completion has no bound record")
        return self._record.status

    def ready(self) -> bool:
        """Return whether sampling, transfers, media, CPU work, and completion copies are all ready."""

        if self._record is None:
            raise RuntimeError("completion has no bound record")
        if self._done:
            return True
        if self._parent is not None and not self._parent.ready():
            return False
        if self._buffer is None or not self._buffer.ready():
            return False
        for task in self._completion_tasks:
            if not task.ready():
                return False
        return True

    def finalize(self) -> ModelOutput:
        """Materialize one operation result, publish payload handles, and attach measured timing."""

        if not self._done:
            if not self.ready():
                raise RuntimeError("completion was resolved before query-ready")
            record = self._record
            if record is None:
                raise RuntimeError("completion has no bound record")
            parent = self._parent
            if parent is not None:
                parent.finalize()
                self._parent = None
            if self._predicated:
                self._resolve_predicated()
            else:
                try:
                    for task in self._completion_tasks:
                        result = task.finalize()
                        if isinstance(result, MediaOutput):
                            if self._media_output is not None:
                                raise RuntimeError("completion produced more than one media output")
                            self._media_output = result
                except Exception:
                    self._completion_error = True
                else:
                    try:
                        concrete = _concrete_record(record)
                        if self._resolved_callback is not None:
                            self._resolved_callback(concrete)
                    except _PredicatedOperation:
                        self._predicated = True
                        self._resolve_predicated()
                    except _InvalidSamplingDistribution:
                        self._invalid_sampling = True
            buffer = self._buffer
            if buffer is None:
                raise RuntimeError("completion lost its pinned output buffer")
            buffer.observe(self._row, self._generation)
            self._completion_timing = buffer.timing()
            self._observed = True
            self._buffer = None
            self._done = True
            timing = TimingCounters(
                queued_us=self._completion_timing[0],
                device_us=self._completion_timing[1],
                copy_us=self._completion_timing[2],
                host_us=self._completion_timing[3],
            )
            materialized_record = (
                replace(record, committed_tokens=(), sampling=None)
                if self._completion_error or self._invalid_sampling or self._predicated
                else record
            )
            concrete = _concrete_record(
                materialized_record,
                timing=timing,
                media_output=self._media_output,
            )
            if self._completion_error:
                concrete = _completion_error_record(concrete)
            elif self._invalid_sampling:
                concrete = _invalid_sampling_record(concrete)
            elif self._predicated:
                selected_runtime = self._selected_runtime
                if selected_runtime is None:
                    raise RuntimeError("predicated operation lost its selected runtime state")
                concrete = _predicated_record(
                    concrete,
                    self._selected_point,
                    selected_runtime,
                )
            concrete.validate()
            self._value = concrete
        if self._value is None:
            raise RuntimeError("completion output was not materialized")
        return self._value

    def _resolve_predicated(self) -> None:
        """Resolve inactive output rows without waiting for model or CPU work."""

        predicated_parent = self._predicated_parent
        if predicated_parent is None:
            raise RuntimeError("predicated completion lost its parent resolver")
        selected, runtime = predicated_parent()
        point = selected.point
        if not isinstance(point, FixedCheckpoint):
            raise RuntimeError("predicated operation selected a non-fixed parent")
        self._selected_point = int(point.point_index)
        self._selected_runtime = runtime

    def completion_timing(self) -> tuple[int, int, int, int]:
        """Finalize the record and expose its output-buffer lifecycle timestamps."""

        self.finalize()
        return self._completion_timing or (0, 0, 0, 0)

    @property
    def media_output(self) -> MediaOutput | None:
        """Finalize the record and expose its published media artifact, if any."""

        self.finalize()
        return self._media_output

    @property
    def invalid_sampling(self) -> bool:
        """Indicate whether token selection produced no valid finite candidate."""

        self.finalize()
        return self._invalid_sampling

    @property
    def predicated(self) -> bool:
        """Indicate whether device predicate resolution suppressed this operation."""

        self.finalize()
        return self._predicated

    @property
    def completion_error(self) -> bool:
        """Indicate whether deferred CPU or media completion failed."""

        self.finalize()
        return self._completion_error

    @property
    def selected_point(self) -> int:
        """Expose the checkpoint point selected after predicate and sampling resolution."""

        self.finalize()
        return int(self._selected_point)

    @property
    def selected_runtime(self) -> RequestRuntime:
        """Expose the request state selected by a predicated operation."""

        self.finalize()
        if self._selected_runtime is None:
            raise RuntimeError("predicated operation lost its selected runtime state")
        return self._selected_runtime

    def __deepcopy__(self, memo: dict[int, object]) -> PendingOutput:
        """Preserve identity because this object uniquely owns asynchronous completion state."""

        memo[id(self)] = self
        return self

    def __del__(self) -> None:
        """Abandon unresolved completion ownership during finalization."""

        buffer = self._buffer
        if buffer is not None and not self._observed:
            buffer.discard(self._row, self._generation)


def _record_ready(record: ModelOutput | PendingOutput) -> bool:
    """Return whether a completion's device and CPU output work has landed."""

    return record.ready() if isinstance(record, PendingOutput) else True


def _finalized_record(record: ModelOutput | PendingOutput) -> ModelOutput:
    """Require and return a concrete model-output record."""

    return record.finalize() if isinstance(record, PendingOutput) else record


def _concrete_record(
    record: OutputRecord,
    *,
    timing: TimingCounters = TimingCounters(),
    media_output: MediaOutput | None = None,
) -> ModelOutput:
    """Freeze a host-visible output record after resolving deferred sampling fields."""

    lengths = record.logical_lengths
    span = record.token_span
    selected_point = int(record.selected_point)
    tokens = record.committed_tokens
    sampling = record.sampling
    if sampling is not None:
        # Sampling decides both the accepted prefix and the selected checkpoint;
        # logical token/cache lengths advance only by that accepted prefix.
        tokens, selected_point, _accepted = sampling.materialize()
        if sampling.logical_base is not None:
            lengths = replace(
                lengths,
                token_len=int(sampling.logical_base) + selected_point,
            )
        if sampling.kv_base is not None:
            lengths = replace(
                lengths,
                kv_visible_len=int(sampling.kv_base) + selected_point,
            )
        span = replace(span, len=selected_point)
    if type(lengths.token_len) is not int or any(
        type(value) is not int
        for value in (
            lengths.kv_visible_len,
            lengths.kv_computed_len,
            lengths.latent_len,
        )
    ):
        # Wire records contain builtin integers even when counters originated as
        # scalar tensors or NumPy-compatible integer values.
        lengths = LogicalLengths(
            token_len=int(lengths.token_len),
            kv_visible_len=int(lengths.kv_visible_len),
            kv_computed_len=int(lengths.kv_computed_len),
            latent_len=int(lengths.latent_len),
        )
    if type(span.base) is not int or type(span.len) is not int:
        span = TokenSpan(base=int(span.base), len=int(span.len))
    payload_type = (
        ArResult
        if record.kind in {RunKind.AR_EXTEND, RunKind.AR_DECODE, RunKind.AR_VERIFY}
        else EncoderResult
        if record.kind in {RunKind.ENCODER_VISION, RunKind.ENCODER_LATENT}
        else DiffusionResult
        if record.kind in {
            RunKind.DIFFUSION_PREPARE,
            RunKind.DIFFUSION_STEP,
            RunKind.DIFFUSION_DECODE,
            RunKind.DIFFUSION_FINALIZE,
        }
        else TransferResult
    )

    # Select the closed result schema from the operation family. Diffusion is
    # the only family with trajectory cursor and terminal-state fields.
    payload_args = (lengths, span, tokens, record.finish_flags, media_output)
    payload = (
        DiffusionResult(
            *payload_args,
            next_cursor=int(record.next_cursor),
            done=bool(record.done),
        )
        if payload_type is DiffusionResult
        else payload_type(*payload_args)
    )
    return ModelOutput(
        request_key=record.request_key,
        op_id=record.op_id,
        completion_slot_generation=record.completion_slot_generation,
        status=record.status,
        selected_point=selected_point,
        product_generations=record.product_generations,
        error_code=record.error_code,
        timing_counters=timing,
        payload=payload,
    )


def _invalid_sampling_record(record: ModelOutput) -> ModelOutput:
    """Return an output record representing a sampling-policy rejection."""

    return replace(
        record,
        status=OpStatus.ERROR,
        selected_point=max(0, int(record.selected_point) - 1),
        payload=replace(
            record.payload,
            token_span=replace(record.token_span, len=0),
            committed_tokens=(),
            finish_flags=FinishFlags(),
        ),
        product_generations=(),
        error_code=ErrorCode.INVALID_OPERATION,
    )


def _completion_error_record(record: ModelOutput) -> ModelOutput:
    """Return an output record for a failed completion capture or host artifact."""

    return replace(
        record,
        status=OpStatus.ERROR,
        payload=replace(
            record.payload,
            token_span=replace(record.token_span, len=0),
            committed_tokens=(),
            finish_flags=FinishFlags(),
        ),
        product_generations=(),
        error_code=ErrorCode.COMPUTE_ERROR,
    )


def _predicated_record(
    record: ModelOutput,
    selected_point: int,
    runtime: RequestRuntime,
) -> ModelOutput:
    """Apply a resolved speculative point to one output record and request runtime."""

    return replace(
        record,
        status=OpStatus.PREDICATED,
        selected_point=int(selected_point),
        payload=replace(
            record.payload,
            logical_lengths=LogicalLengths(
                token_len=runtime.logical_position,
                kv_visible_len=runtime.kv_visible_len,
                kv_computed_len=runtime.kv_computed_len,
                latent_len=0,
            ),
            token_span=replace(record.token_span, len=0),
            committed_tokens=(),
            finish_flags=FinishFlags(),
        ),
        product_generations=(),
        error_code=None,
    )


def run_result_ready(report: RunResult) -> bool:
    """Return whether every completion token and artifact can be read
    without a stall."""

    for record in report.completions:
        if not _record_ready(record):
            return False
    for product in report.products:
        if not _completion_payload_ready(product.payload):
            return False
    return True


def lane_completion_ready(lane: LaneResult) -> bool:
    """Return whether every pending output in a lane can be finalized without blocking."""

    for record in lane.completions:
        if not _record_ready(record):
            return False
    for product in lane.products:
        if not _completion_payload_ready(product.payload):
            return False
    return True


def _completion_payload_ready(payload: object) -> bool:
    """Return whether a completion payload's asynchronous work is readable."""

    return (
        not isinstance(
            payload,
            (ImagePayload, LogprobPayload, TransferPayload),
        )
        or payload.ready()
    )


def finalize_run_result(report: RunResult) -> RunResult:
    """Materialize query-ready lanes into host-owned values."""

    changed = False
    lanes: list[LaneResult] = []
    for lane in report.lanes:
        completions = tuple(
            _finalized_record(record) if _record_ready(record) else record
            for record in lane.completions
        )
        nonpublishing_ops = {
            int(record.op_id)
            for record in completions
            if record.status is not OpStatus.OK
        }
        retained_products = tuple(
            product
            for product in lane.products
            if int(product.product.producer_op_id) not in nonpublishing_ops
        )
        products = tuple(
            replace(product, payload=product.payload.finalize())
            if isinstance(
                product.payload,
                (
                    ImagePayload,
                    LogprobPayload,
                    TransferPayload,
                ),
            )
            and product.payload.ready()
            else product
            for product in retained_products
        )
        for product in products:
            if (
                isinstance(product.payload, bytes)
                and len(product.payload) > int(product.product.max_bytes)
            ):
                raise invalid_descriptor(
                    "completion product exceeds its registered product byte bound"
                )
        publication = lane.publication
        publication_ready = (
            publication is not None
            and all(isinstance(record, ModelOutput) for record in completions)
            and all(
                not isinstance(
                    product.payload,
                    (ImagePayload, LogprobPayload, TransferPayload),
                )
                for product in products
            )
        )
        if publication_ready:
            publication.finish(cast(tuple[ModelOutput, ...], completions))
        if (
            not all(new is old for new, old in zip(completions, lane.completions, strict=True))
            or len(products) != len(lane.products)
            or not all(new is old for new, old in zip(products, lane.products))
            or publication_ready
        ):
            changed = True
            lane = replace(
                lane,
                completions=completions,
                products=products,
                publication=None if publication_ready else publication,
            )
        lanes.append(lane)
    return replace(report, lanes=tuple(lanes)) if changed else report
