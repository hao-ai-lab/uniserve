"""Pinned completion staging, immutable step outputs, and retained outcomes."""

from __future__ import annotations

import struct
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, cast, overload

import torch

from ..batch import (
    CompletionReport,
    DeferredSemanticDigest,
    ErrorCode,
    FinishFlags,
    FixedPoint,
    LogicalLengths,
    ModelOutput,
    OpStatus,
    PartitionCompletion,
    TokenSpan,
    VersionRef,
)
from ..foundation.errors import WorkerError, WorkerErrorCode, invalid_descriptor, resource_error
from ..runtime.device import canonical_device
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

if TYPE_CHECKING:
    from .app import InflightStep, TerminalStep

__all__ = [
    "DeferredDerivedInteger",
    "DeferredCompletionTask",
    "DeferredDigest",
    "DeferredErrorDigest",
    "DeferredImagePayload",
    "DeferredInteger",
    "DeferredLogprobBatch",
    "DeferredLogprobPayload",
    "DeferredLogprobValue",
    "DeferredSampleSpan",
    "DeferredSampleToken",
    "DeferredSpeculativePoint",
    "DeferredSpeculativeTokens",
    "DeferredToken",
    "DeferredTokenSpan",
    "DeferredTopLogprobs",
    "DeferredTransferPayload",
    "CompletedStepCache",
    "PinnedByteCapture",
    "PinnedOutputBuffer",
    "PinnedTokenCapture",
    "StepOutputs",
    "completion_report_ready",
    "finalize_completion_report",
    "partition_completion_ready",
]

_SAMPLING_FIELDS_PER_OPERATION: Final[int] = 4
_next_buffer_generation = 1


def _invariant(message: str) -> WorkerError:
    return WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


class CompletedStepCache:
    """Weighted LRU ownership for quiescent terminal step records."""

    def __init__(self, capacity: int) -> None:
        value = int(capacity)
        if value < 1:
            raise ValueError("completed step cache capacity must be positive")
        self.capacity = value
        self._steps: OrderedDict[int, InflightStep | TerminalStep] = OrderedDict()
        self._weight = 0

    def __contains__(self, step_id: object) -> bool:
        return step_id in self._steps

    def take(self, step_id: int) -> InflightStep | TerminalStep | None:
        step = self._steps.pop(int(step_id), None)
        if step is not None:
            self._weight -= int(step.weight)
        return step

    def touch(self, step_id: int) -> None:
        key = int(step_id)
        if key in self._steps:
            self._steps.move_to_end(key)

    def put(
        self,
        step: InflightStep | TerminalStep,
    ) -> tuple[InflightStep | TerminalStep, ...]:
        key = int(step.step_id)
        existing = self._steps.pop(key, None)
        if existing is not None:
            self._weight -= int(existing.weight)
        self._steps[key] = step
        self._weight += int(step.weight)
        evicted: list[InflightStep | TerminalStep] = []
        while self._weight > self.capacity:
            _key, victim = self._steps.popitem(last=False)
            self._weight -= int(victim.weight)
            evicted.append(victim)
        return tuple(evicted)

    def remove(self, step_id: int) -> InflightStep | TerminalStep | None:
        return self.take(step_id)

    def values(self) -> tuple[InflightStep | TerminalStep, ...]:
        return tuple(self._steps.values())


class StepOutputs:
    """One independent response cursor over an execution step's partition order."""

    __slots__ = (
        "_step",
        "_sent_partitions",
        "_empty_sent",
        "_error_sent",
        "_closed",
        "_on_close",
    )

    def __init__(
        self,
        step: InflightStep | TerminalStep,
        on_close: Callable[[StepOutputs], None],
    ) -> None:
        self._step = step
        self._sent_partitions: set[int] = set()
        self._empty_sent = False
        self._error_sent = False
        self._closed = False
        self._on_close = on_close

    @property
    def step_id(self) -> int:
        return int(self._current().step_id)

    @property
    def session_ids(self) -> frozenset[int]:
        return frozenset(int(value) for value in self._current().session_ids)

    @property
    def source(self) -> object | None:
        return self._current().source

    @property
    def error(self) -> WorkerError | None:
        return self._current().error

    @property
    def complete(self) -> bool:
        return bool(self._current().complete)

    def _current(self) -> InflightStep | TerminalStep:
        current = self._step.current()
        if current is not self._step:
            self._step = current
        return current

    def ready(self) -> bool:
        current = self._current()
        current.advance_materialization()
        current = self._current()
        if current.error is not None:
            return not self._error_sent
        partitions = current.materialized_partitions()
        if any(
            int(partition.partition_id) not in self._sent_partitions for partition in partitions
        ):
            return True
        return bool(current.complete and not current.partition_order and not self._empty_sent)

    def execution_complete(self) -> bool:
        current = self._current()
        complete = bool(current.advance_execution())
        self._current()
        return complete

    def take_ready(self) -> CompletionReport:
        current = self._current()
        current.advance_materialization()
        current = self._current()
        if current.error is not None:
            raise RuntimeError("terminal error must be consumed through take_error")
        partitions = tuple(
            partition
            for partition in current.materialized_partitions()
            if int(partition.partition_id) not in self._sent_partitions
        )
        if partitions:
            self._sent_partitions.update(int(partition.partition_id) for partition in partitions)
            return CompletionReport(step_id=int(current.step_id), partitions=partitions)
        if current.complete and not current.partition_order and not self._empty_sent:
            self._empty_sent = True
            return CompletionReport(step_id=int(current.step_id), partitions=())
        raise RuntimeError("step output cursor has no query-ready partition")

    def take_error(self) -> WorkerError:
        error = self.error
        if error is None or self._error_sent:
            raise RuntimeError("step output cursor has no unread terminal error")
        self._error_sent = True
        return error

    def pending(self) -> bool:
        current = self._current()
        if current.error is not None:
            return not self._error_sent
        if not current.complete:
            return True
        if not current.partition_order:
            return not self._empty_sent
        return any(
            int(partition.partition_id) not in self._sent_partitions
            for partition in current.materialized_partitions()
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_close(self)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class PinnedTokenCapture:
    """One token range copied into a partition's pinned output buffer."""

    buffer: PinnedOutputBuffer
    offset: int
    count: int

    def ready(self) -> bool:
        return self.buffer.ready()

    def values(self) -> tuple[int, ...]:
        return self.buffer.read_tokens(self)


@dataclass(frozen=True, slots=True)
class PinnedByteCapture:
    """One shaped byte range copied into a partition's pinned output buffer."""

    buffer: PinnedOutputBuffer
    offset: int
    count: int
    shape: tuple[int, ...]
    external: torch.Tensor | None = None

    def ready(self) -> bool:
        return self.buffer.ready()

    def tensor(self) -> torch.Tensor:
        return self.buffer.read_bytes(self)

    def numpy(self) -> Any:
        return self.tensor().numpy()


class PinnedOutputBuffer:
    """Pinned host storage and completion events for one partition commit."""

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
        "_deferred",
        "_reserved_ns",
        "_device_started_ns",
        "_copy_started_ns",
        "_sealed_ns",
        "_ready_ns",
        "_timing",
        "_retained_until_ready",
    )

    def __init__(
        self,
        rows: int,
        *,
        token_capacity: int,
        devices: Sequence[torch.device | str] = (),
        event_pool: DeviceEventPool,
    ) -> None:
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
        self._deferred = False
        self._reserved_ns = time.perf_counter_ns()
        self._device_started_ns = 0
        self._copy_started_ns = 0
        self._sealed_ns = 0
        self._ready_ns = 0
        self._timing: tuple[int, int, int, int] | None = None
        self._retained_until_ready: list[object] = []

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def row_count(self) -> int:
        return self._rows

    def register_device(self, device: torch.device | str) -> None:
        if self._sealed:
            raise _invariant("completion device was registered after its buffer was sealed")
        target = canonical_device(device)
        if target.type == "cuda" and target not in self.devices:
            raise _invariant("completion work uses an undeclared CUDA device")

    def begin_device(self, device: torch.device | str) -> None:
        if self._sealed:
            raise _invariant("completion device timing began after its buffer was sealed")
        target = canonical_device(device)
        if self._device_started_ns == 0:
            self._device_started_ns = time.perf_counter_ns()
        if target.type != "cuda":
            return
        self.register_device(target)
        name = str(target)
        if name in self._start_events:
            return
        event = self.event_pool.acquire(target, timing=True)
        self.event_pool.retain(event, target)
        self.event_pool.record(event, target)
        self._start_events[name] = event

    def _mark_copy_started(self, device: torch.device) -> None:
        if self._copy_started_ns == 0:
            self._copy_started_ns = time.perf_counter_ns()
        name = str(device)
        if name in self._producer_events:
            return
        if name not in self._start_events:
            self.begin_device(device)
        event = self.event_pool.acquire(device, timing=True)
        self.event_pool.retain(event, device)
        self.event_pool.record(event, device)
        self._producer_events[name] = event

    def capture(self, tokens: torch.Tensor) -> PinnedTokenCapture:
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
        return PinnedTokenCapture(self, offset, count)

    def capture_bytes(self, value: torch.Tensor) -> PinnedByteCapture:
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
        return PinnedByteCapture(
            self,
            offset,
            count,
            tuple(int(value) for value in contiguous.shape),
        )

    def capture_bytes_into(
        self,
        value: torch.Tensor,
        storage: torch.Tensor,
    ) -> PinnedByteCapture:
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
        return PinnedByteCapture(
            self,
            0,
            count,
            tuple(int(item) for item in contiguous.shape),
            host,
        )

    def _byte_floor(self) -> int:
        return int(self._host.numel()) * int(self._host.element_size()) - self._byte_cursor

    def seal(self) -> None:
        if self._sealed:
            return
        for device in self.devices:
            name = str(device)
            if name not in self._start_events:
                self.begin_device(device)
            if name not in self._producer_events:
                self._mark_copy_started(device)
            event = self.event_pool.acquire(device, timing=True)
            self.event_pool.retain(event, device)
            self.event_pool.record(event, device)
            self._events[name] = event
            self.event_pool.schedule_completion_wake(device, event)
        self._sealed = True
        self._sealed_ns = time.perf_counter_ns()

    def ready(self) -> bool:
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

    def read_tokens(self, capture: PinnedTokenCapture) -> tuple[int, ...]:
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

    def read_bytes(self, capture: PinnedByteCapture) -> torch.Tensor:
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
        if self._timing is None:
            raise _invariant("completion timing was read before observation")
        return self._timing

    def discard(self, row: int, generation: int) -> None:
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
        return tuple(
            (
                *self._start_events.values(),
                *self._producer_events.values(),
                *self._events.values(),
            )
        )

    def _release_events(self) -> None:
        if self._events_released or self._deferred:
            return
        for event in self._all_events():
            self.event_pool.release(event)
        self._events_released = True

    def _defer_release(self) -> None:
        if self._events_released or self._deferred:
            return
        events = self._all_events()
        if events:
            self._deferred = True
            self.event_pool.defer_release(events, self)
        else:
            self._events_released = True


class DeferredTokenSpan:
    """One token vector backed exclusively by host-observation storage."""

    __slots__ = ("capture", "count", "_values")

    def __init__(self, capture: PinnedTokenCapture) -> None:
        self.capture = capture
        self.count = int(capture.count)
        self._values: tuple[int, ...] | None = None

    def ready(self) -> bool:
        return self._values is not None or self.capture.ready()

    def finalize(self) -> tuple[int, ...]:
        if self._values is None:
            self._values = self.capture.values()
        return self._values


class DeferredToken:
    """A protocol integer finalized only when the worker serializes its result."""

    __slots__ = ("span", "index")

    def __init__(self, span: DeferredTokenSpan, index: int) -> None:
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
        if isinstance(other, DeferredToken):
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


class DeferredSampleSpan:
    """Selected tokens, row validity, predicates, and accepted counts."""

    __slots__ = ("capture", "count", "_values")

    def __init__(
        self,
        capture: PinnedTokenCapture | None,
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


class DeferredSampleToken(DeferredToken):
    __slots__ = ("sample_span",)

    def __init__(self, span: DeferredSampleSpan, index: int) -> None:
        self.sample_span = span
        self.span = cast(DeferredTokenSpan, span)
        self.index = int(index)

    def ready(self) -> bool:
        return self.sample_span.ready()

    def finalize(self) -> int:
        return self.sample_span.token(self.index)


class DeferredInteger:
    __slots__ = ("span", "index")

    def __init__(self, span: DeferredSampleSpan, index: int) -> None:
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
        if isinstance(other, DeferredInteger):
            return self.finalize() == other.finalize()
        if isinstance(other, int):
            return self.finalize() == other
        return NotImplemented


class DeferredDerivedInteger:
    __slots__ = ("source", "offset")

    def __init__(self, source: DeferredInteger, offset: int) -> None:
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


class DeferredSpeculativePoint:
    __slots__ = ("accepted", "terminal_prefix")

    def __init__(self, accepted: DeferredInteger, terminal_prefix: int | None) -> None:
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


class DeferredSpeculativeTokens(Sequence[int]):
    __slots__ = ("draft", "accepted", "continuation", "terminal_prefix", "_value")

    def __init__(
        self,
        draft: tuple[int, ...],
        accepted: DeferredInteger,
        continuation: int | DeferredToken,
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
            not isinstance(continuation, DeferredToken) or continuation.ready()
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


class DeferredLogprobBatch:
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
        capture: PinnedTokenCapture | None,
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


class DeferredLogprobValue:
    __slots__ = ("batch", "index")

    def __init__(self, batch: DeferredLogprobBatch, index: int) -> None:
        self.batch = batch
        self.index = int(index)

    def ready(self) -> bool:
        return self.batch.ready()

    def finalize(self) -> float:
        return self.batch.finalize()[self.index][0]

    def __float__(self) -> float:
        return self.finalize()


class DeferredTopLogprobs:
    __slots__ = ("batch", "index")

    def __init__(self, batch: DeferredLogprobBatch, index: int) -> None:
        self.batch = batch
        self.index = int(index)

    def ready(self) -> bool:
        return self.batch.ready()

    def finalize(self) -> tuple[tuple[int, float, int], ...]:
        return self.batch.finalize()[self.index][1]

    def max_entries(self) -> int:
        local = self.batch.rows.index(self.index)
        return 1 + int(self.batch.counts[local]) + len(self.batch.requested_ids[local])


class DeferredCompletionTask(ABC):
    """A completion-owned asynchronous action with nominal readiness semantics."""

    __slots__ = ()

    @abstractmethod
    def ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> object:
        raise NotImplementedError


class DeferredLogprobPayload(DeferredCompletionTask):
    __slots__ = ("logprob", "top_logprobs", "prompt_logprobs", "_value")

    def __init__(
        self,
        logprob: float | DeferredLogprobValue | None,
        top_logprobs: tuple[tuple[int, float, int], ...] | DeferredTopLogprobs | None,
        prompt_logprobs: tuple[
            tuple[tuple[int, float, int], ...] | DeferredTopLogprobs,
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
            (not isinstance(self.logprob, DeferredLogprobValue) or self.logprob.ready())
            and (
                not isinstance(self.top_logprobs, DeferredTopLogprobs) or self.top_logprobs.ready()
            )
            and all(
                not isinstance(position, DeferredTopLogprobs) or position.ready()
                for position in self.prompt_logprobs
            )
        )

    def max_encoded_bytes(self) -> int:
        def entry_bound(
            entries: tuple[tuple[int, float, int], ...] | DeferredTopLogprobs | None,
        ) -> int:
            if isinstance(entries, DeferredTopLogprobs):
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
            if isinstance(self.top_logprobs, DeferredTopLogprobs)
            else self.top_logprobs or ()
        )
        out = bytearray(b"\x00" if logprob is None else b"\x01" + struct.pack("<f", logprob))
        out += struct.pack("<I", len(top))
        for token_id, value, rank in top:
            out += struct.pack("<IfI", int(token_id), float(value), int(rank))
        out += struct.pack("<I", len(self.prompt_logprobs))
        for position in self.prompt_logprobs:
            entries = position.finalize() if isinstance(position, DeferredTopLogprobs) else position
            out += struct.pack("<I", len(entries))
            for token_id, value, rank in entries:
                out += struct.pack("<IfI", int(token_id), float(value), int(rank))
        self._value = bytes(out)
        return self._value

    def __bytes__(self) -> bytes:
        return self.finalize()


class DeferredTransferPayload:
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


class DeferredImagePayload(DeferredCompletionTask):
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
        capture: PinnedByteCapture,
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


class DeferredDigest(DeferredSemanticDigest):
    """A semantic digest finalized from a query-ready completion generation.

    The digest includes committed tokens copied asynchronously into the pinned
    output buffer. Finalization reads that host storage only after every copy
    event reports ready, validates the buffer generation, and releases the
    observed row. A device-parent successor may retain its predecessor's
    pending digest, so finalization follows the request lineage while unrelated
    completions remain independently dispatchable.
    """

    __slots__ = (
        "_record",
        "_parent",
        "_plan_digest",
        "_buffer",
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
        parent: str | DeferredSemanticDigest,
        plan_digest: str,
        buffer: PinnedOutputBuffer,
        row: int,
        predicated_parent: Callable[[], tuple[VersionRef, RequestRuntime]],
        *,
        status: OpStatus,
        selected_point: int,
        resolved_callback: Callable[[ModelOutput, str, str], None] | None = None,
        completion_tasks: tuple[DeferredCompletionTask, ...] = (),
    ) -> None:
        self._record: ModelOutput | None = None
        self._parent = parent
        self._plan_digest = plan_digest
        self._buffer: PinnedOutputBuffer | None = buffer
        self._row = int(row)
        self._generation = int(buffer.generation)
        self._completion_timing: tuple[int, int, int, int] | None = None
        self._value: str | None = None
        self._observed = False
        self._invalid_sampling = False
        self._predicated = status is OpStatus.PREDICATED
        self._predicated_parent: Callable[[], tuple[VersionRef, RequestRuntime]] | None = (
            predicated_parent
        )
        self._selected_point = int(selected_point)
        self._selected_runtime: RequestRuntime | None = None
        self._resolved_callback = resolved_callback
        self._completion_tasks = completion_tasks
        self._completion_error = False

    def bind_record(self, record: ModelOutput) -> ModelOutput:
        """Bind the one final record that carries this deferred digest."""

        if self._record is not None:
            raise RuntimeError("completion digest record was bound more than once")
        if record.semantic_digest is not self:
            raise RuntimeError("completion record does not carry its bound digest")
        if int(record.completion_slot_generation) != self._generation:
            raise RuntimeError("completion record generation does not match its output buffer")
        if (record.status is OpStatus.PREDICATED) != self._predicated:
            raise RuntimeError("completion record status changed during digest binding")
        if int(record.selected_point) != int(self._selected_point):
            raise RuntimeError("completion selected point changed during digest binding")
        self._record = record
        return record

    def ready(self) -> bool:
        if self._record is None:
            raise RuntimeError("completion digest has no bound record")
        if self._value is not None:
            return True
        if isinstance(self._parent, DeferredSemanticDigest) and not self._parent.ready():
            return False
        if self._buffer is None or not self._buffer.ready():
            return False
        for task in self._completion_tasks:
            if not task.ready():
                return False
        return True

    def finalize(self) -> str:
        if self._value is None:
            if not self.ready():
                raise RuntimeError("completion digest was resolved before query-ready")
            record = self._record
            if record is None:
                raise RuntimeError("completion digest has no bound record")
            parent = (
                self._parent.finalize()
                if isinstance(self._parent, DeferredSemanticDigest)
                else self._parent
            )
            if self._predicated:
                self._resolve_predicated(cast(str, parent))
            else:
                try:
                    for task in self._completion_tasks:
                        task.finalize()
                except Exception:
                    self._completion_error = True
                    self._value = _completion_error_record(record).compute_semantic_digest(
                        parent_semantic=cast(str, parent),
                        plan_digest=self._plan_digest,
                    )
                else:
                    try:
                        # The digest packs each committed token via ``__index__``, which
                        # finalizes a deferred token exactly as ``int(value)`` would, so
                        # the record is hashed in place without a concrete-token copy.
                        digest = record.compute_semantic_digest(
                            parent_semantic=cast(str, parent),
                            plan_digest=self._plan_digest,
                        )
                        if self._resolved_callback is not None:
                            self._resolved_callback(record, digest, cast(str, parent))
                        self._value = digest
                    except _PredicatedOperation:
                        self._predicated = True
                        self._resolve_predicated(cast(str, parent))
                    except _InvalidSamplingDistribution:
                        self._invalid_sampling = True
                        self._value = _invalid_sampling_record(record).compute_semantic_digest(
                            parent_semantic=cast(str, parent),
                            plan_digest=self._plan_digest,
                        )
            buffer = self._buffer
            if buffer is None:
                raise RuntimeError("completion digest lost its pinned output buffer")
            buffer.observe(self._row, self._generation)
            self._completion_timing = buffer.timing()
            self._observed = True
            self._buffer = None
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
        return self.finalize()

    def completion_timing(self) -> tuple[int, int, int, int]:
        self.finalize()
        return self._completion_timing or (0, 0, 0, 0)

    @property
    def invalid_sampling(self) -> bool:
        self.finalize()
        return self._invalid_sampling

    @property
    def predicated(self) -> bool:
        self.finalize()
        return self._predicated

    @property
    def completion_error(self) -> bool:
        self.finalize()
        return self._completion_error

    @property
    def selected_point(self) -> int:
        self.finalize()
        return int(self._selected_point)

    @property
    def selected_runtime(self) -> RequestRuntime:
        self.finalize()
        if self._selected_runtime is None:
            raise RuntimeError("predicated operation lost its selected runtime state")
        return self._selected_runtime

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DeferredDigest):
            return self.finalize() == other.finalize()
        if isinstance(other, str):
            return self.finalize() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.finalize())

    def __deepcopy__(self, memo: dict[int, object]) -> DeferredDigest:
        # A committed session snapshot shares ownership of the exact pinned
        # completion generation and its lineage digest.
        memo[id(self)] = self
        return self

    def __del__(self) -> None:
        buffer = self._buffer
        if buffer is not None and not self._observed:
            buffer.discard(self._row, self._generation)


class DeferredErrorDigest(DeferredSemanticDigest):
    """An error digest causally chained to an unobserved parent completion."""

    __slots__ = ("_parent", "_plan_digest", "_record", "_value")

    def __init__(
        self,
        parent: DeferredSemanticDigest,
        record: ModelOutput,
        plan_digest: str,
    ) -> None:
        self._parent = parent
        self._record = record
        self._plan_digest = plan_digest
        self._value: str | None = None

    def ready(self) -> bool:
        return self._value is not None or self._parent.ready()

    def finalize(self) -> str:
        if self._value is None:
            if not self.ready():
                raise RuntimeError("error digest was resolved before its parent was query-ready")
            self._value = self._record.compute_semantic_digest(
                self._parent.finalize(),
                self._plan_digest,
            )
        return self._value

    def __str__(self) -> str:
        return self.finalize()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (DeferredDigest, DeferredErrorDigest)):
            return self.finalize() == other.finalize()
        if isinstance(other, str):
            return self.finalize() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.finalize())

    def __deepcopy__(self, memo: dict[int, object]) -> DeferredErrorDigest:
        memo[id(self)] = self
        return self


def _record_ready(record: ModelOutput) -> bool:
    """Whether a completion's deferred token copy and digest chain have landed."""

    digest = record.semantic_digest
    if isinstance(digest, DeferredDigest):
        return digest.ready()
    if isinstance(digest, DeferredErrorDigest):
        return digest.ready()
    for value in cast(tuple[object, ...], record.committed_tokens):
        if isinstance(value, DeferredToken) and not value.ready():
            return False
    return True


def _finalized_record(record: ModelOutput) -> ModelOutput:
    digest = record.semantic_digest
    if isinstance(digest, DeferredErrorDigest):
        return replace(record, semantic_digest=digest.finalize())
    if isinstance(digest, DeferredDigest):
        resolved = digest.finalize()
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
    lengths = record.logical_lengths
    span = record.token_span
    if type(lengths.token_len) is not int or any(
        type(value) is not int
        for value in (
            lengths.kv_visible_len,
            lengths.kv_computed_len,
            lengths.latent_len,
        )
    ):
        lengths = LogicalLengths(
            token_len=int(lengths.token_len),
            kv_visible_len=int(lengths.kv_visible_len),
            kv_computed_len=int(lengths.kv_computed_len),
            latent_len=int(lengths.latent_len),
        )
    if type(span.base) is not int or type(span.len) is not int:
        span = TokenSpan(base=int(span.base), len=int(span.len))
    tokens = record.committed_tokens
    if type(tokens) is not tuple or any(type(value) is not int for value in tokens):
        tokens = tuple(int(value) for value in tokens)
    return replace(
        record,
        selected_point=int(record.selected_point),
        logical_lengths=lengths,
        token_span=span,
        committed_tokens=tokens,
        semantic_digest=resolved,
        timing_counters=timing,
    )


def _invalid_sampling_record(record: ModelOutput) -> ModelOutput:
    return replace(
        record,
        status=OpStatus.ERROR,
        selected_point=max(0, int(record.selected_point) - 1),
        token_span=replace(record.token_span, len=0),
        committed_tokens=(),
        finish_flags=FinishFlags(),
        product_generations=(),
        error_code=ErrorCode.INVALID_OPERATION,
    )


def _completion_error_record(record: ModelOutput) -> ModelOutput:
    return replace(
        record,
        status=OpStatus.ERROR,
        token_span=replace(record.token_span, len=0),
        committed_tokens=(),
        finish_flags=FinishFlags(),
        product_generations=(),
        error_code=ErrorCode.COMPUTE_ERROR,
    )


def _predicated_record(
    record: ModelOutput,
    selected_point: int,
    runtime: RequestRuntime,
) -> ModelOutput:
    return replace(
        record,
        status=OpStatus.PREDICATED,
        selected_point=int(selected_point),
        logical_lengths=LogicalLengths(
            token_len=runtime.logical_position,
            kv_visible_len=runtime.kv_visible_len,
            kv_computed_len=runtime.kv_computed_len,
            latent_len=0,
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
            (DeferredImagePayload, DeferredLogprobPayload, DeferredTransferPayload),
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
                    DeferredImagePayload,
                    DeferredLogprobPayload,
                    DeferredTransferPayload,
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
