"""Persistent output storage and concrete lane result materialization."""

from __future__ import annotations

import concurrent.futures
import logging
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial
from threading import RLock
from typing import Final

import torch

from uniserve.runtime import EventPool
from uniserve.runtime.device import canonical_device
from uniserve.runtime.resources import close_resources
from uniserve_worker.protocol.call import CallKind
from uniserve_worker.protocol.identity import (
    BufferId,
    CallId,
    RequestKey,
)

from ..foundation.errors import (
    WorkerError,
    WorkerErrorCode,
    resource_error,
)
from ..media.storage import publish_media_bytes
from ..profiling import timing_events_enabled
from ..protocol.batch import LatentParams, TensorPublication
from ..protocol.call import Call, CallStatus, ErrorCode
from ..protocol.output import (
    FinishFlags,
    MediaOutput,
    PosixShmArtifact,
    RequestOutput,
    TimingCounters,
)
from ..protocol.transfer import KvTransfer, Locator
from ..runtime.host_lane import HostTask
from ..runtime.latent_pool import LatentStaging
from ..runtime.request import RequestProgress, RequestState
from ..runtime.tensor_store import TensorRead, TensorRecord
from ..transfer.exports import ExportLocations
from .sampling import LogprobValues, SamplerRow

logger = logging.getLogger(__name__)

__all__ = [
    "PendingOutput",
    "OutputBuffer",
    "OutputPool",
]

# The sampler packs one call's completion column as four consecutive
# row-major fields: [valid | active | token | accepted], each `count` wide.
# `valid` marks a usable sampling distribution, `active` the resolved device
# predicate, `token` the selected token, and `accepted` the speculative
# acceptance count.
_SAMPLING_FIELDS_PER_CALL: Final[int] = 4
_next_buffer_generation = 1


def _invariant(message: str) -> WorkerError:
    """Construct a classified invariant error for output-buffer misuse."""
    return WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


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
        "logprob_layouts",
        "_logprob_cache",
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
        "_completion_future",
        "_completion_registered",
        "_release_to_pool",
        "_released_to_pool",
        "_cpu_readers",
        "_reader_lock",
    )

    def __init__(
        self,
        rows: int,
        *,
        token_capacity: int,
        devices: Sequence[torch.device | str] = (),
        event_pool: EventPool,
        release_to_pool: Callable[[OutputBuffer], None] | None = None,
    ) -> None:
        """Reserve pinned completion rows and generation-tagged CUDA copy.

        state.
        """
        global _next_buffer_generation
        count = int(rows)
        capacity = int(token_capacity)
        if count < 1:
            raise ValueError(
                "a pinned output buffer must contain at least one call row"
            )
        if capacity < count:
            raise resource_error(
                "completion row count exceeds pinned output capacity"
            )

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

        # Token captures fill this int64 buffer from the head upward; byte
        # captures share the same allocation from the tail downward.
        self._host = torch.empty(
            capacity,
            dtype=torch.long,
            device="cpu",
            pin_memory=bool(self.devices),
        )
        self._token_cursor = 0
        self._byte_cursor = 0

        self._token_cache: dict[tuple[int, int], tuple[int, ...]] = {}
        self.logprob_layouts: dict[
            tuple[int, int],
            tuple[
                tuple[int, ...],
                tuple[int, ...],
                tuple[tuple[int, ...], ...],
                int,
                int,
            ],
        ] = {}
        self._logprob_cache: dict[
            tuple[int, int],
            dict[int, tuple[float, tuple[tuple[int, float, int], ...]]],
        ] = {}
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

        self._completion_future: concurrent.futures.Future[None] | None = None
        self._completion_registered = False

        self._release_to_pool = release_to_pool
        self._released_to_pool = False
        self._cpu_readers = 0
        self._reader_lock = RLock()

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
            raise _invariant(
                "output storage was leased before its prior events were "
                "released"
            )
        count = int(rows)
        capacity = int(token_capacity)
        if count < 1 or capacity < count:
            raise resource_error("output lease shape is invalid")

        normalized: list[torch.device] = []
        for value in devices:
            device = canonical_device(value)
            if device.type == "cuda" and device not in normalized:
                normalized.append(device)

        # Grow the pinned allocation when the new lease needs more words, or
        # re-pin it when this lease introduces CUDA producers.
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
        self.logprob_layouts.clear()
        self._logprob_cache.clear()
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

        self._completion_future = None
        self._completion_registered = False
        self._released_to_pool = False

    @property
    def generation(self) -> int:
        """Identify the lease generation guarding all captures from this buffer.

        use.
        """
        return self._generation

    def register_device(self, device: torch.device | str) -> None:
        """Verify that a CUDA producer belongs to the devices declared for this.

        lease.
        """
        if self._sealed:
            raise _invariant(
                "completion device was registered after its buffer was sealed"
            )
        target = canonical_device(device)
        if target.type == "cuda" and target not in self.devices:
            raise _invariant("completion work uses an undeclared CUDA device")

    def begin_device(self, device: torch.device | str) -> None:
        """Mark device execution start and record its optional timing event."""
        if self._sealed:
            raise _invariant(
                "completion device timing began after its buffer was sealed"
            )
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
        """Record the stream event that protects one device-to-host completion.

        copy.
        """
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

    def capture(self, tokens: torch.Tensor) -> tuple[int, int]:
        """Copy a token tensor into the next bounded span of pinned host.

        storage.
        """
        if self._sealed:
            raise _invariant(
                "completion capture was registered after its buffer was sealed"
            )
        flat = tokens.reshape(-1).to(dtype=torch.long)
        count = int(flat.numel())

        offset = self._token_cursor
        end = offset + count
        if end * int(self._host.element_size()) > self._byte_floor():
            raise resource_error(
                "completion token span exceeds pinned output capacity"
            )

        host = self._host[offset:end]
        if flat.device.type == "cuda":
            if not bool(host.is_pinned()):
                raise _invariant(
                    "CUDA completion copy targets pageable host storage"
                )
            device = canonical_device(flat.device)
            self.register_device(device)
            self._mark_copy_started(device)
            host.copy_(flat, non_blocking=True)
        else:
            host.copy_(flat.to(device="cpu"))

        self._token_cursor = end
        return offset, count

    def capture_bytes(self, value: torch.Tensor) -> torch.Tensor:
        """Copy uint8 values into the buffer tail and borrow their shaped host.

        view.

        The caller must wait for this buffer's completion before reading the
        view, and retain a CPU reader until its last asynchronous use finishes.
        """
        if value.dtype is not torch.uint8:
            raise ValueError("completion byte capture requires uint8 storage")
        if self._sealed:
            raise _invariant(
                "completion byte capture was registered after its buffer was "
                "sealed"
            )

        contiguous = value.detach().contiguous()
        count = int(contiguous.numel())
        if count < 1:
            raise ValueError("completion byte capture must not be empty")

        # Byte captures descend from the allocation tail toward the token head.
        end = self._byte_floor()
        offset = end - count
        if offset < self._token_cursor * int(self._host.element_size()):
            raise resource_error(
                "completion byte span exceeds pinned output capacity"
            )

        host = self._host.view(torch.uint8)[offset:end]
        flat = contiguous.reshape(-1)
        if flat.device.type == "cuda":
            if not bool(host.is_pinned()):
                raise _invariant(
                    "CUDA completion byte copy targets pageable host storage"
                )
            device = canonical_device(flat.device)
            self.register_device(device)
            self._mark_copy_started(device)
            host.copy_(flat, non_blocking=True)
        else:
            host.copy_(flat.to(device="cpu"))

        self._byte_cursor += count
        return host.view(contiguous.shape)

    def _byte_floor(self) -> int:
        """Return the byte offset where the captured byte tail begins.

        Byte captures fill the allocation from its end downward, so this is
        both the end offset of the next capture and the lower bound that token
        captures from the head must not cross.
        """
        return (
            int(self._host.numel()) * int(self._host.element_size())
            - self._byte_cursor
        )

    def seal(self) -> None:
        """Record completion events for every producer device and prohibit.

        additional captures.
        """
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
        self._bind_completion()

    def ready(self) -> bool:
        """Return whether all sealed device-copy events have completed."""
        if not self._sealed:
            return False
        if self._ready_ns:
            return True
        if any(not bool(event.query()) for event in self._events.values()):
            return False
        self._ready_ns = time.perf_counter_ns()
        self._complete_dependents()
        return True

    def completion_future(self) -> concurrent.futures.Future[None]:
        """Expose this output lease's existing device fence to physical storage.

        owners.

        The future belongs to this lease even after the pinned buffer is reused.
        It adds no CUDA event or host/device payload allocation.
        """
        if self._completion_future is None:
            self._completion_future = concurrent.futures.Future()
        future = self._completion_future
        self._bind_completion()
        if self.ready():
            self._complete_dependents()
        return future

    def _bind_completion(self) -> None:
        future = self._completion_future
        if not self._sealed or future is None or self._completion_registered:
            return
        self._completion_registered = True
        if self._ready_ns or self._events_released or not self._events:
            self._resolve_completion(future)
            return

        # Physical retirement is independent of whether a host reads the output
        # report. Retain the existing fences until the event loop observes them,
        # and bind the callback to this lease's future across buffer reuse.
        for device in self.devices:
            self.event_pool.retain(self._events[str(device)], device)
        self.event_pool.defer_release(
            tuple(self._events.values()),
            future,
            completed=partial(self._resolve_completion, future),
        )

    @staticmethod
    def _resolve_completion(future: concurrent.futures.Future[None]) -> None:
        if not future.done():
            future.set_result(None)

    def _complete_dependents(self) -> None:
        future = self._completion_future
        if future is not None:
            self._resolve_completion(future)

    def read_tokens(self, offset: int, count: int) -> tuple[int, ...]:
        """Read a registered integer range only after its producer copy.

        completes.
        """
        key = (int(offset), int(count))
        cached = self._token_cache.get(key)
        if cached is not None:
            return cached
        if not self.ready():
            raise _invariant(
                "completion storage was observed before its copy event was "
                "ready"
            )
        end = offset + count
        if offset < 0 or end > self._token_cursor:
            raise _invariant(
                "completion capture range is outside its registered token "
                "extent"
            )
        values = tuple(int(value) for value in self._host[offset:end].tolist())
        self._token_cache[key] = values
        return values

    def logprob_values(
        self, span: tuple[int, int, int]
    ) -> tuple[float, tuple[tuple[int, float, int], ...]]:
        """Decode one row.

        sharing parsing of its packed column with other rows.
        """
        offset, count, index = span
        key = (offset, count)
        details = self._logprob_cache.get(key)
        if details is None:
            details = decode_logprobs(
                self.read_tokens(offset, count), self.logprob_layouts[key]
            )
            self._logprob_cache[key] = details
        return details[index]

    def observe(self, row: int, generation: int) -> tuple[int, int]:
        """Mark one result row observed and return copy and host-observation.

        timing.
        """
        index = int(row)
        if int(generation) != self._generation:
            raise _invariant(
                "completion record carries a stale buffer generation"
            )
        if index < 0 or index >= self._rows:
            raise _invariant(
                "completion record row is outside its pinned output buffer"
            )
        if not self.ready():
            raise _invariant(
                "completion record was observed before query-ready"
            )
        observed_ns = time.perf_counter_ns()
        if self._timing is None:
            # All reported durations are microseconds. CUDA elapsed_time()
            # yields milliseconds; host-clock deltas are nanoseconds.
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
                        raise _invariant(
                            "completion timing events are incomplete"
                        )
                    device_us = max(
                        device_us,
                        max(
                            0,
                            round(
                                float(start_event.elapsed_time(producer_event))
                                * 1000.0
                            ),
                        ),
                    )
                    copy_us = max(
                        copy_us,
                        max(
                            0,
                            round(
                                float(producer_event.elapsed_time(end_event))
                                * 1000.0
                            ),
                        ),
                    )

            # Without CUDA events (host-only copies), derive the same phases
            # from host-clock milestones.
            if not self._events and self._device_started_ns:
                copy_started_ns = self._copy_started_ns or self._sealed_ns
                device_us = (
                    max(0, copy_started_ns - self._device_started_ns) // 1000
                )
                copy_us = max(0, self._sealed_ns - copy_started_ns) // 1000

            ready_to_observed_us = max(0, observed_ns - self._ready_ns) // 1000
            self._timing = (queued_us, device_us, copy_us, ready_to_observed_us)

        self._observed.add(index)
        if len(self._observed) == self._rows:
            self._release_events()
        return self._timing[2], self._timing[3]

    def timing(self) -> tuple[int, int, int, int]:
        """Expose submit, seal, ready.

        and observation timestamps for the completed lease.
        """
        if self._timing is None:
            raise _invariant("completion timing was read before observation")
        return self._timing

    def discard(self, row: int, generation: int) -> None:
        """Retire one unobserved result row while preserving unfinished device.

        copies.
        """
        index = int(row)
        if (
            int(generation) != self._generation
            or index < 0
            or index >= self._rows
        ):
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
        return (
            *self._start_events.values(),
            *self._producer_events.values(),
            *self._events.values(),
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
        """Defer buffer reuse until every outstanding device-copy event.

        completes.
        """
        if self._events_released or self._release_pending:
            return
        events = self._all_events()
        if events:
            self._release_pending = True
            self.event_pool.defer_release(
                events, self, completed=self.events_released
            )
        else:
            self._events_released = True
            self._return_to_pool()

    def events_released(self) -> None:
        """Receive completion of an event-pool asynchronous release."""
        self._release_pending = False
        self._events_released = True
        self._complete_dependents()
        self._return_to_pool()

    def retain_cpu_reader(self) -> Callable[[], None]:
        """Retain pinned storage until a configured CPU task finishes reading.

        it.
        """
        with self._reader_lock:
            if self._released_to_pool:
                raise _invariant("CPU reader acquired a retired output buffer")
            self._cpu_readers += 1
        return self._release_cpu_reader

    def _release_cpu_reader(self) -> None:
        with self._reader_lock:
            if self._cpu_readers < 1:
                raise _invariant("output CPU reader count underflow")
            self._cpu_readers -= 1
        self._return_to_pool()

    def _return_to_pool(self) -> None:
        """Return storage only after both device writes and CPU readers.

        retire.
        """
        with self._reader_lock:
            if (
                self._released_to_pool
                or not self._events_released
                or self._cpu_readers
                or self._release_to_pool is None
            ):
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
        event_pool: EventPool,
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
        """Lease a reset or newly allocated output buffer within startup row.

        and byte bounds.
        """
        words = int(token_capacity)
        if words > self.max_words:
            raise resource_error(
                "lane output exceeds its startup storage bound"
            )

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
        """Accept a released completion buffer back into the bounded free.

        list.
        """
        with self._lock:
            if self._closed:
                return
            if buffer not in self._buffers or buffer in self._free:
                raise _invariant("output pool received an invalid lease return")
            self._free.append(buffer)

    def close(self) -> None:
        """Stop admission and retire every output lease through its existing.

        device fences.
        """
        with self._lock:
            self._closed = True
            buffers = tuple(self._buffers)
            self._free.clear()
            self._buffers.clear()

        for buffer in buffers:
            if not buffer._events_released:
                buffer.seal()
                for event in buffer._all_events():
                    event.synchronize()
            buffer.abandon()
        self.event_pool.reap()


class _InvalidSamplingDistribution(RuntimeError):  # noqa: N818  # deliberate taxonomy name
    """Marks a sampling row whose filtered probability mass is unusable."""

    pass


class _PredicatedCall(RuntimeError):  # noqa: N818  # deliberate taxonomy name
    """Marks an call suppressed by its resolved device predicate."""

    pass


def capture_logprobs(
    details: LogprobValues | None,
    output: OutputBuffer,
) -> dict[int, tuple[int, int, int]]:
    """Store one packed score column and return its call row ranges."""
    if details is None:
        return {}
    packed, rows, counts, requested_ids, max_count, max_requested = details
    capture = output.capture(packed)
    key = capture
    output.logprob_layouts[key] = (
        rows,
        counts,
        requested_ids,
        max_count,
        max_requested,
    )
    return {index: (*key, index) for index in rows}


def capture_samples(
    samples: Sequence[SamplerRow],
    requests: Sequence[PendingOutput],
    output: OutputBuffer,
) -> None:
    """Attach row ranges while copying each shared sampling column only once.

    Call on the producer stream before sealing the output buffer. Its fence
    protects all sampling and score ranges until their PendingOutput retires.
    """
    spans: dict[int, tuple[int, int]] = {}
    details: dict[int, dict[int, tuple[int, int, int]]] = {}
    for sample, request in zip(samples, requests, strict=True):
        metadata = sample.batch.completion
        count = int(metadata.numel()) // _SAMPLING_FIELDS_PER_CALL
        if metadata.numel() != count * _SAMPLING_FIELDS_PER_CALL or not (
            0 <= sample.index < count
        ):
            raise RuntimeError("sampling completion vectors do not align")

        # Rows of a shared batch reference one completion column; capture it
        # on first encounter and hand each request its row span.
        key = id(metadata)
        span = spans.get(key)
        if span is None:
            capture = output.capture(metadata)
            span = capture
            spans[key] = span
        request.sampling_range = (*span, sample.index)

        if sample.batch.logprobs is not None:
            key = id(sample.batch.logprobs)
            if key not in details:
                details[key] = capture_logprobs(sample.batch.logprobs, output)
            request.logprob_range = details[key].get(sample.index)


def sampled_tokens(record: PendingOutput) -> tuple[int, ...]:
    """Resolve validity and speculative acceptance from one captured sampling.

    row.
    """
    if record.sampling_range is None:
        return record.committed_tokens
    offset, extent, index = record.sampling_range
    values = record._sampling_values
    if values is None:
        if record._buffer is None:
            raise RuntimeError("sampling output lost its pinned range")
        values = record._buffer.read_tokens(offset, extent)
        record._sampling_values = values
    # Field layout of the packed sampling column is documented at
    # _SAMPLING_FIELDS_PER_CALL.
    count = extent // _SAMPLING_FIELDS_PER_CALL
    if not bool(values[count + index]):
        raise _PredicatedCall("call predicate selected no state")
    if not bool(values[index]):
        raise _InvalidSamplingDistribution(
            "sampling policy produced an invalid distribution"
        )
    token = values[count * 2 + index]
    accepted = values[count * 3 + index]

    draft = record.draft_tokens
    if not draft:
        return (token,)
    if accepted < 0 or accepted > len(draft):
        raise RuntimeError(
            "speculative acceptance count is outside the draft span"
        )
    if (
        record.terminal_prefix is not None
        and accepted >= record.terminal_prefix
    ):
        return draft[:accepted]
    return (*draft[:accepted], token)


def logprob_entries(record: PendingOutput, span: tuple[int, int, int]) -> int:
    """Return the declared maximum score entries for completion-byte.

    validation.
    """
    if record._buffer is None:
        raise RuntimeError("logprob output lost its pinned range")
    offset, count, index = span
    rows, counts, requested_ids, _max_count, _max_requested = (
        record._buffer.logprob_layouts[(offset, count)]
    )
    local = rows.index(index)
    return 1 + counts[local] + len(requested_ids[local])


def decode_logprobs(
    values: tuple[int, ...],
    layout: tuple[
        tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...], int, int
    ],
) -> dict[int, tuple[float, tuple[tuple[int, float, int], ...]]]:
    """Decode the sampler's packed float bits and row-major rank columns."""
    rows, counts, requested_ids, max_count, max_requested = layout

    # Score columns travel as little-endian float bits inside int64 words.
    def float_value(value: int) -> float:
        return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]

    row_count = len(rows)
    cursor = 0

    def vector(width: int) -> tuple[tuple[int, ...], ...]:
        """Consume one row-major field of fixed width from the packed.

        capture.
        """
        nonlocal cursor
        total = row_count * width
        part = values[cursor : cursor + total]
        if len(part) != total:
            raise RuntimeError("logprob completion metadata is truncated")
        cursor += total
        return tuple(
            tuple(part[row * width : (row + 1) * width])
            for row in range(row_count)
        )

    # The packed column is a fixed sequence of row-major vectors; each
    # vector(width) call consumes the next one in this exact order.
    selected_tokens = vector(1)
    selected_values = vector(1)
    selected_ranks = vector(1)
    top_indexes = vector(max_count)
    top_values = vector(max_count)
    top_ranks = vector(max_count)
    candidate_values = vector(max_requested)
    candidate_ranks = vector(max_requested)
    if cursor != len(values):
        raise RuntimeError("logprob completion metadata has trailing values")

    details: dict[int, tuple[float, tuple[tuple[int, float, int], ...]]] = {}
    for local, result_index in enumerate(rows):
        selected = selected_tokens[local][0]
        selected_value = float_value(selected_values[local][0])
        entries: list[tuple[int, float, int]] = [
            (selected, selected_value, selected_ranks[local][0])
        ]
        seen = {selected}

        for index in range(counts[local]):
            candidate = top_indexes[local][index]
            if candidate not in seen:
                entries.append(
                    (
                        candidate,
                        float_value(top_values[local][index]),
                        top_ranks[local][index],
                    )
                )
                seen.add(candidate)

        for index, candidate in enumerate(requested_ids[local]):
            if candidate not in seen:
                entries.append(
                    (
                        candidate,
                        float_value(candidate_values[local][index]),
                        candidate_ranks[local][index],
                    )
                )
                seen.add(candidate)

        details[result_index] = (selected_value, tuple(entries))
    return details


class PendingOutput:
    """One call's stable predecessor, projected progress.

    and eventual output.

    The output row and CPU tasks retain their actual storage until materialized
    or abandoned. No request progress is installed by this object: RequestPool
    applies accepted snapshots in predecessor order.
    """

    def __init__(
        self,
        call: Call,
        request: RequestState,
        buffer: OutputBuffer,
        row: int,
    ) -> None:
        self.call = call
        self.request = request
        # The call states the coordinates it runs at, so the rank reads them
        # rather than deriving them from a predecessor's record. The device
        # state a previous call left behind stays with the request.
        self.progress: RequestProgress = RequestProgress(
            logical_position=call.coordinates.logical_position,
            rng_counter=request.rng_counter,
            flow_step=call.coordinates.flow_step,
            kv_visible_len=call.coordinates.kv_visible_len,
            kv_computed_len=call.coordinates.kv_computed_len,
            prompt_logits_ready=request.prompt_logits_ready,
        )
        self.accepted_progress: RequestProgress | None = None

        self.status = CallStatus.OK
        self.committed_tokens: tuple[int, ...] = ()
        self.sampling_range: tuple[int, int, int] | None = None
        self._sampling_values: tuple[int, ...] | None = None
        self.logprob_range: tuple[int, int, int] | None = None
        self.prompt_logprob_ranges: tuple[tuple[int, int, int], ...] = ()
        self.logprobs: (
            tuple[float, tuple[tuple[int, float, int], ...]] | None
        ) = None
        self.prompt_logprobs: tuple[
            tuple[tuple[int, float, int], ...], ...
        ] = ()
        self.finish_flags = FinishFlags()
        self.product_generations: tuple[int, ...] = ()
        self.error_code: ErrorCode | None = None

        # Numerical updates are borrowed until the completed group commits to
        # DecodeState. Host acceptance continues to use progress and
        # the completion ranges, independently of these device references.
        self.tensor_exports: dict[BufferId, ExportLocations] = {}
        self.cache_exports: dict[BufferId, ExportLocations] = {}
        self.latent_exports: dict[BufferId, ExportLocations] = {}
        self.exported_locators: list[Locator] = []
        self.cache_publication: tuple[BufferId, KvTransfer] | None = None
        self.cache_installation: (
            tuple[BufferId, BufferId, KvTransfer] | None
        ) = None
        self.device_reads: list[TensorRead] = []
        self.feature_reads: list[TensorRead] = []
        self.writes: list[TensorRecord] = []
        self.predicate: tuple[torch.Tensor, bool] | None = None
        self.token_write: TensorRecord | None = None
        self.transition_write: TensorRecord | None = None
        self.completion_write: TensorRecord | None = None
        self.producer_write: TensorRecord | None = None
        self.sampled: SamplerRow | None = None

        self.runtime_logical_position: int | torch.Tensor = 0
        self.runtime_sampling_position: int | torch.Tensor = 0
        self.runtime_penalty_base: torch.Tensor | None = None
        self.runtime_decode_increment = False
        self.runtime_cache_length: int | torch.Tensor | None = None
        self.runtime_prompt_logits: torch.Tensor | None = None

        self.kv_output: KvTransfer | None = None
        self.products: tuple[TensorPublication, ...] = ()
        # A product whose bytes a host task produces is published with its
        # batch and filled when the task completes. Its consumer is scheduled
        # only after this call completes, so the bytes are in place before
        # any rank can read them.

        # Physical latent versions are staged here and committed with the output
        # group.
        self.input_latent_params: LatentParams | None = None
        self.latent_staging: LatentStaging | None = None
        self.latent_imported = False
        self.latent_params: LatentParams | None = None
        self.latent_expected_generation = 0
        self.latent_expected_step = 0
        self.latent_generation = 0
        self.latent_step = 0
        self.latent_release = False

        # Verification keeps only the host metadata needed to check acceptance.
        self.draft_tokens: tuple[int, ...] | None = None
        self.terminal_prefix: int | None = None
        self.base_logical_position = 0
        self.base_rng_counter = 0
        self.base_kv_visible = 0
        self.initialized_kv = 0

        self._buffer: OutputBuffer | None = buffer
        self._row = int(row)
        self._generation = int(buffer.generation)
        self._completion_timing: tuple[int, int, int, int] | None = None
        self._observed = False

        self.completion_tasks: tuple[HostTask, ...] = ()
        # Host work that produces this call's products publishes them once
        # the tasks complete, through this hook, on the worker thread.
        self.finish: Callable[[tuple[object, ...]], None] | None = None
        self._reports_output = True
        self._media_output: MediaOutput | None = None
        self.value: RequestOutput | None = None

    def release_execution_references(self) -> None:
        """Drop borrowed views after stores commit/abandon writes and fence.

        reads.

        Host completion may outlive every product, so retaining the request tail
        must not keep these numerical allocations alive after their owners free
        them. The caller completes store handoff before invoking this method.
        """
        self.writes.clear()
        self.tensor_exports.clear()
        self.cache_exports.clear()
        self.latent_exports.clear()
        self.exported_locators.clear()
        self.cache_publication = None
        self.cache_installation = None

        self.input_latent_params = None
        self.latent_staging = None
        self.latent_imported = False

        self.predicate = None
        self.token_write = None
        self.transition_write = None
        self.completion_write = None
        self.producer_write = None
        self.sampled = None

        self.runtime_penalty_base = None
        self.runtime_prompt_logits = None
        self.runtime_cache_length = None
        self.runtime_logical_position = 0
        self.runtime_sampling_position = 0

    @property
    def request_key(self) -> RequestKey:
        return self.call.request_key

    @property
    def call_id(self) -> CallId:
        return self.call.call_id

    @property
    def kind(self) -> CallKind:
        return self.call.kind

    def ready(self) -> bool:
        """Query host output readiness without changing request acceptance."""
        if self.value is not None:
            return True
        if self._buffer is None or not self._buffer.ready():
            return False
        for task in self.completion_tasks:
            if not task.ready():
                return False
        return True

    def materialize(self) -> RequestOutput:
        """Resolve output fields once.

        A call that did not run keeps the request's committed coordinates, so
        its completion reports where the request still stands.
        """
        if self.value is not None:
            return self.value
        if not self.ready():
            raise RuntimeError("completion was resolved before query-ready")
        accepted_parent = self.request.accepted_progress

        status = self.status
        error_code = self.error_code
        runtime = self.progress
        tokens = self.committed_tokens
        suppressed = status is CallStatus.PREDICATED
        if not suppressed:
            try:
                results = tuple(task.result() for task in self.completion_tasks)
                finish, self.finish = self.finish, None
                if finish is not None:
                    finish(results)
                    results = ()
                for result in results:
                    if isinstance(result, bytes) and self._reports_output:
                        result = MediaOutput(
                            handle=PosixShmArtifact(
                                name=publish_media_bytes(result)
                            ),
                            bytes=len(result),
                        )
                    if isinstance(result, MediaOutput):
                        if self._media_output is not None:
                            raise RuntimeError(
                                "completion produced more than one media output"
                            )
                        self._media_output = result

                if self._buffer is None:
                    raise RuntimeError(
                        "completion lost its pinned output buffer"
                    )
                if self.logprob_range is not None:
                    self.logprobs = self._buffer.logprob_values(
                        self.logprob_range
                    )
                self.prompt_logprobs = tuple(
                    self._buffer.logprob_values(span)[1]
                    for span in self.prompt_logprob_ranges
                )
            except Exception:
                logger.exception(
                    "completion materialization failed: request=%s "
                    "call=%s computation=%s",
                    self.request_key,
                    self.call_id,
                    self.kind,
                )
                status = CallStatus.ERROR
                error_code = ErrorCode.COMPUTE_ERROR
                suppressed = True
            else:
                try:
                    tokens = sampled_tokens(self)
                except _PredicatedCall:
                    status = CallStatus.PREDICATED
                    suppressed = True
                except _InvalidSamplingDistribution:
                    status = CallStatus.ERROR
                    error_code = ErrorCode.INVALID_CALL
                    suppressed = True
                else:
                    if self.sampling_range is not None:
                        if runtime is None:
                            raise RuntimeError(
                                "sampling output has no request progress"
                            )
                        if self.draft_tokens is not None:
                            accepted = len(tokens)
                            visible = self.base_kv_visible + accepted
                            if (
                                accepted > len(self.draft_tokens) + 1
                                or runtime.kv_computed_len
                                != self.initialized_kv
                                or visible > runtime.kv_computed_len
                            ):
                                raise RuntimeError(
                                    "speculative acceptance exceeds "
                                    "initialized KV state"
                                )
                            # Rejected drafts remain initialized but invisible.
                            # Resolve logical, RNG and visible KV coordinates
                            # together once.
                            runtime = replace(
                                runtime,
                                logical_position=self.base_logical_position
                                + accepted,
                                rng_counter=self.base_rng_counter + accepted,
                                kv_visible_len=visible,
                            )
        if status is CallStatus.PREDICATED:
            runtime = accepted_parent
            error_code = None
        if suppressed:
            tokens = ()
        scores = (
            None if suppressed or not self._reports_output else self.logprobs
        )

        buffer = self._buffer
        if buffer is None:
            raise RuntimeError("completion lost its pinned output buffer")
        buffer.observe(self._row, self._generation)
        self._completion_timing = buffer.timing()
        self._observed = True
        self._buffer = None

        timing = TimingCounters(
            queued_us=self._completion_timing[0],
            device_us=self._completion_timing[1],
            copy_us=self._completion_timing[2],
            host_us=self._completion_timing[3],
        )
        concrete = RequestOutput(
            request_key=self.request_key,
            call_id=self.call_id,
            status=status,
            product_generations=() if suppressed else self.product_generations,
            error_code=error_code,
            timing_counters=timing,
            kind=self.kind,
            position=0 if runtime is None else int(runtime.logical_position),
            kv_visible_len=0
            if runtime is None
            else int(runtime.kv_visible_len),
            kv_computed_len=0
            if runtime is None
            else int(runtime.kv_computed_len),
            num_completed_steps=0
            if runtime is None
            else int(runtime.flow_step),
            committed_tokens=tokens,
            sampled_logprob=None if scores is None else scores[0],
            top_logprobs=() if scores is None else scores[1],
            prompt_logprobs=(
                ()
                if suppressed or not self._reports_output
                else self.prompt_logprobs
            ),
            finish_flags=FinishFlags() if suppressed else self.finish_flags,
            media_output=self._media_output,
            kv_output=None if suppressed else self.kv_output,
        )
        # Ordinary successful calls accept the immutable projection itself;
        # failures keep their predecessor and verification uses its resolved
        # span.
        self.accepted_progress = (
            accepted_parent
            if status in (CallStatus.PREDICATED, CallStatus.ERROR)
            else runtime
        )

        concrete.validate()
        self.value = concrete
        self.completion_tasks = ()
        return concrete

    def abandon(self) -> None:
        """Stop result delivery while actual CPU and GPU readers retain their.

        buffers.
        """
        tasks, self.completion_tasks = self.completion_tasks, ()
        self.finish = None
        actions = [task.abandon for task in tasks]

        buffer, self._buffer = self._buffer, None
        if buffer is not None and not self._observed:
            self._observed = True
            actions.append(partial(buffer.discard, self._row, self._generation))

        close_resources(*actions)
