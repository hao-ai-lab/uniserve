"""Pinned output allocations and their CPU/GPU retirement fences."""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Sequence
from functools import partial
from threading import RLock

import torch

from uniserve.runtime import EventPool
from uniserve.runtime.device import canonical_device
from uniserve_worker.errors import WorkerError, WorkerErrorCode, resource_error
from uniserve_worker.profiling import timing_events_enabled
from uniserve_worker.sampling.output import decode_logprobs

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
