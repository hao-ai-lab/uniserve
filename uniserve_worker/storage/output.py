"""Pinned output allocations and their CPU/GPU retirement fences.

Batch execution copies small device results (sampling columns, packed
logprob columns, call predicates, quantized image bytes) into pinned
``OutputBuffer`` leases from the worker's bounded ``OutputPool``. Sealing a
buffer records a completion event per declared CUDA device; ``read_tokens``
refuses reads until those events complete, and a ``capture_bytes`` caller
must wait for them itself. Every lease carries a process-wide generation, so
``observe`` rejects and ``discard`` ignores a completion record from an
earlier lease of the same buffer. The buffer returns to its pool only once
every row is observed or discarded (or the lease is abandoned), its events
are released to the shared ``EventPool``, and every retained CPU reader has
finished.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from functools import partial
from threading import RLock

import torch

from uniserve.runtime import EventPool
from uniserve.runtime.device import canonical_device
from uniserve_worker._uniserve_ipc import Completion
from uniserve_worker.errors import WorkerError, WorkerErrorCode, resource_error
from uniserve_worker.profiling import timing_events_enabled
from uniserve_worker.sampling.output import decode_logprobs

# Next lease generation. It spans every buffer in the process, so a lease's
# generation is never reused by another lease.
_next_buffer_generation = 1


def _invariant(message: str) -> WorkerError:
    """Construct a classified invariant error for output-buffer misuse."""
    return WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


class OutputBuffer:
    """Pinned host storage and completion events for one lane commit.

    One int64 allocation holds all captures: token captures fill it from the
    head upward in words, byte captures from the tail downward in bytes, and
    a capture that would cross the other region fails.

    Each lease reports queued, device, copy and host-observation durations.
    Device and copy durations come from CUDA timing events when the lease has
    a CUDA device and ``timing_events_enabled`` holds, from host clocks when
    the lease has no CUDA device and ``begin_device`` was called, and are
    zero otherwise.
    """

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
        "_completion",
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
        """Reserve pinned completion rows and generation-tagged CUDA copy state.

        Args:
            rows: Result rows; each must be observed or discarded before the
                buffer can retire, unless the lease is abandoned.
            token_capacity: Allocation size in int64 words, shared by token
                and byte captures.
            devices: Devices that may produce captures. Non-CUDA devices are
                dropped, and the allocation is pinned only when a CUDA device
                remains.
            event_pool: Owner of every CUDA event this buffer records.
            release_to_pool: Called once when the lease fully retires.

        Raises:
            ValueError: ``rows`` is below one.
            WorkerError: From ``resource_error`` when ``token_capacity`` is
                below ``rows``.
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

        self._completion: Completion | None = None
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
        """Begin a new lease over this persistent pinned allocation.

        The previous lease's events must already be released. The allocation
        is kept unless the lease needs more words or introduces CUDA producers
        to pageable storage, and the lease receives a new generation.
        """
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

        self._completion = None
        self._completion_registered = False
        self._released_to_pool = False

    @property
    def generation(self) -> int:
        """Identify the lease generation guarding all captures from this buffer.

        Completion records carry it; ``observe`` rejects and ``discard``
        ignores a record from another lease.
        """
        return self._generation

    def register_device(self, device: torch.device | str) -> None:
        """Verify that a CUDA producer belongs to this lease's declared devices.

        Non-CUDA devices are always accepted. Raises an invariant violation
        after the buffer is sealed or for an undeclared CUDA device.
        """
        if self._sealed:
            raise _invariant(
                "completion device was registered after its buffer was sealed"
            )
        target = canonical_device(device)
        if target.type == "cuda" and target not in self.devices:
            raise _invariant("completion work uses an undeclared CUDA device")

    def begin_device(self, device: torch.device | str) -> None:
        """Mark device execution start and record its optional timing event.

        The first call on any device stamps the host start time. With timing
        events enabled, the first call per CUDA device records a start event
        on its current stream.
        """
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
        """Mark the start of one device's device-to-host copies for timing.

        The first call stamps the host copy-start time. With timing events
        enabled, the first call per device records a producer event on its
        current stream, which ends that device's device interval and starts
        its copy interval in ``observe``. It is not a completion fence.
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
        """Copy a token tensor into the next bounded span of pinned storage.

        Values are flattened and converted to int64. A CUDA source is copied
        asynchronously, so the span is readable only through ``read_tokens``
        once the buffer is ready; a CPU source is copied synchronously.

        Returns:
            The span's ``(offset, count)`` in int64 words.

        Raises:
            WorkerError: After sealing, when the span would cross the byte
                tail, when a CUDA source is on an undeclared device, or when a
                CUDA copy would target pageable storage.
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
        """Copy uint8 values into the buffer tail and borrow their host view.

        The returned view has the shape of ``value``. The caller must wait for
        this buffer's completion before reading the view, and retain a CPU
        reader until its last asynchronous use finishes.
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
        """Record completion events for every declared device and seal captures.

        Each event is recorded on its device's current stream, so it covers
        the copies enqueued there, and schedules the event pool's completion
        wake when one is registered. Sealing twice does nothing.
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
        """Return whether all sealed device-copy events have completed.

        The first true result stamps the ready time and resolves the
        completion signal.
        """
        if not self._sealed:
            return False
        if self._ready_ns:
            return True
        if any(not bool(event.query()) for event in self._events.values()):
            return False
        self._ready_ns = time.perf_counter_ns()
        self._complete_dependents()
        return True

    def completion(self) -> Completion:
        """Expose this output lease's device fence to physical storage owners.

        The signal belongs to this lease even after the pinned buffer is reused.
        It adds no CUDA event or host/device payload allocation.
        """
        if self._completion is None:
            self._completion = Completion()
        completion = self._completion
        self._bind_completion()
        if self.ready():
            self._complete_dependents()
        return completion

    def _bind_completion(self) -> None:
        """Resolve the completion signal once sealed events finish.

        Runs once per lease after both sealing and a completion request.
        The signal resolves at once when the lease is already ready,
        its events are released, or it has no CUDA events.
        """
        completion = self._completion
        if (
            not self._sealed
            or completion is None
            or self._completion_registered
        ):
            return
        self._completion_registered = True
        if self._ready_ns or self._events_released or not self._events:
            self._resolve_completion(completion)
            return

        # Physical retirement is independent of whether a host reads the output
        # report. Retain the existing fences until the event loop observes them,
        # and bind the callback to this lease's signal across buffer reuse.
        for device in self.devices:
            self.event_pool.retain(self._events[str(device)], device)
        self.event_pool.defer_release(
            tuple(self._events.values()),
            completion,
            completed=partial(self._resolve_completion, completion),
        )

    @staticmethod
    def _resolve_completion(completion: Completion) -> None:
        if not completion.done():
            completion.set_result(None)

    def _complete_dependents(self) -> None:
        completion = self._completion
        if completion is not None:
            self._resolve_completion(completion)

    def read_tokens(self, offset: int, count: int) -> tuple[int, ...]:
        """Read a registered integer range once its producer copy completes.

        Values are cached per range. Raises an invariant violation before the
        buffer is ready or when the range leaves the captured token extent.
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
        """Decode one row, sharing parsing of its packed column with other rows.

        ``span`` is the ``(offset, count, row)`` that ``capture_logprobs``
        returned; the column is decoded once through its recorded
        ``logprob_layouts`` entry.
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
        """Mark one result row observed and return copy and observation timing.

        Returns ``(copy_us, ready_to_observed_us)`` of the lease, computed at
        its first observation. Observing the last row releases the lease's
        events. Raises an invariant violation for another lease's generation,
        an out-of-range row, a buffer that is not ready, or incomplete timing
        events.
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
        """Return the observed lease's durations in microseconds.

        The tuple is ``(queued_us, device_us, copy_us, ready_to_observed_us)``.
        Raises an invariant violation before the first ``observe``.
        """
        if self._timing is None:
            raise _invariant("completion timing was read before observation")
        return self._timing

    def discard(self, row: int, generation: int) -> None:
        """Retire one unobserved result row while preserving unfinished copies.

        A row of another lease or out of range is ignored. When the last row
        retires, the events are released at once if the buffer is ready and
        otherwise deferred to the event pool until the copies complete.
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
        """Seal and retire the entire output lease without exposing its rows.

        Event release is deferred to the event pool while copies are still
        in flight. Abandoning twice does nothing.
        """
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
        """Return all owned device-copy events to the shared event pool.

        Callers first establish readiness: ``EventPool.release`` rejects an
        event whose last reference drops before it completes.
        """
        if self._events_released or self._release_pending:
            return
        for event in self._all_events():
            self.event_pool.release(event)
        self._events_released = True
        self._return_to_pool()

    def _defer_release(self) -> None:
        """Defer buffer reuse until every outstanding copy event completes.

        The event pool calls ``events_released`` once they do.
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
        """Receive completion of an event-pool asynchronous release.

        Also resolves the completion signal and returns the buffer to its
        pool when no CPU reader remains.
        """
        self._release_pending = False
        self._events_released = True
        self._complete_dependents()
        self._return_to_pool()

    def retain_cpu_reader(self) -> Callable[[], None]:
        """Retain pinned storage until a CPU task finishes reading it.

        Returns the release callable, which the reader calls once; the buffer
        is not returned to its pool while a reader remains. Raises an
        invariant violation once the buffer has returned to its pool.
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
        """Return storage only after both device writes and CPU readers retire.

        Runs at most once per lease; a buffer without a pool owner is never
        returned.
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
    """Bounded owner of reusable pinned lane-output allocations.

    At most ``capacity`` buffers are ever allocated, and no lease may exceed
    ``max_words`` int64 words. A retired buffer is reused, grown if needed,
    by a later lease.
    """

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
        """Lease a reset or newly allocated output buffer within pool bounds.

        Raises ``resource_error`` when ``token_capacity`` exceeds
        ``max_words``, the pool is closed, or all ``capacity`` buffers are
        leased; the leased buffer's own row-count checks also apply.
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
        """Accept a released completion buffer back into the bounded free list.

        Returns after close are dropped.
        """
        with self._lock:
            if self._closed:
                return
            if buffer not in self._buffers or buffer in self._free:
                raise _invariant("output pool received an invalid lease return")
            self._free.append(buffer)

    def close(self) -> None:
        """Stop admission and retire every lease through its device fences.

        Seals and synchronizes every lease whose events are not yet released,
        then abandons every lease, so the final ``EventPool.reap`` finds each
        buffer's events complete and runs any deferred release.
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
