"""Executes immediate or Poisson arrivals.

``run_point`` drives one benchmark point through ``run_load`` with a
``submit`` coroutine that sends one request. This module owns only
arrival timing, the client-side concurrency limit, and the measured window;
request construction, transport, and metrics live elsewhere. Submission
records supply monotonic start and terminal timestamps for the load window.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from ..types import Example, RequestRecord

T = TypeVar("T", bound=RequestRecord)
# ``submit(example, scheduled)`` sends one request. ``scheduled`` is the
# ``time.perf_counter()`` arrival time for measured requests and ``None`` for
# warmup requests.
Submit = Callable[[Example, float | None], Coroutine[Any, Any, T]]


@dataclass(frozen=True)
class LoadResult:
    """Contains warmup outputs, measured outputs, and the measured window.

    ``window_start`` and ``window_end`` are wall-clock ``time.time()``
    seconds bounding the measured window, for aligning it with telemetry
    sampled on that clock; ``duration_s`` is the same window measured on
    the monotonic clock. ``excluded_outputs`` holds a steady-state load's
    requests that completed outside its window.
    """

    warmup_outputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    duration_s: float
    window_start: float = 0.0
    window_end: float = 0.0
    excluded_outputs: tuple[Any, ...] = ()


class WarmupFailure(RuntimeError):  # noqa: N818  # deliberate taxonomy name
    """Reports outputs from a warmup batch containing a failed request."""

    def __init__(self, outputs: list[Any]) -> None:
        """Capture warmup outputs and summarize the first failure."""
        self.outputs = tuple(outputs)
        first = next(
            (
                output
                for output in outputs
                if not getattr(output, "success", False)
            ),
            None,
        )
        super().__init__(
            "Warmup failed -- check the benchmark arguments and server. "
            f"First classifier: {getattr(first, 'classifier', None)}; "
            f"status: {getattr(first, 'status_code', None)}; "
            f"error: {getattr(first, 'error', None)}"
        )


async def get_request(
    rows: list[Example],
    request_rate: float,
) -> AsyncGenerator[Example, None]:
    """Yield rows with exponential inter-arrival delays at a finite rate.

    An infinite rate yields every row without delay. Otherwise the first row
    is yielded immediately and every later row after an exponential delay
    with mean ``1 / request_rate`` seconds, so ``len(rows) - 1`` delays are
    drawn and the generator finishes at the final arrival. Delays draw from
    the process-global NumPy generator, which ``run_point`` seeds with the
    load seed.
    """
    for index, row in enumerate(rows):
        # A delay separates consecutive arrivals only; none follows the last
        # row, whose arrival ends the schedule.
        if index > 0 and request_rate != float("inf"):
            interval = float(np.random.exponential(1.0 / request_rate))
            await asyncio.sleep(interval)
        yield row


async def run_load(
    rows: list[Example],
    *,
    request_rate: float,
    max_concurrency: int | None,
    submit: Submit[T],
    warmup_requests: int = 1,
    warmup_rows: list[Example] | None = None,
    inspect_warmup: Callable[[list[T]], None] | None = None,
    before_measure: Callable[[], Awaitable[None]] | None = None,
    window_s: float | None = None,
    settle_s: float = 0.0,
) -> LoadResult:
    """Run excluded warmup and measured requests under a concurrency limit.

    Warmup sends ``rows[0]`` ``warmup_requests`` times concurrently, within
    the concurrency limit. The measured phase then submits every row once at
    its arrival time and waits for all of them.

    With ``window_s``, the measured phase is a steady-state load instead:
    ``max_concurrency`` clients each submit the next row, cycling through
    ``rows`` in order, as soon as their previous request ends. The window
    opens ``settle_s`` after these submissions start and lasts ``window_s``;
    no client submits after it closes. The measured outputs are the requests
    that completed inside the window, and ``duration_s`` is ``window_s``.

    Args:
        rows: Examples in submission order.
        request_rate: Mean arrivals per second; ``inf`` submits all rows at
            once.
        max_concurrency: Maximum in-flight warmup and measured submissions,
            or ``None`` (or ``0``) for no limit.
        submit: Coroutine that sends one example; see ``Submit``.
        warmup_requests: Number of first-row warmup submissions; ``0`` skips
            them.
        warmup_rows: Explicit excluded corpus, replacing first-row repetition.
        inspect_warmup: Inspect excluded outputs after their slots are released
            and before checking warmup success.
        before_measure: Coroutine awaited before the measured window opens;
            the runner snapshots server counters there.
        window_s: Length of a steady-state measured window in seconds, or
            ``None`` to send each row once. Requires ``max_concurrency`` and
            an infinite ``request_rate``.
        settle_s: Seconds of steady-state load before the window opens.

    Returns:
        Warmup and measured outputs in submission order, and the measured
        duration in seconds. Empty rows yield an empty result without
        submitting anything.

    Raises:
        WarmupFailure: If any warmup output lacks a truthy ``success``; no
            measured request is sent.
        BaseException: Any exception from a measured submission, or the
            cancellation of this call, after every outstanding measured
            submission has been cancelled and awaited. Outputs are not
            returned; a caller that needs the finished ones collects them
            in ``submit``.
    """
    if not rows:
        return LoadResult((), (), 0.0)

    if window_s is not None and (
        not max_concurrency or request_rate != float("inf")
    ):
        raise ValueError(
            "a steady-state window requires max_concurrency and an infinite "
            "request_rate"
        )

    # Warmup and arrival-driven measurement share the declared in-flight
    # limit. The arrival timestamp is taken before the semaphore, so time
    # queued here appears as the gap between ``scheduled`` and the moment
    # ``submit`` starts.
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited(row: Example, scheduled: float | None) -> T:
        """Submit one row within the optional concurrency semaphore."""
        if semaphore is None:
            return await submit(row, scheduled)
        async with semaphore:
            return await submit(row, scheduled)

    # Warmup exercises the same request path outside the reported duration.
    # Repeating the first row is the warmup of a load that names no corpus.
    warmup_outputs: list[T] = []
    if warmup_rows is not None:
        excluded = warmup_rows
    else:
        excluded = [rows[0]] * warmup_requests
    if excluded:
        warmup_outputs = await asyncio.gather(
            *[limited(row, None) for row in excluded]
        )
        if inspect_warmup is not None:
            inspect_warmup(warmup_outputs)
        if not all(
            getattr(output, "success", False) for output in warmup_outputs
        ):
            raise WarmupFailure(list(warmup_outputs))

    if before_measure is not None:
        await before_measure()
    # Align monotonic transport timestamps with wall-clock GPU telemetry.
    clock_offset = time.time() - time.perf_counter()

    if window_s is not None:
        assert max_concurrency is not None
        return await _steady_window(
            rows,
            clients=max_concurrency,
            submit=submit,
            window_s=window_s,
            settle_s=settle_s,
            warmup_outputs=warmup_outputs,
            clock_offset=clock_offset,
        )

    tasks: list[asyncio.Task[T]] = []
    try:
        async for row in get_request(rows, request_rate):
            tasks.append(asyncio.create_task(limited(row, time.perf_counter())))
        outputs = await asyncio.gather(*tasks)
    except BaseException:
        # ``gather`` leaves sibling tasks running when one raises. They are
        # cancelled and awaited here, so no submission outlives the failed
        # window or finishes after the caller has handled the failure.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    # Every attempted request, including errors and deadlines, contributes to
    # the window. Coroutine cleanup and subsequent inspection do not.
    # Request dispatch closes every record on each path, failures included.
    ends = []
    for output in outputs:
        if output.final_event_time is None:
            raise RuntimeError("request dispatch returned an unclosed record")
        ends.append(output.final_event_time)
    first = min(output.start_time for output in outputs)
    last = max(ends)
    return LoadResult(
        tuple(warmup_outputs),
        tuple(outputs),
        last - first,
        clock_offset + first,
        clock_offset + last,
    )


async def _steady_window(
    rows: list[Example],
    *,
    clients: int,
    submit: Submit[T],
    window_s: float,
    settle_s: float,
    warmup_outputs: list[T],
    clock_offset: float,
) -> LoadResult:
    """Keep ``clients`` requests in flight and measure a fixed window.

    Each client submits its next row as soon as its previous request ends,
    so the offered load stays at ``clients`` in-flight requests, and the
    clients share one cursor that cycles through ``rows`` in order. The
    window spans ``[open + settle_s, open + settle_s + window_s)`` on the
    monotonic clock, where ``open`` is when the clients start. A request is
    measured when its final event falls inside the window, wherever it
    started; clients stop submitting when the window closes and the requests
    still in flight then are awaited and excluded.
    """
    opened = time.perf_counter()
    start = opened + settle_s
    end = start + window_s
    cursor = 0
    outputs: list[T] = []

    async def client() -> None:
        nonlocal cursor
        while time.perf_counter() < end:
            # The event loop runs one client at a time between awaits, so the
            # shared cursor needs no lock.
            row = rows[cursor % len(rows)]
            cursor += 1
            outputs.append(await submit(row, time.perf_counter()))

    tasks = [asyncio.create_task(client()) for _ in range(clients)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    measured: list[T] = []
    excluded: list[T] = []
    for output in outputs:
        if output.final_event_time is None:
            raise RuntimeError("request dispatch returned an unclosed record")
        inside = start <= output.final_event_time < end
        (measured if inside else excluded).append(output)
    # Submission order, as the fixed-count load reports its outputs.
    measured.sort(key=lambda output: output.start_time)
    excluded.sort(key=lambda output: output.start_time)
    return LoadResult(
        tuple(warmup_outputs),
        tuple(measured),
        window_s,
        clock_offset + start,
        clock_offset + end,
        tuple(excluded),
    )
