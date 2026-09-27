"""Executes warmup and measured loads with immediate or Poisson arrivals.

``run_point`` drives one benchmark point through ``run_load`` with a
``submit`` coroutine that builds and sends one request. This module owns only
arrival timing, the client-side concurrency limit, and the measured window;
request construction, transport, and metrics live elsewhere. Submission
outputs are opaque here except for warmup, which reads their ``success``
attribute and, for the failure message, ``classifier``, ``status_code``, and
``error``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from ..types import Example

T = TypeVar("T")
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
    the monotonic clock.
    """

    warmup_outputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    duration_s: float
    window_start: float = 0.0
    window_end: float = 0.0


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
    before_measure: Callable[[], Awaitable[None]] | None = None,
) -> LoadResult:
    """Run warmup, settle, and measured requests under a concurrency limit.

    Warmup sends ``rows[0]`` ``warmup_requests`` times concurrently, within
    the concurrency limit. The measured phase then submits every row once at
    its arrival time and waits for all of them.

    Args:
        rows: Examples in submission order.
        request_rate: Mean arrivals per second; ``inf`` submits all rows at
            once.
        max_concurrency: Maximum in-flight submissions, or ``None`` (or
            ``0``) for no limit.
        submit: Coroutine that sends one example; see ``Submit``.
        warmup_requests: Number of warmup submissions; ``0`` skips warmup.
        before_measure: Coroutine awaited after the settle pause and before
            the measured window opens, outside its duration; the runner
            snapshots server counters there.

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

    # Warmup and measurement share the declared in-flight limit. The arrival
    # timestamp is taken before the semaphore, so time queued here appears
    # as the gap between ``scheduled`` and the moment ``submit`` starts.
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited(row: Example, scheduled: float | None) -> T:
        """Submit one row within the optional concurrency semaphore."""
        if semaphore is None:
            return await submit(row, scheduled)
        async with semaphore:
            return await submit(row, scheduled)

    # Warmup exercises the same request path outside the reported duration.
    warmup_outputs: list[T] = []
    if warmup_requests > 0:
        warmup_outputs = await asyncio.gather(
            *[limited(rows[0], None) for _ in range(warmup_requests)]
        )
        if not all(
            getattr(output, "success", False) for output in warmup_outputs
        ):
            raise WarmupFailure(list(warmup_outputs))

    # A fixed pause precedes the measured window, with or without warmup,
    # and lies outside the reported duration.
    await asyncio.sleep(1.0)
    if before_measure is not None:
        await before_measure()

    window_start = time.time()
    benchmark_start_time = time.perf_counter()

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
    benchmark_end_time = time.perf_counter()
    window_end = time.time()

    # Duration runs from the first arrival until every measured request has
    # finished. The arrival loop ends at the final arrival (see
    # ``get_request``), so the window closes at the last completion and
    # holds no inter-arrival delay beyond it. ``metrics.summarize`` divides
    # its throughputs by this duration.
    return LoadResult(
        tuple(warmup_outputs),
        tuple(outputs),
        benchmark_end_time - benchmark_start_time,
        window_start,
        window_end,
    )
