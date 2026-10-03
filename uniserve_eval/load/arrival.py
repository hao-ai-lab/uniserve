"""Executes immediate, Poisson, or closed-loop session arrivals.

``run_point`` drives one benchmark point through ``run_load`` with a
``submit`` coroutine that builds and sends one request. This module owns only
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
    warmup_rows: list[Example] | None = None,
    inspect_warmup: Callable[[list[T]], None] | None = None,
    before_measure: Callable[[], Awaitable[None]] | None = None,
) -> LoadResult:
    """Run excluded warmup and measured requests under a concurrency limit.

    Warmup sends ``rows[0]`` ``warmup_requests`` times concurrently, within
    the concurrency limit. The measured phase then submits every row once at
    its arrival time and waits for all of them. When every row has a
    ``session_id``, sessions run concurrently and each awaits its previous
    response before submitting its next row. Session traces require an
    infinite request rate, zero repeated-row warmup, and, if specified, a
    concurrency equal to their session count. Initial cold requests are
    measured as part of the trace.

    Args:
        rows: Examples in submission order.
        request_rate: Mean arrivals per second; ``inf`` submits all rows at
            once, or starts all declared closed-loop sessions.
        max_concurrency: Maximum in-flight submissions, or ``None`` (or
            ``0``) for no limit.
        submit: Coroutine that sends one example; see ``Submit``.
        warmup_requests: Number of warmup submissions; ``0`` skips warmup.
        warmup_rows: Explicit excluded corpus, replacing first-row repetition.
        inspect_warmup: Inspect excluded outputs after their slots are released
            and before checking warmup success.
        before_measure: Coroutine awaited before the measured window opens;
            the runner snapshots server counters there.

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

    sessions: dict[str, list[tuple[int, Example]]] = {}
    if any(getattr(row, "session_id", None) is not None for row in rows):
        if request_rate != float("inf"):
            raise ValueError(
                "closed-loop sessions require infinite request_rate"
            )
        for index, row in enumerate(rows):
            if not row.session_id:
                raise ValueError(
                    "closed-loop sessions require every session_id"
                )
            sessions.setdefault(row.session_id, []).append((index, row))
        if max_concurrency and len(sessions) != max_concurrency:
            raise ValueError(
                "closed-loop session count must equal max_concurrency"
            )
        if warmup_requests:
            raise ValueError(
                "session traces require warmup_requests=0; "
                "include each session's initial cold request in the trace"
            )

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
    excluded = (
        warmup_rows if warmup_rows is not None else [rows[0]] * warmup_requests
    )
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

    tasks: list[asyncio.Task[T]] = []
    try:
        if sessions:

            async def run_session(
                sequence: list[tuple[int, Example]],
            ) -> list[tuple[int, T]]:
                completed = []
                for index, row in sequence:
                    # Arrival follows this client's previous completion;
                    # another client's speed cannot advance its trace.
                    output = await submit(row, time.perf_counter())
                    completed.append((index, output))
                return completed

            session_tasks = [
                asyncio.create_task(run_session(sequence))
                for sequence in sessions.values()
            ]
            try:
                completed = await asyncio.gather(*session_tasks)
            except BaseException:
                for task in session_tasks:
                    task.cancel()
                await asyncio.gather(*session_tasks, return_exceptions=True)
                raise
            ordered = sorted(
                item for sequence in completed for item in sequence
            )
            outputs = [output for _, output in ordered]
        else:
            async for row in get_request(rows, request_rate):
                tasks.append(
                    asyncio.create_task(limited(row, time.perf_counter()))
                )
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
