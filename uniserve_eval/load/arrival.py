"""Executes warmup and measured loads with immediate or Poisson arrivals."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from ..types import Example

if TYPE_CHECKING:
    from ..nsys import NsysCapture

T = TypeVar("T")
Submit = Callable[[Example, float | None], Coroutine[Any, Any, T]]


@dataclass(frozen=True)
class LoadResult:
    """Contains warmup outputs, measured outputs, and measured duration."""

    warmup_outputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    duration_s: float


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
    """Yield rows with exponential inter-arrival delays at a finite rate."""
    for row in rows:
        yield row
        if request_rate == float("inf"):
            continue
        interval = float(np.random.exponential(1.0 / request_rate))
        await asyncio.sleep(interval)


async def run_load(
    rows: list[Example],
    *,
    request_rate: float,
    max_concurrency: int | None,
    submit: Submit[T],
    warmup_requests: int = 1,
    measurement: NsysCapture | None = None,
) -> LoadResult:
    """Run warmup, settle, and measured requests under a concurrency limit."""
    if not rows:
        return LoadResult((), (), 0.0)

    # Warmup and measurement share the declared in-flight limit. Queueing at
    # this boundary leaves measured arrival timestamps unchanged.
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited(row: Example, scheduled: float | None) -> T:
        """Submit one row within the optional concurrency semaphore."""
        if semaphore is None:
            return await submit(row, scheduled)
        async with semaphore:
            return await submit(row, scheduled)

    # Warmup exercises the same request path but remains outside both the
    # profiler window and the reported duration.
    warmup_outputs: list[T] = []
    if warmup_requests > 0:
        warmup_outputs = await asyncio.gather(
            *[limited(rows[0], None) for _ in range(warmup_requests)]
        )
        if not all(
            getattr(output, "success", False) for output in warmup_outputs
        ):
            raise WarmupFailure(list(warmup_outputs))

    await asyncio.sleep(1.0)

    if measurement is not None:
        measurement.start()
    benchmark_start_time = time.perf_counter()

    try:
        tasks: list[asyncio.Task[T]] = []
        async for row in get_request(rows, request_rate):
            tasks.append(asyncio.create_task(limited(row, time.perf_counter())))
        outputs = await asyncio.gather(*tasks)
    finally:
        # Profiler report draining is outside the request completion window.
        benchmark_end_time = time.perf_counter()
        if measurement is not None:
            measurement.stop()

    # Duration covers scheduled arrival generation through completion of the
    # final measured request, matching the throughput denominator.
    return LoadResult(
        tuple(warmup_outputs),
        tuple(outputs),
        benchmark_end_time - benchmark_start_time,
    )
