"""Warmup, settle, and Poisson or immediate arrival."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np

from ..types import Example

T = TypeVar("T")
Submit = Callable[[Example, float | None], Coroutine[Any, Any, T]]


class MeasurementWindow(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class LoadResult:
    warmup_outputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    duration_s: float


class WarmupFailure(RuntimeError):
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = tuple(outputs)
        first = outputs[0] if outputs else None
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
    measurement: MeasurementWindow | None = None,
) -> LoadResult:
    if not rows:
        return LoadResult((), (), 0.0)

    warmup_outputs: list[T] = []
    if warmup_requests > 0:
        warmup_outputs = await asyncio.gather(
            *[submit(rows[0], None) for _ in range(warmup_requests)]
        )
        if not all(getattr(output, "success", False) for output in warmup_outputs):
            raise WarmupFailure(list(warmup_outputs))

    await asyncio.sleep(1.0)

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited(row: Example, scheduled: float | None) -> T:
        if semaphore is None:
            return await submit(row, scheduled)
        async with semaphore:
            return await submit(row, scheduled)

    if measurement is not None:
        measurement.start()
    benchmark_start_time = time.perf_counter()
    try:
        tasks: list[asyncio.Task[T]] = []
        async for row in get_request(rows, request_rate):
            tasks.append(asyncio.create_task(limited(row, time.perf_counter())))
        outputs = await asyncio.gather(*tasks)
    finally:
        if measurement is not None:
            measurement.stop()
    return LoadResult(
        tuple(warmup_outputs),
        tuple(outputs),
        time.perf_counter() - benchmark_start_time,
    )
