"""Arrival process + concurrency gating, mirroring ``refs/sglang`` serving.py.

The timed region matches SGLang's ``benchmark()``:

1. ``warmup_requests`` warmup requests are sent first (output capped) and their
   results discarded. If every warmup fails we raise -- the run is misconfigured.
2. ``await asyncio.sleep(1.0)`` -- the same fixed settle before timing starts.
3. ``benchmark_start_time = perf_counter()``.
4. Requests arrive via ``get_request`` (Poisson when ``request_rate`` is finite,
   all-at-once when ``request_rate == inf``) and are dispatched as fire-and-forget
   tasks, each optionally gated by a ``Semaphore(max_concurrency)`` (the
   vLLM/SGLang concurrency cap).
5. ``dur_s = perf_counter() - benchmark_start_time`` after every task completes.

Determinism: arrival intervals use ``np.random.exponential`` which the caller
seeds with ``np.random.seed(seed)`` (default 42), exactly like SGLang.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any

import numpy as np

Row = dict[str, Any]


async def get_request(
    rows: list[Row],
    request_rate: float,
) -> AsyncGenerator[Row, None]:
    """Yield rows back-to-back (``request_rate == inf``) or Poisson-paced."""
    for row in rows:
        yield row
        if request_rate == float("inf"):
            continue
        interval = float(np.random.exponential(1.0 / request_rate))
        await asyncio.sleep(interval)


async def run_load(
    rows: list[Row],
    *,
    request_rate: float,
    max_concurrency: int | None,
    submit: Callable[[Row], Coroutine[Any, Any, Any]],
    warmup_submit: Callable[[Row], Coroutine[Any, Any, Any]] | None = None,
    warmup_requests: int = 1,
) -> tuple[list[Any], float]:
    """Run the warmup + timed region; return ``(outputs, dur_s)``."""
    if not rows:
        return [], 0.0

    if warmup_requests > 0 and warmup_submit is not None:
        warmup_tasks: list[asyncio.Task[Any]] = [
            asyncio.create_task(warmup_submit(rows[0])) for _ in range(warmup_requests)
        ]
        warmup_outputs = await asyncio.gather(*warmup_tasks)
        if not any(getattr(output, "success", False) for output in warmup_outputs):
            first = warmup_outputs[0] if warmup_outputs else None
            error = getattr(first, "error", None)
            raise RuntimeError(
                "Warmup failed -- check the benchmark arguments and server. "
                f"First error: {error}"
            )

    await asyncio.sleep(1.0)

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited(row: Row) -> Any:
        if semaphore is None:
            return await submit(row)
        async with semaphore:
            return await submit(row)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task[Any]] = []
    async for row in get_request(rows, request_rate):
        tasks.append(asyncio.create_task(limited(row)))
    outputs = await asyncio.gather(*tasks)
    dur_s = time.perf_counter() - benchmark_start_time
    return list(outputs), dur_s
