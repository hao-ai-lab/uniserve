"""Arrival process + concurrency gating, mirroring ``refs/sglang`` serving.py.

The timed region matches SGLang's ``benchmark()``:

1. ``warmup_requests`` warmup requests are sent first with the configured workload and
   their results discarded. If every warmup fails we raise -- the run is misconfigured.
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
from dataclasses import dataclass
from typing import Any

import numpy as np

Row = dict[str, Any]


@dataclass(frozen=True)
class ClosedLoopResult:
    """Measured outputs and occupancy evidence from one steady-state run."""

    outputs: tuple[Any, ...]
    guard_outputs: tuple[Any, ...]
    duration_s: float
    target_concurrency: int
    warmup_completions: int
    measured_requests: int
    maximum_inflight: int
    minimum_inflight: int
    mean_inflight: float
    target_occupancy_fraction: float


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
    warmup_rows: list[Row] | None = None,
) -> tuple[list[Any], float]:
    """Run the warmup + timed region; return ``(outputs, dur_s)``."""
    if not rows:
        return [], 0.0

    if warmup_requests > 0 and warmup_submit is not None:
        selected_warmups = warmup_rows or [rows[0] for _ in range(warmup_requests)]
        if len(selected_warmups) != warmup_requests:
            raise ValueError(
                f"warmup workload has {len(selected_warmups)} rows; expected {warmup_requests}"
            )
        warmup_tasks: list[asyncio.Task[Any]] = [
            asyncio.create_task(warmup_submit(row)) for row in selected_warmups
        ]
        warmup_outputs = await asyncio.gather(*warmup_tasks)
        if not any(getattr(output, "success", False) for output in warmup_outputs):
            first = warmup_outputs[0] if warmup_outputs else None
            classifier = getattr(first, "classifier", None)
            status = getattr(first, "status_code", None)
            error = getattr(first, "error", None)
            raise RuntimeError(
                "Warmup failed -- check the benchmark arguments and server. "
                f"First classifier: {classifier}; status: {status}; error: {error}"
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
        scheduled = dict(row)
        scheduled["_harness_scheduled_time"] = time.perf_counter()
        tasks.append(asyncio.create_task(limited(scheduled)))
    outputs = await asyncio.gather(*tasks)
    dur_s = time.perf_counter() - benchmark_start_time
    return list(outputs), dur_s


async def run_closed_loop(
    measured_rows: list[Row],
    guard_rows: list[Row],
    *,
    concurrency: int,
    submit: Callable[[Row], Coroutine[Any, Any, Any]],
    ramp_interval_s: float = 0.25,
    warmup_completions: int | None = None,
) -> ClosedLoopResult:
    """Measure replacement-driven service at fixed logical concurrency.

    Slots enter gradually. Once all slots are occupied, guard requests complete
    for one configured churn interval. Subsequent slot completions launch the
    measured rows, and completed measured rows are replaced by tail guards.
    The window ends only after every measured request completes, so neither the
    entry prefill burst nor a draining tail changes its concurrency.
    """

    if concurrency < 1:
        raise ValueError("closed-loop concurrency must be positive")
    if not measured_rows:
        raise ValueError("closed-loop measurement requires rows")
    if not guard_rows:
        raise ValueError("closed-loop measurement requires guard rows")
    if ramp_interval_s < 0:
        raise ValueError("closed-loop ramp interval cannot be negative")
    required_warmups = concurrency if warmup_completions is None else warmup_completions
    if required_warmups < concurrency:
        raise ValueError("closed-loop warmup must complete at least one request per slot")

    lock = asyncio.Lock()
    all_active = asyncio.Event()
    measured_done = asyncio.Event()
    phase = "ramp"
    stop = False
    inflight = 0
    maximum_inflight = 0
    completed_warmups = 0
    next_measured = 0
    next_guard = 0
    measured_outputs: list[tuple[int, Any]] = []
    guard_outputs: list[Any] = []
    occupancy: list[tuple[float, int]] = []
    measurement_start: float | None = None
    measurement_end: float | None = None

    def stamp(now: float) -> None:
        if occupancy and occupancy[-1][1] == inflight:
            return
        occupancy.append((now, inflight))

    async def acquire(slot: int) -> tuple[Row, bool, int] | None:
        nonlocal inflight, maximum_inflight, next_measured, next_guard, measurement_start
        async with lock:
            if stop:
                return None
            measured = phase == "measure" and next_measured < len(measured_rows)
            if measured:
                row = dict(measured_rows[next_measured])
                next_measured += 1
                sequence = next_measured
                row["_harness_stage"] = "measure"
            else:
                row = dict(guard_rows[next_guard % len(guard_rows)])
                next_guard += 1
                sequence = next_guard
                row["id"] = f"guard-slot-{slot:02d}-{sequence:06d}"
                row["_harness_stage"] = "guard"
            row["_harness_slot"] = slot
            row["_harness_sequence"] = sequence
            scheduled = time.perf_counter()
            row["_harness_scheduled_time"] = scheduled
            inflight += 1
            maximum_inflight = max(maximum_inflight, inflight)
            stamp(scheduled)
            if inflight == concurrency:
                all_active.set()
            if measured and measurement_start is None:
                measurement_start = scheduled
            return row, measured, sequence

    async def complete(output: Any, measured: bool, sequence: int) -> None:
        nonlocal inflight, completed_warmups, phase, stop, measurement_end
        async with lock:
            completed = time.perf_counter()
            inflight -= 1
            stamp(completed)
            if measured:
                measured_outputs.append((sequence, output))
                if len(measured_outputs) == len(measured_rows):
                    measurement_end = completed
                    stop = True
                    measured_done.set()
            else:
                guard_outputs.append(output)
                if phase == "warmup":
                    completed_warmups += 1
                    if completed_warmups == required_warmups:
                        phase = "measure"

    async def slot_loop(slot: int) -> None:
        while True:
            assignment = await acquire(slot)
            if assignment is None:
                return
            row, measured, sequence = assignment
            output = await submit(row)
            await complete(output, measured, sequence)

    slots: list[asyncio.Task[None]] = []
    for slot in range(concurrency):
        slots.append(asyncio.create_task(slot_loop(slot)))
        if ramp_interval_s and slot + 1 < concurrency:
            await asyncio.sleep(ramp_interval_s)

    await all_active.wait()
    async with lock:
        phase = "warmup"
    await measured_done.wait()
    for task in slots:
        if not task.done():
            task.cancel()
    await asyncio.gather(*slots, return_exceptions=True)

    if measurement_start is None or measurement_end is None:
        raise RuntimeError("closed-loop measurement did not establish a complete window")
    duration = measurement_end - measurement_start
    if duration <= 0:
        raise RuntimeError("closed-loop measurement window is not positive")

    weighted = 0.0
    target_duration = 0.0
    observed: list[int] = []
    events = [
        (max(moment, measurement_start), count)
        for moment, count in occupancy
        if moment <= measurement_end
    ]
    state = concurrency
    cursor = measurement_start
    for moment, count in events:
        if moment < cursor:
            state = count
            continue
        span = moment - cursor
        if span > 0:
            weighted += span * state
            target_duration += span if state == concurrency else 0.0
            observed.append(state)
        cursor = moment
        state = count
    if cursor < measurement_end:
        span = measurement_end - cursor
        weighted += span * state
        target_duration += span if state == concurrency else 0.0
        observed.append(state)

    return ClosedLoopResult(
        outputs=tuple(output for _, output in sorted(measured_outputs)),
        guard_outputs=tuple(guard_outputs),
        duration_s=duration,
        target_concurrency=concurrency,
        warmup_completions=completed_warmups,
        measured_requests=len(measured_outputs),
        maximum_inflight=maximum_inflight,
        minimum_inflight=min(observed, default=concurrency),
        mean_inflight=weighted / duration,
        target_occupancy_fraction=target_duration / duration,
    )
