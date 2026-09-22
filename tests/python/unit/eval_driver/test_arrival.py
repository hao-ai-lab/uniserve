"""The declared concurrency limit applies to warmup and measured submissions."""

import asyncio
from types import SimpleNamespace

import pytest

from uniserve_eval.load.arrival import run_load

pytestmark = pytest.mark.unit


def test_warmup_and_measurement_share_the_submission_concurrency_limit():
    active = {"warmup": 0, "measured": 0}
    peak = dict(active)

    async def submit(row, scheduled):
        phase = "warmup" if scheduled is None else "measured"
        active[phase] += 1
        peak[phase] = max(peak[phase], active[phase])
        await asyncio.sleep(0)
        active[phase] -= 1
        return SimpleNamespace(success=True)

    result = asyncio.run(
        run_load(
            [SimpleNamespace(), SimpleNamespace()],
            request_rate=float("inf"),
            max_concurrency=1,
            submit=submit,
            warmup_requests=2,
        )
    )
    assert peak == {"warmup": 1, "measured": 1}
    assert len(result.warmup_outputs) == len(result.outputs) == 2


def test_measured_duration_excludes_warmup(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])

    async def submit(row, scheduled):
        clock[0] += 120.0 if scheduled is None else 7.0
        return SimpleNamespace(success=True)

    result = asyncio.run(
        run_load(
            [SimpleNamespace()],
            request_rate=float("inf"),
            max_concurrency=1,
            submit=submit,
            warmup_requests=1,
        )
    )
    assert result.duration_s == 7.0
