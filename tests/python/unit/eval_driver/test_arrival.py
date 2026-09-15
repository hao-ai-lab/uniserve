"""The declared concurrency limit applies to warmup and measured submissions."""

import asyncio
import subprocess
from types import SimpleNamespace

from uniserve_eval.load.arrival import run_load
from uniserve_eval.nsys import NsysCapture


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


def test_measured_duration_excludes_external_profiler_control(
    monkeypatch, tmp_path
):
    clock = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nsys")

    def run_profiler(command, **kwargs):
        clock[0] += 120.0
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr("subprocess.run", run_profiler)

    async def submit(row, scheduled):
        clock[0] += 7.0
        return SimpleNamespace(success=True)

    result = asyncio.run(
        run_load(
            [SimpleNamespace()],
            request_rate=float("inf"),
            max_concurrency=1,
            submit=submit,
            warmup_requests=1,
            measurement=NsysCapture("duration", tmp_path),
        )
    )
    assert result.duration_s == 7.0
