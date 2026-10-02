"""The declared concurrency limit applies to warmup and measured submissions."""

import asyncio
from types import SimpleNamespace

import pytest

from uniserve_eval.load.arrival import run_load
from uniserve_eval.types import Example

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


def test_finite_rate_duration_ends_at_the_last_completion(monkeypatch):
    """No inter-arrival gap after the final arrival enters the window."""
    clock = [0.0]
    real_sleep = asyncio.sleep

    async def advance(delay):
        clock[0] += delay
        await real_sleep(0)

    # Simulated time: every sleep advances the clock at once, and every
    # inter-arrival gap is 0.25 s.
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("asyncio.sleep", advance)
    monkeypatch.setattr("numpy.random.exponential", lambda scale: 0.25)

    async def submit(row, scheduled):
        return SimpleNamespace(success=True)

    result = asyncio.run(
        run_load(
            [SimpleNamespace() for _ in range(3)],
            request_rate=4.0,
            max_concurrency=None,
            submit=submit,
            warmup_requests=0,
        )
    )

    # Three instantly served rows arrive at 0, 0.25, and 0.5 s into the
    # window, so it closes when the third one finishes.
    assert result.duration_s == 0.5


@pytest.mark.parametrize("concurrency", [None, 0, 2])
def test_closed_loop_sessions_wait_for_their_own_previous_response(concurrency):
    async def run():
        first_arrivals = set()
        all_started = asyncio.Event()
        completed = {"a": [], "b": []}
        rows = [
            Example(id=f"{session}{step}", prompt="", session_id=session)
            for step in range(3)
            for session in ("a", "b")
        ]

        async def submit(row, scheduled):
            session, step = row.session_id, int(row.id[1:])
            assert completed[session] == list(range(step))
            if step == 0:
                first_arrivals.add(session)
                if len(first_arrivals) == 2:
                    all_started.set()
                await all_started.wait()
            await asyncio.sleep(0)
            completed[session].append(step)
            return row.id

        result = await asyncio.wait_for(
            run_load(
                rows,
                request_rate=float("inf"),
                max_concurrency=concurrency,
                submit=submit,
                warmup_requests=0,
            ),
            timeout=5,
        )
        assert result.outputs == tuple(row.id for row in rows)
        assert completed == {"a": [0, 1, 2], "b": [0, 1, 2]}

    asyncio.run(run())


@pytest.mark.parametrize(
    "settings",
    [
        {"request_rate": 1.0},
        {"max_concurrency": 1},
        {"warmup_requests": 1},
    ],
)
def test_closed_loop_sessions_reject_incompatible_arrival_settings(settings):
    async def submit(row, scheduled):
        return SimpleNamespace(success=True)

    arguments = {
        "request_rate": float("inf"),
        "max_concurrency": 2,
        "warmup_requests": 0,
        **settings,
    }
    with pytest.raises(ValueError, match="session"):
        asyncio.run(
            run_load(
                [
                    Example(id="a0", prompt="", session_id="a"),
                    Example(id="b0", prompt="", session_id="b"),
                ],
                submit=submit,
                **arguments,
            )
        )
