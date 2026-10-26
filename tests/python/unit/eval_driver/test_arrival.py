"""The declared concurrency limit applies to warmup and measured submissions."""

import asyncio
from types import SimpleNamespace

import pytest

from uniserve_eval.load.arrival import run_load
from uniserve_eval.types import Example, LoadConfig, RequestRecord

pytestmark = pytest.mark.unit


def test_warmup_and_measurement_share_the_submission_concurrency_limit():
    active = {"warmup": 0, "measured": 0}
    peak = dict(active)

    async def submit(row, scheduled):
        record = RequestRecord(request_id="row", task="video")
        record.begin(endpoint="/test", scheduled_time=scheduled)
        phase = "warmup" if scheduled is None else "measured"
        active[phase] += 1
        peak[phase] = max(peak[phase], active[phase])
        await asyncio.sleep(0)
        active[phase] -= 1
        record.close_now()
        record.mark_success()
        return record

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
        record = RequestRecord(
            request_id="row", task="video", start_time=clock[0]
        )
        clock[0] += 120.0 if scheduled is None else 7.0
        record.close_now()
        record.mark_success()
        # Returning later must not extend the transport measurement window.
        clock[0] += 20.0
        return record

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
        await real_sleep(0)
        clock[0] += delay
        await real_sleep(0)

    # Simulated time: every sleep advances the clock at once, and every
    # inter-arrival gap is 0.25 s.
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("asyncio.sleep", advance)
    monkeypatch.setattr("numpy.random.exponential", lambda scale: 0.25)

    async def submit(row, scheduled):
        record = RequestRecord(
            request_id="row", task="video", start_time=clock[0]
        )
        record.close_now()
        record.mark_success()
        return record

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


def test_steady_window_measures_the_requests_that_complete_inside_it(
    monkeypatch,
):
    """One client, 5 s requests: the window [12, 32) holds four completions."""
    clock = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    order = []

    async def submit(row, scheduled):
        record = RequestRecord(
            request_id=row.id, task="text", start_time=clock[0]
        )
        order.append(row.id)
        await asyncio.sleep(0)
        clock[0] += 5.0
        record.close_now()
        record.mark_success()
        return record

    rows = [Example(id=name, prompt="") for name in "abc"]
    result = asyncio.run(
        run_load(
            rows,
            request_rate=float("inf"),
            max_concurrency=1,
            submit=submit,
            warmup_requests=0,
            window_s=20.0,
            settle_s=12.0,
        )
    )

    # Requests end at 5, 10, ..., 35 s and the rows cycle a, b, c, a, ...;
    # the client stops submitting once the window closes at 32 s.
    assert order == ["a", "b", "c", "a", "b", "c", "a"]
    assert [output.request_id for output in result.outputs] == [
        "c",
        "a",
        "b",
        "c",
    ]
    assert [output.final_event_time for output in result.outputs] == [
        15.0,
        20.0,
        25.0,
        30.0,
    ]
    assert len(result.excluded_outputs) == 3
    assert result.duration_s == 20.0


def test_steady_window_keeps_every_client_in_flight():
    in_flight = [0]
    peak = [0]

    async def submit(row, scheduled):
        record = RequestRecord(request_id="row", task="text")
        record.begin(endpoint="/test", scheduled_time=scheduled)
        in_flight[0] += 1
        peak[0] = max(peak[0], in_flight[0])
        await asyncio.sleep(0.005)
        in_flight[0] -= 1
        record.close_now()
        record.mark_success()
        return record

    result = asyncio.run(
        run_load(
            [Example(id="row", prompt="")],
            request_rate=float("inf"),
            max_concurrency=3,
            submit=submit,
            warmup_requests=0,
            window_s=0.05,
        )
    )
    assert peak[0] == 3
    assert result.outputs


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"window_s": 60.0}, "requires max_concurrency"),
        (
            {"window_s": 60.0, "max_concurrency": 4, "request_rate": 2.0},
            "infinite request_rate",
        ),
        ({"settle_s": 30.0}, "requires a steady-state window_s"),
        ({"window_s": 0.0, "max_concurrency": 4}, "finite and positive"),
    ],
)
def test_load_config_rejects_an_incomplete_steady_state_load(fields, message):
    with pytest.raises(ValueError, match=message):
        LoadConfig(**fields)
