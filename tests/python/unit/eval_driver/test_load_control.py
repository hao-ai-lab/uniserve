from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from uniserve_eval.harness.core.arrival import run_closed_loop

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Output:
    request_id: str
    success: bool = True


def test_closed_loop_measures_after_full_churn_and_holds_target_occupancy() -> None:
    active = 0
    maximum = 0

    async def submit(row: dict[str, object]) -> _Output:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.002 + 0.001 * int(row["_harness_slot"]))
        active -= 1
        return _Output(str(row["id"]))

    measured = [{"id": f"measured-{index}"} for index in range(7)]
    guards = [{"id": f"source-{index}"} for index in range(9)]
    result = asyncio.run(
        run_closed_loop(
            measured,
            guards,
            concurrency=3,
            submit=submit,
            ramp_interval_s=0.0,
            warmup_completions=3,
        )
    )

    assert tuple(output.request_id for output in result.outputs) == tuple(
        f"measured-{index}" for index in range(7)
    )
    assert result.warmup_completions == 3
    assert result.measured_requests == 7
    assert result.maximum_inflight == maximum == 3
    assert result.minimum_inflight >= 2
    assert result.mean_inflight > 2.9
    assert result.target_occupancy_fraction > 0.9


@pytest.mark.parametrize("concurrency", [0, -1])
def test_closed_loop_rejects_nonpositive_concurrency(concurrency: int) -> None:
    async def submit(row: dict[str, object]) -> _Output:
        return _Output(str(row["id"]))

    with pytest.raises(ValueError, match="concurrency must be positive"):
        asyncio.run(
            run_closed_loop(
                [{"id": "measured"}],
                [{"id": "guard"}],
                concurrency=concurrency,
                submit=submit,
            )
        )
