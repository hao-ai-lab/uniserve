"""A host rank's lane, for tests that run media tasks on it."""

import pytest

from uniserve_worker.execution.host import HostLane


@pytest.fixture
def lane():
    """A host rank is one codec slot: one task at a time on one thread."""
    lane = HostLane(max_inflight=1, workers=1)
    try:
        yield lane
    finally:
        lane.close()


def run_on_lane(lane: HostLane, schedule):
    """Run one task on a reserved slot of ``lane`` and return its result."""
    task = schedule(lane.reserve())
    task.submit_if_ready()
    return task.promise.result(timeout=30)
