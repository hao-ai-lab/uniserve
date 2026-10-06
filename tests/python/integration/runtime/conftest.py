"""Runtime integration resources."""

import pytest

from tests.python.fixtures.depth_one import reset_scheduler
from tests.python.fixtures.worker_ipc import worker_channel  # noqa: F401


@pytest.fixture(autouse=True)
def scheduler_state():
    """Scheduler bookkeeping belongs to one request scenario."""
    reset_scheduler()
    yield
    reset_scheduler()
