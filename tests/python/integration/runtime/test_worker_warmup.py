"""Public startup qualification behavior."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker

pytestmark = pytest.mark.integration


def test_warmup_is_a_safe_noop_off_cuda() -> None:
    worker = execution_worker()
    worker.warmup()
    worker.warmup()
