"""Worker capability conformance across the Python wire boundary."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker import batch
from uniserve_worker.batch import SamplingOwnership
from uniserve_worker.bootstrap.capacity import operation_window
from uniserve_worker.server.app import dispatch

pytestmark = pytest.mark.contract


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def test_worker_capability_wire_reports_schedulable_work_and_bounds() -> None:
    worker = execution_worker()
    wire = dispatch(worker, {"kind": "get_capabilities"})["capabilities"]

    assert wire["num_layers"] > 0
    assert wire["num_kv_heads"] > 0
    assert wire["head_dim"] > 0
    assert wire["max_batch_operations"] > 0
    assert wire["max_unresolved_window"] == operation_window(
        wire["pipeline_depth"], wire["max_batch_operations"]
    )
    assert wire["incremental_kv_publication"] is True
    assert wire["sampling_ownership"] == SamplingOwnership.DESIGNATED_RANK.value
    assert _is_digest(wire["protocol_layout_digest"])
    assert wire["protocol_layout_digest"] == batch.protocol_layout_digest()
