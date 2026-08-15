"""Canonical scheduler-to-worker protocol conformance at the worker server."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.server.app import WorkerServer, dispatch

pytestmark = pytest.mark.contract

_RESPONSE_FIELDS = {
    "kind",
    "call_id",
    "capabilities",
    "completion_report",
    "pressure",
    "message",
    "code",
    "retryable",
    "fatal",
    "phase",
    "route",
    "operations",
    "snapshot",
}


def test_worker_response_variants_project_the_complete_typed_schema():
    worker = execution_worker()
    capabilities = dispatch(worker, {"kind": "get_capabilities"})
    error = WorkerServer(worker, None).handle({"kind": "unknown"})

    assert set(capabilities) == _RESPONSE_FIELDS
    assert capabilities["operations"] == []
    assert set(error) == _RESPONSE_FIELDS
    assert error["kind"] == "error"
