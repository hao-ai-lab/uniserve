"""Standard profiler artifacts.

They are exposed by the worker diagnostic configuration.
"""

import gzip
import json
import logging

import pytest
import torch

from uniserve_worker.profiling import WorkerProfiler

pytestmark = pytest.mark.integration


def test_uncreatable_output_directory_disables_capture_once(tmp_path, caplog):
    """A trace directory that cannot be created never fails a step.

    The window fails to start once, is logged once, and every step, including
    those after the start step, still runs its body.
    """
    blocker = tmp_path / "regular-file"
    blocker.write_text("")
    profiler = WorkerProfiler.from_env(
        {
            "UNISERVE_TORCH_PROFILER_DIR": str(blocker / "traces"),
            "UNISERVE_PROFILE_ACTIVITIES": "CPU",
            "UNISERVE_PROFILE_START_STEP": "2",
        }
    )

    executed = []
    with caplog.at_level(logging.ERROR, logger="uniserve_worker.profiling"):
        for step in range(1, 5):
            with profiler.step(f"worker-step-{step}"):
                executed.append(step)
        profiler.close()

    assert executed == [1, 2, 3, 4]
    failures = [
        record
        for record in caplog.records
        if record.getMessage().startswith("failed to start")
    ]
    assert len(failures) == 1


def test_worker_profiler_exports_only_the_selected_execution_window(tmp_path):
    profiler = WorkerProfiler.from_env(
        {
            "UNISERVE_TORCH_PROFILER_DIR": str(tmp_path),
            "UNISERVE_PROFILE_ACTIVITIES": "CPU",
            "UNISERVE_PROFILE_START_STEP": "2",
            "UNISERVE_PROFILE_STEPS": "2",
        }
    )
    for step in range(1, 5):
        with profiler.step(f"worker-step-{step}"):
            torch.ones(4).add_(1)
    profiler.close()

    traces = list(tmp_path.glob("*.trace.json.gz"))
    assert len(traces) == 1
    with gzip.open(traces[0], "rt") as source:
        events = json.load(source)["traceEvents"]
    captured = {
        event["name"]
        for event in events
        if event["name"].startswith("worker-step-")
    }
    assert captured == {"worker-step-2", "worker-step-3"}


def test_bound_worker_exports_the_executed_batch_in_its_window(
    tmp_path, monkeypatch
):
    from tests.python.fixtures.depth_one import (
        ar_params,
        execution_batch,
        root_parent,
        token_call,
    )
    from tests.python.fixtures.execution_worker import execution_worker
    from tests.python.fixtures.worker_ipc import QueuedWorkerIpc
    from uniserve_worker.protocol.call import ForwardMode
    from uniserve_worker.protocol.identity import CallId

    monkeypatch.setenv("UNISERVE_TORCH_PROFILER_DIR", str(tmp_path))
    monkeypatch.setenv("UNISERVE_PROFILE_ACTIVITIES", "CPU")
    monkeypatch.setenv("UNISERVE_PROFILE_START_STEP", "1")
    monkeypatch.setenv("UNISERVE_PROFILE_STEPS", "1")

    worker = execution_worker()
    worker.warmup()
    admission = ar_params(87, block_ids=(0,))
    call = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    batch = execution_batch(batch_id=1, admissions=(admission,), calls=(call,))

    class Endpoint(QueuedWorkerIpc):
        def respond(self, response):
            super().respond(response)
            if response.get("message_id") == 1:
                self.submit({"kind": "close", "message_id": 2})

    endpoint = Endpoint(({"kind": "submit", "batch": batch, "message_id": 1},))
    try:
        worker.bind(endpoint).run()
    finally:
        worker.close()

    assert len(endpoint.responses[0]["result"]["completions"]) == 1
    traces = list(tmp_path.glob("*.trace.json.gz"))
    assert len(traces) == 1
    with gzip.open(traces[0], "rt") as source:
        events = json.load(source)["traceEvents"]
    assert any(event["name"] == "batch:prefill:model" for event in events)
