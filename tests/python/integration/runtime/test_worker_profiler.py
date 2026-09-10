"""Standard profiler artifacts exposed by the worker diagnostic configuration."""

import gzip
import json

import pytest
import torch

from uniserve_worker.profiling import WorkerProfiler

pytestmark = pytest.mark.integration


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
    captured = {event["name"] for event in events if event["name"].startswith("worker-step-")}
    assert captured == {"worker-step-2", "worker-step-3"}
