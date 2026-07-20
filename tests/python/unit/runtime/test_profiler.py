from __future__ import annotations

import pytest

from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.server.app import WorkerServer
from uniserve_worker.server.profiler import WorkerProfiler
from uniserve_worker.server.stub import StubWorker

pytestmark = pytest.mark.unit


_PROFILER_ENVS = (
    "UNISERVE_PROFILE_DIR",
    "UNISERVE_TORCH_PROFILER_DIR",
    "UNISERVE_PROFILE_ACTIVITIES",
    "UNISERVE_PROFILE_START_STEP",
    "UNISERVE_PROFILE_STEPS",
    "UNISERVE_PROFILE_PREFIX",
    "UNISERVE_PROFILE_WITH_STACK",
    "UNISERVE_PROFILE_RECORD_SHAPES",
    "UNISERVE_PROFILE_NVTX",
    "UNISERVE_NVTX",
    "UNISERVE_CUDA_PROFILER",
)


def _clear_profiler_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROFILER_ENVS:
        monkeypatch.delenv(name, raising=False)


def _execute_req() -> dict:
    return {
        "kind": "execute",
        "batch": seal_batch(
            1,
            [
                {
                    "req_id": 1,
                    "kind": "prefill_und",
                    "modality": "und",
                    "pos_range": [0, 2],
                    "token_ids": [11, 12],
                }
            ],
            new_reqs=[{"req_id": 1}],
        ),
    }


def test_worker_profiler_is_disabled_without_output_dir(monkeypatch):
    _clear_profiler_env(monkeypatch)

    profiler = WorkerProfiler.from_env()

    assert profiler.enabled is False
    with profiler.step("unit.noop"):
        pass


def test_worker_server_cpu_profiler_exports_execute_trace(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    _clear_profiler_env(monkeypatch)
    monkeypatch.setenv("UNISERVE_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("UNISERVE_PROFILE_ACTIVITIES", "CPU")
    monkeypatch.setenv("UNISERVE_PROFILE_STEPS", "1")
    monkeypatch.setenv("UNISERVE_PROFILE_PREFIX", "unit-worker")

    response = WorkerServer(
        StubWorker(block_size=256),
        ipc_endpoint=None,
    ).handle(_execute_req())

    assert response["kind"] == "result"
    traces = list(tmp_path.glob("unit-worker-*.trace.json.gz"))
    summaries = list(tmp_path.glob("unit-worker-*.summary.txt"))
    assert traces
    assert summaries
    assert "uniserve.worker.execute" in summaries[0].read_text(encoding="utf-8")
