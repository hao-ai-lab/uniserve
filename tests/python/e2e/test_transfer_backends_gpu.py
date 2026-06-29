"""Cross-process data-plane Tier-2 backends (docs/staged-workers.md §4).

Each backend is proven by moving a real tensor between two *separate* processes
(CUDA IPC and Mooncake RDMA are inherently cross-process). The harness lives in
``e2e-artifacts/staged-workers/test_transfer_xproc.py``; this wraps it as a
gpu-marked test that skips when the transport is unavailable.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HARNESS = os.path.join(REPO, "e2e-artifacts", "staged-workers", "test_transfer_xproc.py")


def _run(backend: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, HARNESS, "orchestrate", backend],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.parametrize("backend", ["shm", "cuda_ipc", "mooncake"])
def test_transfer_backend_moves_tensor_cross_process(backend):
    if backend in ("cuda_ipc", "mooncake"):
        _require_cuda()
    if backend == "mooncake":
        # Use the backend's own cu12 preload so this check matches the subprocess.
        from uniserve_worker.runtime.transfer import _ensure_mooncake_runtime

        _ensure_mooncake_runtime()
        try:
            import mooncake.engine  # noqa: F401
        except Exception as exc:  # pragma: no cover - env-dependent
            pytest.skip(f"mooncake unavailable: {exc}")
    proc = _run(backend)
    if backend == "mooncake" and proc.returncode != 0 and "PASS" not in proc.stdout:
        # Mooncake needs an active RDMA device; skip if the engine can't init.
        if "Failed to initialize" in proc.stdout + proc.stderr or "RDMA" in proc.stderr:
            pytest.skip("mooncake RDMA engine could not initialize in this environment")
    assert proc.returncode == 0, f"{backend} transfer failed:\nSTDOUT{proc.stdout}\nSTDERR{proc.stderr[-2000:]}"
    assert "result=PASS" in proc.stdout, proc.stdout
