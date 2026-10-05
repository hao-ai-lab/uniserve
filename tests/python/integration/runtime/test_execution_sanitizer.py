"""Diagnose real CUDA calls, including native Driver API waits."""

import ctypes
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing

import pytest
import torch

from uniserve.sanitizer.nsys import analyze
from uniserve_worker._uniserve_ipc import EventPool

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_cuda_timeline_hints_distinguish_host_and_device_dependencies(tmp_path):
    nsys = shutil.which("nsys")
    if nsys is None or not torch.cuda.is_available():
        pytest.skip("Nsight Systems and CUDA are required")

    capture = str(tmp_path / "execution")
    result = subprocess.run(
        [
            nsys,
            "profile",
            "--trace=cuda,nvtx,python-gil",
            "--sample=process-tree",
            "--cuda-trace-all-apis=true",
            "--cudabacktrace=sync:0,memory:0",
            "--export=sqlite",
            f"--output={capture}",
            sys.executable,
            "-m",
            "tests.python.integration.runtime.test_execution_sanitizer",
        ],
        env={**os.environ, "UNISERVE_NVTX": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    report = analyze(capture + ".sqlite")

    by_scope = {}
    for hint in report.hints:
        by_scope.setdefault(hint.evidence[0].scope.name, []).append(hint)

    # Each blocking call is counted once even when Nsight also records its
    # call stack. Stream dependencies and nonblocking queries do not block
    # the host, so the sanitizer must distinguish them from synchronize().
    waits = by_scope["uniserve.capture.runtime_wait"]
    assert len([h for h in waits if h.rule == "host-sync"]) == 3
    native = by_scope["uniserve.capture.driver_wait"]
    assert len(native) == 1
    assert native[0].evidence[0].name.startswith("cuEventSynchronize")
    assert "uniserve.capture.stream_dependency" not in by_scope
    assert "uniserve.capture.pinned_copy" not in by_scope

    assert any(
        h.rule == "pageable-copy"
        for h in by_scope["uniserve.capture.pageable_copy"]
    )
    synchronous = by_scope["uniserve.capture.synchronous_copy"]
    assert [h.rule for h in synchronous] == ["sync-copy"]
    roundtrip = by_scope["uniserve.capture.roundtrip"]
    pairs = [h for h in roundtrip if h.rule == "device-roundtrip"]
    assert len(pairs) == 1
    first, second = pairs[0].evidence
    assert first.copies[0].end_ns <= second.start_ns
    assert first.stack or native[0].evidence[0].stack
    assert report.coverage["driver_calls"] > 0

    held = by_scope["uniserve.capture.gil_held_wait"]
    assert any(h.rule == "gil-held-wait" for h in held)
    released = by_scope["uniserve.capture.gil_released_wait"]
    assert all(h.rule == "host-sync" for h in released)
    assert report.coverage["gil_holds"] > 0

    # Batch phase names and payloads are part of the profiler interface.
    # They must come from the actual native executor, including preparation.
    with closing(sqlite3.connect(capture + ".sqlite")) as db:
        phases = dict(
            db.execute(
                "SELECT coalesce(n.text, s.value), n.uint64Value "
                "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId "
                "WHERE coalesce(n.text, s.value) LIKE 'uniserve.worker.%'"
            )
        )
    assert phases["uniserve.worker.prepare"] == 1
    assert phases["uniserve.worker.execute"] == 1
    assert phases["uniserve.worker.poll"] == 1


def _capture_workload():
    device = torch.device("cuda:0")
    stream = torch.cuda.current_stream(device)
    other = torch.cuda.Stream(device=device)
    runtime_event = torch.cuda.Event()
    pool = EventPool()
    event = pool.acquire(device)
    readback = pool.acquire(device)
    pool.retain(event, device)
    pool.retain(readback, device)
    values = torch.arange(16, device=device, dtype=torch.int32)
    pageable = torch.arange(16, dtype=torch.int32)
    pinned = pageable.pin_memory()
    output = torch.empty_like(pinned, pin_memory=True)
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("uniserve.capture.runtime_wait"):
        runtime_event.record(stream)
        runtime_event.synchronize()
        stream.synchronize()
        torch.cuda.synchronize()

    pool.record(event, device)
    with torch.cuda.nvtx.range("uniserve.capture.driver_wait"):
        event.synchronize()

    with torch.cuda.nvtx.range("uniserve.capture.stream_dependency"):
        event.wait(other)
        event.query()

    with torch.cuda.nvtx.range("uniserve.capture.pageable_copy"):
        values.copy_(pageable, non_blocking=True)
    with torch.cuda.nvtx.range("uniserve.capture.pinned_copy"):
        values.copy_(pinned, non_blocking=True)

    # Call a synchronous CUDA API directly: PyTorch's blocking copy instead
    # uses cudaMemcpyAsync followed by a stream synchronization.
    driver = ctypes.CDLL("libcuda.so.1")
    copy = driver.cuMemcpyDtoH_v2
    copy.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    copy.restype = ctypes.c_int
    with torch.cuda.nvtx.range("uniserve.capture.synchronous_copy"):
        assert copy(output.data_ptr(), values.data_ptr(), 64) == 0

    with torch.cuda.nvtx.range("uniserve.capture.roundtrip"):
        output.copy_(values, non_blocking=True)
        pool.record(readback, device)
        readback.synchronize()
        values.copy_(output, non_blocking=True)

    torch.cuda.synchronize()
    assert torch.equal(values.cpu(), pageable)
    pool.release(event)
    pool.release(readback)
    pool.close()

    from tests.python.fixtures.depth_one import (
        ar_params,
        execution_batch,
        finalized_report,
        root_parent,
        token_call,
    )
    from tests.python.fixtures.execution_worker import execution_worker
    from uniserve_worker.protocol.call import ForwardMode
    from uniserve_worker.protocol.identity import CallId

    admission = ar_params(1, block_ids=(0,))
    call = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    with execution_worker(device=str(device)) as worker:
        submitted = worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(call,),
            )
        )
        assert len(finalized_report(worker, submitted).completions) == 1


def _capture_gil_waits():
    # ctypes is an external native boundary with explicit GIL behavior:
    # PyDLL holds it during a call, while CDLL releases it. The CUDA work is
    # identical, so the trace must distinguish ownership from API duration.
    torch.cuda._sleep(1)
    torch.cuda.synchronize()

    for loader, name in (
        (ctypes.PyDLL, "gil_held_wait"),
        (ctypes.CDLL, "gil_released_wait"),
    ):
        synchronize = loader("libcuda.so.1").cuEventSynchronize
        synchronize.argtypes = [ctypes.c_void_p]
        synchronize.restype = ctypes.c_int
        event = torch.cuda.Event()

        torch.cuda._sleep(100_000_000)
        event.record()

        with torch.cuda.nvtx.range("uniserve.capture." + name):
            assert synchronize(event.cuda_event) == 0


if __name__ == "__main__":
    _capture_workload()
    _capture_gil_waits()
