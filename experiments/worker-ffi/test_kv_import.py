"""KV destination ownership and numerical failures through native copy tasks."""

import gc
import multiprocessing
import traceback
import weakref
from threading import Event

import pytest
import torch
import tvm_ffi
from bindings import KVImporter, WorkerRequest, load_library

from tests.python.fixtures.cuda_stream import blocked_stream


def test_queued_copy_cancellation_releases_destination_and_allows_reuse(
    kv_request,
):
    pool = KVImporter(2, 1, None)
    source = torch.arange(1, 5, dtype=torch.float32).reshape(2, 2)
    destination = torch.full((2, 2, 2, 2), -1.0)
    scratch = torch.empty(2, 2, 2)
    entered, release = Event(), Event()

    def reset(slot):
        def numerical_reset(_workspace, _stream):
            destination[slot].zero_()

        return tvm_ffi.convert_func(numerical_reset)

    def copy(slot):
        def numerical_copy(_workspace, _stream):
            scratch.copy_(source)
            destination[slot].copy_(scratch)

        return tvm_ffi.convert_func(numerical_copy)

    def waiting_copy(_workspace, _stream):
        entered.set()
        assert release.wait(5)
        destination[0].copy_(source)

    try:
        first = pool.reserve(
            kv_request, 0, 1, reset(0), tvm_ffi.convert_func(waiting_copy)
        )
        assert entered.wait(5)
        queued = pool.reserve(kv_request, 1, 2, reset(1), copy(1))
        assert queued.cancel()
        pool.abandon(queued)
        assert queued.retired()
        with pytest.raises(RuntimeError, match="cancelled"):
            queued.result()
        torch.testing.assert_close(destination[1], torch.full_like(scratch, -1))

        replacement = pool.reserve(kv_request, 1, 2, reset(1), copy(1))
        release.set()
        first.result()
        replacement.result()
        pool.adopt(first)
        pool.adopt(replacement)
        assert first.retired() and replacement.retired()
        torch.testing.assert_close(
            destination, source.expand_as(destination), rtol=0, atol=0
        )
    finally:
        release.set()
        pool.stop()


def _cuda_copy(library, wire_request, fails):
    load_library(library)
    kv_request = WorkerRequest(wire_request)
    # Construct native streams before PyTorch initializes the device. Their
    # CUDA context must remain alive after constructor device guards leave.
    pool = KVImporter(1, 1, 0)
    source = torch.arange(1, 5, dtype=torch.float32, device="cuda:0").reshape(
        2, 2
    )
    destination = torch.full((2, 2, 2), -1.0, device="cuda:0")
    # Load the broadcast copy kernel before holding device work behind a gate.
    destination.copy_(source)
    destination.zero_()
    torch.cuda.synchronize()
    retained = weakref.ref(source)
    entered = Event()

    def reset(_workspace, stream):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=0)):
            destination.zero_()

    try:
        with blocked_stream("cuda:0") as producer:

            def copy(_workspace, stream, source=source):
                numerical = torch.cuda.ExternalStream(stream, device=0)
                numerical.wait_stream(producer)
                with torch.cuda.stream(numerical):
                    destination.copy_(source)
                entered.set()
                if fails:
                    raise ValueError("KV conversion failed")

            write = pool.reserve(
                kv_request,
                0,
                1,
                tvm_ffi.convert_func(reset),
                tvm_ffi.convert_func(copy),
            )
            del copy, source
            assert entered.wait(5)
            pool.abandon(write)
            assert not write.done() and not write.retired()
            assert retained() is not None

        if fails:
            with pytest.raises(
                ValueError, match="KV conversion failed"
            ) as raised:
                write.result()
            assert "in copy" in "".join(
                traceback.format_exception(raised.value)
            )
        else:
            write.result()
        pool.stop()
        assert write.retired()
        torch.testing.assert_close(
            destination.cpu(),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]).expand(2, 2, 2),
            rtol=0,
            atol=0,
        )
        gc.collect()
        assert retained() is None
    finally:
        pool.stop()


@pytest.mark.parametrize("fails", (False, True))
def test_cuda_copy_retains_abandoned_destination_through_drain(
    request, kv_request, fails
):
    # A fresh process exercises CUDA initialization order and contains a
    # device/GIL deadlock independently of the test runner.
    process = multiprocessing.get_context("spawn").Process(
        target=_cuda_copy,
        args=(
            request.config.getoption("--ffi-library"),
            kv_request.encode(),
            fails,
        ),
    )
    process.start()
    try:
        process.join(timeout=30)
        assert not process.is_alive(), "KV copy did not retire"
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
