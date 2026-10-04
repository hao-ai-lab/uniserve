"""Observable tensor values, cancellation, and physical lifetime across FFI."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch
import tvm_ffi
from bindings import Executor, RequestPool, WorkerRequest


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@torch.inference_mode()
def test_ordinary_torch_model(device):
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 4)
    ).to(device)
    values = torch.arange(32, dtype=torch.float32, device=device).view(4, 8)
    expected = model(values)
    executor = Executor(tvm_ffi.convert_func(model, tensor_cls=torch.Tensor), 2)
    try:
        batch = executor.submit(1, values)
        del values, model
        batch.wait()
        assert batch.retired()
        actual = torch.from_dlpack(executor.poll(batch))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        executor.close()


def test_strided_tensor_keeps_shared_storage():
    values = torch.arange(48, dtype=torch.float32).view(6, 8)
    view = values[1:5, ::2]
    executor = Executor(
        tvm_ffi.convert_func(torch.nn.Identity(), tensor_cls=torch.Tensor), 1
    )
    try:
        batch = executor.submit(1, view)
        actual = torch.from_dlpack(executor.poll(batch))
        actual.add_(3)
        torch.testing.assert_close(view, actual, rtol=0, atol=0)
        assert actual.stride() == view.stride()
        del values, view
        torch.testing.assert_close(
            actual,
            torch.tensor(
                [
                    [11, 13, 15, 17],
                    [19, 21, 23, 25],
                    [27, 29, 31, 33],
                    [35, 37, 39, 41],
                ],
                dtype=torch.float32,
            ),
            rtol=0,
            atol=0,
        )
    finally:
        executor.close()


def test_completed_results_hold_capacity_until_consumed():
    executor = Executor(
        tvm_ffi.convert_func(torch.nn.Identity(), tensor_cls=torch.Tensor), 1
    )
    try:
        first = executor.submit(17, torch.tensor([2.0]))
        assert first.retired()
        with pytest.raises(RuntimeError, match="queue is full"):
            executor.submit(18, torch.tensor([3.0]))

        torch.testing.assert_close(
            torch.from_dlpack(executor.poll(first)),
            torch.tensor([2.0]),
            rtol=0,
            atol=0,
        )
        with pytest.raises(RuntimeError, match="no longer owned"):
            executor.poll(first)

        # Refusing admission leaves this batch number available to retry.
        second = executor.submit(18, torch.tensor([3.0]))
        torch.testing.assert_close(
            torch.from_dlpack(executor.poll(second)),
            torch.tensor([3.0]),
            rtol=0,
            atol=0,
        )
    finally:
        executor.close()


@torch.inference_mode()
def test_ready_result_does_not_wait_for_another_batch():
    stream = torch.cuda.Stream()

    def forward(value):
        if value.is_cuda:
            torch.cuda._sleep(2_000_000_000)
        return value + 7

    executor = Executor(
        tvm_ffi.convert_func(forward, tensor_cls=torch.Tensor), 2
    )
    values = torch.ones(16, device="cuda")
    try:
        with torch.cuda.stream(stream), tvm_ffi.use_torch_stream(stream):
            warmup = executor.submit(1, values)
            warmup.wait()
            executor.poll(warmup)
            pending = executor.submit(2, values)

        entered = Event()

        def wait_for_pending():
            entered.set()
            pending.wait()

        with ThreadPoolExecutor(max_workers=1) as waiter:
            waiting = waiter.submit(wait_for_pending)
            assert entered.wait(5)
            ready = executor.submit(3, torch.tensor([2.0]))
            torch.testing.assert_close(
                torch.from_dlpack(executor.poll(ready)),
                torch.tensor([9.0]),
                rtol=0,
                atol=0,
            )
            assert executor.poll(pending) is None
            waiting.result(timeout=5)

        torch.testing.assert_close(
            torch.from_dlpack(executor.poll(pending)).cpu(),
            torch.full((16,), 8.0),
            rtol=0,
            atol=0,
        )
    finally:
        executor.close()


@torch.inference_mode()
def test_cancelled_batch_holds_capacity_until_cuda_finishes():
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()

    def forward(value):
        torch.cuda._sleep(2_000_000_000)
        return value + 7

    executor = Executor(
        tvm_ffi.convert_func(forward, tensor_cls=torch.Tensor), 1
    )
    values = torch.full((4096,), 3.0, device="cuda")
    try:
        with torch.cuda.stream(producer), tvm_ffi.use_torch_stream(producer):
            # Prepare allocator blocks and CUDA kernels before requiring a
            # submission to return with device work still pending.
            warmup = executor.submit(1, values)
            warmup.wait()
            executor.poll(warmup)
            batch = executor.submit(2, values)
            assert not batch.retired()
            batch.cancel()
            assert executor.poll(batch) is None
            with pytest.raises(RuntimeError, match="queue is full"):
                executor.submit(3, values)

        batch.wait()
        assert batch.retired()
        with pytest.raises(RuntimeError, match="cancelled"):
            executor.poll(batch)
        with torch.cuda.stream(producer), tvm_ffi.use_torch_stream(producer):
            successor = executor.submit(3, values)
        del values
        successor.wait()
        with torch.cuda.stream(consumer), tvm_ffi.use_torch_stream(consumer):
            result = torch.from_dlpack(executor.poll(successor))
            actual = (result * 2).cpu()
        torch.testing.assert_close(
            actual, torch.full((4096,), 20.0), rtol=0, atol=0
        )
    finally:
        executor.close()


def test_callback_failure_preserves_cuda_retirement():
    stream = torch.cuda.Stream()

    def forward(value):
        torch.cuda._sleep(2_000_000_000)
        value.add_(5)
        raise ValueError("numerical callback failed")

    executor = Executor(
        tvm_ffi.convert_func(forward, tensor_cls=torch.Tensor), 1
    )
    values = torch.ones(16, device="cuda")
    try:
        with torch.cuda.stream(stream), tvm_ffi.use_torch_stream(stream):
            values.add_(0)
            stream.synchronize()
            batch = executor.submit(1, values)
            assert executor.poll(batch) is None
            assert not batch.retired()
        batch.wait()
        assert batch.retired()
        with pytest.raises(ValueError, match="numerical callback failed"):
            executor.poll(batch)
        torch.testing.assert_close(
            values.cpu(), torch.full((16,), 6.0), rtol=0, atol=0
        )
    finally:
        executor.close()


def test_close_preserves_delivered_results_and_rejects_submission():
    executor = Executor(
        tvm_ffi.convert_func(torch.nn.Identity(), tensor_cls=torch.Tensor), 1
    )
    batch = executor.submit(1, torch.tensor([2.0]))
    result = executor.poll(batch)
    executor.close()
    torch.testing.assert_close(
        torch.from_dlpack(result), torch.tensor([2.0]), rtol=0, atol=0
    )
    with pytest.raises(RuntimeError, match="closed"):
        executor.submit(2, torch.tensor([3.0]))


@pytest.mark.parametrize("release", ("close", "drop"))
@torch.inference_mode()
def test_executor_retirement_drains_outstanding_gpu_work(release):
    stream = torch.cuda.Stream()

    def forward(value):
        torch.cuda._sleep(2_000_000_000)
        return value.add_(5)

    executor = Executor(
        tvm_ffi.convert_func(forward, tensor_cls=torch.Tensor), 1
    )
    values = torch.zeros(16, device="cuda")
    try:
        with torch.cuda.stream(stream), tvm_ffi.use_torch_stream(stream):
            warmup = executor.submit(1, values)
            warmup.wait()
            executor.poll(warmup)
            values.zero_()
            pending = executor.submit(2, values)
            assert not pending.retired()

        if release == "close":
            executor.close()
        else:
            del executor

        assert pending.retired()
        torch.testing.assert_close(
            values.cpu(), torch.full((16,), 5.0), rtol=0, atol=0
        )
    finally:
        if release == "close":
            executor.close()
        stream.synchronize()


def test_concurrent_submission_preserves_the_active_callback():
    entered = Event()
    released = Event()

    def forward(value):
        entered.set()
        if not released.wait(5):
            raise RuntimeError("numerical callback was not released")
        return value + 3

    executor = Executor(
        tvm_ffi.convert_func(forward, tensor_cls=torch.Tensor), 2
    )
    with ThreadPoolExecutor(max_workers=1) as thread:
        pending = thread.submit(executor.submit, 1, torch.tensor([4.0]))
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="busy"):
                executor.submit(2, torch.tensor([8.0]))
            released.set()
            batch = pending.result(timeout=5)
            torch.testing.assert_close(
                torch.from_dlpack(executor.poll(batch)),
                torch.tensor([7.0]),
                rtol=0,
                atol=0,
            )
        finally:
            released.set()
            pending.result(timeout=5)
            executor.close()


def test_worker_request_stays_native(wire_request):
    request = WorkerRequest(wire_request)
    assert request.kind() == "submit"
    assert request.batch_id() == 17
    # Preserve every field in the existing wire format across a native object.
    assert request.encode() == wire_request
    with pytest.raises(RuntimeError):
        WorkerRequest(b"not a worker frame")


def test_native_admission_and_retirement(wire_request):
    request = WorkerRequest(wire_request)
    pool = RequestPool(2)
    # This IPC batch admits and immediately finishes a request, as when a
    # cancellation reaches the rank before its first numerical call.
    assert tuple(pool.apply_commands(request)) == (2,)
    assert not pool.has_open_requests()
    pool.retire(5)
    assert tuple(pool.apply_commands(request)) == ()
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.apply_commands(request)
