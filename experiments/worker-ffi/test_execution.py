"""Observable tensor values, cancellation, and physical lifetime across FFI."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch
import tvm_ffi
from bindings import Executor, WorkerRequest


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
        batch = executor.submit(values)
        del values, model
        actual = torch.from_dlpack(batch.result())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        batch.wait()
        assert batch.retired()
    finally:
        executor.close()


def test_strided_tensor_keeps_shared_storage():
    values = torch.arange(48, dtype=torch.float32).view(6, 8)
    view = values[1:5, ::2]
    executor = Executor(
        tvm_ffi.convert_func(torch.nn.Identity(), tensor_cls=torch.Tensor), 1
    )
    try:
        batch = executor.submit(view)
        actual = torch.from_dlpack(batch.result())
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
            executor.submit(values).wait()
            batch = executor.submit(values)
            assert not batch.retired()
            batch.cancel()
            with pytest.raises(RuntimeError, match="cancelled"):
                batch.result()
            with pytest.raises(RuntimeError, match="capacity"):
                executor.submit(values)

        batch.wait()
        assert batch.retired()
        with torch.cuda.stream(producer), tvm_ffi.use_torch_stream(producer):
            successor = executor.submit(values)
        del values
        with torch.cuda.stream(consumer), tvm_ffi.use_torch_stream(consumer):
            result = torch.from_dlpack(successor.result())
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
            batch = executor.submit(values)
            with pytest.raises(ValueError, match="numerical callback failed"):
                batch.result()
            assert not batch.retired()
        batch.wait()
        assert batch.retired()
        torch.testing.assert_close(
            values.cpu(), torch.full((16,), 6.0), rtol=0, atol=0
        )
    finally:
        executor.close()


def test_close_preserves_results_and_rejects_submission():
    executor = Executor(
        tvm_ffi.convert_func(torch.nn.Identity(), tensor_cls=torch.Tensor), 1
    )
    batch = executor.submit(torch.tensor([2.0]))
    executor.close()
    torch.testing.assert_close(
        torch.from_dlpack(batch.result()), torch.tensor([2.0]), rtol=0, atol=0
    )
    with pytest.raises(RuntimeError, match="closed"):
        executor.submit(torch.tensor([3.0]))


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
        pending = thread.submit(executor.submit, torch.tensor([4.0]))
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="already submitting"):
                executor.submit(torch.tensor([8.0]))
            released.set()
            batch = pending.result(timeout=5)
            torch.testing.assert_close(
                torch.from_dlpack(batch.result()),
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
