"""Native host execution using only TVM-FFI callbacks and native results."""

import gc
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch
import tvm_ffi
from bindings import HostLane


def test_host_callback_retains_inputs_and_returns_numerical_bytes():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU())
    values = torch.arange(32, dtype=torch.float32).view(4, 8)
    release = Event()
    with torch.inference_mode():
        expected = model(values).numpy().tobytes()

    def encode(model=model, values=values):
        assert release.wait(5)
        with torch.inference_mode():
            return model(values).numpy().tobytes()

    model_ref, values_ref = weakref.ref(model), weakref.ref(values)
    lane = HostLane(1, 1)
    try:
        task = lane.submit(tvm_ffi.convert_func(encode))
        del model, values, encode
        gc.collect()
        assert model_ref() is not None and values_ref() is not None
        release.set()
        assert task.result() == expected
    finally:
        release.set()
        lane.close()

    gc.collect()
    assert model_ref() is None and values_ref() is None
    assert task.result() == expected


def test_cancelled_host_work_releases_capacity_and_callback_inputs():
    lane = HostLane(2, 1)
    entered, release = Event(), Event()

    def first():
        entered.set()
        assert release.wait(5)
        return b"first"

    running = lane.submit(tvm_ffi.convert_func(first))
    try:
        assert entered.wait(5)
        values = torch.tensor([7.0])
        reference = weakref.ref(values)

        def encode(values=values):
            return values.numpy().tobytes()

        queued = lane.submit(tvm_ffi.convert_func(encode))
        del encode, values
        with pytest.raises(RuntimeError, match="capacity"):
            lane.submit(tvm_ffi.convert_func(lambda: None))
        assert not running.cancel()
        assert queued.cancel()
        assert queued.done()
        with pytest.raises(RuntimeError, match="cancelled"):
            queued.result()

        gc.collect()
        assert reference() is None
        replacement = lane.submit(tvm_ffi.convert_func(lambda: b"replacement"))
        release.set()
        assert running.result() == b"first"
        assert replacement.result() == b"replacement"
    finally:
        release.set()
        lane.close()


def test_host_error_preserves_type_message_and_callback_traceback():
    def numerical_failure():
        raise ValueError("host numerical failure")

    lane = HostLane(1, 1)
    try:
        task = lane.submit(tvm_ffi.convert_func(numerical_failure))
        with pytest.raises(
            ValueError, match="host numerical failure"
        ) as raised:
            task.result()
        assert "numerical_failure" in "".join(
            traceback.format_exception(raised.value)
        )
        recovered = lane.submit(tvm_ffi.convert_func(lambda: b"recovered"))
        assert recovered.result() == b"recovered"
    finally:
        lane.close()


def test_host_observer_can_read_results_and_submit_more_work():
    lane = HostLane(1, 1)
    release, observed = Event(), Event()
    results = []
    successors = []

    def first():
        assert release.wait(5)
        return b"first"

    def observe(task):
        results.append(task.result())
        successors.append(lane.submit(tvm_ffi.convert_func(lambda: b"second")))
        observed.set()

    task = lane.submit(tvm_ffi.convert_func(first))
    try:
        task.add_done_callback(tvm_ffi.convert_func(observe))
        release.set()
        assert observed.wait(5)
        assert results == [b"first"]
        assert successors[0].result() == b"second"
        task.add_done_callback(
            tvm_ffi.convert_func(
                lambda completed: results.append(completed.result())
            )
        )
        assert results == [b"first", b"first"]
    finally:
        release.set()
        lane.close()


def test_host_close_drains_staging_and_releases_the_gil():
    lane = HostLane(1, 1)
    entered, release = Event(), Event()
    destination = torch.zeros(4)

    def stage():
        entered.set()
        assert release.wait(5)
        destination.fill_(7)

    task = lane.submit(tvm_ffi.convert_func(stage))
    try:
        assert entered.wait(5)
        with ThreadPoolExecutor(max_workers=1) as caller:
            closing = caller.submit(lane.close)
            try:
                with pytest.raises(TimeoutError):
                    closing.result(timeout=0.05)
            finally:
                release.set()
            closing.result(timeout=5)

        assert task.result() is None
        torch.testing.assert_close(
            destination, torch.full((4,), 7.0), rtol=0, atol=0
        )
        with pytest.raises(RuntimeError, match="closed"):
            lane.submit(tvm_ffi.convert_func(stage))
    finally:
        release.set()
        lane.close()
