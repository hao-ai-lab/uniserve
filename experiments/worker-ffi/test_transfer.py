"""Read admission and physical lifetime through the production native pool."""

import gc
import traceback
import weakref
from threading import Event

import pytest
import torch
import tvm_ffi
from bindings import TransferPool

from tests.python.fixtures.cuda_stream import blocked_stream


def test_queued_cancellation_returns_capacity_and_releases_the_source():
    pool = TransferPool(8, 2, 1)
    started, release = Event(), Event()
    first_value = torch.zeros(1)

    def waiting_read(_stream):
        started.set()
        assert release.wait(5)
        first_value.fill_(7)

    try:
        first = pool.submit(tvm_ffi.convert_func(waiting_read), first_value, 4)
        assert started.wait(5)
        source = torch.tensor([13.0])
        reference = weakref.ref(source)
        destination = torch.zeros_like(source)

        def read(_stream, source=source):
            destination.copy_(source)

        queued = pool.submit(tvm_ffi.convert_func(read), destination, 4)
        del read, source
        queued.cancel()
        assert queued.retirement_ready()
        assert pool.used() == 4
        with pytest.raises(RuntimeError, match="cancelled"):
            queued.result()
        gc.collect()
        assert reference() is None
        assert destination.item() == 0

        replacement = pool.submit(
            tvm_ffi.convert_func(lambda _stream: destination.fill_(19)),
            destination,
            4,
        )
        release.set()
        pool.close()
        assert torch.from_dlpack(first.result()).item() == 7
        assert torch.from_dlpack(replacement.result()).item() == 19
        assert pool.used() == 0
        with pytest.raises(RuntimeError, match="closed"):
            pool.submit(tvm_ffi.convert_func(waiting_read), first_value, 4)
        assert pool.used() == 0
    finally:
        release.set()
        pool.close()


def test_failed_numerical_read_preserves_exception_and_allows_reuse():
    pool = TransferPool(4, 1, 1)
    destination = torch.zeros(1)

    def failing_read(_stream):
        raise ValueError("read source unavailable")

    try:
        failed = pool.submit(tvm_ffi.convert_func(failing_read), destination, 4)
        finished = Event()
        failed.add_retirement_callback(tvm_ffi.convert_func(finished.set))
        assert finished.wait(5)
        with pytest.raises(
            ValueError, match="read source unavailable"
        ) as raised:
            failed.result()
        assert "failing_read" in "".join(
            traceback.format_exception(raised.value)
        )

        assert failed.retirement_ready()
        assert pool.used() == 0
        recovered = pool.submit(
            tvm_ffi.convert_func(lambda _stream: destination.fill_(17)),
            destination,
            4,
        )
        pool.close()
        assert torch.from_dlpack(recovered.result()).item() == 17
    finally:
        pool.close()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("cancelled", (False, True))
def test_cuda_read_joins_streams_and_retains_credits_until_completion(
    cancelled,
):
    source = torch.full((32,), 23.0, device="cuda:0")
    destination = torch.zeros_like(source)
    bytes_ = source.numel() * source.element_size()
    pool = TransferPool(bytes_, 1, 1)
    consumer = torch.cuda.Stream()
    observed = Event()

    def read(stream, source=source, destination=destination):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=0)):
            destination.copy_(source)

    try:
        with blocked_stream("cuda:0") as producer:
            with torch.cuda.stream(producer):
                destination.fill_(-1)
            with tvm_ffi.use_torch_stream(producer):
                ticket = pool.submit(
                    tvm_ffi.convert_func(read), destination, bytes_
                )
            ticket.add_done_callback(tvm_ffi.convert_func(observed.set))
            del read, source
            assert observed.wait(5), "readiness waited for device completion"
            assert ticket.ready() and not ticket.retirement_ready()
            assert pool.used() == bytes_

            if cancelled:
                ticket.cancel()
                with pytest.raises(RuntimeError, match="cancelled"):
                    ticket.result()
                with pytest.raises(RuntimeError, match="capacity"):
                    pool.submit(
                        tvm_ffi.convert_func(lambda _stream: None),
                        destination,
                        bytes_,
                    )
            else:
                with (
                    torch.cuda.stream(consumer),
                    tvm_ffi.use_torch_stream(consumer),
                ):
                    result = torch.from_dlpack(ticket.result()).clone()

        pool.close()
        assert ticket.retirement_ready()
        assert pool.used() == 0
        consumer.synchronize()
        actual = destination if cancelled else result
        torch.testing.assert_close(
            actual, torch.full_like(actual, 23), rtol=0, atol=0
        )
    finally:
        pool.close()
