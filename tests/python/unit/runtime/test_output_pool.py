"""Output values and reuse through the public readback interface."""

import pytest
import torch

from tests.python.fixtures.cuda_stream import blocked_stream
from uniserve.runtime import EventPool
from uniserve_worker.errors import WorkerError
from uniserve_worker.storage.output import OutputPool

pytestmark = pytest.mark.unit


def test_tokens_and_bytes_share_capacity_without_overwriting_each_other():
    events = EventPool()
    outputs = OutputPool(capacity=1, max_words=8, event_pool=events)
    try:
        buffer = outputs.acquire(2, token_capacity=4)
        tokens = buffer.capture(torch.tensor([[1, 2], [3, 4]]).t()[:, 0])
        image = buffer.capture_bytes(torch.arange(16, dtype=torch.uint8))
        with pytest.raises(WorkerError, match="capacity"):
            buffer.capture(torch.tensor([99]))
        with pytest.raises(WorkerError, match="capacity"):
            buffer.capture_bytes(torch.tensor([99], dtype=torch.uint8))
        with pytest.raises(WorkerError, match="before"):
            buffer.read_tokens(*tokens)

        buffer.seal()
        assert buffer.read_tokens(*tokens) == (1, 2)
        assert image.tolist() == list(range(16))
        buffer.observe(0)
        with pytest.raises(WorkerError, match="leases are active"):
            outputs.acquire(1, token_capacity=8)
        buffer.discard(1)
        replacement = outputs.acquire(1, token_capacity=8)
        capture = replacement.capture(torch.arange(8))
        replacement.seal()
        assert replacement.read_tokens(*capture) == tuple(range(8))
        replacement.abandon()
    finally:
        outputs.close()
        events.close()


def test_consumed_lease_cannot_retire_or_complete_a_subsequent_batch():
    events = EventPool()
    outputs = OutputPool(capacity=1, max_words=8, event_pool=events)
    try:
        first = outputs.acquire(1, token_capacity=8)
        first_done = first.completion()
        span = first.capture(torch.tensor([3, 5]))
        first.seal()
        assert first.read_tokens(span[0], 1) == (3,)
        first.observe(0)

        second = outputs.acquire(1, token_capacity=8)
        second.capture(torch.tensor([7, 11]))
        second_done = second.completion()
        first.discard(0)
        first.abandon()
        assert first_done.done()
        assert not second_done.done()
        # Recycled storage cannot change an earlier result, including a range
        # whose values were not individually requested before retirement.
        assert first.read_tokens(*span) == (3, 5)
        with pytest.raises(WorkerError, match="leases are active"):
            outputs.acquire(1, token_capacity=8)

        # Shutdown seals accepted captures and resolves their physical fence.
        outputs.close()
        assert second_done.done()
        with pytest.raises(WorkerError, match="closed"):
            outputs.acquire(1, token_capacity=8)
    finally:
        outputs.close()
        events.close()


@pytest.mark.gpu
@pytest.mark.parametrize("timing", [False, True])
@pytest.mark.parametrize("default_stream", [False, True])
def test_multiple_device_readbacks_keep_values_and_cpu_readers(
    timing, default_stream, monkeypatch
):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    monkeypatch.setenv("UNISERVE_NVTX", "1" if timing else "0")
    events = EventPool()
    outputs = OutputPool(capacity=1, max_words=16, event_pool=events)
    streams = [
        torch.cuda.default_stream(index)
        if default_stream
        else torch.cuda.Stream(device=index)
        for index in range(2)
    ]
    try:
        # Reusing a CPU allocation for CUDA readback must supply pinned storage.
        initial = outputs.acquire(1, token_capacity=4)
        initial.abandon()
        buffer = outputs.acquire(
            2, token_capacity=16, devices=("cuda:0", "cuda:1")
        )
        completion = buffer.completion()
        token_values = torch.arange(6, device="cuda:0")
        image_values = (
            torch.arange(12, dtype=torch.uint8, device="cuda:1")
            .reshape(3, 4)
            .t()
        )
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream(stream.device))

        with blocked_stream("cuda:0") as pending:
            streams[0].wait_stream(pending)
            with torch.cuda.stream(streams[0]):
                buffer.begin_device("cuda:0")
                tokens = buffer.capture(token_values)
                with torch.cuda.stream(streams[1]):
                    buffer.begin_device("cuda:1")
                    image = buffer.capture_bytes(image_values)
                    buffer.seal()
            events.reap()
            assert not completion.done()
            assert not buffer.ready()
            with pytest.raises(WorkerError, match="before"):
                buffer.read_tokens(*tokens)
        release = buffer.retain_cpu_reader()
        for stream in streams:
            stream.synchronize()
        events.reap()
        assert completion.done()
        assert buffer.read_tokens(*tokens) == tuple(range(6))
        assert image.tolist() == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
        buffer.observe(0)
        buffer.observe(1)
        queued, device, copy, host = buffer.timing()
        assert queued >= 0 and host >= 0
        if timing:
            assert device > 0 and copy > 0
        else:
            assert device == 0 and copy == 0
        with pytest.raises(WorkerError, match="leases are active"):
            outputs.acquire(1, token_capacity=8)
        release()
        replacement = outputs.acquire(1, token_capacity=8)
        replacement.abandon()
    finally:
        outputs.close()
        events.close()
