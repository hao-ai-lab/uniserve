"""Execution stream ordering and SM partition lifetime on real CUDA devices."""

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest
import torch

from tests.python.fixtures.cuda_stream import blocked_stream
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    partition_streams,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def test_fork_keeps_partition_and_graph_execution_after_parent_closes():
    device = torch.device("cuda:0")
    parent = partition_streams(device, (64,))[0]
    child = parent.fork()
    parent.close()

    module = torch.nn.Linear(32, 32, bias=False).to(device)
    value = torch.randn(8, 32, device=device)
    expected = module(value)
    try:
        assert child.sm_count == 64
        assert not child.full_device

        with ExecutionContext(module, stream=child) as context:
            context.prepare(None)
            with context.activate():
                module(value)

            with CUDAGraph(context=context) as graph:
                graph.capture(lambda: module(value))
                actual = graph.replay()
                child.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        child.close()


def test_output_fence_survives_partition_stream_close():
    device = torch.device("cuda:0")
    with partition_streams(device, (64,))[0] as stream:
        value = torch.zeros(32, device=device)
        stream.wait(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream.stream):
            value.add_(7)
        completed = stream.record()

    # The event remains usable after both the producer stream and its Python
    # owner retire. Its context cannot disappear before the final consumer.
    del stream
    assert completed is not None
    completed.wait(torch.cuda.current_stream(device))
    torch.testing.assert_close(value.cpu(), torch.full((32,), 7.0))


def test_submission_fences_order_device_work_without_host_waiting():
    device = torch.device("cuda:0")
    with CUDAStream.external(torch.cuda.Stream(device=device)) as stream:
        source = torch.zeros(32, device=device)
        target = torch.empty_like(source)
        stream.wait(torch.cuda.current_stream(device))

        for iteration in range(4):
            with blocked_stream(device) as producer:
                with torch.cuda.stream(producer):
                    source.fill_(iteration)

                stream.wait(producer)
                with torch.cuda.stream(stream.stream):
                    target.copy_(source)
                completed = stream.record()
                assert completed is not None
                assert not completed.query()
                completed.wait(torch.cuda.current_stream(device))

            torch.testing.assert_close(
                target.cpu(), torch.full((32,), float(iteration))
            )


@pytest.mark.parametrize("explicit_close", (True, False))
@pytest.mark.timeout(30)
def test_retirement_releases_the_gil_while_device_work_is_pending(
    explicit_close,
):
    owners = [CUDAStream.external(torch.cuda.Stream(device=0))]
    entering_retirement = threading.Event()

    def retire():
        stream = owners.pop()
        entering_retirement.set()
        if explicit_close:
            stream.close()
        del stream

    with ThreadPoolExecutor(max_workers=1) as threads:
        with blocked_stream("cuda:0") as producer:
            owners[0].wait(producer)
            closing = threads.submit(retire)
            assert entering_retirement.wait(5)
            with pytest.raises(TimeoutError):
                closing.result(timeout=0.1)

        closing.result(timeout=5)
