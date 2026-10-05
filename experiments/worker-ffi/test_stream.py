"""Native streams order ordinary PyTorch work through TVM-FFI alone."""

import pytest
import torch
import tvm_ffi
from bindings import CUDAStream, partition_streams

from tests.python.fixtures.cuda_stream import blocked_stream


def test_fork_and_fence_retain_the_sm_partition():
    parent = partition_streams(0, [64])[0]
    child = parent.fork()
    parent.close()
    del parent

    view = torch.cuda.ExternalStream(child.handle(), device=0)
    value = torch.zeros(32, device="cuda:0")
    try:
        assert child.sm_count() == 64
        with tvm_ffi.use_torch_stream(torch.cuda.current_stream(0)):
            child.wait()
        with torch.cuda.stream(view):
            value.add_(7)
        completed = child.record()
    finally:
        child.close()
    del child

    completed.wait()
    assert value.cpu().tolist() == [7] * 32


@pytest.mark.timeout(30)
def test_foreign_stream_owner_and_device_only_submission():
    view = torch.cuda.Stream(device=0)
    retained = tvm_ffi.Array([CUDAStream(0, view.cuda_stream, 2)])
    owner = retained[0]
    del retained
    source = torch.zeros(32, device="cuda:0")
    target = torch.empty_like(source)
    try:
        for value in (7, 13, 29):
            with blocked_stream("cuda:0") as producer:
                with torch.cuda.stream(producer):
                    source.fill_(value)
                with tvm_ffi.use_torch_stream(producer):
                    owner.wait()
                with torch.cuda.stream(view):
                    target.copy_(source)
                completed = owner.record()
                assert not completed.query()
                completed.wait()

            assert target.cpu().tolist() == [value] * 32
    finally:
        owner.close()
