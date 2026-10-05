"""Native VMM retirement keeps DLPack storage and remote-reader claims alive."""

import time

import pytest
import torch
import tvm_ffi
from bindings import CUDAStream, VmmPool
from uniserve_kernels.peer_storage import empty


def test_retired_reader_keeps_capacity_until_its_acknowledgment():
    storage = empty(
        (1 << 20,), dtype=torch.uint8, device=torch.device("cuda:0")
    )
    capacity = storage.numel()
    stream = torch.cuda.Stream()
    producer = CUDAStream(0, stream.cuda_stream, 1)
    with torch.cuda.stream(stream), tvm_ffi.use_torch_stream(stream):
        pool = VmmPool(storage)
        chunk = pool.reserve(capacity - 256)
        words = torch.from_dlpack(chunk.acknowledgments()).view(torch.int32)
        words[1] = 1

    try:
        pool.retire(chunk, [1], producer.record(), None, "")
        pool.reap()
        torch.cuda.synchronize()
        pool.reap()
        with pytest.raises(RuntimeError, match="does not fit"):
            pool.reserve(1)

        words[1] = 2
        torch.cuda.synchronize()
        deadline = time.monotonic() + 5
        while pool.awaiting_acknowledgment():
            pool.reap()
            assert time.monotonic() < deadline
            time.sleep(0.001)
        with tvm_ffi.use_torch_stream(torch.cuda.current_stream()):
            replacement = pool.reserve(capacity - 256)
        view = torch.from_dlpack(replacement.tensor())
        view.fill_(37)
        torch.testing.assert_close(view, torch.full_like(view, 37))
    finally:
        pool.close()
        producer.close()


def test_payload_view_survives_the_last_pool_owner():
    storage = empty(
        (1 << 20,), dtype=torch.uint8, device=torch.device("cuda:0")
    )
    pool = VmmPool(storage)
    del storage
    with tvm_ffi.use_torch_stream(torch.cuda.current_stream()):
        chunk = pool.reserve(64)
    view = torch.from_dlpack(chunk.tensor())
    owners = tvm_ffi.Array([pool])
    del pool
    # The DLPack payload retains its allocation independently of the pool.
    del chunk, owners
    view.fill_(19)
    torch.testing.assert_close(view, torch.full_like(view, 19))
