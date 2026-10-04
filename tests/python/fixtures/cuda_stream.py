"""Device-side waits for deterministic asynchronous storage tests."""

import ctypes
from contextlib import contextmanager

import torch


@contextmanager
def blocked_stream(device):
    """Keep a stream pending until the scope exits, without a host callback."""
    driver = ctypes.CDLL("libcuda.so.1")
    for name in ("cuStreamWaitValue32_v2", "cuStreamWriteValue32_v2"):
        function = getattr(driver, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int

    gate = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.current_stream(device).synchronize()
    stream = torch.cuda.Stream(device=device)
    release = torch.cuda.Stream(device=device)

    assert (
        driver.cuStreamWaitValue32_v2(stream.cuda_stream, gate.data_ptr(), 1, 1)
        == 0
    )
    try:
        yield stream
    finally:
        assert (
            driver.cuStreamWriteValue32_v2(
                release.cuda_stream, gate.data_ptr(), 1, 0
            )
            == 0
        )
        stream.synchronize()
        release.synchronize()
