"""Native shared storage keeps DLPack views and CUDA copies alive."""

import torch
import tvm_ffi
from bindings import SharedBuffer

from tests.python.fixtures.cuda_stream import blocked_stream


def test_dlpack_view_survives_the_last_shared_buffer_owner():
    buffer = SharedBuffer(64, [], None)
    owners = tvm_ffi.Array([buffer])
    del buffer
    view = torch.from_dlpack(owners[0].tensor()).view(torch.float32)
    view.copy_(torch.arange(16, dtype=torch.float32))
    owners[0].mark_ready()
    assert owners[0].settled()
    del owners

    # Final native destruction unlinks the name; this tensor retains only the
    # mapped payload and remains writable until its DLPack reference retires.
    view.add_(1)
    torch.testing.assert_close(view, torch.arange(1, 17, dtype=torch.float32))


def test_cuda_readiness_follows_the_copy_on_the_ffi_stream():
    source = torch.arange(16, dtype=torch.float32, device="cuda:0")
    with blocked_stream("cuda:0") as stream:
        with torch.cuda.stream(stream), tvm_ffi.use_torch_stream(stream):
            buffer = SharedBuffer(64, [], 0)
            view = torch.from_dlpack(buffer.tensor()).view(torch.float32)
            buffer.begin_copy()
            view.copy_(source, non_blocking=True)
            buffer.mark_ready()
        assert not buffer.settled()

    try:
        assert buffer.settled()
        torch.testing.assert_close(view, torch.arange(16, dtype=torch.float32))
    finally:
        buffer.close()

    del buffer
    view.add_(1)
    torch.testing.assert_close(view, torch.arange(1, 17, dtype=torch.float32))
