"""CUDA graph ownership and PyTorch capture preparation.

Rust owns capture segments, replay order and resource retirement. PyTorch
supplies numerical stream scopes, allocator pools and graph executables.
"""

from contextlib import ExitStack, contextmanager

import torch

from uniserve_worker._uniserve_ipc import CUDAGraph as CUDAGraph
from uniserve_worker._uniserve_ipc import CUDAGraphError as CUDAGraphError


@contextmanager
def _capture_scope(context, pools, capture, computation):
    """Prepare PyTorch state and order the complete capture against its caller.

    Pool cleanup happens once for the whole call. Segments share allocation
    lifetime even when an eager copy submission separates their captures.
    """
    device = context._device
    current = torch.cuda.current_stream(device)
    capture.wait_stream(current)
    try:
        with (
            torch.inference_mode(),
            ExitStack() as scope,
            torch.cuda.device(device),
            torch.cuda.stream(capture),
        ):
            for target, pool in pools.items():
                if target != device:
                    scope.enter_context(torch.cuda.use_mem_pool(pool, target))

            yield
    finally:
        current.wait_stream(computation)
        current.wait_stream(capture)


def _prepare_capture(device, computation):
    """Retire warmup allocations and prepare this thread's numerical handles."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()

    # cuBLAS handles are thread-local. Warmup may have used another host
    # thread, so create this thread's handle before capture.
    with torch.cuda.stream(computation):
        torch.cuda.current_blas_handle()
