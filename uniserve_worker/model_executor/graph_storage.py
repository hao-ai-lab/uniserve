"""PyTorch allocation operations for native graph storage.

The native owner manages pool sharing and per-device residency budgets.
These operations enter the numerical allocator and report its physical use;
graph replay performs neither allocation inspection nor budget accounting.
"""

from contextlib import ExitStack, contextmanager

import torch

from uniserve.runtime.device import process_device_bytes
from uniserve_worker._uniserve_ipc import GraphStorage as GraphStorage


def _new_pool(device):
    with torch.cuda.device(device):
        pool = torch.cuda.MemPool()
    return pool, pool.id, torch.cuda.get_device_properties(device).total_memory


@contextmanager
def _allocate_scope(pools):
    with ExitStack() as scope:
        for device, pool in pools:
            scope.enter_context(torch.cuda.use_mem_pool(pool, device))
        yield


def _snapshot():
    return [
        (
            segment["device"],
            tuple(segment["segment_pool_id"]),
            segment["total_size"],
        )
        for segment in torch.cuda.memory_snapshot()
    ]


def _process_bytes(device):
    return process_device_bytes(torch.device("cuda", device))
