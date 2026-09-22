"""One byte budget for all captured calls on a worker's devices."""

from contextlib import ExitStack, contextmanager

import torch

from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.runtime.device import canonical_device
from uniserve_worker.config import graph_memory_budget_bytes


class GraphMemory:
    """Own private pools through the lifetime of their contexts and graphs.

    Numerical owners keep separate capture catalogs and pools. Allocation and
    admission account for all catalogs together, including prepared workspace
    and static inputs. Replays need no allocator inspection.
    """

    def __init__(self, *, budgets=None):
        self._pools = {}
        self._budgets = {
            canonical_device(device): int(amount)
            for device, amount in (budgets or {}).items()
        }
        if any(amount < 0 for amount in self._budgets.values()):
            raise ValueError("graph memory budgets must be nonnegative")

    def reserve(self, owner, devices):
        if owner in self._pools:
            raise RuntimeError("graph memory owner is already registered")
        pools = {}
        for device in dict.fromkeys(map(canonical_device, devices)):
            if device.type != "cuda":
                continue
            with torch.cuda.device(device):
                pools[device] = torch.cuda.MemPool()
            self._budgets.setdefault(
                device,
                graph_memory_budget_bytes(
                    torch.cuda.get_device_properties(device).total_memory
                ),
            )
        self._pools[owner] = pools
        return pools

    @contextmanager
    def allocate(self, owner):
        """Charge persistent preparation and input allocations to the owner."""
        with ExitStack() as scope:
            for device, pool in self._pools[owner].items():
                scope.enter_context(torch.cuda.use_mem_pool(pool, device))
            yield

    def check(self):
        """Reject residency above the byte bound after preparation/capture."""
        if not self._budgets:
            return
        sizes = dict.fromkeys(self._budgets, 0)
        ids = {
            (device.index, tuple(pool.id)): device
            for pools in self._pools.values()
            for device, pool in pools.items()
        }
        for segment in torch.cuda.memory_snapshot():
            device = ids.get(
                (
                    segment["device"],
                    tuple(segment.get("segment_pool_id", ())),
                )
            )
            if device is not None:
                sizes[device] += segment["total_size"]
        for device, used in sizes.items():
            if used > self._budgets[device]:
                raise CUDAGraphError(
                    f"graph residency on {device} exceeds its byte budget "
                    f"({used}>{self._budgets[device]})"
                )

    def release(self, owner):
        """Release a pool only after its graphs and borrowed views retire."""
        self._pools.pop(owner, None)

    def close(self):
        self._pools.clear()
