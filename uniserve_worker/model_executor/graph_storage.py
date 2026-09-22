"""One byte budget for all captured calls on a worker's devices."""

from contextlib import ExitStack, contextmanager

import torch

from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.runtime.device import canonical_device
from uniserve_worker.config.execution import graph_storage_budget_bytes


class GraphStorage:
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
            raise ValueError("graph storage budgets must be nonnegative")

    def reserve(self, owner, devices):
        if owner in self._pools:
            raise RuntimeError("graph storage owner is already registered")
        pools = {}
        for device in dict.fromkeys(map(canonical_device, devices)):
            if device.type != "cuda":
                continue
            with torch.cuda.device(device):
                pools[device] = torch.cuda.MemPool()
            self._budgets.setdefault(
                device,
                graph_storage_budget_bytes(
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
        for device, used in self.resident_bytes().items():
            if used > self._budgets[device]:
                raise CUDAGraphError(
                    f"graph residency on {device} exceeds its byte budget "
                    f"({used}>{self._budgets[device]})"
                )

    def set_budget(self, device, amount):
        """Bind the graph share of an owner's remaining device-storage grant."""
        if amount < 0:
            raise ValueError("graph storage budgets must be nonnegative")
        self._budgets[canonical_device(device)] = int(amount)
        self.check()

    def resident_bytes(self):
        """Return reserved pool bytes, including reusable capture workspace."""
        sizes = dict.fromkeys(self._budgets, 0)
        ids = {
            (device.index, tuple(pool.id)): device
            for pools in self._pools.values()
            for device, pool in pools.items()
        }
        if not ids:
            return sizes
        for segment in torch.cuda.memory_snapshot():
            device = ids.get(
                (
                    segment["device"],
                    tuple(segment.get("segment_pool_id", ())),
                )
            )
            if device is not None:
                sizes[device] += segment["total_size"]
        return sizes

    def release(self, owner):
        """Release a pool only after its graphs and borrowed views retire."""
        self._pools.pop(owner, None)

    def close(self):
        self._pools.clear()
