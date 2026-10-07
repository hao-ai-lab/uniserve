"""One byte budget for all captured calls on a worker's devices.

``ModelExecutor`` owns one ``GraphStorage`` shared by every runner
(``Execution``). Each runner reserves a private ``torch.cuda.MemPool`` per
CUDA device and allocates its graph captures, prepared workspace and fixed
graph inputs from it; the storage sums those pools per device against one
budget. ``Worker`` rebinds each budget with ``set_budget`` from its
remaining device-storage grant before startup warmup and capture, and from
then until ``seal`` the storage charges everything the process's device
footprint gains, inside the pools or not.
"""

from contextlib import ExitStack, contextmanager

import torch

from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.runtime.device import canonical_device, process_device_bytes
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
        # Per bound device, the bytes the process held outside the pools when
        # ``set_budget`` bound it. Residency on a bound device is the
        # process's footprint less these bytes.
        self._outside = {}

    def reserve(self, owner, devices, *, share=None):
        """Create ``owner``'s private pools and return them by device.

        With ``share``, a registered owner, ``owner`` borrows that owner's
        pools on the devices both name instead. Owners sharing pools must
        never replay their graphs concurrently, and every allocation they
        make outside capture that outlives it must precede the first capture
        into the shared pools: captured graphs reuse free pool blocks as
        their intermediates. Non-CUDA devices are skipped, so the result may
        be empty. A device without a budget gets the default share of its
        total memory from ``graph_storage_budget_bytes``.

        Raises:
            RuntimeError: If ``owner`` already holds pools here, or ``share``
                holds none.
        """
        if owner in self._pools:
            raise RuntimeError("graph storage owner is already registered")
        if share is not None and share not in self._pools:
            raise RuntimeError("shared graph storage owner is not registered")
        pools = {}
        for device in dict.fromkeys(map(canonical_device, devices)):
            if device.type != "cuda":
                continue
            shared = None if share is None else self._pools[share].get(device)
            if shared is not None:
                pools[device] = shared
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
        """Charge persistent preparation and input allocations to the owner.

        Allocations made inside the block on the owner's devices come from
        its private pools. The owner must be reserved.
        """
        with ExitStack() as scope:
            for device, pool in self._pools[owner].items():
                scope.enter_context(torch.cuda.use_mem_pool(pool, device))
            yield

    def check(self):
        """Reject residency above the byte bound after preparation/capture.

        Raises:
            CUDAGraphError: If any device's residency exceeds its budget.
        """
        if not self._budgets:
            return
        for device, used in self.resident_bytes().items():
            if used > self._budgets[device]:
                raise CUDAGraphError(
                    f"graph residency on {device} exceeds its byte budget "
                    f"({used}>{self._budgets[device]})"
                )

    def set_budget(self, device, amount):
        """Bind the graph share of an owner's remaining device-storage grant.

        From binding until ``seal``, the device's residency is its pool bytes
        at binding plus everything the process's device footprint
        (``process_device_bytes``) gains afterwards. Preparation leaves
        storage outside the pools as well: instantiated graph executables,
        communicator resources, loaded modules and work areas created by a
        first eager call. The grant must hold it too, or the device runs out
        of storage while the pools still fit their budget.

        Raises:
            ValueError: ``amount`` is negative.
            CUDAGraphError: The residency already exceeds ``amount``.
        """
        if amount < 0:
            raise ValueError("graph storage budgets must be nonnegative")
        device = canonical_device(device)
        pooled = self.pool_bytes().get(device, 0)
        self._outside[device] = process_device_bytes(device) - pooled
        self._budgets[device] = int(amount)
        self.check()

    def seal(self):
        """Charge only the pools once startup preparation is complete.

        Nothing captures after startup, so the pools stop growing, while the
        footprint grows with serving: products becoming resident and eager
        work. Those belong to the owner's storage grant, not to graph
        residency.
        """
        self._outside.clear()

    def resident_bytes(self):
        """Return the bytes each budgeted device charges against its budget.

        A device bound by ``set_budget`` and not yet sealed charges its pool
        bytes at binding plus the process's footprint growth since; any
        other device charges its pools (``pool_bytes``).
        """
        unbound = self._budgets.keys() - self._outside.keys()
        sizes = (
            self.pool_bytes() if unbound else dict.fromkeys(self._budgets, 0)
        )
        for device, outside in self._outside.items():
            sizes[device] = process_device_bytes(device) - outside
        return sizes

    def pool_bytes(self):
        """Return reserved pool bytes, including reusable capture workspace.

        Sums the allocator segments of every owner's pools, by device, from
        ``torch.cuda.memory_snapshot``; budgeted devices without pooled
        segments report zero.
        """
        sizes = dict.fromkeys(self._budgets, 0)
        for (_, device), used in self.owner_bytes().items():
            sizes[device] += used
        return sizes

    def owner_bytes(self):
        """Return reserved pool bytes by (owner, device).

        Owners without pooled segments on a device are omitted.
        """
        # A pool several owners share is attributed to the first of them.
        ids: dict[
            tuple[int | None, tuple[int, ...]], tuple[object, torch.device]
        ] = {}
        for owner, pools in self._pools.items():
            for device, pool in pools.items():
                ids.setdefault((device.index, tuple(pool.id)), (owner, device))
        sizes: dict = {}
        if not ids:
            return sizes
        for segment in torch.cuda.memory_snapshot():
            key = ids.get(
                (
                    segment["device"],
                    tuple(segment.get("segment_pool_id", ())),
                )
            )
            if key is not None:
                sizes[key] = sizes.get(key, 0) + segment["total_size"]
        return sizes

    def release(self, owner):
        """Stop accounting for ``owner``'s pools and drop this reference.

        Callers release a pool only after its graphs and borrowed views
        retire.
        """
        self._pools.pop(owner, None)

    def close(self):
        self._pools.clear()
        self._outside.clear()
