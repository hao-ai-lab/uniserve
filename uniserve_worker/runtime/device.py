"""Device identity and host integer staging for physical resource owners."""

from __future__ import annotations

import os
from collections.abc import Sequence
from multiprocessing import shared_memory
from typing import Any

import torch

__all__ = [
    "HostStagingRing",
    "canonical_device",
    "fill_cpu_bools",
    "cpu_int_staging_buffer",
    "fill_cpu_ints",
]

_np: Any | None
try:
    import numpy as _numpy_module
except Exception:
    _np = None
else:
    _np = _numpy_module

_NUMPY_DTYPES = {torch.int32: "int32", torch.int64: "int64"}


def allocate_shared_memory(size: int) -> shared_memory.SharedMemory:
    """Allocate physically backed POSIX storage or raise before a mapped write.

    The caller owns close/unlink and any ownership transfer after publication.
    Reserving tmpfs pages avoids an uncatchable SIGBUS from a later copy when
    the shared-memory filesystem is full.
    """

    if size < 1:
        raise ValueError("shared-memory capacity must be positive")
    storage = shared_memory.SharedMemory(create=True, size=size)
    try:
        descriptor = os.open(f"/dev/shm/{storage.name}", os.O_RDWR)
        try:
            os.posix_fallocate(descriptor, 0, size)
        finally:
            os.close(descriptor)
    except BaseException:
        storage.close()
        storage.unlink()
        raise
    return storage


def device_memory_budget(device: torch.device | str, fraction: float) -> tuple[int, int]:
    """Return static-pool and total free bytes after loaded CUDA resources settle."""

    target = canonical_device(device)
    if target.type != "cuda" or not 0 < fraction <= 1:
        raise ValueError("device memory sizing requires CUDA and a fraction in (0, 1]")
    torch.cuda.synchronize(target)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(target)
    return max(0, int(total * fraction) - (total - free)), free


class HostStagingRing:
    """Own generation-safe CPU sources for asynchronous host-to-device copies."""

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        *,
        dtype: torch.dtype,
        depth: int,
        device: torch.device | str,
    ) -> None:
        """Allocate a generation-safe ring of pinned host copy sources."""

        count = int(depth)
        if count < 1:
            raise ValueError("host staging depth must be positive")
        self.device = canonical_device(device)
        pin = self.device.type == "cuda"
        self._buffers = tuple(
            torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin) for _ in range(count)
        )
        self._events: list[torch.cuda.Event | None] = [None] * count
        self._cursor = 0

    def acquire(self) -> tuple[int, torch.Tensor]:
        """Lease the next pinned host integer buffer and return its generation-tagged slot."""

        if not self._buffers:
            raise RuntimeError("host staging storage is closed")
        slot = self._cursor % len(self._buffers)
        self._cursor += 1
        event = self._events[slot]
        if event is not None and not event.query():
            event.synchronize()
        return slot, self._buffers[slot]

    def release(self, slot: int) -> None:
        """Return a validated host-staging slot to the free ring."""

        if self.device.type != "cuda":
            return
        index = int(slot)
        if index < 0 or index >= len(self._buffers):
            raise ValueError("host staging slot is outside its ring")
        event = self._events[index]
        if event is None:
            event = torch.cuda.Event(blocking=False)
            self._events[index] = event
        event.record(torch.cuda.current_stream(self.device))

    def close(self) -> None:
        """Drain copies and release pinned storage while its streams still exist."""

        for event in self._events:
            if event is not None:
                event.synchronize()
        # PyTorch records allocator retirement events when pinned storage is
        # freed. Dropping only the owner's reference can postpone this until
        # after an external CUDA stream has been destroyed, for example when a
        # diagnostic traceback retains the staging owner.
        self._buffers = ()
        self._events.clear()


def canonical_device(device: torch.device | str) -> torch.device:
    """Resolve a bare ``cuda`` device to the current explicit CUDA index."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def cpu_int_staging_buffer(
    numel: int,
    *,
    dtype: torch.dtype,
    pin: bool,
) -> torch.Tensor:
    """Allocate a CPU integer staging tensor with pinned storage when available."""
    if pin:
        try:
            return torch.empty(int(numel), dtype=dtype, pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(int(numel), dtype=dtype)


def fill_cpu_ints(cpu: torch.Tensor, values: Sequence[int]) -> None:
    """Bulk-fill a CPU integer tensor from a Python int sequence."""
    if len(values) == 0:
        return
    if _np is not None and cpu.dtype in _NUMPY_DTYPES:
        cpu[: len(values)].copy_(
            torch.from_numpy(_np.asarray(values, dtype=_NUMPY_DTYPES[cpu.dtype]))
        )
        return
    cpu[: len(values)].copy_(torch.as_tensor(values, dtype=cpu.dtype))


def fill_cpu_bools(cpu: torch.Tensor, values: Sequence[bool]) -> None:
    """Bulk-fill a CPU boolean tensor from a Python boolean sequence."""
    if cpu.dtype is not torch.bool:
        raise TypeError("boolean staging requires a torch.bool destination")
    if len(values) == 0:
        return
    if _np is not None:
        cpu[: len(values)].copy_(torch.from_numpy(_np.asarray(values, dtype="bool")))
        return
    cpu[: len(values)].copy_(torch.as_tensor(values, dtype=torch.bool))
