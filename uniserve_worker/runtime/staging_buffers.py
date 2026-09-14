"""Fixed pinned sources with per-slot H2D reuse fences."""

from __future__ import annotations

import torch

from uniserve.runtime.device import canonical_device


class StagingBuffers:
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

    def record_copy(self, slot: int) -> None:
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
