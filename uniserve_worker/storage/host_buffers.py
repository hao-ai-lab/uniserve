"""Fixed pinned sources with per-slot H2D reuse fences.

``HostBuffers`` is a small ring of preallocated CPU tensors that callers
fill on the host and copy to a CUDA device with ``non_blocking=True``. An
asynchronous copy reads its pinned source after the host call returns, so a
source must not be rewritten until that copy finishes; each slot therefore
carries the CUDA event recorded after its last copy. Users include
``BlockTables``, ``LatentPool``, the model executor's input buffers and
``EncoderRunner``.
"""

from __future__ import annotations

import torch

from uniserve.runtime.device import canonical_device


class HostBuffers:
    """Own generation-safe CPU sources for asynchronous host-to-device copies.

    Callers pair every ``acquire`` with a ``record_copy`` of the returned slot
    after enqueueing the copy that reads it, on the stream that is current at
    that point. Slots are handed out round-robin rather than tracked as
    leased: the ``depth``-th later ``acquire`` returns the same slot, and
    without a ``record_copy`` it waits only on the slot's previous fence.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        *,
        dtype: torch.dtype,
        depth: int,
        device: torch.device | str,
    ) -> None:
        """Allocate a ring of ``depth`` host copy sources of one shape.

        Sources are pinned only when ``device`` is CUDA; for other devices
        no fences are recorded.

        Raises:
            ValueError: When ``depth`` is not positive.
        """
        count = int(depth)
        if count < 1:
            raise ValueError("host staging depth must be positive")

        self.device = canonical_device(device)
        pin = self.device.type == "cuda"
        self._buffers = tuple(
            torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin)
            for _ in range(count)
        )
        self._events: list[torch.cuda.Event | None] = [None] * count
        self._cursor = 0

    def acquire(self) -> tuple[int, torch.Tensor]:
        """Return the next ring slot and its host tensor.

        Blocks the calling thread until the slot's previous copy, if any, has
        completed. The tensor keeps the contents of its previous use.

        Raises:
            RuntimeError: When the ring has been closed.
        """
        if not self._buffers:
            raise RuntimeError("host staging storage is closed")
        slot = self._cursor % len(self._buffers)
        self._cursor += 1
        # A slot is reusable only after the fence of its previous H2D copy.
        event = self._events[slot]
        if event is not None and not event.query():
            event.synchronize()
        return slot, self._buffers[slot]

    def record_copy(self, slot: int) -> None:
        """Fence the copy just enqueued from ``slot`` on the current stream.

        The next ``acquire`` of this slot waits for the recorded event. Does
        nothing for a non-CUDA device.

        Raises:
            ValueError: On a CUDA device, when ``slot`` is outside the ring.
        """
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
        """Drain copies and release pinned storage while its streams exist.

        Waits for every recorded copy, then drops the sources and events;
        a later ``acquire`` raises ``RuntimeError``.
        """
        for event in self._events:
            if event is not None:
                event.synchronize()
        # PyTorch records allocator retirement events when pinned storage is
        # freed. Dropping only the owner's reference can postpone this until
        # after an external CUDA stream has been destroyed, for example when a
        # diagnostic traceback retains the staging owner.
        self._buffers = ()
        self._events.clear()
