"""Worker backing for engine-placed cross-call buffers.

The engine places each cross-call buffer at a byte offset in the pool of
`WorkerInfo.buffer_pool_bytes` this rank reports (`BufferAllocation`), from
the components it knows this rank executes and reads. `BufferPool` backs that
pool with one fixed byte arena per worker device and returns typed tensor
views over it; `TensorStore` binds and releases those views for the products
it records. A binding occupies exactly the engine's
``[offset, offset + bytes)`` span.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock

import torch

from uniserve.runtime.device import canonical_device
from uniserve_worker.errors import (
    ErrorCode,
    WorkerError,
    invalid_descriptor,
)
from uniserve_worker.protocol.batch import BufferAllocation
from uniserve_worker.protocol.identity import BufferId
from uniserve_worker.protocol.tensor import TensorRef


def _invariant(message: str) -> WorkerError:
    """Construct a classified invariant error for persistent-buffer misuse."""
    return WorkerError(
        code=ErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


@dataclass(frozen=True, slots=True)
class BufferBinding:
    """Binds a logical buffer allocation to its validated tensor view.

    ``offset`` and ``bytes`` give the byte span reserved in the device arena,
    which can exceed the tensor's own bytes.
    ``binding_id`` is unique per pool and distinguishes successive bindings
    of the same buffer, so `BufferPool.release` rejects a stale binding.
    """

    buffer: BufferId
    offset: int
    bytes: int
    binding_id: int
    device_name: str
    tensor: torch.Tensor


class BufferPool:
    """One fixed byte-addressed arena per worker-owned device.

    Each arena is ``byte_capacity`` uint8 bytes. Live bindings on one device
    never overlap; `bind` enforces this under a lock.
    """

    def __init__(
        self,
        *,
        byte_capacity: int,
        devices: tuple[torch.device | str, ...],
    ) -> None:
        """Allocate one persistent arena per distinct device.

        Devices are canonicalized and deduplicated; an empty ``devices``
        tuple allocates a single CPU arena.

        Raises:
            ValueError: When ``byte_capacity`` is negative.
        """
        self.byte_capacity = int(byte_capacity)
        if self.byte_capacity < 0:
            raise ValueError("persistent buffer capacity must not be negative")

        normalized: list[torch.device] = []
        for raw in devices:
            device = canonical_device(raw)
            if device not in normalized:
                normalized.append(device)
        if not normalized:
            normalized.append(torch.device("cpu"))
        self.devices = tuple(normalized)

        self._arenas = {}
        for device in self.devices:
            if device.type == "cuda":
                from uniserve_kernels.peer_storage import empty

                # Publications borrow these persistent allocations directly.
                # Their physical mappings retire independently of model streams.
                arena = empty(
                    (self.byte_capacity,), dtype=torch.uint8, device=device
                )
            else:
                arena = torch.empty(
                    (self.byte_capacity,), dtype=torch.uint8, device=device
                )
            self._arenas[str(device)] = arena

        self._active: dict[tuple[str, BufferId], BufferBinding] = {}
        self._next_binding_id = 1
        self._lock = RLock()

    def bind(
        self,
        reference: TensorRef,
        allocation: BufferAllocation,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> BufferBinding:
        """Validate a buffer allocation and return its device tensor view.

        The view has ``shape`` and ``dtype`` and starts at the reserved
        span's first byte. The binding stays live until `release` receives
        this exact binding.

        Raises:
            WorkerError: ``invalid_descriptor`` when the device is not one of
                the pool's devices, the allocation names another buffer, the
                tensor is empty or larger than the allocation, its offset is
                not aligned for ``dtype``, the buffer is already bound on the
                device, the span exceeds the arena, or it overlaps a live
                binding.
        """
        target = canonical_device(device)
        device_name = str(target)
        arena = self._arenas.get(device_name)
        if arena is None:
            raise invalid_descriptor(
                "buffer allocation names an undeclared worker device"
            )
        if allocation.buffer != reference.buffer_id:
            raise invalid_descriptor(
                "buffer allocation does not name its output"
            )
        required = int(math.prod(shape)) * int(
            torch.empty((), dtype=dtype).element_size()
        )
        if required < 1 or required > int(allocation.bytes):
            raise invalid_descriptor(
                "buffer allocation is smaller than its output tensor"
            )
        element_bytes = int(torch.empty((), dtype=dtype).element_size())
        if int(allocation.offset) % element_bytes != 0:
            raise invalid_descriptor(
                "buffer allocation is not aligned for its output dtype"
            )

        key = (device_name, allocation.buffer)
        with self._lock:
            if key in self._active:
                raise invalid_descriptor("buffer allocation is already bound")

            start = int(allocation.offset)
            extent = int(allocation.bytes)
            end = start + extent
            if end > self.byte_capacity:
                raise invalid_descriptor(
                    "buffer allocation exceeds the worker buffer pool"
                )

            for active in self._active.values():
                if active.device_name != device_name:
                    continue
                if start < active.offset + active.bytes and active.offset < end:
                    raise invalid_descriptor(
                        "buffer allocation overlaps a live worker buffer"
                    )

            # The view covers only the tensor's bytes; the rest of the extent
            # stays reserved but unaddressed.
            tensor = arena.narrow(0, start, required).view(dtype).reshape(shape)
            binding = BufferBinding(
                buffer=allocation.buffer,
                offset=start,
                bytes=extent,
                binding_id=self._next_binding_id,
                device_name=device_name,
                tensor=tensor,
            )
            self._next_binding_id += 1
            self._active[key] = binding
            return binding

    def release(self, binding: BufferBinding) -> None:
        """Release one generation-tagged persistent buffer binding.

        The caller must have retired every use of the binding's tensor; the
        pool does not track device work or reader grants.

        Raises:
            WorkerError: A fatal ``INVARIANT_VIOLATION`` when the binding is
                no longer the live one for its buffer.
        """
        key = (binding.device_name, binding.buffer)
        with self._lock:
            current = self._active.get(key)
            if current is None or current.binding_id != binding.binding_id:
                raise _invariant("stale persistent buffer binding")
            self._active.pop(key)

    def close(self) -> None:
        """Drop every binding and arena reference.

        The caller must first retire all device uses and reader grants of the
        arenas; this method does not wait for them.
        """
        with self._lock:
            self._active.clear()
            self._arenas.clear()
