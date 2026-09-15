"""Worker backing for scheduler-placed cross-operation buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock

import torch

from uniserve.runtime.device import canonical_device

from ..foundation.errors import WorkerError, WorkerErrorCode, invalid_descriptor
from ..protocol.batch import BufferAllocation
from ..protocol.identity import BufferId
from ..protocol.tensor import TensorRef


def _invariant(message: str) -> WorkerError:
    """Construct a classified invariant error for persistent-buffer misuse."""

    return WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


@dataclass(frozen=True, slots=True)
class BufferBinding:
    """Binds a logical buffer allocation to its validated tensor view."""

    buffer: BufferId
    physical_offset: int
    physical_bytes: int
    binding_id: int
    device_name: str
    tensor: torch.Tensor


class BufferPool:
    """One fixed byte-addressed arena per worker-owned device."""

    def __init__(
        self,
        *,
        byte_capacity: int,
        devices: tuple[torch.device | str, ...],
        compact: bool = False,
    ) -> None:
        """Allocate one persistent arena, optionally remapping logical allocations."""

        self.byte_capacity = int(byte_capacity)
        if self.byte_capacity < 0:
            raise ValueError("persistent buffer capacity must not be negative")
        self.compact = bool(compact)

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
                from uniserve_kernel.peer_memory import empty

                # Publications borrow these persistent allocations directly.
                # Their physical mappings retire independently of model streams.
                arena = empty((self.byte_capacity,), dtype=torch.uint8, device=device)
            else:
                arena = torch.empty((self.byte_capacity,), dtype=torch.uint8, device=device)
            self._arenas[str(device)] = arena

        self._active: dict[tuple[str, BufferId], BufferBinding] = {}
        self._next_binding_id = 1
        self._lock = RLock()

    def _compact_offset_locked(self, device_name: str, extent: int) -> int:
        """Return the first aligned gap that can hold one physical binding."""

        cursor = 0
        active_bindings = sorted(
            (binding for binding in self._active.values() if binding.device_name == device_name),
            key=lambda binding: binding.physical_offset,
        )

        # Bindings start on 256-byte boundaries so any dtype view stays aligned.
        for active in active_bindings:
            start = ((cursor + 255) // 256) * 256
            if start + extent <= active.physical_offset:
                return start
            cursor = max(cursor, active.physical_offset + active.physical_bytes)

        start = ((cursor + 255) // 256) * 256
        if start + extent > self.byte_capacity:
            spans = tuple(
                (binding.physical_offset, binding.physical_bytes) for binding in active_bindings
            )
            raise invalid_descriptor(
                "physical buffer allocation exceeds the worker buffer pool: "
                f"{extent} bytes requested from {self.byte_capacity} bytes with "
                f"live spans {spans}"
            )
        return start

    def bind(
        self,
        reference: TensorRef,
        allocation: BufferAllocation,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> BufferBinding:
        """Validate a buffer allocation and return its device tensor view with generation ownership."""

        target = canonical_device(device)
        device_name = str(target)
        arena = self._arenas.get(device_name)
        if arena is None:
            raise invalid_descriptor("buffer allocation names an undeclared worker device")
        if allocation.buffer != reference.buffer_id:
            raise invalid_descriptor("buffer allocation does not name its output")
        required = int(math.prod(shape)) * int(torch.empty((), dtype=dtype).element_size())
        if required < 1 or required > int(allocation.bytes):
            raise invalid_descriptor("buffer allocation is smaller than its output tensor")
        element_bytes = int(torch.empty((), dtype=dtype).element_size())
        if int(allocation.offset) % element_bytes != 0:
            raise invalid_descriptor("buffer allocation is not aligned for its output dtype")

        key = (device_name, allocation.buffer)
        with self._lock:
            if key in self._active:
                raise invalid_descriptor("buffer allocation is already bound")

            extent = ((required + 255) // 256) * 256 if self.compact else int(allocation.bytes)
            start = (
                self._compact_offset_locked(device_name, extent)
                if self.compact
                else int(allocation.offset)
            )
            end = start + extent
            if end > self.byte_capacity:
                raise invalid_descriptor("buffer allocation exceeds the worker buffer pool")

            for active in self._active.values():
                if active.device_name != device_name:
                    continue
                if (
                    start < active.physical_offset + active.physical_bytes
                    and active.physical_offset < end
                ):
                    raise invalid_descriptor("buffer allocation overlaps a live worker buffer")

            tensor = arena.narrow(0, start, required).view(dtype).reshape(shape)
            binding = BufferBinding(
                buffer=allocation.buffer,
                physical_offset=start,
                physical_bytes=extent,
                binding_id=self._next_binding_id,
                device_name=device_name,
                tensor=tensor,
            )
            self._next_binding_id += 1
            self._active[key] = binding
            return binding

    def release(self, binding: BufferBinding) -> None:
        """Release one generation-tagged persistent buffer binding."""

        key = (binding.device_name, binding.buffer)
        with self._lock:
            current = self._active.get(key)
            if current is None or current.binding_id != binding.binding_id:
                raise _invariant("stale persistent buffer binding")
            self._active.pop(key)

    def close(self) -> None:
        """Release arenas after the caller retires their device uses and reader grants."""

        with self._lock:
            self._active.clear()
            self._arenas.clear()
