"""Worker backing for scheduler-placed cross-operation buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock

import torch

from ..execution.batch import BufferId, BufferPlacement, ProductRef
from ..foundation.errors import WorkerError, WorkerErrorCode, invalid_descriptor
from .device import canonical_device


def _invariant(message: str) -> WorkerError:
    return WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


@dataclass(frozen=True, slots=True)
class PersistentBufferBinding:
    buffer: BufferId
    offset: int
    bytes: int
    binding_id: int
    device_name: str
    tensor: torch.Tensor


class PersistentBuffers:
    """One fixed byte-addressed arena per worker-owned device."""

    def __init__(
        self,
        *,
        byte_capacity: int,
        devices: tuple[torch.device | str, ...],
    ) -> None:
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
        self._arenas = {
            str(device): torch.empty(
                (self.byte_capacity,), dtype=torch.uint8, device=device
            )
            for device in self.devices
        }
        self._active: dict[tuple[str, BufferId], PersistentBufferBinding] = {}
        self._next_binding_id = 1
        self._lock = RLock()

    def bind(
        self,
        reference: ProductRef,
        placement: BufferPlacement,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> PersistentBufferBinding:
        target = canonical_device(device)
        device_name = str(target)
        arena = self._arenas.get(device_name)
        if arena is None:
            raise invalid_descriptor("buffer placement names an undeclared worker device")
        if placement.buffer != reference.buffer_id:
            raise invalid_descriptor("buffer placement does not name its output")
        required = int(math.prod(shape)) * int(torch.empty((), dtype=dtype).element_size())
        end = int(placement.offset) + int(placement.bytes)
        if required < 1 or required > int(placement.bytes):
            raise invalid_descriptor("buffer placement is smaller than its output tensor")
        if end > self.byte_capacity:
            raise invalid_descriptor("buffer placement exceeds the worker buffer pool")
        element_bytes = int(torch.empty((), dtype=dtype).element_size())
        if int(placement.offset) % element_bytes != 0:
            raise invalid_descriptor("buffer placement is not aligned for its output dtype")
        key = (device_name, placement.buffer)
        with self._lock:
            if key in self._active:
                raise invalid_descriptor("buffer placement is already bound")
            start = int(placement.offset)
            for active in self._active.values():
                if active.device_name != device_name:
                    continue
                if start < active.offset + active.bytes and active.offset < end:
                    raise invalid_descriptor("buffer placement overlaps a live worker buffer")
            tensor = arena.narrow(0, start, required).view(dtype).reshape(shape)
            binding = PersistentBufferBinding(
                buffer=placement.buffer,
                offset=start,
                bytes=int(placement.bytes),
                binding_id=self._next_binding_id,
                device_name=device_name,
                tensor=tensor,
            )
            self._next_binding_id += 1
            self._active[key] = binding
            return binding

    def release(self, binding: PersistentBufferBinding) -> None:
        key = (binding.device_name, binding.buffer)
        with self._lock:
            current = self._active.get(key)
            if current is None or current.binding_id != binding.binding_id:
                raise _invariant("stale persistent buffer binding")
            self._active.pop(key)

    def close(self) -> None:
        with self._lock:
            self._active.clear()
            self._arenas.clear()
