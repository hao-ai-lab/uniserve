"""Shared host-to-device staging primitives for integer sidecar tensors.

Token, position, length, and scheduler block-table staging use canonical device
resolution, bounded pinned buffers, bulk CPU fill, and reusable device slots.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..foundation.errors import resource_error
from ..foundation.torch_compat import torch_is_compiling

_np: Any | None
try:  # Optional fast host packing path; minimal environments may not carry numpy.
    import numpy as _numpy_module
except Exception:  # pragma: no cover - availability depends on worker image.
    _np = None
else:  # pragma: no cover
    _np = _numpy_module

__all__ = [
    "canonical_device",
    "copy_cpu_to_device",
    "cpu_int_staging_buffer",
    "fill_cpu_ints",
    "is_pinned",
    "pack_integer_tensors",
    "TensorStager",
    "TensorStagingSlot",
    "torch_is_compiling",
]

_NUMPY_DTYPES = {torch.int32: "int32", torch.int64: "int64"}
_QUERY_BUDGET = 16


class TensorStager:
    """Bounded generation-safe staging storage for one execution pipeline."""

    def __init__(self, *, capacity: int, byte_capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.byte_capacity = int(byte_capacity)
        if self.byte_capacity < 1:
            raise ValueError("staging byte capacity must be positive")
        self._allocated_bytes = 0
        self._slots: list[dict[str, torch.Tensor]] = [{} for _ in range(self.capacity)]
        self._completion_events: list[dict[str, torch.cuda.Event]] = [
            {} for _ in range(self.capacity)
        ]
        self._generations = [0] * self.capacity
        self._active = [False] * self.capacity
        self._vacant = deque(range(self.capacity))
        self._available: deque[int] = deque()
        self._pending: deque[int] = deque()
        self._pin_memory_supported = True

    @property
    def allocated_bytes(self) -> int:
        return self._allocated_bytes

    def _install_buffer(
        self,
        slot: dict[str, torch.Tensor],
        key: str,
        buffer: torch.Tensor,
    ) -> torch.Tensor:
        existing = slot.get(key)
        prior = 0 if existing is None else int(existing.numel()) * int(existing.element_size())
        current = int(buffer.numel()) * int(buffer.element_size())
        projected = self._allocated_bytes - prior + current
        if projected > self.byte_capacity:
            raise resource_error(
                f"staging byte capacity is exhausted ({projected}>{self.byte_capacity})"
            )
        slot[key] = buffer
        self._allocated_bytes = projected
        return buffer

    def acquire(self, device: torch.device | str) -> TensorStagingSlot:
        target = canonical_device(device)
        self._reclaim_ready(_QUERY_BUDGET)
        if self._available:
            index = self._available.popleft()
        elif self._vacant:
            index = self._vacant.popleft()
        else:
            self._reclaim_ready(len(self._pending))
            if not self._available:
                raise resource_error(f"staging storage for {target} has no query-ready generation")
            index = self._available.popleft()
        generation = self._generations[index] + 1
        if generation > (1 << 32) - 1:
            generation = 1
        self._generations[index] = generation
        self._active[index] = True
        return TensorStagingSlot(self, self._slots[index], index, generation)

    def mark_submitted(
        self,
        slot: TensorStagingSlot,
        device: torch.device | str,
    ) -> None:
        target = canonical_device(device)
        if (
            slot.stager is not self
            or slot.index < 0
            or slot.index >= self.capacity
            or slot.buffers is not self._slots[slot.index]
            or not self._active[slot.index]
            or slot.generation != self._generations[slot.index]
        ):
            raise ValueError("staging slot is not the active generation owned by this stager")
        self._active[slot.index] = False
        if target.type != "cuda":
            self._available.append(slot.index)
            return
        events = self._completion_events[slot.index]
        event = events.get(str(target))
        if event is None:
            event = torch.cuda.Event(blocking=False)
            events[str(target)] = event
        event.record(torch.cuda.current_stream(target))
        self._pending.append(slot.index)

    def _reclaim_ready(self, budget: int) -> None:
        for _ in range(min(max(0, int(budget)), len(self._pending))):
            index = self._pending.popleft()
            events = self._completion_events[index]
            if all(bool(event.query()) for event in events.values()):
                self._available.append(index)
            else:
                self._pending.append(index)

    def _host_buffer(
        self,
        slot: dict[str, torch.Tensor],
        name: str,
        numel: int,
        *,
        dtype: torch.dtype,
        pin: bool,
    ) -> torch.Tensor:
        want_pin = bool(pin and self._pin_memory_supported)
        key = f"host:{dtype}:{name}"
        buffer = slot.get(key)
        if (
            buffer is None
            or int(buffer.numel()) < int(numel)
            or buffer.dtype != dtype
            or is_pinned(buffer) != want_pin
        ):
            try:
                buffer = torch.empty(int(numel), dtype=dtype, pin_memory=want_pin)
            except RuntimeError:
                self._pin_memory_supported = False
                buffer = torch.empty(int(numel), dtype=dtype)
            buffer = self._install_buffer(slot, key, buffer)
        return buffer[: int(numel)]

    def _device_buffer(
        self,
        slot: dict[str, torch.Tensor],
        name: str,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        target = canonical_device(device)
        key = f"device:{target}:{dtype}:{name}"
        buffer = slot.get(key)
        if (
            buffer is None
            or int(buffer.numel()) < int(numel)
            or buffer.dtype != dtype
            or buffer.device != target
        ):
            buffer = torch.empty(int(numel), dtype=dtype, device=target)
            buffer = self._install_buffer(slot, key, buffer)
        return buffer[: int(numel)]


@dataclass(frozen=True, slots=True)
class TensorStagingSlot:
    stager: TensorStager
    buffers: dict[str, torch.Tensor]
    index: int
    generation: int

    def host_buffer(
        self,
        name: str,
        numel: int,
        *,
        dtype: torch.dtype,
        pin: bool,
    ) -> torch.Tensor:
        return self.stager._host_buffer(
            self.buffers,
            name,
            numel,
            dtype=dtype,
            pin=pin,
        )

    def int_buffer(self, name: str, numel: int, *, pin: bool) -> torch.Tensor:
        return self.host_buffer(name, numel, dtype=torch.int32, pin=pin)

    def long_buffer(self, name: str, numel: int, *, pin: bool) -> torch.Tensor:
        return self.host_buffer(name, numel, dtype=torch.long, pin=pin)

    def device_buffer(
        self,
        name: str,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        return self.stager._device_buffer(
            self.buffers,
            name,
            numel,
            dtype=dtype,
            device=device,
        )


def canonical_device(device: torch.device | str) -> torch.device:
    """Resolve a bare ``cuda`` device to the current explicit CUDA index."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def is_pinned(tensor: torch.Tensor) -> bool:
    return bool(getattr(tensor, "is_pinned", lambda: False)())


def cpu_int_staging_buffer(
    numel: int,
    *,
    dtype: torch.dtype,
    pin: bool,
    slot: Any | None = None,
    name: str = "buffer",
) -> torch.Tensor:
    """Acquire a CPU staging tensor, recycling through ``slot`` when present.

    ``slot`` supplies reusable host and device integer buffers.
    (``int_buffer`` for int32; the concrete ``TextTensorStagingSlot`` also
    provides ``long_buffer`` for int64).
    """
    if slot is not None:
        if dtype == torch.long:
            return slot.long_buffer(name, numel, pin=pin)
        return slot.int_buffer(name, numel, pin=pin)
    if pin and not torch_is_compiling():
        try:
            return torch.empty(int(numel), dtype=dtype, pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(int(numel), dtype=dtype)


def fill_cpu_ints(cpu: torch.Tensor, values: Sequence[int]) -> None:
    """Bulk-fill a CPU integer tensor from a Python int sequence.

    One bulk host->host conversion for every size: a Python list of a few
    hundred ints converts in a single C call, which is cheaper than the N
    individual tensor ``__setitem__`` ATen ops a scalar loop issues on the
    per-forward staging path.
    """
    if len(values) == 0:
        return
    if _np is not None and cpu.dtype in _NUMPY_DTYPES:
        cpu[: len(values)].copy_(
            torch.from_numpy(_np.asarray(values, dtype=_NUMPY_DTYPES[cpu.dtype]))
        )
        return
    cpu[: len(values)].copy_(torch.as_tensor(values, dtype=cpu.dtype))


def copy_cpu_to_device(
    cpu: torch.Tensor,
    *,
    device: torch.device,
    non_blocking: bool,
    slot: Any | None,
    name: str,
) -> torch.Tensor:
    """Copy a staged CPU tensor to ``device``, reusing a slot device buffer."""
    device = canonical_device(device)
    if slot is None or device.type != "cuda":
        return cpu.to(device=device, non_blocking=non_blocking)
    out = slot.device_buffer(name, int(cpu.numel()), dtype=cpu.dtype, device=device)
    out.copy_(cpu, non_blocking=non_blocking)
    return out


def pack_integer_tensors(
    values: Sequence[torch.Tensor],
    *,
    device: torch.device | str,
    slot: TensorStagingSlot | None,
    name: str,
) -> torch.Tensor:
    """Pack integer tensors from host and target-device storage in input order."""
    if not values:
        raise ValueError("at least one tensor is required")
    target = canonical_device(device)
    flattened = tuple(value.reshape(-1) for value in values)
    dtype = flattened[0].dtype
    if dtype not in _NUMPY_DTYPES or any(value.dtype != dtype for value in flattened):
        raise ValueError("packed staging requires one integer dtype")
    unsupported = tuple(
        value.device for value in flattened if value.device.type != "cpu" and value.device != target
    )
    if unsupported:
        raise ValueError(f"cannot stage tensors from {unsupported[0]} to {target}")

    count = sum(int(value.numel()) for value in flattened)
    if target.type != "cuda":
        return torch.cat(tuple(value.to(device=target) for value in flattened))

    host_values = tuple(value for value in flattened if value.device.type == "cpu")
    device_values = tuple(value for value in flattened if value.device == target)
    if not device_values:
        host = cpu_int_staging_buffer(
            count,
            dtype=dtype,
            pin=True,
            slot=slot,
            name=name,
        )
        offset = 0
        for value in flattened:
            width = int(value.numel())
            host[offset : offset + width].copy_(value)
            offset += width
        return copy_cpu_to_device(
            host,
            device=target,
            non_blocking=is_pinned(host),
            slot=slot,
            name=name,
        )

    if not host_values:
        if slot is None:
            return torch.cat(device_values)
        output = slot.device_buffer(name, count, dtype=dtype, device=target)
        torch.cat(device_values, out=output)
        return output

    host = cpu_int_staging_buffer(
        count,
        dtype=dtype,
        pin=True,
        slot=slot,
        name=name,
    )
    host.zero_()
    device_indexes: list[int] = []
    offset = 0
    for value in flattened:
        width = int(value.numel())
        if value.device.type == "cpu":
            host[offset : offset + width].copy_(value)
        else:
            device_indexes.extend(range(offset, offset + width))
        offset += width
    output = copy_cpu_to_device(
        host,
        device=target,
        non_blocking=is_pinned(host),
        slot=slot,
        name=name,
    )
    if slot is None:
        packed_device_values = torch.cat(device_values)
        indexes = torch.tensor(device_indexes, dtype=torch.long, device=target)
    else:
        packed_device_values = slot.device_buffer(
            f"{name}_device_values",
            len(device_indexes),
            dtype=dtype,
            device=target,
        )
        torch.cat(device_values, out=packed_device_values)
        indexes_host = slot.long_buffer(
            f"{name}_device_indexes",
            len(device_indexes),
            pin=True,
        )
        fill_cpu_ints(indexes_host, device_indexes)
        indexes = copy_cpu_to_device(
            indexes_host,
            device=target,
            non_blocking=is_pinned(indexes_host),
            slot=slot,
            name=f"{name}_device_indexes",
        )
    output.index_copy_(0, indexes, packed_device_values)
    return output
