"""Closed transfer-handle family for typed product components (dormant).

Stage 3's transfer slice from ``specs/unified_forward_execution.md``:
`TransferHandle` is a closed tagged union — local residency, shared memory,
CUDA IPC, and Mooncake — where each variant carries only the descriptor
fields its genuine transport requires. Only residency and the transfer layer
interpret the tag; schedulers route the opaque typed descriptor.

This dormant slice binds two genuine transports and types the rest:

* :class:`LocalResidencyHandle` — same-process consumption of a committed
  product lease (identity only; no bytes move).
* :class:`SharedMemoryHandle` — a real POSIX shared-memory adapter
  (create/export/attach/close/unlink) used for cross-process host staging.
* :class:`CudaIpcHandle` and :class:`MooncakeHandle` — typed variants whose
  transports register at startup; a deployment configured for them without
  the dependency fails readiness rather than substituting a fake transport
  (`TransferUnavailable`).

Per the spec, all memory registration happens before READY, and transfer
selection is static per deployment edge; the registry here enforces the
fail-closed half of that contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Union

__all__ = [
    "CudaIpcHandle",
    "LocalResidencyHandle",
    "MooncakeHandle",
    "SharedMemoryHandle",
    "SharedMemoryTransport",
    "TransferHandle",
    "TransferRegistry",
    "TransferUnavailable",
]


class TransferUnavailable(RuntimeError):
    """A configured transport dependency is missing (startup failure)."""


@dataclass(frozen=True, slots=True)
class LocalResidencyHandle:
    """Same-process product consumption: identity, never bytes."""

    lease_id: int
    byte_extent: int


@dataclass(frozen=True, slots=True)
class SharedMemoryHandle:
    """One POSIX shared-memory segment holding a product component."""

    segment_name: str
    byte_extent: int


@dataclass(frozen=True, slots=True)
class CudaIpcHandle:
    """CUDA IPC memory handle bytes plus the exact component extent."""

    handle_bytes: bytes
    device_index: int
    byte_extent: int


@dataclass(frozen=True, slots=True)
class MooncakeHandle:
    """Mooncake segment descriptor (external library owns the format)."""

    descriptor: bytes
    byte_extent: int


TransferHandle = Union[
    LocalResidencyHandle,
    SharedMemoryHandle,
    CudaIpcHandle,
    MooncakeHandle,
]


class SharedMemoryTransport:
    """Genuine shared-memory adapter: export bytes, attach elsewhere."""

    def __init__(self) -> None:
        self._owned: dict[str, shared_memory.SharedMemory] = {}

    def export(self, name: str, payload: bytes) -> SharedMemoryHandle:
        segment = shared_memory.SharedMemory(
            name=name, create=True, size=max(len(payload), 1)
        )
        buffer = segment.buf
        assert buffer is not None
        buffer[: len(payload)] = payload
        self._owned[segment.name] = segment
        return SharedMemoryHandle(
            segment_name=segment.name, byte_extent=len(payload)
        )

    @staticmethod
    def attach(handle: SharedMemoryHandle) -> bytes:
        segment = shared_memory.SharedMemory(name=handle.segment_name)
        try:
            buffer = segment.buf
            assert buffer is not None
            return bytes(buffer[: handle.byte_extent])
        finally:
            segment.close()

    def release(self, handle: SharedMemoryHandle) -> None:
        segment = self._owned.pop(handle.segment_name, None)
        if segment is not None:
            segment.close()
            segment.unlink()

    def close(self) -> None:
        for name in list(self._owned):
            self.release(
                SharedMemoryHandle(segment_name=name, byte_extent=0)
            )


class TransferRegistry:
    """Static per-deployment transport binding; missing deps fail readiness."""

    def __init__(
        self,
        *,
        shared_memory_transport: SharedMemoryTransport | None = None,
        cuda_ipc_available: bool = False,
        mooncake_available: bool = False,
    ) -> None:
        self._shared_memory = shared_memory_transport
        self._cuda_ipc = cuda_ipc_available
        self._mooncake = mooncake_available

    def require(self, handle_type: type) -> None:
        if handle_type is LocalResidencyHandle:
            return
        if handle_type is SharedMemoryHandle:
            if self._shared_memory is None:
                raise TransferUnavailable(
                    "shared-memory transport is not configured"
                )
            return
        if handle_type is CudaIpcHandle:
            if not self._cuda_ipc:
                raise TransferUnavailable("CUDA IPC transport is not configured")
            return
        if handle_type is MooncakeHandle:
            if not self._mooncake:
                raise TransferUnavailable("Mooncake transport is not configured")
            return
        raise TransferUnavailable(
            f"{handle_type.__name__} is not a transfer handle variant"
        )
