"""Closed transfer-handle family (Stage 3 transfer slice, dormant)."""

from __future__ import annotations

import pytest

from uniserve_worker.runtime.transfer_target import (
    CudaIpcHandle,
    LocalResidencyHandle,
    MooncakeHandle,
    SharedMemoryHandle,
    SharedMemoryTransport,
    TransferRegistry,
    TransferUnavailable,
)

pytestmark = pytest.mark.unit


def test_shared_memory_transport_round_trips_bytes():
    transport = SharedMemoryTransport()
    payload = bytes(range(200)) * 3
    handle = transport.export("uniserve-test-transfer", payload)
    try:
        assert handle.byte_extent == len(payload)
        assert SharedMemoryTransport.attach(handle) == payload
    finally:
        transport.release(handle)
    # Released segments are gone: attaching again fails at the OS level.
    with pytest.raises(FileNotFoundError):
        SharedMemoryTransport.attach(handle)


def test_missing_configured_transports_fail_closed():
    registry = TransferRegistry()
    registry.require(LocalResidencyHandle)
    with pytest.raises(TransferUnavailable, match="shared-memory"):
        registry.require(SharedMemoryHandle)
    with pytest.raises(TransferUnavailable, match="CUDA IPC"):
        registry.require(CudaIpcHandle)
    with pytest.raises(TransferUnavailable, match="Mooncake"):
        registry.require(MooncakeHandle)
    configured = TransferRegistry(
        shared_memory_transport=SharedMemoryTransport(),
        cuda_ipc_available=True,
    )
    configured.require(SharedMemoryHandle)
    configured.require(CudaIpcHandle)
    with pytest.raises(TransferUnavailable):
        configured.require(MooncakeHandle)
