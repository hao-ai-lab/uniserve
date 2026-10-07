"""CUDA allocation, mapping, and asynchronous host-storage primitives.

This package is the Python face of ``csrc/peer_storage.cpp``, a C++
extension built on the CUDA driver's virtual memory management API and
compiled with the package into :mod:`uniserve_kernels.peer_storage._C`. It
supplies physical allocations whose shareable handles other processes can
import, the mapping of those handles into tensors, and strided host/device
DMA.

The primitives own no transfer policy. Callers such as
``uniserve.runtime._peer_storage``, ``uniserve_worker.transport.cuda_vmm``
and ``uniserve_worker.storage.buffer_pool`` decide how handles travel between
processes, when grants are withdrawn, and when storage retires.
"""

from math import prod

import torch


def _extension():
    # Imported on first use: a CPU build of the package has no native module.
    from uniserve_kernels.peer_storage import _C

    return _C


def allocation_granularity(device: torch.device) -> int:
    """Return the CUDA device allocation granularity in bytes."""
    if device.type != "cuda":
        raise ValueError("peer tensor mappings require CUDA")
    index = (
        torch.cuda.current_device() if device.index is None else device.index
    )
    return _extension().allocation_granularity(index)


def allocate(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
):
    """Create exactly page-aligned physical storage for one peer's tensor.

    ``shape`` must be nonempty with positive extents, and its byte size must
    be an integral multiple of :func:`allocation_granularity`; the extension
    raises otherwise and does not round up (:func:`empty` does).

    The returned ``PeerAllocation`` exposes ``export_handle()``, the
    allocation's shareable handle as bytes of this device's probed type (see
    :func:`exports_fabric_handles`); ``map_local()``, a tensor of ``shape``
    over its own storage; and ``map_peers(handles)``, which maps an ordered
    list of handles, one per owner allocating the same shape and dtype, into
    one tensor whose first dimension is ``shape[0] * len(handles)``. Handle
    transport, publication, reuse, and retirement belong to the distributed
    runtime.
    """
    return _extension().PeerAllocation(
        torch.empty(0, dtype=dtype, device=device), list(shape)
    )


def empty(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Allocate exportable CUDA storage.

    Physical backing is rounded up to the device allocation granularity, and
    the storage's ``nbytes()`` reports the rounded size. The tensor owns its
    allocation and mapping. Its owner must retain it until all local device
    accesses and remote grants have retired. Logical shape and storage size
    remain distinct; views retain the complete physical backing.

    The mapping keeps the originating allocation handle, so
    :func:`export_handle` succeeds on the returned tensor and its views. An
    empty shape returns an ordinary tensor with no exportable backing.
    """
    elements = prod(shape)
    if elements == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    itemsize = torch.empty((), dtype=dtype).element_size()
    page_bytes = allocation_granularity(device)
    nbytes = ((elements * itemsize + page_bytes - 1) // page_bytes) * page_bytes
    owner = allocate((nbytes // itemsize,), dtype=dtype, device=device)
    # The mapping retains its own handle to the physical allocation, so the
    # tensor keeps the storage alive after ``owner`` is released.
    storage = owner.map_local()
    return storage[:elements].view(shape)


def exports_fabric_handles(device: int) -> bool:
    """Report whether this device exports a handle another host can import.

    A fabric handle crosses hosts inside the fabric domain; a process
    descriptor reaches only the host that created it, so an instance whose
    devices export descriptors cannot place a transfer edge across hosts.

    ``device`` is a CUDA device index, not a ``torch.device``. The extension
    probes once per device with one minimum-granularity fabric allocation and
    caches the result for the process.
    """
    return _extension().exports_fabric_handles(device)


def export_handle(tensor: torch.Tensor) -> tuple[bytes, int, int] | None:
    """Export shared storage as a shareable handle, byte capacity and offset.

    The handle is the device's probed type: a fabric handle where the driver
    exports one, which another host inside the fabric domain can import, and a
    process descriptor otherwise. Both travel as bytes so one publication
    shape carries either.

    The capacity is the byte size of the tensor's whole storage and the
    offset is the byte offset of the tensor's first element within it.

    Return None when the driver rejects the storage base as a virtual memory
    mapping (``CUDA_ERROR_INVALID_VALUE``) or when its allocation was not
    created with this device's probed handle type. Raise for an empty or
    non-CUDA tensor and for other CUDA failures. The caller retains the
    source through every reader grant.
    """
    return _extension().export_handle(tensor)


def import_handle(
    prototype: torch.Tensor, exported: bytes, allocation_bytes: int
) -> torch.Tensor:
    """Map a granted allocation on the prototype device as a flat typed tensor.

    ``exported`` is a handle from :func:`export_handle` or
    ``PeerAllocation.export_handle`` and ``allocation_bytes`` the byte size
    of the allocation it names; the result has
    ``allocation_bytes // prototype.element_size()`` elements of the
    prototype's dtype, and the caller applies the exported offset. A POSIX
    descriptor must be open in this process for the duration of the call.
    The handle's byte length must match this device's probed handle type, so
    producer and consumer devices must probe the same type.

    The returned tensor retains the imported physical handle. Retain its
    mapping until all device reads complete, then release it before
    acknowledging the source grant. This primitive does not synchronize
    consumer streams.
    """
    return _extension().import_handle(prototype, exported, allocation_bytes)


def copy_host_device(
    destination: torch.Tensor, source: torch.Tensor, stream: torch.cuda.Stream
) -> None:
    """Enqueue an exact strided copy between pinned host and CUDA storage.

    The caller owns both views through stream completion; nothing here
    records the host tensor with PyTorch's pinned allocator (see
    :func:`record_host_usage`). Shape and dtype must agree; this primitive
    performs no conversion or GPU packing allocation. Destination elements
    must be disjoint; this primitive does not check it (for caller-supplied
    transfer read destinations,
    ``uniserve_worker.transport.layout.validate_destination`` does).

    Raises:
        ValueError: ``stream`` is not on the CUDA view's device, including
            when neither view is on CUDA.
        RuntimeError: The views are not one CUDA and one pinned host tensor,
            disagree in shape or dtype, have a negative stride, or a driver
            call fails.
    """
    device = source.device if source.is_cuda else destination.device
    if device != stream.device:
        raise ValueError("host/device copy stream belongs to another device")
    _extension().copy_host_device(destination, source, int(stream.cuda_stream))


def record_host_usage(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """Retain pinned storage through a native asynchronous DMA submission.

    Call after native code enqueues a host/device copy outside PyTorch's
    copy operator. The pinned allocator delays storage reuse until this
    stream retires the copy; the owner must leave the submitted contents
    immutable meanwhile.
    """
    _extension().record_host_usage(
        tensor, stream.device_index, int(stream.cuda_stream)
    )
