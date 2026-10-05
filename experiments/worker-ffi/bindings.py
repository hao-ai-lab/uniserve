"""Reflected objects from the standalone TVM-FFI worker experiment."""

import tvm_ffi

_module = None


class Executor(tvm_ffi.Object):
    """Bounded numerical execution; methods and state are defined in Rust."""


class Submission(tvm_ffi.Object):
    """An executor submission retaining its outstanding numerical accesses."""


class WorkerRequest(tvm_ffi.Object):
    """The existing Rust worker request, decoded without Python records."""


class RequestPool(tvm_ffi.Object):
    """The production request lifecycle, shared with the numerical worker."""


class HostLane(tvm_ffi.Object):
    """The production native host executor, called through TVM-FFI."""


class HostTask(tvm_ffi.Object):
    """A native host result; numerical tensors remain inside the callback."""


class HostBuffers(tvm_ffi.Object):
    """Host input tensors and native copy fences shared with production."""


class TransferPool(tvm_ffi.Object):
    """Native reads, copy streams and physical credit returns."""


class TransferTicket(tvm_ffi.Object):
    """Native read readiness and retirement with a DLPack destination view."""


class CUDAStream(tvm_ffi.Object):
    """Native stream, SM partition and submission fences."""


class CUDAEvent(tvm_ffi.Object):
    """A native device dependency that retains its producer's SM partition."""


class DescriptorGrants(tvm_ffi.Object):
    """Allocation descriptors served by the shared native transport owner."""


class SharedBuffer(tvm_ffi.Object):
    """Native shared host allocation with a borrowed DLPack payload view."""


class SharedRead(tvm_ffi.Object):
    """Native reader claim, readiness wait and DLPack payload view."""


class VmmPool(tvm_ffi.Object):
    """Export ranges and asynchronous retirement over a DLPack allocation."""


class PoolChunk(tvm_ffi.Object):
    """A pool reservation with borrowed payload and acknowledgment tensors."""


def fetch_descriptor(endpoint, export):
    return _module.fetch_descriptor(endpoint, export)


def store_media_bytes(payload):
    return _module.store_media_bytes(payload)


def open_shared_memory(name):
    return _module.open_shared_memory(name)


def partition_streams(device, counts, event_slots=2):
    return _module.partition_streams(device, counts, event_slots)


def load_library(filename: str) -> None:
    """Load the explicitly selected library and bind its reflected methods."""
    global _module
    _module = tvm_ffi.load_module(filename)
    _module.register()
    tvm_ffi.register_object("uniserve.ffi.Executor")(Executor)
    tvm_ffi.register_object("uniserve.ffi.Submission")(Submission)
    tvm_ffi.register_object("uniserve.ffi.WorkerRequest")(WorkerRequest)
    tvm_ffi.register_object("uniserve.ffi.RequestPool")(RequestPool)
    tvm_ffi.register_object("uniserve.ffi.HostLane")(HostLane)
    tvm_ffi.register_object("uniserve.ffi.HostTask")(HostTask)
    tvm_ffi.register_object("uniserve.ffi.HostBuffers")(HostBuffers)
    tvm_ffi.register_object("uniserve.ffi.TransferPool")(TransferPool)
    tvm_ffi.register_object("uniserve.ffi.TransferTicket")(TransferTicket)
    tvm_ffi.register_object("uniserve.ffi.CUDAStream")(CUDAStream)
    tvm_ffi.register_object("uniserve.ffi.CUDAEvent")(CUDAEvent)
    tvm_ffi.register_object("uniserve.ffi.DescriptorGrants")(DescriptorGrants)
    tvm_ffi.register_object("uniserve.ffi.SharedBuffer")(SharedBuffer)
    tvm_ffi.register_object("uniserve.ffi.SharedRead")(SharedRead)
    tvm_ffi.register_object("uniserve.ffi.VmmPool")(VmmPool)
    tvm_ffi.register_object("uniserve.ffi.PoolChunk")(PoolChunk)
