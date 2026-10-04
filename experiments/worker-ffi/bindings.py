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
