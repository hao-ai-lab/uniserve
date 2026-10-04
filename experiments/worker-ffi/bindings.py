"""Reflected objects from the standalone TVM-FFI worker experiment."""

import tvm_ffi

_module = None


class Executor(tvm_ffi.Object):
    """Bounded numerical execution; methods and state are defined in Rust."""


class Batch(tvm_ffi.Object):
    """A result whose tensor backing survives asynchronous device execution."""


class WorkerRequest(tvm_ffi.Object):
    """The existing Rust worker request, decoded without Python records."""


def load_library(filename: str) -> None:
    """Load the explicitly selected library and bind its reflected methods."""
    global _module
    _module = tvm_ffi.load_module(filename)
    _module.register()
    tvm_ffi.register_object("uniserve.ffi.Executor")(Executor)
    tvm_ffi.register_object("uniserve.ffi.Batch")(Batch)
    tvm_ffi.register_object("uniserve.ffi.WorkerRequest")(WorkerRequest)
