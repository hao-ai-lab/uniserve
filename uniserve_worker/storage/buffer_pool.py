"""Native physical backing for scheduler-placed cross-call buffers.

The pool owns fixed device arenas and validates binding and release in Rust.
Numerical callers borrow the pool-issued tensor views; storage owners retire
device work and transport readers before releasing a binding.
"""

from uniserve_worker._uniserve_ipc import BufferBinding, BufferPool

__all__ = ["BufferBinding", "BufferPool"]
