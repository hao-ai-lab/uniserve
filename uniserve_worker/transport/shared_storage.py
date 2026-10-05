"""Reader mappings of native media and tensor shared-memory allocations.

Readers resolve POSIX names with ``shm_open``; no shared-memory mount path
is required. The native producer owns allocation and unlinking.
"""

from __future__ import annotations

import ctypes
import mmap
import os

_SHM_LIBC = ctypes.CDLL(None, use_errno=True)
_SHM_LIBC.shm_open.restype = ctypes.c_int
_SHM_LIBC.shm_open.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_uint)


def _open_descriptor(name: str) -> int:
    canonical_name = name if name.startswith("/") else f"/{name}"
    descriptor = _SHM_LIBC.shm_open(canonical_name.encode(), os.O_RDWR, 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), canonical_name)
    return descriptor


def open_shared_storage(name: str, size: int) -> mmap.mmap:
    """Open an existing shared storage segment.

    The mapping is writable, because a consumer writes its own acknowledgment
    word in the segment's header once it has copied the payload out. The
    caller closes the returned mapping and never unlinks the segment, which
    stays owned by its producer.

    Raises:
        OSError: When `shm_open` or the mapping fails; a missing segment raises
            `FileNotFoundError`, which `ShmTransport` reports as a retired
            export.
        ValueError: When `size` exceeds the segment's size.
    """
    descriptor = _open_descriptor(name)
    try:
        return mmap.mmap(
            descriptor,
            int(size),
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(descriptor)
