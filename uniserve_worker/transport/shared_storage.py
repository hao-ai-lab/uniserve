"""Physically backed shared allocations used by media and transport owners.

A producer creates a POSIX shared-memory segment with
`allocate_shared_storage` (used by `ShmTransport` and `media.storage`), and
a reader on the same host maps it by name with `open_shared_storage`, which
resolves the name with `shm_open`. `allocate_shared_storage` reserves the
new segment's pages through its path under `/dev/shm`, where
`multiprocessing.shared_memory` places segments on Linux.
"""

from __future__ import annotations

import ctypes
import mmap
import os
from multiprocessing import shared_memory


def allocate_shared_storage(size: int) -> shared_memory.SharedMemory:
    """Allocate physically backed POSIX storage or raise before a mapped write.

    The caller owns close/unlink and any ownership transfer after publication.
    Reserving tmpfs pages avoids an uncatchable SIGBUS from a later copy when
    the shared storage filesystem is full.

    Raises:
        ValueError: When `size` is not positive.
        OSError: When the segment cannot be created or its pages cannot be
            reserved; a segment already created is closed and unlinked
            first.
    """
    if size < 1:
        raise ValueError("shared storage capacity must be positive")

    storage = shared_memory.SharedMemory(create=True, size=size)
    try:
        descriptor = os.open(f"/dev/shm/{storage.name}", os.O_RDWR)
        try:
            os.posix_fallocate(descriptor, 0, size)
        finally:
            os.close(descriptor)
    except BaseException:
        storage.close()
        storage.unlink()
        raise
    return storage


_SHM_LIBC = ctypes.CDLL(None, use_errno=True)
_SHM_LIBC.shm_open.restype = ctypes.c_int


def open_shared_storage(name: str, size: int) -> mmap.mmap:
    """Open an existing shared storage segment.

    The mapping is writable, because a consumer writes its own acknowledgment
    word in the segment's header once it has copied the payload out. The
    caller closes the returned mapping and never unlinks the segment, which
    stays owned by its producer.

    Raises:
        OSError: When `shm_open` or the mapping fails; a missing segment raises
            `FileNotFoundError`, which `ShmTransport` reports as a retired
            publication.
        ValueError: When `size` exceeds the segment's size.
    """
    canonical_name = name if name.startswith("/") else f"/{name}"
    descriptor = _SHM_LIBC.shm_open(canonical_name.encode(), os.O_RDWR)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), canonical_name)
    try:
        return mmap.mmap(
            descriptor,
            int(size),
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(descriptor)
