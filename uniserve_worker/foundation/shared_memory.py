"""Physically backed shared allocations used by media and transport owners."""

from __future__ import annotations

import os
from multiprocessing import shared_memory


def allocate_shared_memory(size: int) -> shared_memory.SharedMemory:
    """Allocate physically backed POSIX storage or raise before a mapped write.

    The caller owns close/unlink and any ownership transfer after publication.
    Reserving tmpfs pages avoids an uncatchable SIGBUS from a later copy when
    the shared-memory filesystem is full.
    """

    if size < 1:
        raise ValueError("shared-memory capacity must be positive")

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
