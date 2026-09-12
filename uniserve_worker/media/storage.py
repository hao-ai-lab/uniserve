"""Ownership transfer of immutable media bytes through POSIX shared memory."""

from ..runtime.device import allocate_shared_memory


def publish_media_bytes(payload: bytes) -> str:
    """Transfer ownership of final media storage to the host artifact consumer."""

    from multiprocessing import resource_tracker

    if not payload:
        raise ValueError("shared-memory media publication must not be empty")
    shm = allocate_shared_memory(len(payload))
    try:
        buffer = shm.buf
        if buffer is None:
            raise RuntimeError("shared-memory artifact has no writable buffer")
        buffer[: len(payload)] = payload
    except BaseException:
        shm.unlink()
        raise
    finally:
        shm.close()
    resource_tracker.unregister("/" + shm.name.lstrip("/"), "shared_memory")
    return shm.name
