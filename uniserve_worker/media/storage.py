"""Ownership transfer of immutable media bytes through POSIX shared storage."""

from uniserve_worker.transport.shared_storage import allocate_shared_storage


def publish_media_bytes(payload: bytes) -> str:
    """Copy final media bytes into a new shared segment and hand it off.

    Returns the segment name without its leading slash, the form
    `PosixShmArtifact` carries. On return this process holds no mapping and
    no unlink responsibility: the engine claims the segment by name when it
    receives the batch result, and its claim (`SharedMedia::open` in
    ``uniserve-core``) unlinks the name. When the copy fails, the segment is
    unlinked before the error propagates.

    Raises:
        ValueError: When ``payload`` is empty.
        RuntimeError: When the mapped segment exposes no writable buffer.
        OSError: When `allocate_shared_storage` cannot create or reserve the
            segment.
    """
    from multiprocessing import resource_tracker

    if not payload:
        raise ValueError("shared-storage media publication must not be empty")

    shm = allocate_shared_storage(len(payload))
    try:
        buffer = shm.buf
        if buffer is None:
            raise RuntimeError("shared-storage artifact has no writable buffer")
        buffer[: len(payload)] = payload
    except BaseException:
        shm.unlink()
        raise
    finally:
        shm.close()

    # Ownership passes to the consuming process, so detach this segment from
    # the local resource tracker; it must not unlink storage it no longer owns.
    # The tracker registered the slash-prefixed POSIX name, while ``shm.name``
    # reports it without the slash.
    resource_tracker.unregister("/" + shm.name.lstrip("/"), "shared_memory")
    return shm.name
