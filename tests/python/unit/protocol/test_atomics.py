"""Native atomics accept only writable, contiguous, aligned storage."""

import ctypes

import pytest

from uniserve_worker._uniserve_ipc import atomic_load_u32, atomic_store_u32


def test_atomic_access_checks_the_actual_view_address_and_extent():
    storage = bytearray(16)
    address = ctypes.addressof(ctypes.c_ubyte.from_buffer(storage))
    aligned = (-address) % 4
    view = memoryview(storage)[aligned : aligned + 8]
    atomic_store_u32(view, 4, 0xFFFFFFFF)
    assert atomic_load_u32(view, 4) == 0xFFFFFFFF

    for invalid, offset in [
        (view[1:], 0),
        (view[::2], 0),
        (view.toreadonly(), 0),
        (view, 5),
        (view, 2**64 - 1),
    ]:
        with pytest.raises(RuntimeError):
            atomic_load_u32(invalid, offset)
        with pytest.raises(RuntimeError):
            atomic_store_u32(invalid, offset, 1)

    assert atomic_load_u32(view, 4) == 0xFFFFFFFF
