"""Native media publication hands readable bytes to a separate owner."""

from multiprocessing import shared_memory

import pytest
from bindings import publish_media_bytes


def test_media_bytes_are_readable_until_the_receiver_unlinks():
    payload = b"encoded media\x00\xff"
    segment = shared_memory.SharedMemory(name=publish_media_bytes(payload))
    try:
        assert bytes(segment.buf) == payload
    finally:
        segment.close()
        segment.unlink()


def test_empty_media_is_refused():
    with pytest.raises(RuntimeError, match="empty"):
        publish_media_bytes(b"")
