"""A media publisher hands its segment to a later receiving process."""

import subprocess
import sys
from multiprocessing import shared_memory

import pytest

from uniserve_worker._uniserve_ipc import publish_media_bytes

pytestmark = pytest.mark.unit


def test_media_remains_available_after_its_publisher_exits():
    producer = subprocess.run(
        [
            sys.executable,
            "-c",
            "from uniserve_worker._uniserve_ipc import publish_media_bytes; "
            "print(publish_media_bytes(b'encoded media'))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    segment = shared_memory.SharedMemory(name=producer.stdout.strip())
    try:
        assert bytes(segment.buf) == b"encoded media"
    finally:
        segment.close()
        segment.unlink()


def test_empty_media_is_refused():
    with pytest.raises(ValueError, match="empty"):
        publish_media_bytes(b"")
