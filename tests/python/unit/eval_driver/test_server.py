"""ManagedServer releases every resource it acquires for a launch."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from uniserve_eval.config import ServerLaunch, ServerProfile
from uniserve_eval.server import ManagedServer

pytestmark = pytest.mark.unit


def _open_descriptors(path: Path) -> list[str]:
    """Return this process's file descriptors that refer to `path`."""
    descriptors = []
    for descriptor in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except FileNotFoundError:
            # The descriptor `listdir` itself used is closed by now.
            continue
        if target == str(path):
            descriptors.append(descriptor)
    return descriptors


def test_failed_launch_closes_the_server_log(tmp_path: Path) -> None:
    log_path = tmp_path / "server.log"
    executable = tmp_path / "missing-server"
    server = ManagedServer(
        ServerProfile("local", (str(executable),), "127.0.0.1", 9, {}),
        ServerLaunch((str(executable),), tmp_path, {}),
        log_path,
        timeout_s=1.0,
    )

    with pytest.raises(FileNotFoundError):
        with server:
            pass

    assert _open_descriptors(log_path) == []
