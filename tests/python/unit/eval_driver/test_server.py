"""ManagedServer releases every resource it acquires for a launch."""

from __future__ import annotations

import os
import signal
import socket
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from uniserve_eval.config import ServerLaunch, ServerProfile
from uniserve_eval.server import ManagedDeployment, ManagedServer

pytestmark = pytest.mark.unit

# A stand-in server: it optionally starts a worker that stays in its process
# group, records the worker's pid, then listens until it is killed. The
# arguments are the pid file, the port, and "1" to start the worker.
_SERVER = """
import socket, subprocess, sys, time

pid_path, port, start_worker = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if start_worker == "1":
    sleeper = "import time; time.sleep(300)"
    worker = subprocess.Popen([sys.executable, "-c", sleeper])
    with open(pid_path, "w") as handle:
        handle.write(str(worker.pid))
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", port))
listener.listen()
time.sleep(300)
"""


# A stand-in server that loads for a while before it listens. The arguments
# are the port and the load time in seconds.
_LATE_SERVER = """
import socket, sys, time

port, load_s = int(sys.argv[1]), float(sys.argv[2])
time.sleep(load_s)
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", port))
listener.listen()
time.sleep(300)
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _running(pid: int) -> bool:
    """Report whether a process exists and has not exited (zombie)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    # The state letter follows the parenthesized command name.
    return stat.rpartition(")")[2].split()[0] != "Z"


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


@pytest.mark.parametrize("start_worker", [True, False], ids=["worker", "alone"])
def test_stop_terminates_the_group_after_the_server_was_killed(
    tmp_path: Path, start_worker: bool
) -> None:
    port = _free_port()
    pid_path = tmp_path / "worker.pid"
    command = (
        sys.executable,
        "-c",
        _SERVER,
        str(pid_path),
        str(port),
        "1" if start_worker else "0",
    )
    server = ManagedServer(
        ServerProfile("local", command, "127.0.0.1", port, {}),
        ServerLaunch(command, tmp_path, {}),
        tmp_path / "server.log",
        timeout_s=30.0,
    )

    worker: int | None = None
    try:
        with server:
            if start_worker:
                worker = int(pid_path.read_text())
            # An abrupt server exit, such as SIGKILL or an OOM kill, skips
            # the server's own worker cleanup.
            assert server.process is not None
            os.kill(server.process.pid, signal.SIGKILL)
            server.process.wait()

        # Exiting the context signals the group; a group left with no
        # member must not make it raise.
        if worker is not None:
            deadline = time.monotonic() + 5.0
            while _running(worker) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not _running(worker)
    finally:
        if worker is not None:
            with suppress(ProcessLookupError):
                os.kill(worker, signal.SIGKILL)


def test_startup_time_runs_from_launch_to_the_first_accepted_connection(
    tmp_path: Path,
) -> None:
    """A deployment reports how long its slowest process took to listen."""
    ports = [_free_port(), _free_port()]
    # Each stand-in loads for its delay before it binds its listener.
    delays = (0.2, 0.6)
    processes = []
    for index, (port, delay) in enumerate(zip(ports, delays, strict=True)):
        command = (
            sys.executable,
            "-c",
            _LATE_SERVER,
            str(port),
            str(delay),
        )
        processes.append(
            (
                ServerProfile(
                    f"replica-{index}", command, "127.0.0.1", port, {}
                ),
                ServerLaunch(command, tmp_path, {}),
                tmp_path / f"{index}.log",
            )
        )

    deployment = ManagedDeployment(processes, timeout_s=30.0)
    assert deployment.startup_s is None
    with deployment:
        startup = deployment.startup_s

    assert startup is not None and startup >= max(delays)
    assert startup == max(server.startup_s for server in deployment.servers)


# A stand-in server that listens at once and answers its health path with
# 503 while it warms up, then 200. The arguments are the port and the warmup
# time in seconds.
_WARMING_SERVER = """
import sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

port, warmup_s = int(sys.argv[1]), float(sys.argv[2])
started = time.monotonic()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        warm = time.monotonic() - started >= warmup_s
        self.send_response(200 if warm and self.path == "/health" else 503)
        self.end_headers()

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


@pytest.mark.parametrize("ready_path", [None, "/health"])
def test_a_readiness_path_holds_startup_until_the_server_reports_ready(
    tmp_path: Path, ready_path: str | None
) -> None:
    """A server that warms up behind its listener is ready only when warm."""
    port = _free_port()
    warmup_s = 1.0
    command = (sys.executable, "-c", _WARMING_SERVER, str(port), str(warmup_s))
    server = ManagedServer(
        ServerProfile(
            "local", command, "127.0.0.1", port, {}, ready_path=ready_path
        ),
        ServerLaunch(command, tmp_path, {}),
        tmp_path / "server.log",
        timeout_s=30.0,
    )

    with server:
        startup = server.startup_s

    assert startup is not None
    if ready_path is None:
        # The listener alone accepts long before the warmup ends.
        assert startup < warmup_s
    else:
        assert startup >= warmup_s
