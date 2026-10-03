"""Owns benchmark server processes from launch through termination.

`cli.run` wraps each deployment in one `ManagedDeployment`: a single
`ManagedServer`, or one per replica. Each process's combined stdout and
stderr go to its own log file.
"""

from __future__ import annotations

import contextlib
import http.client
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from .config import ServerLaunch, ServerProfile

# Seconds a stopped server group has to exit after SIGTERM before SIGKILL.
# A serving rank leaves SIGTERM to the server, which drains its worker groups
# and closes each group's ranks together so they release the communicators
# they share; FastH3 on four GPUs takes about 12 seconds. SIGKILL during that
# teardown abandons the communicators' shared-memory segments, so the grace
# covers it with margin and only bounds a server that has stopped responding.
_STOP_GRACE_S = 60.0


class ManagedServer:
    """Runs one server process group and waits until it is ready.

    Readiness is a successful TCP connect to `ServerProfile.host` and `port`,
    so those must name the address the launch command binds; configuration
    loading does not derive one from the other. With `uniserve serve`, the
    HTTP listener is bound only after `uniserve_server::build_state` has
    resolved the model assets and started the engine. A server that listens
    earlier, such as one that warms up behind its listener, names the HTTP
    path that reports its readiness as `ServerProfile.ready_path`; readiness
    then also requires that path to answer 200.
    """

    def __init__(
        self,
        profile: ServerProfile,
        launch: ServerLaunch,
        log_path: Path,
        *,
        timeout_s: float,
    ) -> None:
        """Configure the managed launch, log, and readiness deadline."""
        self.profile = profile
        self.launch = launch
        self.log_path = log_path
        self.timeout_s = timeout_s
        self.process: subprocess.Popen[str] | None = None
        self.log: IO[str] | None = None
        # Monotonic launch time, and the seconds from it until the server was
        # first found ready.
        self.started_at: float | None = None
        self.startup_s: float | None = None

    def __enter__(self) -> Self:
        """Start the server and return once it is ready."""
        # `__exit__` does not run when `__enter__` raises, so a failed launch
        # or readiness wait releases the log and any started process here.
        try:
            self.start()
            self.wait_until_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def start(self) -> None:
        """Launch the server process without waiting for its listener.

        The caller owns `stop` from this point, including when the launch
        itself raises after opening the log.
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w", encoding="utf-8")

        # Launch values override the inherited environment. A new session
        # makes the server the leader of its own process group, which `stop`
        # signals as a whole.
        environment = dict(os.environ)
        environment.update(self.launch.environment)
        self.started_at = time.monotonic()
        self.process = subprocess.Popen(
            self.launch.command,
            cwd=self.launch.working_directory,
            env=environment,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Terminate the server process group and close its log."""
        self.stop()

    def stop(self) -> None:
        """Terminate the server process group and release process resources.

        The group receives SIGTERM whether or not the server itself is still
        running: the workers it starts inherit its group, and after an abrupt
        server exit (SIGKILL, an OOM kill) nothing else stops them. If any
        member, the server included, remains `_STOP_GRACE_S` seconds later,
        the group receives SIGKILL. Clearing `process` first makes a repeated
        call a no-op.
        """
        process = self.process
        self.process = None
        if process is not None:
            # The server's pid names the group. The kernel does not reuse it
            # while any member remains, so signalling it after the server has
            # exited reaches exactly the surviving members.
            group = process.pid
            _signal_group(group, signal.SIGTERM)

            # `poll` reaps an exited server, whose zombie would otherwise
            # keep the group non-empty. A member that exited but awaits
            # reaping by its new parent still counts, which bounds this wait
            # at the grace period.
            deadline = time.monotonic() + _STOP_GRACE_S
            while (
                process.poll() is None or _group_exists(group)
            ) and time.monotonic() < deadline:
                time.sleep(0.1)
            if process.poll() is None or _group_exists(group):
                _signal_group(group, signal.SIGKILL)
            process.wait()
        if self.log is not None:
            self.log.close()
            self.log = None

    def wait_until_ready(self) -> None:
        """Wait until the server is ready or report early process exit.

        On success, `startup_s` holds the seconds from the launch until the
        server was first found ready, its startup time under the readiness
        rule of this class.

        Raises:
            RuntimeError: If the server process exits before it is ready.
            TimeoutError: If the server is not ready before `timeout_s`
                elapses.
        """  # noqa: E501
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self.process.returncode}; "
                    f"see {self.log_path}"
                )
            if self._ready():
                assert self.started_at is not None
                self.startup_s = time.monotonic() - self.started_at
                return
            time.sleep(0.5)
        raise TimeoutError(
            f"server was not ready on "
            f"{self.profile.host}:{self.profile.port} "
            f"within {self.timeout_s}s"
        )

    def _ready(self) -> bool:
        """Probe the listener and, when configured, the readiness path once."""
        address = (self.profile.host, self.profile.port)
        try:
            with socket.create_connection(address, timeout=1):
                pass
        except OSError:
            return False
        if self.profile.ready_path is None:
            return True

        # A warming server answers its readiness path with an error status
        # until it can serve; any failure to answer counts as not ready.
        url = f"http://{address[0]}:{address[1]}{self.profile.ready_path}"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.status == 200
        except (OSError, http.client.HTTPException):
            return False


class ManagedDeployment:
    """Runs every process of one deployment and waits until all are ready.

    Replicas launch together so their model loading overlaps, and readiness
    requires every process to be ready. Any launch or readiness failure stops
    all of them, as does leaving the context.
    """

    def __init__(
        self,
        processes: list[tuple[ServerProfile, ServerLaunch, Path]],
        *,
        timeout_s: float,
    ) -> None:
        """Configure each process's profile, launch, and log path."""
        self.servers = [
            ManagedServer(profile, launch, log_path, timeout_s=timeout_s)
            for profile, launch, log_path in processes
        ]

    def __enter__(self) -> Self:
        """Start every process and return once all are ready."""
        try:
            for server in self.servers:
                server.start()
            for server in self.servers:
                server.wait_until_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Terminate every process group and close the logs."""
        self.stop()

    def stop(self) -> None:
        """Stop every process; each stop is a no-op once completed."""
        for server in self.servers:
            server.stop()

    @property
    def startup_s(self) -> float | None:
        """Seconds from the launch until every process was ready, if ready.

        The processes launch together, so the deployment is ready when its
        slowest process is.
        """
        times = [server.startup_s for server in self.servers]
        if any(value is None for value in times):
            return None
        return max(value for value in times if value is not None)

    def exited(self) -> list[Path]:
        """Return the logs of processes that are no longer running."""
        return [
            server.log_path
            for server in self.servers
            if server.process is None or server.process.poll() is not None
        ]


def _signal_group(group: int, signum: int) -> None:
    """Signal a process group, which may already have no members."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(group, signum)


def _group_exists(group: int) -> bool:
    """Report whether any process, including a zombie, is in the group."""
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True
