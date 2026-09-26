"""Owns benchmark server processes from launch through termination.

`cli.run` wraps each deployment in one `ManagedDeployment`: a single
`ManagedServer`, or one per replica. Each process's combined stdout and
stderr go to its own log file.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from .config import ServerLaunch, ServerProfile

# Seconds a stopped server group has to exit after SIGTERM before SIGKILL.
_STOP_GRACE_S = 10.0


class ManagedServer:
    """Runs one server process group and waits for its TCP listener.

    Readiness is a successful TCP connect to `ServerProfile.host` and `port`,
    so those must name the address the launch command binds; configuration
    loading does not derive one from the other. With `uniserve serve`, the
    HTTP listener is bound only after `uniserve_server::build_state` has
    resolved the model assets and started the engine. A server that listens
    earlier would be reported ready too soon.
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

    def __enter__(self) -> Self:
        """Start the server and return after its listener accepts connections."""  # noqa: E501
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
        """Wait for the configured TCP listener or report early process exit.

        Raises:
            RuntimeError: If the server process exits before listening.
            TimeoutError: If no connection succeeds before `timeout_s`
                elapses.
        """  # noqa: E501
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self.process.returncode}; "
                    f"see {self.log_path}"
                )
            try:
                with socket.create_connection(
                    (self.profile.host, self.profile.port), timeout=1
                ):
                    return
            except OSError:
                time.sleep(0.5)
        raise TimeoutError(
            f"server did not listen on "
            f"{self.profile.host}:{self.profile.port} "
            f"within {self.timeout_s}s"
        )


class ManagedDeployment:
    """Runs every process of one deployment and waits until all listen.

    Replicas launch together so their model loading overlaps, and readiness
    requires every listener. Any launch or readiness failure stops all of
    them, as does leaving the context.
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
        """Start every process and return once all listeners accept."""
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
