"""Owns a benchmark server process from launch through termination.

`cli.run` wraps each benchmark point in one `ManagedServer`, so every point
measures a freshly launched server whose combined stdout and stderr go to a
per-point log file.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from .config import ServerLaunch, ServerProfile


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

        try:
            self._wait_until_ready()
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
        """Terminate the server process group and close its log."""
        self.stop()

    def stop(self) -> None:
        """Terminate the server process group and release process resources.

        The group receives SIGTERM, then SIGKILL if the leader has not exited
        within 10 seconds. Clearing `process` first makes a repeated call a
        no-op.
        """
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if self.log is not None:
            self.log.close()
            self.log = None

    def _wait_until_ready(self) -> None:
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
