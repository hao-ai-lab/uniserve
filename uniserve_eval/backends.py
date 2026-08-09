"""Lifecycle for the single server owned by one benchmark point."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from .profiles import ServerLaunch, ServerProfile


class ManagedServer:
    def __init__(
        self,
        profile: ServerProfile,
        launch: ServerLaunch,
        log_path: Path,
        *,
        timeout_s: float,
    ) -> None:
        self.profile = profile
        self.launch = launch
        self.log_path = log_path
        self.timeout_s = timeout_s
        self.process: subprocess.Popen[str] | None = None
        self.log: IO[str] | None = None

    def __enter__(self) -> Self:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w", encoding="utf-8")
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
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def stop(self) -> None:
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
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self.process.returncode}; see {self.log_path}"
                )
            try:
                with socket.create_connection((self.profile.host, self.profile.port), timeout=1):
                    return
            except OSError:
                time.sleep(0.5)
        raise TimeoutError(
            f"server did not listen on {self.profile.host}:{self.profile.port} within {self.timeout_s}s"
        )
