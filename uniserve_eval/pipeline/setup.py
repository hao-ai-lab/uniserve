"""Prepares an exclusive, resolved server launch for measurement."""

from __future__ import annotations

import fcntl
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..config import (
    EvaluationConfig,
    ServerLaunch,
    require_resolved,
    server_launch,
)
from ..types import BenchmarkPoint


@contextmanager
def host_lock() -> Iterator[None]:
    """Acquire the process-wide lock that serializes benchmark servers."""
    path = Path("/tmp/uniserve-eval.lock")
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another uniserve-eval process holds the host lock"
            ) from error
        yield


@contextmanager
def applied_environment(values: dict[str, str]) -> Iterator[None]:
    """Apply launch environment values and restore the prior process state."""
    previous: dict[str, str | None] = {
        name: os.environ.get(name) for name in values
    }
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def prepare_launch(
    config: EvaluationConfig,
    point: BenchmarkPoint,
    executable: Path | None,
) -> ServerLaunch:
    """Resolve a point's server launch and require all environment references."""  # noqa: E501
    server = config.servers[point.server]
    launch = server_launch(server, executable)
    require_resolved(launch.command, context=f"server {server.name}")
    require_resolved(point.workload_dict(), context=f"benchmark {point.name}")
    return launch


def describe_launch(launch: ServerLaunch) -> dict[str, Any]:
    """Capture command, environment, revision, worktree, and GPU provenance."""
    record: dict[str, Any] = {
        "command": list(launch.command),
        "working_directory": str(launch.working_directory),
        "environment": dict(launch.environment),
    }
    head = _capture(["git", "rev-parse", "HEAD"], cwd=launch.working_directory)
    if head:
        record["git_head"] = head
        record["dirty"] = bool(
            _capture(["git", "status", "--short"], cwd=launch.working_directory)
        )
    gpu = _gpu(launch.environment.get("CUDA_VISIBLE_DEVICES"))
    if gpu is not None:
        record["gpu"] = gpu
    return record


def _gpu(selector: str | None) -> dict[str, str] | None:
    """Return the selected GPU name and driver version when available."""
    command = ["nvidia-smi"]
    if selector and "," not in selector:
        command.append(f"--id={selector}")
    command.extend(["--query-gpu=name,driver_version", "--format=csv,noheader"])
    value = _capture(command)
    if not value:
        return None
    name, separator, driver = value.splitlines()[0].partition(",")
    return (
        {"name": name.strip(), "driver_version": driver.strip()}
        if separator
        else None
    )


def _capture(command: list[str], *, cwd: Path | None = None) -> str:
    """Return stdout from a successful read-only provenance command."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""
