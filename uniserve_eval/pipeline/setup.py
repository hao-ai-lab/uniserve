"""Prepares an exclusive, resolved server launch for measurement.

`cli.run` takes `host_lock` for the whole selection, then for each point
resolves the launch with `prepare_launch`, records its provenance with
`describe_launch`, and starts the server inside `applied_environment`.
"""

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
    """Acquire the host-wide lock that serializes benchmark servers.

    The lock is an advisory `flock` on a fixed file, so it excludes only other
    uniserve-eval drivers on the same host, not servers started by other
    means. It is released when the file handle closes on exit.

    Raises:
        RuntimeError: Immediately, without waiting, if another process holds
            the lock.
    """
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
    """Apply launch environment values and restore the prior process state.

    The values apply to this process's `os.environ` for the duration of the
    block, and so to subprocesses it starts. On exit each named variable
    returns to its prior value, and one that was unset before is removed.
    """
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
    """Resolve a point's server launch and require all environment references.

    Configuration loading leaves `${NAME}` references to unset variables in
    place, and `cli.plan` renders them as written; a run rejects those in the
    launch command and the point's workload here, before any server starts.

    Raises:
        ValueError: If the launch command or the point's workload still holds
            an unresolved environment reference, or `server_launch` finds
            `--worker-python` without a path.
    """  # noqa: E501
    server = config.servers[point.server]
    launch = server_launch(server, executable, config.root)
    require_resolved(launch.command, context=f"server {server.name}")
    require_resolved(point.workload_dict(), context=f"benchmark {point.name}")
    return launch


def describe_launch(launch: ServerLaunch) -> dict[str, Any]:
    """Capture command, environment, revision, worktree, and GPU provenance.

    Git fields describe the launch working directory; `git_head` and `dirty`
    are recorded only when `git rev-parse HEAD` succeeds there. `dirty` is
    true when `git status --short` succeeds and prints anything, including
    untracked files, and `build_summary` turns it into the `dirty_workspace`
    warning. The GPU selector is read from the launch environment only, not
    from this process's inherited `CUDA_VISIBLE_DEVICES`; `gpu` is omitted
    when `_gpu` finds no name and driver version.
    """
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
    """Return the selected GPU name and driver version when available.

    Only a single-device `CUDA_VISIBLE_DEVICES` selector is passed to
    `nvidia-smi --id`. With several devices or none selected, the query
    covers every GPU and the first line is used, which describes the first
    GPU `nvidia-smi` lists rather than necessarily a selected one.
    """
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
    """Return stdout from a successful read-only provenance command.

    Returns an empty string when the executable cannot be started or exits
    with a non-zero status; stderr is discarded.
    """
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
