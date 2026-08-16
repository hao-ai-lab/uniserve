"""Small diagnostic provenance record for one evaluator point."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "TORCHINDUCTOR_CACHE_DIR",
)


def collect_provenance(
    command: Sequence[str],
    working_directory: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    head = _command(["git", "rev-parse", "HEAD"], cwd=working_directory)
    status = _command(["git", "status", "--short"], cwd=working_directory)
    return {
        "server_command": list(command),
        "server_working_directory": str(working_directory),
        "git_head": head or None,
        "dirty": bool(status),
        "gpu": _gpu(environment.get("CUDA_VISIBLE_DEVICES")),
        "performance_environment": {
            name: environment.get(name, os.environ.get(name))
            for name in PERFORMANCE_ENVIRONMENT
            if environment.get(name, os.environ.get(name)) is not None
        },
    }


def _gpu(selector: str | None) -> dict[str, str] | None:
    command = ["nvidia-smi"]
    if selector and "," not in selector:
        command.append(f"--id={selector}")
    command.extend(["--query-gpu=name,driver_version", "--format=csv,noheader"])
    value = _command(command)
    if not value:
        return None
    first = value.splitlines()[0]
    name, separator, driver = first.partition(",")
    return {"name": name.strip(), "driver_version": driver.strip()} if separator else None


def _command(command: list[str], *, cwd: Path = ROOT) -> str:
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
