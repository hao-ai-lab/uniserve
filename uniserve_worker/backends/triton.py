"""Triton toolchain setup for fused worker kernels."""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

from ..foundation.env import env_flag

__all__ = [
    "configure_triton_toolchain",
    "triton_available",
]


@lru_cache(maxsize=1)
def configure_triton_toolchain() -> bool:
    """Select a CUDA 13+ assembler or honor the unsupported-toolchain opt-in."""

    existing = os.environ.get("TRITON_PTXAS_PATH")
    if existing and _ptxas_supports_blackwell(Path(existing)):
        return True
    for path in (
        Path("/usr/local/cuda/bin/ptxas"),
        Path("/usr/local/cuda-13.0/bin/ptxas"),
        Path("/usr/local/cuda-13/bin/ptxas"),
    ):
        if _ptxas_supports_blackwell(path):
            os.environ["TRITON_PTXAS_PATH"] = str(path)
            return True
    return env_flag("UNISERVE_ENABLE_UNSUPPORTED_TRITON_SM100")


@torch.compiler.assume_constant_result
def triton_available(device: torch.device | str) -> bool:
    """Return whether fused Triton kernels may run on ``device``."""

    try:
        major, _minor = torch.cuda.get_device_capability(torch.device(device))
    except Exception:
        return False
    if major >= 10:
        return configure_triton_toolchain()
    return True


def _ptxas_supports_blackwell(path: Path) -> bool:
    """Return whether the installed assembler advertises Blackwell target support."""

    if not path.is_file():
        return False
    try:
        out = subprocess.check_output(
            [str(path), "--version"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2,
        )
    except Exception:
        return False
    match = re.search(r"release\s+(\d+)\.(\d+)", out)
    if match is None:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    return (major, minor) >= (13, 0)
