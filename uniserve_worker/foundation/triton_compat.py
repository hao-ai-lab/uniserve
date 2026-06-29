"""Triton runtime compatibility helpers."""
from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

from .env import env_flag

__all__ = [
    'triton_fused_layers_enabled',
    'ensure_blackwell_ptxas',
    'triton_device_supported',
]


def triton_fused_layers_enabled() -> bool:
    return env_flag("UNISERVE_TRITON_FUSED_LAYERS")


@lru_cache(maxsize=1)
def ensure_blackwell_ptxas() -> bool:
    """Point Triton at a CUDA 13+ ptxas when serving on Blackwell."""

    # TRITON_PTXAS_PATH is Triton's own env var; we intentionally read and write
    # it directly rather than through core/env.py to drive the external toolchain.
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
    return False


def triton_device_supported(device: torch.device | str) -> bool:
    dev = torch.device(device)
    try:
        major, _minor = torch.cuda.get_device_capability(dev)
    except Exception:
        return False
    if major >= 10:
        return ensure_blackwell_ptxas() or env_flag("UNISERVE_ENABLE_UNSUPPORTED_TRITON_SM100")
    return True


def _ptxas_supports_blackwell(path: Path) -> bool:
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
