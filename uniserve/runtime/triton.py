"""Device eligibility for fused numerical Triton kernels."""

from __future__ import annotations

import torch

__all__ = ["triton_available"]


@torch.compiler.assume_constant_result
def triton_available(device: torch.device | str) -> bool:
    """Return whether the device targets the fused kernels' CUDA backend.

    Triton selects its bundled assembler for the target architecture. Compilation
    errors propagate from the kernel instead of silently disabling fused execution.
    """

    return torch.device(device).type == "cuda" and torch.cuda.is_available()
