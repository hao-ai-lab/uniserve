"""Device eligibility shared by UniServe's Triton kernels."""

from __future__ import annotations

import torch

try:  # pragma: no cover - depends on the installed accelerator stack.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

__all__ = ["launchable", "tl", "triton"]


@torch.compiler.assume_constant_result
def launchable(device: torch.device | str) -> bool:
    """Return whether Triton kernels can launch on ``device``.

    Triton selects its bundled assembler for the target architecture.
    Compilation errors propagate from the kernel instead of silently
    disabling fused execution.
    """
    return (
        triton is not None
        and torch.device(device).type == "cuda"
        and torch.cuda.is_available()
    )
