"""Device eligibility shared by UniServe's Triton kernels.

This module holds a guarded Triton import that most ``uniserve_kernels``
modules reuse; a few attention modules guard their own import. When the
import fails, ``triton`` and ``tl`` are ``None``; modules importing them
define their ``@triton.jit`` functions only under ``if triton is not None``,
and eligibility checks such as ``activation.can_run`` call :func:`launchable`.
"""

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

    The check does not probe the compiler toolchain: Triton selects its
    bundled assembler for the target architecture, and compilation errors
    propagate from the kernel launch instead of silently disabling fused
    execution. ``torch.compile`` treats the result as a constant while
    tracing.
    """
    return (
        triton is not None
        and torch.device(device).type == "cuda"
        and torch.cuda.is_available()
    )
