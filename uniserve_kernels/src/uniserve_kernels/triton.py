"""Device eligibility shared by UniServe's Triton kernels.

This module holds a guarded Triton import that most ``uniserve_kernels``
modules reuse; a few attention modules guard their own import. When the
import fails, ``triton`` and ``tl`` are ``None``; modules importing them
define their ``@triton.jit`` functions only under ``if triton is not None``.

Eligibility checks such as ``activation.unsupported`` start from
:func:`unsupported_operands` and return the first unmet condition as a
reason string, or ``None`` when a kernel accepts the operands. Callers on
CUDA raise with that reason instead of evaluating a slower substitute.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - depends on the installed accelerator stack.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

__all__ = [
    "launchable",
    "records_autograd",
    "require_kernel",
    "tl",
    "triton",
    "unsupported_operands",
]


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


def records_autograd(*tensors: torch.Tensor | None) -> bool:
    """Return whether autograd would record an operation on ``tensors``.

    The kernels register no backward, so a launch while autograd records
    would return results detached from the graph. Grad mode alone does not
    matter: operands that do not require grad record nothing.
    """
    return torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    )


def unsupported_operands(*tensors: torch.Tensor | None) -> str | None:
    """Return why no Triton kernel can take these operands, or ``None``.

    Checks the conditions every kernel shares: Triton is importable, the
    operands (``None`` entries are absent optional operands) reside on one
    CUDA device, and autograd is not recording them.
    """
    present = tuple(tensor for tensor in tensors if tensor is not None)
    device = present[0].device
    if triton is None:
        return "Triton is not importable"
    if device.type != "cuda":
        return f"operands reside on {device}, not a CUDA device"
    if any(tensor.device != device for tensor in present):
        return "operands reside on different devices"
    if records_autograd(*present):
        return (
            "autograd is recording operands that require grad, and the "
            "kernel defines no backward; call under torch.inference_mode()"
        )
    return None


def require_kernel(
    function: str, reason: str | None, **operands: torch.Tensor | None
) -> None:
    """Raise when no CUDA kernel accepts a call's operands.

    ``reason`` is the first unmet kernel condition reported by a
    ``uniserve_kernels`` eligibility check, or ``None`` when a kernel accepts
    the call. CUDA calls never evaluate a tensor-operation substitute, so the
    ``ValueError`` names the function, the shape, dtype and strides of every
    named operand (``None`` marks an absent optional operand), and the
    missing capability.
    """
    if reason is None:
        return
    described = "; ".join(
        f"{name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"stride={tuple(tensor.stride())}"
        for name, tensor in operands.items()
        if tensor is not None
    )
    raise ValueError(f"{function} has no CUDA kernel for {described}: {reason}")
