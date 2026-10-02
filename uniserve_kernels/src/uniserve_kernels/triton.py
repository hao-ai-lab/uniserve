"""Device eligibility shared by UniServe's Triton kernels.

This module holds a guarded Triton import that most ``uniserve_kernels``
modules reuse; a few attention modules guard their own import. When the
import fails, ``triton`` and ``tl`` are ``None``; modules importing them
define their ``@triton.jit`` functions only under ``if triton is not None``.

Eligibility checks such as ``activation.unsupported`` start from
:func:`unsupported_operands` and return the first unmet condition as a
reason string, or ``None`` when a kernel accepts the operands. Callers on
CUDA raise with that reason instead of evaluating a slower substitute.

:func:`dependent_launch` decides per device whether kernels that opt in
launch as programmatic dependents of the preceding kernel on their stream;
Triton kernels open with :func:`pdl_prologue` to take part.
"""

from __future__ import annotations

import functools

import torch

try:  # pragma: no cover - depends on the installed accelerator stack.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None
    # Numerical modules remain importable without the optional GPU compiler.
    pdl_prologue = None

__all__ = [
    "dependent_launch",
    "launchable",
    "pdl_prologue",
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


@torch.compiler.assume_constant_result
@functools.cache
def dependent_launch(device: torch.device) -> bool:
    """Return whether kernels on ``device`` use programmatic dependent launch.

    ``device`` is a CUDA device with an index, such as a tensor's device.
    Programmatic dependent launch (PDL) lets the GPU schedule a kernel's
    grid while the preceding kernel on the stream still runs: the preceding
    grid releases it with ``griddepcontrol.launch_dependents``, and the
    dependent grid's ``griddepcontrol.wait`` blocks until every preceding
    grid has completed and its memory operations are visible. A kernel that
    opts in executes the wait in every thread before its first global
    memory access, so only launch, scheduling and the prologue before the
    wait overlap the preceding kernel, and results match ordinary stream
    order; a kernel launched without the attribute still waits for the
    preceding kernel to complete. CUDA graph capture records the dependency
    as a programmatic edge, so graph replays overlap the same boundaries.

    PDL is enabled on compute capability 10.x; serving latency improvements
    have been measured on GB200. Other devices launch every kernel in
    ordinary stream order.
    """
    major, _minor = torch.cuda.get_device_capability(device)
    return major == 10


if triton is not None:

    @triton.jit
    def pdl_prologue(PDL: tl.constexpr):  # noqa: N803
        """Wait for the preceding grid, then release the next one.

        A kernel launched with ``launch_pdl=PDL`` calls this before any
        global memory access, with ``PDL`` from :func:`dependent_launch`.
        The wait precedes every read of the preceding kernel's outputs and
        every write the preceding kernel could still read. The release sits
        right after it: the next grid becomes eligible once every program
        has signaled or completed. Its programs still wait for this grid's
        completion before accessing dependent data; concurrent execution
        is an optimization and is not required for progress. Without
        ``PDL`` the kernel contains neither instruction.
        """
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()


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
