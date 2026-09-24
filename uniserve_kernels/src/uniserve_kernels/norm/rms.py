"""Row-wise RMS normalization and residual-add RMS normalization.

These Triton kernels back ``uniserve.nn.functional.rms_norm`` and
``add_rms_norm``, which check the weight, residual and outputs against ``x``,
allocate outputs the caller does not supply, and evaluate the same formula
with tensor operations whenever :func:`can_run` rejects a call. One program
normalizes one contiguous last-axis row with FP32 statistics and rounds once
when storing to the output dtype.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

#: One program reduces a complete row, bounding the normalized width.
MAX_WIDTH = 8192
# Power-of-two row blocks at least this wide launch with 8 warps, narrower
# blocks with 4 (see ``_launch``).
_WIDE_BLOCK = 2048


if triton is not None:

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        """Normalize one flattened hidden-state row per Triton program."""
        # ``n_cols`` and ``eps`` are constexpr, so Triton compiles one kernel
        # variant per distinct (width, eps) pair. Rows are addressed as
        # ``row * n_cols``, which requires contiguous rows.
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # Variance and scaling stay in fp32; the output store performs the
        # conversion to the destination tensor's dtype.
        var = tl.sum(x * x, axis=0) / n_cols
        y = x * tl.rsqrt(var + eps) * w
        tl.store(y_ptr + row * n_cols + offs, y, mask=mask)

    @triton.jit
    def _add_rms_norm_kernel(
        x_ptr,
        r_ptr,
        w_ptr,
        y_ptr,
        c_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        """Add a residual, preserve the sum, and normalize it in one program."""
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        r = tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        c = x + r

        # Both outputs derive from the same fp32 sum: ``c_ptr`` carries the
        # residual stream and ``y_ptr`` carries its normalized projection.
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(c * c, axis=0) / n_cols
        y = c * tl.rsqrt(var + eps) * w
        tl.store(c_ptr + row * n_cols + offs, c, mask=mask)
        tl.store(y_ptr + row * n_cols + offs, y, mask=mask)


def can_run(x: torch.Tensor, weight: torch.Tensor, *outputs) -> bool:
    """Return whether contiguous CUDA rows and outputs fit one program each.

    ``outputs`` lists every other row tensor the kernel touches: the result
    buffers, and for ``add_rms_norm`` also the residual input. Each must share
    ``x``'s shape and device and be contiguous. Outputs may alias the inputs:
    each program loads its whole row before storing. Residual and output
    dtypes are not checked here; the ``uniserve.nn.functional`` callers
    enforce them. Returns False while grad mode is enabled; the kernels
    define no autograd backward.
    """
    width = int(x.shape[-1]) if x.ndim else 0
    return (
        launchable(x.device)
        and not torch.is_grad_enabled()
        and x.is_cuda
        and x.numel() > 0
        and 0 < width <= MAX_WIDTH
        and weight.shape == (width,)
        and weight.device == x.device
        and x.is_contiguous()
        and weight.is_contiguous()
        and all(
            value.shape == x.shape
            and value.device == x.device
            and value.is_contiguous()
            for value in outputs
        )
    )


def _launch(width: int) -> tuple[int, int]:
    """Return the power-of-two row block covering ``width`` and its warps."""
    block = triton.next_power_of_2(width)
    return block, 8 if block >= _WIDE_BLOCK else 4


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor
) -> None:
    """Store ``x * rsqrt(mean(x^2) + eps) * weight`` into ``out``."""
    width = int(x.shape[-1])
    block, warps = _launch(width)
    _rms_norm_kernel[(x.numel() // width,)](
        x, weight, out, width, float(eps), block, num_warps=warps
    )


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    summed: torch.Tensor,
) -> None:
    """Store the residual sum and its RMS normalization.

    Both outputs derive from the unrounded FP32 sum ``x + residual``.
    """
    width = int(x.shape[-1])
    block, warps = _launch(width)
    _add_rms_norm_kernel[(x.numel() // width,)](
        x,
        residual,
        weight,
        out,
        summed,
        width,
        float(eps),
        block,
        num_warps=warps,
    )
