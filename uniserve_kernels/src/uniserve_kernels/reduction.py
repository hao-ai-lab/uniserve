"""Reductions that finish per-program partial statistics.

The magnitude variants in ``uniserve_kernels.norm.residual`` (one partial per
row) and ``uniserve_kernels.activation`` (one partial per program) write FP32
absolute-maximum partials. ``uniserve.nn.functional`` then calls
:func:`absmax` to reduce them to one scalar on the device.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import tl, triton

if triton is not None:

    @triton.jit
    def _finish_absmax_kernel(
        partials_ptr,
        output_ptr,
        count: tl.constexpr,
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Reduce per-program magnitude partials to one scalar.

        A single program loads all ``count`` partials in one ``BLOCK`` tile;
        padding lanes read ``-inf`` so they never win the maximum.
        """
        offsets = tl.arange(0, BLOCK)
        values = tl.load(
            partials_ptr + offsets, mask=offsets < count, other=-float("inf")
        )
        tl.store(output_ptr, tl.max(values, axis=0))


def absmax(partials: torch.Tensor, out: torch.Tensor) -> None:
    """Store the maximum of FP32 magnitude partials into scalar ``out``.

    ``partials`` must be contiguous. Its length is a compile-time constant of
    the kernel, so each distinct length compiles its own variant.
    """
    _finish_absmax_kernel[(1,)](
        partials,
        out,
        count=int(partials.numel()),
        BLOCK=triton.next_power_of_2(int(partials.numel())),
        num_warps=8,
    )
