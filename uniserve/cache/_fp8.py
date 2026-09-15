"""Re-encode initialized FP8 blocks when their numerical scale grows."""

from __future__ import annotations

import torch


def rescale_(
    values: torch.Tensor,
    old: torch.Tensor,
    new: torch.Tensor,
    initialized: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> None:
    """Preserve decoded values, including logical-dtype rounding, on growing blocks.

    Scale tensors are read-only during this operation. The caller commits new
    scales after re-encoding, on the same stream, then writes incoming values.
    """

    if values.is_cuda:
        # Numerical views select their backing device, independently of the
        # caller's ambient CUDA device. The context restores it on return.
        with torch.cuda.device(values.device):
            _rescale(values, old, new, initialized, dtype=dtype)
        return
    for block in range(values.shape[0]):
        if initialized[block] and new[block] > old[block]:
            decoded = (values[block].float() * old[block]).to(dtype).float()
            values[block].copy_((decoded / new[block]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn))


def _rescale(values, old, new, initialized, *, dtype):
    import triton
    import triton.language as tl

    from ._fp8_kernel import rescale_blocks

    width = values[0].numel() if values.shape[0] else 0
    if not width:
        return
    rescale_blocks[(values.shape[0],)](
        values,
        old,
        new,
        initialized,
        width,
        triton.next_power_of_2(min(width, 1024)),
        {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[dtype],
        num_warps=4,
    )
