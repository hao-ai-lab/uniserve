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
    """Preserve decoded values on growing blocks.

    Including logical-dtype rounding. Scale tensors are read-only during
    this call. The caller commits new scales after re-encoding, on
    the same stream, then writes incoming values.
    """
    if values.is_cuda:
        # Numerical views select their backing device, independently of the
        # caller's ambient CUDA device. The context restores it on return.
        with torch.cuda.device(values.device):
            _rescale(values, old, new, initialized, dtype=dtype)
        return

    # Host fallback: a per-block scalar loop mirroring the Triton kernel.
    for block in range(values.shape[0]):
        if initialized[block] and new[block] > old[block]:
            decoded = (values[block].float() * old[block]).to(dtype).float()
            values[block].copy_(
                (decoded / new[block])
                .clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
            )


def _rescale(values, old, new, initialized, *, dtype):
    from uniserve_kernels import cache

    cache.rescale_fp8_blocks(values, old, new, initialized, dtype)
