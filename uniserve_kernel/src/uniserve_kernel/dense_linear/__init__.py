"""Dense half-precision projections with explicit FP32 reduction semantics."""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    return load(
        "uniserve_dense_linear",
        sources=[str(Path(__file__).parent / "csrc" / "linear.cpp")],
        extra_cflags=["-O2", "-std=c++20"],
        extra_ldflags=["-lcublas"],
        with_cuda=True,
    )


def initialize() -> None:
    """Resolve the native provider before stream capture or request execution."""

    _extension()


def linear_fp32_accum(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Project contiguous FP16/BF16 matrices with FP32 partial reductions.

    The result retains input dtype. The current CUDA stream and thread-local
    cuBLAS handle are used; handle math and pointer modes are restored after
    submission. Process-wide PyTorch precision settings remain caller-owned.
    """

    return _extension().forward(input, weight, bias)
