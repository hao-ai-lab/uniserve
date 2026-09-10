"""SM100 block-64 attention with independent query and key extents."""

from functools import lru_cache
from pathlib import Path

import torch


def available(device: torch.device | None = None) -> bool:
    """Resolve the required native artifact before serving or CUDA capture."""

    # The extension contains sm_100a code, whose architecture-conditional
    # instructions do not carry the major-version cubin compatibility promise.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (10, 0):
        return False
    _extension()
    return True


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    source = Path(__file__).parent / "csrc" / "attention.cu"
    return load(
        "uniserve_sparse_attention_sm100",
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--use_fast_math",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xcompiler=-fno-strict-aliasing",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        extra_ldflags=["-lcuda"],
    )


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    valid_sizes: torch.Tensor,
) -> torch.Tensor:
    """Attend from head-major Q to its selected complete key blocks.

    Q/K/V have shape ``[1, heads, rows, 128]`` with independent Q/K row
    counts. Q is contiguous; K/V accept independent aligned row/head strides
    with contiguous channels, including runtime-owned peer tensor mappings.
    Metadata uses global key-block indices and per-block valid sizes.
    Callers supply indices in ``[0, key_rows / 64)``, counts within the
    indices' final extent, and valid key sizes in ``[0, 64]``. These device
    values are consumed without a host synchronization.
    The output has Q's shape, dtype and device. Compilation precedes capture.
    """

    return _extension().forward(query, key, value, indices, counts, valid_sizes)
