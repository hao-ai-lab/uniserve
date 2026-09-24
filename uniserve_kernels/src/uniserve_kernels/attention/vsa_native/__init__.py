"""SM100a CUDA block-64 attention with independent query and key extents.

The kernel sources live in ``csrc/`` and build as a PyTorch JIT extension.
The extension compiles on first :func:`load`; support queries never compile.
``uniserve.runtime.backends.attention.vsa.sm100`` calls :func:`load` while
preparing its operator and routes complete calls here when
``vsa_cute.should_use`` declines a shape.
"""

from functools import lru_cache
from pathlib import Path

import torch


def supported(device: torch.device | None = None) -> bool:
    """Report whether ``device`` can run the extension, without compiling it.

    The extension contains sm_100a code, whose architecture-conditional
    instructions do not carry the major-version cubin compatibility promise.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability(
        device
    ) == (10, 0)


def load() -> None:
    """Compile or load the cached extension before serving or CUDA capture."""
    _extension()


@lru_cache(maxsize=1)
def _extension():
    # One build or cache lookup per process. The explicit -gencode flag pins
    # the sm_100a target; torch adds no TORCH_CUDA_ARCH_LIST targets when the
    # CUDA flags already name an architecture. -lcuda links the driver API
    # that the launcher uses to encode TMA tensor maps.
    from torch.utils.cpp_extension import load as load_extension

    source = Path(__file__).parent / "csrc" / "attention.cu"
    return load_extension(
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
    out: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    valid_sizes: torch.Tensor,
    *,
    scale: float,
) -> None:
    """Attend from query tiles to their selected complete key blocks.

    Q, K, V and ``out`` are BF16 ``[rows, heads, 128]`` tensors with
    contiguous channels and 16-byte-aligned row and head strides, such as
    row-major views of merged projections; each may use its own strides. Q and
    K row counts are independent multiples of 64. Metadata uses global key-block
    indices ``[heads, query tiles, selected]``, counts ``[heads, query
    tiles]`` and per-key-block valid sizes in ``[0, 64]``; these device values
    are consumed without a host synchronization. Call :func:`load` before
    CUDA capture.

    Only the first ``counts[h, t]`` indices of each query tile are read. The
    kernel does not range-check metadata values: counts must not exceed the
    selected width and indices must address existing key blocks. With
    ``scale * log2(e) <= 1``, keys past a block's valid size never contribute,
    and a query tile with a zero count or no valid selected key produces zero
    output; a larger scale can turn fully masked scores into NaN.
    Every query row is computed, including padded ones. The call launches on
    the current CUDA stream and writes ``out`` in place.
    """
    _extension().forward(
        query, key, value, out, indices, counts, valid_sizes, float(scale)
    )
