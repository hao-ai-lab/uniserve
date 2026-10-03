"""SM100a CUDA block-sparse attention with independent query and key extents.

The kernel sources live in ``csrc/`` and build through
:mod:`uniserve_kernels.jit`, once per sparse-block size (64 or 128
rows): the first :func:`load` with a given source, block size and toolchain
compiles them, later processes import the finished module, and support
queries never compile.
``uniserve.runtime.backends.attention.vsa.sm100`` calls :func:`load` while
preparing its operator and routes complete calls here: block-64 calls when
``vsa_cute.should_use`` declines a shape, and every block-128 call.
"""

from functools import lru_cache
from pathlib import Path

import torch

# Sparse-block sizes the kernel source instantiates.
BLOCKS = (64, 128)


def supported(device: torch.device | None = None) -> bool:
    """Report whether ``device`` can run the extension, without compiling it.

    The extension contains sm_100a code, whose architecture-conditional
    instructions do not carry the major-version cubin compatibility promise.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability(
        device
    ) == (10, 0)


def load(block: int) -> None:
    """Compile or load the ``block``-row build before serving or CUDA capture.

    Raises:
        ValueError: ``block`` is not one of ``BLOCKS``.
    """
    _extension(block)


@lru_cache(maxsize=len(BLOCKS))
def _extension(block: int):
    if block not in BLOCKS:
        raise ValueError(f"the SM100 sparse kernel builds blocks of {BLOCKS}")
    # Build each tile size by source content. Each configuration has its own
    # symbols; the explicit architecture flag suppresses extra torch targets.
    from uniserve_kernels import jit

    directory = Path(__file__).parent / "csrc"
    configuration = ["-DVSA_BLK128=true"] if block == 128 else []
    return jit.load(
        "uniserve_sparse_attention_sm100"
        + ("_block128" if block == 128 else ""),
        [directory / "attention.cu"],
        headers=sorted(directory.glob("*.cuh")),
        cxx_flags=["-O3", "-std=c++20", *configuration],
        cuda_flags=[
            "-O3",
            "-std=c++20",
            "--use_fast_math",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xcompiler=-fno-strict-aliasing",
            "-gencode=arch=compute_100a,code=sm_100a",
            *configuration,
        ],
        ldflags=["-lcuda"],
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
    block: int,
) -> None:
    """Attend from query tiles to their selected complete key blocks.

    Q, K, V and ``out`` are BF16 ``[rows, heads, 128]`` tensors with
    contiguous channels and 16-byte-aligned row and head strides, such as
    row-major views of merged projections; each may use its own strides. Q and
    K row counts are independent multiples of ``block`` (64 or 128), the rows
    of one query tile and of one key block. Metadata uses global key-block
    indices ``[heads, query tiles, selected]``, counts ``[heads, query
    tiles]`` and per-key-block valid sizes in ``[0, block]``; these device
    values are consumed without a host synchronization. Call :func:`load`
    with the same ``block`` before CUDA capture.

    Only the first ``counts[h, t]`` indices of each query tile are read. The
    kernel does not range-check metadata values: counts must not exceed the
    selected width and indices must address existing key blocks. With
    ``scale * log2(e) <= 1``, keys past a block's valid size never contribute,
    and a query tile with a zero count or no valid selected key produces zero
    output; a larger scale can turn fully masked scores into NaN.
    Every query row is computed, including padded ones. The call launches on
    the current CUDA stream and writes ``out`` in place.
    """
    _extension(block).forward(
        query, key, value, out, indices, counts, valid_sizes, float(scale)
    )
