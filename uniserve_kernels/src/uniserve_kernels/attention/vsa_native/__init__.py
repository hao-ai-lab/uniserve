"""SM100a CUDA block-sparse attention with independent query and key extents.

The kernel sources live in ``csrc/`` and compile with the package once per
sparse-block size, into :mod:`uniserve_kernels.attention.vsa_native._block64`
and :mod:`uniserve_kernels.attention.vsa_native._block128`.
``uniserve.runtime.backends.attention.vsa.sm100`` routes complete calls
here: block-64 calls when ``vsa_cute.should_use`` declines a shape, and
every block-128 call.
"""

from importlib import import_module

import torch

# Sparse-block sizes the kernel source instantiates.
BLOCKS = (64, 128)


def supported(device: torch.device | None = None) -> bool:
    """Report whether ``device`` can run the extension.

    The extension contains sm_100a code, whose architecture-conditional
    instructions do not carry the major-version cubin compatibility promise.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability(
        device
    ) == (10, 0)


def _extension(block: int):
    if block not in BLOCKS:
        raise ValueError(f"the SM100 sparse kernel builds blocks of {BLOCKS}")
    # Imported on first use: a CPU build of the package has no native module.
    return import_module(f"uniserve_kernels.attention.vsa_native._block{block}")


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
    values are consumed without a host synchronization.

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
