"""SM100a CUDA block-64 attention with independent query and key extents.

The kernel sources live in ``csrc/`` and compile with the package into
:mod:`uniserve_kernels.attention.vsa_native._block64`.
``uniserve.runtime.backends.attention.vsa.sm100`` routes complete calls here
when ``vsa_cute.should_use`` declines a shape.
"""

import torch


def supported(device: torch.device | None = None) -> bool:
    """Report whether ``device`` can run the extension.

    The extension contains sm_100a code, whose architecture-conditional
    instructions do not carry the major-version cubin compatibility promise.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability(
        device
    ) == (10, 0)


def _extension():
    # Imported on first use: a CPU build of the package has no native module.
    from uniserve_kernels.attention.vsa_native import _block64

    return _block64


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
    are consumed without a host synchronization.

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
