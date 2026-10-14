"""SM90a CUDA block-sparse attention over 128-row tiles.

The kernel source lives in ``csrc/`` and compiles with the package into
:mod:`uniserve_kernels.attention.vsa_sm90._C`.
``uniserve.runtime.backends.attention.vsa.sm90`` routes complete 128-row
calls here.
"""

from importlib import import_module

import torch

# Rows of one query tile and of one key block.
TILE = 128


def supported(device: torch.device | None = None) -> bool:
    """Report whether ``device`` can run the extension.

    The extension contains sm_90a code, whose architecture-conditional
    instructions (GMMA, register reconfiguration) run on compute capability
    9.0 alone.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability(
        device
    ) == (9, 0)


def _extension():
    # Imported on first use: a CPU build of the package has no native module.
    return import_module("uniserve_kernels.attention.vsa_sm90._C")


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    scale: float,
    query_valid_rows: torch.Tensor | None = None,
) -> None:
    """Attend from query tiles to their selected complete key blocks.

    Q, K, V and ``out`` are BF16 ``[rows, heads, 128]`` tensors with
    contiguous channels and 16-byte-aligned row and head strides, such as
    row-major views of merged projections; each may use its own strides. Q and
    K row counts are independent multiples of 128, the rows of one query tile
    and of one key block. Metadata uses global key-block indices ``[heads,
    query tiles, selected]``, counts ``[heads, query tiles]`` and per-key-block
    valid sizes in ``[0, 128]``; these device values are consumed without a
    host synchronization.

    Only the first ``counts[h, t]`` indices of each query tile are read. The
    kernel does not range-check metadata values: counts must not exceed the
    selected width and indices must address existing key blocks. Keys past a
    block's valid size never contribute, at any positive scale, and a query
    tile with a zero count or no valid selected key produces zero output.

    ``query_valid_rows``, int32 ``[query tiles]`` in ``[0, 128]``, states the
    valid rows that lead each query tile; ``None`` states every row valid.
    Valid query rows are always computed. A tile with at most 64 valid rows
    computes its first 64 rows and writes zeros to the rest, which spares half
    its work; padded rows of other tiles are computed. The call launches on
    the current CUDA stream and writes ``out`` in place.
    """
    _extension().forward(
        query,
        key,
        value,
        out,
        indices,
        counts,
        variable_block_sizes,
        query_valid_rows,
        float(scale),
    )
