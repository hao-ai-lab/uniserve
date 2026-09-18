"""Masked complete-block stores from a contiguous numerical snapshot."""

from math import prod

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(
    source,
    target,
    indices,
    index_stride: tl.constexpr,
    shape: tl.constexpr,
    strides: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    target_row = tl.load(indices + row * index_stride)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    remaining = columns
    # Initialization bits are rank-one fields. Keep their addresses vectorized
    # too; only column zero participates for that one-element block.
    address = target_row * strides[0] + tl.zeros((block,), tl.int64)
    for axis in tl.static_range(len(shape) - 1, 0, -1):
        address += (remaining % shape[axis]) * strides[axis]
        remaining //= shape[axis]
    valid = (target_row >= 0) & (target_row < shape[0]) & (columns < width)
    value = tl.load(source + row * width + columns, valid, 0)
    tl.store(target + address, value, valid)


def scatter_blocks(target, indices, snapshot):
    """Write selected blocks without materializing a host selection.

    Index validation precedes this call. Bounds are also masked here so
    invalid asynchronous inputs cannot access outside the allocated backing.
    Target fields may be strided; index_select supplies a contiguous snapshot.
    """
    width = prod(target.shape[1:])
    if not indices.numel() or not width:
        return

    # One program per (block slot, 1024-column tile) of the flattened block.
    with torch.cuda.device(target.device):
        _scatter_kernel[(indices.numel(), triton.cdiv(width, 1024))](
            snapshot,
            target,
            indices,
            indices.stride(0),
            tuple(target.shape),
            target.stride(),
            width,
            1024,
        )
