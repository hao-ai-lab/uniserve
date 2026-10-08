"""Row kernels of region-wise video sparse attention over 128-row tiles.

Region attention (``uniserve.nn.attention.vsa.RegionAttention``) keeps the
eager arithmetic of its checkpoint's recipe; these kernels perform two of its
row passes without materializing full-sequence temporaries and reproduce the
eager results bit for bit:

- ``zero_tile_padding`` writes zeros to the rows past each tile's valid size,
  so a plain per-tile sum of the rows equals the sum of the masked rows. Only
  padding rows are stored; a launch's shape depends on the tile count alone.
- ``add_gated_tiles`` adds each tile's gated compression to its rows,
  ``out + compressed[tile] * gate`` with the product and the sum each rounded
  to the tensors' dtype, as two eager element-wise operations round them.

Tensors are ``[rows, heads, width]`` with any row and head strides and a unit
column stride; compressed tiles are ``[heads, tiles, width]``.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

# Rows one program covers: a tile is a whole number of row blocks.
_BLOCK_ROWS = 16

if triton is not None:

    @triton.jit
    def _add_rn(a, b):
        # Round-to-nearest FP32 sum the compiler cannot contract with a
        # neighbouring multiply.
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=r,r,r",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=r,r,r",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _zero_tile_padding_kernel(
        values,
        valid_sizes,
        stride_row,
        stride_head,
        heads: tl.constexpr,
        tile_rows: tl.constexpr,
        width: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Store zeros to one tile's row block past the tile's valid size."""
        tile = tl.program_id(0)
        block = tl.program_id(1)
        valid = tl.load(valid_sizes + tile)
        # A row block wholly inside the valid rows stores nothing.
        if (block + 1) * block_rows > valid:
            local = block * block_rows + tl.arange(0, block_rows)
            rows = (tile * tile_rows + local).to(tl.int64)
            columns = tl.arange(0, width)
            padding = (local >= valid)[:, None]
            zeros = tl.zeros((block_rows, width), dtype=values.dtype.element_ty)
            for head in tl.static_range(heads):
                offsets = (
                    rows[:, None] * stride_row
                    + head * stride_head
                    + columns[None, :]
                )
                tl.store(values + offsets, zeros, mask=padding)

    @triton.jit
    def _add_gated_tiles_kernel(
        output,
        compressed,
        gate,
        output_stride_row,
        output_stride_head,
        gate_stride_row,
        gate_stride_head,
        compressed_stride_head,
        compressed_stride_tile,
        tile_rows: tl.constexpr,
        width: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Add one row block's gated tile compression for one head."""
        block = tl.program_id(0)
        head = tl.program_id(1)
        rows = (block * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        columns = tl.arange(0, width)
        # A row block lies within one tile.
        tile = (block * block_rows) // tile_rows
        tile_values = tl.load(
            compressed
            + head * compressed_stride_head
            + tile * compressed_stride_tile
            + columns
        ).to(tl.float32)
        gate_values = tl.load(
            gate
            + rows[:, None] * gate_stride_row
            + head * gate_stride_head
            + columns[None, :]
        ).to(tl.float32)
        destination = (
            output
            + rows[:, None] * output_stride_row
            + head * output_stride_head
            + columns[None, :]
        )
        # The product rounds to the storage dtype before the sum, as the two
        # eager operations store it in between.
        product = _mul_rn(tile_values[None, :], gate_values)
        product = product.to(
            output.dtype.element_ty, fp_downcast_rounding="rtne"
        ).to(tl.float32)
        total = _add_rn(tl.load(destination).to(tl.float32), product)
        tl.store(
            destination,
            total.to(output.dtype.element_ty, fp_downcast_rounding="rtne"),
        )


def _rows_layout(value: torch.Tensor, name: str) -> None:
    if value.ndim != 3 or value.stride(2) != 1:
        raise ValueError(
            f"{name} must be [rows, heads, width] with unit column stride"
        )


def zero_tile_padding(
    values: torch.Tensor, valid_sizes: torch.Tensor, tile: int
) -> None:
    """Store zeros to every row past its tile's valid size, in place.

    ``values`` is ``[tiles * tile, heads, width]`` with unit column stride
    and ``valid_sizes`` the int32 valid rows of each ``tile``-row tile, read
    on the device. Rows within a tile's valid size are not touched.

    Raises:
        RuntimeError: Triton cannot launch on the tensors' device.
        ValueError: The rows are not whole tiles of the table's tiles.
    """
    if not launchable(values.device):
        raise RuntimeError("tile padding requires Triton")
    _rows_layout(values, "tile rows")
    tiles = int(valid_sizes.numel())
    if (
        tile % _BLOCK_ROWS
        or values.shape[0] != tiles * tile
        or valid_sizes.dtype != torch.int32
    ):
        raise ValueError(
            "tile padding needs whole tiles and an int32 valid-size table"
        )
    rows, heads, width = (int(size) for size in values.shape)
    if rows == 0:
        return
    _zero_tile_padding_kernel[(tiles, tile // _BLOCK_ROWS)](
        values,
        valid_sizes,
        int(values.stride(0)),
        int(values.stride(1)),
        heads,
        tile,
        width,
        _BLOCK_ROWS,
        num_warps=4,
    )


def add_gated_tiles(
    output: torch.Tensor,
    compressed: torch.Tensor,
    gate: torch.Tensor,
    tile: int,
) -> None:
    """Add each tile's gated compression to its rows, in place.

    For every row ``r`` of tile ``t``, ``output[r] = output[r] +
    compressed[:, t] * gate[r]``, the product and the sum each rounded to
    nearest even in the tensors' common dtype: the result of
    ``output.view(tiles, tile, heads, width).add_(compressed.permute(1, 0,
    2)[:, None] * gate.view(tiles, tile, heads, width))`` evaluated eagerly.
    ``output`` and ``gate`` are ``[tiles * tile, heads, width]`` and
    ``compressed`` is ``[heads, tiles, width]``, each with unit column
    stride.

    Raises:
        RuntimeError: Triton cannot launch on the tensors' device.
        ValueError: The shapes, dtypes or devices disagree.
    """
    if not launchable(output.device):
        raise RuntimeError("gated tile compression requires Triton")
    for value, name in ((output, "output"), (gate, "gate")):
        _rows_layout(value, name)
    rows, heads, width = (int(size) for size in output.shape)
    if (
        gate.shape != output.shape
        or compressed.ndim != 3
        or compressed.stride(2) != 1
        or compressed.shape != (heads, rows // tile, width)
        or tile % _BLOCK_ROWS
        or rows % tile
        or len({value.dtype for value in (output, compressed, gate)}) != 1
        or len({value.device for value in (output, compressed, gate)}) != 1
    ):
        raise ValueError(
            "gated tile compression needs whole tiles of one dtype and device"
        )
    if rows == 0:
        return
    _add_gated_tiles_kernel[(rows // _BLOCK_ROWS, heads)](
        output,
        compressed,
        gate,
        int(output.stride(0)),
        int(output.stride(1)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        int(compressed.stride(0)),
        int(compressed.stride(1)),
        tile,
        width,
        _BLOCK_ROWS,
        num_warps=4,
    )


__all__ = ["add_gated_tiles", "zero_tile_padding"]
