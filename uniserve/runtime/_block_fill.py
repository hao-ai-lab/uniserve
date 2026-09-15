"""Bound initialization of contiguous blocks across physical state fields."""

from math import prod

import torch

from .triton import triton_available

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only library installations use ordinary tensor fills.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fill_kernel(
        tensors,
        widths: tl.constexpr,
        values: tl.constexpr,
        tiles: tl.constexpr,
        start,
        block: tl.constexpr,
    ):
        # Grid axis 0 walks the per-field 1024-element tiles of one block,
        # concatenated across fields; axis 1 indexes blocks along the leading
        # axis, offset by ``start``. Each tensor is [block, *widths[field]].
        tile = tl.program_id(0)
        page = start + tl.program_id(1)
        first: tl.constexpr = 0
        for field in tl.static_range(len(widths)):
            if tile >= first and tile < first + tiles[field]:
                offsets = (tile - first) * block + tl.arange(0, block)
                tl.store(
                    tensors[field] + page.to(tl.int64) * widths[field] + offsets,
                    values[field],
                    offsets < widths[field],
                )
            first += tiles[field]


class BlockFill:
    """Borrow physical fields with one fill value and a leading block axis each.

    The cache owner validates block bounds and serializes reuse before calling.
    Binding performs layout inspection once; no device address table or mutable
    launch workspace is needed. A completed launch resets values and flags on
    the caller's stream before a subsequent numerical consumer can use them.
    """

    def __init__(self, tensors: tuple[torch.Tensor, ...], values: tuple[int, ...]):
        self.tensors, self.values = tensors, values
        self.widths = tuple(prod(tensor.shape[1:]) for tensor in tensors)
        self.tiles = tuple((width + 1023) // 1024 for width in self.widths)

        self._cuda = (
            triton is not None
            and bool(tensors)
            and triton_available(tensors[0].device)
            and all(
                tensor.device == tensors[0].device
                and tensor.is_contiguous()
                and tensor.dtype
                in {
                    torch.bool,
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                }
                for tensor in tensors
            )
        )

    def __call__(self, start: int, stop: int) -> None:
        """Fill the leading-axis block interval ``[start, stop)`` on every field."""

        if self._cuda:
            # A public cache operation follows its backing device even when
            # the calling thread currently has another CUDA device selected.
            with torch.cuda.device(self.tensors[0].device):
                _fill_kernel[(sum(self.tiles), stop - start)](
                    self.tensors, self.widths, self.values, self.tiles, start, 1024
                )
        else:
            for tensor, value in zip(self.tensors, self.values, strict=True):
                tensor[start:stop].fill_(value)
