"""Bound initialization of contiguous blocks across physical state fields."""

from math import prod

import torch


class BlockFill:
    """Borrow physical fields with one fill value and a leading block axis each.

    The cache owner validates block bounds and serializes reuse before calling.
    Binding performs layout inspection once; no device address table or mutable
    launch workspace is needed. A completed launch resets values and flags on
    the caller's stream before a subsequent numerical consumer can use them.
    """

    def __init__(
        self, tensors: tuple[torch.Tensor, ...], values: tuple[int, ...]
    ):
        self.tensors, self.values = tensors, values
        self.widths = tuple(prod(tensor.shape[1:]) for tensor in tensors)

        from uniserve_kernels import cache

        self._cuda = cache.can_fill_blocks(tensors)

    def __call__(self, start: int, stop: int) -> None:
        """Fill a block interval on every field.

        Fill the leading-axis block interval ``[start, stop)`` on every
        field.
        """
        if self._cuda:
            from uniserve_kernels import cache

            # A public cache call follows its backing device even when the
            # calling thread currently has another CUDA device selected.
            cache.fill_blocks(
                self.tensors, self.values, self.widths, start, stop
            )
        else:
            for tensor, value in zip(self.tensors, self.values, strict=True):
                tensor[start:stop].fill_(value)
