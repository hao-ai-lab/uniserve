"""Fine attention over an explicit per-head selected key-block domain."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn

from uniserve.nn import _binding

from .inputs import BlockInput


class BlockAttention(nn.Module):
    """Compute attention over an explicit per-head selected key-block domain.

    Each block input declares the rows of its query tiles and key blocks
    (``Pattern.tile``), and the backend that evaluates it is resolved for
    that tile.
    """

    def __init__(self, scale: float):
        super().__init__()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("VSA requires a positive scale")
        self.scale = scale

    @contextmanager
    def _operator(self, q, batch):
        """Yield the bound operator, or build a transient one outside
        serving.
        """  # noqa: D205
        binding = _binding.vsa.get().get(id(self))
        if binding is not None:
            yield binding.prepare(batch.pattern, q)
            return

        from uniserve.runtime.backends.attention import vsa
        from uniserve.runtime.tensor_buffers import TensorBuffers

        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("bind and prepare VSA before CUDA graph capture")
        provider = vsa.resolve("auto", device=q.device, tile=batch.pattern.tile)
        options = {
            "num_heads": q.shape[1],
            "head_dim": q.shape[2],
            "dtype": q.dtype,
        }
        requirements = provider.workspace_buffers(batch.pattern, **options)
        buffers = TensorBuffers.allocate(requirements, device=q.device)
        operator = provider.prepare(
            batch.pattern, **options, workspace=buffers.view(requirements)
        )
        try:
            operator.bind(batch)
            yield operator
        finally:
            operator.close()
            buffers.close()

    def forward(self, q, k, v, batch: BlockInput, *, out=None):
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        with self._operator(q, batch) as operator:
            return operator(q, k, v, batch, scale=self.scale, out=out)
