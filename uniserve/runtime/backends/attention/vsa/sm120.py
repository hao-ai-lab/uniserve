"""SM120 block attention through FlashInfer's 64-row block-sparse kernel.

64-row tile selections reach the kernel as they are; a 128-row tile
selection is first rewritten in the kernel's 64-row tiles
(``_half_tiles``). ``_flashinfer_sm120`` launches the kernel on the
caller's row-major ``[rows, heads, dim]`` views.
"""

from __future__ import annotations

import torch

from uniserve.tensors import BufferConfig

from . import Backend as BaseBackend
from . import Operator as BaseOperator
from . import _flashinfer_sm120, _half_tiles

# Tile sizes this provider serves.
TILES = (64, 128)


def available(device):
    """Report whether ``device`` is an SM120-family GPU the kernel serves."""
    return _half_tiles.available() and _flashinfer_sm120.available(device)


class _Operator(BaseOperator):
    def __call__(self, q, k, v, batch, *, scale, out):
        self._validate(q, k, v, batch, out)

        indices, counts, sizes = (
            batch.q2k_index,
            batch.q2k_num,
            batch.variable_block_sizes,
        )
        heads, query_tiles, width = (int(size) for size in indices.shape)
        # ``TILES`` holds the kernel's 64-row tile and 128-row tiles, so
        # every other tile is a 128-row tile.
        split = self.pattern.tile != _flashinfer_sm120.TILE

        # The kernel writes the log-sum-exp of every row; a 128-row selection
        # also needs its kernel-tile form. Both live for this call only.
        requirements = {
            "lse": BufferConfig((1, heads, q.shape[0]), torch.float32)
        }
        if split:
            requirements |= _half_tiles.work_areas(
                heads=heads,
                query_tiles=query_tiles,
                width=width,
                key_tiles=int(sizes.numel()),
            )
        work = self._work(requirements, q.device)
        if split:
            indices, counts, sizes = _half_tiles.split(
                indices, counts, sizes, work
            )

        _flashinfer_sm120.launch(
            q,
            k,
            v,
            indices,
            counts,
            sizes,
            scale=scale,
            out=out,
            lse=work["lse"],
        )
        return out

    def _work(self, requirements, device):
        # Every layer of the owning context runs on its serialized stream, so
        # they can share one backing for these per-call work areas.
        if self.transient is not None:
            return self.transient("vsa_sm120", requirements, device)
        return {
            name: torch.empty(config.shape, dtype=config.dtype, device=device)
            for name, config in requirements.items()
        }


class Backend(BaseBackend):
    name = "sm120"
    operator_class = _Operator
