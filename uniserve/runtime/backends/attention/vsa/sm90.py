"""SM90 block attention over 128-row tiles with the native CUDA kernel."""

from uniserve_kernels.attention import vsa_sm90

from . import Backend as BaseBackend
from . import Operator as BaseOperator

# Tile sizes this provider's kernels address.
TILES = (vsa_sm90.TILE,)


def available(device):
    """Report whether ``device`` runs the SM90 kernel."""
    return vsa_sm90.supported(device)


class _Operator(BaseOperator):
    def __call__(self, q, k, v, batch, *, scale, out):
        self._validate(q, k, v, batch, out)

        # The query tiles are the sequence tiles from the batch's offset, so
        # their valid rows are that window of the tiles' valid sizes; rows
        # past a tile's valid size are padding the kernel need not compute.
        start = batch.query_tile_offset
        query_valid_rows = batch.variable_block_sizes[
            start : start + q.shape[0] // self.pattern.tile
        ]
        vsa_sm90.block_sparse_attention(
            q,
            k,
            v,
            out,
            batch.q2k_index,
            batch.q2k_num,
            batch.variable_block_sizes,
            scale=scale,
            query_valid_rows=query_valid_rows,
        )
        return out


class Backend(BaseBackend):
    name = "sm90"
    operator_class = _Operator
