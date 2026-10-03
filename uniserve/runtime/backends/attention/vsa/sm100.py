"""SM100 block attention choosing its kernel by tile size and shape.

64-row complete calls use the native CUDA kernel for the large key domains it
was measured to serve and the CuTe kernel below that boundary; 128-row calls
use the native kernel's 128-row build, the only SM100 kernel of that tile.
Row production composes 64-row tiles and always uses the CuTe kernel.
"""

from uniserve_kernels.attention import vsa_cute, vsa_native

from . import Backend as BaseBackend
from . import Operator as BaseOperator

# Tile sizes this provider's kernels address.
TILES = (64, 128)


def available(device):
    """Report whether ``device`` runs both SM100 kernels."""
    return vsa_cute.available(device) and vsa_native.supported(device)


class _Operator(BaseOperator):
    row_kernel = staticmethod(vsa_cute.block_sparse_attention)

    def kernel(self, q, k, v, out, indices, counts, valid_sizes, *, scale):
        tile = self.pattern.tile
        if tile == 64 and vsa_cute.should_use(
            rows=k.shape[0], prefix_tiles=self.pattern.dense_prefix_tiles
        ):
            vsa_cute.block_sparse_attention(
                q, k, v, out, indices, counts, valid_sizes, scale=scale
            )
            return
        vsa_native.block_sparse_attention(
            q,
            k,
            v,
            out,
            indices,
            counts,
            valid_sizes,
            scale=scale,
            block=tile,
        )


class Backend(BaseBackend):
    operator_class = _Operator
