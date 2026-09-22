"""SM100 block attention choosing its kernel by the measured shape boundary.

Complete calls use the native CUDA kernel for the large key domains it was
measured to serve and the CuTe kernel below that boundary. Row production
always uses the CuTe kernel.
"""

from uniserve_kernels.attention import vsa_cute, vsa_native

from . import Backend as BaseBackend
from . import Operator as BaseOperator


def available(device):
    """Report support without compiling the native extension."""
    return vsa_cute.available(device) and vsa_native.supported(device)


class _Operator(BaseOperator):
    row_kernel = staticmethod(vsa_cute.block_sparse_attention)

    def kernel(self, q, k, v, out, indices, counts, valid_sizes, *, scale):
        attend = (
            vsa_cute.block_sparse_attention
            if vsa_cute.should_use(
                rows=k.shape[0], prefix_tiles=self.pattern.dense_prefix_tiles
            )
            else vsa_native.block_sparse_attention
        )
        attend(q, k, v, out, indices, counts, valid_sizes, scale=scale)


class Backend(BaseBackend):
    operator_class = _Operator

    def prepare(self, pattern, **options):
        # Compilation belongs to preparation, before serving or capture.
        vsa_native.load()
        return super().prepare(pattern, **options)
