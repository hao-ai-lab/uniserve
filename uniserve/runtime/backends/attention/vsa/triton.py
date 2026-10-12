"""Triton block-64 attention with live device selection and validity."""

from uniserve_kernels.attention import vsa_triton

from . import Backend as BaseBackend
from . import Operator as BaseOperator

# Tile sizes this provider's kernels address.
# FIXME: no 128-row build exists here, so 128-row tiles (the ref2va FastH3
# students) run only on SM100 (``sm100.TILES``).
TILES = (64,)


def available(device):
    return vsa_triton.available(device)


class _Operator(BaseOperator):
    kernel = staticmethod(vsa_triton.block_sparse_attention)


class Backend(BaseBackend):
    name = "triton"
    operator_class = _Operator
