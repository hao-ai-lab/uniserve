"""Triton block-64 attention with live device selection and validity."""

from uniserve_kernels.attention import vsa_triton

from . import Backend as BaseBackend
from . import Operator as BaseOperator


def available(device):
    return vsa_triton.available(device)


class _Operator(BaseOperator):
    kernel = staticmethod(vsa_triton.block_sparse_attention)


class Backend(BaseBackend):
    name = "triton"
    operator_class = _Operator
