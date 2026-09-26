"""CuTe block-64 attention on Blackwell."""

from uniserve_kernels.attention import vsa_cute

from . import Backend as BaseBackend
from . import Operator as BaseOperator


def available(device):
    return vsa_cute.available(device)


class _Operator(BaseOperator):
    kernel = staticmethod(vsa_cute.block_sparse_attention)


class Backend(BaseBackend):
    name = "cute"
    operator_class = _Operator
