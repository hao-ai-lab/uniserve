"""DeepEP elastic dispatch/combine primitives from the FastAFD source tree.

The host transport compiles with the package into
:mod:`uniserve_kernels.deepep._C`, linked against this environment's NCCL.
Its device kernels compile at run time with NVRTC from the headers in
``csrc/deep_ep``.
"""

from functools import cache
from importlib.metadata import distribution
from pathlib import Path


@cache
def load():
    """Return the native transport with its device compiler initialized.

    This initializes the device compiler only. Collective construction and
    communication-buffer lifetime belong to the calling runtime.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("DeepEP requires a CUDA toolkit")
    # Imported on first use: a CPU build of the package has no native module.
    from uniserve_kernels.deepep import _C

    nccl = Path(distribution("nvidia-nccl-cu13").locate_file("nvidia/nccl"))
    headers = Path(__file__).parent / "csrc" / "deep_ep"
    _C.init_jit(str(headers), CUDA_HOME, str(nccl))
    return _C
