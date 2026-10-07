"""DeepGEMM's native grouped expert and split dispatch/compute kernels.

The kernel sources include FastAFD's attention/expert split of MegaMoE.
The runtime owns collective buffers, streams, weights and launch ordering;
this package supplies the native numerical and communication primitives.
The host binding compiles with the package into
:mod:`uniserve_kernels.deepgemm._C`; device kernels compile at run time with
NVRTC from the headers in ``csrc/deep_gemm`` and the vendored CUTLASS.
"""

from functools import cache
from pathlib import Path


@cache
def load():
    """Return the native host bindings with their kernel compiler initialized.

    Device compilation uses the installed CUDA toolkit.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("DeepGEMM requires a CUDA toolkit")
    # Imported on first use: a CPU build of the package has no native module.
    from uniserve_kernels.deepgemm import _C

    _C.init(str(Path(__file__).parent / "csrc" / "deep_gemm"), CUDA_HOME)
    return _C
