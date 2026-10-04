"""DeepGEMM's native grouped expert and split dispatch/compute kernels.

The kernel sources include FastAFD's attention/expert split of MegaMoE.
The runtime owns collective buffers, streams, weights and launch ordering;
this package supplies the native numerical and communication primitives.
"""

from functools import cache
from pathlib import Path

from uniserve_kernels import jit


@cache
def load():
    """Build the native host bindings and initialize their kernel compiler.

    Compilation uses the installed CUDA toolkit and the repository's pinned
    PyTorch ABI. Both host compilation and subsequent device compilation
    consume the same immutable source snapshot.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("DeepGEMM requires a CUDA toolkit")
    root = Path(__file__).parent / "csrc"
    source = root / "binding.cpp"
    headers = sorted(
        file
        for file in root.rglob("*")
        if file.suffix in {".h", ".hpp", ".cuh", ".inl"}
    )
    module = jit.load(
        "uniserve_deepgemm",
        [source],
        headers=headers,
        source_root=root,
        include_dirs=[
            "csrc",
            "deep_gemm/include",
            "third-party/cutlass/include",
            "third-party/fmt/include",
            Path(CUDA_HOME) / "include" / "cccl",
        ],
        cxx_flags=[
            "-std=c++17",
            "-O3",
            "-Wno-psabi",
            "-Wno-deprecated-declarations",
        ],
        ldflags=["-lnvrtc", "-lcuda"],
        with_cuda=True,
    )
    snapshot = Path(module.__file__).parent / "sources"
    module.init(str(snapshot / "deep_gemm"), CUDA_HOME)
    return module
