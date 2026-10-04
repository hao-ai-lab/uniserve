"""DeepEP elastic dispatch/combine primitives from the FastAFD source tree."""

from functools import cache
from importlib.metadata import distribution
from pathlib import Path

from uniserve_kernels import jit


@cache
def load():
    """Build the native transport against the installed CUDA and NCCL ABI.

    This initializes the device compiler only. Collective construction and
    communication-buffer lifetime belong to the calling runtime.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("DeepEP requires a CUDA toolkit")
    package = distribution("nvidia-nccl-cu13")
    nccl = Path(package.locate_file("nvidia/nccl"))
    root = Path(__file__).parent / "csrc"
    sources = [
        root / "csrc" / "python_api.cpp",
        root / "csrc" / "kernels" / "backend" / "nccl.cu",
        root / "csrc" / "kernels" / "backend" / "cuda_driver.cu",
    ]
    headers = sorted(
        file
        for file in root.rglob("*")
        if file.suffix in {".h", ".hpp", ".cuh", ".inl"}
    )
    module = jit.load(
        "uniserve_deepep",
        sources,
        headers=headers,
        source_root=root,
        include_dirs=["csrc", "deep_ep/include", nccl / "include"],
        dependencies=[f"nvidia-nccl-cu13=={package.version}"],
        cxx_flags=[
            "-std=c++20",
            "-O3",
            "-Wno-deprecated-declarations",
            "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
        ],
        cuda_flags=[
            "-std=c++20",
            "-O3",
            "--extended-lambda",
            "--diag-suppress=128,2417",
            "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
            jit.device_architecture(),
        ],
        ldflags=[
            "-lcuda",
            "-lnvrtc",
            f"-L{nccl / 'lib'}",
            "-l:libnccl.so.2",
            f"-Wl,-rpath,{nccl / 'lib'}",
        ],
    )
    snapshot = Path(module.__file__).parent / "sources"
    module.init_jit(str(snapshot / "deep_ep"), CUDA_HOME, str(nccl))
    return module
