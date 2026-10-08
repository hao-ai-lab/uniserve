"""Compile UniServe's native kernels when the package builds.

Every C++ and CUDA extension of ``uniserve_kernels`` is declared here and
compiled by PyTorch's ``BuildExtension`` while the package builds, so a
process imports finished modules and runs no compiler. Each module installs
inside the Python package that wraps it.

The extensions link against the libraries they run with, so the package
builds without isolation in the serving environment: ``uv`` does so after
installing the locked PyTorch (``no-build-isolation-package`` in the
repository's ``pyproject.toml``), and other front ends build with
``--no-build-isolation``.

``UNISERVE_KERNELS_DEVICE`` names the build target. ``cuda``, the default,
compiles every extension and requires a CUDA toolkit; ``cpu`` builds only
the Python package, for hosts that run the portable implementations. A
kernel whose instructions are specific to one architecture names its
``-gencode`` target; the others compile for ``TORCH_CUDA_ARCH_LIST``, or for
the visible GPUs when it is unset.
"""

import os
from importlib.metadata import distribution
from pathlib import Path

from setuptools import setup

_DEVICES = ("cuda", "cpu")


def _source(*parts):
    # setuptools requires sources relative to this directory.
    return str(Path("src", "uniserve_kernels", *parts))


def _include(*parts):
    # The compiler runs in the build directory, so include paths are absolute.
    return str(Path(__file__).resolve().parent.joinpath(_source(*parts)))


def _cuda_extensions():
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

    if CUDA_HOME is None:
        raise RuntimeError(
            "uniserve-kernels compiles CUDA extensions; install a CUDA "
            "toolkit or build with UNISERVE_KERNELS_DEVICE=cpu"
        )
    # The driver API resolves at run time to the installed driver; the
    # toolkit's stub library links it on build hosts without one.
    driver = {
        "libraries": ["cuda"],
        "library_dirs": [str(Path(CUDA_HOME, "lib64", "stubs"))],
    }
    # SM100a block-sparse attention, one module per sparse-block size; the
    # 128-row configuration defines VSA_BLK128. Architecture-conditional
    # instructions carry no cross-architecture compatibility, so the target
    # is exactly sm_100a.
    sparse_attention = [
        CUDAExtension(
            f"uniserve_kernels.attention.vsa_native._block{block}",
            [_source("attention", "vsa_native", "csrc", "attention.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", *configuration],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "--use_fast_math",
                    "--expt-extended-lambda",
                    "--expt-relaxed-constexpr",
                    "-Xcompiler=-fno-strict-aliasing",
                    "-gencode=arch=compute_100a,code=sm_100a",
                    *configuration,
                ],
            },
            **driver,
        )
        for block, configuration in ((64, []), (128, ["-DVSA_BLK128=true"]))
    ]
    return [
        # CUDA virtual memory management, peer handle export and import,
        # and strided host/device DMA.
        CUDAExtension(
            "uniserve_kernels.peer_storage._C",
            [_source("peer_storage", "csrc", "peer_storage.cpp")],
            extra_compile_args={"cxx": ["-O2", "-std=c++20"]},
            **driver,
        ),
        *sparse_attention,
        # Softmax top-k expert routing, for every target architecture.
        CUDAExtension(
            "uniserve_kernels._topk_softmax",
            [_source("csrc", "topk_softmax.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3", "-std=c++20"],
            },
        ),
        # Per-frame group normalization, SiLU and causal padding. The sources
        # compile without fast math, as PyTorch's normalization and SiLU
        # kernels do, whose arithmetic they reproduce bit for bit.
        CUDAExtension(
            "uniserve_kernels.norm._frame",
            [_source("norm", "csrc", "frame.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3", "-std=c++20", "--expt-relaxed-constexpr"],
            },
        ),
        # SM100a block-diffusion canvas steps and their cuBLASLt
        # self-conditioning product. The sources compile without fast math:
        # the Gumbel scores use the accurate logf, and the divisions and
        # temperature use explicit round-to-nearest intrinsics.
        CUDAExtension(
            "uniserve_kernels.diffusion._canvas",
            [
                _source("diffusion", "csrc", "canvas.cu"),
                _source("diffusion", "csrc", "product.cpp"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "--expt-relaxed-constexpr",
                    "-gencode=arch=compute_100a,code=sm_100a",
                ],
            },
            libraries=["cublasLt"],
        ),
        *_expert_extensions(CUDA_HOME, driver),
    ]


def _expert_extensions(cuda_home, driver):
    """Host bindings of the vendored DeepEP and DeepGEMM sources.

    Both libraries compile their device kernels at run time with NVRTC from
    the headers installed beside these modules, so the bindings link NVRTC
    and the build compiles host code only.
    """
    from torch.utils.cpp_extension import CUDAExtension

    # DeepEP's elastic transport links the NCCL of this environment, the
    # runtime PyTorch also loads; the run path names that installation.
    nccl = Path(distribution("nvidia-nccl-cu13").locate_file("nvidia/nccl"))
    deepep = CUDAExtension(
        "uniserve_kernels.deepep._C",
        [
            _source("deepep", "csrc", "csrc", "python_api.cpp"),
            _source("deepep", "csrc", "csrc", "kernels", "backend", "nccl.cu"),
            _source(
                "deepep", "csrc", "csrc", "kernels", "backend", "cuda_driver.cu"
            ),
        ],
        include_dirs=[
            _include("deepep", "csrc", "csrc"),
            _include("deepep", "csrc", "deep_ep", "include"),
            str(nccl / "include"),
        ],
        extra_compile_args={
            "cxx": [
                "-std=c++20",
                "-O3",
                "-Wno-deprecated-declarations",
                "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
            ],
            "nvcc": [
                "-std=c++20",
                "-O3",
                "--extended-lambda",
                "--diag-suppress=128,2417",
                "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
            ],
        },
        libraries=[*driver["libraries"], "nvrtc", ":libnccl.so.2"],
        library_dirs=[*driver["library_dirs"], str(nccl / "lib")],
        extra_link_args=[f"-Wl,-rpath,{nccl / 'lib'}"],
    )
    # DeepGEMM's host binding, including FastAFD's split MegaMoE. CUTLASS and
    # fmt are vendored headers, and CCCL ships with the CUDA toolkit.
    deepgemm = CUDAExtension(
        "uniserve_kernels.deepgemm._C",
        [_source("deepgemm", "csrc", "binding.cpp")],
        include_dirs=[
            _include("deepgemm", "csrc", "csrc"),
            _include("deepgemm", "csrc", "deep_gemm", "include"),
            _include("deepgemm", "csrc", "third-party", "cutlass", "include"),
            _include("deepgemm", "csrc", "third-party", "fmt", "include"),
            str(Path(cuda_home, "include", "cccl")),
        ],
        extra_compile_args={
            "cxx": [
                "-std=c++17",
                "-O3",
                "-Wno-psabi",
                "-Wno-deprecated-declarations",
            ]
        },
        libraries=[*driver["libraries"], "nvrtc"],
        library_dirs=driver["library_dirs"],
    )
    return [deepep, deepgemm]


def _build():
    device = os.environ.get("UNISERVE_KERNELS_DEVICE", "cuda")
    if device not in _DEVICES:
        raise ValueError(
            f"UNISERVE_KERNELS_DEVICE={device!r} is not one of {_DEVICES}"
        )
    if device == "cpu":
        return {}

    from torch.utils.cpp_extension import BuildExtension

    return {
        "ext_modules": _cuda_extensions(),
        "cmdclass": {"build_ext": BuildExtension},
    }


setup(**_build())
