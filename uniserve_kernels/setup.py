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
from pathlib import Path

from setuptools import setup

_DEVICES = ("cuda", "cpu")


def _source(*parts):
    # setuptools requires sources relative to this directory.
    return str(Path("src", "uniserve_kernels", *parts))


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
    ]


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
