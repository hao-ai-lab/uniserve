"""Build or load PyTorch JIT extensions without stale-lock hangs.

``torch.utils.cpp_extension.load`` serializes builds of one extension name
with a lock file in its build directory and waits, without a deadline, for
that file to disappear. A builder killed mid-compilation leaves the file
behind and every later process waits forever. :func:`load` first takes an
``fcntl`` lock on a file beside the build directory; the kernel releases it
when its holder exits for any reason. Every UniServe build of the name holds
that lock, so a PyTorch lock file found while holding it belongs to a dead
builder and is removed before PyTorch builds or loads the extension.
"""

from __future__ import annotations

import fcntl
from collections.abc import Sequence
from pathlib import Path


def load(
    name: str,
    sources: Sequence[Path],
    *,
    cuda_flags: Sequence[str] = (),
    cxx_flags: Sequence[str] = ("-O3", "-std=c++20"),
):
    """Compile ``sources`` into extension ``name`` or load the cached build.

    The device code targets the current CUDA device's architecture only
    (rather than every entry of ``TORCH_CUDA_ARCH_LIST``). PyTorch's build
    cache (``TORCH_EXTENSIONS_DIR`` or its default) keys the build by name
    and rebuilds when the sources or flags change. Returns the loaded
    module. Compilation errors propagate.
    """
    import torch
    from torch.utils.cpp_extension import _get_build_directory
    from torch.utils.cpp_extension import load as load_extension

    major, minor = torch.cuda.get_device_capability()
    architecture = (
        f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    )
    build = Path(_get_build_directory(name, verbose=False))
    with open(build.parent / f"{name}.flock", "a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            # Only a builder holding this lock creates PyTorch's lock file,
            # so one present now outlived a process that died mid-build.
            (build / "lock").unlink(missing_ok=True)
            return load_extension(
                name,
                sources=[str(source) for source in sources],
                extra_cflags=list(cxx_flags),
                extra_cuda_cflags=[*cuda_flags, architecture],
                build_directory=str(build),
            )
        finally:
            fcntl.flock(guard, fcntl.LOCK_UN)
