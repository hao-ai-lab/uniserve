"""Build or load PyTorch JIT extensions by content, without stale-lock hangs.

Every native extension of this package builds through :func:`load`. A build
is named ``<name>_<digest>``, the digest covering the bytes of every source
and header and all build flags, and compiles copies of those files staged in
``<build directory>/sources/``. Processes of any revision or worktree
therefore agree on the build that matches their files: identical files share
one build (the staged paths are identical too, so PyTorch's ninja file and
its up-to-date check match and nothing recompiles), and different files build
under different names instead of rebuilding over one another.

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
import hashlib
import os
from collections.abc import Sequence
from pathlib import Path


def device_architecture() -> str:
    """Return the nvcc flag that targets the current CUDA device only.

    Extensions that pass it compile one architecture instead of every entry
    of ``TORCH_CUDA_ARCH_LIST``.
    """
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"


def load(
    name: str,
    sources: Sequence[Path],
    *,
    headers: Sequence[Path] = (),
    cxx_flags: Sequence[str] = ("-O3", "-std=c++20"),
    cuda_flags: Sequence[str] = (),
    ldflags: Sequence[str] = (),
    with_cuda: bool | None = None,
):
    """Compile ``sources`` into an extension named after ``name`` or load it.

    ``headers`` are the local files the sources include by relative path;
    they enter the digest and are staged beside the sources, so every file
    name among sources and headers must be distinct. ``cuda_flags`` must
    name the device architecture (a ``-gencode`` flag, for example
    :func:`device_architecture`) when a source is CUDA; PyTorch otherwise
    compiles every ``TORCH_CUDA_ARCH_LIST`` entry. ``with_cuda`` is
    PyTorch's flag for C++ sources that use the CUDA headers and runtime.
    The build lives in PyTorch's cache (``TORCH_EXTENSIONS_DIR`` or its
    default). Returns the loaded module; compilation errors propagate.
    """
    from torch.utils.cpp_extension import _get_build_directory
    from torch.utils.cpp_extension import load as load_extension

    files = [Path(file) for file in (*sources, *headers)]
    if len({file.name for file in files}) != len(files):
        raise ValueError(f"extension {name}: file names must be distinct")
    contents = [file.read_bytes() for file in files]

    # Length-prefixed entries keep the digest unambiguous across files.
    digest = hashlib.sha256()
    for file, content in zip(files, contents, strict=True):
        digest.update(file.name.encode())
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    flags = (tuple(cxx_flags), tuple(cuda_flags), tuple(ldflags), with_cuda)
    digest.update(repr(flags).encode())
    name = f"{name}_{digest.hexdigest()[:16]}"

    build = Path(_get_build_directory(name, verbose=False))
    with open(build.parent / f"{name}.flock", "a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            # Only a builder holding this lock creates PyTorch's lock file,
            # so one present now outlived a process that died mid-build.
            (build / "lock").unlink(missing_ok=True)

            # Stage the files under the lock. A copy that differs from the
            # digest's content (a writer died mid-copy) is replaced
            # atomically; an identical copy is left untouched so ninja sees
            # nothing new.
            staged = build / "sources"
            staged.mkdir(exist_ok=True)
            for file, content in zip(files, contents, strict=True):
                target = staged / file.name
                if target.exists() and target.read_bytes() == content:
                    continue
                partial = staged / f"{file.name}.partial"
                partial.write_bytes(content)
                os.replace(partial, target)

            return load_extension(
                name,
                sources=[str(staged / Path(source).name) for source in sources],
                extra_cflags=list(cxx_flags),
                extra_cuda_cflags=list(cuda_flags),
                extra_ldflags=list(ldflags),
                build_directory=str(build),
                with_cuda=with_cuda,
            )
        finally:
            fcntl.flock(guard, fcntl.LOCK_UN)
