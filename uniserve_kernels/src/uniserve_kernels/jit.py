"""Build or load PyTorch JIT extensions by content, without stale-lock hangs.

Every native extension builds through :func:`load`. Its module name includes
a digest of its sources, headers, build flags, and Python/PyTorch ABI. A
finished shared object is published atomically, so processes with identical
inputs import it without invoking build tools.

An operating-system lock serializes builders and is released even after
SIGKILL. Each builder compiles in its own private directory: an interrupted
builder's PyTorch lock and surviving compiler cannot block or overwrite the
next builder's files. Only a successful build publishes an importable module.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
import shutil
import sysconfig
import tempfile
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
    source_root: Path | None = None,
    include_dirs: Sequence[Path | str] = (),
    dependencies: Sequence[str] = (),
    cxx_flags: Sequence[str] = ("-O3", "-std=c++20"),
    cuda_flags: Sequence[str] = (),
    ldflags: Sequence[str] = (),
    with_cuda: bool | None = None,
):
    """Compile ``sources`` into an extension named after ``name`` or load it.

    ``headers`` enter the digest alongside the sources. With ``source_root``
    their paths relative to that directory are preserved; otherwise each
    file is staged by its basename, which must be unique. Relative
    ``include_dirs`` refer to the staged tree; absolute ones name installed
    external dependencies. ``dependencies`` records their versions in the
    build identity (for example, the NCCL ABI). The immutable source snapshot
    remains beside the returned module under ``sources/`` for extensions
    that compile kernels at runtime. ``cuda_flags`` must name the device
    architecture (a ``-gencode`` flag, for example
    :func:`device_architecture`) when a source is CUDA; PyTorch otherwise
    compiles every ``TORCH_CUDA_ARCH_LIST`` entry. ``with_cuda`` is
    PyTorch's flag for C++ sources that use the CUDA headers and runtime.
    The build lives in PyTorch's cache (``TORCH_EXTENSIONS_DIR`` or its
    default). Returns the loaded module; compilation errors propagate.
    """
    import torch
    from torch.utils.cpp_extension import _get_build_directory
    from torch.utils.cpp_extension import load as load_extension

    files = [Path(file) for file in (*sources, *headers)]
    root = None if source_root is None else Path(source_root).resolve()
    paths = [
        Path(file.name) if root is None else file.resolve().relative_to(root)
        for file in files
    ]
    if len(set(paths)) != len(paths):
        raise ValueError(f"extension {name}: staged paths must be distinct")
    contents = [file.read_bytes() for file in files]

    # Length-prefixed entries keep the digest unambiguous across files.
    digest = hashlib.sha256()
    for path, content in zip(paths, contents, strict=True):
        digest.update(str(path).encode())
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    flags = (
        tuple(map(str, paths[: len(sources)])),
        tuple(map(str, include_dirs)),
        tuple(dependencies),
        tuple(cxx_flags),
        tuple(cuda_flags),
        tuple(ldflags),
        with_cuda,
        torch.__version__,
        torch.version.git_version,
        torch.version.cuda,
        torch._C._GLIBCXX_USE_CXX11_ABI,
        sysconfig.get_config_var("EXT_SUFFIX"),
    )
    digest.update(repr(flags).encode())
    name = f"{name}_{digest.hexdigest()[:16]}"

    build = Path(_get_build_directory(name, verbose=False))
    published = build / f"{name}.so"
    if published.exists():
        return _import(name, published)
    with open(build.parent / f"{name}.flock", "a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if published.exists():
            return _import(name, published)
        private = Path(tempfile.mkdtemp(prefix="build-", dir=build))
        try:
            # Compile exactly the bytes that entered the digest, even if a
            # source checkout is edited while the compiler is running.
            staged = private / "sources"
            staged.mkdir()
            for path, content in zip(paths, contents, strict=True):
                target = staged / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

            load_extension(
                name,
                sources=[str(staged / path) for path in paths[: len(sources)]],
                extra_include_paths=[
                    str(staged / path) for path in include_dirs
                ],
                extra_cflags=list(cxx_flags),
                extra_cuda_cflags=list(cuda_flags),
                extra_ldflags=list(ldflags),
                build_directory=str(private),
                with_cuda=with_cuda,
            )
            # Publish the source snapshot first: a process that sees the
            # library must also see the headers its runtime compiler needs.
            # A killed publisher may leave only the source directory behind.
            shutil.rmtree(build / "sources", ignore_errors=True)
            os.replace(staged, build / "sources")
            os.replace(private / f"{name}.so", published)
            return _import(name, published)
        finally:
            shutil.rmtree(private, ignore_errors=True)


def _import(name: str, path: Path):
    """Import an atomically published extension without taking a build lock."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} is not an extension module", path=str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # CPython may return the extension object first loaded by the builder.
    # Its origin must describe the published library after the atomic move.
    module.__file__ = str(path)
    module.__spec__ = spec
    return module
