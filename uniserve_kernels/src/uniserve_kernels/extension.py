"""Build-once loading of the package's C++ and CUDA extensions.

``torch.utils.cpp_extension.load`` guards a shared build directory with a
lock file that only its creator removes, and every fresh process creates it,
even when the build is current. A process killed while holding it leaves
every later load of that extension waiting forever. :func:`load` never
makes one process depend on another's cleanup:

* A finished build is published as ``<root>/uniserve_kernels/<name>/<key>.so``
  by an atomic rename. ``<key>`` digests the sources, the build flags, and the
  torch, CUDA and Python ABI they compile against. A process that finds the
  published file imports it, taking no lock and running no build tools.
* Otherwise the process takes an exclusive ``flock`` on ``<key>.lock`` next to
  it. The kernel releases that lock when its holder exits by any means,
  including SIGKILL, so a waiter proceeds once a killed builder is gone.
* The lock holder runs torch's build in a fresh private directory, so torch's
  own lock file there is never visible to another process. A killed build
  leaves that directory behind; later builds use their own and ignore it.

``<root>`` is ``TORCH_EXTENSIONS_DIR`` when set and torch's default extension
root otherwise. Its file system must support ``flock``.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sysconfig
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import torch


def load(
    name: str,
    source_dir: Path,
    sources: Sequence[str],
    *,
    extra_cflags: Sequence[str] = (),
    extra_cuda_cflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
    with_cuda: bool | None = None,
) -> ModuleType:
    """Import extension ``name``, building it first when no process has.

    Blocks while another live process builds the same key and raises what
    the build raises; a failed build publishes nothing. An extension module
    initializes once per process, so callers keep the returned module rather
    than loading it again.

    Args:
        name: Extension module name; the sources bind it through
            ``TORCH_EXTENSION_NAME``.
        source_dir: Directory holding the sources and every header they
            include besides torch's and CUDA's. All files under it enter the
            build key, so any edit there selects a fresh build.
        sources: Translation units, relative to ``source_dir``.
        extra_cflags: Host compiler flags.
        extra_cuda_cflags: ``nvcc`` flags. CUDA sources must name their
            targets with ``-gencode`` here: the key does not cover the
            architectures torch would otherwise derive from visible devices.
        extra_ldflags: Linker flags.
        with_cuda: Passed to torch; ``None`` infers it from the sources.

    Returns:
        The imported extension module.
    """
    source_dir = Path(source_dir)
    flags = {
        "cflags": list(extra_cflags),
        "cuda_cflags": list(extra_cuda_cflags),
        "ldflags": list(extra_ldflags),
    }
    key = _build_key(name, source_dir, sources, flags, with_cuda)
    directory = _root() / "uniserve_kernels" / name
    published = directory / f"{key}.so"
    if published.exists():
        return _import(name, published)

    directory.mkdir(parents=True, exist_ok=True)
    # The lock file is never removed: unlinking it would let a later process
    # lock a new inode while an earlier holder still owns the old one.
    with open(directory / f"{key}.lock", "ab") as lock:
        # Held until this file closes or the process exits by any means.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if published.exists():
            return _import(name, published)
        return _build(name, source_dir, sources, flags, with_cuda, published)


def _build_key(
    name: str,
    source_dir: Path,
    sources: Sequence[str],
    flags: dict[str, list[str]],
    with_cuda: bool | None,
) -> str:
    """Digest every input that determines the built module's contents."""
    files = {
        path.relative_to(source_dir).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(source_dir.rglob("*"))
        if path.is_file()
    }
    inputs = {
        "name": name,
        "sources": list(sources),
        "files": files,
        "flags": flags,
        "with_cuda": with_cuda,
        # The objects compile against torch's headers and link its
        # libraries; the suffix names the Python ABI and platform.
        "torch": [
            torch.__version__,
            torch.version.git_version,
            torch.version.cuda,
        ],
        "python": sysconfig.get_config_var("EXT_SUFFIX"),
    }
    encoded = json.dumps(inputs, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _root() -> Path:
    from torch.utils.cpp_extension import get_default_build_root

    return Path(
        os.environ.get("TORCH_EXTENSIONS_DIR") or get_default_build_root()
    )


def _build(
    name: str,
    source_dir: Path,
    sources: Sequence[str],
    flags: dict[str, list[str]],
    with_cuda: bool | None,
    published: Path,
) -> ModuleType:
    """Build in a private directory and publish the module; lock held."""
    from torch.utils.cpp_extension import load as build

    build_directory = Path(
        tempfile.mkdtemp(prefix=f"{published.stem}.", dir=published.parent)
    )
    try:
        # Torch imports the module from the private directory. The rename
        # keeps that mapping valid, so the builder does not import it again.
        module = build(
            name,
            sources=[str(source_dir / source) for source in sources],
            extra_cflags=flags["cflags"],
            extra_cuda_cflags=flags["cuda_cflags"],
            extra_ldflags=flags["ldflags"],
            build_directory=str(build_directory),
            with_cuda=with_cuda,
        )
        os.replace(build_directory / f"{name}.so", published)
    finally:
        shutil.rmtree(build_directory, ignore_errors=True)
    return module


def _import(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} is not an extension module", path=str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
