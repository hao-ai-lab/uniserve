"""Deterministic checkpoint iteration, lazy tensor handles, and page-cache advice."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import safe_open

from .config import LoadConfig, LoadFormat
from .handles import (
    PrefixedWeightHandle,
    PtFileWeightHandle,
    SafetensorFileWeightHandle,
    SafetensorWeightHandle,
    TensorWeightHandle,
    WeightHandle,
    safetensor_dtype,
    weight_handle_materialization,
)
from .source import WeightSourceSet

__all__ = ["iter_weight_handles"]


def iter_weight_handles(source: WeightSourceSet, load: LoadConfig) -> Iterator[WeightHandle]:
    """Yield unique handles from a closed source set in deterministic file/name order.

    Reader threads may decode files concurrently, but results are consumed in source
    order so parameter assignment and duplicate diagnostics remain reproducible.
    """

    # Establish the file schedule and issue optional kernel page-cache advice once.
    files = _rank_stagger(source.weight_files, load)
    seen: set[str] = set()
    if not files:
        return
    if load.prefetch:
        _prefetch_files(files)

    # Layered loading must retain durable file-backed handles because assignment is
    # deferred until placements have been grouped by owning module.
    if load.load_format is LoadFormat.LAYERED:
        for path in files:
            yield from _unique_handles(
                _read_durable_file(path, source.name_prefix, load),
                seen,
            )
            _drop_cache(path, load)
        return

    # A single reader can stream open safetensors slices directly into the consumer.
    if load.reader_count == 1:
        for path in files:
            yield from _unique_handles(_read_file(path, source.name_prefix, load), seen)
            _drop_cache(path, load)
        return

    # Decode files concurrently, then publish each completed list in canonical order.
    with ThreadPoolExecutor(max_workers=load.reader_count) as pool:
        futures: dict[Path, Future[list[WeightHandle]]] = {
            path: pool.submit(_read_file_list, path, source.name_prefix, load)
            for path in files
        }
        for path in files:
            handles = futures[path].result()
            with weight_handle_materialization():
                yield from _unique_handles(handles, seen)
            _drop_cache(path, load)


def _read_file(path: Path, prefix: str, load: LoadConfig) -> Iterator[WeightHandle]:
    """Yield materializable handles while keeping any required file scope open."""

    # Safetensors mmap exposes slices without reading unrelated tensor payloads.
    if path.suffix == ".safetensors":
        if load.mmap:
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for name in sorted(checkpoint.keys()):
                    handle: WeightHandle = SafetensorWeightHandle.from_slice(
                        name, checkpoint.get_slice(name)
                    )
                    yield PrefixedWeightHandle(prefix, handle) if prefix else handle
            return
        state: Any = load_safetensors(path.read_bytes())

    # PT containers and non-mmap safetensors become resident tensor mappings.
    elif path.suffix in {".bin", ".pt"}:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
    else:
        raise ValueError(f"unsupported checkpoint weight file {path}")
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint weight file {path} does not contain a tensor mapping")

    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            continue
        handle = TensorWeightHandle(str(name), tensor)
        yield PrefixedWeightHandle(prefix, handle) if prefix else handle


def _unique_handles(
    handles: Iterable[WeightHandle],
    seen: set[str],
) -> Iterator[WeightHandle]:
    """Reject duplicate checkpoint keys while forwarding handles unchanged."""

    for handle in handles:
        if handle.name in seen:
            raise ValueError(f"checkpoint source repeats tensor {handle.name!r}")
        seen.add(handle.name)
        yield handle


def _read_file_list(path: Path, prefix: str, load: LoadConfig) -> list[WeightHandle]:
    """Collect one file into handles safe to consume after the reader task returns."""

    if path.suffix == ".safetensors" and load.mmap:
        return list(_read_durable_file(path, prefix, load))
    return list(_read_file(path, prefix, load))


def _read_durable_file(path: Path, prefix: str, load: LoadConfig) -> Iterator[WeightHandle]:
    """Yield file-backed handles whose payloads can be materialized after discovery."""

    # Safetensors metadata is cheap to enumerate and sufficient to build lazy handles.
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            handles: list[WeightHandle] = [
                SafetensorFileWeightHandle(
                    name=str(name),
                    path=path,
                    shape=tuple(
                        int(size) for size in checkpoint.get_slice(name).get_shape()
                    ),
                    dtype=safetensor_dtype(str(checkpoint.get_slice(name).get_dtype())),
                    mmap=load.mmap,
                )
                for name in sorted(checkpoint.keys())
            ]

    # PT files must be decoded to discover shape and dtype, while the returned handles
    # retain only metadata and reopen through the scoped materialization cache.
    elif path.suffix in {".bin", ".pt"}:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"checkpoint weight file {path} does not contain a tensor mapping")
        handles = [
            PtFileWeightHandle(
                name=str(name),
                path=path,
                shape=tuple(int(size) for size in tensor.shape),
                dtype=tensor.dtype,
            )
            for name, tensor in sorted(state.items())
            if isinstance(tensor, torch.Tensor)
        ]
    else:
        raise ValueError(f"unsupported checkpoint weight file {path}")

    for handle in handles:
        yield PrefixedWeightHandle(prefix, handle) if prefix else handle


def _prefetch_files(files: tuple[Path, ...]) -> None:
    """Advise the operating system that every checkpoint file will be read soon."""

    advice = getattr(os, "POSIX_FADV_WILLNEED", None)
    advise = getattr(os, "posix_fadvise", None)
    if advice is None or advise is None:
        return
    with ExitStack() as stack:
        descriptors = [stack.enter_context(path.open("rb")) for path in files]
        for descriptor in descriptors:
            advise(descriptor.fileno(), 0, 0, advice)


def _drop_cache(path: Path, load: LoadConfig) -> None:
    """Release a consumed checkpoint file from the page cache when configured."""

    if not load.drop_cache_after_load:
        return
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    advise = getattr(os, "posix_fadvise", None)
    if advice is None or advise is None:
        return
    with path.open("rb") as descriptor:
        advise(descriptor.fileno(), 0, 0, advice)


def _rank_stagger(files: tuple[Path, ...], load: LoadConfig) -> tuple[Path, ...]:
    """Return the deterministic checkpoint file order for this rank."""

    del load
    return files
