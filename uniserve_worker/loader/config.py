"""Immutable inputs for checkpoint source selection and model loading."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from ..bootstrap.execution_config import ExecutionConfig
from ..bootstrap.plan import ModelLoadScope
from ..nn.mesh import TensorParallelSpec

__all__ = ["LoadConfig", "LoadFormat", "LoadRequest"]


class LoadFormat(StrEnum):
    AUTO = "auto"
    SAFETENSORS = "safetensors"
    PT = "pt"
    DUMMY = "dummy"
    SHARDED_STATE = "sharded_state"
    LAYERED = "layered"


_DEFAULT_THREADS = object()


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True, slots=True, init=False)
class LoadConfig:
    load_format: LoadFormat
    download_dir: str | None
    ignore_patterns: tuple[str, ...]
    num_threads: int
    mmap: bool
    prefetch: bool
    drop_cache_after_load: bool
    checksum_manifest: str | None
    revision: str | None
    _num_threads_explicit: bool

    def __init__(
        self,
        load_format: LoadFormat | str = LoadFormat.AUTO,
        download_dir: str | None = None,
        ignore_patterns: tuple[str, ...] | list[str] = ("original/**/*",),
        num_threads: int | object = _DEFAULT_THREADS,
        mmap: bool = True,
        prefetch: bool | None = None,
        drop_cache_after_load: bool = False,
        checksum_manifest: str | None = None,
        revision: str | None = None,
    ) -> None:
        try:
            resolved_format = LoadFormat(str(load_format))
        except ValueError as error:
            choices = ", ".join(value.value for value in LoadFormat)
            raise ValueError(f"unknown load format {load_format!r}; expected one of {choices}") from error
        explicit_threads = num_threads is not _DEFAULT_THREADS
        if explicit_threads and (
            isinstance(num_threads, bool) or not isinstance(num_threads, int)
        ):
            raise TypeError("load num_threads must be an integer")
        resolved_threads = 8 if not explicit_threads else int(cast(int, num_threads))
        if resolved_threads < 1:
            raise ValueError("load num_threads must be positive")
        if isinstance(ignore_patterns, (str, bytes)):
            raise TypeError("load ignore_patterns must be a sequence of patterns")
        patterns = tuple(str(value) for value in ignore_patterns)
        if any(not value for value in patterns):
            raise ValueError("load ignore patterns must not be empty")
        use_mmap = bool(mmap)
        object.__setattr__(self, "load_format", resolved_format)
        object.__setattr__(self, "download_dir", _optional_text(download_dir))
        object.__setattr__(self, "ignore_patterns", patterns)
        object.__setattr__(self, "num_threads", resolved_threads)
        object.__setattr__(self, "mmap", use_mmap)
        object.__setattr__(self, "prefetch", use_mmap if prefetch is None else bool(prefetch))
        object.__setattr__(self, "drop_cache_after_load", bool(drop_cache_after_load))
        object.__setattr__(self, "checksum_manifest", _optional_text(checksum_manifest))
        object.__setattr__(self, "revision", _optional_text(revision))
        object.__setattr__(self, "_num_threads_explicit", explicit_threads)

    @property
    def reader_count(self) -> int:
        if self.prefetch and not self._num_threads_explicit:
            return 1
        return self.num_threads


@dataclass(frozen=True, slots=True)
class LoadRequest:
    model_path: str
    device: str
    execution: ExecutionConfig
    parallel: TensorParallelSpec
    scope: ModelLoadScope
    load: LoadConfig = LoadConfig()
    attention_backend: str | None = None

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must not be empty")
        if not self.device:
            raise ValueError("load device must not be empty")
