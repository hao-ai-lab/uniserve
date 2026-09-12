"""Immutable inputs for checkpoint source selection and model loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from ..config import WorkerConfig
from ..execution.model_entry import ModelEntry

__all__ = ["LoadConfig", "LoadFormat", "LoadRequest"]


class LoadFormat(StrEnum):
    """Checkpoint source and materialization strategies accepted by the loader."""

    AUTO = "auto"
    SAFETENSORS = "safetensors"
    PT = "pt"
    DUMMY = "dummy"
    LAYERED = "layered"


_DEFAULT_THREADS = object()


def _optional_text(value: object | None) -> str | None:
    """Normalize an optional string-like setting and discard blank values."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True, slots=True, init=False)
class LoadConfig:
    """Immutable checkpoint discovery, I/O, caching, and integrity settings."""

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
        """Normalize checkpoint format, reader concurrency, cache, and revision options."""

        # Resolve and validate caller-facing values before publishing the frozen state.
        try:
            resolved_format = LoadFormat(str(load_format))
        except ValueError as error:
            choices = ", ".join(value.value for value in LoadFormat)
            raise ValueError(
                f"unknown load format {load_format!r}; expected one of {choices}"
            ) from error
        explicit_threads = num_threads is not _DEFAULT_THREADS
        if explicit_threads and (isinstance(num_threads, bool) or not isinstance(num_threads, int)):
            raise TypeError("load num_threads must be an integer")
        resolved_threads = 8 if not explicit_threads else int(cast(int, num_threads))
        if resolved_threads < 1:
            raise ValueError("load num_threads must be positive")
        if isinstance(ignore_patterns, (str, bytes)):
            raise TypeError("load ignore_patterns must be a sequence of patterns")
        patterns = tuple(str(value) for value in ignore_patterns)
        if any(not value for value in patterns):
            raise ValueError("load ignore patterns must not be empty")

        # Prefetch follows mmap by default because page advice only benefits mapped reads.
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
        """Limit implicit mmap-prefetch loading to one reader to avoid competing page walks."""

        if self.prefetch and not self._num_threads_explicit:
            return 1
        return self.num_threads


@dataclass(frozen=True, slots=True)
class LoadRequest:
    """Computation bindings, execution configuration, and checkpoint policy for one load."""

    model_path: str
    execution: WorkerConfig
    bindings: Mapping[str, ModelEntry]
    load: LoadConfig = LoadConfig()
    quantization_config: Mapping[str, object] = field(default_factory=dict)
    max_text_rows: int = 8192
    max_video_seconds: float = 15.0
    pipeline_depth: int | None = None

    def __post_init__(self) -> None:
        """Reject requests that cannot identify a checkpoint or destination device."""

        if not self.model_path:
            raise ValueError("model_path must not be empty")
        if self.pipeline_depth is not None and self.pipeline_depth < 1:
            raise ValueError("load pipeline depth must be positive")
