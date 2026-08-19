"""Task and dataset registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .types import BenchmarkPoint, Example, TaskName

TaskFactory = Callable[[BenchmarkPoint], Any]
DatasetLoader = Callable[[BenchmarkPoint, Any], list[Example]]


@dataclass(frozen=True)
class TaskSpec:
    name: TaskName
    allowed_endpoints: tuple[str, ...]
    default_endpoint: str
    default_stream: bool
    accepts_image: bool
    accepts_question: bool
    requires_image_count: bool
    forbids_image_count: bool
    factory: TaskFactory


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: DatasetLoader
    requires_tokenizer: bool
    requires_path: bool


_TASKS: dict[str, TaskSpec] = {}
_DATASETS: dict[str, DatasetSpec] = {}
_LOADED = False


def register_task(spec: TaskSpec) -> TaskSpec:
    key = spec.name.value
    if key in _TASKS:
        raise ValueError(f"task {key!r} is already registered")
    _TASKS[key] = spec
    return spec


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    if spec.name in _DATASETS:
        raise ValueError(f"dataset {spec.name!r} is already registered")
    _DATASETS[spec.name] = spec
    return spec


def get_task(name: str | TaskName) -> TaskSpec:
    _autoload()
    key = name.value if isinstance(name, TaskName) else name
    if key not in _TASKS:
        known = ", ".join(sorted(_TASKS)) or "(none)"
        raise KeyError(f"unknown task {key!r}; known: {known}")
    return _TASKS[key]


def get_dataset(name: str) -> DatasetSpec:
    _autoload()
    if name not in _DATASETS:
        known = ", ".join(sorted(_DATASETS)) or "(none)"
        raise KeyError(f"unknown dataset {name!r}; known: {known}")
    return _DATASETS[name]


def list_tasks() -> tuple[TaskSpec, ...]:
    _autoload()
    return tuple(_TASKS[name] for name in sorted(_TASKS))


def list_datasets() -> tuple[DatasetSpec, ...]:
    _autoload()
    return tuple(_DATASETS[name] for name in sorted(_DATASETS))


def _autoload() -> None:
    global _LOADED
    if _LOADED:
        return
    from . import datasets as _datasets
    from . import tasks as _tasks

    _ = _datasets
    _ = _tasks
    _LOADED = True
