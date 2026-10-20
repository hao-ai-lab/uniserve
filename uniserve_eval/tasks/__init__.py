"""Registers task adapters for request construction and output validation."""

from __future__ import annotations

from ..types import TaskName
from .base import BenchmarkTask
from .video import VideoTask

# Keyed by `TaskName` value strings, the form profiles declare.
TASKS: dict[str, type[BenchmarkTask]] = {
    TaskName.VIDEO.value: VideoTask,
}


def get_task(name: str | TaskName) -> type[BenchmarkTask]:
    """Return the task adapter registered under a task identifier.

    Raises:
        KeyError: If no adapter is registered under the identifier.
    """
    key = name.value if isinstance(name, TaskName) else name
    if key not in TASKS:
        known = ", ".join(sorted(TASKS)) or "(none)"
        raise KeyError(f"unknown task {key!r}; known: {known}")
    return TASKS[key]


__all__ = [
    "TASKS",
    "BenchmarkTask",
    "VideoTask",
    "get_task",
]
