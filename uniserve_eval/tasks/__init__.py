from __future__ import annotations

from ..types import TaskName
from .base import BenchmarkTask, ImageCountRule
from .i2i import I2ITask
from .i2t import I2TTask
from .interleave import InterleaveTask
from .t2i import T2ITask
from .text import TextTask

TASKS: dict[str, type[BenchmarkTask]] = {
    TaskName.TEXT.value: TextTask,
    TaskName.T2I.value: T2ITask,
    TaskName.I2I.value: I2ITask,
    TaskName.I2T.value: I2TTask,
    TaskName.INTERLEAVE.value: InterleaveTask,
}


def get_task(name: str | TaskName) -> type[BenchmarkTask]:
    key = name.value if isinstance(name, TaskName) else name
    if key not in TASKS:
        known = ", ".join(sorted(TASKS)) or "(none)"
        raise KeyError(f"unknown task {key!r}; known: {known}")
    return TASKS[key]


def list_tasks() -> tuple[type[BenchmarkTask], ...]:
    return tuple(TASKS[name] for name in sorted(TASKS))


__all__ = [
    "TASKS",
    "BenchmarkTask",
    "I2ITask",
    "I2TTask",
    "ImageCountRule",
    "InterleaveTask",
    "T2ITask",
    "TextTask",
    "get_task",
    "list_tasks",
]
