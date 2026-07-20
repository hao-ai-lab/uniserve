"""System-owned model execution interface."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runner import ExecutorConfig, ModelExecutor, ModelRunner

_EXECUTION_EXPORTS = frozenset({"ModelExecutor", "ModelRunner", "ExecutorConfig"})


def __getattr__(name: str) -> Any:
    if name not in _EXECUTION_EXPORTS:
        raise AttributeError(name)
    from . import runner

    value = getattr(runner, name)
    globals()[name] = value
    return value


__all__ = [
    "ModelExecutor",
    "ModelRunner",
    "ExecutorConfig",
]
