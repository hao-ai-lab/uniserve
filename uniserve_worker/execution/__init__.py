"""System-owned model execution interface."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runner import ModelRunner, RunnerConfig

_RUNNER_EXPORTS = frozenset({"ModelRunner", "RunnerConfig"})


def __getattr__(name: str) -> Any:
    if name not in _RUNNER_EXPORTS:
        raise AttributeError(name)
    from . import runner

    value = getattr(runner, name)
    globals()[name] = value
    return value

__all__ = [
    "ModelRunner",
    "RunnerConfig",
]
