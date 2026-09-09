"""Execution-scoped sum reduction providers for model communicators."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

import torch


class SumReduction(Protocol):
    """Reduce an eligible tensor in place, or leave an unsupported tensor untouched."""

    def try_reduce(self, value: torch.Tensor) -> bool: ...


_ACTIVE_REDUCTIONS: ContextVar[Mapping[Any, SumReduction]] = ContextVar(
    "active_sum_reductions", default={}
)


@contextmanager
def collective_scope(reductions: Mapping[Any, SumReduction]) -> Iterator[None]:
    """Bind execution-owned workspaces while enqueueing or capturing one model call.

    Captured kernels retain their workspace addresses. Replay requires no host
    dispatch; the runner keeps each binding alive until its graphs retire.
    """

    token = _ACTIVE_REDUCTIONS.set(reductions)
    try:
        yield
    finally:
        _ACTIVE_REDUCTIONS.reset(token)


def try_sum_reduction(group: Any, value: torch.Tensor) -> bool:
    """Dispatch an in-place sum through the current execution scope."""

    reduction = _ACTIVE_REDUCTIONS.get().get(group)
    return reduction is not None and reduction.try_reduce(value)
