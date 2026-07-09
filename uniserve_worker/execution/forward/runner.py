"""Forward runner contracts and eager runner implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ...contracts.forward_batch import ForwardBatch
from .plan import ForwardPlan
from .result import ForwardResult, coerce_forward_result

__all__ = ["EagerForwardRunner", "ForwardRunner"]


class ForwardRunner(ABC):
    @abstractmethod
    def run(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        *,
        forward_fn: Callable[[ForwardBatch], Any] | None = None,
    ) -> ForwardResult: ...


class EagerForwardRunner(ForwardRunner):
    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def run(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        *,
        forward_fn: Callable[[ForwardBatch], Any] | None = None,
    ) -> ForwardResult:
        del plan
        if forward_fn is not None:
            return coerce_forward_result(forward_fn(batch))
        if self.model is None or not callable(getattr(self.model, "forward", None)):
            raise TypeError("eager forward requires a model.forward(batch) callable")
        return coerce_forward_result(self.model.forward(batch))
