"""Selection among physical graph execution paths."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from uniserve_worker.contracts.forward_batch import (
    ForwardBatch,
    ForwardPlan,
    ForwardResult,
    GraphInfo,
)
from uniserve_worker.contracts.forward_context import get_forward_context, use_forward_context
from uniserve_worker.foundation.errors import invalid_descriptor

from .bucket import Capacity, key

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Match:
    """Whether a path owns the supplied plan."""

    accepted: bool
    reason: str = ""


class Path(ABC):
    """A real physical graph execution path.

    Implementations perform one graph-backed run. Capture and replay remain an
    internal concern of their physical runners; this interface does not pretend
    to own CUDA graph state.
    """

    @abstractmethod
    def name(self, plan: ForwardPlan) -> str: ...

    @abstractmethod
    def match(self, batch: ForwardBatch, plan: ForwardPlan) -> Match: ...

    def capacity(self, batch: ForwardBatch, plan: ForwardPlan) -> Capacity:
        return key(path=self.name(plan), batch=batch, plan=plan)

    @abstractmethod
    def run(self, batch: ForwardBatch, plan: ForwardPlan) -> ForwardResult | None: ...


class Dispatch:
    """Choose one path and publish capture policy to its physical runner."""

    def __init__(self, paths: tuple[Path, ...] = ()) -> None:
        self.paths = list(paths)

    def register(self, path: Path) -> None:
        self.paths.append(path)

    def run(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        *,
        allow_capture: bool = True,
    ) -> ForwardResult | None:
        ctx = get_forward_context()
        strict = bool(getattr(getattr(plan, "graph_policy", None), "strict", False))
        rejected: list[str] = []
        for path in self.paths:
            match = path.match(batch, plan)
            name = path.name(plan)
            if not match.accepted:
                rejected.append(f"{name}:{match.reason or 'ineligible'}")
                continue
            capacity = path.capacity(batch, plan)
            with use_forward_context(replace(ctx, allow_capture=bool(allow_capture))):
                result = path.run(batch, plan)
            if result is None:
                if strict:
                    logger.warning(
                        "strict graph path miss: path=%s rows=%d tokens=%d capture=%s",
                        name,
                        plan.shape.row_count,
                        plan.shape.token_count,
                        bool(allow_capture),
                    )
                return None
            if not isinstance(result, ForwardResult):
                raise invalid_descriptor("graph path must return a ForwardResult")
            result.graph = GraphInfo(
                path=name,
                capacity=capacity,
            )
            stats = ctx.stats
            if stats is not None:
                shape = repr(capacity)
                stats.forward_graph_shape_counts[shape] = (
                    int(stats.forward_graph_shape_counts.get(shape, 0)) + 1
                )
                stats.record_runtime_graph_topology(name)
            return result
        if strict:
            logger.warning(
                "strict graph miss: no path rows=%d tokens=%d reasons=%s",
                plan.shape.row_count,
                plan.shape.token_count,
                ";".join(rejected),
            )
        return None
