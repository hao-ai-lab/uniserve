"""Generic paged-attention plan-pool seam."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "PagedAttentionPlan",
    "PagedAttentionPlanPool",
]


@dataclass(frozen=True)
class PagedAttentionPlan:
    """Pairs a plan cache key with its workspace and backend wrapper."""

    kind: str
    key: tuple[Any, ...]
    workspace: torch.Tensor | None = None
    wrapper: Any | None = None


@dataclass
class PagedAttentionPlanPool:
    """Owns plan cache keys, workspaces, and graph bindings for paged attention."""

    plans: dict[tuple[Any, ...], PagedAttentionPlan] = field(default_factory=dict)
    graph_bindings: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    workspaces: dict[tuple[str, int], torch.Tensor] = field(default_factory=dict)

    def plan_decode(
        self,
        key: tuple[Any, ...],
        *,
        workspace: torch.Tensor | None = None,
        wrapper: Any | None = None,
    ) -> PagedAttentionPlan:
        """Intern a decode plan by backend key and retain its workspace and wrapper."""

        return self._plan("decode", key, workspace=workspace, wrapper=wrapper)

    def plan_prefill(
        self,
        key: tuple[Any, ...],
        *,
        workspace: torch.Tensor | None = None,
        wrapper: Any | None = None,
    ) -> PagedAttentionPlan:
        """Intern a prefill plan by backend key and retain its workspace and wrapper."""

        return self._plan("prefill", key, workspace=workspace, wrapper=wrapper)

    def bind_graph(self, key: tuple[Any, ...], binding: Any) -> None:
        """Associate a plan key with the stable buffers owned by one CUDA graph."""

        self.graph_bindings[tuple(key)] = binding

    def workspace(
        self, device: torch.device | str, size: int, *, dtype: torch.dtype = torch.uint8
    ) -> torch.Tensor:
        """Return a reusable device workspace with at least the requested byte capacity."""

        target = torch.device(device)
        key = (str(target), int(size))
        cached = self.workspaces.get(key)
        if cached is None:
            with torch.inference_mode(False):
                cached = torch.empty(int(size), dtype=dtype, device=target)
            self.workspaces[key] = cached
        return cached

    def _plan(
        self,
        kind: str,
        key: tuple[Any, ...],
        *,
        workspace: torch.Tensor | None,
        wrapper: Any | None,
    ) -> PagedAttentionPlan:
        """Return or create a bounded decode or prefill plan for one cache key."""

        cache_key = (str(kind), *tuple(key))
        plan = self.plans.get(cache_key)
        if plan is None:
            plan = PagedAttentionPlan(
                kind=str(kind), key=cache_key, workspace=workspace, wrapper=wrapper
            )
            self.plans[cache_key] = plan
        return plan
