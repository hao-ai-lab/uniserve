"""System authority for runtime-scoped compiled graph executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["GraphStore", "GraphView"]


@dataclass(frozen=True, slots=True)
class GraphView:
    """Borrowed graph executors available to one forward."""

    text: Any = None
    flow: Any = None
    segment: Any = None


class GraphStore:
    """Own every runtime-scoped graph executor for one loaded worker."""

    def __init__(self, *, text: Any = None, flow: Any = None, segment: Any = None) -> None:
        self._view = GraphView(text=text, flow=flow, segment=segment)

    def view(self) -> GraphView:
        return self._view
