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
    """Own every runtime-scoped graph executor for one loaded worker.

    ``spec_digest`` is the resolved ModelSpec + DeploymentOverlay identity that
    scopes every graph key in this store; captures are per-process, so one
    store never holds graphs for two model identities.
    """

    def __init__(
        self,
        *,
        text: Any = None,
        flow: Any = None,
        segment: Any = None,
        spec_digest: str | None = None,
    ) -> None:
        self._view = GraphView(text=text, flow=flow, segment=segment)
        self.spec_digest = spec_digest

    def view(self) -> GraphView:
        return self._view
