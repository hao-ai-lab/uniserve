"""Attention backend selection over provider support checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...ops.requests import AttentionReq

__all__ = [
    "AttentionAdapter",
    "AttentionBackendSelector",
]


@dataclass(frozen=True)
class AttentionAdapter:
    """Selected attention provider plus its support contract."""

    provider: Any

    @property
    def name(self) -> str:
        return str(getattr(self.provider, "name", type(self.provider).__name__))

    def capabilities(self) -> Any:
        return self.provider.capabilities()

    def supports(self, req: AttentionReq) -> bool:
        can_run = getattr(self.provider, "can_run", None)
        return bool(can_run(req)) if callable(can_run) else False

    def run(self, req: AttentionReq) -> Any:
        return self.provider.run(req)


class AttentionBackendSelector:
    """Owns attention preference order, compatibility checks, and fallback."""

    def __init__(self, dispatcher: Any | None = None) -> None:
        self._dispatcher = dispatcher

    def adapters(self, preference: str | None = "auto") -> tuple[AttentionAdapter, ...]:
        dispatcher = self._dispatcher
        if dispatcher is None:
            import uniserve_worker.ops as ops

            dispatcher = ops.attention_dispatcher()
        return tuple(
            AttentionAdapter(provider)
            for provider in dispatcher.ordered(preference or "auto")
        )

    def select(
        self,
        attention_req: AttentionReq,
        preference: str | None = "auto",
    ) -> AttentionAdapter:
        for adapter in self.adapters(preference):
            if adapter.supports(attention_req):
                return adapter
        raise RuntimeError("no attention backend supports the request")

    def capability_rows(self, preference: str | None = "auto") -> tuple[tuple[str, dict[str, Any]], ...]:
        rows: list[tuple[str, dict[str, Any]]] = []
        for adapter in self.adapters(preference):
            if adapter.name == "context":
                continue
            try:
                caps = adapter.capabilities()
            except Exception:
                continue
            attrs = dict(getattr(caps, "attrs", {}) or {})
            rows.append((adapter.name, attrs))
        return tuple(rows)
