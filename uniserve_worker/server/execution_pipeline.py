"""Execution pipeline and pending-result finalization state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ExecutionPipeline",
    "PendingResult",
]


@dataclass
class PendingResult:
    """Execute response state that may require deferred CPU finalization."""

    response: dict[str, Any]
    batch: dict[str, Any] | None = None
    wait_start_ns: int | None = None

    @property
    def result(self) -> dict[str, Any] | None:
        result = self.response.get("result")
        return result if isinstance(result, dict) else None

    @property
    def has_deferred(self) -> bool:
        per_seq = (self.result or {}).get("per_seq")
        return isinstance(per_seq, list) and any(
            hasattr(item, "finalize") and callable(item.finalize)
            for item in per_seq
        )

    def ready(self) -> bool:
        per_seq = (self.result or {}).get("per_seq")
        if not isinstance(per_seq, list):
            return True
        for item in per_seq:
            ready = getattr(item, "ready", None)
            if callable(ready) and not bool(ready()):
                return False
        return True


class ExecutionPipeline:
    """Small execute-response shaper used by the worker runtime."""

    def immediate(self, response: dict[str, Any]) -> PendingResult:
        return PendingResult(response=response)

    def pending(
        self,
        *,
        result: dict[str, Any],
        batch: dict[str, Any],
        wait_start_ns: int | None,
    ) -> PendingResult:
        response = {"kind": "result", "result": result}
        pending = PendingResult(response=response)
        per_seq = result.get("per_seq")
        if isinstance(per_seq, list) and any(
            hasattr(item, "finalize") and callable(item.finalize)
            for item in per_seq
        ):
            pending.batch = batch
            pending.wait_start_ns = wait_start_ns
        return pending
