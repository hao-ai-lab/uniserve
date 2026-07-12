"""Eager fallback policy and accounting for unified forward execution."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ...contracts.forward_mode import ForwardMode

__all__ = [
    "EagerFallbackReason",
    "EagerFallbackRecorder",
    "EagerFallbackWarning",
    "ForwardGraphPolicy",
    "StrictForwardGraphError",
]

logger = logging.getLogger(__name__)


class EagerFallbackReason(StrEnum):
    GRAPH_DISABLED = "graph_disabled"
    GRAPH_MISS = "graph_miss"
    GRAPH_INELIGIBLE = "graph_ineligible"
    CAPTURE_FAILURE = "capture_failure"
    REPLAY_FAILURE = "replay_failure"
    BACKEND_INELIGIBLE = "backend_ineligible"
    SHAPE_UNSUPPORTED = "shape_unsupported"
    STRICT_MODE_DISABLED = "strict_mode_disabled"
    CUDA_UNAVAILABLE = "cuda_unavailable"


@dataclass(frozen=True)
class ForwardGraphPolicy:
    prefer_graph: bool = True
    strict: bool = True
    allow_capture: bool = True
    graph_selection_delegated: bool = False


@dataclass(frozen=True)
class EagerFallbackWarning:
    reason: EagerFallbackReason
    mode: ForwardMode
    op_modes: tuple[ForwardMode, ...]
    tokens: int
    rows: int
    padded_tokens: int
    padded_rows: int
    program: str | None = None
    shape_key: Any | None = None
    backend: str | None = None


class StrictForwardGraphError(RuntimeError):
    def __init__(self, warning: EagerFallbackWarning) -> None:
        super().__init__(
            "strict forward graph policy rejected eager execution: "
            f"reason={warning.reason.value} mode={warning.mode.value}"
        )
        self.warning = warning


class EagerFallbackRecorder:
    """Rate-limit warning logs while counting every eager fallback."""

    def __init__(self) -> None:
        self._warned: set[tuple[Any, ...]] = set()
        self.counts: dict[str, int] = {}

    def record(self, warning: EagerFallbackWarning, *, stats: Any | None = None) -> None:
        key = (
            warning.reason.value,
            warning.program,
            repr(warning.shape_key),
            warning.mode.value,
            tuple(mode.value for mode in warning.op_modes),
        )
        self.counts[warning.reason.value] = self.counts.get(warning.reason.value, 0) + 1
        _bump_stats(stats, warning)
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(
            "forward eager fallback: reason=%s mode=%s rows=%d tokens=%d padded_rows=%d padded_tokens=%d program=%s backend=%s",
            warning.reason.value,
            warning.mode.value,
            warning.rows,
            warning.tokens,
            warning.padded_rows,
            warning.padded_tokens,
            warning.program,
            warning.backend,
        )


def _bump_stats(stats: Any | None, warning: EagerFallbackWarning) -> None:
    if stats is None:
        return
    for attr in ("cuda_graph_fallbacks", "forward_eager_fallbacks"):
        try:
            setattr(stats, attr, int(getattr(stats, attr, 0)) + 1)
        except Exception:
            pass
    try:
        setattr(stats, "forward_eager_tokens", int(getattr(stats, "forward_eager_tokens", 0)) + warning.tokens)
        setattr(stats, "forward_eager_rows", int(getattr(stats, "forward_eager_rows", 0)) + warning.rows)
    except Exception:
        pass
