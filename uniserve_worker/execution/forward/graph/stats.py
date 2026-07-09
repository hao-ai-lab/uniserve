"""Unified graph metrics aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ForwardGraphStats"]


@dataclass
class ForwardGraphStats:
    captures: int = 0
    replays: int = 0
    misses: int = 0
    fallbacks: int = 0
    capture_failures: int = 0
    replay_failures: int = 0
    shape_counts: dict[str, int] = field(default_factory=dict)
    mode_counts: dict[str, int] = field(default_factory=dict)
    padded_tokens: int = 0
    unpadded_tokens: int = 0

    def record_capture(self, key: Any) -> None:
        self.captures += 1
        self._record_key(key)

    def record_replay(self, key: Any) -> None:
        self.replays += 1
        self._record_key(key)

    def record_miss(self, mode: str) -> None:
        self.misses += 1
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1

    def _record_key(self, key: Any) -> None:
        text = repr(key)
        self.shape_counts[text] = self.shape_counts.get(text, 0) + 1
        mode = getattr(getattr(key, "mode", None), "value", str(getattr(key, "mode", "")))
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1
        self.padded_tokens += int(getattr(key, "token_bucket", 0) or 0)

    def to_wire(self) -> dict[str, Any]:
        return {
            "forward_graph_captures": self.captures,
            "forward_graph_replays": self.replays,
            "forward_graph_misses": self.misses,
            "forward_graph_fallbacks": self.fallbacks,
            "forward_graph_capture_failures": self.capture_failures,
            "forward_graph_replay_failures": self.replay_failures,
            "forward_graph_runtime_mode_counts": dict(self.mode_counts),
            "forward_graph_shape_counts": dict(self.shape_counts),
            "forward_graph_padded_tokens": self.padded_tokens,
            "forward_graph_unpadded_tokens": self.unpadded_tokens,
        }
