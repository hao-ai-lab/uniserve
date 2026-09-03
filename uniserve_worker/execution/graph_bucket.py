"""Worker-local captured execution shapes."""

from __future__ import annotations

from dataclasses import dataclass

from ..foundation.errors import invalid_descriptor


@dataclass(frozen=True, slots=True)
class GraphBucket:
    decode_rows: int
    flow_rows: int
    height: int
    width: int
    cfg_branches: int

    def __post_init__(self) -> None:
        if min(
            self.decode_rows,
            self.flow_rows,
            self.height,
            self.width,
            self.cfg_branches,
        ) < 1:
            raise invalid_descriptor("captured execution bucket dimensions must be positive")


__all__ = ["GraphBucket"]
