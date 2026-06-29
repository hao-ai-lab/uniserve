"""Typed batching policy for runner-owned grouping."""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts.forward_mode import ForwardMode
from ..foundation.errors import invalid_descriptor

__all__ = [
    'BatchPolicy',
]


@dataclass(frozen=True)
class BatchPolicy:
    max_batch_ops: int = 1
    supports_mixed_modes: bool = False
    mode_order: tuple[ForwardMode, ...] = (
        ForwardMode.ENCODE,
        ForwardMode.EXTEND,
        ForwardMode.DECODE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.DENOISE,
        ForwardMode.COMMIT,
    )

    def __post_init__(self) -> None:
        if self.max_batch_ops < 1:
            raise invalid_descriptor("BatchPolicy.max_batch_ops must be at least 1")
        seen: set[ForwardMode] = set()
        for mode in self.mode_order:
            if not isinstance(mode, ForwardMode):
                raise invalid_descriptor("BatchPolicy.mode_order entries must be ForwardMode values")
            if mode in seen:
                raise invalid_descriptor("BatchPolicy.mode_order must not contain duplicates")
            seen.add(mode)

    def allows_group(self, modes: list[ForwardMode]) -> bool:
        if not modes:
            return False
        if len(modes) > self.max_batch_ops:
            return False
        return self.supports_mixed_modes or all(mode == modes[0] for mode in modes)
