"""Admission gate for mixed-modality forward execution."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from ..contracts.forward_mode import ForwardMode, mode_for_op

__all__ = [
    'Route',
    'ForwardAdmissionDecision',
    'ForwardAdmissionRouter',
]


class Route(str, Enum):
    """Admission outcome. Values are the wire/log strings, preserved verbatim.

    Subclassing ``str`` keeps each member equal to (and serializable as) its
    string value, so ``decision.route`` stays string-compatible for callers and
    logging while the cascade dispatches on the enum.
    """

    PER_MODE = "per_mode"
    FORWARD = "forward"
    MODEL_CHECKED_FORWARD = "model_checked_forward"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ForwardAdmissionDecision:
    route: Route
    reason: str
    modes: tuple[ForwardMode, ...]

    @property
    def use_forward(self) -> bool:
        return self.route in {Route.FORWARD, Route.MODEL_CHECKED_FORWARD}

    @property
    def requires_model_acceptance(self) -> bool:
        return self.route is Route.MODEL_CHECKED_FORWARD


@dataclass(frozen=True)
class ForwardAdmissionRouter:
    @classmethod
    def from_runtime_config(cls) -> "ForwardAdmissionRouter":
        return cls()

    def decide(self, ops: Sequence[Mapping[str, object]]) -> ForwardAdmissionDecision:
        modes = tuple(mode_for_op(str(op.get("kind"))) for op in ops)
        if not ops:
            return ForwardAdmissionDecision(Route.PER_MODE, "empty batch", modes)
        text_modes = {ForwardMode.EXTEND, ForwardMode.DECODE}
        if set(modes).issubset(text_modes) and all(mode in modes for mode in text_modes):
            if any(_has_values(op.get("spec_token_ids")) for op in ops):
                return ForwardAdmissionDecision(
                    Route.PER_MODE,
                    "text mixed forward does not route speculative rows",
                    modes,
                )
            return ForwardAdmissionDecision(
                Route.MODEL_CHECKED_FORWARD, "text extend+decode mixed forward", modes
            )
        has_text = any(mode in text_modes for mode in modes)
        gen_modes = {ForwardMode.DENOISE, ForwardMode.COMMIT}
        has_gen = any(mode in gen_modes for mode in modes)
        if has_text and has_gen:
            return ForwardAdmissionDecision(Route.FORWARD, "und/gen mixed forward", modes)
        supported = {ForwardMode.EXTEND, ForwardMode.DECODE, *gen_modes}
        if any(mode not in supported for mode in modes):
            return ForwardAdmissionDecision(
                Route.PER_MODE,
                "mixed forward supports only text and gen ops",
                modes,
            )
        return ForwardAdmissionDecision(Route.PER_MODE, "requires concurrent und and gen ops", modes)


def _has_values(raw: object) -> bool:
    if raw is None:
        return False
    try:
        return len(raw) > 0  # type: ignore[arg-type]
    except TypeError:
        return bool(raw)
