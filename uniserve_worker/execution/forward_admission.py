"""Admission gate for mixed-modality forward execution."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from ..contracts.forward_mode import ForwardMode, mode_for_op
from ..foundation.env import env_int

__all__ = [
    'Route',
    'ForwardAdmissionDecision',
    'ForwardAdmissionConfig',
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

_MAX_MEMORY_BOUND_TOKENS_ENV = "UNISERVE_FORWARD_MAX_MEMORY_BOUND_TOKENS"

# Default memory-bound window for the decode+denoise route. Below this token
# count the batch stays memory-bound (the forward kernel's precondition); above
# it the batch tips compute-bound and is rejected. Hardware-tunable via
# ``_MAX_MEMORY_BOUND_TOKENS_ENV``. The authoritative admission budget lives in
# the Rust scheduler — this is only the worker-side narrowing gate.
_DEFAULT_MAX_MEMORY_BOUND_TOKENS = 281


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
class ForwardAdmissionConfig:
    """Env-derived admission tunables, resolved once at the composition root."""

    max_memory_bound_decode_tokens: int = _DEFAULT_MAX_MEMORY_BOUND_TOKENS

    @classmethod
    def from_env(cls) -> "ForwardAdmissionConfig":
        max_memory_bound = env_int(
            _MAX_MEMORY_BOUND_TOKENS_ENV, default=_DEFAULT_MAX_MEMORY_BOUND_TOKENS
        )
        # A non-positive crossover would admit nothing; fall back to the default.
        if max_memory_bound <= 0:
            max_memory_bound = _DEFAULT_MAX_MEMORY_BOUND_TOKENS
        return cls(max_memory_bound_decode_tokens=max_memory_bound)


@dataclass(frozen=True)
class ForwardAdmissionRouter:
    config: ForwardAdmissionConfig = ForwardAdmissionConfig()

    @property
    def max_memory_bound_decode_tokens(self) -> int:
        return self.config.max_memory_bound_decode_tokens

    @classmethod
    def from_env(cls) -> "ForwardAdmissionRouter":
        return cls(config=ForwardAdmissionConfig.from_env())

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
        supported = {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.DENOISE}
        if any(mode not in supported for mode in modes):
            return ForwardAdmissionDecision(
                Route.PER_MODE,
                "mixed forward supports only text and denoise ops",
                modes,
            )
        has_decode = any(mode is ForwardMode.DECODE for mode in modes)
        has_denoise = any(mode is ForwardMode.DENOISE for mode in modes)
        if not (has_decode and has_denoise):
            return ForwardAdmissionDecision(Route.PER_MODE, "requires concurrent decode and denoise", modes)
        # The memory-bound precondition is about total text-token width, not just
        # decode rows. EXTEND (prefill) chunks are compute-bound and the host
        # budgets their full chunk width (op_token_cost), so count them against the
        # same window -- otherwise a large prefill could ride into the forward kernel
        # and make the batch compute-bound, violating the documented precondition.
        text_tokens = sum(
            len(op.get("token_ids") or [])
            for op, mode in zip(ops, modes)
            if mode in (ForwardMode.DECODE, ForwardMode.EXTEND)
        )
        if text_tokens > self.max_memory_bound_decode_tokens:
            return ForwardAdmissionDecision(
                Route.PER_MODE,
                "text batch is beyond the memory-bound forward window",
                modes,
            )
        return ForwardAdmissionDecision(Route.FORWARD, "decode+denoise forward window", modes)


def _has_values(raw: object) -> bool:
    if raw is None:
        return False
    try:
        return len(raw) > 0  # type: ignore[arg-type]
    except TypeError:
        return bool(raw)
