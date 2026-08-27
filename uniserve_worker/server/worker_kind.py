"""Deployment roles and the exact wire work variants each is admitted to run."""

from __future__ import annotations

from enum import StrEnum

from ..batch import ForwardMode


class WorkerKind(StrEnum):
    FULL = "full"
    ENCODER = "encoder"
    PREFILL = "prefill"
    DECODE = "decode"
    UND = "und"
    GEN = "gen"

    @property
    def allowed_work_variants(self) -> frozenset[ForwardMode]:
        """The wire work variants this deployment role is admitted to run."""

        return _ROUTES[self]

    @classmethod
    def wire_values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


_ROUTES = {
    WorkerKind.FULL: frozenset(ForwardMode) - {ForwardMode.DRAFT},
    WorkerKind.ENCODER: frozenset({ForwardMode.ENCODE_VISION, ForwardMode.ENCODE_LATENT}),
    WorkerKind.PREFILL: frozenset({ForwardMode.TOKEN_EXTEND}),
    WorkerKind.DECODE: frozenset(
        {
            ForwardMode.TOKEN_DECODE,
            ForwardMode.TOKEN_VERIFY,
            ForwardMode.TRANSFER_KV_PUBLISH,
            ForwardMode.TRANSFER_KV_INSTALL,
        }
    ),
    WorkerKind.UND: frozenset(
        {
            ForwardMode.TOKEN_EXTEND,
            ForwardMode.TOKEN_DECODE,
            ForwardMode.TOKEN_VERIFY,
            ForwardMode.ENCODE_VISION,
            ForwardMode.ENCODE_LATENT,
            ForwardMode.TRANSFER_KV_PUBLISH,
            ForwardMode.TRANSFER_KV_INSTALL,
        }
    ),
    WorkerKind.GEN: frozenset(
        {
            ForwardMode.GEN_TRANSITION,
            ForwardMode.GEN_FLOW,
            ForwardMode.GEN_DECODE,
            ForwardMode.MATERIALIZE,
        }
    ),
}


__all__ = ["WorkerKind"]
