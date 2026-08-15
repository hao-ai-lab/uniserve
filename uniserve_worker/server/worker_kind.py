"""Deployment roles and the exact wire work variants each is admitted to run."""

from __future__ import annotations

from enum import StrEnum

from ..batch import WorkVariant


class WorkerKind(StrEnum):
    FULL = "full"
    ENCODER = "encoder"
    PREFILL = "prefill"
    DECODE = "decode"
    UND = "und"
    GEN = "gen"

    @property
    def allowed_work_variants(self) -> frozenset[WorkVariant]:
        """The wire work variants this deployment role is admitted to run."""

        return _ROUTES[self]

    @classmethod
    def wire_values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


_ROUTES = {
    WorkerKind.FULL: frozenset(WorkVariant),
    WorkerKind.ENCODER: frozenset({WorkVariant.ENCODE_VISION, WorkVariant.ENCODE_LATENT}),
    WorkerKind.PREFILL: frozenset({WorkVariant.TOKEN_EXTEND}),
    WorkerKind.DECODE: frozenset(
        {
            WorkVariant.TOKEN_DECODE,
            WorkVariant.TOKEN_VERIFY,
            WorkVariant.GEN_TRANSITION,
            WorkVariant.GEN_FLOW,
            WorkVariant.MATERIALIZE,
            WorkVariant.TRANSFER_KV_PUBLISH,
            WorkVariant.TRANSFER_KV_INSTALL,
        }
    ),
    WorkerKind.UND: frozenset(
        {
            WorkVariant.TOKEN_EXTEND,
            WorkVariant.TOKEN_DECODE,
            WorkVariant.TOKEN_VERIFY,
            WorkVariant.ENCODE_VISION,
            WorkVariant.ENCODE_LATENT,
            WorkVariant.TRANSFER_KV_PUBLISH,
            WorkVariant.TRANSFER_KV_INSTALL,
        }
    ),
    WorkerKind.GEN: frozenset(
        {
            WorkVariant.GEN_TRANSITION,
            WorkVariant.GEN_FLOW,
            WorkVariant.MATERIALIZE,
        }
    ),
}


__all__ = ["WorkerKind"]
