"""Worker roles and the work variants each is admitted to run."""

from __future__ import annotations

from enum import StrEnum

from ..execution.batch import ForwardMode


class WorkerRole(StrEnum):
    FULL = "full"
    ENCODER = "encoder"
    PREFILL = "prefill"
    DECODE = "decode"
    UND = "und"
    GEN = "gen"

    @property
    def allowed_work_variants(self) -> frozenset[ForwardMode]:
        """The work variants this deployment role is admitted to run."""

        return _ROUTES[self]

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


_ROUTES = {
    WorkerRole.FULL: frozenset(ForwardMode),
    WorkerRole.ENCODER: frozenset({ForwardMode.ENCODE_VISION, ForwardMode.ENCODE_LATENT}),
    WorkerRole.PREFILL: frozenset({ForwardMode.TOKEN_EXTEND}),
    WorkerRole.DECODE: frozenset(
        {
            ForwardMode.TOKEN_DECODE,
            ForwardMode.TOKEN_VERIFY,
            ForwardMode.MEDIA_PREPARE,
            ForwardMode.MEDIA_DENOISE,
            ForwardMode.MEDIA_RECONSTRUCT,
            ForwardMode.MATERIALIZE,
            ForwardMode.TRANSFER_KV_PUBLISH,
            ForwardMode.TRANSFER_KV_INSTALL,
        }
    ),
    WorkerRole.UND: frozenset(
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
    WorkerRole.GEN: frozenset(
        {
            ForwardMode.MEDIA_PREPARE,
            ForwardMode.MEDIA_DENOISE,
            ForwardMode.MEDIA_RECONSTRUCT,
            ForwardMode.MATERIALIZE,
        }
    ),
}


__all__ = ["WorkerRole"]
