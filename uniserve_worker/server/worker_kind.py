"""Deployment roles and their exact execution-operation routes."""

from __future__ import annotations

from enum import StrEnum

from ..spec import OperationType


class WorkerKind(StrEnum):
    FULL = "full"
    ENCODER = "encoder"
    PREFILL = "prefill"
    DECODE = "decode"
    SAMPLER = "sampler"
    POSTPROCESS = "postprocess"
    UND = "und"
    GEN = "gen"

    @property
    def allowed_operation_types(self) -> frozenset[OperationType]:
        """The route operation types this deployment role is admitted to run."""

        return _ROUTES[self]

    @classmethod
    def wire_values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


_ROUTES = {
    WorkerKind.FULL: frozenset(OperationType),
    WorkerKind.ENCODER: frozenset(
        {OperationType.ENCODE_VISION, OperationType.ENCODE_LATENT}
    ),
    WorkerKind.PREFILL: frozenset({OperationType.SEQUENCE_EXTEND}),
    WorkerKind.DECODE: frozenset(
        {
            OperationType.SEQUENCE_DECODE,
            OperationType.SEQUENCE_VERIFY,
            OperationType.FLOW,
            OperationType.MATERIALIZE_IMAGE,
            OperationType.TRANSFER_KV,
        }
    ),
    WorkerKind.SAMPLER: frozenset({OperationType.SEQUENCE_SAMPLE}),
    WorkerKind.POSTPROCESS: frozenset({OperationType.MATERIALIZE_FRAME}),
    WorkerKind.UND: frozenset(
        {
            OperationType.SEQUENCE_EXTEND,
            OperationType.SEQUENCE_DECODE,
            OperationType.SEQUENCE_VERIFY,
            OperationType.SEQUENCE_SAMPLE,
            OperationType.ENCODE_VISION,
            OperationType.ENCODE_LATENT,
            OperationType.TRANSFER_KV,
        }
    ),
    WorkerKind.GEN: frozenset(
        {
            OperationType.FLOW,
            OperationType.MATERIALIZE_IMAGE,
            OperationType.MATERIALIZE_FRAME,
        }
    ),
}


__all__ = ["WorkerKind"]
