"""Cross-language worker-role vocabulary.

``WorkerKind`` is a deployment role used by the host to select a pool. It owns
only the role's wire token and operation envelope; Python model-loading and
execution implementation choices belong to :mod:`uniserve_worker.bootstrap`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping

from ..contracts.op_kinds import (
    COMMIT_GEN,
    COMMIT_WRITEBACK,
    DECODE_UND,
    DENOISE_GEN,
    PREFILL_UND,
    TARGET_VERIFY_UND,
    VAE_ENCODE,
    VIT_ENCODE,
)

__all__ = [
    "WorkerKind",
]


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
    def supported_ops(self) -> frozenset[str]:
        """Operation envelope routed to this worker role."""
        return _SUPPORTED_OPS_BY_KIND[self]

    @classmethod
    def wire_values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


# Mirrors Rust ``WorkerKind::supported_ops``. This is protocol vocabulary, not a
# Python implementation registry.
_SUPPORTED_OPS_BY_KIND: Mapping[WorkerKind, frozenset[str]] = {
    WorkerKind.FULL: frozenset(
        {
            PREFILL_UND,
            DECODE_UND,
            TARGET_VERIFY_UND,
            DENOISE_GEN,
            COMMIT_GEN,
            COMMIT_WRITEBACK,
            VAE_ENCODE,
            VIT_ENCODE,
        }
    ),
    WorkerKind.ENCODER: frozenset({VIT_ENCODE, VAE_ENCODE}),
    WorkerKind.PREFILL: frozenset({PREFILL_UND}),
    WorkerKind.DECODE: frozenset(
        {DECODE_UND, TARGET_VERIFY_UND, DENOISE_GEN, COMMIT_GEN, COMMIT_WRITEBACK}
    ),
    WorkerKind.SAMPLER: frozenset({"sample"}),
    WorkerKind.POSTPROCESS: frozenset({"encode_frame"}),
    WorkerKind.UND: frozenset(
        {
            PREFILL_UND,
            DECODE_UND,
            TARGET_VERIFY_UND,
            VIT_ENCODE,
            VAE_ENCODE,
            "sample",
            COMMIT_WRITEBACK,
        }
    ),
    WorkerKind.GEN: frozenset({DENOISE_GEN, COMMIT_GEN, "encode_frame"}),
}
