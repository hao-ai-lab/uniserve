"""Worker stage kind: the OpKind subset a worker pool serves.

Mirrors the Rust ``uniserve_executor::WorkerKind``. A worker's kind determines
which OpKinds it accepts — the host's ``StageRouter`` routes by OpKind→pool — and
which driver :mod:`uniserve_worker.main` builds. ``full`` is the default: the
whole model, every model op in one mixed-batch forward.

The peeled kinds (encoder / prefill / decode / sampler / postprocess) are stages
split off ``full``. ``prefill`` and ``decode`` still load the whole model and run
the same runner; their kind only narrows the advertised OpKind subset so the
router sends each pool its phase's ops.
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
    "FULL",
    "ENCODER",
    "PREFILL",
    "DECODE",
    "SAMPLER",
    "POSTPROCESS",
    "UND",
    "GEN",
    "WORKER_KINDS",
    "SUPPORTED_OPS",
    "RUNNER_BACKED_KINDS",
    "validate_worker_kind",
    "supported_ops_for",
    "restrict_supported_ops",
    "tower_role_for_kind",
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


FULL = WorkerKind.FULL.value
ENCODER = WorkerKind.ENCODER.value
PREFILL = WorkerKind.PREFILL.value
DECODE = WorkerKind.DECODE.value
SAMPLER = WorkerKind.SAMPLER.value
POSTPROCESS = WorkerKind.POSTPROCESS.value
# Understanding/generation tower split: ``und`` runs text/vision-encode/sampling ops;
# ``gen`` runs image generation. Each pool is a tower device profile that loads
# only its tower's modules (via ``tower_role``); the host's StageRouter routes
# each tower its ops over the data plane.
UND = WorkerKind.UND.value
GEN = WorkerKind.GEN.value

WORKER_KINDS = frozenset(
    {FULL, ENCODER, PREFILL, DECODE, SAMPLER, POSTPROCESS, UND, GEN}
)

# OpKind subset each kind handles (mirrors Rust ``WorkerKind::supported_ops``).
SUPPORTED_OPS: Mapping[str, frozenset[str]] = {
    FULL: frozenset(
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
    ENCODER: frozenset({VIT_ENCODE, VAE_ENCODE}),
    PREFILL: frozenset({PREFILL_UND}),
    DECODE: frozenset({DECODE_UND, TARGET_VERIFY_UND, DENOISE_GEN, COMMIT_GEN, COMMIT_WRITEBACK}),
    SAMPLER: frozenset({"sample"}),
    POSTPROCESS: frozenset({"encode_frame"}),
    UND: frozenset(
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
    GEN: frozenset({DENOISE_GEN, COMMIT_GEN, "encode_frame"}),
}

# Kinds whose worker is model-backed and runs the shared ``ModelRunner``. ``full``,
# ``prefill``, and ``decode`` load the whole model; ``encoder`` loads vision-only
# weights; ``und``/``gen`` each load only their tower's modules; ``sampler`` and
# ``postprocess`` load no model.
RUNNER_BACKED_KINDS = frozenset({FULL, PREFILL, DECODE, UND, GEN})


def tower_role_for_kind(kind: str) -> str | None:
    """The tower role a worker kind materializes, or ``None`` for whole-model kinds.

    ``und``/``gen`` are Mode-A tower device profiles that load only their tower's
    modules (§6). The und side publishes any text-only CFG prefixes the gen side
    needs, so gen denoise does not re-enter the und text path.
    """
    if kind not in (UND, GEN):
        return None
    return "und" if kind == UND else "gen"


def validate_worker_kind(kind: str) -> str:
    if kind not in WORKER_KINDS:
        raise SystemExit(
            f"--worker-kind must be one of {sorted(WORKER_KINDS)}, got {kind!r}"
        )
    return kind


def supported_ops_for(kind: str) -> frozenset[str]:
    return SUPPORTED_OPS[kind]


def restrict_supported_ops(kind: str, model_ops: "list[str] | tuple[str, ...]") -> list[str]:
    """Intersect a model's declared ops with the kind's allowed subset, in the
    model's declared order. For ``full`` this is the model's set unchanged (every
    model op is in ``SUPPORTED_OPS[full]``); for a phase kind it narrows to that
    phase. A peeled stage (sampler/postprocess) keeps its own ops even when the
    backing object declares none.
    """
    allowed = SUPPORTED_OPS[kind]
    restricted = [op for op in model_ops if op in allowed]
    if restricted:
        return restricted
    # No overlap: the stage's ops are entirely its own (sampler/postprocess).
    return sorted(allowed)
