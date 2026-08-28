"""Pure compilation of a deployment role into one worker plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..execution.batch import ForwardMode
from ..server.worker_kind import WorkerKind


class ModelLoadScope(StrEnum):
    WHOLE = "whole"
    UNDERSTANDING = "understanding"
    GENERATION = "generation"


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    worker_kind: WorkerKind
    model_scope: ModelLoadScope
    allowed_work_variants: frozenset[ForwardMode]


def resolve_worker_plan(worker_kind: WorkerKind) -> WorkerPlan:
    if worker_kind in {WorkerKind.FULL, WorkerKind.PREFILL, WorkerKind.DECODE, WorkerKind.ENCODER}:
        scope = ModelLoadScope.WHOLE
    elif worker_kind is WorkerKind.UND:
        scope = ModelLoadScope.UNDERSTANDING
    elif worker_kind is WorkerKind.GEN:
        scope = ModelLoadScope.GENERATION
    else:
        raise AssertionError(f"unhandled worker kind {worker_kind!r}")
    return WorkerPlan(
        worker_kind=worker_kind,
        model_scope=scope,
        allowed_work_variants=worker_kind.allowed_work_variants,
    )


__all__ = ["ModelLoadScope", "WorkerPlan", "resolve_worker_plan"]
