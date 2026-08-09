"""Pure compilation of a deployment role into one worker plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..batch import WorkVariant
from ..loader.schema import ModelLoadScope
from ..server.worker_kind import WorkerKind


class WorkerImplementation(StrEnum):
    MODEL = "model"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    worker_kind: WorkerKind
    implementation: WorkerImplementation
    model_scope: ModelLoadScope | None
    allowed_work_variants: frozenset[WorkVariant]

    @property
    def requires_model(self) -> bool:
        return self.model_scope is not None


def resolve_worker_plan(worker_kind: WorkerKind) -> WorkerPlan:
    if worker_kind in {WorkerKind.FULL, WorkerKind.PREFILL, WorkerKind.DECODE, WorkerKind.ENCODER}:
        scope = ModelLoadScope.WHOLE
        implementation = WorkerImplementation.MODEL
    elif worker_kind is WorkerKind.UND:
        scope = ModelLoadScope.UNDERSTANDING
        implementation = WorkerImplementation.MODEL
    elif worker_kind is WorkerKind.GEN:
        scope = ModelLoadScope.GENERATION
        implementation = WorkerImplementation.MODEL
    elif worker_kind in {WorkerKind.SAMPLER, WorkerKind.POSTPROCESS}:
        scope = None
        implementation = WorkerImplementation.SYSTEM
    else:
        raise AssertionError(f"unhandled worker kind {worker_kind!r}")
    return WorkerPlan(
        worker_kind=worker_kind,
        implementation=implementation,
        model_scope=scope,
        allowed_work_variants=worker_kind.allowed_work_variants,
    )


__all__ = ["WorkerImplementation", "WorkerPlan", "resolve_worker_plan"]
