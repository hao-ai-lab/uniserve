"""Pure compilation of a deployment role into one worker plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..execution.batch import ForwardMode
from .role import WorkerRole


class ModelLoadScope(StrEnum):
    WHOLE = "whole"
    UNDERSTANDING = "understanding"
    GENERATION = "generation"


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    worker_role: WorkerRole
    model_scope: ModelLoadScope
    allowed_work_variants: frozenset[ForwardMode]


def resolve_worker_plan(worker_role: WorkerRole) -> WorkerPlan:
    if worker_role in {WorkerRole.FULL, WorkerRole.PREFILL, WorkerRole.DECODE, WorkerRole.ENCODER}:
        scope = ModelLoadScope.WHOLE
    elif worker_role is WorkerRole.UND:
        scope = ModelLoadScope.UNDERSTANDING
    elif worker_role is WorkerRole.GEN:
        scope = ModelLoadScope.GENERATION
    else:
        raise AssertionError(f"unhandled worker role {worker_role!r}")
    return WorkerPlan(
        worker_role=worker_role,
        model_scope=scope,
        allowed_work_variants=worker_role.allowed_work_variants,
    )


__all__ = ["ModelLoadScope", "WorkerPlan", "resolve_worker_plan"]
