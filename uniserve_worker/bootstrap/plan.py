"""Pure compilation of configured operation support into one worker plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..execution.batch import RunKind


class ModelLoadScope(StrEnum):
    WHOLE = "whole"
    UNDERSTANDING = "understanding"
    GENERATION = "generation"


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    model_scope: ModelLoadScope
    allowed_work_variants: frozenset[RunKind]


def resolve_worker_plan(supported_ops: frozenset[RunKind]) -> WorkerPlan:
    if not supported_ops:
        raise ValueError("worker pool must support at least one operation")
    return WorkerPlan(
        model_scope=ModelLoadScope.WHOLE,
        allowed_work_variants=supported_ops,
    )


__all__ = ["ModelLoadScope", "WorkerPlan", "resolve_worker_plan"]
