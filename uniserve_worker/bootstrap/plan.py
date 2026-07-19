"""Pure compilation of a deployment role into a Python worker plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..contracts.model_family import ModelLoadScope
from ..server.worker_kind import WorkerKind
from ..worker.protocol import ResultPolicy


class WorkerImplementation(StrEnum):
    MODEL = "model"
    ENCODER = "encoder"
    SAMPLER = "sampler"
    FRAME_ACCUMULATOR = "frame_accumulator"


@dataclass(frozen=True)
class WorkerPlan:
    """Resolved implementation choices for one worker process."""

    worker_kind: WorkerKind
    implementation: WorkerImplementation
    model_scope: ModelLoadScope | None
    allowed_ops: frozenset[str]
    result_policy: ResultPolicy

    @property
    def requires_model(self) -> bool:
        return self.model_scope is not None


def resolve_worker_plan(worker_kind: WorkerKind) -> WorkerPlan:
    """Resolve all role-sensitive Python choices in one place."""

    if worker_kind in {
        WorkerKind.FULL,
        WorkerKind.PREFILL,
        WorkerKind.DECODE,
    }:
        implementation = WorkerImplementation.MODEL
        model_scope = ModelLoadScope.WHOLE
    elif worker_kind is WorkerKind.UND:
        implementation = WorkerImplementation.MODEL
        model_scope = ModelLoadScope.UNDERSTANDING
    elif worker_kind is WorkerKind.GEN:
        implementation = WorkerImplementation.MODEL
        model_scope = ModelLoadScope.GENERATION
    elif worker_kind is WorkerKind.ENCODER:
        implementation = WorkerImplementation.ENCODER
        # SenseNova's encode path extends the language-model KV state, so an
        # encoder-only weight slice is not a valid general contract today.
        model_scope = ModelLoadScope.WHOLE
    elif worker_kind is WorkerKind.SAMPLER:
        implementation = WorkerImplementation.SAMPLER
        model_scope = None
    elif worker_kind is WorkerKind.POSTPROCESS:
        implementation = WorkerImplementation.FRAME_ACCUMULATOR
        model_scope = None
    else:  # pragma: no cover - exhaustive over WorkerKind
        raise AssertionError(f"unhandled worker kind {worker_kind!r}")

    result_policy = (
        ResultPolicy.SYNCHRONOUS
        if worker_kind is WorkerKind.UND
        else ResultPolicy.DEFER_WHEN_AVAILABLE
    )
    return WorkerPlan(
        worker_kind=worker_kind,
        implementation=implementation,
        model_scope=model_scope,
        allowed_ops=worker_kind.supported_ops,
        result_policy=result_policy,
    )
