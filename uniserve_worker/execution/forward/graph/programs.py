"""Graph program interfaces for unified forward execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from ....contracts.forward_batch import ForwardBatch
from ..plan import ForwardPlan
from ..result import ForwardResult
from .key import ForwardGraphShapeKey, graph_shape_key

__all__ = [
    "CapturedForwardGraph",
    "DecodeGraphProgram",
    "DenoiseStepGraphProgram",
    "ForwardGraphProgram",
    "GraphEligibility",
    "PackedVisibleGraphProgram",
    "PrefillGraphProgram",
]


@dataclass(frozen=True)
class GraphEligibility:
    eligible: bool
    reason: str = ""


@dataclass
class CapturedForwardGraph:
    key: ForwardGraphShapeKey
    program: "ForwardGraphProgram"
    payload: Any = None


class ForwardGraphProgram(ABC):
    program_id: str = "program"

    @abstractmethod
    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility: ...

    def shape_key(self, batch: ForwardBatch, plan: ForwardPlan) -> ForwardGraphShapeKey:
        return graph_shape_key(program=self.program_id, batch=batch, plan=plan)

    def capture(
        self,
        key: ForwardGraphShapeKey,
        batch: ForwardBatch,
        plan: ForwardPlan,
        forward_fn: Callable[[ForwardBatch], ForwardResult],
    ) -> CapturedForwardGraph:
        del plan
        return CapturedForwardGraph(key=key, program=self, payload=forward_fn(batch))

    def replay(
        self,
        graph: CapturedForwardGraph,
        batch: ForwardBatch,
        plan: ForwardPlan,
    ) -> ForwardResult:
        del batch, plan
        if not isinstance(graph.payload, ForwardResult):
            raise TypeError("captured forward graph payload must be a ForwardResult")
        return graph.payload


class DecodeGraphProgram(ForwardGraphProgram):
    program_id = "decode"

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count == plan.shape.row_count and all(
            segment.segment_class.value == "decode" for segment in plan.segments
        ):
            return GraphEligibility(True)
        return GraphEligibility(False, "not a pure decode shape")


class PrefillGraphProgram(ForwardGraphProgram):
    program_id = "prefill"

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count == plan.shape.row_count and plan.shape.token_count > 0:
            return GraphEligibility(True)
        return GraphEligibility(False, "not a text prefill shape")


class PackedVisibleGraphProgram(ForwardGraphProgram):
    program_id = "packed_visible"

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count and plan.shape.denoise_row_count:
            return GraphEligibility(True)
        return GraphEligibility(False, "not a packed visible mixed shape")


class DenoiseStepGraphProgram(ForwardGraphProgram):
    program_id = "denoise_step"

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.denoise_row_count == plan.shape.row_count:
            return GraphEligibility(True)
        return GraphEligibility(False, "not a pure denoise shape")
