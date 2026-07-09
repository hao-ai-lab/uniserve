"""Unified CUDA graph runner orchestration."""
from __future__ import annotations

from typing import Callable

from ....contracts.forward_batch import ForwardBatch
from ..plan import ForwardPlan
from ..result import ForwardGraphExecutionInfo, ForwardResult
from .programs import CapturedForwardGraph, ForwardGraphProgram
from .stats import ForwardGraphStats

__all__ = ["CudaGraphForwardRunner"]


class CudaGraphForwardRunner:
    def __init__(
        self,
        programs: tuple[ForwardGraphProgram, ...] = (),
        *,
        stats: ForwardGraphStats | None = None,
    ) -> None:
        self.programs = list(programs)
        self.stats = stats or ForwardGraphStats()
        self._graphs: dict[object, CapturedForwardGraph] = {}

    def register(self, program: ForwardGraphProgram) -> None:
        self.programs.append(program)

    def run(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        *,
        forward_fn: Callable[[ForwardBatch], ForwardResult],
        allow_capture: bool = True,
    ) -> ForwardResult | None:
        for program in self.programs:
            eligibility = program.can_run(batch, plan)
            if not eligibility.eligible:
                continue
            key = program.shape_key(batch, plan)
            graph = self._graphs.get(key)
            if graph is not None:
                result = program.replay(graph, batch, plan)
                result.graph = ForwardGraphExecutionInfo(
                    program=program.program_id,
                    shape_key=key,
                    replayed=True,
                )
                self.stats.record_replay(key)
                return result
            if not allow_capture:
                self.stats.record_miss(plan.forward_mode.value)
                return None
            graph = program.capture(key, batch, plan, forward_fn)
            self._graphs[key] = graph
            result = graph.payload
            if not isinstance(result, ForwardResult):
                result = forward_fn(batch)
                graph.payload = result
            result.graph = ForwardGraphExecutionInfo(
                program=program.program_id,
                shape_key=key,
                captured=True,
                replayed=True,
            )
            self.stats.record_capture(key)
            return result
        self.stats.record_miss(plan.forward_mode.value)
        return None
