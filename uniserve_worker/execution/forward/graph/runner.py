"""Unified CUDA graph runner orchestration."""
from __future__ import annotations

import logging
from typing import Callable

from ....contracts.forward_batch import ForwardBatch
from ....contracts.forward_context import get_forward_context
from ..plan import ForwardPlan
from ..result import ForwardGraphExecutionInfo, ForwardResult
from .programs import CapturedForwardGraph, ForwardGraphProgram
from .stats import ForwardGraphStats

__all__ = ["CudaGraphForwardRunner"]

logger = logging.getLogger(__name__)


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
        forward_stats = get_forward_context().stats
        ineligible: list[str] = []
        strict = bool(getattr(getattr(plan, "graph_policy", None), "strict", False))
        for program in self.programs:
            eligibility = program.can_run(batch, plan)
            if not eligibility.eligible:
                ineligible.append(f"{program.program_id}:{eligibility.reason or 'ineligible'}")
                continue
            key = program.shape_key(batch, plan)
            graph = self._graphs.get(key)
            if graph is not None:
                result = program.replay(graph, batch, plan)
                if result is None:
                    if strict:
                        logger.warning(
                            "strict forward graph replay miss: program=%s mode=%s rows=%d tokens=%d",
                            program.program_id,
                            plan.forward_mode.value,
                            plan.shape.row_count,
                            plan.shape.token_count,
                        )
                    self.stats.record_miss(plan.forward_mode.value, forward_stats=forward_stats)
                    return None
                result.graph = ForwardGraphExecutionInfo(
                    program=program.program_id,
                    shape_key=key,
                    replayed=True,
                )
                self.stats.record_replay(
                    key,
                    unpadded_tokens=plan.shape.token_count,
                    forward_stats=forward_stats,
                )
                return result
            if not allow_capture:
                if strict:
                    logger.warning(
                        "strict forward graph capture disabled: program=%s mode=%s rows=%d tokens=%d",
                        program.program_id,
                        plan.forward_mode.value,
                        plan.shape.row_count,
                        plan.shape.token_count,
                    )
                self.stats.record_miss(plan.forward_mode.value, forward_stats=forward_stats)
                return None
            graph = program.capture(key, batch, plan, forward_fn)
            if graph is None:
                if strict:
                    logger.warning(
                        "strict forward graph capture miss: program=%s mode=%s rows=%d tokens=%d",
                        program.program_id,
                        plan.forward_mode.value,
                        plan.shape.row_count,
                        plan.shape.token_count,
                    )
                self.stats.record_miss(plan.forward_mode.value, forward_stats=forward_stats)
                return None
            self._graphs[key] = graph
            result = graph.payload
            if not isinstance(result, ForwardResult):
                result = forward_fn(batch)
                graph.payload = result
            result.graph = ForwardGraphExecutionInfo(
                program=program.program_id,
                shape_key=key,
                captured=True,
            )
            self.stats.record_capture(
                key,
                unpadded_tokens=plan.shape.token_count,
                forward_stats=forward_stats,
            )
            return result
        if strict:
            logger.warning(
                "strict forward graph miss: no eligible program mode=%s rows=%d tokens=%d reasons=%s",
                plan.forward_mode.value,
                plan.shape.row_count,
                plan.shape.token_count,
                ";".join(ineligible),
            )
        self.stats.record_miss(plan.forward_mode.value, forward_stats=forward_stats)
        return None
