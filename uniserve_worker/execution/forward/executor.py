"""Unified forward executor with graph selection and eager fallback policy."""
from __future__ import annotations

from typing import Any, Callable

from ...contracts.forward_batch import ForwardBatch
from ...contracts.forward_context import get_forward_context
from .fallback import (
    EagerFallbackReason,
    EagerFallbackRecorder,
    EagerFallbackWarning,
    ForwardGraphPolicy,
    StrictForwardGraphError,
)
from .plan import ForwardPlan
from .result import ForwardResult
from .runner import EagerForwardRunner

__all__ = ["ForwardExecutor"]


class ForwardExecutor:
    def __init__(
        self,
        *,
        model: Any | None = None,
        graph_runner: Any | None = None,
        eager_runner: EagerForwardRunner | None = None,
        graph_policy: ForwardGraphPolicy | None = None,
        fallback_recorder: EagerFallbackRecorder | None = None,
    ) -> None:
        self.model = model
        self.graph_runner = graph_runner
        self.eager_runner = eager_runner or EagerForwardRunner(model)
        self.graph_policy = graph_policy or ForwardGraphPolicy()
        self.fallback_recorder = fallback_recorder or EagerFallbackRecorder()

    def execute(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        *,
        forward_fn: Callable[[ForwardBatch], Any] | None = None,
    ) -> ForwardResult:
        policy = plan.graph_policy or self.graph_policy
        stats = get_forward_context().stats
        if policy.graph_selection_delegated:
            return self.eager_runner.run(batch, plan, forward_fn=forward_fn)
        if policy.prefer_graph and self.graph_runner is not None:
            try:
                graph_result = self.graph_runner.run(
                    batch,
                    plan,
                    forward_fn=lambda graph_batch: self.eager_runner.run(
                        graph_batch,
                        plan,
                        forward_fn=forward_fn,
                    ),
                    allow_capture=policy.allow_capture,
                )
            except Exception as exc:
                warning = self._warning(EagerFallbackReason.REPLAY_FAILURE, plan)
                if policy.strict:
                    raise StrictForwardGraphError(warning) from exc
                self.fallback_recorder.record(warning, stats=stats)
            else:
                if graph_result is not None:
                    return graph_result
                warning = self._warning(EagerFallbackReason.GRAPH_MISS, plan)
                if policy.strict:
                    raise StrictForwardGraphError(warning)
                self.fallback_recorder.record(warning, stats=stats)
        else:
            warning = self._warning(EagerFallbackReason.GRAPH_DISABLED, plan)
            if policy.strict:
                raise StrictForwardGraphError(warning)
            self.fallback_recorder.record(warning, stats=stats)
        return self.eager_runner.run(batch, plan, forward_fn=forward_fn)

    @staticmethod
    def _warning(reason: EagerFallbackReason, plan: ForwardPlan) -> EagerFallbackWarning:
        return EagerFallbackWarning(
            reason=reason,
            mode=plan.forward_mode,
            op_modes=plan.op_modes,
            tokens=plan.shape.token_count,
            rows=plan.shape.row_count,
            padded_tokens=plan.shape.padded_token_count,
            padded_rows=plan.shape.padded_row_count,
        )
