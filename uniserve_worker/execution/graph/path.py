"""Bindings from forward plans to physical graph owners."""

from __future__ import annotations

import inspect
from typing import Any

from uniserve_worker.contracts.batches import UniForwardBatch
from uniserve_worker.contracts.forward_batch import ForwardBatch, ForwardPlan, ForwardResult
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.foundation.errors import invalid_descriptor

from .dispatch import Match, Path


class Segment(Path):
    """Run any active segment composition through its system executor."""

    def __init__(
        self,
        *,
        executor: Any | None = None,
        states: Any | None = None,
        publisher: Any | None = None,
    ) -> None:
        self.executor = executor
        self.states = states
        self.publisher = publisher

    def name(self, plan: ForwardPlan) -> str:
        del plan
        return "segment"

    def match(self, batch: ForwardBatch, plan: ForwardPlan) -> Match:
        del batch
        if plan.shape.segment_count <= 0:
            return Match(False, "plan has no active segments")
        if self.executor is None or self.states is None:
            return Match(False, "segment executor is unavailable")
        return Match(True)

    def run(self, batch: ForwardBatch, plan: ForwardPlan) -> ForwardResult | None:
        del batch
        if self.executor is None or self.states is None:
            return None
        result = self.executor.run_segment_graph(
            plan,
            request_states=self.states,
            result_publisher=self.publisher,
        )
        if result is not None and not isinstance(result, ForwardResult):
            raise invalid_descriptor("segment path must return a ForwardResult")
        return result


class Batch(Path):
    """Run token-span rows through the system or owner-provided executor."""

    def __init__(
        self,
        *,
        driver: Any | None = None,
        model: Any | None = None,
        states: Any | None = None,
        executor: Any | None = None,
    ) -> None:
        self.driver = driver
        self.model = model
        self.states = states
        self.executor = executor

    def name(self, plan: ForwardPlan) -> str:
        if len(plan.segments) == len(plan.rows) and all(
            segment.segment_class.value == "decode" for segment in plan.segments
        ):
            return "step"
        return "span"

    def match(self, batch: ForwardBatch, plan: ForwardPlan) -> Match:
        del batch
        if not all(row.token_span is not None for row in plan.rows):
            return Match(False, "rows are not token spans")
        if plan.shape.token_count <= 0:
            return Match(False, "plan has no tokens")
        if not self._bound:
            return Match(False, "batch executor is unavailable")
        for row in plan.rows:
            if row.op.get("spec_token_ids"):
                return Match(False, "speculative rows are not graphable")
            if (
                row.mode is ForwardMode.DECODE
                and int(row.op.get("decode_token_count") or 1) > 1
                and not callable(getattr(self.driver, "forward_graph_result", None))
            ):
                return Match(False, "multi-step rows are not graphable")
        return Match(True)

    @property
    def _bound(self) -> bool:
        if self.driver is None or self.model is None or self.states is None:
            return False
        return self.executor is not None or callable(
            getattr(self.model, "try_run_graph_logits_batch", None)
        )

    def run(self, batch: ForwardBatch, plan: ForwardPlan) -> ForwardResult | None:
        del batch
        if not self._bound:
            return None
        dispatch = plan.runtime_handles.get("dispatch_batch")
        if not isinstance(dispatch, UniForwardBatch):
            dispatch = UniForwardBatch.from_ops(plan.ops)
        forward = getattr(self.driver, "forward_graph_result", None)
        if callable(forward):
            result = forward(
                dispatch,
                self.states,
                self.model,
                graph_runner=self.executor,
                defer_cpu_results=bool(plan.runtime_handles.get("defer_text_cpu_results", False)),
                defer_sampling=bool(plan.runtime_handles.get("defer_sampling", False)),
            )
            if result is not None and not isinstance(result, ForwardResult):
                raise invalid_descriptor("batch path must return a ForwardResult")
            return result
        forward = getattr(self.driver, "forward_logits_graph", None)
        if not callable(forward):
            return None
        result = forward(
            dispatch,
            self.states,
            self.model,
            graph_runner=self.executor,
            defer_cpu_results=bool(plan.runtime_handles.get("defer_text_cpu_results", False)),
            defer_sampling=bool(plan.runtime_handles.get("defer_sampling", False)),
        )
        if result is None:
            return None
        expected = tuple(int(row.req_id) for row in plan.rows)
        if tuple(int(req_id) for req_id in getattr(result, "req_ids", ())) != expected:
            raise invalid_descriptor("batch graph result must align with forward rows")
        return ForwardResult(
            text_logits=result.logits,
            text_cuda_ready_start_event=getattr(result, "cuda_ready_start_event", None),
        )


class Flow(Path):
    """Run a uniform flow plan through its graph-capable executor."""

    def __init__(
        self,
        *,
        driver: Any | None = None,
        model: Any | None = None,
        states: Any | None = None,
    ) -> None:
        self.driver = driver
        self.model = model
        self.states = states

    def name(self, plan: ForwardPlan) -> str:
        del plan
        return "flow"

    def match(self, batch: ForwardBatch, plan: ForwardPlan) -> Match:
        del batch
        if plan.shape.denoise_row_count != plan.shape.row_count:
            return Match(False, "plan is not uniformly flow-shaped")
        if not self._bound:
            return Match(False, "flow executor is unavailable")
        if any(int(op.get("denoise_step_count") or 1) > 1 for op in plan.ops):
            return Match(False, "multi-step rows are not graphable")
        return Match(True)

    @property
    def _bound(self) -> bool:
        return (
            self.driver is not None
            and self.model is not None
            and self.states is not None
            and callable(getattr(self.driver, "forward_result", None))
        )

    def run(self, batch: ForwardBatch, plan: ForwardPlan) -> ForwardResult | None:
        del batch
        if not self._bound:
            return None
        forward = self.driver.forward_result
        kwargs: dict[str, Any] = {"row_indices": tuple(int(row.row_index) for row in plan.rows)}
        if _accepts(forward, "graph_mode"):
            kwargs["graph_mode"] = "require"
        items = [(int(row.req_id), self.states.get(int(row.req_id)), row.op) for row in plan.rows]
        result = forward(items, self.model, **kwargs)
        if result is not None and not isinstance(result, ForwardResult):
            raise invalid_descriptor("flow path must return a ForwardResult")
        return result


def _accepts(hook: Any, name: str) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in signature.parameters.values()
    )
