"""Graph program interfaces for unified forward execution."""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from ....contracts.batches import UniForwardBatch
from ....contracts.forward_batch import ForwardBatch
from ....contracts.forward_mode import ForwardMode
from ....foundation.errors import invalid_descriptor
from ..plan import ForwardPlan
from ..result import ForwardResult
from .key import ForwardGraphShapeKey, graph_shape_key

__all__ = [
    "CapturedForwardGraph",
    "DecodeGraphProgram",
    "DenoiseStepGraphProgram",
    "ForwardGraphProgram",
    "GraphEligibility",
    "ModelOwnedTextGraphProgram",
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
    replay_fn: Callable[[ForwardBatch], ForwardResult] | None = None


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
    ) -> CapturedForwardGraph | None:
        del plan
        return CapturedForwardGraph(
            key=key,
            program=self,
            payload=forward_fn(batch),
            replay_fn=forward_fn,
        )

    def replay(
        self,
        graph: CapturedForwardGraph,
        batch: ForwardBatch,
        plan: ForwardPlan,
    ) -> ForwardResult | None:
        del plan
        if graph.replay_fn is not None:
            return graph.replay_fn(batch)
        if not isinstance(graph.payload, ForwardResult):
            raise TypeError("captured forward graph payload must be a ForwardResult")
        return graph.payload


class _TextGraphProgramMixin(ForwardGraphProgram):
    def __init__(
        self,
        *,
        text_driver: Any | None = None,
        model: Any | None = None,
        request_states: Any | None = None,
        text_graph_runner: Any | None = None,
    ) -> None:
        self.text_driver = text_driver
        self.model = model
        self.request_states = request_states
        self.text_graph_runner = text_graph_runner

    @property
    def _bound_text_graph(self) -> bool:
        return (
            self.text_driver is not None
            and self.model is not None
            and self.request_states is not None
            and self.text_graph_runner is not None
        )

    def _text_graph_eligible(self, plan: ForwardPlan) -> GraphEligibility:
        if not self._bound_text_graph:
            return GraphEligibility(True)
        for row in plan.rows:
            op = row.op
            if op.get("spec_token_ids"):
                return GraphEligibility(False, "speculative text rows are not graphable")
            if row.mode is ForwardMode.DECODE and int(op.get("decode_token_count") or 1) > 1:
                forward_graph_result = getattr(self.text_driver, "forward_graph_result", None)
                if not callable(forward_graph_result):
                    return GraphEligibility(False, "decode burst rows are not graphable")
        return GraphEligibility(True)

    def _run_text_graph(self, plan: ForwardPlan) -> ForwardResult | None:
        if not self._bound_text_graph:
            return None
        forward_graph_result = getattr(self.text_driver, "forward_graph_result", None)
        if callable(forward_graph_result):
            dispatch_batch = plan.runtime_handles.get("dispatch_batch")
            if not isinstance(dispatch_batch, UniForwardBatch):
                dispatch_batch = UniForwardBatch.from_ops(plan.ops)
            result = forward_graph_result(
                dispatch_batch,
                self.request_states,
                self.model,
                graph_runner=self.text_graph_runner,
                defer_cpu_results=bool(plan.runtime_handles.get("defer_text_cpu_results", False)),
                defer_sampling=bool(plan.runtime_handles.get("defer_sampling", False)),
            )
            if result is not None and not isinstance(result, ForwardResult):
                raise invalid_descriptor("text graph result must be a ForwardResult")
            return result
        forward_logits_graph = getattr(self.text_driver, "forward_logits_graph", None)
        if not callable(forward_logits_graph):
            return None
        dispatch_batch = plan.runtime_handles.get("dispatch_batch")
        if not isinstance(dispatch_batch, UniForwardBatch):
            dispatch_batch = UniForwardBatch.from_ops(plan.ops)
        text_result = forward_logits_graph(
            dispatch_batch,
            self.request_states,
            self.model,
            graph_runner=self.text_graph_runner,
            defer_cpu_results=bool(plan.runtime_handles.get("defer_text_cpu_results", False)),
            defer_sampling=bool(plan.runtime_handles.get("defer_sampling", False)),
        )
        if text_result is None:
            return None
        expected_req_ids = tuple(int(row.req_id) for row in plan.rows)
        req_ids = tuple(int(req_id) for req_id in getattr(text_result, "req_ids", ()))
        if req_ids != expected_req_ids:
            raise invalid_descriptor("text graph result req_ids must align with forward rows")
        return ForwardResult(
            text_logits=text_result.logits,
            text_cuda_ready_start_event=getattr(text_result, "cuda_ready_start_event", None),
        )

    def capture(
        self,
        key: ForwardGraphShapeKey,
        batch: ForwardBatch,
        plan: ForwardPlan,
        forward_fn: Callable[[ForwardBatch], ForwardResult],
    ) -> CapturedForwardGraph | None:
        if not self._bound_text_graph:
            return super().capture(key, batch, plan, forward_fn)
        result = self._run_text_graph(plan)
        if result is None:
            return None
        return CapturedForwardGraph(key=key, program=self, payload=result)

    def replay(
        self,
        graph: CapturedForwardGraph,
        batch: ForwardBatch,
        plan: ForwardPlan,
    ) -> ForwardResult | None:
        if not self._bound_text_graph:
            return super().replay(graph, batch, plan)
        return self._run_text_graph(plan)


class DecodeGraphProgram(_TextGraphProgramMixin):
    program_id = "decode"

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count == plan.shape.row_count and all(
            segment.segment_class.value == "decode" for segment in plan.segments
        ):
            return self._text_graph_eligible(plan)
        return GraphEligibility(False, "not a pure decode shape")


class PrefillGraphProgram(_TextGraphProgramMixin):
    program_id = "prefill"

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count == plan.shape.row_count and plan.shape.token_count > 0:
            return self._text_graph_eligible(plan)
        return GraphEligibility(False, "not a text prefill shape")


class ModelOwnedTextGraphProgram(_TextGraphProgramMixin):
    program_id = "model_owned_text"

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count != plan.shape.row_count:
            return GraphEligibility(False, "not a pure model-owned text shape")
        if not self._bound_model_owned_text_graph:
            return GraphEligibility(False, "model-owned text graph program is not bound")
        return self._text_graph_eligible(plan)

    @property
    def _bound_model_owned_text_graph(self) -> bool:
        return (
            self.text_driver is not None
            and self.model is not None
            and self.request_states is not None
            and callable(getattr(self.model, "try_run_text_graph_logits_batch", None))
        )

    @property
    def _bound_text_graph(self) -> bool:
        return self._bound_model_owned_text_graph


class PackedVisibleGraphProgram(ForwardGraphProgram):
    program_id = "packed_visible"

    def __init__(
        self,
        *,
        owner: Any | None = None,
        request_states: Any | None = None,
        image_decode_driver: Any | None = None,
    ) -> None:
        self.owner = owner
        self.request_states = request_states
        self.image_decode_driver = image_decode_driver

    @property
    def _bound_packed_visible(self) -> bool:
        return (
            self.owner is not None
            and self.request_states is not None
            and callable(getattr(self.owner, "prepare_denoise", None))
            and callable(getattr(self.owner, "packed_decoder_forward", None))
            and callable(getattr(self.owner, "packed_text_embeddings", None))
            and callable(getattr(self.owner, "packed_text_logits", None))
            and callable(getattr(self.owner, "packed_graph_attention", None))
        )

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count and (
            plan.shape.denoise_row_count or plan.shape.commit_row_count
        ):
            if not self._bound_packed_visible:
                if self.owner is None and self.request_states is None:
                    return GraphEligibility(True)
                return GraphEligibility(False, "packed visible graph program is not bound")
            if plan.shape.denoise_row_count and not callable(
                getattr(self.owner, "packed_hidden_to_velocity", None)
            ):
                return GraphEligibility(False, "packed denoise projection is not bound")
            if self._has_burst_rows(plan) and not callable(
                getattr(self.owner, "_run_forward_adapter", None)
            ):
                return GraphEligibility(False, "packed burst graph adapter is not bound")
            if plan.shape.commit_row_count and self.image_decode_driver is None:
                return GraphEligibility(False, "packed visible commit publication is not bound")
            return GraphEligibility(True)
        return GraphEligibility(False, "not a packed visible generation shape")

    def capture(
        self,
        key: ForwardGraphShapeKey,
        batch: ForwardBatch,
        plan: ForwardPlan,
        forward_fn: Callable[[ForwardBatch], ForwardResult],
    ) -> CapturedForwardGraph | None:
        del batch, forward_fn
        result = self._run_packed_visible_graph(plan)
        if result is None:
            return None
        return CapturedForwardGraph(key=key, program=self, payload=result)

    def replay(
        self,
        graph: CapturedForwardGraph,
        batch: ForwardBatch,
        plan: ForwardPlan,
    ) -> ForwardResult | None:
        del graph, batch
        return self._run_packed_visible_graph(plan)

    def _run_packed_visible_graph(self, plan: ForwardPlan) -> ForwardResult | None:
        owner = self.owner
        request_states = self.request_states
        if not self._bound_packed_visible or owner is None or request_states is None:
            return None
        from ..programs.packed_visible import run_packed_visible_forward_result

        dispatch_batch = plan.runtime_handles.get("dispatch_batch")
        if not isinstance(dispatch_batch, UniForwardBatch):
            dispatch_batch = UniForwardBatch.from_ops(plan.ops)
        if self._has_burst_rows(plan):
            run = getattr(owner, "_run_forward_adapter", None)
            if not callable(run):
                return None
            outputs = run(
                dispatch_batch,
                request_states=request_states,
                group=list(enumerate(dispatch_batch.ops)),
                defer_text_cpu_results=bool(
                    plan.runtime_handles.get("defer_text_cpu_results", False)
                ),
            )
            if isinstance(outputs, ForwardResult):
                return outputs
            if not isinstance(outputs, Sequence) or isinstance(
                outputs, (str, bytes, bytearray)
            ):
                raise invalid_descriptor(
                    "packed burst graph adapter must return one result per forward row"
                )
            if len(outputs) != len(plan.rows):
                raise invalid_descriptor(
                    "packed burst graph adapter returned the wrong number of results"
                )
            return ForwardResult(runtime_outputs=tuple(outputs))
        denoise_steps = []
        for row in plan.rows:
            if row.mode is not ForwardMode.DENOISE:
                continue
            state = request_states.get(int(row.req_id))
            step = owner.prepare_denoise(state, dict(row.op))
            denoise_steps.append((int(row.row_index), step))
            extra = getattr(step, "extra", None)
            img = extra.get("img") if isinstance(extra, dict) else None
            residual_state = getattr(img, "residual_cache", None)
            if residual_state is not None:
                residual_state.invalidate()
        result = run_packed_visible_forward_result(
            owner,
            dispatch_batch,
            request_states,
            denoise_steps,
            defer_text_cpu_results=bool(plan.runtime_handles.get("defer_text_cpu_results", False)),
            allow_graph=True,
            require_graph=True,
        )
        if result is None or not plan.shape.commit_row_count:
            return result
        image_decode_driver = self.image_decode_driver
        if image_decode_driver is None:
            raise invalid_descriptor("packed visible commit publication is not bound")
        commit_rows = tuple(row for row in plan.rows if row.mode is ForwardMode.COMMIT)
        commit_result = image_decode_driver.forward_result(
            tuple(
                (int(row.req_id), request_states.get(int(row.req_id)), row.op)
                for row in commit_rows
            ),
            owner,
            row_indices=tuple(int(row.row_index) for row in commit_rows),
        )
        if not isinstance(commit_result, ForwardResult) or commit_result.commit_outputs is None:
            raise invalid_descriptor("packed visible commit publication returned no commit outputs")
        commit_outputs = dict(result.commit_outputs or {})
        commit_outputs.update(commit_result.commit_outputs)
        result.commit_outputs = commit_outputs
        return result

    @staticmethod
    def _has_burst_rows(plan: ForwardPlan) -> bool:
        return any(
            int(op.get("decode_token_count") or 1) > 1
            or int(op.get("denoise_step_count") or 1) > 1
            for op in plan.ops
        )


class DenoiseStepGraphProgram(ForwardGraphProgram):
    program_id = "denoise_step"

    def __init__(
        self,
        *,
        denoise_driver: Any | None = None,
        model: Any | None = None,
        request_states: Any | None = None,
    ) -> None:
        self.denoise_driver = denoise_driver
        self.model = model
        self.request_states = request_states

    @property
    def _bound_denoise_graph(self) -> bool:
        return (
            self.denoise_driver is not None
            and self.model is not None
            and self.request_states is not None
            and callable(getattr(self.denoise_driver, "forward_result", None))
        )

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.denoise_row_count == plan.shape.row_count:
            if not self._bound_denoise_graph:
                if (
                    self.denoise_driver is None
                    and self.model is None
                    and self.request_states is None
                ):
                    return GraphEligibility(True)
                return GraphEligibility(False, "denoise graph program is not bound")
            for op in plan.ops:
                if int(op.get("denoise_step_count") or 1) > 1:
                    return GraphEligibility(False, "denoise burst rows are not graphable")
            return GraphEligibility(True)
        return GraphEligibility(False, "not a pure denoise shape")

    def capture(
        self,
        key: ForwardGraphShapeKey,
        batch: ForwardBatch,
        plan: ForwardPlan,
        forward_fn: Callable[[ForwardBatch], ForwardResult],
    ) -> CapturedForwardGraph | None:
        del batch, forward_fn
        result = self._run_denoise_graph(plan)
        if result is None:
            return None
        return CapturedForwardGraph(key=key, program=self, payload=result)

    def replay(
        self,
        graph: CapturedForwardGraph,
        batch: ForwardBatch,
        plan: ForwardPlan,
    ) -> ForwardResult | None:
        del graph, batch
        return self._run_denoise_graph(plan)

    def _run_denoise_graph(self, plan: ForwardPlan) -> ForwardResult | None:
        denoise_driver = self.denoise_driver
        model = self.model
        request_states = self.request_states
        if (
            not self._bound_denoise_graph
            or denoise_driver is None
            or model is None
            or request_states is None
        ):
            return None
        forward_result = denoise_driver.forward_result
        kwargs: dict[str, Any] = {"row_indices": tuple(int(row.row_index) for row in plan.rows)}
        if _accepts_keyword(forward_result, "graph_mode"):
            kwargs["graph_mode"] = "require"
        items = [
            (int(row.req_id), request_states.get(int(row.req_id)), row.op)
            for row in plan.rows
        ]
        return forward_result(items, model, **kwargs)


def _accepts_keyword(hook: Any, name: str) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters
