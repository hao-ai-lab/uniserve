"""Forward-step execution over parsed worker batches."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from ...contracts.batch_policy import BatchPolicy
from ...contracts.batches import ExecuteBatch, UniForwardBatch
from ...contracts.forward_context import ForwardContext, use_forward_context
from ...contracts.forward_mode import ForwardMode, mode_for_op
from ...contracts.forward_stats import ForwardStats
from ...contracts.outputs import DeferredForwardOutput, ForwardOutput, ForwardOutputBase
from ...foundation.env import env_flag
from ...foundation.errors import capability_mismatch, invalid_descriptor
from ...foundation.profiling import profile_range
from .plan import ForwardAdmissionRouter, ForwardRuntimeHandles

__all__ = [
    "ForwardGroupPlanner",
    "ForwardStepExecutor",
    "ForwardStepOptions",
]

_GRAPH_SELECTION_DELEGATED_MODES = frozenset({ForwardMode.COMMIT, ForwardMode.ENCODE})


@dataclass(frozen=True)
class ForwardStepOptions:
    defer_text_cpu_results: bool = False


class ForwardGroupPlanner:
    """Plans one parsed worker step into executable op groups."""

    def __init__(
        self,
        batch_policy: BatchPolicy,
        *,
        log_text_mixed_split: Callable[[list[Mapping[str, Any]], Any], None],
        can_run_forward: Callable[[UniForwardBatch], bool] | None = None,
    ) -> None:
        self.batch_policy = batch_policy
        self._log_text_mixed_split = log_text_mixed_split
        self._can_run_forward = can_run_forward

    def groups(self, ops: list[Mapping[str, Any]]) -> list[list[tuple[int, Mapping[str, Any]]]]:
        if self.batch_policy.supports_mixed_modes:
            decision = ForwardAdmissionRouter.from_runtime_config().decide(ops)
            if decision.use_forward:
                batch = UniForwardBatch.from_ops(ops)
                if self._can_run_forward is None or self._can_run_forward(batch):
                    return [list(enumerate(ops))]
                raise capability_mismatch(
                    "admitted mixed forward has no whole-batch executor",
                    details={
                        "modes": [mode.value for mode in decision.modes],
                        "reason": decision.reason,
                    },
                )
            self._log_text_mixed_split(ops, decision)
            return self._mode_ordered_groups(ops)
        return self._contiguous_groups(ops)

    def _contiguous_groups(
        self,
        ops: list[Mapping[str, Any]],
    ) -> list[list[tuple[int, Mapping[str, Any]]]]:
        groups: list[list[tuple[int, Mapping[str, Any]]]] = []
        for idx, op in enumerate(ops):
            mode = self._validated_mode(idx, op)
            if groups:
                modes = [mode_for_op(item[1]["kind"]) for item in groups[-1]]
                if self.batch_policy.allows_group([*modes, mode]):
                    groups[-1].append((idx, op))
                    continue
            groups.append([(idx, op)])
        return groups

    def _mode_ordered_groups(
        self,
        ops: list[Mapping[str, Any]],
    ) -> list[list[tuple[int, Mapping[str, Any]]]]:
        buckets: dict[ForwardMode, list[tuple[int, Mapping[str, Any]]]] = {}
        first_seen: list[ForwardMode] = []
        for idx, op in enumerate(ops):
            mode = self._validated_mode(idx, op)
            if mode not in buckets:
                buckets[mode] = []
                first_seen.append(mode)
            buckets[mode].append((idx, op))

        ordered_modes: list[ForwardMode] = []
        for mode in self.batch_policy.mode_order:
            if mode in buckets:
                ordered_modes.append(mode)
        for mode in first_seen:
            if mode not in ordered_modes:
                ordered_modes.append(mode)

        groups: list[list[tuple[int, Mapping[str, Any]]]] = []
        max_batch_ops = self.batch_policy.max_batch_ops
        for mode in ordered_modes:
            items = buckets[mode]
            for start in range(0, len(items), max_batch_ops):
                groups.append(items[start:start + max_batch_ops])
        return groups

    @staticmethod
    def _validated_mode(idx: int, op: Mapping[str, Any]) -> ForwardMode:
        if not isinstance(op, Mapping):
            raise invalid_descriptor(f"execute batch.ops[{idx}] must be a map")
        kind = op.get("kind")
        if not isinstance(kind, str):
            raise invalid_descriptor(f"execute batch.ops[{idx}].kind must be a string")
        return mode_for_op(kind)


class ForwardStepExecutor:
    """Owns per-step grouping, dispatch context, and result alignment."""

    def __init__(self, runner: Any, *, group_planner: ForwardGroupPlanner) -> None:
        self.runner = runner
        self.group_planner = group_planner

    def execute(self, parsed: ExecuteBatch, options: ForwardStepOptions) -> dict[str, Any]:
        forward_stats = ForwardStats() if env_flag("UNISERVE_FORWARD_METRICS") else None
        self.runner._register_new_reqs(parsed.new_reqs)
        ops = parsed.ops
        results: list[dict[str, Any] | None] = [None] * len(ops)
        with profile_range("uniserve.runner.group_ops"):
            groups = self.group_planner.groups(list(ops))
        for group in groups:
            self._run_group(
                group,
                results,
                forward_stats=forward_stats,
                defer_text_cpu_results=options.defer_text_cpu_results,
            )
        if any(result is None for result in results):
            raise invalid_descriptor("runner missed at least one op result")
        out: dict[str, Any] = {"step_id": parsed.step_id, "per_seq": results}
        if forward_stats is not None:
            out["forward_stats"] = forward_stats.to_wire()
        return out

    def _run_group(
        self,
        group: list[tuple[int, Mapping[str, Any]]],
        results: list[dict[str, Any] | None],
        *,
        forward_stats: ForwardStats | None,
        defer_text_cpu_results: bool,
    ) -> None:
        indices = [idx for idx, _ in group]
        fb = UniForwardBatch.from_ops([op for _, op in group])
        with profile_range(f"uniserve.runner.group.{fb.mode.value}"):
            self.runner._accountant.account_group(group)
            if forward_stats is not None:
                self.runner._record_group_shape(forward_stats, fb)
            ctx = ForwardContext(
                attention_backend=self.runner.attention_backend,
                attention_backend_name=self.runner.attention_backend_name,
                stats=forward_stats,
            )
            group_start = time.perf_counter_ns() if forward_stats is not None else 0
            stream_ctx = self.runner._forward_stream_context(fb)
            with torch.inference_mode(), use_forward_context(ctx), stream_ctx:
                outputs = self._execute_unified_group(
                    fb,
                    group,
                    results,
                    forward_stats=forward_stats,
                    defer_text_cpu_results=defer_text_cpu_results,
                )
            if forward_stats is not None:
                forward_stats.record_mode_wall_time(
                    fb.mode.value,
                    time.perf_counter_ns() - group_start,
                )
            if len(outputs) != len(group):
                raise invalid_descriptor(
                    f"model returned {len(outputs)} outputs for {len(group)} ops"
                )
            for idx, result in zip(indices, outputs):
                results[idx] = result

    def _execute_unified_group(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        results: list[dict[str, Any] | None],
        *,
        forward_stats: ForwardStats | None,
        defer_text_cpu_results: bool,
    ) -> list[Any]:
        indices = [idx for idx, _ in group]

        def normalize_output(output: Any) -> Any:
            return _to_seq_result(output)

        def postprocess_side_effects(group_results: list[Any]) -> None:
            staged_results = list(results)
            for idx, result in zip(indices, group_results):
                staged_results[idx] = result
            self.runner._advance_state(fb, group_results)
            self.runner._stamp_conditioning_locators(fb, group, staged_results)
            for row, idx in enumerate(indices):
                group_results[row] = staged_results[idx]

        runtime_handles = ForwardRuntimeHandles(
            request_states=self.runner.request_states,
            residency=self.runner.residency,
            tensor_store=self.runner.tensor_store,
            values={
                "dispatch_batch": fb,
                "defer_text_cpu_results": defer_text_cpu_results,
                "defer_sampling": self.runner.defer_sampling,
                "output_normalizer": normalize_output,
                "postprocess_side_effects": postprocess_side_effects,
            },
        )
        plan = self.runner.forward_plan_builder.build(
            group,
            request_states=self.runner.request_states,
            graph_policy=_graph_policy_for_group(self.runner.forward_graph_policy, fb),
            runtime_handles=runtime_handles,
        )
        device = torch.device(str(getattr(self.runner.model, "device", "cpu") or "cpu"))
        batch = self.runner.unified_forward_batch_builder.build(plan, device=device)

        with self.runner.forward_adapter.bind(
            dispatch_batch=fb,
            group=group,
            defer_text_cpu_results=defer_text_cpu_results,
        ):
            result = self.runner.forward_executor.execute(batch, plan)
        if forward_stats is not None and result.graph is not None:
            forward_stats.cuda_graph_runtime_mode_counts[result.graph.program] = (
                forward_stats.cuda_graph_runtime_mode_counts.get(result.graph.program, 0) + 1
            )
        return self.runner.forward_postprocessor.apply(plan, result)


def _to_seq_result(output: ForwardOutput | Mapping[str, Any]) -> Any:
    if isinstance(output, ForwardOutputBase):
        return output.to_seq_result()
    if isinstance(output, DeferredForwardOutput):
        return output.to_seq_result()
    if isinstance(output, Mapping):
        return dict(output)
    raise invalid_descriptor(f"unsupported forward output type {type(output).__name__}")


def _graph_policy_for_group(policy: Any, fb: UniForwardBatch) -> Any:
    if getattr(fb, "mode", None) not in _GRAPH_SELECTION_DELEGATED_MODES:
        return policy
    if policy is None:
        return None
    return replace(policy, graph_selection_delegated=True)
