"""Model forward preparation, execution, and result projection.

The runner owns model-facing row plans, device batch construction, graph/eager
execution, and postprocessing for sequence, flow, encode, and materialize
operations. Transaction lifecycle and schema lowering live in their dedicated
torch-free modules; graph capture and replay live in ``execution.graph``.
"""

from __future__ import annotations

import base64
import logging
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import torch

from uniserve_worker.backends.attention import (
    get_attention_backend,
    normalize_attention_backend_name,
)
from uniserve_worker.contracts.batch_policy import BatchPolicy
from uniserve_worker.contracts.batches import ExecuteBatch as WireExecuteBatch
from uniserve_worker.contracts.batches import TextBatch
from uniserve_worker.contracts.caps import Caps
from uniserve_worker.contracts.forward_batch import (
    BranchSpec,
    CacheSpanPlan,
    CfgPlan,
    CommitInputs,
    CommitRowPlan,
    DenoiseBranchKey,
    DenoiseInputs,
    DenoiseRowPlan,
    EncodeInputs,
    EncodeRowPlan,
    ForwardBatch,
    ForwardExecutionOptions,
    ForwardGraphPolicy,
    ForwardModality,
    ForwardOutputKind,
    ForwardOutputSlot,
    ForwardPlan,
    ForwardPostprocessPolicy,
    ForwardResult,
    ForwardResultProjection,
    ForwardRowPlan,
    ForwardSegmentClass,
    ForwardSegmentPlan,
    ForwardShapeSummary,
    KvSource,
    KvWritePolicy,
    SegmentSpec,
    TextPostprocessEntry,
    TextTokenSpanPlan,
    VisiblePolicy,
    coerce_forward_result,
)
from uniserve_worker.contracts.forward_context import (
    ForwardContext,
    component_timer_start,
    get_forward_context,
    record_component_elapsed,
    use_forward_context,
)
from uniserve_worker.contracts.forward_mode import ForwardMode, mode_for_op
from uniserve_worker.contracts.forward_stats import ForwardStats
from uniserve_worker.contracts.model_protocols import ModelHooks, UniModel
from uniserve_worker.contracts.outputs import (
    CommitOutput,
    DeferredForwardOutput,
    EncodeOutput,
    FlowOutput,
    ForwardOutput,
    ForwardOutputBase,
    TextTokenOutput,
)
from uniserve_worker.contracts.resource_plan import LatentTokens, ResourcePlan
from uniserve_worker.foundation.env import env_flag
from uniserve_worker.foundation.errors import capability_mismatch, invalid_descriptor
from uniserve_worker.foundation.runtime_config import get_execution_config
from uniserve_worker.nn.diffusion import (
    euler_step,
)
from uniserve_worker.nn.sampler import (
    apply_sampling_batched_with_device_tokens,
    finalize_sampling_result,
    is_deferred_sampling_result,
)
from uniserve_worker.runtime.forward_batch_builder import ForwardBatchBuilder
from uniserve_worker.runtime.paged_text_cache import copy_paged_text_cache_spans
from uniserve_worker.runtime.request_session import RequestSessionTable
from uniserve_worker.runtime.residency_manager import ResidencyLeaseManager
from uniserve_worker.runtime.resources import ResourceRuntime

from .autoregressive import (
    _DECODE_RELAY,
    DeferredDecodeBurstSeqResult,
    DeferredTerminalDecodeBurstSeqResult,
    DeferredTextSeqResult,
    TextDecodeRelay,
    _AutoregressiveRuntime,
    sample_logits_result,
    text_input_id_replacements_from_relays,
)
from .codec import (
    _coerce_encode_output,
    _commit_output_from_dict,
    _image_to_result,
    commit_result,
    encode_result,
)
from .diffusion import (
    _DiffusionRuntime,
)
from .flow import (
    PreparedFlowStep,
    combine_flow_velocity,
    flow_branches,
    flow_cfg_branch_count,
    flow_cfg_plan,
)

if TYPE_CHECKING:
    from uniserve_worker.contracts.forward_batch import ForwardBatch, ForwardGraphPolicy
    from uniserve_worker.contracts.forward_stats import ForwardStats
    from uniserve_worker.runtime.residency import ResidencyManager


__all__ = [
    "DeferredDecodeBurstSeqResult",
    "DeferredTerminalDecodeBurstSeqResult",
    "DeferredTextSeqResult",
    "ModelRunner",
    "PreparedFlowStep",
    "RunnerConfig",
    "TextDecodeRelay",
    "combine_flow_velocity",
    "flow_branches",
    "flow_cfg_branch_count",
    "flow_cfg_plan",
    "text_input_id_replacements_from_relays",
]


def _execute_required_denoise(items: Any, model: Any) -> list[FlowOutput]:
    return _DiffusionRuntime().step_many(items, model, graph_mode="require")


# ---------------------
# Canonical plan construction (rows, segments, output slots)
# ---------------------

_TEXT_MODES = frozenset({ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT})


class ForwardPlanBuilder:
    """Build immutable control plans from admitted worker op groups."""

    def build(
        self,
        group: Sequence[Mapping[str, Any] | tuple[int, Mapping[str, Any]]],
        *,
        request_states: Any = None,
        step_id: int | None = None,
        graph_policy: "ForwardGraphPolicy | None" = None,
    ) -> ForwardPlan:
        if not group:
            raise invalid_descriptor("forward plan group must not be empty")
        rows: list[ForwardRowPlan] = []
        segments: list[ForwardSegmentPlan] = []
        output_slots: list[ForwardOutputSlot] = []
        for row_index, item in enumerate(group):
            original_index, op = _group_item(row_index, item)
            row = self._row_plan(row_index, original_index, op, request_states)
            rows.append(row)
            segments.extend(self._segments_for_row(row, len(segments)))
            output_slots.append(self._output_slot(row))
        forward_mode = _summary_mode(tuple(row.mode for row in rows))
        shape = ForwardShapeSummary.from_parts(
            forward_mode=forward_mode,
            rows=rows,
            segments=segments,
        )
        plan = ForwardPlan(
            step_id=step_id,
            rows=tuple(rows),
            segments=tuple(segments),
            output_slots=tuple(output_slots),
            shape=shape,
            graph_policy=graph_policy,
        )
        plan.validate()
        return plan

    def _row_plan(
        self,
        row_index: int,
        original_index: int,
        op: Mapping[str, Any],
        request_states: Any,
    ) -> ForwardRowPlan:
        kind = op.get("kind")
        if not isinstance(kind, str):
            raise invalid_descriptor("forward op kind must be a string")
        req_id = _int_field(op, "req_id")
        mode = mode_for_op(kind)
        token_span = self._text_span(op, mode)
        cache_span = self._cache_span(op, request_states, req_id, token_span)
        return ForwardRowPlan(
            row_index=row_index,
            original_index=original_index,
            req_id=req_id,
            op=MappingProxyType(dict(op)),
            mode=mode,
            token_span=token_span,
            cache_span=cache_span,
            denoise=self._denoise_plan(op, mode),
            commit=self._commit_plan(op, mode),
            encode=self._encode_plan(op, mode),
        )

    @staticmethod
    def _text_span(op: Mapping[str, Any], mode: ForwardMode) -> TextTokenSpanPlan | None:
        if mode not in _TEXT_MODES:
            return None
        tokens = tuple(int(token) for token in (op.get("token_ids") or ()))
        start, end = _pos_range(op, len(tokens))
        return TextTokenSpanPlan(
            token_ids=tokens,
            position_start=start,
            position_end=end,
            token_source=str(op.get("token_source") or "wire"),
            last_token_only=mode
            in {ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.VERIFY_DRAFT},
        )

    @staticmethod
    def _cache_span(
        op: Mapping[str, Any],
        request_states: Any,
        req_id: int,
        token_span: TextTokenSpanPlan | None,
    ) -> CacheSpanPlan | None:
        if token_span is None:
            return None
        state = None
        if request_states is not None:
            get = getattr(request_states, "get", None)
            if callable(get):
                try:
                    state = get(req_id)
                except Exception:
                    state = None
        state_blocks = tuple(int(block) for block in getattr(state, "block_ids", ()) or ())
        new_blocks = tuple(int(block) for block in (op.get("new_block_ids") or ()))
        block_ids = state_blocks
        if new_blocks and not block_ids[-len(new_blocks) :] == new_blocks:
            block_ids = (*block_ids, *new_blocks)
        return CacheSpanPlan(
            block_ids=block_ids,
            base_len=int(token_span.position_start),
            append_len=token_span.q_len,
            pool_identity=str(op.get("kv_pool") or "text"),
            persistent=True,
        )

    @staticmethod
    def _denoise_plan(op: Mapping[str, Any], mode: ForwardMode) -> DenoiseRowPlan | None:
        if mode is not ForwardMode.DENOISE:
            return None
        cfg = dict(op.get("cfg") or {})
        branch_count = int(cfg.get("branch_count") or op.get("branch_count") or 1)
        if branch_count < 1:
            raise invalid_descriptor("denoise branch count must be positive")
        return DenoiseRowPlan(
            step_index=int(op.get("timestep_idx") or 0),
            total_steps=max(1, int(op.get("num_steps") or op.get("total_steps") or 1)),
            branch_count=branch_count,
            branch_ids=tuple(_plan_branch_name(i, branch_count) for i in range(branch_count)),
            image_token_count=max(1, _image_token_count(op)),
            latent_handle=_plan_optional_int(op.get("latent_handle")),
            grid_hw=_grid_hw(op.get("grid_hw")),
            cfg=MappingProxyType(cfg),
        )

    @staticmethod
    def _commit_plan(op: Mapping[str, Any], mode: ForwardMode) -> CommitRowPlan | None:
        if mode is not ForwardMode.COMMIT:
            return None
        return CommitRowPlan(
            latent_handle=_plan_optional_int(op.get("latent_handle")),
            fold_back=bool(op.get("fold_back", False)),
            image_token_count=max(1, _image_token_count(op)),
        )

    @staticmethod
    def _encode_plan(op: Mapping[str, Any], mode: ForwardMode) -> EncodeRowPlan | None:
        if mode is not ForwardMode.ENCODE:
            return None
        return EncodeRowPlan(
            kind=str(op.get("kind")),
            out_handle=_plan_optional_int(op.get("out_handle") or op.get("encoder_handle")),
            mm_hash=_plan_optional_int(op.get("mm_hash")),
            num_tokens=max(1, int(op.get("num_tokens") or op.get("image_token_count") or 1)),
        )

    @staticmethod
    def _segments_for_row(
        row: ForwardRowPlan,
        next_segment_index: int,
    ) -> list[ForwardSegmentPlan]:
        segments: list[ForwardSegmentPlan] = []
        if row.token_span is not None and row.token_span.q_len > 0:
            segment_class = (
                ForwardSegmentClass.DECODE
                if row.mode is ForwardMode.DECODE
                else ForwardSegmentClass.EXTEND
            )
            segments.append(
                ForwardSegmentPlan(
                    segment_index=next_segment_index,
                    row_index=row.row_index,
                    mode=row.mode,
                    modality=ForwardModality.TEXT,
                    segment_class=segment_class,
                    q_len=row.token_span.q_len,
                    prefix_len=row.token_span.position_start,
                    visible_policy=VisiblePolicy.CAUSAL,
                    branch_id=0,
                    position_source=row.token_span.token_source,
                    kv_write_policy=KvWritePolicy.PERSISTENT,
                )
            )
            next_segment_index += 1
        if row.denoise is not None:
            for branch_index in range(row.denoise.branch_count):
                segments.append(
                    ForwardSegmentPlan(
                        segment_index=next_segment_index,
                        row_index=row.row_index,
                        mode=row.mode,
                        modality=ForwardModality.GENERATION,
                        segment_class=ForwardSegmentClass.DENOISE,
                        q_len=row.denoise.image_token_count,
                        prefix_len=0,
                        visible_policy=VisiblePolicy.BIDIRECTIONAL,
                        branch_id=branch_index,
                        position_source="generation_grid",
                        kv_write_policy=KvWritePolicy.TRANSIENT,
                    )
                )
                next_segment_index += 1
        if row.commit is not None:
            segments.append(
                ForwardSegmentPlan(
                    segment_index=next_segment_index,
                    row_index=row.row_index,
                    mode=row.mode,
                    modality=ForwardModality.GENERATION,
                    segment_class=ForwardSegmentClass.COMMIT_INPUT,
                    q_len=row.commit.image_token_count,
                    visible_policy=VisiblePolicy.BIDIRECTIONAL,
                    kv_write_policy=KvWritePolicy.NONE,
                )
            )
            next_segment_index += 1
        if row.encode is not None:
            segments.append(
                ForwardSegmentPlan(
                    segment_index=next_segment_index,
                    row_index=row.row_index,
                    mode=row.mode,
                    modality=ForwardModality.GENERATION,
                    segment_class=ForwardSegmentClass.ENCODE_INPUT,
                    q_len=row.encode.num_tokens,
                    visible_policy=VisiblePolicy.BIDIRECTIONAL,
                    kv_write_policy=KvWritePolicy.NONE,
                )
            )
        return segments

    @staticmethod
    def _output_slot(row: ForwardRowPlan) -> ForwardOutputSlot:
        if row.mode in _TEXT_MODES:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.TEXT_TOKEN,
                result_projection=ForwardResultProjection.LAST_TEXT_ROW,
                postprocess_policy=ForwardPostprocessPolicy.SAMPLE,
            )
        if row.mode is ForwardMode.DENOISE:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.DENOISE_STEP,
                result_projection=ForwardResultProjection.DENOISE_BRANCHES,
                postprocess_policy=ForwardPostprocessPolicy.LATENT_UPDATE,
            )
        if row.mode is ForwardMode.COMMIT:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.COMMIT,
                result_projection=ForwardResultProjection.COMMIT_ROW,
                postprocess_policy=ForwardPostprocessPolicy.COMMIT_DECODE,
            )
        if row.mode is ForwardMode.ENCODE:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.ENCODE,
                result_projection=ForwardResultProjection.ENCODE_ROW,
                postprocess_policy=ForwardPostprocessPolicy.ENCODE_PUBLISH,
            )
        return ForwardOutputSlot(
            row_index=row.row_index,
            req_id=row.req_id,
            kind=ForwardOutputKind.COMBINED,
            result_projection=ForwardResultProjection.RUNTIME_OUTPUT,
            postprocess_policy=ForwardPostprocessPolicy.NONE,
        )


def _summary_mode(modes: tuple[ForwardMode, ...]) -> ForwardMode:
    first = modes[0]
    if any(mode is not first for mode in modes):
        return ForwardMode.MIXED
    return first


def _group_item(
    fallback_index: int,
    item: Mapping[str, Any] | tuple[int, Mapping[str, Any]],
) -> tuple[int, Mapping[str, Any]]:
    if isinstance(item, tuple) and len(item) == 2:
        index, op = item
        if not isinstance(op, Mapping):
            raise invalid_descriptor("forward group item op must be a mapping")
        return int(index), op
    if not isinstance(item, Mapping):
        raise invalid_descriptor("forward group item must be an op mapping")
    return fallback_index, item


def _int_field(op: Mapping[str, Any], field_name: str) -> int:
    value = op.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_descriptor(f"forward op {field_name} must be an integer")
    return int(value)


def _plan_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _pos_range(op: Mapping[str, Any], token_count: int) -> tuple[int, int]:
    raw = op.get("pos_range")
    if raw is None:
        return 0, int(token_count)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise invalid_descriptor("text op pos_range must be [start, end]")
    start, end = int(raw[0]), int(raw[1])
    if end < start:
        raise invalid_descriptor("text op pos_range end must be >= start")
    return start, end


def _image_token_count(op: Mapping[str, Any]) -> int:
    for key in ("image_token_count", "latent_tokens", "num_tokens"):
        if op.get(key) is not None:
            return int(op[key])
    shape = op.get("latent_shape")
    if isinstance(shape, (list, tuple)) and shape:
        total = 1
        for value in shape:
            total *= max(1, int(value))
        return total
    return 1


def _grid_hw(raw: Any) -> tuple[int, int]:
    if raw is None:
        return (0, 0)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise invalid_descriptor("grid_hw must be [height, width]")
    return (int(raw[0]), int(raw[1]))


def _plan_branch_name(index: int, count: int) -> str:
    if count == 1:
        return "cond"
    if index == 0:
        return "cond"
    return f"branch_{index}"


logger = logging.getLogger(__name__)


# ---------------------
# Device batch construction
# ---------------------


class UnifiedForwardBatchBuilder:
    """Build the single device snapshot consumed by the executor."""

    def __init__(
        self,
        *,
        runtime_builder: Any | None = None,
        kv_pool: Any | None = None,
        request_states: Any | None = None,
        default_device: torch.device | str = "cpu",
    ) -> None:
        self.runtime_builder = runtime_builder
        self.kv_pool = kv_pool
        self.request_states = request_states
        self.default_device = torch.device(default_device)

    def build(
        self,
        plan: ForwardPlan,
        *,
        device: torch.device | str | None = None,
    ) -> ForwardBatch:
        plan.validate()
        target_device = torch.device(device) if device is not None else self.default_device
        runtime_batch = self._try_runtime_text_batch(plan, target_device)
        if runtime_batch is not None:
            return runtime_batch
        return self._build_generic(plan, target_device)

    def _try_runtime_text_batch(
        self,
        plan: ForwardPlan,
        device: torch.device,
    ) -> ForwardBatch | None:
        if self.runtime_builder is None or self.kv_pool is None or self.request_states is None:
            return None
        if not plan.rows or any(row.mode not in _TEXT_MODES for row in plan.rows):
            return None
        text = TextBatch.from_ops(
            plan.forward_mode,
            plan.ops,
            op_modes=plan.op_modes,
            allow_mixed_text=plan.forward_mode is ForwardMode.MIXED,
        )
        runtime_batch = self.runtime_builder.build_text(
            text,
            device=device,
            kv_pool=self.kv_pool,
            request_states=self.request_states,
        )
        runtime_batch.op_modes = plan.op_modes
        return runtime_batch

    def _build_generic(self, plan: ForwardPlan, device: torch.device) -> ForwardBatch:
        input_ids: list[int] = []
        positions: list[int] = []
        is_gen: list[bool] = []
        last_token_indices: list[int] = []
        token_offset = 0
        for row in plan.rows:
            if row.token_span is None:
                continue
            for offset, token in enumerate(row.token_span.token_ids):
                input_ids.append(int(token))
                positions.append(int(row.token_span.position_start + offset))
                is_gen.append(False)
            if row.token_span.q_len:
                token_offset += row.token_span.q_len
                last_token_indices.append(token_offset - 1)
        padded_tokens = max(plan.shape.padded_token_count, len(input_ids))
        if padded_tokens > len(input_ids):
            pad = padded_tokens - len(input_ids)
            input_ids.extend([0] * pad)
            positions.extend([0] * pad)
            is_gen.extend([True] * pad)
        input_tensor = (
            torch.tensor(input_ids, dtype=torch.long, device=device) if input_ids else None
        )
        position_tensor = (
            torch.tensor(positions, dtype=torch.long, device=device) if positions else None
        )
        is_gen_tensor = torch.tensor(is_gen, dtype=torch.bool, device=device) if is_gen else None
        last_token_tensor = (
            torch.tensor(last_token_indices, dtype=torch.long, device=device)
            if last_token_indices
            else None
        )
        return ForwardBatch(
            forward_mode=plan.forward_mode,
            req_ids=plan.req_ids,
            op_modes=plan.op_modes,
            ops=plan.ops,
            device=device,
            input_ids=input_tensor,
            positions=position_tensor,
            last_token_indices=last_token_tensor,
            num_token_non_padded=sum(
                row.token_span.q_len for row in plan.rows if row.token_span is not None
            ),
            padded_num_tokens=padded_tokens,
            is_gen=is_gen_tensor,
            segments=_segment_specs(plan),
            denoise=_denoise_inputs(plan, device),
            encode=_encode_inputs(plan),
            commit=_commit_inputs(plan),
            sampling=None,
        )


def _segment_specs(plan: ForwardPlan) -> tuple[SegmentSpec, ...]:
    specs: list[SegmentSpec] = []
    start = 0
    for segment in plan.segments:
        specs.append(
            SegmentSpec(
                start=start,
                length=int(segment.q_len),
                visible_policy=segment.visible_policy,
                branch_id=int(segment.branch_id),
                is_gen=segment.modality is ForwardModality.GENERATION,
            )
        )
        start += int(segment.q_len)
    return tuple(specs)


def _denoise_inputs(plan: ForwardPlan, device: torch.device) -> DenoiseInputs | None:
    row = next((row for row in plan.rows if row.denoise is not None), None)
    if row is None or row.denoise is None:
        return None
    branches = tuple(
        BranchSpec(
            name=name,
            kv_source=KvSource.SCRATCH,
            kv_handle=0,
            kv_len=row.denoise.image_token_count,
            position=branch_index,
        )
        for branch_index, name in enumerate(row.denoise.branch_ids)
    )
    cfg = CfgPlan(branches=branches)
    return DenoiseInputs(
        latent_handle=int(row.denoise.latent_handle or 0),
        step_index=row.denoise.step_index,
        total_steps=row.denoise.total_steps,
        t=_scalar_tensor(row.op.get("t"), device),
        t_next=_scalar_tensor(row.op.get("t_next"), device),
        grid_hw=row.denoise.grid_hw,
        branches=branches,
        cfg=cfg,
        rng_handle=int(row.op.get("rng_handle") or 0),
    )


def _encode_inputs(plan: ForwardPlan) -> EncodeInputs | None:
    row = next((row for row in plan.rows if row.encode is not None), None)
    if row is None or row.encode is None:
        return None
    return EncodeInputs(
        kind=row.encode.kind,
        out_handle=int(row.encode.out_handle or 0),
        mm_hash=row.encode.mm_hash,
        cond_pos=int(row.op.get("cond_pos") or 0),
    )


def _commit_inputs(plan: ForwardPlan) -> CommitInputs | None:
    row = next((row for row in plan.rows if row.commit is not None), None)
    if row is None or row.commit is None:
        return None
    return CommitInputs(
        latent_handle=int(row.commit.latent_handle or 0),
        fold_back=bool(row.commit.fold_back),
    )


def _scalar_tensor(value: Any, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    try:
        return torch.tensor([float(value)], dtype=torch.float32, device=device)
    except (TypeError, ValueError) as exc:
        raise invalid_descriptor("denoise scalar tensors must be numeric") from exc


# ---------------------
# Result projection and postprocess
# ---------------------


class ForwardPostprocessor:
    """Project neural results using explicitly bound runtime services."""

    def __init__(self, *, request_states: Any = None, tensor_store: Any = None) -> None:
        self.request_states = request_states
        self.tensor_store = tensor_store

    def apply(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        result: ForwardResult,
        options: ForwardExecutionOptions = ForwardExecutionOptions(),
    ) -> list[Any]:
        result.validate_for_plan(plan)
        if result.runtime_outputs is not None:
            return self._normalize_outputs(result.runtime_outputs)
        if self._can_apply_text_batch(plan, result):
            return self._normalize_outputs(self._apply_text_batch(batch, plan, result, options))
        text_outputs = (
            self._apply_text_entries(plan, result, options) if result.text_postprocess else {}
        )
        outputs: list[Any] = []
        text_row = 0
        for slot in plan.output_slots:
            row = plan.rows[slot.row_index]
            if slot.kind is ForwardOutputKind.TEXT_TOKEN:
                if slot.row_index in text_outputs:
                    outputs.append(text_outputs[slot.row_index])
                    continue
                if result.text_logits is None:
                    raise invalid_descriptor("text output slot requires logits")
                logits_rows = result.text_logits.reshape(-1, result.text_logits.shape[-1])
                outputs.append(self._sample_text(row.req_id, row.op, logits_rows[text_row]))
                text_row += 1
            elif slot.kind is ForwardOutputKind.DENOISE_STEP:
                outputs.append(self._apply_denoise(plan, slot.row_index, result))
            elif slot.kind is ForwardOutputKind.COMMIT:
                outputs.append(self._apply_commit(plan, slot.row_index, result))
            elif slot.kind is ForwardOutputKind.ENCODE:
                outputs.append(self._apply_encode(plan, slot.row_index, result))
            else:
                outputs.append({"req_id": row.req_id})
        return self._normalize_outputs(outputs)

    @staticmethod
    def _normalize_outputs(outputs: tuple[Any, ...] | list[Any]) -> list[Any]:
        return list(outputs)

    @staticmethod
    def _can_apply_text_batch(plan: ForwardPlan, result: ForwardResult) -> bool:
        return (
            isinstance(result.text_logits, torch.Tensor)
            and bool(plan.output_slots)
            and all(slot.kind is ForwardOutputKind.TEXT_TOKEN for slot in plan.output_slots)
        )

    def _apply_text_batch(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        result: ForwardResult,
        options: ForwardExecutionOptions,
    ) -> list[Any]:
        text = batch.as_text(allow_mixed_text=plan.forward_mode is ForwardMode.MIXED)
        req_ids = [int(row.req_id) for row in plan.rows]
        if result.text_logits is None:
            raise invalid_descriptor("text batch postprocess requires logits")
        logits_batch = self._text_logits_rows(result.text_logits, len(req_ids))
        if options.defer_sampling and self.tensor_store is not None:
            published_outputs = self._publish_logits(
                list(text.ops),
                req_ids,
                logits_batch,
                self.tensor_store,
            )
            self._publish_decode_position_relays(text, logits_batch.device)
            self._advance_text_kv_lengths(text)
            return published_outputs
        sampled_outputs = self._sample_text_logits_batch(
            plan,
            list(text.ops),
            req_ids,
            logits_batch,
            defer_cpu_results=options.defer_text_cpu_results,
            cuda_ready_start_event=result.text_cuda_ready_start_event,
        )
        self._publish_decode_position_relays(text, logits_batch.device)
        self._advance_text_kv_lengths(text)
        return sampled_outputs

    @staticmethod
    def _text_logits_rows(logits: torch.Tensor, row_count: int) -> torch.Tensor:
        if logits.ndim != 2:
            logits = logits.reshape(-1, logits.shape[-1])
        if int(logits.shape[0]) < int(row_count):
            raise invalid_descriptor("text logits row count is smaller than text output slots")
        return logits[: int(row_count)]

    def _sample_text_logits_batch(
        self,
        plan: ForwardPlan,
        ops: list[Mapping[str, Any]],
        req_ids: list[int],
        logits_batch: torch.Tensor,
        *,
        defer_cpu_results: bool,
        cuda_ready_start_event: torch.cuda.Event | None,
    ) -> list[TextTokenOutput | DeferredTextSeqResult]:
        if logits_batch.ndim != 2:
            raise invalid_descriptor("batched text logits rows must form a [batch, vocab] tensor")
        if int(logits_batch.shape[0]) != len(req_ids):
            raise invalid_descriptor("batched text logits row count must match req_ids")
        request_states = self.request_states
        if request_states is None:
            return [
                TextTokenOutput(
                    req_id=req_id,
                    sampled_token_id=int(torch.argmax(logits_batch[row].float()).item()),
                )
                for row, req_id in enumerate(req_ids)
            ]
        stats = get_forward_context().stats
        start = component_timer_start(stats)
        params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        generators: list[torch.Generator] = []
        for op, req_id in zip(ops, req_ids, strict=True):
            state = request_states.get(int(req_id))
            params.append(dict(state.sampling or {}))
            recent.append(op.get("recent_tokens") or [])
            allowed.append(op.get("allowed_tokens"))
            suppress.append(op.get("suppress_tokens"))
            generators.append(state.device_rng(logits_batch.device, stream="text_sampling"))
        sampling_result = apply_sampling_batched_with_device_tokens(
            logits_batch,
            params,
            recent,
            allowed,
            suppress,
            generators=generators,
            defer_cpu=defer_cpu_results,
            enable_cuda_timing=cuda_ready_start_event is not None,
        )
        if is_deferred_sampling_result(sampling_result):
            sampling_result.set_ready_start_event(cuda_ready_start_event)
            outputs: list[TextTokenOutput | DeferredTextSeqResult] = []
            for row, req_id in enumerate(req_ids):
                state = request_states.get(int(req_id))
                relay_token_tensor = sampling_result.device_tokens[row : row + 1]
                _DECODE_RELAY.publish_sample(
                    state,
                    token_id=None,
                    token_tensor=relay_token_tensor,
                )
                outputs.append(
                    DeferredTextSeqResult(
                        req_id=req_id,
                        row=row,
                        state=state,
                        sampling_result=sampling_result,
                        relay_token_tensor=relay_token_tensor,
                    )
                )
            record_component_elapsed(stats, "text_sample", start)
            return outputs

        immediate_result = finalize_sampling_result(sampling_result)
        outputs = []
        for row, (req_id, (tok, lp, top)) in enumerate(
            zip(req_ids, immediate_result.samples, strict=True)
        ):
            _DECODE_RELAY.publish_sample(
                request_states.get(int(req_id)),
                token_id=int(tok),
                token_tensor=immediate_result.device_tokens[row : row + 1],
            )
            outputs.append(
                TextTokenOutput(
                    req_id=int(req_id),
                    sampled_token_id=int(tok),
                    sampled_logprob=lp,
                    top_logprobs=(
                        [(int(item[0]), float(item[1]), int(item[2])) for item in top]
                        if top
                        else None
                    ),
                )
            )
        record_component_elapsed(stats, "text_sample", start)
        return outputs

    @staticmethod
    def _publish_logits(
        ops: list[Mapping[str, Any]],
        req_ids: list[int],
        logits_batch: torch.Tensor,
        tensor_store: Any,
    ) -> list[dict[str, Any]]:
        if logits_batch.ndim != 2 or int(logits_batch.shape[0]) != len(ops):
            raise invalid_descriptor("deferred-sampler logits must be shaped [ops, vocab]")
        if logits_batch.is_cuda:
            torch.cuda.synchronize(logits_batch.device)
        results: list[dict[str, Any]] = []
        for row, op in enumerate(ops):
            handle = tensor_store.publish(logits_batch[row].contiguous(), "logits")
            result: dict[str, Any] = {"req_id": int(op["req_id"]), "logits_handle": int(handle)}
            locator = tensor_store.locator_of(handle)
            if locator is not None:
                result["locator"] = base64.b64encode(locator).decode("ascii")
            results.append(result)
        return results

    def _publish_decode_position_relays(self, text: Any, device: torch.device) -> None:
        if text.mode is not ForwardMode.DECODE:
            return
        request_states = self.request_states
        if request_states is None:
            return
        if any(len(tokens) != 1 for tokens in text.token_ids):
            return
        stats = get_forward_context().stats
        start = component_timer_start(stats)
        states = [request_states.get(int(req_id)) for req_id in text.req_ids]
        positions = [int(pos_range[1]) for pos_range in text.pos_ranges]
        _DECODE_RELAY.publish_positions(states, position_ids=positions, device=device)
        record_component_elapsed(stats, "text_decode_position_store", start)

    def _advance_text_kv_lengths(self, text: Any) -> None:
        if text.mode not in _TEXT_MODES:
            return
        request_states = self.request_states
        if request_states is None:
            return
        for req_id, pos_range in zip(text.req_ids, text.pos_ranges, strict=True):
            request_states.get(int(req_id)).set_kv_length(int(pos_range[1]), lane="text")

    def _sample_text(self, req_id: int, op: Any, logits: torch.Tensor) -> dict[str, Any]:
        request_states = self.request_states
        if request_states is not None:
            state = request_states.get(int(req_id))
            return sample_logits_result(req_id=int(req_id), state=state, logits=logits, op=op)
        token = int(torch.argmax(logits.float()).item())
        return {"req_id": int(req_id), "sampled_token_id": token}

    def _apply_text_entries(
        self,
        plan: ForwardPlan,
        result: ForwardResult,
        options: ForwardExecutionOptions,
    ) -> dict[int, Any]:
        entries = tuple(result.text_postprocess or ())
        if not entries:
            return {}
        if result.text_logits is None:
            raise invalid_descriptor("text postprocess entries require text logits")
        logits_rows = self._text_logits_rows(result.text_logits, len(entries))
        entries_by_index = sorted(entries, key=lambda entry: int(entry.logits_index))
        if [int(entry.logits_index) for entry in entries_by_index] != list(
            range(len(entries_by_index))
        ):
            raise invalid_descriptor("text postprocess logits indices must be contiguous")
        request_states = self.request_states
        params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        generators: list[torch.Generator | None] = []
        for entry in entries_by_index:
            row = plan.rows[int(entry.row_index)]
            state = request_states.get(int(entry.req_id)) if request_states is not None else None
            params.append(dict(getattr(state, "sampling", {}) or {}))
            recent.append(row.op.get("recent_tokens") or [])
            allowed.append(row.op.get("allowed_tokens"))
            suppress.append(row.op.get("suppress_tokens"))
            generators.append(
                None
                if state is None
                else state.device_rng(logits_rows.device, stream="text_sampling")
            )
        sampling_result = apply_sampling_batched_with_device_tokens(
            logits_rows[: len(entries_by_index)],
            params,
            recent,
            allowed,
            suppress,
            generators=generators,
            defer_cpu=options.defer_text_cpu_results,
        )
        deferred: Any | None
        if is_deferred_sampling_result(sampling_result):
            deferred = sampling_result
            samples = []
            device_tokens = sampling_result.device_tokens
        else:
            deferred = None
            immediate_result = finalize_sampling_result(sampling_result)
            samples = immediate_result.samples
            device_tokens = immediate_result.device_tokens
        promotions = [
            entry.kv_promotion for entry in entries_by_index if entry.kv_promotion is not None
        ]
        if promotions:
            num_layers = max(int(entry.num_layers) for entry in entries_by_index)
            copy_paged_text_cache_spans(
                promotions,
                num_layers=num_layers,
                missing_message="forward text K/V span is missing from staged cache",
            )
        outputs: dict[int, Any] = {}
        for sample_index, entry in enumerate(entries_by_index):
            row = plan.rows[int(entry.row_index)]
            state = request_states.get(int(entry.req_id)) if request_states is not None else None
            logits = logits_rows[int(entry.logits_index) : int(entry.logits_index) + 1].unsqueeze(0)
            self._publish_text_entry_state(entry, logits)
            token_tensor = device_tokens[sample_index : sample_index + 1]
            position_tensor = torch.tensor(
                [int(entry.position_id)],
                dtype=torch.long,
                device=logits_rows.device,
            )
            if state is not None:
                _DECODE_RELAY.publish_sample(
                    state,
                    token_id=None if deferred is not None else int(samples[sample_index].token_id),
                    token_tensor=token_tensor,
                )
                _DECODE_RELAY.publish_position(
                    state,
                    position_id=int(entry.position_id),
                    position_tensor=position_tensor,
                )
            output: Any
            if deferred is not None:
                if state is None:
                    raise invalid_descriptor("deferred text postprocess requires request state")
                output = DeferredTextSeqResult(
                    req_id=int(entry.req_id),
                    row=sample_index,
                    state=state,
                    sampling_result=deferred,
                    relay_token_tensor=token_tensor,
                )
            else:
                sample = samples[sample_index]
                top_logprobs = (
                    [(int(item[0]), float(item[1]), int(item[2])) for item in sample.top_logprobs]
                    if sample.top_logprobs is not None
                    else None
                )
                output = TextTokenOutput(
                    req_id=int(entry.req_id),
                    sampled_token_id=int(sample.token_id),
                    sampled_logprob=sample.logprob,
                    top_logprobs=top_logprobs,
                )
            outputs[int(row.row_index)] = output
        return outputs

    @staticmethod
    def _publish_text_entry_state(entry: TextPostprocessEntry, logits: torch.Tensor) -> None:
        state = entry.program_state
        cond = getattr(state, "cond", None)
        if cond is not None:
            cond.t_index = int(entry.position_id) - 1
            cond.last_logits = logits
            cond.last_token_id = int(entry.last_input_token)
        persistent_cache = entry.persistent_cache
        if persistent_cache is not None:
            persistent_cache.length = int(entry.kv_new_length)
        if (
            entry.staged_cache is not None
            and persistent_cache is not None
            and entry.staged_cache is not persistent_cache
            and entry.mark_staging_advanced is not None
        ):
            entry.mark_staging_advanced(
                entry.staged_cache, persistent_cache, int(entry.kv_new_length)
            )

    @staticmethod
    def _apply_denoise(plan: ForwardPlan, row_index: int, result: ForwardResult) -> FlowOutput:
        row = plan.rows[row_index]
        denoise = row.denoise
        if denoise is None:
            raise invalid_descriptor("denoise output slot references a non-denoise row")
        velocities = result.denoise_velocities or {}
        update = (result.denoise_updates or {}).get(int(row_index))
        if update is not None:
            branch_velocities: dict[Any, torch.Tensor] = {}
            for branch_id, branch_name in enumerate(update.branch_names):
                key = DenoiseBranchKey(row_index, branch_id)
                velocity = velocities.get(key)
                if velocity is None:
                    raise invalid_descriptor("denoise output is missing branch velocity")
                branch_velocities[branch_name] = velocity
            combined = update.combine_velocity(branch_velocities)
            if not isinstance(combined, torch.Tensor):
                raise invalid_descriptor("denoise combined velocity must be a tensor")
            if tuple(combined.shape) != tuple(update.latent.shape):
                raise invalid_descriptor("denoise combined velocity shape must match latent shape")
            updated = euler_step(update.latent, combined, update.t, update.t_next)
            update.accept_update(updated)
            done = update.step_index + 1 >= update.total_steps
            return FlowOutput(
                req_id=update.req_id,
                denoise_done=done,
                num_steps_done=update.step_index + 1,
            )
        for branch_id in range(denoise.branch_count):
            key = DenoiseBranchKey(row_index, branch_id)
            if key not in velocities:
                raise invalid_descriptor("denoise output is missing branch velocity")
        done = denoise.step_index + 1 >= denoise.total_steps
        return FlowOutput(
            req_id=row.req_id,
            denoise_done=done,
            num_steps_done=denoise.step_index + 1,
        )

    def _apply_commit(
        self,
        plan: ForwardPlan,
        row_index: int,
        result: ForwardResult,
    ) -> CommitOutput:
        row = plan.rows[row_index]
        if row.commit is None:
            raise invalid_descriptor("commit output slot references a non-commit row")
        if result.commit_outputs is None:
            return CommitOutput(req_id=row.req_id)
        return _commit_output_from_value(
            int(row.req_id),
            self.request_states.get(int(row.req_id)) if self.request_states is not None else None,
            row.op,
            result.commit_outputs[int(row_index)],
        )

    @staticmethod
    def _apply_encode(plan: ForwardPlan, row_index: int, result: ForwardResult) -> EncodeOutput:
        row = plan.rows[row_index]
        if row.encode is None:
            raise invalid_descriptor("encode output slot references a non-encode row")
        if result.encode_outputs is None:
            return EncodeOutput(req_id=row.req_id, encoder_handle=0)
        return _coerce_encode_output(result.encode_outputs[int(row_index)])


def _commit_output_from_value(
    req_id: int,
    state: Any,
    op: Mapping[str, Any],
    value: Any,
) -> CommitOutput:
    out = dict(value) if isinstance(value, Mapping) else _image_to_result(req_id, value)
    logits = out.pop("logits", None)
    if logits is not None:
        if state is None:
            raise invalid_descriptor("commit logits require request state for sampling")
        sampled = sample_logits_result(req_id=int(req_id), state=state, logits=logits, op=op)
        sampled.pop("req_id", None)
        out.update(sampled)
    return _commit_output_from_dict(req_id, out)


def _to_seq_result(output: ForwardOutput | Mapping[str, Any]) -> Any:
    if isinstance(output, ForwardOutputBase):
        return output.to_seq_result()
    if isinstance(output, DeferredForwardOutput):
        return output.to_seq_result()
    if isinstance(output, Mapping):
        return dict(output)
    raise invalid_descriptor(f"unsupported forward output type {type(output).__name__}")


# ---------------------
# Complete-batch execution
# ---------------------

_STREAM_OVERLAP_MODES = frozenset(
    {ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.VERIFY_DRAFT}
)


def _overlap_eligible(group: list[tuple[int, Mapping[str, Any]]]) -> bool:
    """Plan/forward stream overlap covers text-only groups.

    Flow, encode, and commit groups run packed graph programs with their own
    buffer ownership; their prepare phases stay on the forward stream.
    """

    return all(mode_for_op(op["kind"]) in _STREAM_OVERLAP_MODES for _, op in group)


# ---------------------
# Plan/forward stream overlap
# ---------------------

_T = TypeVar("_T")


@dataclass(frozen=True)
class PreparedOnPlanStream:
    """One prepare-phase result plus the event that fences its consumption."""

    value: Any
    plan_done: torch.cuda.Event


@dataclass(frozen=True)
class _InflightForward:
    plan_done: torch.cuda.Event
    forward_done: torch.cuda.Event
    retained: Any


class PlanStreamOverlap:
    """Runs batch preparation on a dedicated stream, fenced against reuse.

    ``max_inflight`` must equal the staging-ring depth: it is the reuse period
    of the pinned/device staging buffers, and the coordinator's fences are what
    make that reuse safe across streams.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        max_inflight: int,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("PlanStreamOverlap requires a CUDA device")
        self.device = device
        self.plan_stream = torch.cuda.Stream(device=device)
        self.max_inflight = max(1, int(max_inflight))
        self._inflight: deque[_InflightForward] = deque()

    def prepare(self, build: Callable[[], _T]) -> PreparedOnPlanStream:
        """Run ``build`` under ``plan_stream`` and record its completion event.

        Applies the ring-reuse fences before running: host-sync on the oldest
        retired prepare (pinned-buffer WAR) and a device-side wait on its
        forward (device-buffer WAR), then releases that batch's references.
        """

        while len(self._inflight) >= self.max_inflight:
            oldest = self._inflight.popleft()
            oldest.plan_done.synchronize()
            self.plan_stream.wait_event(oldest.forward_done)
        with torch.cuda.stream(self.plan_stream):
            value = build()
            plan_done = torch.cuda.Event()
            plan_done.record(self.plan_stream)
        return PreparedOnPlanStream(value=value, plan_done=plan_done)

    def launch(
        self,
        prepared: PreparedOnPlanStream,
        run: Callable[[], _T],
        *,
        retain: Any,
    ) -> _T:
        """Launch the forward on the current stream after the plan event.

        ``retain`` (the prepared batch) is held until the ring-reuse fence for
        its slot passes, keeping plan-stream allocations alive while the
        forward may still read them.
        """

        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(prepared.plan_done)
        result = run()
        forward_done = torch.cuda.Event()
        forward_done.record(stream)
        self._inflight.append(
            _InflightForward(
                plan_done=prepared.plan_done,
                forward_done=forward_done,
                retained=retain,
            )
        )
        return result

    def drain(self) -> None:
        """Retire every in-flight forward (host-blocking); used by tests."""

        while self._inflight:
            entry = self._inflight.popleft()
            entry.plan_done.synchronize()
            entry.forward_done.synchronize()


# ---------------------
# Worker model runner assembly
# ---------------------

_MIXED_PROOF_LOG = logging.getLogger("uniserve.mixed_proof")
# Per-step mixed-forward trace (opt-in). Fallback warnings are always emitted.
_MIXED_PROOF_ENABLED = env_flag("UNISERVE_MIXED_PROOF_LOG")


def _model_max_context_len(model: Any) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, "max_position_embeddings", None)
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


@dataclass
class RunnerConfig:
    """ModelRunner configuration knobs.

    Groups the sampler-stage split (``defer_sampling``/``tensor_store``),
    multimodal processor, batch policy, and attention-backend selection.
    """

    batch_policy: BatchPolicy | None = None
    attention_backend: Any | None = None
    multimodal_processor: Any | None = None
    defer_sampling: bool = False
    tensor_store: Any | None = None
    simulation: bool = False


@dataclass
class _ResolvedRunnerDependencies:
    batch_policy: BatchPolicy | None
    attention_backend: Any | None
    multimodal_processor: Any | None
    defer_sampling: bool
    tensor_store: Any | None
    simulation: bool


@dataclass(frozen=True)
class _TextExecutionStack:
    builder: ForwardBatchBuilder | None
    gate: Any | None
    graph_runner: Any | None


class ModelRunner:
    """Prepare, execute, and project one intact model batch."""

    def __init__(
        self,
        model: UniModel,
        request_states: RequestSessionTable | None = None,
        *,
        config: RunnerConfig | None = None,
        resource_runtime: ResourceRuntime | None = None,
        residency: "ResidencyManager | None" = None,
        batch_policy: BatchPolicy | None = None,
        attention_backend: Any | None = None,
        multimodal_processor: Any | None = None,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
    ):
        if not isinstance(model, ModelHooks):
            raise capability_mismatch("runner model must inherit ModelHooks")
        dependencies = self._resolve_dependencies(
            config=config,
            batch_policy=batch_policy,
            attention_backend=attention_backend,
            multimodal_processor=multimodal_processor,
            defer_sampling=defer_sampling,
            tensor_store=tensor_store,
        )

        self.model = model
        self.residency = residency
        self.request_states = request_states or RequestSessionTable()
        # Deferred sampling: text decode/extend ops publish logits to
        # ``tensor_store`` and return handles — a separate Sampler worker samples.
        # Off = sample inline (default).
        self.defer_sampling = (
            bool(dependencies.defer_sampling) and dependencies.tensor_store is not None
        )
        self.tensor_store = dependencies.tensor_store
        self.simulation = bool(dependencies.simulation)
        self.batch_policy = dependencies.batch_policy or self._model_batch_policy()
        self.attention_backend, self.attention_preference = self._resolve_attention_backend(
            dependencies.attention_backend
        )
        self._diffusion = _DiffusionRuntime()
        self._init_text_execution(model, residency)
        self._init_unified_forward_execution(model, residency)
        self.multimodal_processor = dependencies.multimodal_processor
        self._init_resource_accounting(resource_runtime, residency)
        # CUDA Green Context SM partitioning. ``None`` unless runtime config
        # enables it and the model runs on a CUDA device.
        self.stream_manager = self._maybe_build_stream_manager()

    def _init_unified_forward_execution(
        self,
        model: UniModel,
        residency: "ResidencyManager | None",
    ) -> None:
        device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
        kv_pool = residency.kv if residency is not None else None
        self.forward_plan_builder = ForwardPlanBuilder()
        self.unified_forward_batch_builder = UnifiedForwardBatchBuilder(
            runtime_builder=self.forward_batch_builder,
            kv_pool=kv_pool,
            request_states=self.request_states,
            default_device=device,
        )
        self.forward_graph_policy = ForwardGraphPolicy(
            prefer_graph=bool(get_execution_config().cuda_graph),
            strict=not self.simulation,
        )
        self.forward_postprocessor = ForwardPostprocessor(
            request_states=self.request_states,
            tensor_store=self.tensor_store,
        )
        self.plan_stream_overlap = self._maybe_build_plan_stream_overlap(device)

    def _maybe_build_plan_stream_overlap(self, device: torch.device):
        """Build the plan/forward stream-overlap coordinator when enabled.

        Gated behind ``UNISERVE_STREAM_OVERLAP=1`` per
        ``specs/intra-worker-stream-overlap.md``; requires a CUDA device. The
        in-flight bound is the staging-ring reuse period so the coordinator's
        WAR fences cover pinned and device staging-buffer recycling.
        """
        if not env_flag("UNISERVE_STREAM_OVERLAP"):
            return None
        if device.type != "cuda":
            return None
        ring_depth = getattr(self.forward_batch_builder, "staging_ring_depth", 3)
        overlap = PlanStreamOverlap(device, max_inflight=int(ring_depth))
        logger.info(
            "intra-worker stream overlap enabled (plan stream, max_inflight=%d)",
            overlap.max_inflight,
        )
        return overlap

    def _init_text_execution(
        self,
        model: UniModel,
        residency: "ResidencyManager | None",
    ) -> None:
        # System text execution: a thin model declares its KV geometry and the
        # runtime owns builder/gate/graph/sampler around the graph-unaware model.
        text_stack = self._build_text_execution(model, residency)
        self.forward_batch_builder = text_stack.builder
        self.text_gate = text_stack.gate
        self.text_graph_runner = text_stack.graph_runner
        self._autoregressive = _AutoregressiveRuntime(
            builder=self.forward_batch_builder,
            gate=self.text_gate,
            kv_pool=residency.kv if residency is not None else None,
            graph_runner=self.text_graph_runner,
        )

    def _init_resource_accounting(
        self,
        resource_runtime: ResourceRuntime | None,
        residency: "ResidencyManager | None",
    ) -> None:
        resource_plan = self._model_resource_plan()
        classes = resource_plan.classes()
        self.resource_runtime = resource_runtime or ResourceRuntime(
            classes,
            totals=self._model_resource_totals(classes),
        )
        # The accountant holds ``resource_plan`` as the single source of truth;
        # ``ModelRunner.resource_plan`` forwards to it so a runtime reassignment
        # is seen by both.
        self._accountant = ResidencyLeaseManager(
            self.resource_runtime,
            self.request_states,
            resource_plan,
            residency=residency,
        )

    @staticmethod
    def _resolve_dependencies(
        *,
        config: RunnerConfig | None,
        batch_policy: BatchPolicy | None,
        attention_backend: Any | None,
        multimodal_processor: Any | None,
        defer_sampling: bool,
        tensor_store: Any | None,
    ) -> _ResolvedRunnerDependencies:
        config = config or RunnerConfig()
        return _ResolvedRunnerDependencies(
            batch_policy=batch_policy if batch_policy is not None else config.batch_policy,
            attention_backend=(
                attention_backend if attention_backend is not None else config.attention_backend
            ),
            multimodal_processor=(
                multimodal_processor
                if multimodal_processor is not None
                else config.multimodal_processor
            ),
            defer_sampling=defer_sampling or config.defer_sampling,
            tensor_store=tensor_store if tensor_store is not None else config.tensor_store,
            simulation=bool(config.simulation),
        )

    @staticmethod
    def _resolve_attention_backend(attention_backend: Any | None) -> tuple[Any | None, str | None]:
        if isinstance(attention_backend, str) or attention_backend is None:
            attention_preference = normalize_attention_backend_name(attention_backend or "auto")
            if attention_preference != "auto":
                get_attention_backend(attention_preference)
            return None, attention_preference
        return attention_backend, getattr(attention_backend, "name", None)

    @property
    def resource_plan(self) -> ResourcePlan:
        return self._accountant.resource_plan

    @resource_plan.setter
    def resource_plan(self, plan: ResourcePlan) -> None:
        self._accountant.resource_plan = plan

    def _build_text_execution(
        self,
        model: UniModel,
        residency: "ResidencyManager | None",
    ) -> _TextExecutionStack:
        """Build the system text builder + backend gate + CUDA-graph runner.

        A thin model declares ``kv_cache_spec`` and the runtime owns its KV pool
        (``residency.kv``); the gate decides batched-paged vs per-op-dense from
        the model geometry + the system pool's storage flag, and the graph runner
        captures/replays decode/prefill graphs around the graph-unaware model.
        Models without ``kv_cache_spec`` (self-managed KV) get none of these.
        """

        if residency is None or residency.kv is None:
            return _TextExecutionStack(None, None, None)
        if self._model_kv_cache_spec(model) is None:
            return _TextExecutionStack(None, None, None)
        import torch

        from uniserve_worker.backends.attention.text_dispatch import TextBackendGate

        device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
        max_context_len = _model_max_context_len(model)
        gate = TextBackendGate(
            head_dim=int(getattr(model, "head_dim", residency.kv.head_dim)),
            block_size=int(residency.kv.block_size),
            device_type=device.type,
            paged_storage_ok=bool(getattr(residency.kv, "supports_paged_attention_storage", True)),
        )
        graph_runner = None
        if device.type == "cuda":
            from uniserve_worker.execution.graph import Executor

            graph_runner = Executor(
                kv_pool=residency.kv,
                num_blocks=int(residency.kv.num_blocks),
                block_size=int(residency.kv.block_size),
                device=device,
                attention_preference=self.attention_preference,
                max_context_len=max_context_len,
            )
            try:
                graph_runner.warmup(model)
            except Exception:  # noqa: BLE001 - a warmup failure must never block serving.
                logger.warning(
                    "text CUDA-graph warmup failed; falling back to eager", exc_info=True
                )
        return _TextExecutionStack(
            ForwardBatchBuilder(max_context_len=max_context_len),
            gate,
            graph_runner,
        )

    @staticmethod
    def _model_kv_cache_spec(model: UniModel) -> Any | None:
        return model.kv_cache_spec()

    def _maybe_build_stream_manager(self):
        if not get_execution_config().green_contexts:
            return None
        import torch

        device = torch.device(str(getattr(self.model, "device", "cpu") or "cpu"))
        if device.type != "cuda":
            return None
        try:
            from uniserve_worker.runtime.stream_manager import StreamManager

            gpu_id = device.index if device.index is not None else torch.cuda.current_device()
            manager = StreamManager(int(gpu_id))
            logger.info(
                "green contexts enabled: %d SMs, %d stream groups (partitioned=%s)",
                manager.total_sms,
                len(manager.stream_groups),
                manager.using_green_contexts,
            )
            return manager
        except Exception:  # noqa: BLE001 - never let a stream-setup failure block serving.
            logger.warning("StreamManager init failed; green contexts disabled", exc_info=True)
            return None

    def _forward_stream_context(self, plan: ForwardPlan):
        """Context manager that runs a single-mode group on its SM partition.

        Prefill/verify groups run on the prefill (large-SM) partition; decode
        groups on the decode partition sized by the running decode batch; mixed
        and non-text groups run full-SM (no partitioning helps a mixed forward).
        A no-op ``nullcontext`` when green contexts are disabled.
        """
        from contextlib import nullcontext

        if self.stream_manager is None:
            return nullcontext()
        mode = plan.forward_mode
        if mode == ForwardMode.EXTEND or mode == ForwardMode.VERIFY_DRAFT:
            stream = self.stream_manager.select_streams(0)[0]
        elif mode == ForwardMode.DECODE:
            stream = self.stream_manager.select_streams(plan.shape.row_count)[1]
        else:
            stream = self.stream_manager.default_stream()
        import torch

        return torch.cuda.stream(stream)

    def drop_request(self, req_id: int) -> None:
        self.model.drop_request(req_id)
        self._accountant.release_request(int(req_id))
        self.request_states.drop(req_id)

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        parsed = WireExecuteBatch.from_wire(batch)
        options = ForwardExecutionOptions(
            defer_text_cpu_results=defer_text_cpu_results,
            defer_sampling=self.defer_sampling,
        )
        self._register_new_reqs(parsed.new_reqs)
        if len(parsed.ops) > int(self.batch_policy.max_batch_ops):
            raise invalid_descriptor("execute batch exceeds the model's maximum operation count")
        group = list(enumerate(parsed.ops))
        self._accountant.account_group(group)
        plan = self.forward_plan_builder.build(
            group,
            request_states=self.request_states,
            step_id=parsed.step_id,
            graph_policy=self.forward_graph_policy,
        )
        forward_stats = ForwardStats() if env_flag("UNISERVE_FORWARD_METRICS") else None
        if forward_stats is not None:
            self._record_group_shape(forward_stats, plan)
        device = torch.device(str(getattr(self.model, "device", "cpu") or "cpu"))
        context = ForwardContext(
            attention_backend=self.attention_backend,
            attention_preference=self.attention_preference,
            kv_pool=self.residency.kv if self.residency is not None else None,
            stats=forward_stats,
            request_states=self.request_states,
            execution_options=options,
            default_model_forward=self._default_model_forward,
            tensor_store=self.tensor_store,
        )

        def prepare() -> ForwardBatch:
            text_batch = self._autoregressive.prepare_batch(
                plan,
                self.request_states,
                device=device,
            )
            if text_batch is not None:
                return text_batch
            return self.unified_forward_batch_builder.build(plan, device=device)

        overlap = self.plan_stream_overlap
        if overlap is None or device.type != "cuda" or not _overlap_eligible(group):
            overlap = None
        with torch.inference_mode(), use_forward_context(context):
            prepared = overlap.prepare(prepare) if overlap is not None else None
            forward_batch = prepared.value if prepared is not None else prepare()
        if _MIXED_PROOF_ENABLED and forward_batch.mode is ForwardMode.MIXED:
            self._log_mixed_proof(forward_batch, group)
        context = replace(context, attention_plan=forward_batch.attn_plan)
        started = time.perf_counter_ns() if forward_stats is not None else 0

        def run_forward() -> ForwardResult:
            with (
                torch.inference_mode(),
                use_forward_context(context),
                self._forward_stream_context(plan),
            ):
                return coerce_forward_result(self.model.forward(forward_batch))

        if overlap is not None and prepared is not None:
            result = overlap.launch(prepared, run_forward, retain=forward_batch)
        else:
            result = run_forward()
        if forward_stats is not None:
            forward_stats.record_mode_wall_time(
                plan.forward_mode.value,
                time.perf_counter_ns() - started,
            )
        with torch.inference_mode():
            outputs = [
                _to_seq_result(output)
                for output in self.forward_postprocessor.apply(
                    forward_batch,
                    plan,
                    result,
                    options,
                )
            ]
        self._advance_state(plan, outputs)
        self._stamp_conditioning_locators(plan, outputs)
        response: dict[str, Any] = {"step_id": parsed.step_id, "per_seq": outputs}
        if forward_stats is not None:
            response["forward_stats"] = forward_stats.to_wire()
        return response

    def _default_model_forward(self, model: ModelHooks, batch: ForwardBatch) -> ForwardResult:
        """Execute a homogeneous batch for models using the shared runtime."""

        context = get_forward_context()
        options = context.execution_options
        if not isinstance(options, ForwardExecutionOptions):
            options = ForwardExecutionOptions()
        if batch.op_modes and all(mode in _TEXT_MODES for mode in batch.op_modes):
            return self._autoregressive.forward_result(
                batch,
                self.request_states,
                model,
                options=options,
                tensor_store=self.tensor_store,
            )
        items = [
            (int(op["req_id"]), self.request_states.get(int(op["req_id"])), op) for op in batch.ops
        ]
        rows = tuple(range(len(items)))
        if batch.mode is ForwardMode.DENOISE:
            result = self._diffusion.forward_result(
                items,
                model,
                row_indices=rows,
                graph_mode="eager",
            )
            if result is not None:
                return result
            return ForwardResult(runtime_outputs=tuple(self._diffusion.step_many(items, model)))
        if batch.mode is ForwardMode.COMMIT:
            return commit_result(items, model, row_indices=rows)
        if batch.mode is ForwardMode.ENCODE:
            return encode_result(batch, model, row_indices=rows)
        raise invalid_descriptor("mixed-mode models must implement forward(batch)")

    def _register_new_reqs(self, new_reqs: tuple[Mapping[str, Any], ...]) -> None:
        """Create/refresh request state and account resident blocks for new reqs.

        A block-accounting failure rolls back any request *this* call freshly
        created (existing requests are left untouched) before re-raising.
        """
        for nr in new_reqs:
            req_id = nr["req_id"]
            existed = req_id in self.request_states
            state = self.request_states.create_or_update(req_id, dict(nr))
            try:
                self._accountant.account_blocks(req_id, state.block_ids, append_to_state=False)
            except Exception:
                if not existed:
                    self._accountant.release_request(int(req_id))
                    self.request_states.drop(req_id)
                raise
            self.model.on_new_request(req_id, state)

    def _stamp_conditioning_locators(
        self,
        plan: ForwardPlan,
        results: list[Any],
    ) -> None:
        """Mode A: stamp the und->gen conditioning locator on a text result that
        begins an image, so the gen pool can fetch the conditioning KV.

        Delegates the decision to the model's ``maybe_publish_conditioning`` hook
        (a no-op unless a data-plane handoff is bound). Only fires for text-decode
        results carrying an inline sampled token (the image-start trigger)."""
        if plan.forward_mode not in _TEXT_MODES:
            return
        for row in plan.rows:
            result = results[row.original_index]
            if not isinstance(result, dict):
                continue
            sampled = result.get("sampled_token_id")
            if sampled is None:
                continue
            locator = self.model.maybe_publish_conditioning(int(row.req_id), int(sampled))
            if locator:
                result["locator"] = locator

    def _log_mixed_proof(
        self,
        fb: ForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
    ) -> None:
        # Per-step "mixed forward executed" trace (opt-in via the proof flag).
        n_ext = sum(1 for m in fb.op_modes if m == ForwardMode.EXTEND)
        n_dec = sum(1 for m in fb.op_modes if m == ForwardMode.DECODE)
        n_den = sum(1 for m in fb.op_modes if m == ForwardMode.DENOISE)
        n_tok = sum(len(op.get("token_ids") or []) for _, op in group)
        _MIXED_PROOF_LOG.info(
            "MIXED FORWARD executed: %d ops (%d extend + %d decode + %d denoise), %d tokens",
            len(fb.ops),
            n_ext,
            n_dec,
            n_den,
            n_tok,
        )

    def _log_text_mixed_split(self, ops: list[Mapping[str, Any]], decision: Any) -> None:
        # Warn when a prefill+decode text mix is split to per-mode groups.
        modes = {mode_for_op(str(op.get("kind"))) for op in ops}
        if {ForwardMode.EXTEND, ForwardMode.DECODE} <= modes:
            _MIXED_PROOF_LOG.warning(
                "MIXED TEXT SPLIT: scheduler co-batched %d ops with both prefill+decode "
                "and runner is splitting per-mode (use_forward=%s)",
                len(ops),
                decision.use_forward,
            )

    def _model_batch_policy(self) -> BatchPolicy:
        policy = self.model.batch_policy()
        if isinstance(policy, BatchPolicy):
            return policy
        raise invalid_descriptor("model.batch_policy() must return BatchPolicy")

    def _model_resource_plan(self) -> ResourcePlan:
        return self.model.resource_plan

    def _model_resource_totals(self, classes: tuple[str, ...]) -> dict[str, int]:
        caps = self._model_caps()
        totals = {cls: 0 for cls in classes}
        if "kv_block" in totals:
            totals["kv_block"] = int(caps.num_blocks if caps is not None else 0)
        if "scratch" in totals:
            totals["scratch"] = int(caps.scratch_capacity_tokens if caps is not None else 0)
        if "image_latent" in totals:
            totals["image_latent"] = int(caps.max_latent_size if caps is not None else 0)
        if "encoder_output" in totals:
            totals["encoder_output"] = int(
                caps.encoder_cache_budget
                if caps is not None and caps.encoder_cache_budget is not None
                else 0
            )
        if "adapter" in totals:
            totals["adapter"] = 0
        return totals

    def _model_caps(self) -> Caps | None:
        return self.model.caps() if isinstance(self.model, ModelHooks) else None

    def _advance_state(self, plan: ForwardPlan, results: list[Any]) -> None:
        if plan.forward_mode == ForwardMode.MIXED:
            for mode, result in zip(plan.op_modes, results, strict=True):
                self._advance_op_state(mode, result)
            return
        for result in results:
            self._advance_op_state(plan.forward_mode, result)

    def _advance_op_state(self, mode: ForwardMode, result: Any) -> None:
        if mode == ForwardMode.DENOISE:
            self.request_states.advance_denoise(
                int(result["req_id"]),
                int(result["num_steps_done"]) if result.get("num_steps_done") is not None else None,
            )
        elif mode == ForwardMode.COMMIT:
            req_id = int(result["req_id"])
            self._accountant.release_generation(req_id, committed=True)

    def _record_group_shape(self, stats: ForwardStats, plan: ForwardPlan) -> None:
        if plan.forward_mode is ForwardMode.MIXED:
            for mode, op in zip(plan.op_modes, plan.ops, strict=True):
                stats.record_mode_shape(
                    mode.value,
                    ops=1,
                    tokens=self._op_token_count(op),
                )
            return
        stats.record_mode_shape(
            plan.forward_mode.value,
            ops=plan.shape.row_count,
            tokens=sum(self._op_token_count(op) for op in plan.ops),
        )

    def _op_token_count(self, op: Mapping[str, Any]) -> int:
        mode = mode_for_op(str(op.get("kind")))
        if mode in _TEXT_MODES:
            if mode == ForwardMode.DECODE:
                try:
                    return max(1, int(op.get("decode_token_count") or 1))
                except (TypeError, ValueError):
                    raise invalid_descriptor(
                        "decode_token_count must be a positive integer"
                    ) from None
            tokens = op.get("token_ids") or []
            return len(tokens) if isinstance(tokens, (list, tuple)) else 0
        if mode == ForwardMode.DENOISE:
            req_id = int(op["req_id"])
            state = self.request_states.get(req_id)
            cfg = op.get("cfg")
            # Token throughput counts every denoise CFG branch iteration.
            branch_count = int(cfg.get("branch_count") or 1) if isinstance(cfg, Mapping) else 1
            step_count = max(1, int(op.get("denoise_step_count") or 1))
            latent_rule = self.resource_plan.image_latent or LatentTokens(downsample=16)
            return (
                self._accountant.latent_units(op, state.image, latent_rule)
                * max(1, branch_count)
                * step_count
            )
        if mode == ForwardMode.COMMIT:
            req_id = int(op["req_id"])
            state = self.request_states.get(req_id)
            latent_rule = self.resource_plan.image_latent or LatentTokens(downsample=16)
            return self._accountant.latent_units(op, state.image, latent_rule)
        return 0
