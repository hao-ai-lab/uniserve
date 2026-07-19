"""Internal construction of immutable forward plans and device batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import (
    Any,
)

import torch

from uniserve_worker.contracts.batches import TextBatch
from uniserve_worker.contracts.forward_batch import (
    BranchSpec,
    CacheSpanPlan,
    CfgPlan,
    CommitInputs,
    CommitRowPlan,
    DenoiseInputs,
    DenoiseRowPlan,
    EncodeInputs,
    EncodeRowPlan,
    ForwardBatch,
    ForwardGraphPolicy,
    ForwardModality,
    ForwardOutputKind,
    ForwardOutputSlot,
    ForwardPlan,
    ForwardPostprocessPolicy,
    ForwardResultProjection,
    ForwardRowPlan,
    ForwardSegmentClass,
    ForwardSegmentPlan,
    ForwardShapeSummary,
    KvSource,
    KvWritePolicy,
    SegmentSpec,
    TextTokenSpanPlan,
    VisiblePolicy,
)
from uniserve_worker.contracts.forward_mode import ForwardMode, mode_for_op
from uniserve_worker.foundation.errors import invalid_descriptor

# ---------------------
# Canonical plan construction (rows, segments, output slots)
# ---------------------

_TEXT_MODES = frozenset({ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT})


class _ForwardPlanBuilder:
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


# ---------------------
# Device batch construction
# ---------------------


class _UnifiedForwardBatchBuilder:
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
