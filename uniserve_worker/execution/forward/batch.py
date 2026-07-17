"""Materialize execution-layer ``ForwardBatch`` values from forward plans."""
from __future__ import annotations

from typing import Any

import torch

from ...contracts.batches import UniForwardBatch
from ...contracts.forward_batch import (
    BranchSpec,
    CfgPlan,
    CommitInputs,
    DenoiseInputs,
    EncodeInputs,
    ForwardBatch,
    KvSource,
    SegmentSpec,
)
from ...contracts.forward_mode import ForwardMode
from ...foundation.errors import invalid_descriptor
from .plan import ForwardModality, ForwardPlan

__all__ = ["ForwardBatchBuilder"]

_TEXT_MODES = frozenset(
    {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT}
)


class ForwardBatchBuilder:
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
        text = UniForwardBatch.from_ops(plan.ops).as_text(
            allow_mixed_text=plan.forward_mode is ForwardMode.MIXED
        )
        return self.runtime_builder.build_text(
            text,
            device=device,
            kv_pool=self.kv_pool,
            request_states=self.request_states,
        )

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
