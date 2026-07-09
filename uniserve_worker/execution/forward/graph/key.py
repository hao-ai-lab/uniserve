"""Static CUDA graph shape keys for unified forward execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....contracts.forward_batch import ForwardBatch
from ....contracts.forward_mode import ForwardMode
from ..plan import ForwardPlan

__all__ = ["ForwardGraphShapeKey", "graph_shape_key"]


@dataclass(frozen=True)
class ForwardGraphShapeKey:
    program: str
    mode: ForwardMode
    op_modes: tuple[ForwardMode, ...]
    token_bucket: int
    row_bucket: int
    segment_geometry: tuple[tuple[int, str, str, int], ...]
    kv_geometry: tuple[Any, ...]
    tensor_geometry: tuple[Any, ...]
    backend: str | None = None
    descriptor_variant: str | None = None


def graph_shape_key(
    *,
    program: str,
    batch: ForwardBatch,
    plan: ForwardPlan,
    backend: str | None = None,
    descriptor_variant: str | None = None,
) -> ForwardGraphShapeKey:
    token_bucket = int(batch.padded_num_tokens or plan.shape.padded_token_count)
    row_bucket = max(int(plan.shape.padded_row_count), len(batch.req_ids))
    segment_geometry = tuple(
        (
            int(segment.q_len),
            str(segment.visible_policy.value),
            str(segment.segment_class.value),
            int(segment.branch_id),
        )
        for segment in plan.segments
    )
    max_block_width = 0
    block_table = getattr(batch, "block_table", None)
    if block_table is not None and getattr(block_table, "ndim", 0) >= 2:
        max_block_width = int(block_table.shape[-1])
    kv_geometry = (
        max_block_width,
        tuple(str(segment.kv_write_policy.value) for segment in plan.segments),
        int(plan.shape.branch_count),
    )
    tensor_geometry = (
        str(getattr(getattr(batch, "input_ids", None), "dtype", "")),
        str(getattr(getattr(batch, "input_ids", None), "device", getattr(batch, "device", ""))),
        tuple(getattr(getattr(batch, "input_ids", None), "shape", ())),
        tuple(getattr(getattr(batch, "is_gen", None), "shape", ())),
    )
    return ForwardGraphShapeKey(
        program=program,
        mode=plan.forward_mode,
        op_modes=plan.op_modes,
        token_bucket=token_bucket,
        row_bucket=row_bucket,
        segment_geometry=segment_geometry,
        kv_geometry=kv_geometry,
        tensor_geometry=tensor_geometry,
        backend=backend,
        descriptor_variant=descriptor_variant,
    )
