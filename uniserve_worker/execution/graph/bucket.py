"""Finite capacity identities and bucket selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uniserve_worker.contracts.forward_batch import ForwardBatch, ForwardPlan
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.foundation.runtime_config import (
    DEFAULT_DECODE_GRAPH_BATCH_SIZES,
    DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
)

STEP_SIZES = DEFAULT_DECODE_GRAPH_BATCH_SIZES
SPAN_SIZES = DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS


@dataclass(frozen=True, slots=True)
class Capacity:
    """Stable physical layout plus finite capacity axes.

    Request ids, token values, positions, and operation composition are dynamic
    inputs and therefore never participate in this identity.
    """

    path: str
    rows: int
    segments: int
    tokens: int
    branches: int
    blocks: int
    dtype: str
    device: str
    backend: str | None = None
    descriptor: str | None = None


def key(
    *,
    path: str,
    batch: ForwardBatch,
    plan: ForwardPlan,
    backend: str | None = None,
    descriptor: str | None = None,
) -> Capacity:
    """Build the finite identity for a graph execution path."""

    input_tensor = getattr(batch, "input_ids", None)
    block_table = getattr(batch, "block_table", None)
    width = 0
    if block_table is not None and getattr(block_table, "ndim", 0) >= 2:
        width = int(block_table.shape[-1])
    return Capacity(
        path=str(path),
        rows=ceil(max(int(plan.shape.padded_row_count), len(batch.req_ids))),
        segments=ceil(len(plan.segments)),
        tokens=ceil(int(batch.padded_num_tokens or plan.shape.padded_token_count)),
        branches=ceil(int(plan.shape.branch_count)),
        blocks=ceil(width),
        dtype=str(getattr(input_tensor, "dtype", "")),
        device=str(getattr(input_tensor, "device", getattr(batch, "device", ""))),
        backend=backend,
        descriptor=descriptor,
    )


def ceil(value: int) -> int:
    """Round a non-negative capacity to its power-of-two bucket."""

    value = max(0, int(value))
    return 0 if value == 0 else 1 << (value - 1).bit_length()


def padding_blocks(pool: Any) -> tuple[int, ...]:
    """Validate and return the pool's reserved graph-padding blocks."""

    block_ids = tuple(int(block_id) for block_id in getattr(pool, "reserved_block_ids", ()))
    if len(set(block_ids)) != len(block_ids):
        raise invalid_descriptor("graph padding block ids must be unique")
    num_blocks = int(getattr(pool, "num_blocks", 0) or 0)
    if any(block_id < 0 or block_id >= num_blocks for block_id in block_ids):
        raise invalid_descriptor("graph padding block id is out of range")
    return block_ids
