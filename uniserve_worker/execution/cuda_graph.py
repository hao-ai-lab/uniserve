"""Capacity-only CUDA graph runtime: buckets, capture, refresh, replay.

Dormant Stage 7 deliverable from ``specs/unified_forward_execution.md``.
Graph identity is capacity identity: a finite ordered set of
:class:`~uniserve_worker.contracts.segment_table.GraphCapacity` buckets is
allocated and captured once at startup, and production execution can only
refresh and replay an existing bucket. No graph key contains operation,
phase, route, request, overlay, or composition values, and nothing may
capture, compile, or allocate persistent storage after readiness.

Each bucket owns pointer-stable device columns for the closed
``SegmentTable`` schema, the packed ``ResidencyBatch`` mapping, and a fixed
result slot per row. Because the torch-free host arrays already carry their
zero-sentinel inactive tails at full capacity, one whole-column copy per
transaction is simultaneously the active refresh and the neutral tail fill —
stale contents cannot survive a refresh, which the maximal/minimal
alternation test proves.

The adapter captured here is whatever resident callable the engine binds
(tests capture a deterministic structural reduction; the family root arrives
with Stage 6). Nothing in production routes through this module until the
vertical slice activates.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

import torch

from ..contracts.residency_batch import ResidencyBatchArrays
from ..contracts.segment_table import GraphCapacity, SegmentTableArrays

__all__ = [
    "BucketTensors",
    "CudaGraphError",
    "CudaGraphRuntime",
    "GraphReplayView",
]


class CudaGraphError(RuntimeError):
    """A graph-runtime law was violated (capacity miss, pointer drift, ...)."""


@dataclass(frozen=True)
class BucketTensors:
    """Pointer-stable device columns owned by one captured bucket."""

    # SegmentTable columns (int32, capacity `segments`).
    segment_active: torch.Tensor
    row_id: torch.Tensor
    operation_tag: torch.Tensor
    route_id: torch.Tensor
    local_segment_id: torch.Tensor
    token_begin: torch.Tensor
    token_count: torch.Tensor
    query_begin: torch.Tensor
    query_count: torch.Tensor
    context_length: torch.Tensor
    position_begin: torch.Tensor
    position_count: torch.Tensor
    branch_id: torch.Tensor
    branch_count: torch.Tensor
    attention_pattern: torch.Tensor
    attention_region_id: torch.Tensor
    kv_group: torch.Tensor
    kv_read_index: torch.Tensor
    kv_write_index: torch.Tensor
    cache_effect: torch.Tensor
    cache_write_count: torch.Tensor
    input_product_index: torch.Tensor
    output_product_index: torch.Tensor
    overlay_slot: torch.Tensor
    candidate_begin: torch.Tensor
    candidate_count: torch.Tensor
    result_slot: torch.Tensor
    # ResidencyBatch columns.
    binding_active: torch.Tensor
    binding_domain_id: torch.Tensor
    binding_committed_rows: torch.Tensor
    binding_provisional_rows: torch.Tensor
    binding_page_indptr: torch.Tensor
    page_ids: torch.Tensor
    write_page_ids: torch.Tensor
    write_page_offsets: torch.Tensor
    write_active: torch.Tensor
    # Fixed result slots (int64, capacity `rows`).
    row_results: torch.Tensor


@dataclass(frozen=True)
class GraphReplayView:
    """Graph-owned result view; copy out before the next replay."""

    bucket_index: int
    row_results: torch.Tensor


class _Bucket:
    def __init__(self, capacity: GraphCapacity, device: torch.device) -> None:
        self.capacity = capacity
        segment_columns = {
            name: torch.zeros(capacity.segments, dtype=torch.int32, device=device)
            for name in SegmentTableArrays.__dataclass_fields__
        }
        residency = capacity.residency
        residency_columns = {
            "binding_active": torch.zeros(
                residency.bindings + 1, dtype=torch.uint8, device=device
            ),
            "binding_domain_id": torch.zeros(
                residency.bindings + 1, dtype=torch.int32, device=device
            ),
            "binding_committed_rows": torch.zeros(
                residency.bindings + 1, dtype=torch.int32, device=device
            ),
            "binding_provisional_rows": torch.zeros(
                residency.bindings + 1, dtype=torch.int32, device=device
            ),
            "binding_page_indptr": torch.zeros(
                residency.bindings + 2, dtype=torch.int32, device=device
            ),
            "page_ids": torch.zeros(
                residency.page_references, dtype=torch.int32, device=device
            ),
            "write_page_ids": torch.zeros(
                residency.tokens, dtype=torch.int32, device=device
            ),
            "write_page_offsets": torch.zeros(
                residency.tokens, dtype=torch.int32, device=device
            ),
            "write_active": torch.zeros(
                residency.tokens, dtype=torch.uint8, device=device
            ),
        }
        self.tensors = BucketTensors(
            **segment_columns,
            **residency_columns,
            row_results=torch.zeros(
                max(capacity.rows, 1), dtype=torch.int64, device=device
            ),
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.replay_count = 0
        self._pointers = tuple(
            getattr(self.tensors, field.name).data_ptr()
            for field in fields(BucketTensors)
        )

    def assert_pointer_stability(self) -> None:
        current = tuple(
            getattr(self.tensors, field.name).data_ptr()
            for field in fields(BucketTensors)
        )
        if current != self._pointers:
            raise CudaGraphError("bucket tensor pointers drifted after capture")

    def refresh(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
    ) -> None:
        # The host arrays carry zero-sentinel tails at full capacity, so one
        # whole-column copy is both the active refresh and the neutral fill.
        for name in SegmentTableArrays.__dataclass_fields__:
            column = getattr(self.tensors, name)
            column.copy_(
                torch.as_tensor(
                    list(getattr(segments, name)), dtype=column.dtype
                )
            )
        for name in (
            "binding_active",
            "binding_domain_id",
            "binding_committed_rows",
            "binding_provisional_rows",
            "binding_page_indptr",
            "page_ids",
            "write_page_ids",
            "write_page_offsets",
            "write_active",
        ):
            column = getattr(self.tensors, name)
            column.copy_(
                torch.as_tensor(
                    list(getattr(residency, name)), dtype=column.dtype
                )
            )


class CudaGraphRuntime:
    """Finite ordered capacities, captured once, replayed forever."""

    def __init__(
        self,
        capacities: tuple[GraphCapacity, ...],
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        if not capacities:
            raise CudaGraphError("a graph runtime needs at least one capacity")
        self.device = torch.device(device)
        self._buckets = [
            _Bucket(capacity, self.device) for capacity in capacities
        ]
        self._ready = False
        self.capture_count = 0

    def capture_all(
        self,
        adapter_fn: Callable[[BucketTensors], None],
    ) -> None:
        """Warm and capture every configured bucket exactly once."""

        if self._ready:
            raise CudaGraphError("capture after readiness is prohibited")
        for bucket in self._buckets:
            # Warmup on a side stream, then capture one adapter invocation.
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                adapter_fn(bucket.tensors)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                adapter_fn(bucket.tensors)
            bucket.graph = graph
            bucket.assert_pointer_stability()
            self.capture_count += 1
        self._ready = True

    def execute(
        self,
        demand: GraphCapacity,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
    ) -> GraphReplayView:
        """Select the smallest dominating bucket, refresh, replay once."""

        if not self._ready:
            raise CudaGraphError("the runtime is not ready")
        bucket_index = self._select(demand)
        bucket = self._buckets[bucket_index]
        bucket.refresh(segments, residency)
        allocated_before = torch.cuda.memory_allocated(self.device)
        assert bucket.graph is not None
        bucket.graph.replay()
        torch.cuda.synchronize(self.device)
        if torch.cuda.memory_allocated(self.device) != allocated_before:
            raise CudaGraphError("replay changed persistent device allocation")
        bucket.assert_pointer_stability()
        bucket.replay_count += 1
        return GraphReplayView(
            bucket_index=bucket_index,
            row_results=bucket.tensors.row_results,
        )

    def replay_counts(self) -> tuple[int, ...]:
        return tuple(bucket.replay_count for bucket in self._buckets)

    def _select(self, demand: GraphCapacity) -> int:
        candidates = [
            index
            for index, bucket in enumerate(self._buckets)
            if bucket.capacity.dominates(demand)
        ]
        if not candidates:
            raise CudaGraphError(
                "no configured graph capacity dominates the demand"
            )
        return min(
            candidates,
            key=lambda index: (
                self._buckets[index].capacity.tokens,
                self._buckets[index].capacity.segments,
                self._buckets[index].capacity.rows,
            ),
        )
