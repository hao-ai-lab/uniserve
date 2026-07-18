"""Packed page-reference ABI for the target ``ResidencyBatch``.

Device-contract companion to :mod:`uniserve_worker.contracts.cache_schema`
(``specs/unified_kv_attention_runtime.md`` — "Packed Page References"). The
canonical residency mapping is a packed set of logical page references, not a
persistent maximum-context rectangular table:

* binding ``0`` is the sink binding; real active bindings occupy
  ``1 : active_binding_count + 1``;
* ``binding_page_indptr[b] : binding_page_indptr[b + 1]`` selects binding
  ``b``'s pages in logical order;
* page identifier ``0`` names the residency-owned sink page of the binding's
  domain;
* per-token write locations align with canonical packed query-token indices.

This module owns the capacity math and the structural invariants so graph
buckets, residency, and providers validate against one description. It is
torch-free: the checks run over any integer sequences (host lists, numpy, or
tensor ``.tolist()`` views) and therefore run in contract tests without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "ResidencyBatchArrays",
    "ResidencyBatchCapacity",
    "ResidencyBatchError",
    "SINK_BINDING",
    "SINK_PAGE_ID",
]

SINK_BINDING = 0
SINK_PAGE_ID = 0


class ResidencyBatchError(ValueError):
    """A packed residency mapping violates the device ABI."""


@dataclass(frozen=True, slots=True)
class ResidencyBatchCapacity:
    """Graph-capacity axes for one bucket's packed cache metadata.

    ``bindings`` is the maximum number of real cache bindings (``B``, excluding
    the sink binding), ``page_references`` the packed page-reference capacity
    (``P``), and ``tokens`` the packed query-token capacity (``T``).
    """

    bindings: int
    page_references: int
    tokens: int

    def __post_init__(self) -> None:
        if self.bindings < 0 or self.page_references < 0 or self.tokens < 0:
            raise ResidencyBatchError("capacity axes must be non-negative")

    def canonical_metadata_bytes(self) -> int:
        """Exact canonical cache-tensor bytes before allocator alignment.

        ``4 * (4B + P + 2T + 7) + B + T + 1``: the int32 tensors
        (``binding_domain_id``, ``binding_committed_rows``,
        ``binding_provisional_rows`` at ``B + 1``, ``binding_page_indptr`` at
        ``B + 2``, ``page_ids`` at ``P``, ``write_page_ids`` and
        ``write_page_offsets`` at ``T``, and the two active counts) plus the
        uint8 ``binding_active`` (``B + 1``) and ``write_active`` (``T``).
        """

        b, p, t = self.bindings, self.page_references, self.tokens
        return 4 * (4 * b + p + 2 * t + 7) + b + t + 1


@dataclass(frozen=True, slots=True)
class ResidencyBatchArrays:
    """One populated packed mapping, expressed as host integer sequences."""

    active_binding_count: int
    active_page_reference_count: int
    binding_active: Sequence[int]
    binding_domain_id: Sequence[int]
    binding_committed_rows: Sequence[int]
    binding_provisional_rows: Sequence[int]
    binding_page_indptr: Sequence[int]
    page_ids: Sequence[int]
    write_page_ids: Sequence[int]
    write_page_offsets: Sequence[int]
    write_active: Sequence[int]

    def validate(self, capacity: ResidencyBatchCapacity, *, page_tokens: int) -> None:
        """Prove the packed-mapping invariants for one transaction.

        Structural rules only; content rules that need reservation state
        (page-ownership, disjoint provisional spans, domain-scoped uniqueness
        across bindings sharing leases) belong to residency.
        """

        self._validate_shapes(capacity)
        self._validate_counts(capacity)
        self._validate_sink_binding()
        self._validate_bindings(capacity)
        self._validate_write_locations(capacity, page_tokens=page_tokens)

    def _validate_shapes(self, capacity: ResidencyBatchCapacity) -> None:
        b, p, t = capacity.bindings, capacity.page_references, capacity.tokens
        expected = {
            "binding_active": (self.binding_active, b + 1),
            "binding_domain_id": (self.binding_domain_id, b + 1),
            "binding_committed_rows": (self.binding_committed_rows, b + 1),
            "binding_provisional_rows": (self.binding_provisional_rows, b + 1),
            "binding_page_indptr": (self.binding_page_indptr, b + 2),
            "page_ids": (self.page_ids, p),
            "write_page_ids": (self.write_page_ids, t),
            "write_page_offsets": (self.write_page_offsets, t),
            "write_active": (self.write_active, t),
        }
        for name, (array, size) in expected.items():
            if len(array) != size:
                raise ResidencyBatchError(
                    f"{name} has {len(array)} entries; capacity requires {size}"
                )

    def _validate_counts(self, capacity: ResidencyBatchCapacity) -> None:
        if not 0 <= self.active_binding_count <= capacity.bindings:
            raise ResidencyBatchError(
                f"active_binding_count {self.active_binding_count} exceeds "
                f"binding capacity {capacity.bindings}"
            )
        if not 0 <= self.active_page_reference_count <= capacity.page_references:
            raise ResidencyBatchError(
                f"active_page_reference_count {self.active_page_reference_count} "
                f"exceeds page-reference capacity {capacity.page_references}"
            )

    def _validate_sink_binding(self) -> None:
        if (
            self.binding_active[SINK_BINDING] != 0
            or self.binding_domain_id[SINK_BINDING] != 0
            or self.binding_committed_rows[SINK_BINDING] != 0
            or self.binding_provisional_rows[SINK_BINDING] != 0
        ):
            raise ResidencyBatchError("sink binding must stay inactive and zeroed")
        if self.binding_page_indptr[0] != 0 or self.binding_page_indptr[1] != 0:
            raise ResidencyBatchError("sink binding owns no page references")

    def _validate_bindings(self, capacity: ResidencyBatchCapacity) -> None:
        active = self.active_binding_count
        previous = 0
        for binding in range(1, capacity.bindings + 1):
            begin = self.binding_page_indptr[binding]
            end = self.binding_page_indptr[binding + 1]
            if binding <= active:
                if self.binding_active[binding] != 1:
                    raise ResidencyBatchError(
                        f"binding {binding} within active prefix is not active"
                    )
                if self.binding_domain_id[binding] <= 0:
                    raise ResidencyBatchError(
                        f"active binding {binding} needs a positive cache domain"
                    )
                committed = self.binding_committed_rows[binding]
                provisional = self.binding_provisional_rows[binding]
                if committed < 0 or provisional < committed:
                    raise ResidencyBatchError(
                        f"active binding {binding} rows are inconsistent: "
                        f"committed={committed} provisional={provisional}"
                    )
                if begin != previous or end < begin:
                    raise ResidencyBatchError(
                        f"active binding {binding} page indptr is not monotonic"
                    )
                previous = end
            else:
                if self.binding_active[binding] != 0:
                    raise ResidencyBatchError(
                        f"binding {binding} beyond the active prefix is active"
                    )
                if (
                    self.binding_domain_id[binding] != 0
                    or self.binding_committed_rows[binding] != 0
                    or self.binding_provisional_rows[binding] != 0
                ):
                    raise ResidencyBatchError(
                        f"inactive binding {binding} must be zeroed"
                    )
                if begin != self.active_page_reference_count or end != begin:
                    raise ResidencyBatchError(
                        f"inactive binding {binding} indptr must pin to the "
                        "active page-reference count"
                    )
        if active and previous != self.active_page_reference_count:
            raise ResidencyBatchError(
                "final active binding indptr must equal active_page_reference_count"
            )
        for index in range(self.active_page_reference_count):
            if self.page_ids[index] < 0:
                raise ResidencyBatchError(f"page reference {index} is negative")
        for index in range(self.active_page_reference_count, capacity.page_references):
            if self.page_ids[index] != SINK_PAGE_ID:
                raise ResidencyBatchError(
                    f"inactive page reference {index} must name the sink page"
                )

    def _validate_write_locations(
        self,
        capacity: ResidencyBatchCapacity,
        *,
        page_tokens: int,
    ) -> None:
        if page_tokens <= 0:
            raise ResidencyBatchError("page_tokens must be positive")
        for token in range(capacity.tokens):
            active = self.write_active[token]
            if active not in (0, 1):
                raise ResidencyBatchError(f"write_active[{token}] must be 0 or 1")
            if active:
                offset = self.write_page_offsets[token]
                if not 0 <= offset < page_tokens:
                    raise ResidencyBatchError(
                        f"write offset {offset} at token {token} exceeds the "
                        f"page geometry {page_tokens}"
                    )
                if self.write_page_ids[token] < 0:
                    raise ResidencyBatchError(
                        f"active write at token {token} names a negative page"
                    )
            else:
                if (
                    self.write_page_ids[token] != SINK_PAGE_ID
                    or self.write_page_offsets[token] != 0
                ):
                    raise ResidencyBatchError(
                        f"inactive write slot {token} must name the sink page"
                    )
