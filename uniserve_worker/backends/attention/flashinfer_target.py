"""FlashInfer provider for the packed cache ABI (production kernel path).

The fused counterpart of
:mod:`uniserve_worker.backends.attention.reference_target`, consuming the
same canonical inputs — `SegmentTable` regions, packed `ResidencyBatch` page
chains, and the FlashInfer-layout `CacheDeviceBinding` storage — through
`BatchPrefillWithPagedKVCacheWrapper` kernels. This is the component that
carries the target stack to production speed: numerical parity against the
reference provider is the conformance obligation, and both providers stand
behind the same two-method interface.

Lowering per the companion's provider ABI:

* regions group by attention pattern into one causal plan (``CAUSAL_PREFIX``:
  queries are the trailing positions of their kv extent, FlashInfer's causal
  alignment) and one full plan (``FULL_QUERY_PREFIX``: ``causal=False``);
* each region contributes ``qo_indptr`` from its packed query span and
  ``paged_kv_indptr/indices/last_page_len`` from its binding's page chain
  covering ``context_length + query_count`` physical rows;
* ``prepare`` plans both wrappers once per transaction; ``forward`` scatters
  the layer's current K/V into the reservation-owned page locations and runs
  the planned kernels, scattering outputs back to canonical token order.

Named residency follow-up: a transient overlay whose committed prefix ends
mid-page leaves a physical hole the paged kernel cannot express; such
regions require the private-tail continuation clone in residency before this
provider accepts them (typed error today, never a silent fallback).
"""
from __future__ import annotations

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from ...contracts.cache_schema import AttentionPattern
from ...contracts.residency_batch import ResidencyBatchArrays
from ...contracts.segment_table import AttentionLayerSpec, SegmentTableArrays
from .reference_target import CacheDeviceBinding, ReferenceAttentionError

__all__ = ["FlashInferTargetBackend"]

_WORKSPACE_BYTES = 128 * 1024 * 1024


class _RegionPlan:
    __slots__ = ("query_begin", "query_count")

    def __init__(self, query_begin: int, query_count: int) -> None:
        self.query_begin = query_begin
        self.query_count = query_count


class FlashInferTargetBackend:
    """Two-method fused provider over the packed ABI.

    One provider instance is statically bound to one attention-site geometry
    before any planning, per the companion's static-binding law.
    """

    def __init__(self, binding: CacheDeviceBinding, site: AttentionLayerSpec) -> None:
        self._binding = binding
        self._num_qo_heads = site.query_heads
        self._num_kv_heads = site.kv_heads
        self._head_dim = site.qk_head_dim
        self._sm_scale = site.scale
        device = binding.storage.device
        self._causal_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device),
            kv_layout="NHD",
        )
        self._full_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device),
            kv_layout="NHD",
        )
        self._causal_regions: list[_RegionPlan] = []
        self._full_regions: list[_RegionPlan] = []
        self._residency: ResidencyBatchArrays | None = None

    # ------------------------------------------------------------------ #

    def prepare(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
    ) -> None:
        """Group regions by pattern and plan both kernels once."""

        self._residency = residency
        page_tokens = self._binding.page_tokens
        by_region: dict[int, list[int]] = {}
        for index, active in enumerate(segments.segment_active):
            if active:
                by_region.setdefault(
                    segments.attention_region_id[index], []
                ).append(index)
        causal: list[tuple[_RegionPlan, int, int]] = []
        full: list[tuple[_RegionPlan, int, int]] = []
        for indices in by_region.values():
            first = indices[0]
            begin = segments.query_begin[first]
            count = sum(segments.query_count[i] for i in indices)
            context = segments.context_length[first]
            binding_index = segments.kv_read_index[first]
            if segments.kv_group[first] == 0:
                raise ReferenceAttentionError(
                    "no-cache regions are not lowered by this provider yet"
                )
            provisional = residency.binding_provisional_rows[binding_index]
            if provisional - context > count and context % page_tokens:
                raise ReferenceAttentionError(
                    "a mid-page committed prefix under a page-aligned overlay "
                    "needs the residency continuation clone first"
                )
            pattern = segments.attention_pattern[first]
            entry = (_RegionPlan(begin, count), binding_index, context)
            if pattern == int(AttentionPattern.CAUSAL_PREFIX):
                causal.append(entry)
            elif pattern == int(AttentionPattern.FULL_QUERY_PREFIX):
                full.append(entry)
            else:
                raise ReferenceAttentionError(
                    "this provider covers causal-prefix and full-query-prefix "
                    "regions"
                )
        self._causal_regions = [entry[0] for entry in causal]
        self._full_regions = [entry[0] for entry in full]
        for wrapper, group, is_causal in (
            (self._causal_wrapper, causal, True),
            (self._full_wrapper, full, False),
        ):
            if not group:
                continue
            self._plan_group(wrapper, group, residency, is_causal)

    def _plan_group(
        self,
        wrapper: BatchPrefillWithPagedKVCacheWrapper,
        group: list[tuple[_RegionPlan, int, int]],
        residency: ResidencyBatchArrays,
        is_causal: bool,
    ) -> None:
        page_tokens = self._binding.page_tokens
        device = self._binding.storage.device
        qo_indptr = [0]
        kv_indptr = [0]
        kv_indices: list[int] = []
        last_page_len: list[int] = []
        for region, binding_index, context in group:
            # For a page-aligned overlay the current rows physically follow
            # the aligned begin; the visible kv extent is prefix + region.
            provisional = residency.binding_provisional_rows[binding_index]
            overlay_begin = provisional - region.query_count
            physical_end = (
                context + region.query_count
                if overlay_begin == context
                else provisional
            )
            begin = residency.binding_page_indptr[binding_index]
            pages_needed = -(-physical_end // page_tokens)
            chain = list(
                residency.page_ids[begin : begin + pages_needed]
            )
            qo_indptr.append(qo_indptr[-1] + region.query_count)
            kv_indptr.append(kv_indptr[-1] + pages_needed)
            kv_indices.extend(chain)
            last_page_len.append(
                physical_end - (pages_needed - 1) * page_tokens
            )
        wrapper.plan(
            torch.tensor(qo_indptr, device=device, dtype=torch.int32),
            torch.tensor(kv_indptr, device=device, dtype=torch.int32),
            torch.tensor(kv_indices, device=device, dtype=torch.int32),
            torch.tensor(last_page_len, device=device, dtype=torch.int32),
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim,
            page_tokens,
            causal=is_causal,
            sm_scale=self._sm_scale,
            q_data_type=self._binding.storage.dtype,
            kv_data_type=self._binding.storage.dtype,
        )

    # ------------------------------------------------------------------ #

    def forward(
        self,
        layer: AttentionLayerSpec,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        residency = self._residency
        if residency is None:
            raise ReferenceAttentionError("prepare() must precede forward()")
        self._binding.write(layer.layer_id, k, v, residency)
        paged = self._binding.storage[layer.layer_id]
        output = q.new_zeros((q.shape[0], layer.query_heads, layer.value_head_dim))
        for wrapper, regions in (
            (self._causal_wrapper, self._causal_regions),
            (self._full_wrapper, self._full_regions),
        ):
            if not regions:
                continue
            gathered = torch.cat(
                [
                    q[region.query_begin : region.query_begin + region.query_count]
                    for region in regions
                ],
                dim=0,
            )
            result = wrapper.run(gathered, paged)
            cursor = 0
            for region in regions:
                output[
                    region.query_begin : region.query_begin + region.query_count
                ] = result[cursor : cursor + region.query_count]
                cursor += region.query_count
        return output
