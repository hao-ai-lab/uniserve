"""Reference attention provider for the packed cache ABI (dormant).

First slice of the "Canonical attention" work package from
``specs/unified_kv_attention_runtime.md``: one provider implementation that
consumes the canonical `SegmentTable` and packed `ResidencyBatch` truth,
owns every current K/V write through the reservation's write locations, and
covers the causal-prefix and full-query-prefix patterns numerically.

Two owners live here:

* :class:`CacheDeviceBinding` — the residency-owned device KV storage for
  one cache domain: page-granular tensors plus a write primitive. It exposes
  no allocator, release, commit, or page-ownership operation (Law: the
  backend can read and write reserved bytes; it cannot change ownership).
  In the dormant slice the binding is constructed standalone; at cutover
  residency injects it during engine construction.
* :class:`ReferenceAttentionBackend` — the two-method provider
  (``prepare``/``forward``). It is a *numerical reference*: unbatched
  per-region SDPA over gathered pages, meant to anchor conformance goldens
  and prove the ABI, not to be fast. A production provider lowers the same
  canonical inputs into fused paged kernels behind the same interface.

Family code never sees this module: models hand Q/K/V to the shared
attention layer, and the backend owns cache writes and kernels. Nothing in
production routes through here until the vertical slice activates.
"""
from __future__ import annotations

import torch

from ...contracts.cache_schema import AttentionPattern
from ...contracts.residency_batch import ResidencyBatchArrays
from ...contracts.segment_table import AttentionLayerSpec, SegmentTableArrays

__all__ = [
    "AttentionLayerSpec",
    "CacheDeviceBinding",
    "ReferenceAttentionBackend",
    "ReferenceAttentionError",
]


class ReferenceAttentionError(RuntimeError):
    """The provider received inputs outside its proven coverage."""


class CacheDeviceBinding:
    """Residency-owned page storage for one domain, one layer set.

    Layout is ``[layer, page, 2 (K/V), page_tokens, kv_heads, head_dim]`` —
    per layer exactly FlashInfer's NHD paged KV cache, so fused providers
    consume the same storage the reference provider proves. Page ``0`` is the
    sink page; writes there are legal and meaningless.
    """

    def __init__(
        self,
        *,
        layers: int,
        pages: int,
        page_tokens: int,
        kv_heads: int,
        head_dim: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.page_tokens = page_tokens
        self.storage = torch.zeros(
            (layers, pages, 2, page_tokens, kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

    def write(
        self,
        layer_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        residency: ResidencyBatchArrays,
    ) -> None:
        """Write current K/V into the reservation-owned locations (once)."""

        page_ids = []
        offsets = []
        slots = []
        for token, active in enumerate(residency.write_active):
            if active:
                page_ids.append(residency.write_page_ids[token])
                offsets.append(residency.write_page_offsets[token])
                slots.append(token)
        if not slots:
            return
        device = self.storage.device
        page_index = torch.tensor(page_ids, device=device, dtype=torch.long)
        offset_index = torch.tensor(offsets, device=device, dtype=torch.long)
        slot_index = torch.tensor(slots, device=device, dtype=torch.long)
        self.storage[layer_id, page_index, 0, offset_index] = keys[slot_index].to(
            self.storage.dtype
        )
        self.storage[layer_id, page_index, 1, offset_index] = values[
            slot_index
        ].to(self.storage.dtype)

    def gather_rows(
        self,
        layer_id: int,
        residency: ResidencyBatchArrays,
        binding_index: int,
        rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one binding's first ``rows`` physical rows in logical order."""

        begin = residency.binding_page_indptr[binding_index]
        end = residency.binding_page_indptr[binding_index + 1]
        chain = list(residency.page_ids[begin:end])
        needed_pages = -(-rows // self.page_tokens) if rows else 0
        if needed_pages > len(chain):
            raise ReferenceAttentionError(
                f"binding {binding_index} exposes {len(chain)} pages; "
                f"{rows} rows need {needed_pages}"
            )
        if rows == 0:
            empty = self.storage.new_zeros(
                (0, self.storage.shape[4], self.storage.shape[5])
            )
            return empty, empty.clone()
        page_index = torch.tensor(
            chain[:needed_pages], device=self.storage.device, dtype=torch.long
        )
        gathered = (
            self.storage[layer_id, page_index]
            .permute(0, 2, 1, 3, 4)
            .reshape(needed_pages * self.page_tokens, 2, *self.storage.shape[4:])
        )[:rows]
        return gathered[:, 0].contiguous(), gathered[:, 1].contiguous()


class ReferenceAttentionBackend:
    """Two-method provider over the canonical tables (numerical reference)."""

    def __init__(self, binding: CacheDeviceBinding) -> None:
        self._binding = binding
        self._segments: SegmentTableArrays | None = None
        self._residency: ResidencyBatchArrays | None = None

    def prepare(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
    ) -> None:
        """Bounded validation and metadata refresh; no allocation, no search."""

        for index, active in enumerate(segments.segment_active):
            if not active:
                continue
            pattern = segments.attention_pattern[index]
            if pattern not in (
                int(AttentionPattern.CAUSAL_PREFIX),
                int(AttentionPattern.FULL_QUERY_PREFIX),
            ):
                raise ReferenceAttentionError(
                    "this provider covers causal-prefix and full-query-prefix "
                    "regions only"
                )
        self._segments = segments
        self._residency = residency

    def forward(
        self,
        layer: AttentionLayerSpec,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Write current K/V, then run every declared attention region.

        ``q``: ``[tokens, query_heads, qk_head_dim]``; ``k``/``v``:
        ``[tokens, kv_heads, head_dim]`` in canonical packed token order.
        """

        segments = self._segments
        residency = self._residency
        if segments is None or residency is None:
            raise ReferenceAttentionError("prepare() must precede forward()")
        self._binding.write(layer.layer_id, k, v, residency)
        output = q.new_zeros((q.shape[0], layer.query_heads, layer.value_head_dim))
        for begin, count, region_indices in self._regions(segments):
            self._run_region(
                layer, q, k, v, output, segments, residency,
                begin, count, region_indices,
            )
        return output

    # ------------------------------------------------------------------ #

    @staticmethod
    def _regions(segments: SegmentTableArrays):
        """Yield ``(query_begin, query_count, segment_indices)`` per region."""

        by_region: dict[int, list[int]] = {}
        for index, active in enumerate(segments.segment_active):
            if active:
                by_region.setdefault(
                    segments.attention_region_id[index], []
                ).append(index)
        for indices in by_region.values():
            begin = segments.query_begin[indices[0]]
            count = sum(segments.query_count[i] for i in indices)
            yield begin, count, indices

    def _run_region(
        self,
        layer: AttentionLayerSpec,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
        begin: int,
        count: int,
        region_indices: list[int],
    ) -> None:
        first = region_indices[0]
        pattern = segments.attention_pattern[first]
        context = segments.context_length[first]
        binding_index = segments.kv_read_index[first]
        if segments.kv_group[first] == 0:
            prefix_k = k.new_zeros((0, layer.kv_heads, layer.qk_head_dim))
            prefix_v = v.new_zeros((0, layer.kv_heads, layer.value_head_dim))
        else:
            prefix_k, prefix_v = self._binding.gather_rows(
                layer.layer_id, residency, binding_index, context
            )
            prefix_k = prefix_k.to(q.dtype)
            prefix_v = prefix_v.to(q.dtype)
        span = slice(begin, begin + count)
        keys = torch.cat([prefix_k, k[span]], dim=0)
        values = torch.cat([prefix_v, v[span]], dim=0)
        queries = q[span]
        # Grouped-query expansion for the reference path.
        group = layer.query_heads // layer.kv_heads
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        scores = (
            torch.einsum("qhd,khd->hqk", queries.float(), keys.float())
            * layer.scale
        )
        total = context + count
        if pattern == int(AttentionPattern.CAUSAL_PREFIX):
            key_positions = torch.arange(total, device=q.device)
            query_positions = context + torch.arange(count, device=q.device)
            mask = key_positions[None, :] > query_positions[:, None]
            scores = scores.masked_fill(mask[None, :, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        region_output = torch.einsum(
            "hqk,khd->qhd", weights, values.float()
        ).to(q.dtype)
        output[span] = region_output
