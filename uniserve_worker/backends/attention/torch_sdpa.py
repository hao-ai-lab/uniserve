"""Portable torch SDPA attention backend."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from ...forward import ForwardContext
from .base import AttentionCapabilities

__all__ = [
    'TorchSDPAAttentionBackend',
]


class TorchSDPAAttentionBackend:
    name = "torch_sdpa"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            segment_batched_cfg=True,
            mixed_mode=True,
            paged_kv=True,
            varlen_attention=True,
            varlen_paged_kv=True,
            visible_end=True,
            tree_verify=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float,
        attn_mask: torch.Tensor | None = None,
        context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del context
        if q.ndim == 3:
            return self._forward_lhd(q, k, v, causal=causal, scale=scale, attn_mask=attn_mask)
        if q.ndim == 4:
            return self._forward_bhld(q, k, v, causal=causal, scale=scale, attn_mask=attn_mask)
        raise ValueError(f"unsupported q rank {q.ndim}")

    def forward_paged(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        causal: bool,
        scale: float,
        context: ForwardContext | None = None,
    ) -> torch.Tensor:
        plan = None if context is None else context.attention
        base_lens = getattr(plan, "cache_seqlens_cpu", None)
        if base_lens is None:
            raise ValueError(
                "torch_sdpa paged decode requires a plan carrying host-known "
                "cache_seqlens_cpu instead of a device cache-length tensor"
            )
        del cache_seqlens
        lengths = tuple(int(value) for value in base_lens)
        q_rows, restore = _paged_query_rows(q, int(block_table.shape[0]))
        if len(lengths) != len(q_rows):
            raise ValueError("paged query rows do not match cache sequence lengths")
        current_k = current_v = None
        if k is not None or v is not None:
            if k is None or v is None:
                raise ValueError("paged KV update requires both key and value tensors")
            current_k = _paged_current_rows(k, tuple(row.shape[0] for row in q_rows))
            current_v = _paged_current_rows(v, tuple(row.shape[0] for row in q_rows))
            if len(current_k) != len(q_rows) or len(current_v) != len(q_rows):
                raise ValueError("current paged K/V rows do not match query rows")

        outputs: list[torch.Tensor] = []
        for index, (query, cache_len) in enumerate(zip(q_rows, lengths, strict=True)):
            if cache_len < 0:
                raise ValueError("cache sequence lengths must be non-negative")
            if current_k is not None and current_v is not None:
                key = current_k[index]
                value = current_v[index]
                if key.shape[0] != query.shape[0] or value.shape[0] != query.shape[0]:
                    raise ValueError("current paged K/V length must match its query length")
                _write_paged_row(k_cache, block_table[index], cache_len, key)
                _write_paged_row(v_cache, block_table[index], cache_len, value)
            live_len = cache_len + query.shape[0]
            keys = _read_paged_row(k_cache, block_table[index], live_len)
            values = _read_paged_row(v_cache, block_table[index], live_len)
            outputs.append(
                self._forward_lhd(
                    query,
                    keys,
                    values,
                    causal=causal,
                    scale=scale,
                    attn_mask=None,
                )
            )
        return restore(outputs)

    def forward_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        scale: float,
        block_table: torch.Tensor | None = None,
        context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del max_seqlen_q, max_seqlen_k
        if q.ndim != 3:
            raise ValueError("portable varlen attention expects packed [tokens, heads, dim] queries")
        if block_table is not None:
            # Paged varlen carries a PagedVarlenPlan whose host-mirrored per-row
            # query/key lengths give the packed offsets without reading the
            # device ``cu_seqlens`` tensors back to the host.
            plan = None if context is None else context.attention
            query_lens = getattr(plan, "query_lens_cpu", None)
            kv_lens = getattr(plan, "kv_seqlens_cpu", None)
            if query_lens is None or kv_lens is None:
                raise ValueError(
                    "torch_sdpa paged varlen requires a plan carrying host-known "
                    "query_lens_cpu and kv_seqlens_cpu"
                )
            q_offsets = _offsets_from_lengths(query_lens)
            k_offsets = _offsets_from_lengths(kv_lens)
        else:
            q_offsets = _validated_offsets(cu_seqlens_q, int(q.shape[0]), "query")
            k_offsets = _validated_offsets(cu_seqlens_k, int(k.shape[0]), "key")
        if len(q_offsets) != len(k_offsets):
            raise ValueError("query and key varlen metadata have different row counts")
        if block_table is not None and int(block_table.shape[0]) != len(q_offsets) - 1:
            raise ValueError("paged varlen row count does not match its page table")

        outputs: list[torch.Tensor] = []
        for row in range(len(q_offsets) - 1):
            query = q[q_offsets[row] : q_offsets[row + 1]]
            key_len = k_offsets[row + 1] - k_offsets[row]
            if block_table is None:
                keys = k[k_offsets[row] : k_offsets[row + 1]]
                values = v[k_offsets[row] : k_offsets[row + 1]]
            else:
                keys = _read_paged_row(k, block_table[row], key_len)
                values = _read_paged_row(v, block_table[row], key_len)
            outputs.append(
                self._forward_lhd(
                    query,
                    keys,
                    values,
                    causal=causal,
                    scale=scale,
                    attn_mask=None,
                )
            )
        return torch.cat(outputs, dim=0) if outputs else q.new_empty(q.shape)

    def forward_visible_end(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        visible_end: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        seqused_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        scale: float | None = None,
        use_prefix_bounds: bool = False,
        fully_visible: bool = False,
        context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del max_seqlen_q, max_seqlen_k, use_prefix_bounds, context
        if q.ndim == 3:
            if cu_seqlens_q is None:
                raise ValueError("packed visible-end attention requires query offsets")
            q_offsets = _validated_offsets(cu_seqlens_q, int(q.shape[0]), "query")
            q_rows = [q[q_offsets[row] : q_offsets[row + 1]] for row in range(len(q_offsets) - 1)]
            restore: Callable[[list[torch.Tensor]], torch.Tensor] = _concatenate_rows
        elif q.ndim == 4:
            q_rows = [q[row].transpose(0, 1) for row in range(int(q.shape[0]))]
            restore = _stack_attention_rows
        else:
            raise ValueError("visible-end queries must be packed or batched")
        if visible_end.ndim != 2 or int(visible_end.shape[0]) != len(q_rows):
            raise ValueError("visible-end rows do not match query rows")

        if page_table is not None:
            if seqused_k is None or int(page_table.shape[0]) != len(q_rows):
                raise ValueError("paged visible-end attention requires one key length per row")
            key_lengths = _integer_values(seqused_k, "visible-end key lengths")
            key_rows = [
                _read_paged_row(k, page_table[row], key_lengths[row]) for row in range(len(q_rows))
            ]
            value_rows = [
                _read_paged_row(v, page_table[row], key_lengths[row]) for row in range(len(q_rows))
            ]
        else:
            if cu_seqlens_k is None:
                if k.ndim != 4 or int(k.shape[0]) != len(q_rows):
                    raise ValueError("visible-end K/V rows require key offsets or a batch dimension")
                key_rows = [k[row].transpose(0, 1) for row in range(len(q_rows))]
                value_rows = [v[row].transpose(0, 1) for row in range(len(q_rows))]
            else:
                k_offsets = _validated_offsets(cu_seqlens_k, int(k.shape[0]), "key")
                if len(k_offsets) != len(q_rows) + 1:
                    raise ValueError("visible-end key offsets do not match query rows")
                key_rows = [k[k_offsets[row] : k_offsets[row + 1]] for row in range(len(q_rows))]
                value_rows = [v[k_offsets[row] : k_offsets[row + 1]] for row in range(len(q_rows))]

        outputs: list[torch.Tensor] = []
        for row, (query, keys, values) in enumerate(
            zip(q_rows, key_rows, value_rows, strict=True)
        ):
            if fully_visible:
                mask = None
            else:
                ends = visible_end[row, : query.shape[0]].to(device=q.device, dtype=torch.int64)
                if int(ends.numel()) != int(query.shape[0]):
                    raise ValueError("visible-end metadata is shorter than its query row")
                key_indexes = torch.arange(keys.shape[0], device=q.device)
                allowed = key_indexes.unsqueeze(0) < ends.unsqueeze(1)
                mask = torch.zeros(allowed.shape, device=q.device, dtype=q.dtype)
                mask.masked_fill_(~allowed, float("-inf"))
            outputs.append(
                self._forward_lhd(
                    query,
                    keys,
                    values,
                    causal=False,
                    scale=scale,
                    attn_mask=mask,
                )
            )
        return restore(outputs)

    def _forward_lhd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        lq, n_heads, _ = q.shape
        lk, _, _ = k.shape
        k, v = _expand_gqa(k, v, n_heads, head_axis=1)
        q4 = q.permute(1, 0, 2).unsqueeze(0)
        k4 = k.permute(1, 0, 2).unsqueeze(0)
        v4 = v.permute(1, 0, 2).unsqueeze(0)
        mask = attn_mask
        use_is_causal = False
        if mask is None and causal:
            cache_len = lk - lq
            if cache_len == 0:
                # No KV-cache offset: top-left causal masking is exactly what
                # SDPA's is_causal flag expresses, so let it use the fused
                # causal kernel instead of materializing a dense [Lq, Lk] mask.
                use_is_causal = True
            else:
                iq = torch.arange(lq, device=q.device).unsqueeze(1)
                ik = torch.arange(lk, device=q.device).unsqueeze(0)
                bad = ik > (cache_len + iq)
                mask = torch.zeros(lq, lk, dtype=q.dtype, device=q.device)
                mask.masked_fill_(bad, float("-inf"))
        mask = _normalize_mask(mask, q)
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None]
        out = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=mask, is_causal=use_is_causal, scale=scale
        )
        return out.squeeze(0).permute(1, 0, 2)

    def _forward_bhld(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        k, v = _expand_gqa(k, v, q.shape[1], head_axis=1)
        mask = _normalize_mask(attn_mask, q)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=causal and mask is None,
            scale=scale,
        )
        return out


def _expand_gqa(
    k: torch.Tensor,
    v: torch.Tensor,
    n_heads: int,
    *,
    head_axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat-interleave the KV head groups to match ``n_heads`` query heads.

    The shared grouped-query expansion for both the packed ``[L, H, D]`` and
    batched ``[B, H, L, D]`` paths. ``k``/``v`` are returned unchanged when the
    head counts already match.
    """
    n_kv_heads = int(k.shape[head_axis])
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"num heads {n_heads} is not divisible by kv heads {n_kv_heads}")
    if n_heads == n_kv_heads:
        return k, v
    rep = n_heads // n_kv_heads
    return k.repeat_interleave(rep, dim=head_axis), v.repeat_interleave(rep, dim=head_axis)


def _normalize_mask(mask: torch.Tensor | None, q: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.device != q.device:
        mask = mask.to(device=q.device)
    if mask.dtype.is_floating_point and mask.dtype != q.dtype:
        mask = mask.to(dtype=q.dtype)
    return mask


def _integer_values(value: torch.Tensor, name: str) -> tuple[int, ...]:
    if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    return tuple(int(item) for item in value.detach().to(device="cpu").tolist())


def _offsets_from_lengths(lengths: Sequence[int]) -> tuple[int, ...]:
    """Cumulative packed offsets ``[0, l0, l0+l1, ...]`` from host segment lengths."""

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return tuple(offsets)


def _validated_offsets(
    value: torch.Tensor,
    terminal: int | None,
    name: str,
) -> tuple[int, ...]:
    offsets = _integer_values(value, f"{name} offsets")
    if len(offsets) < 2 or offsets[0] != 0 or any(
        right < left for left, right in zip(offsets, offsets[1:])
    ):
        raise ValueError(f"{name} offsets are invalid")
    if terminal is not None and offsets[-1] != terminal:
        raise ValueError(f"{name} offsets do not span their tensor")
    return offsets


def _paged_query_rows(
    q: torch.Tensor,
    row_count: int,
) -> tuple[list[torch.Tensor], Callable[[list[torch.Tensor]], torch.Tensor]]:
    if q.ndim == 4:
        if int(q.shape[0]) != row_count:
            raise ValueError("paged query batch does not match its page table")
        return (
            [q[row].transpose(0, 1) for row in range(row_count)],
            _stack_attention_rows,
        )
    if q.ndim != 3:
        raise ValueError("paged queries must be [rows, heads, dim] or [batch, heads, tokens, dim]")
    if int(q.shape[0]) == row_count:
        return [q[row : row + 1] for row in range(row_count)], _concatenate_rows
    if row_count == 1:
        return [q], _first_row
    raise ValueError("packed paged queries require explicit varlen metadata")


def _concatenate_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(rows, dim=0)


def _stack_attention_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(rows, dim=0).transpose(1, 2).contiguous()


def _first_row(rows: list[torch.Tensor]) -> torch.Tensor:
    return rows[0]


def _paged_current_rows(value: torch.Tensor, query_lens: tuple[int, ...]) -> list[torch.Tensor]:
    row_count = len(query_lens)
    if value.ndim == 4:
        if int(value.shape[0]) != row_count:
            raise ValueError("current paged K/V batch does not match queries")
        rows = [value[row].transpose(0, 1) for row in range(row_count)]
    elif value.ndim == 3 and int(value.shape[0]) == row_count and all(
        length == 1 for length in query_lens
    ):
        rows = [value[row : row + 1] for row in range(row_count)]
    elif value.ndim == 3 and row_count == 1:
        rows = [value]
    else:
        raise ValueError("current paged K/V layout does not match queries")
    return rows


def _read_paged_row(cache: torch.Tensor, pages: torch.Tensor, length: int) -> torch.Tensor:
    if cache.ndim != 4 or length < 0:
        raise ValueError("paged cache must be [pages, page, heads, dim]")
    if length == 0:
        return cache.new_empty((0, cache.shape[2], cache.shape[3]))
    page_size = int(cache.shape[1])
    page_count = (length + page_size - 1) // page_size
    page_ids = pages[:page_count].to(device=cache.device, dtype=torch.int64)
    if int(page_ids.numel()) != page_count:
        raise ValueError("page table does not cover the requested cache length")
    # Physical page ids are validated host-side when the page table is
    # registered, so no on-device bound check is taken here; an out-of-range id
    # surfaces as a CUDA index error from ``index_select`` (mirroring the
    # fa4_cute append path) rather than a per-read device-to-host sync.
    return cache.index_select(0, page_ids).reshape(-1, cache.shape[2], cache.shape[3])[:length]


def _write_paged_row(
    cache: torch.Tensor,
    pages: torch.Tensor,
    start: int,
    values: torch.Tensor,
) -> None:
    if values.ndim != 3 or values.shape[1:] != cache.shape[2:]:
        raise ValueError("paged cache write geometry does not match cache storage")
    count = int(values.shape[0])
    if count == 0:
        return
    page_size = int(cache.shape[1])
    if start + count > int(pages.numel()) * page_size:
        raise ValueError("page table does not cover the paged cache write")
    # Vectorized on-device scatter: destination pages and offsets are derived
    # from the host-known ``start``/``count`` and gathered from the page table,
    # so the write incurs no per-token ``.item()`` sync. Physical page ids are
    # host-validated at registration (see ``_read_paged_row``).
    positions = start + torch.arange(count, device=cache.device, dtype=torch.int64)
    logical = positions // page_size
    offsets = positions % page_size
    page_ids = pages.to(device=cache.device, dtype=torch.int64).index_select(0, logical)
    flat = page_ids * page_size + offsets
    source = values if values.dtype == cache.dtype else values.to(cache.dtype)
    cache.view(-1, cache.shape[2], cache.shape[3]).index_copy_(0, flat, source.contiguous())
