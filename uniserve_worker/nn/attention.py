"""Model-facing attention module."""
from __future__ import annotations

import enum
from typing import TypeVar

import torch
import torch.nn as nn

import uniserve_worker.ops as ops

from ..contracts.forward_context import get_forward_context
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..foundation.torch_compat import torch_is_compiling as _torch_is_compiling

__all__ = [
    'AttentionPath',
    'RadixAttention',
]

_T = TypeVar("_T")


class AttentionPath(enum.Enum):
    """The forward path selected for a single ``RadixAttention.forward`` call.

    Resolved once per call by :meth:`RadixAttention._resolve_attention_path` and
    routed to the matching ``_forward_<path>`` body. The resolution order is
    load-bearing: paged-backend selection is only performed once the varlen
    branches are ruled out, so the enum is produced by a single short-circuiting
    pass rather than independent probes.
    """

    CONTIGUOUS_VARLEN = "contiguous_varlen"
    PAGED_VARLEN = "paged_varlen"
    TRANSIENT_PAGED_VARLEN = "transient_paged_varlen"
    PAGED_EXTEND = "paged_extend"
    PAGED_DECODE = "paged_decode"
    EMPTY_PAGED_PREFILL = "empty_paged_prefill"
    DENSE = "dense"

# Backend-agnostic paged-KV page-block multiple, matching the declared default of
# AttentionCapabilities.paged_block_size_multiple (1 = no extra alignment beyond a
# whole page). A geometry-restricted backend such as flash_attn_with_kvcache, which
# needs 256-token pages, advertises that via capabilities().paged_block_size_multiple
# and overrides this seed; callers that omit the backend inherit the contract default
# instead of silently imposing flash_attn's stricter 256 requirement.
_DEFAULT_PAGED_BLOCK_SIZE_MULTIPLE = 1


class RadixAttention(nn.Module):
    """Model-facing attention op — thin: holds only ``layer_id`` + shapes + scale.

    The SGLang-shaped attention seam. The model calls ``forward(q, k, v,
    forward_batch, save_kv_cache=…)`` and the op resolves its residency from the
    **system-published** context: the per-forward attention plan
    (``forward_batch.attn_metadata``, built by the ``ForwardBatchBuilder``) names
    the paged request-cache view, so the model never owns a pool or threads a
    cache. It keeps the paged/varlen/dense forward paths and a backward-flexible
    ``kv_cache=``/``update_cache=`` entry for the dense (vision) and per-branch
    scratch-KV (generation) callers that pass their cache directly.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        layer_id: int = 0,
        backend_name: str | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id
        self.scale = head_dim**-0.5
        self.backend_name = backend_name

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch=None,
        *,
        kv_cache=None,
        update_cache: bool | None = None,
        save_kv_cache: bool = True,
        causal: bool = True,
        attn_mask: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        ctx = get_forward_context()
        preferred = ctx.attention_backend_name or self.backend_name or "torch_sdpa"
        effective_scale = self.scale if scale is None else scale
        # System-managed text path: the cache is the per-forward plan the system
        # built into ``forward_batch.attn_metadata`` (published on the context);
        # the model passes no cache. Dense (vision) and scratch-KV (generation)
        # callers pass ``kv_cache=`` directly and that wins.
        if kv_cache is None and forward_batch is not None:
            metadata = getattr(forward_batch, "attn_metadata", None)
            if metadata is not None:
                kv_cache = getattr(metadata, "cache", None)
        update = save_kv_cache if update_cache is None else update_cache

        path = self._resolve_attention_path(
            ctx, preferred, q, k, v, kv_cache=kv_cache, update_cache=update, attn_mask=attn_mask
        )
        if path is AttentionPath.CONTIGUOUS_VARLEN:
            return self._forward_contiguous_varlen(
                ctx, preferred, q, k, v, causal=causal, scale=effective_scale
            )
        if path is AttentionPath.PAGED_VARLEN:
            return self._forward_paged_varlen(
                ctx, preferred, q, k, v, causal=causal, scale=effective_scale
            )
        if path is AttentionPath.TRANSIENT_PAGED_VARLEN:
            return self._forward_transient_paged_varlen(
                ctx, preferred, q, k, v, kv_cache=kv_cache, causal=causal, scale=effective_scale
            )
        if path is AttentionPath.PAGED_EXTEND:
            return self._forward_paged_extend(
                ctx, preferred, q, k, v, kv_cache=kv_cache, causal=causal, scale=effective_scale
            )
        if path is AttentionPath.PAGED_DECODE:
            return self._forward_paged_decode(
                ctx, preferred, q, k, v, kv_cache=kv_cache, causal=causal, scale=effective_scale
            )
        if path is AttentionPath.EMPTY_PAGED_PREFILL:
            return self._forward_empty_paged_prefill(
                ctx, preferred, q, k, v, kv_cache=kv_cache, causal=causal, scale=effective_scale, attn_mask=attn_mask
            )
        return self._forward_dense(
            ctx,
            preferred,
            q,
            k,
            v,
            kv_cache=kv_cache,
            update_cache=update,
            causal=causal,
            scale=effective_scale,
            attn_mask=attn_mask,
        )

    def _resolve_attention_path(
        self,
        ctx,
        preferred: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache,
        update_cache: bool,
        attn_mask: torch.Tensor | None,
    ) -> AttentionPath:
        """Select the forward path and resolved backends for this call.

        Short-circuits in load-bearing order: contiguous varlen, paged varlen,
        paged decode, empty paged prefill, then dense. The probes ask the
        attention dispatcher whether a typed request has an eligible provider;
        the actual forward calls still go through ``ops.attention``.
        """
        if kv_cache is None:
            return AttentionPath.DENSE
        # Shared gate for all paged/varlen branches.
        paged_eligible = update_cache and hasattr(kv_cache, "pool")
        if paged_eligible:
            if self._can_run_contiguous_varlen_prefill(ctx, preferred, kv_cache, q, k, v):
                return AttentionPath.CONTIGUOUS_VARLEN
            if self._can_run_paged_varlen_prefill(ctx, preferred, kv_cache, q, k, v):
                return AttentionPath.PAGED_VARLEN
            if self._can_run_transient_paged_varlen(ctx, preferred, kv_cache, q, k, v):
                return AttentionPath.TRANSIENT_PAGED_VARLEN
            # Self-managing segment decoders pass a single-request paged view
            # directly with token-major [L, H, D] q/k/v and no system-built
            # per-forward plan. A multi-token append+attend on such a view is a
            # one-sequence EXTEND whose plan the view itself supplies; without
            # this branch it would fall through to the paged-decode path, which
            # reads a 3-D q as one-token *rows* and corrupts the write.
            if attn_mask is None and self._can_run_paged_extend(ctx, preferred, kv_cache, q, k, v):
                return AttentionPath.PAGED_EXTEND

        if paged_eligible:
            if self.can_run_paged_attention(q, attn_mask, kv_cache=kv_cache, preferred=preferred, ctx=ctx):
                return AttentionPath.PAGED_DECODE
            if self._can_run_empty_paged_prefill(ctx, kv_cache, q, k, v):
                return AttentionPath.EMPTY_PAGED_PREFILL
        return AttentionPath.DENSE

    @staticmethod
    def _attention_override(ctx, preferred: str) -> str:
        return "context" if getattr(ctx, "attention_backend", None) is not None else preferred

    def _forward_contiguous_varlen(
        self,
        ctx,
        preferred: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        metadata = ctx.attention_metadata
        assert metadata is not None
        raw_tokens = self._raw_varlen_tokens(ctx, q)
        q_run, k_run, v_run = self._trim_padded_varlen(q, k, v, raw_tokens)
        self._record_varlen_padding(ctx, total_tokens=int(q.shape[0]), raw_tokens=raw_tokens)
        out = ops.attention(
            q_run,
            k_run,
            v_run,
            regime=ops.AttentionRegime.EXTEND,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_q,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_q,
            causal=causal,
            scale=scale,
            block_table=None,
            ctx=ctx,
            override=self._attention_override(ctx, preferred),
        )
        metadata.cache.append_varlen(
            self.layer_id,
            k_run,
            v_run,
            metadata.query_lens_cpu,
            block_table=metadata.block_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
        )
        return self._restore_padded_varlen_output(out, q, raw_tokens)

    def _forward_paged_varlen(
        self,
        ctx,
        preferred: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        metadata = ctx.attention_metadata
        assert metadata is not None
        raw_tokens = self._raw_varlen_tokens(ctx, q)
        q_run, k_run, v_run = self._trim_padded_varlen(q, k, v, raw_tokens)
        self._record_varlen_padding(ctx, total_tokens=int(q.shape[0]), raw_tokens=raw_tokens)
        metadata.cache.append_varlen(
            self.layer_id,
            k_run,
            v_run,
            metadata.query_lens_cpu,
            block_table=metadata.block_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
        )
        k_cache, v_cache = metadata.cache.pool.layer_cache(self.layer_id)
        out = ops.attention(
            q_run,
            k_cache,
            v_cache,
            regime=ops.AttentionRegime.EXTEND,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_k,
            causal=causal,
            scale=scale,
            block_table=metadata.block_table,
            ctx=ctx,
            kv_cache=metadata.cache,
            override=self._attention_override(ctx, preferred),
        )
        return self._restore_padded_varlen_output(out, q, raw_tokens)

    def _forward_transient_paged_varlen(
        self,
        ctx,
        preferred: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        metadata = self._transient_varlen_metadata(kv_cache, q)
        if metadata is None:
            raise RuntimeError("transient paged-varlen attention became ineligible")
        query_lens_cpu, block_table, cache_seqlens, cu_seqlens_q, cu_seqlens_k, max_q, max_k = metadata
        q_run = self._flatten_bhld(q)
        k_run = self._flatten_bhld(k)
        v_run = self._flatten_bhld(v)
        kv_cache.append_varlen(
            self.layer_id,
            k_run,
            v_run,
            query_lens_cpu,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )
        k_cache, v_cache = kv_cache.pool.layer_cache(self.layer_id)
        out = ops.attention(
            q_run,
            k_cache,
            v_cache,
            regime=ops.AttentionRegime.EXTEND,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=causal,
            scale=scale,
            block_table=block_table,
            ctx=ctx,
            kv_cache=kv_cache,
            override=self._attention_override(ctx, preferred),
        )
        batch, _heads, q_len, _head_dim = q.shape
        # Return a [batch, heads, tokens, dim] *view* over the kernel's
        # token-major output. Values are identical to the previously returned
        # materialized copy; callers that need a different layout re-transpose,
        # which lands back on the contiguous token-major storage for free.
        return out.view(batch, q_len, int(out.shape[1]), int(out.shape[2])).transpose(1, 2)

    @staticmethod
    def _paged_extend_metadata(kv_cache, q: torch.Tensor):
        """Derive the one-sequence EXTEND plan from a directly-passed view.

        Eligible only for the self-managed shape: token-major 3-D multi-token
        q against a single-request paged view exposing the append surface
        (``append_varlen``/``block_table``/``cache_seqlens``/``length``).
        System-planned batches (4-D q, batched views, per-forward metadata)
        never reach this — their branches resolve earlier.
        """
        if q.ndim != 3 or int(q.shape[0]) <= 1:
            return None
        base_lens = getattr(kv_cache, "base_lens", None)
        if base_lens is None or len(tuple(base_lens)) != 1:
            return None
        if not callable(getattr(kv_cache, "append_varlen", None)):
            return None
        block_table_fn = getattr(kv_cache, "block_table", None)
        cache_seqlens_fn = getattr(kv_cache, "cache_seqlens", None)
        length_fn = getattr(kv_cache, "length", None)
        if not (callable(block_table_fn) and callable(cache_seqlens_fn) and callable(length_fn)):
            return None
        device = q.device
        n_tokens = int(q.shape[0])
        past = int(length_fn())
        block_table = block_table_fn(device=device)
        cache_seqlens = cache_seqlens_fn(device=device)
        block_size = int(getattr(getattr(kv_cache, "pool", None), "block_size", 0) or 0)
        if block_size <= 0:
            return None
        if int(block_table.shape[-1]) * block_size < past + n_tokens:
            raise RuntimeError(
                "paged-extend view capacity is smaller than the appended sequence "
                f"({int(block_table.shape[-1])} blocks x {block_size} < {past} + {n_tokens})"
            )
        cu_seqlens_q = torch.tensor([0, n_tokens], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, past + n_tokens], dtype=torch.int32, device=device)
        return (
            [n_tokens],
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            cu_seqlens_k,
            n_tokens,
            past + n_tokens,
        )

    @staticmethod
    def _can_run_paged_extend(
        ctx,
        preferred: str,
        kv_cache,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        if k.ndim != 3 or v.ndim != 3:
            return False
        if int(k.shape[0]) != int(q.shape[0]) or int(v.shape[0]) != int(q.shape[0]):
            return False
        metadata = RadixAttention._paged_extend_metadata(kv_cache, q)
        if metadata is None:
            return False
        _query_lens_cpu, block_table, _cache_seqlens, cu_seqlens_q, cu_seqlens_k, max_q, max_k = metadata
        q_probe = q.new_empty((1, int(q.shape[1]), int(q.shape[2])))
        k_probe = k.new_empty((1, int(k.shape[1]), int(k.shape[2])))
        v_probe = v.new_empty((1, int(v.shape[1]), int(v.shape[2])))
        return ops.can_run_attention(
            q_probe,
            k_probe,
            v_probe,
            regime=ops.AttentionRegime.EXTEND,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=True,
            scale=1.0,
            block_table=block_table,
            kv_cache=kv_cache,
            ctx=ctx,
            override=RadixAttention._attention_override(ctx, preferred),
        )

    def _forward_paged_extend(
        self,
        ctx,
        preferred: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        metadata = self._paged_extend_metadata(kv_cache, q)
        if metadata is None:
            raise RuntimeError("paged-extend attention became ineligible")
        query_lens_cpu, block_table, cache_seqlens, cu_seqlens_q, cu_seqlens_k, max_q, max_k = metadata
        kv_cache.append_varlen(
            self.layer_id,
            k,
            v,
            query_lens_cpu,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )
        k_cache, v_cache = kv_cache.pool.layer_cache(self.layer_id)
        return ops.attention(
            q,
            k_cache,
            v_cache,
            regime=ops.AttentionRegime.EXTEND,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=causal,
            scale=scale,
            block_table=block_table,
            ctx=ctx,
            kv_cache=kv_cache,
            override=self._attention_override(ctx, preferred),
        )

    def _forward_paged_decode(
        self,
        ctx,
        preferred: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        k_cache, v_cache = kv_cache.pool.layer_cache(self.layer_id)
        block_table, cache_seqlens = self._paged_metadata_tensors(ctx, kv_cache, q.device)
        if q.ndim == 3 and int(q.shape[0]) != int(block_table.shape[0]):
            # One decode row per block-table row is the contract; a mismatch
            # means a multi-token single-sequence segment was misrouted here
            # (its tokens would be scattered as rows) — fail before writing.
            raise RuntimeError(
                f"paged decode row mismatch: q has {int(q.shape[0])} rows but the "
                f"plan covers {int(block_table.shape[0])} request(s)"
            )
        # Decode q/k/v arrive as ``[batch, heads, dim]`` (one token per row). A
        # paged kernel's BLHD/BHD layout normalizer treats a 3-D tensor as
        # ``[L, H, D]`` (one sequence), which would fold the *batch* into the
        # query length and mismatch the ``[batch, max_blocks]`` block table for
        # batch>1. Present them explicitly as ``[batch, heads, 1, dim]`` so the
        # batch dimension is unambiguous, then drop the unit query length on the
        # way out. (For batch==1 this is byte-identical to the 3-D path.)
        decode_rows = q.ndim == 3
        if decode_rows:
            q = q.unsqueeze(2)
            k = k.unsqueeze(2) if k is not None else None
            v = v.unsqueeze(2) if v is not None else None
        out = ops.attention(
            q,
            k_cache,
            v_cache,
            regime=ops.AttentionRegime.DECODE,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            current_k=k,
            current_v=v,
            causal=causal,
            scale=scale,
            ctx=ctx,
            kv_cache=kv_cache,
            override=self._attention_override(ctx, preferred),
        )
        if decode_rows and out.ndim == 4:
            out = out.squeeze(2)
        return out

    def _forward_empty_paged_prefill(
        self,
        ctx,
        preferred: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache,
        causal: bool,
        scale: float,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        out = ops.attention(
            q,
            k,
            v,
            regime=ops.AttentionRegime.DENSE,
            causal=causal,
            scale=scale,
            attn_mask=attn_mask,
            ctx=ctx,
            override=self._attention_override(ctx, preferred),
        )
        self._append_batched_paged_current(kv_cache, k, v)
        return out

    def _forward_dense(
        self,
        ctx,
        preferred: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache,
        update_cache: bool,
        causal: bool,
        scale: float,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if kv_cache is not None:
            cache_get = getattr(kv_cache, "get", None)
            if not callable(cache_get):
                self._raise_unsupported_paged_fallback(kv_cache)
            cached_k, cached_v = cache_get(self.layer_id)
            if cached_k is not None:
                k = torch.cat([cached_k, k], dim=0)
                v = torch.cat([cached_v, v], dim=0)

        out = ops.attention(
            q,
            k,
            v,
            regime=ops.AttentionRegime.DENSE,
            causal=causal,
            scale=scale,
            attn_mask=attn_mask,
            ctx=ctx,
            override=self._attention_override(ctx, preferred),
        )
        if kv_cache is not None and update_cache:
            k_tail = k[-q.shape[0] :]
            v_tail = v[-q.shape[0] :]
            if k_tail.ndim != 3 or v_tail.ndim != 3:
                raise invalid_descriptor("cached KV tensors must be [tokens, heads, dim]")
            kv_cache.append(self.layer_id, k_tail, v_tail)
        return out

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
    ) -> torch.Tensor:
        """Run hybrid ``visible_end`` masked attention via the op dispatcher."""
        ctx = get_forward_context()
        effective_scale = self.scale if scale is None else scale
        override = ctx.attention_backend_name or self.backend_name
        if ctx.attention_backend is not None:
            override = "context"
        try:
            return ops.attention(
                q,
                k,
                v,
                regime=ops.AttentionRegime.VISIBLE_END,
                causal=False,
                scale=effective_scale,
                ctx=ctx,
                visible_end=visible_end,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                page_table=page_table,
                seqused_k=seqused_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                use_prefix_bounds=use_prefix_bounds,
                fully_visible=fully_visible,
                override=override,
            )
        except RuntimeError as exc:
            raise capability_mismatch(
                "visible_end attention requires a backend with paged visible_end "
                "support; install and select the fa4_cute provider pack"
            ) from exc

    @staticmethod
    def _paged_metadata_tensors(ctx, kv_cache, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = getattr(ctx, "attention_metadata", None)
        stats = getattr(ctx, "stats", None)
        if metadata is not None and getattr(metadata, "cache", None) is kv_cache:
            if stats is not None and not _torch_is_compiling():
                stats.attention_metadata_hits += 1
            return metadata.block_table, metadata.cache_seqlens
        if stats is not None and not _torch_is_compiling():
            stats.attention_metadata_misses += 1
        cache_seqlens = kv_cache.cache_seqlens(device=device)
        return kv_cache.block_table(device=device), cache_seqlens

    def can_run_paged_attention(
        self,
        q: torch.Tensor,
        attn_mask: torch.Tensor | None,
        *,
        kv_cache=None,
        preferred: str = "auto",
        ctx=None,
    ) -> bool:
        if attn_mask is not None or q.ndim not in {3, 4}:
            return False
        ctx = get_forward_context() if ctx is None else ctx
        block_table = q.new_empty((1, 1), dtype=torch.int32)
        cache_seqlens = q.new_empty((1,), dtype=torch.int32)
        return ops.can_run_attention(
            q,
            q,
            q,
            regime=ops.AttentionRegime.DECODE,
            causal=True,
            scale=self.scale,
            ctx=ctx,
            kv_cache=kv_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            override=self._attention_override(ctx, preferred),
        )

    @staticmethod
    def _can_run_empty_paged_prefill(ctx, kv_cache, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
        metadata = getattr(ctx, "attention_metadata", None)
        if metadata is None or getattr(metadata, "cache", None) is not kv_cache:
            return False
        if getattr(metadata, "mode", None) not in {"extend", "mixed"}:
            return False
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            return False
        base_lens = getattr(kv_cache, "base_lens", None)
        if not base_lens or any(int(length) != 0 for length in base_lens):
            return False
        return int(k.shape[0]) == len(base_lens) and k.shape == v.shape

    def _append_batched_paged_current(self, kv_cache, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.ndim != 4 or v.ndim != 4:
            raise invalid_descriptor("batched paged prefill append expects [batch, heads, tokens, dim] K/V")
        kv_cache.append(
            self.layer_id,
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
        )

    @staticmethod
    def _is_one_token_decode(q: torch.Tensor) -> bool:
        if q.ndim == 3:
            metadata = getattr(get_forward_context(), "attention_metadata", None)
            if (
                getattr(metadata, "mode", None) == "decode"
                and len(getattr(metadata, "query_lens_cpu", ()) or ()) == int(q.shape[0])
                and all(int(length) == 1 for length in getattr(metadata, "query_lens_cpu", ()) or ())
            ):
                return True
            return int(q.shape[0]) == 1
        if q.ndim == 4:
            return int(q.shape[2]) == 1
        return False

    @staticmethod
    def _raw_varlen_tokens(ctx, q: torch.Tensor) -> int:
        metadata = getattr(ctx, "attention_metadata", None)
        query_lens_cpu = getattr(metadata, "query_lens_cpu", ()) or ()
        raw_tokens = sum(int(length) for length in query_lens_cpu)
        if raw_tokens <= 0 or raw_tokens > int(q.shape[0]):
            return int(q.shape[0])
        return int(raw_tokens)

    @staticmethod
    def _trim_padded_varlen(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_tokens = int(raw_tokens)
        if raw_tokens == int(q.shape[0]):
            return q, k, v
        return q[:raw_tokens], k[:raw_tokens], v[:raw_tokens]

    @staticmethod
    def _restore_padded_varlen_output(
        out: torch.Tensor,
        q: torch.Tensor,
        raw_tokens: int,
    ) -> torch.Tensor:
        raw_tokens = int(raw_tokens)
        if raw_tokens == int(q.shape[0]):
            return out
        padded = q.new_zeros((int(q.shape[0]), int(out.shape[1]), int(out.shape[2])))
        padded[:raw_tokens].copy_(out)
        return padded

    @staticmethod
    def _record_varlen_padding(ctx, *, total_tokens: int, raw_tokens: int) -> None:
        padding = int(total_tokens) - int(raw_tokens)
        if padding <= 0:
            return
        if _torch_is_compiling():
            return
        stats = getattr(ctx, "stats", None)
        if stats is None:
            return
        stats.attention_padded_tokens += padding
        stats.attention_padded_calls += 1

    @staticmethod
    def _can_run_contiguous_varlen_prefill(
        ctx,
        preferred: str,
        kv_cache,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        metadata = getattr(ctx, "attention_metadata", None)
        if metadata is None or getattr(metadata, "cache", None) is not kv_cache:
            return False
        if getattr(metadata, "mode", None) not in {"extend", "mixed"}:
            return False
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            return False
        if any(int(length) != 0 for length in getattr(kv_cache, "base_lens", [])):
            return False
        query_lens = getattr(metadata, "query_lens", None)
        cu_seqlens_q = getattr(metadata, "cu_seqlens_q", None)
        query_lens_cpu = getattr(metadata, "query_lens_cpu", ())
        if query_lens is None or cu_seqlens_q is None or not query_lens_cpu:
            return False
        raw_tokens = sum(int(x) for x in query_lens_cpu)
        if raw_tokens <= 0:
            return False
        if int(q.shape[0]) != int(k.shape[0]) or int(q.shape[0]) != int(v.shape[0]):
            return False
        if int(q.shape[0]) < int(raw_tokens):
            return False
        return ops.can_run_attention(
            q,
            k,
            v,
            regime=ops.AttentionRegime.EXTEND,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_q,
            causal=True,
            scale=1.0,
            ctx=ctx,
            override=RadixAttention._attention_override(ctx, preferred),
        )

    @staticmethod
    def _can_run_paged_varlen_prefill(
        ctx,
        preferred: str,
        kv_cache,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        metadata = getattr(ctx, "attention_metadata", None)
        if metadata is None or getattr(metadata, "cache", None) is not kv_cache:
            return False
        if getattr(metadata, "mode", None) not in {"extend", "mixed"}:
            return False
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            return False
        if not any(int(length) != 0 for length in getattr(kv_cache, "base_lens", [])):
            return False
        required = (
            "block_table",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "max_seqlen_q",
            "max_seqlen_k",
        )
        if any(getattr(metadata, name, None) is None for name in required):
            return False
        query_lens_cpu = getattr(metadata, "query_lens_cpu", ())
        if not query_lens_cpu:
            return False
        raw_tokens = sum(int(x) for x in query_lens_cpu)
        if raw_tokens <= 0:
            return False
        if int(q.shape[0]) != int(k.shape[0]) or int(q.shape[0]) != int(v.shape[0]):
            return False
        if int(q.shape[0]) < int(raw_tokens):
            return False
        return ops.can_run_attention(
            q,
            k,
            v,
            regime=ops.AttentionRegime.EXTEND,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_k,
            causal=True,
            scale=1.0,
            block_table=metadata.block_table,
            kv_cache=kv_cache,
            ctx=ctx,
            override=RadixAttention._attention_override(ctx, preferred),
        )

    @staticmethod
    def _can_run_transient_paged_varlen(
        ctx,
        preferred: str,
        kv_cache,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        if k.ndim != 4 or v.ndim != 4:
            return False
        if int(k.shape[0]) != int(q.shape[0]) or int(v.shape[0]) != int(q.shape[0]):
            return False
        if int(k.shape[2]) != int(q.shape[2]) or int(v.shape[2]) != int(q.shape[2]):
            return False
        metadata = RadixAttention._transient_varlen_metadata(kv_cache, q)
        if metadata is None:
            return False
        _query_lens_cpu, block_table, _cache_seqlens, cu_seqlens_q, cu_seqlens_k, max_q, max_k = metadata
        q_probe = q.new_empty((1, int(q.shape[1]), int(q.shape[3])))
        k_probe = k.new_empty((1, int(k.shape[1]), int(k.shape[3])))
        v_probe = v.new_empty((1, int(v.shape[1]), int(v.shape[3])))
        return ops.can_run_attention(
            q_probe,
            k_probe,
            v_probe,
            regime=ops.AttentionRegime.EXTEND,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=True,
            scale=1.0,
            block_table=block_table,
            kv_cache=kv_cache,
            ctx=ctx,
            override=RadixAttention._attention_override(ctx, preferred),
        )

    @staticmethod
    def _transient_varlen_metadata(kv_cache, q: torch.Tensor):
        if q.ndim != 4 or int(q.shape[2]) <= 1:
            return None
        batch = int(q.shape[0])
        q_len = int(q.shape[2])
        if batch <= 0 or q_len <= 0:
            return None
        if not (
            hasattr(kv_cache, "pool")
            and callable(getattr(kv_cache, "block_table", None))
            and callable(getattr(kv_cache, "cache_seqlens", None))
            and callable(getattr(kv_cache, "append_varlen", None))
        ):
            return None
        base_lens = getattr(kv_cache, "base_lens", None)
        if base_lens is None:
            base_len = getattr(kv_cache, "base_len", None)
            if base_len is None:
                return None
            base_lens = (int(base_len),)
        else:
            base_lens = tuple(int(length) for length in base_lens)
        if len(base_lens) != batch or any(length < 0 for length in base_lens):
            return None
        device = q.device
        key = (
            str(device),
            tuple(base_lens),
            q_len,
        )
        cached = getattr(kv_cache, "_transient_varlen_metadata", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        block_table = kv_cache.block_table(device=device)
        cache_seqlens = kv_cache.cache_seqlens(device=device)
        if int(block_table.shape[0]) != batch or int(cache_seqlens.shape[0]) != batch:
            return None
        query_lens_cpu = tuple(q_len for _ in base_lens)
        kv_lens_cpu = tuple(int(base_len) + q_len for base_len in base_lens)
        cu_q_values = [0]
        cu_k_values = [0]
        for query_len, kv_len in zip(query_lens_cpu, kv_lens_cpu, strict=True):
            cu_q_values.append(cu_q_values[-1] + int(query_len))
            cu_k_values.append(cu_k_values[-1] + int(kv_len))
        cu_seqlens_q = torch.tensor(cu_q_values, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor(cu_k_values, dtype=torch.int32, device=device)
        metadata = (
            query_lens_cpu,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            cu_seqlens_k,
            max(query_lens_cpu, default=0),
            max(kv_lens_cpu, default=0),
        )
        try:
            setattr(kv_cache, "_transient_varlen_metadata", (key, metadata))
        except Exception:
            pass
        return metadata

    @staticmethod
    def _flatten_bhld(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise invalid_descriptor("transient paged-varlen attention expects [batch, heads, tokens, dim]")
        batch, heads, tokens, dim = tensor.shape
        return tensor.transpose(1, 2).reshape(int(batch) * int(tokens), int(heads), int(dim)).contiguous()

    @staticmethod
    def _raise_unsupported_paged_fallback(kv_cache) -> None:
        pool = getattr(kv_cache, "pool", None)
        if pool is not None and not bool(getattr(pool, "supports_paged_attention_storage", True)):
            raise capability_mismatch(
                "quantized paged KV storage requires a scale-aware paged attention "
                "backend for batched page-table caches; dense fallback is only "
                "available for single-request request-cache views"
            )
        raise capability_mismatch(
            "batched paged KV cache requires a supported paged attention backend; "
            "dense fallback is only available for single-request request-cache views"
        )
