"""Attention over row-aligned forward tensors."""

from __future__ import annotations

import torch
from torch import nn

import uniserve_worker.ops as ops

from ..backends.attention.base import AttentionBackend
from ..execution.forward_batch import AttentionMode, AttentionSelection, ForwardBatch
from ..runtime.cache_pool import CachePool
from .attention_storage import attention_exchange_storage
from .mesh import Communicator
from .parallel_attention import AttentionHeadRows, AttentionRowExchange, HeadRowExchange
from .parallel_sequence import SequencePartition


class RadixAttention(nn.Module):
    """Dense and paged attention facade over startup-owned backend resources."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        layer_id: int = 0,
        sequence: Communicator | None = None,
    ) -> None:
        """Validate local head geometry and identify the layer's paged-cache slice."""

        super().__init__()
        if min(num_heads, num_kv_heads, head_dim) < 1:
            raise ValueError("attention geometry must be positive")
        self.exchange = HeadRowExchange(Communicator() if sequence is None else sequence)
        if num_heads % self.exchange.ulysses_group.world_size:
            raise ValueError("query heads must divide sequence membership")
        self.num_heads = self.exchange.head_region(int(num_heads))[0]
        self.num_kv_heads = self.exchange.head_region(int(num_kv_heads))[0]
        self.head_dim = int(head_dim)
        self.layer_id = int(layer_id)
        self.scale = self.head_dim**-0.5
        self._cache_pool: CachePool | None = None
        self._selection: AttentionSelection | None = None
        self._providers: dict[AttentionMode, AttentionBackend] = {}
        self._varlen_provider: AttentionBackend | None = None

    def bind_dense(self, selection: AttentionSelection) -> None:
        """Bind execution-owned providers for cache-free attention requests.

        Dense requests select by their actual dtype, layout and mask; head and
        page geometry alone cannot determine which numerical path is legal.
        """

        self._selection = selection

    def bind(self, cache_pool: CachePool, selection: AttentionSelection) -> None:
        """Bind physical KV storage and select one compatible backend per attention mode."""

        self._cache_pool = cache_pool
        self.bind_dense(selection)
        candidates = {
            AttentionMode.DENSE: _select_provider(
                selection,
                AttentionMode.DENSE,
                head_dim=self.head_dim,
                block_size=cache_pool.block_size,
                device=cache_pool.k.device,
            ),
            AttentionMode.PAGED_DECODE: _select_provider(
                selection,
                AttentionMode.PAGED_DECODE,
                head_dim=self.head_dim,
                block_size=cache_pool.block_size,
                device=cache_pool.k.device,
            ),
            AttentionMode.PAGED_VARLEN: _select_provider(
                selection,
                AttentionMode.PAGED_VARLEN,
                head_dim=self.head_dim,
                block_size=cache_pool.block_size,
                device=cache_pool.k.device,
            ),
            AttentionMode.PACKED: (
                _select_provider(
                    selection,
                    AttentionMode.PACKED,
                    cuda_graph=True,
                    head_dim=self.head_dim,
                    block_size=cache_pool.block_size,
                    device=cache_pool.k.device,
                )
                or _select_provider(
                    selection,
                    AttentionMode.PACKED,
                    head_dim=self.head_dim,
                    block_size=cache_pool.block_size,
                    device=cache_pool.k.device,
                )
            ),
        }
        self._providers = {
            mode: provider for mode, provider in candidates.items() if provider is not None
        }
        self._varlen_provider = _select_varlen_provider(
            selection,
            head_dim=self.head_dim,
            block_size=cache_pool.block_size,
            device=cache_pool.k.device,
        )

    @property
    def selection(self) -> AttentionSelection:
        """Expose the immutable startup backend selection after binding."""

        if self._selection is None:
            raise RuntimeError("attention module has not been bound to a startup backend")
        return self._selection

    def provider(self, mode: AttentionMode) -> AttentionBackend:
        """Resolve the bound backend for a concrete attention execution mode."""

        try:
            return self._providers[mode]
        except KeyError:
            raise RuntimeError(
                f"attention module has no bound provider for {mode.value!r}"
            ) from None

    @property
    def varlen_provider(self) -> AttentionBackend | None:
        """Expose the optional provider used for variable-length cache prefix reads."""

        return self._varlen_provider

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch | None,
        *,
        causal: bool,
        scale: float | None = None,
        attn_mask: torch.Tensor | None = None,
        partition: SequencePartition | None = None,
        consume_row_intervals: bool = False,
    ) -> torch.Tensor | AttentionRowExchange:
        """Dispatch dense or paged attention and commit requested K/V rows to the bound cache."""

        group = self.exchange.ulysses_group
        if group.world_size == 1:
            return self._forward(q, k, v, context, causal=causal, scale=scale, attn_mask=attn_mask)
        if context is None or context.forward_mode is AttentionMode.DENSE or q.ndim != 3:
            raise ValueError("sequence-owned paged attention requires packed Q/K/V rows")
        if partition is None or partition.group != group:
            raise ValueError("sequence attention requires its caller-owned row partition")
        storage = None
        if consume_row_intervals and partition.capacity > AttentionRowExchange.chunk_rows(
            q[:, : self.num_heads]
        ):
            storage = attention_exchange_storage(group)
        q, k, v = (
            self.exchange.exchange_heads(partition.pad(value), storage=storage, role=role)[
                : partition.rows
            ]
            for role, value in zip(("query", "key", "value"), (q, k, v), strict=True)
        )
        return self.forward_heads(
            AttentionHeadRows(q, k, v, partition),
            context,
            causal=causal,
            scale=scale,
            attn_mask=attn_mask,
            consume_row_intervals=consume_row_intervals,
        )

    def forward_heads(
        self,
        inputs: AttentionHeadRows,
        context: ForwardBatch,
        *,
        causal: bool,
        scale: float | None = None,
        attn_mask: torch.Tensor | None = None,
        consume_row_intervals: bool = False,
    ) -> torch.Tensor | AttentionRowExchange:
        """Consume globally visible head rows and restore their sequence owners."""

        partition = inputs.partition
        group = self.exchange.ulysses_group
        if partition.group != group:
            raise ValueError("attention head rows require their sequence communicator")
        q, k, v = inputs.query, inputs.key, inputs.value
        if any(value.shape[0] != partition.rows for value in (q, k, v)):
            raise ValueError("attention head rows must cover the complete logical sequence")
        output = self._forward(q, k, v, context, causal=causal, scale=scale, attn_mask=attn_mask)
        capacity = partition.capacity * group.world_size
        if output.shape[0] != capacity:
            padded = output.new_zeros((capacity, *output.shape[1:]))
            padded[: partition.rows].copy_(output)
            output = padded
        # Small payloads have no later transfer to overlap; preserve the direct
        # exchange and its original GEMM row shape in that case.
        if consume_row_intervals and partition.capacity > AttentionRowExchange.chunk_rows(output):
            storage = attention_exchange_storage(group)
            return AttentionRowExchange(
                self.exchange,
                output.contiguous(),
                torch.empty_like(output)
                if storage is None
                else storage.view("output_send", tuple(output.shape), output),
                logical_rows=partition.count,
                receive_workspace=(
                    None
                    if storage is None
                    else storage.view("output_receive", tuple(output.shape), output)
                ),
            )
        return self.exchange.restore_rows(output)[: partition.count]

    def _forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch | None,
        *,
        causal: bool,
        scale: float | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Execute numerical attention over global logical rows and local heads."""

        effective_scale = self.scale if scale is None else float(scale)
        if self._selection is None:
            raise RuntimeError("attention module has not been bound to a startup backend")
        if context is None or context.forward_mode is AttentionMode.DENSE:
            request = ops.DenseAttention(
                q=q,
                k=k,
                v=v,
                causal=causal,
                scale=effective_scale,
                attn_mask=attn_mask,
                ctx=context,
            )
            for provider in self._selection.providers:
                if ops.can_run_attention(request, provider=provider):
                    return ops.attention(request, provider=provider)
            raise RuntimeError(
                f"attention selection {self._selection.identity!r} does not support "
                f"dense {q.dtype} {tuple(q.shape)} on {q.device} with "
                f"mask={attn_mask is not None}"
            )
        if attn_mask is not None:
            raise ValueError("paged attention does not accept a dense attention mask")
        pool = self._cache_pool
        if pool is None or context.block_table is None:
            raise RuntimeError("paged attention has no bound physical cache")
        if context.forward_mode is AttentionMode.PAGED_DECODE:
            return self._decode(q, k, v, context, causal, effective_scale, pool)
        if context.forward_mode is AttentionMode.PAGED_VARLEN:
            return self._varlen(q, k, v, context, causal, effective_scale, pool)
        if context.forward_mode is AttentionMode.REQUEST_INDEXED_DECODE:
            raise ValueError("request-indexed decode metadata was not staged")
        return self._packed(q, k, v, context, effective_scale, pool)

    def _decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch,
        causal: bool,
        scale: float,
        pool: CachePool,
    ) -> torch.Tensor:
        """Execute one-token paged decode against the selected cache group."""

        assert context.block_table is not None
        if q.ndim != 3 or int(q.shape[0]) != int(context.block_table.shape[0]):
            raise ValueError("paged decode query rows do not match its page table")
        if context.kv_lens is None:
            raise ValueError("paged decode requires resulting KV lengths")
        if context.has_cache_writes:
            pool.write_locations(self.layer_id, context.out_cache_loc, k, v)
        k_cache, v_cache = pool.layer_cache(self.layer_id, context.group_id)
        out = ops.attention(
            ops.PagedDecodeAttention(
                q=q.unsqueeze(2),
                k=k_cache,
                v=v_cache,
                block_table=context.block_table,
                cache_seqlens=context.kv_lens,
                causal=causal,
                scale=scale,
                ctx=context,
            ),
            provider=self.provider(AttentionMode.PAGED_DECODE),
        )
        return out.squeeze(2) if out.ndim == 4 else out

    def _varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch,
        causal: bool,
        scale: float,
        pool: CachePool,
    ) -> torch.Tensor:
        """Execute variable-length paged prefill with packed query boundaries."""

        raw_tokens = sum(context.query_lens_cpu)
        if (
            q.ndim != 3
            or raw_tokens < 1
            or raw_tokens > int(q.shape[0])
            or context.cu_seqlens_q is None
            or context.cu_seqlens_k is None
        ):
            raise ValueError("paged varlen query geometry is invalid")
        q_run, k_run, v_run = q[:raw_tokens], k[:raw_tokens], v[:raw_tokens]
        if context.has_cache_writes:
            pool.write_locations(self.layer_id, context.out_cache_loc[:raw_tokens], k_run, v_run)
        k_cache, v_cache = pool.layer_cache(self.layer_id, context.group_id)
        out = ops.attention(
            ops.VarlenAttention(
                q=q_run,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                causal=causal,
                scale=scale,
                block_table=context.block_table,
                ctx=context,
            ),
            provider=self.provider(AttentionMode.PAGED_VARLEN),
        )
        if raw_tokens == int(q.shape[0]):
            return out
        padded = q.new_zeros(q.shape)
        padded[:raw_tokens].copy_(out)
        return padded

    def _packed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch,
        scale: float,
        pool: CachePool,
    ) -> torch.Tensor:
        """Execute cache-free packed attention across declared route spans."""

        if q.ndim != 3 or context.cu_seqlens_q is None or context.visible_end is None:
            raise ValueError("packed attention geometry is invalid")
        if context.has_cache_writes:
            pool.write_locations(self.layer_id, context.out_cache_loc, k, v)
        k_cache, v_cache = pool.layer_cache(self.layer_id, context.group_id)
        return ops.attention(
            ops.VisibleEndAttention(
                q=q,
                k=k,
                v=v,
                visible_end=context.visible_end,
                scale=scale,
                cu_seqlens_q=context.cu_seqlens_q,
                page_table=context.block_table,
                fully_visible=context.fully_visible,
                prefix_k=k_cache,
                prefix_v=v_cache,
                prefix_lens=context.seq_lens,
                ctx=context,
            ),
            provider=self.provider(AttentionMode.PACKED),
        )


def _select_provider(
    selection: AttentionSelection,
    mode: AttentionMode,
    *,
    head_dim: int,
    block_size: int,
    device: torch.device,
    cuda_graph: bool = False,
) -> AttentionBackend | None:
    """Select the first backend that supports the requested mode and concrete geometry."""

    for provider in selection.providers:
        if provider.can_bind(
            mode,
            head_dim=head_dim,
            block_size=block_size,
            device=device,
            cuda_graph=cuda_graph,
        ):
            return provider
    return None


def _select_varlen_provider(
    selection: AttentionSelection,
    *,
    head_dim: int,
    block_size: int,
    device: torch.device,
) -> AttentionBackend | None:
    """Select paged-varlen attention or raise when no backend supports the geometry."""

    for provider in selection.providers:
        if provider.can_bind_varlen(
            head_dim=head_dim,
            block_size=block_size,
            device=device,
        ):
            return provider
    return None


def bind_dense_attention_modules(model: nn.Module, selection: AttentionSelection) -> None:
    """Share one execution owner's provider selection across its attention layers."""

    for module in model.modules():
        if isinstance(module, RadixAttention):
            module.bind_dense(selection)


def bind_attention_modules(
    model: nn.Module,
    cache_pool: CachePool,
    selection: AttentionSelection,
) -> None:
    """Bind every radix-attention layer in a model to shared cache and backend resources."""

    for module in model.modules():
        if isinstance(module, RadixAttention):
            module.bind(cache_pool, selection)


__all__ = ["RadixAttention", "bind_attention_modules", "bind_dense_attention_modules"]
