"""SGL's FlashAttention kernels over the public numerical inputs."""

import torch

from uniserve.nn.attention.inputs import PagedInput

from . import Operator as _Operator
from .flash_attn import Backend as _Backend
from .flash_attn import _FlashOperator

__all__ = ["Backend"]


class _SGL(_FlashOperator):
    """SGLang's FlashAttention build, sharing the FlashAttention operator flow."""

    def requires_host_lengths(self, batch):
        # This native entry takes rectangular queries; variable paged batches
        # need exact host boundaries to select their separate query views.
        return isinstance(batch, PagedInput) or super().requires_host_lengths(batch)

    def __init__(self, **kwargs):
        _Operator.__init__(self, **kwargs)
        from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

        self._varlen_kernel = flash_attn_varlen_func
        self._paged_kernel = flash_attn_with_kvcache
        self._paged_options = self._varlen_options = {"ver": 3}
        self._table_argument = "page_table"
        self._validate_representation()

    def _dense(self, q, k, v, causal, scale):
        packed = q.ndim == 3
        query, key, value = (
            tensor.unsqueeze(0) if packed else tensor.transpose(1, 2) for tensor in (q, k, v)
        )
        batches, queries, keys = query.shape[0], query.shape[1], key.shape[1]
        if keys == 0:
            return torch.zeros_like(q)

        # No native dense entry: flatten the rectangular batch into varlen form.
        query_offsets = torch.arange(batches + 1, dtype=torch.int32, device=q.device) * queries
        key_offsets = torch.arange(batches + 1, dtype=torch.int32, device=q.device) * keys

        result = self._varlen_kernel(
            query.flatten(0, 1),
            key.flatten(0, 1),
            value.flatten(0, 1),
            query_offsets,
            key_offsets,
            queries,
            keys,
            softmax_scale=scale,
            causal=causal,
            ver=3,
        ).view_as(query)
        return result.squeeze(0) if packed else result.transpose(1, 2)

    def _paged_varlen(self, q, k, v, batch, lengths, scale):
        # The SGL paged entry consumes rectangular query batches. Independent
        # variable-length sequences use the same kernel with one row each.
        outputs = []
        start = 0
        for row, count in enumerate(batch.queries.host):
            if count:
                outputs.append(
                    self._paged_kernel(
                        q[start : start + count].unsqueeze(0),
                        k,
                        v,
                        cache_seqlens=lengths[row : row + 1],
                        page_table=batch.block_table.indices[row : row + 1].to(dtype=torch.int32),
                        softmax_scale=scale,
                        causal=batch.causal[row],
                        ver=3,
                    ).squeeze(0)
                )
            start += count
        return torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)


class Backend(_Backend):
    operator_class = _SGL
