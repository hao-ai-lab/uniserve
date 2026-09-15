"""FlashAttention CUDA kernels with explicit sequence and cache inputs."""

import torch

from uniserve.nn.attention.inputs import DenseInput, PagedInput, VarlenInput
from uniserve.quantization import QuantizedTensor
from uniserve.tensors import BufferConfig

from . import Backend as _Backend
from . import Operator as _Operator
from ._sequences import causal_runs

__all__ = ["Backend"]


class _FlashOperator(_Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache

        self._dense_kernel = flash_attn_func
        self._varlen_kernel = flash_attn_varlen_func
        self._paged_kernel = flash_attn_with_kvcache
        self._paged_options = {}
        self._varlen_options = {"dropout_p": 0.0}
        self._table_argument = "block_table"
        self._validate_representation()

    def _validate_representation(self):
        if self.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("FlashAttention requires FP16 or BF16 computation")
        if self.cache is not None and (
            isinstance(self.cache.key, QuantizedTensor) or self.cache.block_size % 256
        ):
            raise ValueError("this FlashAttention kernel requires dense 256-token cache blocks")

    def requires_host_lengths(self, batch):
        return isinstance(batch, (PagedInput, VarlenInput)) and len(set(batch.causal)) > 1

    def __call__(self, q, k, v, batch, *, scale, out):
        if not q.is_cuda:
            raise ValueError("FlashAttention requires CUDA tensors")
        if not isinstance(batch, (DenseInput, PagedInput, VarlenInput)):
            raise ValueError("this FlashAttention kernel does not implement visible-range masks")
        if isinstance(batch, DenseInput) and batch.mask is not None:
            raise ValueError("this FlashAttention kernel does not implement dense masks")
        self._validate(q, k, v, batch, out)

        if isinstance(batch, PagedInput) and batch.write_indices is not None:
            self.update_cache(k, v, indices=batch.write_indices)

        if q.numel() == 0:
            return out

        if isinstance(batch, DenseInput):
            return out.copy_(self._dense(q, k, v, batch.causal, scale))

        if q.ndim != 3 or (
            batch.queries.num_tokens is not None and q.shape[0] != batch.queries.num_tokens
        ):
            raise ValueError("packed attention rows must match the declared query lengths")

        for query_slice, key_slice, run in causal_runs(batch):
            query = q[query_slice]
            if query.numel() == 0:
                continue

            if isinstance(run, PagedInput):
                key, value = (k, v) if self.cache is None else (self.cache.key, self.cache.value)
                if key.ndim != 4 or key.shape[1] % 256:
                    raise ValueError("paged FlashAttention requires 256-token cache blocks")
                result = self._paged(query, key, value, run, scale)
            else:
                result = self._varlen_kernel(
                    query.contiguous(),
                    k[key_slice].contiguous(),
                    v[key_slice].contiguous(),
                    run.queries.offsets,
                    run.keys.offsets,
                    query.shape[0],
                    k[key_slice].shape[0],
                    softmax_scale=scale,
                    causal=run.causal[0],
                    **self._varlen_options,
                )
            out[query_slice].copy_(result)
        return out

    def _dense(self, q, k, v, causal, scale):
        packed = q.ndim == 3
        values = (tensor.unsqueeze(0) if packed else tensor.transpose(1, 2) for tensor in (q, k, v))
        result = self._dense_kernel(*values, softmax_scale=scale, causal=causal, dropout_p=0.0)
        return result.squeeze(0) if packed else result.transpose(1, 2)

    def _paged(self, q, k, v, batch, scale):
        count = batch.queries.batch_size
        lengths = self.workspace["lengths"][:count]
        torch.add(batch.prefixes.values, batch.queries.values, out=lengths)

        if batch.queries.host is not None and len(set(batch.queries.host)) == 1:
            # Cache updates already occurred through the public State method.
            # Omit current K/V so the native kernel reads each write once.
            result = self._paged_kernel(
                q.reshape(count, batch.queries.maximum, *q.shape[1:]),
                k,
                v,
                cache_seqlens=lengths,
                softmax_scale=scale,
                causal=batch.causal[0],
                **{self._table_argument: batch.block_table.indices.to(dtype=torch.int32)},
                **self._paged_options,
            )
            return result.reshape_as(q)
        return self._paged_varlen(q, k, v, batch, lengths, scale)

    def _paged_varlen(self, q, k, v, batch, lengths, scale):
        # The varlen kernel consumes cumulative KV offsets; build them from
        # the per-sequence lengths held in workspace.
        offsets = self.workspace["offsets"][: lengths.numel() + 1]
        offsets[0].zero_()
        torch.cumsum(lengths, dim=0, out=offsets[1:])

        return self._varlen_kernel(
            q,
            k,
            v,
            batch.queries.offsets,
            offsets,
            q.shape[0],
            batch.block_table.indices.shape[1] * batch.block_table.block_size,
            softmax_scale=scale,
            causal=batch.causal[0],
            block_table=batch.block_table.indices.to(dtype=torch.int32),
            **self._varlen_options,
        )


class Backend(_Backend):
    operator_class = _FlashOperator

    def workspace_buffers(self, *, num_heads, num_kv_heads, head_dim, dtype, size, cache):
        return {
            "lengths": BufferConfig((size.batch_size,), torch.int32),
            "offsets": BufferConfig((size.batch_size + 1,), torch.int32),
        }
