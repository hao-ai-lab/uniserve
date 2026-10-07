"""TensorRT-LLM paged MHA kernels with borrowed execution workspace."""

import torch

from uniserve.nn.attention.inputs import PagedInput
from uniserve.quantization import QuantizedTensor
from uniserve.tensors import BufferConfig

from . import Backend as _Backend
from . import Operator as _Operator
from ._sequences import causal_runs

__all__ = ["Backend"]


def _head_major(value):
    result = value.permute(0, 2, 1, 3)
    if result.shape[1] == 1 and result.stride(1) == result.stride(2):
        # A singleton head axis has no address contribution. Canonicalizing
        # its stride preserves values and satisfies the native HND descriptor.
        strides = list(result.stride())
        strides[1] = result.shape[2] * strides[2]
        result = result.as_strided(result.shape, strides)
    return result


def _require_windowed_causal(window, batch):
    """Reject windows the context kernels cannot evaluate exactly.

    The TensorRT-LLM context kernels implement a history window only with
    the causal mask; a non-causal windowed query would silently read keys
    outside its window.
    """
    if window is not None and not all(batch.causal):
        raise ValueError(
            "TensorRT-LLM MHA supports history windows only for causal queries"
        )


class _TRTLLM(_Operator):
    # A table with start pages is read from its first column: the prepared
    # key lengths count each row's keys from its table origin, and the
    # kernels align queries to the end of their keys, so causal positions
    # and the history window keep their relative meaning while retired
    # pages, which no query of the row may read, are never addressed.
    reads_retired_tables = True
    # The kernels read every length, offset and table origin on the
    # device; ``bind`` only checks the batch.
    builds_launch_plan = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        if self.head_dim not in {64, 128, 256, 512}:
            raise ValueError(
                "TensorRT-LLM MHA requires head dimension 64, 128, 256 or 512"
            )
        if self.head_dim == 512 and self.window is not None:
            # The pinned SM100 cubins provide head dimension 512 context
            # kernels only for causal and dense masks, not sliding windows.
            raise ValueError(
                "TensorRT-LLM MHA has no windowed head dimension 512 kernels"
            )
        if self.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("TensorRT-LLM MHA requires FP16 or BF16 queries")
        if self.cache is not None and isinstance(
            self.cache.key, QuantizedTensor
        ):
            raise ValueError(
                "TensorRT-LLM MHA does not consume per-block FP8 prefix scales"
            )
        self._decode = trtllm_batch_decode_with_kv_cache
        self._prefill = trtllm_batch_context_with_kv_cache
        self.workspace["scratch"].zero_()

    def bind(self, batch):
        super().bind(batch)
        if not isinstance(batch, PagedInput):
            raise ValueError("TensorRT-LLM MHA requires paged attention input")
        if batch.causal_values is not None:
            raise ValueError(
                "TensorRT-LLM MHA does not implement device causality flags"
            )
        _require_windowed_causal(self.window, batch)

    def requires_host_lengths(self, batch):
        return isinstance(batch, PagedInput) and len(set(batch.causal)) > 1

    def __call__(self, q, k, v, batch, *, scale, out):
        from ._paged_inputs import prepare

        if not isinstance(batch, PagedInput):
            raise ValueError("TensorRT-LLM MHA requires paged attention input")
        _require_windowed_causal(self.window, batch)
        if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] != 10:
            raise ValueError("TensorRT-LLM MHA requires an SM100 family GPU")
        self._validate(q, k, v, batch, out)

        if q.shape[0] == 0:
            if batch.write_indices is not None:
                self.update_cache(k, v, indices=batch.write_indices)
            return out

        if q.ndim != 3 or (
            batch.queries.num_tokens is not None
            and q.shape[0] != batch.queries.num_tokens
        ):
            raise ValueError(
                "packed attention rows must match their declared query lengths"
            )

        key, value = (
            (k, v) if self.cache is None else (self.cache.key, self.cache.value)
        )
        if key.ndim != 4:
            raise ValueError(
                "TensorRT-LLM MHA requires physical paged K/V backing"
            )
        cache = (_head_major(key), _head_major(value))

        # Native stores require contiguous, aligned backing. An aliased
        # destination must not overwrite queries or shared prefix values until
        # every causal run has read them.
        direct = (
            out.is_contiguous()
            and out.data_ptr() % 16 == 0
            and not any(
                torch._C._overlaps(out, tensor)
                for tensor in (q, k, v, key, value)
            )
        )
        destination = (
            out
            if direct
            else torch.empty(out.shape, dtype=out.dtype, device=out.device)
        )

        if len(set(batch.causal)) > 1 and batch.write_indices is not None:
            # Causal runs slice query domains. Commit the complete write once
            # before those runs, whose inputs then carry no write indices.
            self.update_cache(k, v, indices=batch.write_indices)

        for query_slice, _, run in causal_runs(batch):
            query = q[query_slice]
            if not query.shape[0]:
                continue

            lengths = self.workspace["lengths"][: run.queries.batch_size]
            offsets = self.workspace["offsets"][: run.queries.batch_size + 1]
            prepare(self.cache, k, v, run, lengths=lengths, offsets=offsets)

            # The table's capacity is invariant across replay; actual key
            # lengths remain device inputs, including a growing decode prefix.
            maximum = (
                run.block_table.indices.shape[1] * run.block_table.block_size
            )
            options = {
                "query": query.contiguous(),
                "kv_cache": cache,
                "workspace_buffer": self.workspace["scratch"],
                "block_tables": run.block_table.indices.to(
                    dtype=torch.int32
                ).contiguous(),
                "seq_lens": lengths,
                "bmm1_scale": float(scale),
                "bmm2_scale": 1.0,
                # Keys from query position - window_left on stay visible.
                "window_left": -1 if self.window is None else self.window,
                "out_dtype": q.dtype,
                "out": destination[query_slice],
                "kv_layout": "HND",
            }
            if run.queries.host is not None and all(
                count == 1 for count in run.queries.host
            ):
                self._decode(max_seq_len=maximum, **options)
            else:
                self._prefill(
                    max_q_len=query.shape[0],
                    max_kv_len=maximum,
                    batch_size=run.queries.batch_size,
                    cum_seq_lens_q=run.queries.offsets,
                    cum_seq_lens_kv=offsets,
                    causal=run.causal[0],
                    **options,
                )

        if destination is not out:
            out.copy_(destination)
        return out


class Backend(_Backend):
    name = "trtllm"

    operator_class = _TRTLLM

    def __init__(self, *, workspace_size=512 * 1024 * 1024):
        if workspace_size < 1:
            raise ValueError("TensorRT-LLM workspace size must be positive")
        self.workspace_size = workspace_size

    def workspace_buffers(
        self,
        *,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype,
        size,
        cache,
        window=None,
    ):
        return {
            "scratch": BufferConfig((self.workspace_size,), torch.uint8),
            "lengths": BufferConfig((size.batch_size,), torch.int32),
            "offsets": BufferConfig((size.batch_size + 1,), torch.int32),
        }
