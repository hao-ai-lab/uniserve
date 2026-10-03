"""FlashAttention-4 over packed sequences and borrowed paged prefix state."""

from __future__ import annotations

from functools import partial

import torch
from uniserve_kernels.attention.merge import merge_attention_states

from uniserve.nn.attention.inputs import (
    DenseInput,
    PagedInput,
    SegmentedInput,
    VarlenInput,
    VisibleInput,
)
from uniserve.quantization import QuantizedTensor
from uniserve.tensors import BufferConfig

from . import Backend as _Backend
from . import Operator as _Operator
from ._sequences import causal_runs


def available() -> bool:
    """Report whether the FlashAttention-4 CuTe runtime can be imported."""
    try:
        _flash_attn_forward()
    except ImportError:
        return False
    return True


def _flash_attn_forward():
    """Return FlashAttention-4's forward entry, raising if it is absent."""
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
    except Exception as error:
        raise ImportError(
            "FlashAttention-4 is unavailable; install uniserve-kernels with "
            "its flash_attn extra"
        ) from error
    return _flash_attn_fwd


def _visible_options(
    batch, keys, *, query_capacity, key_capacity, segmented=False
):
    from uniserve_kernels.attention.visible_end import visible_end_mask

    visible = batch.visible_current_end if segmented else batch.visible_end
    complete = batch.fully_visible_current if segmented else batch.fully_visible
    if complete:
        return {}

    visible = visible.to(dtype=torch.int32).contiguous()
    # The mask reads one sequence-local endpoint. Its inner stride and integer
    # alignment must survive CuTe's dynamic tensor conversion.
    visible.__leading_dim__ = 1
    visible.__assumed_align__ = 4

    result = {"aux_tensors": [visible], "mask_mod": visible_end_mask}
    architecture = torch.cuda.get_device_capability(visible.device)[0]
    if (
        not segmented
        and batch.prefix_bounds
        and batch.block_table is None
        and architecture in (9, 10, 11)
    ):
        # Native block-sparse traversal consumes contiguous K/V. Paged K/V
        # retains the identical elementwise visibility mask above.
        from uniserve_kernels.attention.prefix_bounds import (
            prefix_block_sparsity,
        )

        result.update(
            pack_gqa=False,
            tile_mn=(128, 128),
            num_threads=384,
            block_sparse_tensors=prefix_block_sparsity(
                visible,
                query_lengths=batch.queries.values,
                key_lengths=keys.values,
                max_key_length=key_capacity,
                # Match the compiled SM100 query staging to its full launch
                # capacity; live sequence boundaries can change on replay.
                query_tile=(
                    256
                    if architecture in (10, 11) and query_capacity > 128
                    else 128
                ),
                key_tile=128,
                variable_length=True,
            ),
        )
    return result


def _lse(result):
    output, lse = result[:2]
    if lse is None or lse.shape != (output.shape[1], output.shape[0]):
        raise RuntimeError(
            "FlashAttention-4 returned an incompatible packed LSE layout"
        )
    return output, lse.transpose(0, 1).contiguous()


class _FlashAttentionOperator(_Operator):
    # The kernels read every length, offset and table on the device; ``bind``
    # only checks the batch.
    builds_launch_plan = False

    def requires_host_lengths(self, batch):
        if isinstance(batch, PagedInput) and batch.causal_values is not None:
            return False
        return (
            isinstance(batch, (PagedInput, VarlenInput))
            and len(set(batch.causal)) > 1
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        flash_attn_fwd = _flash_attn_forward()
        if self.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(
                "FlashAttention-4 requires FP16 or BF16 computation"
            )
        if self.window is not None:
            raise ValueError(
                "FlashAttention-4 attention does not take history windows"
            )
        if self.cache is not None and isinstance(
            self.cache.key, QuantizedTensor
        ):
            raise ValueError(
                "FlashAttention-4 does not consume per-block FP8 prefix scales"
            )
        # Partial channel tiles use the ordinary epilogue: the packed-GQA
        # store predicate does not match its vector fragment for these widths.
        # Grouped attention remains native, with explicit query-head traversal.
        self._forward = (
            partial(flash_attn_fwd, pack_gqa=False)
            if self.head_dim % 32
            else flash_attn_fwd
        )

    def __call__(self, q, k, v, batch, *, scale, out):
        if not q.is_cuda:
            raise ValueError("FlashAttention-4 requires CUDA tensors")
        if not isinstance(
            batch, (DenseInput, VarlenInput)
        ) and torch.cuda.get_device_capability(q.device)[0] in (8, 12):
            raise ValueError(
                "paged FlashAttention-4 requires an SM90 or SM100 family GPU"
            )
        if isinstance(batch, DenseInput) and batch.mask is not None:
            raise ValueError(
                "FlashAttention-4 does not consume arbitrary dense masks"
            )
        if self.head_dim == 256 and isinstance(
            batch, (PagedInput, VisibleInput, SegmentedInput)
        ):
            # The dedicated head-dimension-256 kernel rejects per-sequence
            # key lengths and mask functions, which these inputs require.
            raise ValueError(
                "FlashAttention-4 head dimension 256 supports only dense and "
                "variable-length inputs"
            )
        self._validate(q, k, v, batch, out)

        if (
            isinstance(batch, (PagedInput, SegmentedInput))
            and batch.write_indices is not None
        ):
            self.update_cache(k, v, indices=batch.write_indices)

        if q.numel() == 0:
            return out

        if isinstance(batch, DenseInput):
            packed = q.ndim == 3
            query, key, value = (
                tensor.unsqueeze(0) if packed else tensor.transpose(1, 2)
                for tensor in (q, k, v)
            )
            destination = out.unsqueeze(0) if packed else out.transpose(1, 2)
            self._forward(
                query,
                key,
                value,
                softmax_scale=scale,
                causal=batch.causal,
                out=destination,
            )
            return out

        if q.ndim != 3 or (
            batch.queries.num_tokens is not None
            and q.shape[0] != batch.queries.num_tokens
        ):
            raise ValueError(
                "packed attention rows must match the declared query lengths"
            )

        if isinstance(batch, PagedInput) and batch.causal_values is not None:
            self._sequence(q, k, v, batch, scale, out)
        elif isinstance(batch, (PagedInput, VarlenInput)):
            for query_slice, key_slice, run in causal_runs(batch):
                if query_slice.stop == query_slice.start:
                    continue
                self._sequence(
                    q[query_slice],
                    k[key_slice],
                    v[key_slice],
                    run,
                    scale,
                    out[query_slice],
                )
        elif isinstance(batch, VisibleInput):
            key, value = (
                (k, v)
                if batch.block_table is None
                else self._cache_values(k, v)
            )

            used = None if batch.block_table is None else batch.keys.values
            options = {}
            if not batch.fully_visible and batch.visible_end.shape[1] == 1:
                # A shared endpoint bounds the sequence's keys: the kernel
                # reads it as the used key count and evaluates no row mask.
                used = self.workspace["lengths"][: batch.queries.batch_size]
                torch.minimum(
                    batch.visible_end.reshape(-1).to(dtype=torch.int32),
                    batch.keys.values,
                    out=used,
                )
            else:
                options = _visible_options(
                    batch,
                    batch.keys,
                    query_capacity=q.shape[0],
                    key_capacity=k.shape[0],
                )

            self._forward(
                q,
                key,
                value,
                cu_seqlens_q=batch.queries.offsets,
                cu_seqlens_k=batch.keys.offsets
                if batch.block_table is None
                else None,
                page_table=None
                if batch.block_table is None
                else batch.block_table.indices,
                seqused_k=used,
                max_seqlen_q=q.shape[0],
                max_seqlen_k=(
                    k.shape[0]
                    if batch.block_table is None
                    else batch.block_table.indices.shape[1]
                    * batch.block_table.block_size
                ),
                softmax_scale=scale,
                out=out,
                **options,
            )
        elif isinstance(batch, SegmentedInput):
            if self.cache is None:
                raise RuntimeError(
                    "segmented attention requires bound prefix state"
                )

            # Evaluate the current window and the paged prefix independently,
            # then merge both partial states with online softmax.
            common = {
                "cu_seqlens_q": batch.queries.offsets,
                "max_seqlen_q": q.shape[0],
                "softmax_scale": scale,
                "tile_mn": (128, 128),
                "num_threads": 384,
                "return_lse": True,
            }
            current = _lse(
                self._forward(
                    q,
                    k,
                    v,
                    cu_seqlens_k=batch.queries.offsets,
                    max_seqlen_k=k.shape[0],
                    **common,
                    **_visible_options(
                        batch,
                        batch.queries,
                        query_capacity=q.shape[0],
                        key_capacity=k.shape[0],
                        segmented=True,
                    ),
                )
            )
            if batch.block_table.indices.shape[1] == 0:
                return out.copy_(current[0])

            prefix = _lse(
                self._forward(
                    q,
                    self.cache.key,
                    self.cache.value,
                    page_table=batch.block_table.indices,
                    seqused_k=batch.prefixes.values,
                    max_seqlen_k=batch.block_table.indices.shape[1]
                    * batch.block_table.block_size,
                    **common,
                )
            )
            merged, _ = merge_attention_states(*current, *prefix)
            out.copy_(merged)
        else:
            raise TypeError("unsupported numerical attention input")
        return out

    def _cache_values(self, k, v):
        return (
            (k, v) if self.cache is None else (self.cache.key, self.cache.value)
        )

    def _sequence(self, q, k, v, batch, scale, out):
        common = {
            "cu_seqlens_q": batch.queries.offsets,
            "max_seqlen_q": q.shape[0],
            "softmax_scale": scale,
            "causal": batch.causal[0],
            "out": out,
        }

        if isinstance(batch, PagedInput):
            if batch.causal_values is not None:
                from uniserve_kernels.attention.visible_end import (
                    paged_causal_mask,
                )

                flags, prefixes = batch.causal_values, batch.prefixes.values
                for column in (flags, prefixes):
                    column.__leading_dim__ = 0
                    column.__assumed_align__ = 4
                common.update(
                    causal=False,
                    mask_mod=paged_causal_mask,
                    aux_tensors=[flags, prefixes],
                )
            key, value = self._cache_values(k, v)
            lengths = self.workspace["lengths"][: batch.queries.batch_size]
            torch.add(batch.prefixes.values, batch.queries.values, out=lengths)

            self._forward(
                q,
                key,
                value,
                page_table=batch.block_table.indices.to(dtype=torch.int32),
                seqused_k=lengths,
                # Capacity bounds the launch while device lengths select the
                # live prefix after a graph replay changes sequence lengths.
                max_seqlen_k=batch.block_table.indices.shape[1]
                * batch.block_table.block_size,
                tile_mn=(128, 128),
                num_threads=384,
                **common,
            )
        else:
            self._forward(
                q,
                k,
                v,
                cu_seqlens_k=batch.keys.offsets,
                max_seqlen_k=k.shape[0],
                **common,
            )


class Backend(_Backend):
    name = "flash_attn_4"

    operator_class = _FlashAttentionOperator

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
        return {"lengths": BufferConfig((size.batch_size,), torch.int32)}
