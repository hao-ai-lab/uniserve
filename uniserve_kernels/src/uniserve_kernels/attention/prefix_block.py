"""Block attention over a paged prefix window on SM100 (CuTe DSL).

A batch holds ``B`` sequences. Sequence ``b`` contributes the query block
``query[query_offsets[b]:query_offsets[b + 1]]`` of ``L_b`` rows, its own
current keys and values at the same rows of ``key``/``value``, and a
read-only prefix of ``P_b = prefix_lengths[b]`` tokens stored in paged
caches. Query row ``i`` of the block attends without causal ordering to all
``L_b`` current keys and to prefix tokens ``[lower_b(i), P_b)``, where
``lower_b(i)`` is the largest of:

* ``prefix_start[b]`` when that column is given, otherwise 0;
* ``P_b + i - window`` when ``query_window`` is set (an image block whose
  history window follows each query's absolute position ``P_b + i``), or
  ``P_b - window`` otherwise (a canvas read of a fixed window), when a
  ``window`` is given; ``window`` counts history tokens.

Prefix token ``t`` of sequence ``b`` lives in cache page
``block_table[b, t // page_tokens - start_page[b]]`` at row
``t % page_tokens`` (``start_page`` defaults to zero). The kernel reads
only the pages that intersect ``[min_i lower_b(i), P_b)``: other table
entries, other pages, and the unwritten rows at or past ``P_b`` of the last
page may hold anything, NaN included, without affecting the result. The
rows of the first read page that precede the window are loaded and masked;
as written prefix tokens they must be finite. Likewise, packed rows after
the last sequence are never read.

All lengths, offsets and tables are device tensors, so a launch records
into a CUDA graph and replays with new values in the same buffers. The host
supplies ``max_query_len``, an upper bound of every ``L_b`` that sizes the
persistent grid; replays must keep lengths within it. The optional
log-sum-exp output is base 2 when ``lse_base2`` is set and natural
otherwise, matching the ``base2`` flag of
:func:`uniserve_kernels.attention.merge.merge_attention_states`.

Supported configuration: CUDA compute capability 10.x, BF16 Q/K/V, caches
and output, head dimension 256 or 512, query heads a multiple of KV heads
with the query heads per KV head dividing 128 (head dimension 256) or 64
(head dimension 512), and 16, 32 or 64 tokens per page. :func:`can_run`
checks a call without launching it.

Each distinct specialization (the executor cache key) compiles once per
process on first use; :func:`prefix_block_attention` refuses to compile
during CUDA graph capture, so every specialization must run once before
capture.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

_IMPORT_ERROR: BaseException | None = None
_cute: Any | None
_from_dlpack: Any | None
_cuda_driver: Any | None
_kernel_type: Any | None
try:  # pragma: no cover - CUDA-only provider.
    import cuda.bindings.driver as _cuda_driver_module
    import cutlass.cute as _cute_module
    from cutlass.cute.runtime import from_dlpack as _dlpack_converter

    from ._prefix_block_kernel import PrefixBlockAttentionSm100
except BaseException as error:  # pragma: no cover
    _IMPORT_ERROR = error
    _cute = None
    _from_dlpack = None
    _cuda_driver = None
    _kernel_type = None
else:  # pragma: no cover
    _cute = _cute_module
    _from_dlpack = _dlpack_converter
    _cuda_driver = _cuda_driver_module
    _kernel_type = PrefixBlockAttentionSm100

_EXECUTORS: dict[tuple[object, ...], Callable[..., None]] = {}
_SM_COUNTS: dict[int, int] = {}

HEAD_DIMS = (256, 512)
PAGE_TOKENS = (16, 32, 64)
# Packed rows of one work tile, computed by a two-CTA cluster, by head
# dimension: a 512-wide accumulator allows 64 rows per CTA.
_TILE_ROWS = {256: 256, 512: 128}


def available(device: torch.device | None = None) -> bool:
    """Report whether the kernel can compile and run on ``device``.

    Requires the CUTLASS DSL and CUDA driver bindings and a compute
    capability 10.x device. Nothing is compiled.
    """
    if _kernel_type is None or not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major == 10


def import_error() -> BaseException | None:
    """Return the exception that prevented kernel registration, if any."""
    return _IMPORT_ERROR


def _check(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_offsets: torch.Tensor,
    prefix_lengths: torch.Tensor,
    start_page: torch.Tensor | None,
    prefix_start: torch.Tensor | None,
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    window: int | None,
) -> str | None:
    """Return why the call violates the kernel contract, or None."""
    tensors = (query, key, value, key_cache, value_cache)
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        return "query, key, value and caches must be BF16"
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        return "query, key and value must be [tokens, heads, head_dim]"
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        return "caches must share one [pages, page_tokens, heads, dim] shape"

    tokens, query_heads, head_dim = query.shape
    kv_heads = key.shape[1]
    if head_dim not in HEAD_DIMS:
        return f"head_dim {head_dim} is not one of {HEAD_DIMS}"
    if key.shape != (tokens, kv_heads, head_dim) or value.shape != key.shape:
        return "key and value must be [tokens, kv_heads, head_dim]"
    if kv_heads < 1 or query_heads % kv_heads:
        return "query heads must be a multiple of KV heads"
    group = query_heads // kv_heads
    if (_TILE_ROWS[head_dim] // 2) % group:
        return (
            "query heads per KV head must divide "
            f"{_TILE_ROWS[head_dim] // 2} at head_dim {head_dim}"
        )
    pages, page_tokens, cache_heads, cache_dim = key_cache.shape
    if page_tokens not in PAGE_TOKENS:
        return f"page_tokens {page_tokens} is not one of {PAGE_TOKENS}"
    if cache_heads != kv_heads or cache_dim != head_dim or pages < 1:
        return "caches must match the KV heads and head dimension"

    device = query.device
    if not query.is_cuda or any(
        tensor.device != device
        for tensor in (
            key,
            value,
            key_cache,
            value_cache,
            block_table,
            query_offsets,
            prefix_lengths,
        )
    ):
        return "all tensors must share one CUDA device"
    # TMA requires a unit innermost stride, 16-byte aligned base addresses
    # and 16-byte multiples for the other strides.
    for tensor in tensors:
        if tensor.stride(-1) != 1 or tensor.data_ptr() % 16:
            return "Q/K/V and caches need a contiguous 16-byte aligned dim"
        if any(stride % 8 for stride in tensor.stride()[:-1]):
            return "Q/K/V and cache strides must be 16-byte multiples"

    batch = prefix_lengths.shape[0]
    columns = [prefix_lengths, start_page, prefix_start]
    if any(
        column is not None
        and (
            column.shape != (batch,)
            or column.dtype != torch.int32
            or column.device != device
            or column.stride(0) != 1
        )
        for column in columns
    ):
        return "length columns must be contiguous int32 [batch] tensors"
    if (
        query_offsets.shape != (batch + 1,)
        or query_offsets.dtype != torch.int32
        or query_offsets.stride(0) != 1
    ):
        return "query_offsets must be a contiguous int32 [batch + 1] tensor"
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] < 1
        or block_table.dtype != torch.int32
        or block_table.stride(1) != 1
    ):
        return "block_table must be int32 [batch, width] with unit stride"

    if out is not None and (
        out.shape != query.shape
        or out.dtype != torch.bfloat16
        or out.device != device
        or out.stride(-1) != 1
        or out.data_ptr() % 16
        or any(stride % 8 for stride in out.stride()[:-1])
    ):
        return "out must be a 16-byte aligned BF16 tensor shaped like query"
    if lse is not None and (
        lse.shape != (tokens, query_heads)
        or lse.dtype != torch.float32
        or lse.device != device
        or lse.stride(-1) != 1
    ):
        return "lse must be FP32 [tokens, query_heads] with unit head stride"
    if window is not None and (type(window) is not int or window < 0):
        return "window must be a nonnegative history token count"
    return None


def can_run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_offsets: torch.Tensor,
    prefix_lengths: torch.Tensor,
    *,
    start_page: torch.Tensor | None = None,
    prefix_start: torch.Tensor | None = None,
    window: int | None = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> bool:
    """Return whether :func:`prefix_block_attention` accepts the call.

    Checks the device, the optional dependencies and every tensor's shape,
    dtype, device and layout; device values are not inspected.
    """
    return available(query.device) and (
        _check(
            query,
            key,
            value,
            key_cache,
            value_cache,
            block_table,
            query_offsets,
            prefix_lengths,
            start_page,
            prefix_start,
            out,
            lse,
            window,
        )
        is None
    )


def _dynamic_tensor(tensor: torch.Tensor, assumed_align: int = 16) -> Any:
    """Wrap a tensor as a CuTe argument with runtime shape and strides.

    The last dimension is marked as the unit-stride one; ``assumed_align``
    promises the base-pointer alignment in bytes.
    """
    if _from_dlpack is None:
        raise RuntimeError(
            "CuTe tensor ingress is unavailable"
        ) from _IMPORT_ERROR
    return _from_dlpack(
        tensor.detach(),
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def _abi(tensor: torch.Tensor | None) -> tuple[object, ...] | None:
    """Layout properties an executor is specialized on.

    Shapes and non-unit strides stay runtime values; the dtype, rank and
    the unit innermost stride are fixed by the compiled executor.
    """
    if tensor is None:
        return None
    return (tensor.dtype, tensor.ndim, int(tensor.stride(-1)))


def _sm_count(device: torch.device) -> int:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    count = _SM_COUNTS.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNTS[index] = count
    return count


def prefix_block_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_offsets: torch.Tensor,
    prefix_lengths: torch.Tensor,
    *,
    max_query_len: int,
    window: int | None = None,
    query_window: bool = False,
    start_page: torch.Tensor | None = None,
    prefix_start: torch.Tensor | None = None,
    scale: float = 1.0,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    lse_base2: bool = False,
) -> torch.Tensor:
    """Attend each query block to itself and its paged prefix window.

    See the module docstring for the visible key set of each row.

    Args:
        query: BF16 ``[tokens, query_heads, head_dim]`` packed query rows.
        key: BF16 ``[tokens, kv_heads, head_dim]`` current block keys, at the
            query rows. Query head ``h`` reads KV head ``h // G``.
        value: BF16 tensor shaped like ``key``.
        key_cache: BF16 ``[pages, page_tokens, kv_heads, head_dim]``.
        value_cache: BF16 tensor shaped like ``key_cache``.
        block_table: int32 ``[batch, width]`` physical page IDs.
        query_offsets: int32 ``[batch + 1]`` cumulative block row offsets.
        prefix_lengths: int32 ``[batch]`` prefix token counts ``P_b``.
        max_query_len: Host upper bound of every block length; sizes the
            grid.
        window: History tokens visible before a query, or None for the
            whole prefix.
        query_window: Whether the window follows each query position.
        start_page: Optional int32 ``[batch]`` logical page of column 0.
        prefix_start: Optional int32 ``[batch]`` lower bound of the prefix.
        scale: Softmax scale applied to the scores.
        out: Optional output tensor shaped like ``query``; rows outside the
            query blocks are not written.
        lse: Optional FP32 ``[tokens, query_heads]`` log-sum-exp output of
            the scaled scores; rows outside the query blocks are not
            written.
        lse_base2: Store base-2 instead of natural log-sum-exp.

    Returns:
        ``out``, or a new tensor shaped like ``query`` when it is None.

    Raises:
        RuntimeError: If the kernel is unavailable, or if an uncompiled
            specialization is first seen during CUDA graph capture.
        ValueError: If the call violates the kernel contract.
    """
    if not available(query.device):
        raise RuntimeError(
            "the SM100 prefix-block attention kernel is unavailable"
        ) from _IMPORT_ERROR
    problem = _check(
        query,
        key,
        value,
        key_cache,
        value_cache,
        block_table,
        query_offsets,
        prefix_lengths,
        start_page,
        prefix_start,
        out,
        lse,
        window,
    )
    if problem is not None:
        raise ValueError(problem)
    if type(max_query_len) is not int or max_query_len < 0:
        raise ValueError("max_query_len must be a nonnegative integer")
    if out is None:
        out = torch.empty_like(query)

    batch = prefix_lengths.shape[0]
    kv_heads = key.shape[1]
    group = query.shape[1] // kv_heads
    tile_rows = _TILE_ROWS[query.shape[2]]
    num_m_blocks = max(1, -(-max_query_len * group // tile_rows))
    total_tiles = batch * kv_heads * num_m_blocks
    if batch == 0 or max_query_len == 0:
        return out
    num_clusters = max(1, min(total_tiles, _sm_count(query.device) // 2))

    specialization = {
        "head_dim": query.shape[2],
        "group_size": group,
        "page_tokens": key_cache.shape[1],
        "query_window": bool(query_window) and window is not None,
        "has_window": window is not None,
        "has_prefix_start": prefix_start is not None,
        "has_start_page": start_page is not None,
        "has_lse": lse is not None,
        "lse_base2": bool(lse_base2),
    }
    arguments = (
        query,
        key,
        value,
        key_cache,
        value_cache,
        block_table,
        query_offsets,
        prefix_lengths,
        start_page,
        prefix_start,
        out,
        lse,
    )
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cache_key = (
        device_index,
        torch.cuda.get_device_capability(query.device),
        tuple(sorted(specialization.items())),
        *(_abi(tensor) for tensor in arguments),
    )

    with torch.cuda.device(query.device):
        executor = _EXECUTORS.get(cache_key)
        if executor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "prefix-block attention was not compiled before graph "
                    "capture"
                )
            if _cute is None or _kernel_type is None:
                raise RuntimeError(
                    "the SM100 prefix-block attention kernel is unavailable"
                ) from _IMPORT_ERROR
            executor = _cute.compile(
                _kernel_type(**specialization),
                *(
                    None
                    if tensor is None
                    else _dynamic_tensor(
                        tensor,
                        16 if tensor.dtype == torch.bfloat16 else 4,
                    )
                    for tensor in arguments
                ),
                1.0,
                0,
                1,
                1,
                # The stream is an explicit argument of every launch.
                _cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
            _EXECUTORS[cache_key] = executor

        if _cuda_driver is None:
            raise RuntimeError(
                "CUDA driver bindings are unavailable"
            ) from _IMPORT_ERROR
        executor(
            *(
                None if tensor is None else tensor.detach()
                for tensor in arguments
            ),
            float(scale),
            0 if window is None else int(window),
            num_m_blocks,
            num_clusters,
            # Launch on torch's current stream so the call orders with the
            # surrounding work and records into an active capture.
            _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream),
        )
    return out


__all__ = [
    "HEAD_DIMS",
    "PAGE_TOKENS",
    "available",
    "can_run",
    "import_error",
    "prefix_block_attention",
]
