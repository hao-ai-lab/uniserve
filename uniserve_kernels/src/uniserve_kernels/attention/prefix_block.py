"""Block attention over a paged prefix window on SM100 (CuTe DSL).

A batch holds ``B`` sequences. Sequence ``b`` contributes the query block
``query[query_offsets[b]:query_offsets[b + 1]]`` of ``L_b`` rows, its own
current keys and values at the same rows of ``key``/``value``, and a
read-only prefix of ``P_b = prefix_lengths[b]`` tokens stored in paged
caches. Query row ``i`` of the block attends to prefix tokens
``[lower_b(i), P_b)`` and to current keys: without causal ordering to all
``L_b`` of them, with it (``causal``: a chunk continuing its history, whose
row ``i`` is absolute position ``P_b + i``) to current keys ``[0, i]``.
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
(head dimension 512), and 16, 32 or 64 tokens per page. Causal ordering
requires head dimension 512 and no history window.
:func:`unsupported_configuration` checks these dimensions before any tensor
exists; :func:`can_run` checks a call without launching it.

Non-causal blocks run the kernel of :mod:`._prefix_block_kernel`, whose
persistent clusters stride statically over the work tiles. Causal chunks
run the kernel of :mod:`._causal_block_kernel`, whose clusters take a
static first wave and then claim tickets from a counter in ``workspace``
(see :func:`new_workspace`), sized for the batch's sequences. The counter
must be zero when a causal launch starts, and the launch leaves it
nonzero: the launch that precedes it on the stream resets it, such as
:func:`uniserve_kernels.attention.paged.prepare` with its ``semaphore``.
Launches that may run concurrently need distinct workspaces.

Each distinct specialization (the executor cache key) compiles once per
process on first use; :func:`prefix_block_attention` refuses to compile
during CUDA graph capture, so every specialization must run once before
capture.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from uniserve_kernels.triton import dependent_launch

_IMPORT_ERROR: BaseException | None = None
_cute: Any | None
_from_dlpack: Any | None
_cuda_driver: Any | None
_kernel_type: Any | None
_causal_kernel_type: Any | None
try:  # pragma: no cover - CUDA-only provider.
    import cuda.bindings.driver as _cuda_driver_module
    import cutlass.cute as _cute_module
    from cutlass.cute.runtime import from_dlpack as _dlpack_converter

    from ._causal_block_kernel import CausalBlockAttentionSm100
    from ._prefix_block_kernel import PrefixBlockAttentionSm100
except BaseException as error:  # pragma: no cover
    _IMPORT_ERROR = error
    _cute = None
    _from_dlpack = None
    _cuda_driver = None
    _kernel_type = None
    _causal_kernel_type = None
else:  # pragma: no cover
    _cute = _cute_module
    _from_dlpack = _dlpack_converter
    _cuda_driver = _cuda_driver_module
    _kernel_type = PrefixBlockAttentionSm100
    _causal_kernel_type = CausalBlockAttentionSm100

_EXECUTORS: dict[tuple[object, ...], Callable[..., None]] = {}
_SM_COUNTS: dict[int, int] = {}

HEAD_DIMS = (256, 512)
PAGE_TOKENS = (16, 32, 64)
# Packed rows of one work tile, computed by a two-CTA cluster, by head
# dimension, widest first: a 512-wide accumulator allows 64 rows per CTA;
# at head dim 256 the 128-row tile serves batches too small to occupy the
# GPU with 256-row tiles.
_TILE_ROWS = {256: (256, 128), 512: (128,)}
# int32 words of a causal launch workspace besides two per sequence: the
# ticket counter. The per-sequence words hold each sequence's first dynamic
# ticket and tile count.
_COUNTER_WORDS = 1


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


def workspace_words(sequences: int) -> int:
    """int32 words of a causal launch workspace for up to ``sequences``."""
    return _COUNTER_WORDS + 2 * sequences


def new_workspace(device: torch.device, sequences: int) -> torch.Tensor:
    """Return a zeroed causal launch workspace for up to ``sequences``.

    Its first word is the ticket counter, which a causal launch leaves
    nonzero; the launch that precedes the next one on the stream resets it
    (see the module docstring).
    """
    return torch.zeros(
        workspace_words(sequences), dtype=torch.int32, device=device
    )


def unsupported_configuration(
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    page_tokens: int,
    dtype: torch.dtype,
    causal: bool = False,
    window: int | None = None,
) -> str | None:
    """Return why the kernel cannot serve these dimensions, or None.

    ``dtype`` is the element type of Q/K/V, the caches and the output;
    ``causal`` and ``window`` are the call's block ordering and history
    window. The check needs no tensor or device, so a caller can decide
    which kernel serves a layer before its inputs exist; :func:`can_run`
    additionally checks a concrete call's layouts and device.
    """
    if dtype != torch.bfloat16:
        return "query, key, value and caches must be BF16"
    if head_dim not in HEAD_DIMS:
        return f"head_dim {head_dim} is not one of {HEAD_DIMS}"
    if kv_heads < 1 or query_heads % kv_heads:
        return "query heads must be a multiple of KV heads"
    group = query_heads // kv_heads
    if (_TILE_ROWS[head_dim][0] // 2) % group:
        return (
            "query heads per KV head must divide "
            f"{_TILE_ROWS[head_dim][0] // 2} at head_dim {head_dim}"
        )
    if page_tokens not in PAGE_TOKENS:
        return f"page_tokens {page_tokens} is not one of {PAGE_TOKENS}"
    if causal and (head_dim != 512 or window is not None):
        # A causal chunk longer than a window would need a lower bound
        # within the block, and only full-attention head dimension 512
        # layers are served causally.
        return "causal block attention requires head_dim 512 and no window"
    return None


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
    causal: bool = False,
    workspace: torch.Tensor | None = None,
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
    pages, page_tokens, cache_heads, cache_dim = key_cache.shape
    problem = unsupported_configuration(
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_tokens=page_tokens,
        dtype=query.dtype,
        causal=causal,
        window=window,
    )
    if problem is not None:
        return problem
    if key.shape != (tokens, kv_heads, head_dim) or value.shape != key.shape:
        return "key and value must be [tokens, kv_heads, head_dim]"
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
    if workspace is not None and (
        workspace.ndim != 1
        or workspace.shape[0] < workspace_words(batch)
        or workspace.dtype != torch.int32
        or workspace.device != device
        or workspace.stride(0) != 1
    ):
        return (
            "workspace must be a contiguous int32 tensor of at least "
            "workspace_words(sequences) words on the query device"
        )
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
    causal: bool = False,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> bool:
    """Return whether :func:`prefix_block_attention` accepts the call.

    Checks the device, the optional dependencies and every tensor's shape,
    dtype, device and layout; device values are not inspected. A missing
    ``workspace`` is not checked.
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
            causal,
            workspace,
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


def _tile_rows(
    head_dim: int, group: int, blocks: int, block_rows: int, clusters: int
) -> int:
    """Rows per work tile for ``blocks`` (sequence, KV head) row blocks.

    Each block holds up to ``block_rows`` packed rows. A narrower tile,
    whose CTA rows ``group`` divides, is used when its tiles still run in
    one wave of the ``clusters`` two-CTA clusters: the work then spreads over
    more clusters and each runs a shorter tile. Otherwise the widest tile,
    which uses the tensor cores and memory best, is used.
    """
    rows = _TILE_ROWS[head_dim]
    for narrow in rows[1:]:
        tiles = blocks * -(-block_rows // narrow)
        if (narrow // 2) % group == 0 and tiles <= clusters:
            return narrow
    return rows[0]


def _compile(
    specialization: dict[str, object],
    arguments: tuple[torch.Tensor | None, ...],
) -> Callable[..., None]:
    """Compile one kernel specialization for the ABI of ``arguments``."""
    return _cute.compile(
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
    causal: bool = False,
    workspace: torch.Tensor | None = None,
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
        causal: Whether block row ``i`` sees only current keys ``[0, i]``.
        workspace: The int32 workspace of a causal launch for at least
            ``batch`` sequences (:func:`new_workspace`), whose ticket counter
            a preceding launch on the stream has zeroed; not shared with a
            concurrently running launch. Non-causal launches take none.
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
            specialization is first seen during CUDA graph capture. The
            first call of a specialization (dtypes, head and page shape,
            window kind, optional inputs) compiles it for every batch size,
            so one call before capture suffices for all captured batches.
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
        causal,
        workspace,
    )
    if problem is None and causal and workspace is None:
        problem = "causal launches need a workspace"
    if problem is not None:
        raise ValueError(problem)
    if type(max_query_len) is not int or max_query_len < 0:
        raise ValueError("max_query_len must be a nonnegative integer")
    if out is None:
        out = torch.empty_like(query)

    batch = prefix_lengths.shape[0]
    kv_heads = key.shape[1]
    group = query.shape[1] // kv_heads
    if batch == 0 or max_query_len == 0:
        return out
    if causal:
        _causal_attention(
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
            workspace,
            max_query_len=max_query_len,
            scale=scale,
            lse_base2=lse_base2,
        )
        return out
    clusters = _sm_count(query.device) // 2
    tile_rows = _tile_rows(
        query.shape[2], group, batch * kv_heads, max_query_len * group, clusters
    )
    num_m_blocks = max(1, -(-max_query_len * group // tile_rows))
    total_tiles = batch * kv_heads * num_m_blocks
    num_clusters = max(1, min(total_tiles, clusters))

    specialization = {
        "head_dim": query.shape[2],
        "tile_rows": tile_rows,
        "group_size": group,
        "page_tokens": key_cache.shape[1],
        "query_window": bool(query_window) and window is not None,
        "has_window": window is not None,
        "has_prefix_start": prefix_start is not None,
        "has_start_page": start_page is not None,
        "has_lse": lse is not None,
        "lse_base2": bool(lse_base2),
        "pdl": dependent_launch(query.device),
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

    def cache_key(variant: dict[str, object]) -> tuple[object, ...]:
        return (
            device_index,
            torch.cuda.get_device_capability(query.device),
            tuple(sorted(variant.items())),
            *(_abi(tensor) for tensor in arguments),
        )

    with torch.cuda.device(query.device):
        executor = _EXECUTORS.get(cache_key(specialization))
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
            # The batch shape selects the tile rows, so every tile this call's
            # specialization can run with is compiled together; a graph
            # captured at another batch size then finds its executor.
            for rows in _TILE_ROWS[specialization["head_dim"]]:
                if (rows // 2) % group:
                    continue
                variant = dict(specialization, tile_rows=rows)
                if cache_key(variant) not in _EXECUTORS:
                    _EXECUTORS[cache_key(variant)] = _compile(
                        variant, arguments
                    )
            executor = _EXECUTORS[cache_key(specialization)]

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


def _causal_attention(
    *arguments: torch.Tensor | None,
    max_query_len: int,
    scale: float,
    lse_base2: bool,
) -> None:
    """Launch the causal kernel on validated arguments.

    ``arguments`` are the kernel's tensors in :func:`prefix_block_attention`
    order followed by ``out``, ``lse`` and ``workspace``. Its clusters take
    a static first wave of the grid bounded by ``max_query_len`` and claim
    the remaining tiles dynamically; see :mod:`._causal_block_kernel`.
    """
    query, key, _value, key_cache = arguments[:4]
    start_page, prefix_start = arguments[8:10]
    lse = arguments[11]
    batch = arguments[7].shape[0]
    kv_heads = key.shape[1]
    group = query.shape[1] // kv_heads
    clusters = _sm_count(query.device) // 2
    tile_rows = _tile_rows(
        query.shape[2], group, batch * kv_heads, max_query_len * group, clusters
    )
    # The grid of (sequence, row block up to the bound, KV head) tiles; no
    # launch runs more clusters than it has tiles. When every tile has its
    # own cluster, the kernel exchanges no work item.
    row_blocks = max(1, -(-max_query_len * group // tile_rows))
    grid_tiles = batch * kv_heads * row_blocks
    num_clusters = max(1, min(grid_tiles, clusters))

    specialization = {
        "head_dim": query.shape[2],
        "tile_rows": tile_rows,
        "group_size": group,
        "page_tokens": key_cache.shape[1],
        "has_prefix_start": prefix_start is not None,
        "has_start_page": start_page is not None,
        "has_lse": lse is not None,
        "lse_base2": bool(lse_base2),
    }
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    def cache_key(variant: dict[str, object]) -> tuple[object, ...]:
        return (
            "causal",
            device_index,
            torch.cuda.get_device_capability(query.device),
            tuple(sorted(variant.items())),
            *(_abi(tensor) for tensor in arguments),
        )

    with torch.cuda.device(query.device):
        executor = _EXECUTORS.get(cache_key(specialization))
        if executor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "prefix-block attention was not compiled before graph "
                    "capture"
                )
            if _cute is None or _causal_kernel_type is None:
                raise RuntimeError(
                    "the SM100 prefix-block attention kernel is unavailable"
                ) from _IMPORT_ERROR
            executor = _cute.compile(
                _causal_kernel_type(**specialization),
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
                1,
                1,
                1,
                _cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
            _EXECUTORS[cache_key(specialization)] = executor

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
            num_clusters,
            row_blocks,
            int(grid_tiles <= clusters),
            # Launch on torch's current stream so the call orders with the
            # surrounding work and records into an active capture.
            _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream),
        )


__all__ = [
    "HEAD_DIMS",
    "PAGE_TOKENS",
    "available",
    "can_run",
    "import_error",
    "new_workspace",
    "prefix_block_attention",
    "unsupported_configuration",
    "workspace_words",
]
