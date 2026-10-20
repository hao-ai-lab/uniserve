"""FlashInfer's SM120 block-sparse attention kernel on strided views.

The kernel, ``BlockSparseAttnForwardSm120Blk64``, selects 64-row tiles in
both the query and the key domain and reads row-major ``[rows, heads,
dim]`` query, key, value and output through strides given at run time.

FlashInfer's public ``bsa_attn_sm120_blk64_fwd`` copies the query, key and
value to contiguous storage before it launches that kernel. Segment
attention passes strided views of one ``[rows, q | k | v | gate]``
projection matrix, so those copies would move the three projections
through device memory at every layer. ``launch`` repeats the public
function's launch on the caller's views instead, with the same compile
cache and key. It reaches into the public function's module for that:
``_prepare_sm120_sparse_metadata``, ``_sm120_compile_cache``,
``torch2cute_dtype_map``, ``to_cute_tensor`` and the kernel class. These
are not a stable API, so the in-place launch runs only under the FlashInfer
release whose launch it repeats, ``FLASHINFER_VERSION``, and only on views
whose alignment the kernel admits. Every other call goes through the
public function, which computes the same values after its copies.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - CUDA provider dependency.
    from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
        bsa_attn_sm120_blk64_fwd,
    )
except BaseException:  # pragma: no cover
    bsa_attn_sm120_blk64_fwd = None

try:  # pragma: no cover - the internals the in-place launch reuses.
    import cuda.bindings.driver as _driver
    import cutlass
    import flashinfer
    from flashinfer.cute_dsl.sparse import bsa_attn_sm120 as _wrapper
    from flashinfer.cute_dsl.sparse.bsa_utils.cute_tensor_utils import (
        to_cute_tensor as _to_cute_tensor,
    )
    from flashinfer.cute_dsl.sparse.sm120_blk64.flash_fwd_sm120 import (
        BlockSparseAttnForwardSm120Blk64 as _Kernel,
    )

    _installed_version: str | None = flashinfer.__version__
except BaseException:  # pragma: no cover
    _installed_version = None
    _wrapper = None

# The FlashInfer release whose ``bsa_attn_sm120_blk64_fwd`` launch
# ``launch`` repeats; the ``gpu`` extra pins ``flashinfer-python`` to it.
FLASHINFER_VERSION = "0.6.18"

# The kernel's selection tile in rows.
TILE = 64

# The kernel assumes 128-byte aligned query, key, value and output bases and
# loads rows in 16-byte vectors.
_BASE_ALIGNMENT = 128
_ROW_ALIGNMENT = 16


def available(device):
    """Report whether ``device`` is an SM120-family GPU the kernel serves."""
    if bsa_attn_sm120_blk64_fwd is None:
        return False
    return torch.cuda.get_device_capability(device)[0] == 12


def _in_place(value):
    """Whether the kernel reads ``[rows, heads, 128]`` ``value`` as it is."""
    size = value.element_size()
    return (
        value.shape[-1] == 128
        and value.stride(-1) == 1
        and value.data_ptr() % _BASE_ALIGNMENT == 0
        and all(
            value.stride(axis) * size % _ROW_ALIGNMENT == 0 for axis in (0, 1)
        )
    )


def _launches_in_place(q, k, v):
    """Whether ``launch`` attends ``q``, ``k`` and ``v`` without copies.

    That needs the public function's internals of ``FLASHINFER_VERSION``
    and views the kernel's alignment admits.
    """
    return (
        _wrapper is not None
        and _installed_version == FLASHINFER_VERSION
        and all(_in_place(value) for value in (q, k, v))
    )


def launch(q, k, v, indices, counts, sizes, *, scale, out, lse):
    """Attend ``[rows, heads, 128]`` views through the compiled kernel.

    ``indices`` and ``counts`` are the ``[heads, query tiles, ...]``
    selection in kernel tiles and ``sizes`` the valid rows of every key
    tile. This is the launch ``bsa_attn_sm120_blk64_fwd`` makes, with the
    same compile cache and key, less its contiguous copies of ``q``, ``k``
    and ``v``: every view keeps its own row and head strides.
    """
    if not _launches_in_place(q, k, v):
        bsa_attn_sm120_blk64_fwd(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            indices.unsqueeze(0),
            indices.shape[-1],
            block_sizes=sizes,
            q2k_block_nums=counts.unsqueeze(0),
            softmax_scale=scale,
            out=out.unsqueeze(0),
            lse=lse,
        )
        return

    q, k, v, out = (value.unsqueeze(0) for value in (q, k, v, out))
    heads, dim = int(q.shape[2]), int(q.shape[3])
    major, minor = torch.cuda.get_device_capability(q.device)
    (
        has_block_nums,
        has_block_sizes,
        block_sizes_mode,
        q2k_t,
        q2k_nums_t,
        block_sizes_t,
    ) = _wrapper._prepare_sm120_sparse_metadata(
        indices.unsqueeze(0),
        counts.unsqueeze(0),
        sizes,
        int(indices.shape[-1]),
        batch_size=1,
        num_heads=heads,
        num_q_blocks=-(-int(q.shape[1]) // TILE),
        num_kv_blocks=-(-int(k.shape[1]) // TILE),
        device=q.device,
    )
    # (B, S, H, D) to the kernel's (S, D, H, B), V's (D, S, H, B) and the
    # log-sum-exp's (S, H, B), all as views.
    q_t, k_t, out_t = (value.permute(1, 3, 2, 0) for value in (q, k, out))
    v_t = v.permute(3, 1, 2, 0)
    lse_t = lse.permute(2, 1, 0)
    tensors = [
        _to_cute_tensor(
            value,
            assumed_align=align,
            leading_dim=leading,
            enable_tvm_ffi=False,
        )
        for value, align, leading in (
            (q_t, 128, 1),
            (k_t, 128, 1),
            (v_t, 128, 0),
            (out_t, 128, 1),
            (lse_t, 4, 0),
            (q2k_t, None, 0),
            (q2k_nums_t, None, 0),
            (block_sizes_t, None, 0),
        )
    ]
    stream = _driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    args = (
        *tensors[:7],
        cutlass.Int32(int(indices.shape[-1])),
        tensors[7],
        cutlass.Float32(scale),
        stream,
    )
    key = (
        "sm120_blk64_fwd",
        major * 10 + minor,
        q.dtype,
        dim,
        dim,
        1,
        bool(has_block_nums),
        bool(has_block_sizes),
        int(block_sizes_mode),
        q_t.stride(),
        k_t.stride(),
        v_t.stride(),
        out_t.stride(),
        lse_t.stride(),
        q2k_t.stride(),
        q2k_nums_t.stride(),
        block_sizes_t.stride(),
    )
    cache = _wrapper._sm120_compile_cache
    if key not in cache:
        import cutlass.cute as cute

        kernel = _Kernel(
            gqa_ratio=1,
            head_dim=dim,
            value_dim=dim,
            blocksparse_blocksize_q=TILE,
            blocksparse_blocksize_k=TILE,
            dtype=_wrapper.torch2cute_dtype_map[q.dtype],
            acc_dtype=cutlass.Float32,
            has_block_sizes=has_block_sizes,
            has_block_nums=has_block_nums,
            block_sizes_mode=block_sizes_mode,
        )
        cache[key] = cute.compile(kernel, *args)
    cache[key](*args)
