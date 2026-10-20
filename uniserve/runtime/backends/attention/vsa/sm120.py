"""SM120 block attention with FlashInfer's CuTe block-sparse kernel.

The kernel reads the caller's row-major ``[rows, heads, dim]`` query, key,
value and output directly, so a call makes no head-first copies. Its
selection unit is a 64-row tile in both the query and the key domain.

FlashInfer's ``bsa_attn_sm120_blk64_fwd`` copies the query, key and value
to contiguous storage before the launch, although the kernel it compiles
takes their strides at run time. Segment attention passes strided views of
one ``[rows, q | k | v | gate]`` projection matrix, so those copies would
move the three projections through device memory at every layer. ``_launch``
calls the same compiled kernel on the caller's views instead, and uses the
public wrapper only for views the kernel's alignment does not admit.

128-row tiles (the ref2va FastH3 students) attend through the same kernel.
Each 128-row query tile becomes its two 64-row halves, each selecting both
64-row halves of every 128-row key tile the full tile selects, and a key
tile's valid rows, which lead the tile, split between its halves (the
first holds ``min(valid, 64)``, the second the rest). Every query row
therefore attends exactly the valid keys of its 128-row selection, and the
kernel's online softmax gives an empty half no weight.
"""

from __future__ import annotations

import torch

from uniserve.tensors import BufferConfig

from . import Backend as BaseBackend
from . import Operator as BaseOperator

try:  # pragma: no cover - CUDA provider dependencies.
    import triton
    import triton.language as tl
    from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
        bsa_attn_sm120_blk64_fwd,
    )
except BaseException:  # pragma: no cover
    triton = None
    bsa_attn_sm120_blk64_fwd = None

try:  # pragma: no cover - the wrapper internals the strided launch reuses.
    import cuda.bindings.driver as _driver
    import cutlass
    from flashinfer.cute_dsl.sparse import bsa_attn_sm120 as _wrapper
    from flashinfer.cute_dsl.sparse.bsa_utils.cute_tensor_utils import (
        to_cute_tensor as _to_cute_tensor,
    )
    from flashinfer.cute_dsl.sparse.sm120_blk64.flash_fwd_sm120 import (
        BlockSparseAttnForwardSm120Blk64 as _Kernel,
    )
except BaseException:  # pragma: no cover
    _wrapper = None

# The kernel's selection tile in rows.
_KERNEL_TILE = 64

# The kernel assumes 128-byte aligned query, key, value and output bases and
# loads rows in 16-byte vectors.
_BASE_ALIGNMENT = 128
_ROW_ALIGNMENT = 16

# Tile sizes this provider serves: the kernel's own and 128-row tiles split
# into kernel tiles.
TILES = (64, 128)


def available(device):
    """Report whether ``device`` is an SM120-family GPU the kernel serves."""
    if bsa_attn_sm120_blk64_fwd is None or triton is None:
        return False
    return torch.cuda.get_device_capability(device)[0] == 12


if triton is not None:

    @triton.jit
    def _split_selection_kernel(
        source_indices,
        source_counts,
        indices,
        counts,
        source_stride_head,
        source_stride_tile,
        count_stride_head,
        stride_head,
        stride_tile,
        query_tiles,
        width,
        block: tl.constexpr,
    ):
        """Write one 128-row query tile's selection as two 64-row rows.

        Selected key tile ``j`` becomes kernel tiles ``2j`` and ``2j + 1``,
        and both query halves select the same key tiles; entries past the
        live count are copied but never read, since the count bounds them.
        Shapes and strides are runtime values, so every layout shares one
        compiled kernel.
        """
        row = tl.program_id(0)
        head = row // query_tiles
        tile = row % query_tiles
        count = tl.load(source_counts + head * count_stride_head + tile)
        for half in tl.static_range(2):
            tl.store(
                counts + head * (2 * query_tiles) + 2 * tile + half,
                2 * count,
            )
        for start in range(0, width, block):
            offsets = start + tl.arange(0, block)
            active = offsets < width
            selected = tl.load(
                source_indices
                + head * source_stride_head
                + tile * source_stride_tile
                + offsets,
                mask=active,
                other=0,
            )
            for half in tl.static_range(2):
                destination = (
                    indices
                    + head * stride_head
                    + (2 * tile + half) * stride_tile
                    + 2 * offsets
                )
                tl.store(destination, 2 * selected, mask=active)
                tl.store(destination + 1, 2 * selected + 1, mask=active)

    @triton.jit
    def _split_sizes_kernel(
        source_sizes, sizes, key_tiles, block: tl.constexpr
    ):
        """Split each 128-row key tile's leading valid rows between halves."""
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        active = offsets < key_tiles
        valid = tl.load(source_sizes + offsets, mask=active, other=0)
        tl.store(sizes + 2 * offsets, tl.minimum(valid, 64), mask=active)
        tl.store(
            sizes + 2 * offsets + 1, tl.maximum(valid - 64, 0), mask=active
        )


_SPLIT_BLOCK = 128


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


def _launch(q, k, v, indices, counts, sizes, *, scale, out, lse):
    """Attend ``[rows, heads, 128]`` views through the compiled kernel.

    ``indices`` and ``counts`` are the ``[heads, query tiles, ...]``
    selection in kernel tiles and ``sizes`` the valid rows of every key
    tile. This is the launch ``bsa_attn_sm120_blk64_fwd`` makes, with the
    same compile cache and key, less its contiguous copies of ``q``, ``k``
    and ``v``: every view keeps its own row and head strides.
    """
    if _wrapper is None or not all(_in_place(value) for value in (q, k, v)):
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
        num_q_blocks=-(-int(q.shape[1]) // _KERNEL_TILE),
        num_kv_blocks=-(-int(k.shape[1]) // _KERNEL_TILE),
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
            blocksparse_blocksize_q=_KERNEL_TILE,
            blocksparse_blocksize_k=_KERNEL_TILE,
            dtype=_wrapper.torch2cute_dtype_map[q.dtype],
            acc_dtype=cutlass.Float32,
            has_block_sizes=has_block_sizes,
            has_block_nums=has_block_nums,
            block_sizes_mode=block_sizes_mode,
        )
        cache[key] = cute.compile(kernel, *args)
    cache[key](*args)


class _Operator(BaseOperator):
    def __call__(self, q, k, v, batch, *, scale, out):
        self._validate(q, k, v, batch, out)

        indices, counts, sizes = (
            batch.q2k_index,
            batch.q2k_num,
            batch.variable_block_sizes,
        )
        heads, query_tiles, width = (int(size) for size in indices.shape)
        key_tiles = int(sizes.numel())
        split = self.pattern.tile != _KERNEL_TILE

        # The kernel writes the log-sum-exp of every row; a 128-row selection
        # also needs its kernel-tile form. Both live for this call only.
        requirements = {
            "lse": BufferConfig((1, heads, q.shape[0]), torch.float32)
        }
        if split:
            requirements |= {
                "indices": BufferConfig(
                    (heads, 2 * query_tiles, 2 * width), torch.int32
                ),
                "counts": BufferConfig((heads, 2 * query_tiles), torch.int32),
                "sizes": BufferConfig((2 * key_tiles,), torch.int32),
            }
        work = self._work(requirements, q.device)
        if split:
            _split_selection_kernel[(heads * query_tiles,)](
                indices,
                counts,
                work["indices"],
                work["counts"],
                int(indices.stride(0)),
                int(indices.stride(1)),
                int(counts.stride(0)),
                int(work["indices"].stride(0)),
                int(work["indices"].stride(1)),
                query_tiles,
                width,
                _SPLIT_BLOCK,
                num_warps=4,
                num_stages=1,
            )
            _split_sizes_kernel[(triton.cdiv(key_tiles, _SPLIT_BLOCK),)](
                sizes, work["sizes"], key_tiles, _SPLIT_BLOCK, num_warps=4
            )
            indices, counts, sizes = (
                work["indices"],
                work["counts"],
                work["sizes"],
            )

        _launch(
            q,
            k,
            v,
            indices,
            counts,
            sizes,
            scale=scale,
            out=out,
            lse=work["lse"],
        )
        return out

    def _work(self, requirements, device):
        # Every layer of the owning context runs on its serialized stream, so
        # they can share one backing for these per-call work areas.
        if self.transient is not None:
            return self.transient("vsa_sm120", requirements, device)
        return {
            name: torch.empty(config.shape, dtype=config.dtype, device=device)
            for name, config in requirements.items()
        }


class Backend(BaseBackend):
    name = "sm120"
    operator_class = _Operator
