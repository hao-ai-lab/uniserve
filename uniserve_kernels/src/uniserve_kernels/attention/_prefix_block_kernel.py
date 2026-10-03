"""SM100 attention of query blocks over a paged prefix window and themselves.

Each sequence ``b`` contributes a block of ``L_b`` query rows. A query row
attends to every key of its own block (the packed current K/V rows), or to
its causal/windowed current keys when its device flag is set, and to the
keys ``[lower, P_b)`` of the sequence's paged
prefix. ``lower`` is fixed for the sequence (canvas reads) or follows the
query position (image blocks); see :class:`PrefixBlockAttentionSm100`.

Work decomposition. Query heads that share a KV head are packed into the row
dimension: packed row ``r`` of a sequence is query position ``r // G`` of
query head ``kv_head * G + r % G``. A work tile is a block of packed rows of
one sequence and one KV head, computed by a two-CTA cluster that issues
``tcgen05`` two-SM MMAs, so each CTA stages only half of every K/V tile. The
persistent grid walks work tiles with a static stride; tiles beyond a
sequence's device length are skipped, so one launch serves every length
column it is replayed with.

Head-dimension shapes. With head dimension 256 a work tile is 256 rows
(128 per CTA, the full tensor-memory lane count), the output accumulator
takes 256 of the 512 tensor-memory columns and two 128-column score stages
the rest; BF16 probabilities overwrite their scores and feed the PV MMA
from tensor memory. With head dimension 512 a 128-row accumulator per CTA
would fill all 512 columns, so a work tile is 128 rows (64 per CTA): the
two-SM M = 128 MMA stores each CTA's 64 rows in the "2x2" layout, the first
half of the columns in lanes 0-63 and the second half in lanes 64-127, which
keeps every data path busy while halving the columns (256 for the output,
64 per score stage). Each row's scores are then split between two threads
(lanes ``r`` and ``r + 64``), which exchange row maxima through shared
memory, and the probabilities reach the PV MMA through shared memory. Head
dimension 256 also runs 128-row tiles in this layout (the ``tile_rows``
specialization), which the launcher selects when a batch is too small to
occupy the GPU with 256-row tiles.

Key tiles. A work tile streams 128-key tiles: prefix tiles first, then the
tiles of its own block. Prefix tiles start at the page containing the
tile's smallest lower bound; each is assembled from ``128 / page_tokens``
page slots. Each CTA stages the 64 keys of its K half per 128-wide head-dim
chunk, and its contiguous head-dim half of V per pair of chunks and 64-key
half of the tile. Every stage is 64 rows by two 64-dim swizzle atoms,
filled by TMA boxes of (64 dims, rows, 2 atoms): one box per page slot when
a page covers the stage's 64 rows, one per atom otherwise. The TMA global
view of the cache keeps the in-page row and the page as separate
dimensions, which makes two cases exact without reading unowned memory:

* the last, partially written page is loaded with its row coordinate
  shifted down by the unwritten tail length, so the TMA unit zero-fills the
  leading rows of the slot and the written rows land at its end;
* slots past the last page, or whose logical page falls outside the block
  table row, load an out-of-range page coordinate and are zero-filled.

The last tile of a block whose length is not a multiple of 128 is loaded
ending at the block's last row. Its leading rows repeat keys of the
previous tile or precede the block and are masked; they are rows of other
sequences in the packed K/V tensors or out-of-range zeros, never rows past
the packed sequences. Masked scores are replaced (not added to), so a
masked key never matters, and a masked value row contributes zero as long
as it is finite. The only masked rows read from memory are written tokens:
prefix tokens before the window in its first page, and other sequences'
current rows. Unused pages, unwritten page tails and trailing padding rows
may therefore hold NaN.

Warp roles (per CTA, 12 warps): softmax (0-3, one thread per tensor-memory
lane), output correction and epilogue (4-7), MMA issue (8, leader CTA
only), TMA loads (9), and two idle warps. K and V stages share one ring of
16 KiB shared-memory stages. The load warp looks up the page slots of a
prefix tile once, one slot per lane and one tile ahead of its copies. The
softmax raises the row maximum used for the probabilities only when a tile
exceeds it by more than a threshold, so the correction warps rescale the
output only for the few tiles that raise it. These mechanisms follow the
FlashAttention-4 SM100 forward kernels. At head dimension 256 the epilogue
stages each output chunk in shared memory so that global stores write
whole row segments instead of one 16-byte piece per row and thread.
"""

import math
from typing import no_type_check

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Boolean, Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

_LOG2_E = math.log2(math.e)
_LN_2 = math.log(2.0)
_TMEM_COLUMNS = 512
# Named barriers: 1 publishes the tensor-memory allocation; 2 and 3 pair
# softmax warps (0, 2) and (1, 3), whose lanes share rows at head dim 512;
# 4 orders the correction warps' accesses to the output staging buffer.
_TMEM_BARRIER = 1
_PAIR_BARRIER = 2
_OUTPUT_BARRIER = 4
# Dynamic SMEM bytes that Q, the ring, the probability tiles and the output
# staging buffer may use together: the 227 KiB per-CTA limit less 3 KiB for
# the barriers, row statistics and alignment.
_STAGING_SMEM_LIMIT = 224 * 1024
# Tensor-memory addresses carry the lane above bit 16 and the column below.
_TMEM_LANE = 1 << 16
# The running row maximum used for the probabilities is only raised when a
# tile exceeds it by more than this many log2 units, so most tiles need no
# output rescale. Probabilities then stay below 2**8, far inside the BF16
# range, and the row sum uses the same maximum, so the result is exact up to
# rounding.
_RESCALE_THRESHOLD = 8.0
# Scores are masked in fragments of 32 consecutive columns of one row: one
# 32-bit keep mask per fragment.
_FRAGMENT = 32


def _group(size: int) -> pipeline.CooperativeGroup:
    """Cooperative group of ``size`` threads for pipeline arrival counts."""
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


@dsl_user_op
def _keep_bits(low: Int32, high_shift: Int32, *, loc=None, ip=None) -> Uint32:
    """``(~0 << low) & (~0 >> high_shift)`` with PTX shift semantics.

    Both amounts must be non-negative; PTX clamps amounts above 31 to 32,
    which shifts every bit out, where an LLVM shift would be undefined.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(low).ir_value(loc=loc, ip=ip),
                Int32(high_shift).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .b32 ones, below, above;\n\t"
            "mov.b32 ones, 0xFFFFFFFF;\n\t"
            "shl.b32 above, ones, $1;\n\t"
            "shr.b32 below, ones, $2;\n\t"
            "and.b32 $0, above, below;\n\t"
            "}\n",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _interval_bits(start: Int32, end: Int32) -> Uint32:
    """Bits ``i`` of a fragment with ``start <= i < end``."""
    return _keep_bits(
        cutlass.max(start, Int32(0)), cutlass.max(_FRAGMENT - end, Int32(0))
    )


class PrefixBlockAttentionSm100:
    """Two-CTA SM100 kernel for block attention over a paged prefix window.

    Constructor arguments are compile-time specializations:

    Args:
        head_dim: Q/K/V head dimension; 256 or 512.
        tile_rows: Packed rows per cluster tile: 256 or 128 at head dim 256,
            128 at head dim 512.
        group_size: Query heads per KV head ``G``; must divide the CTA rows
            (``tile_rows / 2``).
        page_tokens: Tokens per cache page; 16, 32 or 64.
        query_window: If true, row ``i`` of a block reads prefix keys from
            ``P + i - window`` (an image block whose history window follows
            the query position); otherwise every row reads from
            ``P - window`` (a canvas read). Only meaningful with a window.
        has_window: Whether a history window bounds the prefix; without one
            the prefix is read from its start.
        has_prefix_start: Whether a device column raises each sequence's
            prefix lower bound further.
        has_start_page: Whether block table column ``j`` holds logical page
            ``start_page[b] + j`` rather than page ``j``.
        has_causal: Whether a device column selects causal current keys and
            their history window independently for each sequence.
        has_lse: Whether to store each row's log-sum-exp.
        lse_base2: Store base-2 LSE if true, natural LSE otherwise.
        pdl: Launch as a programmatic dependent of the preceding kernel
            (see ``uniserve_kernels.triton.dependent_launch``): every thread
            waits for the preceding grid after the TMA descriptor prefetch
            and before its first global memory access, then releases the
            next grid.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        tile_rows: int,
        group_size: int,
        page_tokens: int,
        query_window: bool,
        has_window: bool,
        has_prefix_start: bool,
        has_start_page: bool,
        has_causal: bool,
        has_lse: bool,
        lse_base2: bool,
        pdl: bool,
    ) -> None:
        if head_dim not in (256, 512):
            raise ValueError("head_dim must be 256 or 512")
        if tile_rows not in ((256, 128) if head_dim == 256 else (128,)):
            raise ValueError(
                "tile_rows must be 256 or 128 at head_dim 256 and 128 at "
                "head_dim 512"
            )
        if page_tokens not in (16, 32, 64):
            raise ValueError("page_tokens must be 16, 32 or 64")

        self.head_dim = head_dim
        self.group_size = group_size
        self.page_tokens = page_tokens
        self.query_window = query_window
        self.has_window = has_window
        self.has_prefix_start = has_prefix_start
        self.has_start_page = has_start_page
        self.has_causal = has_causal
        self.has_lse = has_lse
        self.lse_base2 = lse_base2
        self.pdl = pdl

        # Rows per cluster tile and per CTA; see the module docstring.
        self.tile_rows = tile_rows
        self.cta_rows = self.tile_rows // 2
        if group_size < 1 or self.cta_rows % group_size:
            raise ValueError("query heads per KV head must divide CTA rows")
        # Threads (tensor-memory lanes) per row: the 2x2 layout of a 64-row
        # accumulator puts the two column halves of a row in two lanes.
        self.row_split = 128 // self.cta_rows
        # Probabilities feed the PV MMA from tensor memory at 128 rows per
        # CTA and from shared memory at 64 rows per CTA.
        self.p_in_tmem = self.cta_rows == 128

        # The head dimension is consumed in 128-wide chunks so that every K
        # or V stage is 16 KiB per CTA; keys come in 128-key tiles.
        self.tile_keys = 128
        self.chunk = 128
        self.qk_mma_tiler = (self.tile_rows, self.tile_keys, self.chunk)
        self.pv_mma_tiler = (self.tile_rows, self.chunk, self.tile_keys)
        self.pv_block_tiler = (self.cta_rows, self.chunk)
        self.qk_chunks = head_dim // self.chunk
        self.pv_chunks = head_dim // self.chunk
        # Each CTA of the pair stages 64 keys of every K chunk (its half of
        # the QK B operand) and 64 head dims of every V chunk (its half of the
        # PV B operand). A K or V stage is 64 rows (keys) by two 64-wide
        # head-dim atoms, one 128-byte swizzle row per row and atom, so every
        # stage is filled by TMA boxes of (64 dims, rows, 2 atoms).
        self.atom_dims = 64
        self.half_keys = self.tile_keys // 2
        self.slots_per_tile = self.tile_keys // page_tokens
        self.slots_per_half = self.half_keys // page_tokens
        # A CTA's V head dims are the contiguous half [v * D / 2, (v + 1) *
        # D / 2): PV chunk c's column n is head dim (n // 64) * D / 2 + 64 c
        # + n % 64. A V stage holds two consecutive chunks of one 64-key
        # half of a tile, so a tile's V takes 2 * pv_pairs stages.
        self.pv_pairs = self.pv_chunks // 2

        # A tile streams qk_chunks K stages and 2 * pv_pairs V stages; eight
        # 16 KiB ring stages keep the loads a full tile ahead at head dim
        # 512.
        self.q_stage = self.qk_chunks
        self.kv_stage = 8
        self.s_stage = 2
        # A tensor-memory lane holds (part of) one output row, so storing
        # rows straight from registers writes 16 bytes to a different row
        # per thread. When SMEM has room next to Q, the ring and the
        # probability tiles, each output chunk goes through a staging buffer
        # and leaves in whole row segments; otherwise (head dim 512) it is
        # stored from registers.
        row_bytes = self.cta_rows * 2
        smem_bytes = (
            row_bytes * head_dim
            + self.kv_stage * 16 * 1024
            + (0 if self.p_in_tmem else self.s_stage * row_bytes * 128)
        )
        self.stage_output = (
            smem_bytes + row_bytes * self.chunk <= _STAGING_SMEM_LIMIT
        )

        self.softmax_warps = (0, 1, 2, 3)
        self.correction_warps = (4, 5, 6, 7)
        self.mma_warp = 8
        self.load_warp = 9
        self.idle_warps = (10, 11)
        self.threads = 32 * 12
        self.lane_threads = 128

        # Tensor-memory columns of one score stage and one output chunk, and
        # their placement: score stages from column 0, the output from 256.
        self.s_columns = self.tile_keys * self.cta_rows // 128
        self.o_chunk_columns = self.chunk * self.cta_rows // 128
        self.tmem_s_offset = 0
        self.tmem_o_offset = 256
        # BF16 probabilities kept in tensor memory occupy the first half of
        # their score stage. The softmax row statistics for the correction
        # warps take two columns of the stage after its scores are read:
        # after the probabilities, or at its start when they go to SMEM.
        self.p_columns = self.tile_keys // 2
        self.stats_column = self.p_columns if self.p_in_tmem else 0

        # Registers per thread of each warpgroup after setmaxnreg: softmax,
        # correction, and the MMA/load/idle group. Their sum times 128 must
        # stay within the launch allocation (168 per thread for 384 threads,
        # 64512), or the softmax increase never completes; the load and MMA
        # warps need the 88 to keep their unrolled issue loops in registers.
        self.regs_softmax = 256
        self.regs_correction = 160
        self.regs_other = 88

    # ------------------------------------------------------------------ host

    @no_type_check
    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        key_cache: cute.Tensor,
        value_cache: cute.Tensor,
        block_table: cute.Tensor,
        query_offsets: cute.Tensor,
        prefix_lengths: cute.Tensor,
        start_page: cute.Tensor | None,
        prefix_start: cute.Tensor | None,
        causal_rows: cute.Tensor | None,
        output: cute.Tensor,
        lse: cute.Tensor | None,
        scale: Float32,
        window: Int32,
        num_m_blocks: Int32,
        num_clusters: Int32,
        stream: cuda.CUstream,
    ):
        """Build tensor views and TMA descriptors, then launch the kernel.

        ``query``/``output`` are ``[tokens, Hq, D]``, ``key``/``value``
        ``[tokens, Hkv, D]``, caches ``[pages, page_tokens, Hkv, D]``,
        ``block_table`` ``[B, W]`` int32, ``query_offsets`` ``[B + 1]``,
        ``prefix_lengths``/``start_page``/``prefix_start`` ``[B]`` int32 and
        ``lse`` ``[tokens, Hq]`` FP32. ``num_m_blocks`` bounds the work tiles
        of any sequence and ``num_clusters`` sizes the grid.
        """
        dtype = query.element_type
        group = self.group_size
        tokens = query.shape[0]
        kv_heads = key.shape[1]
        head_dim = query.shape[2]

        # Packed query/output rows ((G, tokens), D, Hkv): row (g, t) is token
        # t of query head kv_head * G + g.
        q_packed = cute.make_tensor(
            query.iterator,
            cute.make_layout(
                ((group, tokens), head_dim, kv_heads),
                stride=(
                    (query.stride[1], query.stride[0]),
                    1,
                    query.stride[1] * group,
                ),
            ),
        )
        # The launcher guarantees 16-byte aligned rows; stating it lets the
        # epilogue store each thread's row segment with vector stores. The
        # head dim is viewed as ((64, 2), chunks) in the PV column order: the
        # 128 columns of chunk c are head dims 64 c + [0, 64) and D / 2 +
        # 64 c + [0, 64).
        o_head_stride = cute.assume(output.stride[1], divby=8)
        o_token_stride = cute.assume(output.stride[0], divby=8)
        o_packed = cute.make_tensor(
            output.iterator,
            cute.make_layout(
                (
                    (group, tokens),
                    ((self.atom_dims, 2), self.pv_chunks),
                    kv_heads,
                ),
                stride=(
                    (o_head_stride, o_token_stride),
                    ((1, self.head_dim // 2), self.atom_dims),
                    o_head_stride * group,
                ),
            ),
        )
        # K/V as (dim in atom, row, atom, head[, page]): current blocks from
        # the packed tokens and prefixes from the caches, whose in-page row
        # and page stay separate TMA dimensions.
        atoms = self.head_dim // self.atom_dims
        k_current = cute.make_tensor(
            key.iterator,
            cute.make_layout(
                (self.atom_dims, tokens, atoms, kv_heads),
                stride=(1, key.stride[0], self.atom_dims, key.stride[1]),
            ),
        )
        v_current = cute.make_tensor(
            value.iterator,
            cute.make_layout(
                (self.atom_dims, tokens, atoms, kv_heads),
                stride=(1, value.stride[0], self.atom_dims, value.stride[1]),
            ),
        )
        k_pages = cute.make_tensor(
            key_cache.iterator,
            cute.make_layout(
                (
                    self.atom_dims,
                    self.page_tokens,
                    atoms,
                    kv_heads,
                    key_cache.shape[0],
                ),
                stride=(
                    1,
                    key_cache.stride[1],
                    self.atom_dims,
                    key_cache.stride[2],
                    key_cache.stride[0],
                ),
            ),
        )
        v_pages = cute.make_tensor(
            value_cache.iterator,
            cute.make_layout(
                (
                    self.atom_dims,
                    self.page_tokens,
                    atoms,
                    kv_heads,
                    value_cache.shape[0],
                ),
                stride=(
                    1,
                    value_cache.stride[1],
                    self.atom_dims,
                    value_cache.stride[2],
                    value_cache.stride[0],
                ),
            ),
        )

        self.dtype = dtype
        self.o_dtype = output.element_type

        cta_group = tcgen05.CtaGroup.TWO
        qk_mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        # P is K-major, from tensor memory or shared memory; V is MN-major
        # (head dim contiguous).
        pv_mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            Float32,
            cta_group,
            self.pv_mma_tiler[:2],
            tcgen05.OperandSource.TMEM
            if self.p_in_tmem
            else tcgen05.OperandSource.SMEM,
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((2, 1, 1)), (qk_mma.thr_id.shape,)
        )

        q_smem = sm100_utils.make_smem_layout_a(
            qk_mma, self.qk_mma_tiler, dtype, self.q_stage
        )
        k_smem = sm100_utils.make_smem_layout_b(
            qk_mma, self.qk_mma_tiler, dtype, self.kv_stage
        )
        # V sub-stages of 64 keys (one head-dim atom each), two per stage.
        v_smem = sm100_utils.make_smem_layout_b(
            pv_mma,
            (self.tile_rows, self.chunk, self.half_keys),
            dtype,
            2 * self.kv_stage,
        )
        # One probability tile per score stage as the PV A operand, aliasing
        # the stage in tensor memory or in SMEM.
        p_layout = sm100_utils.make_smem_layout_a(
            pv_mma, self.pv_mma_tiler, dtype, self.s_stage
        )
        # Output staging buffer: this CTA's rows of one 128-column chunk,
        # row-major in 128-byte swizzle atoms.
        out_layout = sm100_utils.make_smem_layout_epi(
            self.o_dtype, utils.LayoutEnum.ROW_MAJOR, self.pv_block_tiler, 1
        )
        # K and V stages alias one ring: both are 16 KiB per CTA with the
        # same swizzle, and a stage holds one of them at a time. In both, row
        # (key) n and head dim d of a stage sit at element n * 64 + d % 64 +
        # (d // 64) * 4096, i.e. the stage is (dim in atom, row, atom) =
        # (64, 64, 2):(1, 64, 4096), the order of the TMA boxes below.
        assert cute.cosize(k_smem) == cute.cosize(v_smem)
        assert str(k_smem.inner) == str(v_smem.inner)
        ring = cute.make_composed_layout(
            k_smem.inner,
            0,
            cute.make_layout(
                (self.atom_dims, self.half_keys, 2),
                stride=(1, self.atom_dims, self.atom_dims * self.half_keys),
            ),
        )
        # A page slot is page_tokens rows of a stage: one TMA box when the
        # slot fills the stage, one per atom otherwise.
        slot = cute.composition(ring, (self.atom_dims, self.page_tokens, 2))
        slot_tile = (self.atom_dims, self.page_tokens, 2)
        stage_tile = (self.atom_dims, self.half_keys, 2)

        load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_q, tma_q_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            load_op,
            q_packed,
            cute.select(q_smem, mode=[0, 1, 2]),
            self.qk_mma_tiler,
            qk_mma,
            cluster_layout_vmnk.shape,
        )
        tma_kp, tma_kp_tensor = cpasync.make_tiled_tma_atom(
            load_op, k_pages, slot, slot_tile
        )
        tma_vp, tma_vp_tensor = cpasync.make_tiled_tma_atom(
            load_op, v_pages, slot, slot_tile
        )
        tma_kc, tma_kc_tensor = cpasync.make_tiled_tma_atom(
            load_op, k_current, ring, stage_tile
        )
        tma_vc, tma_vc_tensor = cpasync.make_tiled_tma_atom(
            load_op, v_current, ring, stage_tile
        )
        q_bytes = cute.size_in_bytes(dtype, cute.select(q_smem, mode=[0, 1, 2]))
        kv_bytes = cute.size_in_bytes(
            dtype, cute.select(k_smem, mode=[0, 1, 2])
        )
        # Both CTAs' TMA bytes land on the leader CTA's barriers.
        self.q_tx_bytes = q_bytes * 2
        self.kv_tx_bytes = kv_bytes * 2

        @cute.struct
        class SharedStorage:
            q_full_empty: cute.struct.MemRange[Int64, self.q_stage * 2]
            kv_full_empty: cute.struct.MemRange[Int64, self.kv_stage * 2]
            s_full_empty: cute.struct.MemRange[Int64, self.s_stage * 2]
            p_full_empty: cute.struct.MemRange[Int64, self.s_stage * 2]
            stats_full_empty: cute.struct.MemRange[Int64, self.s_stage * 2]
            sum_full_empty: cute.struct.MemRange[Int64, 2]
            o_full_empty: cute.struct.MemRange[Int64, 2]
            tmem_dealloc: Int64
            tmem_holding: Int32
            # Row sums for the epilogue, and the row-half exchange buffer
            # (two alternating parities of one value per lane).
            row_sums: cute.struct.MemRange[Float32, self.cta_rows]
            pair_values: cute.struct.MemRange[Float32, 2 * self.lane_threads]

        self.shared_storage = SharedStorage

        scale_log2 = scale * Float32(_LOG2_E)
        self.kernel(
            qk_mma,
            pv_mma,
            tma_q,
            tma_q_tensor,
            tma_kp,
            tma_kp_tensor,
            tma_vp,
            tma_vp_tensor,
            tma_kc,
            tma_kc_tensor,
            tma_vc,
            tma_vc_tensor,
            o_packed,
            lse,
            block_table,
            query_offsets,
            prefix_lengths,
            start_page,
            prefix_start,
            causal_rows,
            key_cache.shape[0],
            scale,
            scale_log2,
            window,
            num_m_blocks,
            cluster_layout_vmnk,
            q_smem,
            k_smem,
            v_smem,
            p_layout,
            out_layout,
            ring,
        ).launch(
            grid=[num_clusters * 2, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[2, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.pdl,
        )

    # ---------------------------------------------------------- tile metadata

    @no_type_check
    @cute.jit
    def work_tile(self, tile: Int32, schedule):
        """Decode a linear work tile and its sequence's key ranges.

        Tiles are ordered with the row block fastest, then KV head, then
        sequence, so concurrently running clusters share K/V in L2.

        Returns:
            ``(valid, batch, kv_head, m_block, query_start, query_len,
            prefix_len, first_page, prefix_base, prefix_tiles, key_tiles,
            lower_first, lower_last)``: ``valid`` is false for a tile past
            the sequence's rows; ``prefix_base`` is the first token of the
            first prefix tile (page aligned); ``lower_first`` and
            ``lower_last`` are the prefix lower bounds of the tile's first
            and last query positions.
        """
        (
            _cluster_id,
            _num_clusters,
            _total_tiles,
            num_m_blocks,
            kv_heads,
            query_offsets,
            prefix_lengths,
            start_page,
            prefix_start,
            window,
            causal_rows,
        ) = schedule
        m_block = tile % num_m_blocks
        rest = tile // num_m_blocks
        kv_head = rest % kv_heads
        batch = rest // kv_heads

        query_start = query_offsets[batch]
        query_len = query_offsets[batch + 1] - query_start
        prefix_len = prefix_lengths[batch]
        first_page = Int32(0)
        if const_expr(self.has_start_page):
            first_page = start_page[batch]

        valid = m_block * self.tile_rows < query_len * self.group_size

        first_query = m_block * (self.tile_rows // self.group_size)
        last_query = cutlass.min(
            first_query + self.tile_rows // self.group_size, query_len
        )
        last_query = cutlass.max(last_query - 1, first_query)
        lower_first = self.lower_bound(
            first_query, prefix_len, batch, prefix_start, window
        )
        lower_last = self.lower_bound(
            last_query, prefix_len, batch, prefix_start, window
        )

        # Prefix tiles cover pages [lower_first // page, ceil(P / page)).
        prefix_base = (lower_first // self.page_tokens) * self.page_tokens
        prefix_tiles = Int32(0)
        if lower_first < prefix_len:
            page_end = (prefix_len + self.page_tokens - 1) // self.page_tokens
            prefix_span = page_end * self.page_tokens - prefix_base
            prefix_tiles = (prefix_span + self.tile_keys - 1) // self.tile_keys
        current_tiles = (query_len + self.tile_keys - 1) // self.tile_keys
        if const_expr(self.has_causal):
            if causal_rows[batch] != 0:
                # No query in this work tile can see a later current tile.
                current_tiles = cutlass.min(
                    current_tiles,
                    (last_query + self.tile_keys) // self.tile_keys,
                )
        key_tiles = prefix_tiles + current_tiles
        return (
            valid,
            batch,
            kv_head,
            m_block,
            query_start,
            query_len,
            prefix_len,
            first_page,
            prefix_base,
            prefix_tiles,
            key_tiles,
            lower_first,
            lower_last,
        )

    @no_type_check
    @cute.jit
    def lower_bound(
        self,
        query_pos: Int32,
        prefix_len: Int32,
        batch: Int32,
        prefix_start: cute.Tensor | None,
        window: Int32,
    ) -> Int32:
        """First visible prefix token of block query position ``query_pos``.

        The bound is the larger of the optional explicit ``prefix_start``
        column and the history window's start: ``P + i - window`` when the
        window follows the query, ``P - window`` otherwise. It may exceed
        ``P``, in which case the row sees no prefix key.
        """
        lower = Int32(0)
        if const_expr(self.has_prefix_start):
            lower = prefix_start[batch]
        if const_expr(self.has_window):
            history = prefix_len - window
            if const_expr(self.query_window):
                history = history + query_pos
            lower = cutlass.max(lower, history)
        return lower

    # ------------------------------------------------------------------ kernel

    @no_type_check
    @cute.kernel
    def kernel(
        self,
        qk_mma: cute.TiledMma,
        pv_mma: cute.TiledMma,
        tma_q: cute.CopyAtom,
        tma_q_tensor: cute.Tensor,
        tma_kp: cute.CopyAtom,
        tma_kp_tensor: cute.Tensor,
        tma_vp: cute.CopyAtom,
        tma_vp_tensor: cute.Tensor,
        tma_kc: cute.CopyAtom,
        tma_kc_tensor: cute.Tensor,
        tma_vc: cute.CopyAtom,
        tma_vc_tensor: cute.Tensor,
        o_packed: cute.Tensor,
        lse: cute.Tensor | None,
        block_table: cute.Tensor,
        query_offsets: cute.Tensor,
        prefix_lengths: cute.Tensor,
        start_page: cute.Tensor | None,
        prefix_start: cute.Tensor | None,
        causal_rows: cute.Tensor | None,
        num_pages: Int32,
        scale: Float32,
        scale_log2: Float32,
        window: Int32,
        num_m_blocks: Int32,
        cluster_layout_vmnk: cute.Layout,
        q_smem: cute.ComposedLayout,
        k_smem: cute.ComposedLayout,
        v_smem: cute.ComposedLayout,
        p_layout: cute.ComposedLayout,
        out_layout: cute.ComposedLayout,
        ring: cute.ComposedLayout,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == self.load_warp:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_kp)
            cpasync.prefetch_descriptor(tma_vp)
            cpasync.prefetch_descriptor(tma_kc)
            cpasync.prefetch_descriptor(tma_vc)
        if const_expr(self.pdl):
            # The descriptors are launch parameters, not preceding outputs,
            # so their prefetch may overlap the preceding grid. Every thread
            # waits for that grid before the first global access (tile
            # metadata, page lookups, TMA loads and all stores), so it reads
            # complete inputs and overwrites nothing the grid still reads.
            # The release follows at once: the next grid becomes eligible
            # after every CTA has signaled or completed. Its own wait still
            # protects accesses to this grid's data until completion.
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        bidx, _, _ = cute.arch.block_idx()
        # Position of this CTA in the MMA pair: rows [v * R, v * R + R) of a
        # tile (R rows per CTA), keys [v * 64, v * 64 + 64) of each K tile
        # and head dims [v * 64, v * 64 + 64) of each V chunk.
        cta_v = bidx % 2
        cluster_id = bidx // 2
        num_clusters = cute.arch.grid_dim()[0] // 2
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
        is_leader = cta_rank % 2 == 0
        kv_heads = o_packed.shape[2]
        total_tiles = (query_offsets.shape[0] - 1) * kv_heads * num_m_blocks
        schedule = (
            cluster_id,
            num_clusters,
            total_tiles,
            num_m_blocks,
            kv_heads,
            query_offsets,
            prefix_lengths,
            start_page,
            prefix_start,
            window,
            causal_rows,
        )
        # The load warp reads its first work tile's metadata and page lookups
        # before the barrier and tensor-memory setup, which the other warps
        # carry out meanwhile; the other warps keep placeholder values.
        zero = Int32(0)
        first_lookups = (
            (Boolean(False),) + (zero,) * 12,
            (zero, zero),
            (zero, zero),
        )
        if warp == self.load_warp:
            first_lookups = self.tile_lookups(
                cluster_id, schedule, block_table, num_pages
            )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_warp_group = _group(1)
        mma_warp_group = _group(1)
        softmax_pair = _group(self.lane_threads * 2)
        correction_pair = _group(self.lane_threads * 2)
        q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=load_warp_group,
            consumer_group=mma_warp_group,
            tx_count=self.q_tx_bytes,
            barrier_storage=storage.q_full_empty.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        kv_producer, kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=load_warp_group,
            consumer_group=mma_warp_group,
            tx_count=self.kv_tx_bytes,
            barrier_storage=storage.kv_full_empty.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.s_stage,
            producer_group=mma_warp_group,
            consumer_group=softmax_pair,
            barrier_storage=storage.s_full_empty.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.s_stage,
            producer_group=softmax_pair,
            consumer_group=mma_warp_group,
            barrier_storage=storage.p_full_empty.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        stats_producer, stats_consumer = pipeline.PipelineAsync.create(
            num_stages=self.s_stage,
            producer_group=_group(self.lane_threads),
            consumer_group=_group(self.lane_threads),
            barrier_storage=storage.stats_full_empty.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=_group(self.lane_threads),
            consumer_group=_group(self.lane_threads),
            barrier_storage=storage.sum_full_empty.data_ptr(),
            defer_sync=True,
        ).make_participants()
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=mma_warp_group,
            consumer_group=correction_pair,
            barrier_storage=storage.o_full_empty.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        tmem = utils.TmemAllocator(
            storage.tmem_holding.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=_TMEM_BARRIER, num_threads=self.threads
            ),
            allocator_warp_id=self.correction_warps[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc.ptr,
        )
        tmem.allocate(_TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        pipeline.pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True
        )

        s_q = smem.allocate_tensor(
            element_type=self.dtype,
            layout=q_smem.outer,
            swizzle=q_smem.inner,
            byte_alignment=128,
        )
        s_k = smem.allocate_tensor(
            element_type=self.dtype,
            layout=k_smem.outer,
            swizzle=k_smem.inner,
            byte_alignment=1024,
        )
        # V stages alias the K stages of the shared ring.
        s_v = cute.make_tensor(s_k.iterator, v_smem.outer)
        s_p = None
        if const_expr(not self.p_in_tmem):
            s_p = smem.allocate_tensor(
                element_type=self.dtype,
                layout=p_layout.outer,
                swizzle=p_layout.inner,
                byte_alignment=1024,
            )
        s_out = None
        if const_expr(self.stage_output):
            s_out = smem.allocate_tensor(
                element_type=self.o_dtype,
                layout=out_layout.outer,
                swizzle=out_layout.inner,
                byte_alignment=1024,
            )[None, None, 0]
        s_sum = storage.row_sums.get_tensor(cute.make_layout(self.cta_rows))
        s_pair = storage.pair_values.get_tensor(
            cute.make_layout((self.lane_threads, 2))
        )

        qk_thr = qk_mma.get_slice(cta_v)
        pv_thr = pv_mma.get_slice(cta_v)
        t_q = qk_thr.make_fragment_A(s_q)
        t_k = qk_thr.make_fragment_B(s_k)
        t_v = pv_thr.make_fragment_B(s_v)
        s_shape = qk_thr.partition_shape_C(self.qk_mma_tiler[:2])
        t_s = qk_thr.make_fragment_C(cute.append(s_shape, self.s_stage))
        t_s = cute.make_tensor(t_s.iterator + self.tmem_s_offset, t_s.layout)
        # (row, key) of every score of this CTA, relative to the work tile.
        c_s = qk_thr.partition_C(
            cute.make_identity_tensor(self.qk_mma_tiler[:2])
        )
        o_shape = pv_thr.partition_shape_C(self.pv_mma_tiler[:2])
        t_o = pv_thr.make_fragment_C(o_shape)
        t_o = cute.make_tensor(
            t_o.iterator + self.tmem_o_offset,
            cute.append(
                t_o.layout,
                cute.make_layout(self.pv_chunks, stride=self.o_chunk_columns),
            ),
        )
        # The PV A operand in SMEM; in tensor memory it is built per stage.
        t_p = None
        if const_expr(not self.p_in_tmem):
            t_p = pv_thr.make_fragment_A(s_p)

        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ------------------------------------------------------------ load
        if warp == self.load_warp:
            cute.arch.setmaxregister_decrease(self.regs_other)
            self.load(
                schedule,
                cta_v,
                cta_coord_vmnk,
                cluster_layout_vmnk,
                qk_thr,
                (
                    tma_q,
                    tma_q_tensor,
                    tma_kp,
                    tma_kp_tensor,
                    tma_vp,
                    tma_vp_tensor,
                    tma_kc,
                    tma_kc_tensor,
                    tma_vc,
                    tma_vc_tensor,
                ),
                (s_q, s_k, ring),
                block_table,
                num_pages,
                first_lookups,
                q_producer,
                kv_producer,
            )

        # ------------------------------------------------------------- mma
        if warp == self.mma_warp:
            cute.arch.setmaxregister_decrease(self.regs_other)
            self.mma(
                schedule,
                is_leader,
                qk_mma,
                pv_mma,
                pv_thr,
                (t_q, t_k, t_v, t_s, t_o, t_p, p_layout),
                (q_consumer, kv_consumer, s_producer, p_consumer, o_producer),
            )

        # --------------------------------------------------------- softmax
        if warp < self.correction_warps[0]:
            cute.arch.setmaxregister_increase(self.regs_softmax)
            self.softmax(
                schedule,
                cta_v,
                t_s,
                c_s,
                s_p,
                p_layout,
                s_sum,
                s_pair,
                lse,
                scale,
                scale_log2,
                (s_consumer, p_producer, stats_producer, sum_producer),
            )

        # ------------------------------------------------------ correction
        if warp >= self.correction_warps[0] and warp < self.mma_warp:
            cute.arch.setmaxregister_decrease(self.regs_correction)
            self.correction(
                schedule,
                cta_v,
                t_s,
                t_o,
                s_sum,
                s_out,
                o_packed,
                scale_log2,
                (stats_consumer, sum_consumer, o_consumer),
            )

        if warp > self.load_warp:
            cute.arch.setmaxregister_decrease(self.regs_other)

        # Both CTAs must finish every TMEM access before the pair frees it.
        # Every tensor-memory access has completed through its pipeline by
        # now, so the arrival needs no release ordering; a releasing arrival
        # would wait for the epilogue's global stores to complete.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    # --------------------------------------------------------------- loading
    #
    # CuTe DSL carries a value across a dynamic loop or branch only when the
    # region assigns it, so every helper that advances a pipeline returns the
    # participant and the caller reassigns it.

    @no_type_check
    @cute.jit
    def load(
        self,
        schedule,
        cta_v: Int32,
        cta_coord_vmnk,
        cluster_layout_vmnk: cute.Layout,
        qk_thr,
        tma,
        smem_tensors,
        block_table: cute.Tensor,
        num_pages: Int32,
        first_lookups,
        q_producer,
        kv_producer,
    ):
        """Issue the TMA loads of Q and of every K/V stage of each tile.

        K tile ``j`` is loaded before V tile ``j - 1``, the order in which
        the MMA warp consumes the stages. Each tile's metadata and first
        page lookups (:meth:`tile_lookups`) are read at the end of the
        previous tile, or before the kernel setup for the first one.
        """
        cluster_id, num_clusters, total_tiles = schedule[:3]
        (
            tma_q,
            tma_q_tensor,
            tma_kp,
            tma_kp_tensor,
            tma_vp,
            tma_vp_tensor,
            tma_kc,
            tma_kc_tensor,
            tma_vc,
            tma_vc_tensor,
        ) = tma
        s_q, s_k, ring = smem_tensors

        # (dim in atom, row, atom, stage) view of the ring, split into page
        # slots and into whole stages; both grouped as (box, ...).
        s_ring = cute.make_tensor(
            s_k.iterator,
            cute.append(
                ring.outer,
                cute.make_layout(self.kv_stage, stride=cute.cosize(ring.outer)),
            ),
        )
        slot_tile = (self.atom_dims, self.page_tokens, 2)
        stage_tile = (self.atom_dims, self.half_keys, 2)
        s_slots = cute.group_modes(cute.flat_divide(s_ring, slot_tile), 0, 3)
        s_whole = cute.group_modes(cute.flat_divide(s_ring, stage_tile), 0, 3)
        q_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )

        tile = cluster_id
        info, slots, ahead = first_lookups
        while tile < total_tiles:
            (
                valid,
                batch,
                kv_head,
                m_block,
                query_start,
                query_len,
                prefix_len,
                first_page,
                prefix_base,
                prefix_tiles,
                key_tiles,
                _lower_first,
                _lower_last,
            ) = info
            if valid:
                # Q: this CTA's rows of the tile's packed rows, per chunk.
                q_seq = cute.domain_offset(
                    ((0, query_start), 0, 0), tma_q_tensor
                )
                g_q = cute.flat_divide(
                    q_seq, cute.select(self.qk_mma_tiler, mode=[0, 2])
                )
                t_q_dst, t_q_src = cpasync.tma_partition(
                    tma_q,
                    cta_coord_vmnk[2],
                    q_cta_layout,
                    cute.group_modes(s_q, 0, 3),
                    cute.group_modes(qk_thr.partition_A(g_q), 0, 3),
                )
                for chunk in cutlass.range(self.qk_chunks, unroll=1):
                    handle = q_producer.acquire_and_advance()
                    cute.copy(
                        tma_q,
                        t_q_src[None, m_block, chunk, kv_head],
                        t_q_dst[None, handle.index],
                        tma_bar_ptr=handle.barrier,
                    )

                sequence = (
                    kv_head,
                    cta_v,
                    query_start,
                    query_len,
                    prefix_tiles,
                )
                sources = (
                    (tma_kp, tma_kp_tensor, tma_kc, tma_kc_tensor),
                    (tma_vp, tma_vp_tensor, tma_vc, tma_vc_tensor),
                    (s_slots, s_whole),
                )
                # Lane s holds (physical page, row shift) of page slot s of
                # a prefix tile; the lookups run one tile ahead of the copies
                # that use them.
                lookup = (
                    (
                        batch,
                        prefix_len,
                        first_page,
                        prefix_base,
                        prefix_tiles,
                    ),
                    block_table,
                    num_pages,
                )
                kv_producer = self.load_k(
                    Int32(0), slots, sequence, sources, kv_producer
                )
                for key_tile in cutlass.range(1, key_tiles, unroll=1):
                    following = self.lane_slots(key_tile + 1, lookup)
                    kv_producer = self.load_k(
                        key_tile, ahead, sequence, sources, kv_producer
                    )
                    kv_producer = self.load_v(
                        key_tile - 1, slots, sequence, sources, kv_producer
                    )
                    slots = ahead
                    ahead = following
                kv_producer = self.load_v(
                    key_tiles - 1, slots, sequence, sources, kv_producer
                )
            tile += num_clusters
            info, slots, ahead = self.tile_lookups(
                tile, schedule, block_table, num_pages
            )
        kv_producer.tail()
        q_producer.tail()

    @no_type_check
    @cute.jit
    def tile_lookups(
        self,
        tile: Int32,
        schedule,
        block_table: cute.Tensor,
        num_pages: Int32,
    ):
        """Metadata of work tile ``tile`` and its first two prefix lookups.

        Returns ``(work_tile(tile), lane_slots(0), lane_slots(1))``. A tile
        past the last one is clamped to it, so the lookups ahead of the end
        of the loop stay inside the metadata tensors.
        """
        total_tiles = schedule[2]
        info = self.work_tile(cutlass.min(tile, total_tiles - 1), schedule)
        (
            _valid,
            batch,
            _kv_head,
            _m_block,
            _query_start,
            _query_len,
            prefix_len,
            first_page,
            prefix_base,
            prefix_tiles,
            _key_tiles,
            _lower_first,
            _lower_last,
        ) = info
        lookup = (
            (batch, prefix_len, first_page, prefix_base, prefix_tiles),
            block_table,
            num_pages,
        )
        return (
            info,
            self.lane_slots(Int32(0), lookup),
            self.lane_slots(Int32(1), lookup),
        )

    @no_type_check
    @cute.jit
    def lane_slots(self, key_tile: Int32, lookup):
        """Page slot ``lane`` of prefix tile ``key_tile`` for this lane.

        Each lane below ``128 / page_tokens`` looks up one slot, so a tile's
        block table reads proceed in parallel and are shuffled to the
        issuing lane; other lanes, and tiles that are not prefix tiles,
        return an out-of-range page.
        """
        prefix, block_table, num_pages = lookup
        batch, prefix_len, first_page, prefix_base, prefix_tiles = prefix
        lane = cute.arch.lane_idx()
        physical = num_pages
        shift = Int32(0)
        if key_tile < prefix_tiles and lane < self.slots_per_tile:
            physical, shift = self.page_slot(
                prefix_base // self.page_tokens
                + key_tile * self.slots_per_tile
                + lane,
                batch,
                prefix_len,
                first_page,
                block_table,
                num_pages,
            )
        return physical, shift

    @no_type_check
    @cute.jit
    def page_slot(
        self,
        logical_page: Int32,
        batch: Int32,
        prefix_len: Int32,
        first_page: Int32,
        block_table: cute.Tensor,
        num_pages: Int32,
    ):
        """Return ``(physical page, row shift)`` for one prefix page slot.

        Pages at or past ``ceil(P / page_tokens)``, and logical pages outside
        the sequence's block table row, map to page ``num_pages``, which the
        TMA unit reads as zeros. The last page is shifted down by its
        unwritten row count so that only written rows are read.
        """
        page_end = (prefix_len + self.page_tokens - 1) // self.page_tokens
        column = logical_page - first_page
        physical = num_pages
        shift = Int32(0)
        if (
            logical_page < page_end
            and column >= 0
            and column < block_table.shape[1]
        ):
            physical = block_table[batch, column]
            if logical_page == page_end - 1:
                shift = prefix_len - page_end * self.page_tokens
        return physical, shift

    @no_type_check
    @cute.jit
    def copy_slot(
        self,
        atom,
        tensor,
        s_slots,
        kv_head,
        pair,
        physical,
        shift,
        slot,
        handle,
    ):
        """Copy one page slot into slot ``slot`` of a ring stage.

        The box is rows ``[-shift, page_tokens - shift)`` of page
        ``physical`` over head-dim atoms ``2 * pair`` and ``2 * pair + 1``;
        rows before the page start read as zeros.
        """
        g = cute.domain_offset(
            (0, shift, 0, 0), tensor[None, None, None, kv_head, None]
        )
        g = cute.flat_divide(g, (self.atom_dims, self.page_tokens, 2))
        smem, gmem = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), s_slots, cute.group_modes(g, 0, 3)
        )
        cute.copy(
            atom,
            gmem[None, 0, 0, pair, physical],
            smem[None, 0, slot, 0, handle.index],
            tma_bar_ptr=handle.barrier,
        )

    @no_type_check
    @cute.jit
    def copy_current(self, atom, tensor, s_whole, rows, pair, handle):
        """Copy 64 current-block rows starting at packed row ``rows``."""
        g = cute.domain_offset((0, rows, 0), tensor)
        g = cute.flat_divide(g, (self.atom_dims, self.half_keys, 2))
        smem, gmem = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), s_whole, cute.group_modes(g, 0, 3)
        )
        cute.copy(
            atom,
            gmem[None, 0, 0, pair],
            smem[None, 0, 0, 0, handle.index],
            tma_bar_ptr=handle.barrier,
        )

    @no_type_check
    @cute.jit
    def load_k(self, key_tile: Int32, slots, sequence, sources, kv_producer):
        """Load this CTA's 64-key half of K tile ``key_tile``, per chunk.

        Chunk ``c`` holds head dims ``[128 c, 128 c + 128)``.
        """
        kv_head, cta_v, query_start, query_len, prefix_tiles = sequence
        (atom, pages, current_atom, current), _, (s_slots, s_whole) = sources
        for chunk in cutlass.range(self.qk_chunks, unroll=1):
            handle = kv_producer.acquire_and_advance()
            if key_tile < prefix_tiles:
                for slot in cutlass.range(self.slots_per_half, unroll=1):
                    lane = cta_v * self.slots_per_half + slot
                    self.copy_slot(
                        atom,
                        pages,
                        s_slots,
                        kv_head,
                        chunk,
                        cute.arch.shuffle_sync(slots[0], lane),
                        cute.arch.shuffle_sync(slots[1], lane),
                        slot,
                        handle,
                    )
            else:
                row = self.current_row(key_tile - prefix_tiles, query_len)
                self.copy_current(
                    current_atom,
                    current[None, None, None, kv_head],
                    s_whole,
                    query_start + row + cta_v * self.half_keys,
                    chunk,
                    handle,
                )
        return kv_producer

    @no_type_check
    @cute.jit
    def load_v(self, key_tile: Int32, slots, sequence, sources, kv_producer):
        """Load this CTA's head-dim half of V tile ``key_tile``.

        Stages follow the PV order: for each chunk pair ``pair`` of this
        CTA's contiguous head-dim half (head-dim atoms ``2 * (v * pairs +
        pair)`` and the next), the tile's two 64-key halves.
        """
        kv_head, cta_v, query_start, query_len, prefix_tiles = sequence
        _, (atom, pages, current_atom, current), (s_slots, s_whole) = sources
        for pair in cutlass.range_constexpr(self.pv_pairs):
            for half in cutlass.range_constexpr(2):
                handle = kv_producer.acquire_and_advance()
                atoms = cta_v * self.pv_pairs + pair
                if key_tile < prefix_tiles:
                    for slot in cutlass.range(self.slots_per_half, unroll=1):
                        lane = half * self.slots_per_half + slot
                        self.copy_slot(
                            atom,
                            pages,
                            s_slots,
                            kv_head,
                            atoms,
                            cute.arch.shuffle_sync(slots[0], lane),
                            cute.arch.shuffle_sync(slots[1], lane),
                            slot,
                            handle,
                        )
                else:
                    row = self.current_row(key_tile - prefix_tiles, query_len)
                    self.copy_current(
                        current_atom,
                        current[None, None, None, kv_head],
                        s_whole,
                        query_start + row + half * self.half_keys,
                        atoms,
                        handle,
                    )
        return kv_producer

    @no_type_check
    @cute.jit
    def current_row(self, block_tile: Int32, query_len: Int32) -> Int32:
        """First block row of current-block tile ``block_tile``.

        Every tile starts at a multiple of 128 except the last of a block
        whose length is not a multiple of 128, which ends at the block's
        last row; its leading rows repeat or precede the block and are
        masked by :meth:`current_threshold`.
        """
        row = block_tile * self.tile_keys
        if row + self.tile_keys > query_len:
            row = query_len - self.tile_keys
        return row

    @no_type_check
    @cute.jit
    def current_threshold(self, block_tile: Int32, query_len: Int32) -> Int32:
        """First unmasked column of current-block tile ``block_tile``."""
        threshold = Int32(0)
        row = block_tile * self.tile_keys
        if row + self.tile_keys > query_len:
            threshold = row + self.tile_keys - query_len
        return threshold

    # ------------------------------------------------------------------- mma

    @no_type_check
    @cute.jit
    def mma(
        self, schedule, is_leader, qk_mma, pv_mma, pv_thr, fragments, pipes
    ):
        """Issue the QK and PV MMAs of every tile from the leader CTA.

        Scores of tile ``j`` are computed before the PV product of tile
        ``j - 1`` so that the softmax of one tile overlaps the MMAs of its
        neighbours; the two score stages alternate. The peer CTA's MMA warp
        only walks the tiles.
        """
        cluster_id, num_clusters, total_tiles = schedule[:3]
        t_q, t_k, t_v, t_s, t_o, t_p, p_layout = fragments
        q_consumer, kv_consumer, s_producer, p_consumer, o_producer = pipes

        tile = cluster_id
        while tile < total_tiles:
            info = self.work_tile(tile, schedule)
            valid = info[0]
            key_tiles = info[10]
            if valid and is_leader:
                q_release = q_consumer.clone()
                pv_mma.set(tcgen05.Field.ACCUMULATE, False)
                for key_tile in cutlass.range(key_tiles, unroll=1):
                    # S_j = Q K_j^T over the head-dim chunks.
                    s_handle = s_producer.acquire_and_advance()
                    t_s_stage = t_s[None, None, None, s_handle.index]
                    qk_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for chunk in cutlass.range(self.qk_chunks, unroll=1):
                        if key_tile == 0:
                            q_consumer.wait_and_advance()
                        k_handle = kv_consumer.wait_and_advance()
                        t_q_chunk = t_q[None, None, None, chunk]
                        t_k_stage = t_k[None, None, None, k_handle.index]
                        for kphase in cutlass.range(
                            cute.size(t_q_chunk, mode=[2]), unroll_full=True
                        ):
                            cute.gemm(
                                qk_mma,
                                t_s_stage,
                                t_q_chunk[None, None, kphase],
                                t_k_stage[None, None, kphase],
                                t_s_stage,
                            )
                            qk_mma.set(tcgen05.Field.ACCUMULATE, True)
                        k_handle.release()
                        if key_tile == key_tiles - 1:
                            q_release.release()
                            q_release.advance()
                    s_handle.commit()
                    if key_tile > 0:
                        (
                            pv_mma,
                            kv_consumer,
                            p_consumer,
                            o_producer,
                        ) = self.pv_step(
                            pv_mma,
                            pv_thr,
                            (t_v, t_s, t_o, t_p, p_layout),
                            kv_consumer,
                            p_consumer,
                            o_producer,
                        )
                pv_mma, kv_consumer, p_consumer, o_producer = self.pv_step(
                    pv_mma,
                    pv_thr,
                    (t_v, t_s, t_o, t_p, p_layout),
                    kv_consumer,
                    p_consumer,
                    o_producer,
                )
            tile += num_clusters
        s_producer.tail()
        o_producer.tail()

    @no_type_check
    @cute.jit
    def pv_step(
        self, pv_mma, pv_thr, operands, kv_consumer, p_consumer, o_producer
    ):
        """Accumulate ``P V`` of the oldest pending tile into the output.

        Waits for the tile's probabilities and for the correction warps to
        release the output, which they rescale between consecutive tiles.
        """
        t_v, t_s, t_o, t_p, p_layout = operands
        p_handle = p_consumer.wait_and_advance()
        o_handle = o_producer.acquire_and_advance()
        accumulate = pv_mma.get(tcgen05.Field.ACCUMULATE)
        if const_expr(self.p_in_tmem):
            # BF16 probabilities in place of the stage's scores.
            t_s_stage = t_s[None, None, None, p_handle.index]
            t_p_stage = pv_thr.make_fragment_A(
                cute.make_tensor(
                    t_s_stage.iterator,
                    cute.select(p_layout.outer, mode=[0, 1, 2]),
                )
            )
            t_p_stage = cute.make_tensor(
                cute.recast_ptr(t_s_stage.iterator, dtype=self.dtype),
                t_p_stage.layout,
            )
        else:
            t_p_stage = t_p[None, None, None, p_handle.index]
        # V stages arrive per chunk pair and 64-key half (see load_v); each
        # holds one 64-key sub-stage per chunk of the pair. Both halves of a
        # pair are held so that each output chunk accumulates its 128 keys
        # in one uninterrupted run.
        for pair in cutlass.range_constexpr(self.pv_pairs):
            halves = (
                kv_consumer.wait_and_advance(),
                kv_consumer.wait_and_advance(),
            )
            for step in cutlass.range_constexpr(2):
                t_o_chunk = t_o[None, None, None, pair * 2 + step]
                pv_mma.set(tcgen05.Field.ACCUMULATE, accumulate)
                for half in cutlass.range_constexpr(2):
                    t_v_stage = t_v[
                        None, None, None, halves[half].index * 2 + step
                    ]
                    kphases = cute.size(t_v_stage, mode=[2])
                    for kphase in cutlass.range_constexpr(kphases):
                        cute.gemm(
                            pv_mma,
                            t_o_chunk,
                            t_p_stage[None, None, half * kphases + kphase],
                            t_v_stage[None, None, kphase],
                            t_o_chunk,
                        )
                        pv_mma.set(tcgen05.Field.ACCUMULATE, True)
            halves[0].release()
            halves[1].release()
        o_handle.commit()
        p_handle.release()
        # The accumulate flag lives in the MMA atom; returning the atom carries
        # it to the next tile.
        return pv_mma, kv_consumer, p_consumer, o_producer

    # --------------------------------------------------------------- softmax

    @no_type_check
    @cute.jit
    def softmax(
        self,
        schedule,
        cta_v: Int32,
        t_s: cute.Tensor,
        c_s: cute.Tensor,
        s_p: cute.Tensor | None,
        p_layout: cute.ComposedLayout,
        s_sum: cute.Tensor,
        s_pair: cute.Tensor,
        lse: cute.Tensor | None,
        scale: Float32,
        scale_log2: Float32,
        pipes,
    ):
        """Turn each score tile into BF16 probabilities with online softmax.

        Thread ``t`` reads tensor-memory lane ``t``: a whole row at head dim
        256, half a row at head dim 512. Scores outside the row's visible set
        are replaced by -inf before the row maximum, so masked keys
        contribute exactly zero probability.
        """
        cluster_id, num_clusters, total_tiles = schedule[:3]
        window = schedule[9]
        prefix_start = schedule[8]
        causal_rows = schedule[10]
        s_consumer, p_producer, stats_producer, sum_producer = pipes
        tidx = cute.arch.thread_idx()[0] % self.lane_threads

        # Tensor-memory load of this thread's scores and their coordinates;
        # the layouts are static, only the stage address changes per tile.
        load_s = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32
            ),
            t_s[(None, None), 0, 0, 0],
        )
        thr_load = load_s.get_slice(tidx)
        coords = thr_load.partition_D(c_s[(None, None), 0, 0])
        # Each thread's scores lie in one row; its offset within the tile.
        tile_row = coords[0][0]
        local_row = tile_row - cta_v * self.cta_rows
        s_p_views = None
        if const_expr(not self.p_in_tmem):
            s_p_views = self.probability_views(load_s, tidx, s_p, p_layout)

        exchanges = Int32(0)
        tile = cluster_id
        while tile < total_tiles:
            (
                valid,
                batch,
                kv_head,
                m_block,
                query_start,
                query_len,
                prefix_len,
                _first_page,
                prefix_base,
                prefix_tiles,
                key_tiles,
                _lower_first,
                lower_last,
            ) = self.work_tile(tile, schedule)
            if valid:
                packed_row = m_block * self.tile_rows + tile_row
                query_pos = packed_row // self.group_size
                lower = self.lower_bound(
                    query_pos, prefix_len, batch, prefix_start, window
                )
                page_end = (
                    prefix_len + self.page_tokens - 1
                ) // self.page_tokens
                tail_start = (page_end - 1) * self.page_tokens
                tail_shift = page_end * self.page_tokens - prefix_len
                # Prefix tokens below this bound are free of the shifted
                # last page and its zero-filled leading rows.
                clean_end = prefix_len
                if tail_shift > 0:
                    clean_end = tail_start
                # Prefix tile column c holds token tile_start + c, except in
                # the shifted last page [tail_start, tail_start + page) where
                # it holds token tile_start + c - tail_shift and its first
                # tail_shift columns are zero-filled. In column-token space
                # (tile_start + c) the row's visible keys are the tokens
                # before the last page, [lower, tail_start), and the last
                # page's written rows at or above the bound.
                tail_first = cutlass.max(tail_start, lower) + tail_shift
                tail_end = tail_start + self.page_tokens

                row_max = -Float32.inf
                row_sum = Float32(0.0)
                for key_tile in cutlass.range(key_tiles, unroll=1):
                    tile_start = prefix_base + key_tile * self.tile_keys
                    is_prefix = key_tile < prefix_tiles
                    threshold = self.current_threshold(
                        key_tile - prefix_tiles, query_len
                    )
                    need_mask = threshold > 0
                    # Visible intervals [first_start, first_end) and
                    # [second_start, second_end) of base + column: a current
                    # tile's columns from threshold on, a prefix tile's
                    # intervals in column-token space.
                    base = Int32(0)
                    first_start = threshold
                    first_end = Int32(self.tile_keys)
                    second_start = Int32(0)
                    second_end = Int32(0)
                    if const_expr(self.has_causal):
                        if causal_rows[batch] != 0 and not is_prefix:
                            # The last loaded tile can shift backwards to
                            # avoid reading unwritten packed rows. Express
                            # the causal/window bounds in its local columns.
                            current_start = self.current_row(
                                key_tile - prefix_tiles, query_len
                            )
                            first_end = cutlass.min(
                                first_end, query_pos + 1 - current_start
                            )
                            if const_expr(self.has_window):
                                first_start = cutlass.max(
                                    first_start,
                                    query_pos - window - current_start,
                                )
                            need_mask = True
                    if is_prefix:
                        need_mask = (
                            tile_start < lower_last
                            or tile_start + self.tile_keys > clean_end
                        )
                        base = tile_start
                        first_start = lower
                        first_end = tail_start
                        second_start = tail_first
                        second_end = tail_end
                    step_args = (
                        scale_log2,
                        (
                            base,
                            first_start,
                            first_end,
                            second_start,
                            second_end,
                        ),
                        t_s,
                        load_s,
                        thr_load,
                        coords,
                        s_p_views,
                        s_pair,
                    )
                    (
                        row_max,
                        row_sum,
                        exchanges,
                        s_consumer,
                        p_producer,
                        stats_producer,
                    ) = self.softmax_step(
                        need_mask,
                        row_max,
                        row_sum,
                        exchanges,
                        step_args,
                        s_consumer,
                        p_producer,
                        stats_producer,
                    )
                if const_expr(self.row_split == 2):
                    # Both halves share the running maximum, so the row sum
                    # is the sum of their partial sums.
                    row_sum = row_sum + self.exchange(
                        row_sum, exchanges, s_pair
                    )
                    exchanges += 1
                sum_producer = self.store_row_stats(
                    row_max,
                    row_sum,
                    s_sum,
                    sum_producer,
                    lse,
                    scale,
                    scale_log2,
                    local_row,
                    query_pos,
                    packed_row,
                    query_start,
                    query_len,
                    kv_head,
                )
            tile += num_clusters
        p_producer.tail()
        stats_producer.tail()

    @no_type_check
    @cute.jit
    def probability_views(self, load_s, tidx, s_p, p_layout):
        """SMEM copy of this thread's probabilities into the P tiles.

        The copy has the thread-to-(row, key) mapping of the score load, so
        each thread writes the probabilities of its own scores. Returns the
        tiled copy, the thread's slice, and the destination partition of
        every P stage.
        """
        copy = cute.make_tiled_copy_D(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=128,
            ),
            load_s,
        )
        thr_copy = copy.get_slice(tidx)
        # (row, key, stage) view of the MMA-partitioned P tiles.
        outer = p_layout.outer
        rows_keys = cute.make_tensor(
            s_p.iterator,
            cute.make_layout(
                (
                    (outer.shape[0][0], outer.shape[1]),
                    (outer.shape[0][1], outer.shape[2]),
                    outer.shape[3],
                ),
                stride=(
                    (outer.stride[0][0], outer.stride[1]),
                    (outer.stride[0][1], outer.stride[2]),
                    outer.stride[3],
                ),
            ),
        )
        return copy, thr_copy, thr_copy.partition_D(rows_keys)

    @no_type_check
    @cute.jit
    def exchange(self, value: Float32, exchanges: Int32, s_pair: cute.Tensor):
        """Swap ``value`` with the thread holding the other half of the row.

        Lanes ``t`` and ``t ^ 64`` (warps ``w`` and ``w ^ 2``) share a row at
        head dim 512. Consecutive exchanges alternate between two buffers:
        the barrier of one exchange guarantees that both threads have read
        the buffer of the previous exchange before it is written again.
        """
        tidx = cute.arch.thread_idx()[0] % self.lane_threads
        parity = exchanges % 2
        s_pair[tidx, parity] = value
        cute.arch.barrier(
            barrier_id=_PAIR_BARRIER + (tidx // 32) % 2,
            number_of_threads=64,
        )
        return s_pair[tidx ^ 64, parity]

    @no_type_check
    @cute.jit
    def mask_scores(self, scores: cute.Tensor, coords: cute.Tensor, intervals):
        """Replace scores outside the row's visible key set by -inf.

        A score at column ``c`` is visible when ``base + c`` lies in one of
        the two half-open intervals ``[first_start, first_end)`` and
        ``[second_start, second_end)`` (see :meth:`softmax`). Each fragment
        of 32 consecutive columns turns the intervals into a 32-bit keep
        mask whose constant bits select the scores.
        """
        base, first_start, first_end, second_start, second_end = intervals
        for fragment in cutlass.range_constexpr(cute.size(scores) // _FRAGMENT):
            offset = base + coords[fragment * _FRAGMENT][1]
            keep = _interval_bits(
                first_start - offset, first_end - offset
            ) | _interval_bits(second_start - offset, second_end - offset)
            for i in cutlass.range_constexpr(_FRAGMENT):
                n = fragment * _FRAGMENT + i
                visible = Boolean(keep & (Uint32(1) << i))
                scores[n] = scores[n] if visible else -Float32.inf

    @no_type_check
    @cute.jit
    def softmax_step(
        self,
        need_mask,
        row_max: Float32,
        row_sum: Float32,
        exchanges: Int32,
        step_args,
        s_consumer,
        p_producer,
        stats_producer,
    ):
        """Process one score tile of this thread's lane.

        Loads the lane's FP32 scores, masks them when ``need_mask``,
        updates the running row maximum (combined across the two lanes of a
        row at head dim 512), publishes ``(previous max, new max)`` for the
        output correction, writes ``exp2((s - max) * scale_log2)`` in BF16
        as the PV operand, and returns the running maximum and this lane's
        running sum with the advanced exchange count and pipeline
        participants.
        """
        (
            scale_log2,
            intervals,
            t_s,
            load_s,
            thr_load,
            coords,
            s_p_views,
            s_pair,
        ) = step_args
        tidx = cute.arch.thread_idx()[0] % self.lane_threads

        s_handle = s_consumer.wait_and_advance()
        t_s_stage = t_s[(None, None), 0, 0, s_handle.index]
        scores = cute.make_rmem_tensor(coords.shape, Float32)
        # The load is repeated in both branches: with only the masking under
        # the branch, the compiler predicates it and every tile issues the
        # mask instructions.
        if need_mask:
            cute.copy(load_s, thr_load.partition_S(t_s_stage), scores)
            cute.arch.fence_view_async_tmem_load()
            s_handle.release()
            self.mask_scores(scores, coords, intervals)
        else:
            cute.copy(load_s, thr_load.partition_S(t_s_stage), scores)
            cute.arch.fence_view_async_tmem_load()
            s_handle.release()

        previous_max = row_max
        tile_max = scores.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        if const_expr(self.row_split == 2):
            tile_max = cutlass.max(
                tile_max, self.exchange(tile_max, exchanges, s_pair)
            )
            exchanges += 1
        # Keep the previous maximum unless this tile exceeds it by more than
        # the rescale threshold (both lanes of a row decide alike).
        row_max = tile_max
        if previous_max != -Float32.inf:
            if (previous_max - tile_max) * scale_log2 >= -_RESCALE_THRESHOLD:
                row_max = previous_max
        safe_max = row_max
        if row_max == -Float32.inf:
            safe_max = Float32(0.0)

        # Row statistics for the correction warps, in this lane of the
        # stage's statistics columns.
        stats_handle = stats_producer.acquire_and_advance()
        t_stats = cute.make_tensor(
            t_s_stage.iterator + self.stats_column,
            cute.make_layout((self.lane_threads, 2), stride=(_TMEM_LANE, 1)),
        )
        store_stats = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(2)), Float32
            ),
            t_stats,
        )
        thr_stats = store_stats.get_slice(tidx)
        stats = cute.make_rmem_tensor(
            thr_stats.partition_S(
                cute.make_identity_tensor((self.lane_threads, 2))
            ).shape,
            Float32,
        )
        stats[0] = previous_max
        stats[1] = safe_max
        cute.copy(store_stats, stats, thr_stats.partition_D(t_stats))
        cute.arch.fence_view_async_tmem_store()
        stats_handle.commit()

        neg_max = (Float32(0.0) - safe_max) * scale_log2
        p_handle = p_producer.acquire_and_advance()
        for n in cutlass.range(0, cute.size(scores), 2, unroll_full=True):
            scores[n], scores[n + 1] = cute.arch.fma_packed_f32x2(
                (scores[n], scores[n + 1]),
                (scale_log2, scale_log2),
                (neg_max, neg_max),
            )
            scores[n] = cute.math.exp2(scores[n], fastmath=True)
            scores[n + 1] = cute.math.exp2(scores[n + 1], fastmath=True)
        probs = cute.make_rmem_tensor(scores.shape, self.dtype)
        probs.store(scores.load().to(self.dtype))

        if const_expr(self.p_in_tmem):
            # BF16 pairs as 32-bit words over the stage's first columns.
            t_p = cute.make_tensor(
                t_s_stage.iterator,
                cute.composition(
                    t_s_stage.layout,
                    cute.make_layout((t_s_stage.shape[0], self.p_columns)),
                ),
            )
            store_p = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(32)), Float32
                ),
                t_p,
            )
            thr_store = store_p.get_slice(tidx)
            packed = cute.make_tensor(
                cute.recast_ptr(probs.iterator, dtype=Float32),
                thr_store.partition_S(
                    cute.make_identity_tensor(
                        (self.lane_threads, self.p_columns)
                    )
                ).shape,
            )
            cute.copy(store_p, packed, thr_store.partition_D(t_p))
            cute.arch.fence_view_async_tmem_store()
        else:
            copy, thr_copy, s_p_dst = s_p_views
            cute.copy(
                copy,
                thr_copy.retile(probs),
                s_p_dst[None, None, None, p_handle.index],
            )
            # Make the generic-proxy stores visible to the MMA's reads.
            cute.arch.fence_view_async_shared()
        p_handle.commit()

        # Rescale this lane's running sum to the new maximum and add the
        # tile.
        correction = cute.math.exp2(
            scale_log2 * (previous_max - safe_max), fastmath=True
        )
        tile_sum = scores.load().reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
        row_sum = row_sum * correction + tile_sum
        return (
            row_max,
            row_sum,
            exchanges,
            s_consumer,
            p_producer,
            stats_producer,
        )

    @no_type_check
    @cute.jit
    def store_row_stats(
        self,
        row_max: Float32,
        row_sum: Float32,
        s_sum: cute.Tensor,
        sum_producer,
        lse: cute.Tensor | None,
        scale: Float32,
        scale_log2: Float32,
        local_row: Int32,
        query_pos: Int32,
        packed_row: Int32,
        query_start: Int32,
        query_len: Int32,
        kv_head: Int32,
    ):
        """Hand the row sum to the epilogue and store the row's LSE.

        The LSE is of the scaled scores, ``log(sum_k exp(scale * s_k))``, in
        base 2 or natural logarithm; rows without visible keys store -inf.
        At head dim 512 both lanes of a row hold the same values and the
        first one stores them.
        """
        tidx = cute.arch.thread_idx()[0] % self.lane_threads
        writer = tidx < self.cta_rows
        handle = sum_producer.acquire_and_advance()
        if writer:
            s_sum[local_row] = row_sum
        cute.arch.fence_view_async_shared()
        handle.commit()
        if const_expr(self.has_lse):
            empty = row_sum == Float32(0.0) or row_sum != row_sum
            value = -Float32.inf
            if not empty:
                log2_sum = cute.math.log2(row_sum, fastmath=True)
                if const_expr(self.lse_base2):
                    value = scale_log2 * row_max + log2_sum
                else:
                    value = scale * row_max + log2_sum * Float32(_LN_2)
            if writer and query_pos < query_len:
                head = kv_head * self.group_size + packed_row % self.group_size
                lse[query_start + query_pos, head] = value
        return sum_producer

    # ------------------------------------------------------------ correction

    @no_type_check
    @cute.jit
    def correction(
        self,
        schedule,
        cta_v: Int32,
        t_s: cute.Tensor,
        t_o: cute.Tensor,
        s_sum: cute.Tensor,
        s_out: cute.Tensor | None,
        o_packed: cute.Tensor,
        scale_log2: Float32,
        pipes,
    ):
        """Rescale the output after each tile and store the final rows."""
        cluster_id, num_clusters, total_tiles = schedule[:3]
        stats_consumer, sum_consumer, o_consumer = pipes

        tile = cluster_id
        while tile < total_tiles:
            (
                valid,
                _batch,
                kv_head,
                m_block,
                query_start,
                query_len,
                _prefix_len,
                _first_page,
                _prefix_base,
                _prefix_tiles,
                key_tiles,
                _lower_first,
                _lower_last,
            ) = self.work_tile(tile, schedule)
            if valid:
                # The first tile's statistics need no rescale.
                stats_consumer.wait_and_advance().release()
                for _ in cutlass.range(1, key_tiles, unroll=1):
                    stats_consumer, o_consumer = self.rescale_output(
                        scale_log2, t_s, t_o, stats_consumer, o_consumer
                    )
                o_seq = cute.domain_offset(((0, query_start), 0, 0), o_packed)
                sum_consumer, o_consumer = self.store_output(
                    o_seq,
                    m_block * 2 + cta_v,
                    kv_head,
                    query_len,
                    t_o,
                    s_sum,
                    s_out,
                    sum_consumer,
                    o_consumer,
                )
            tile += num_clusters

    @no_type_check
    @cute.jit
    def rescale_output(self, scale_log2, t_s, t_o, stats_consumer, o_consumer):
        """Scale the accumulated output by ``exp2(old max - new max)``.

        Runs between the PV product of the previous tile and that of the
        current one, which waits for this release. Each thread scales the
        output columns of its own tensor-memory lane with the statistics
        stored in that lane.
        """
        tidx = cute.arch.thread_idx()[0] % self.lane_threads
        stats_handle = stats_consumer.wait_and_advance()
        t_s_stage = t_s[(None, None), 0, 0, stats_handle.index]
        t_stats = cute.make_tensor(
            t_s_stage.iterator + self.stats_column,
            cute.make_layout((self.lane_threads, 2), stride=(_TMEM_LANE, 1)),
        )
        load_stats = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(2)), Float32
            ),
            t_stats,
        )
        thr_stats = load_stats.get_slice(tidx)
        stats = cute.make_rmem_tensor(
            thr_stats.partition_D(
                cute.make_identity_tensor((self.lane_threads, 2))
            ).shape,
            Float32,
        )
        cute.copy(load_stats, thr_stats.partition_S(t_stats), stats)
        factor = cute.math.exp2(
            scale_log2 * (stats[0] - stats[1]), fastmath=True
        )
        stats_handle.release()
        # Tensor-memory copies are warp collectives: a warp rescales when any
        # of its rows raised its maximum.
        rescale = cute.arch.vote_ballot_sync(factor < Float32(1.0)) != 0

        o_handle = o_consumer.wait_and_advance()
        if rescale:
            self.scale_output(t_o, tidx, factor)
        o_handle.release()
        return stats_consumer, o_consumer

    @no_type_check
    @cute.jit
    def scale_output(self, t_o: cute.Tensor, tidx: Int32, factor: Float32):
        """Multiply this lane's output columns by ``factor`` in place.

        Loads, scales and stores 16 columns at a time, overlapping the load
        of one piece with the scaling of the previous one.
        """
        for chunk in cutlass.range(self.pv_chunks, unroll_full=True):
            t_o_epi = cute.zipped_divide(
                t_o[(None, None), 0, 0, chunk], self.pv_block_tiler
            )
            c_o_epi = cute.zipped_divide(
                cute.make_identity_tensor(self.pv_block_tiler),
                self.pv_block_tiler,
            )
            load_o = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), Float32
                ),
                t_o_epi,
            )
            store_o = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(16)), Float32
                ),
                t_o_epi,
            )
            thr_load = load_o.get_slice(tidx)
            thr_store = store_o.get_slice(tidx)
            src = thr_load.partition_S(t_o_epi)
            dst = thr_store.partition_D(t_o_epi)
            piece_shape = thr_load.partition_D(c_o_epi)[None, 0, 0].shape
            values = cute.make_rmem_tensor_like(
                cute.append(
                    cute.make_layout(piece_shape),
                    cute.make_layout(2, stride=cute.size(piece_shape)),
                ),
                Float32,
            )
            pieces = cute.size(src, mode=[1])
            cute.copy(load_o, src[None, 0, 0], values[None, 0])
            for piece in cutlass.range_constexpr(1, pieces + 1):
                if const_expr(piece < pieces):
                    cute.copy(
                        load_o, src[None, piece, 0], values[None, piece % 2]
                    )
                previous = (piece - 1) % 2
                for n in cutlass.range_constexpr(
                    0, cute.size(values, mode=[0]), 2
                ):
                    (
                        values[n, previous],
                        values[n + 1, previous],
                    ) = cute.arch.mul_packed_f32x2(
                        (values[n, previous], values[n + 1, previous]),
                        (factor, factor),
                    )
                cute.copy(
                    store_o, values[None, previous], dst[None, piece - 1, 0]
                )
        cute.arch.fence_view_async_tmem_store()

    @no_type_check
    @cute.jit
    def store_output(
        self,
        o_seq: cute.Tensor,
        cta_block: Int32,
        kv_head: Int32,
        query_len: Int32,
        t_o: cute.Tensor,
        s_sum: cute.Tensor,
        s_out: cute.Tensor | None,
        sum_consumer,
        o_consumer,
    ):
        """Normalize this CTA's output rows and store the valid ones.

        CTA row ``r`` is packed row ``cta_block * R + r`` of the sequence (R
        rows per CTA); the packed view maps it to query position ``row // G``
        of head ``kv_head * G + row % G``. With a staging buffer ``s_out``,
        each 128-column chunk is written there by its rows' threads and
        stored by :meth:`store_rows`; the output accumulator is released as
        soon as its last chunk is in registers. Otherwise each thread stores
        its row directly.
        """
        tidx = cute.arch.thread_idx()[0] % self.lane_threads
        c_seq = cute.make_identity_tensor(o_seq.shape)

        # This lane's row within the CTA block selects its row sum; every
        # chunk maps lanes to rows alike.
        rows = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32
            ),
            cute.zipped_divide(t_o[(None, None), 0, 0, 0], self.pv_block_tiler),
        ).get_slice(tidx)
        local = rows.partition_D(
            cute.zipped_divide(
                cute.make_identity_tensor(self.pv_block_tiler),
                self.pv_block_tiler,
            )
        )
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = s_sum[local[0, 0, 0][0]]
        sum_handle.release()
        empty = row_sum == Float32(0.0) or row_sum != row_sum
        factor = Float32(0.0)
        if not empty:
            factor = cute.arch.rcp_approx(row_sum)

        o_handle = o_consumer.wait_and_advance()
        for chunk in cutlass.range_constexpr(self.pv_chunks):
            # This CTA's rows of the chunk's 128 head dims (PV column order).
            g_o = cute.flat_divide(
                o_seq[None, (None, chunk), kv_head], self.pv_block_tiler
            )[None, None, cta_block, 0]
            c_o = cute.flat_divide(
                c_seq[None, (None, chunk), kv_head], self.pv_block_tiler
            )[None, None, cta_block, 0]
            t_o_epi = cute.zipped_divide(
                t_o[(None, None), 0, 0, chunk], self.pv_block_tiler
            )
            g_o_epi = cute.zipped_divide(g_o, self.pv_block_tiler)
            c_o_epi = cute.zipped_divide(c_o, self.pv_block_tiler)
            load_o = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32
                ),
                t_o_epi,
            )
            thr_load = load_o.get_slice(tidx)
            src = thr_load.partition_S(t_o_epi)
            coords = thr_load.partition_D(c_o_epi)
            dst = None
            if const_expr(s_out is None):
                dst = thr_load.partition_D(g_o_epi)
            else:
                dst = thr_load.partition_D(
                    cute.zipped_divide(s_out, self.pv_block_tiler)
                )
                # The buffer is free once every thread has stored the
                # previous chunk (of this or the previous tile).
                cute.arch.barrier(
                    barrier_id=_OUTPUT_BARRIER,
                    number_of_threads=self.lane_threads,
                )
            for piece in cutlass.range_constexpr(cute.size(src, mode=[1])):
                values = cute.make_rmem_tensor(
                    coords[None, piece, 0].shape, Float32
                )
                cute.copy(load_o, src[None, piece, 0], values)
                for n in cutlass.range_constexpr(0, cute.size(values), 2):
                    values[n], values[n + 1] = cute.arch.mul_packed_f32x2(
                        (values[n], values[n + 1]), (factor, factor)
                    )
                converted = cute.make_rmem_tensor(values.shape, self.o_dtype)
                converted.store(values.load().to(self.o_dtype))
                if const_expr(s_out is None):
                    # The row coordinate is (head in group, query position).
                    if coords[0, piece, 0][0][1] < query_len:
                        cute.autovec_copy(converted, dst[None, piece, 0])
                else:
                    cute.autovec_copy(converted, dst[None, piece, 0])
            if const_expr(s_out is not None):
                if const_expr(chunk == self.pv_chunks - 1):
                    cute.arch.fence_view_async_tmem_load()
                    o_handle.release()
                cute.arch.barrier(
                    barrier_id=_OUTPUT_BARRIER,
                    number_of_threads=self.lane_threads,
                )
                self.store_rows(s_out, g_o, c_o, query_len, tidx)
        if const_expr(s_out is None):
            o_handle.release()
        return sum_consumer, o_consumer

    @no_type_check
    @cute.jit
    def store_rows(
        self,
        s_out: cute.Tensor,
        g_o: cute.Tensor,
        c_o: cute.Tensor,
        query_len: Int32,
        tidx: Int32,
    ):
        """Store the staged chunk's valid rows in 16-byte segments.

        Sixteen consecutive threads cover one row's 128 columns, two runs of
        64 contiguous head dims, so each warp store writes whole 128-byte
        pieces of two rows (the group's heads of one query position when
        G = 2).
        """
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.o_dtype, num_bits_per_copy=128
        )
        copy = cute.make_tiled_copy_tv(
            atom,
            cute.make_layout((8, 16), stride=(16, 1)),
            cute.make_layout((1, 8)),
        )
        thr = copy.get_slice(tidx)
        src = thr.partition_S(s_out)
        dst = thr.partition_D(g_o)
        coords = thr.partition_D(c_o)
        for step in cutlass.range_constexpr(cute.size(src, mode=[1])):
            if coords[0, step, 0][0][1] < query_len:
                cute.copy(atom, src[None, step, 0], dst[None, step, 0])
