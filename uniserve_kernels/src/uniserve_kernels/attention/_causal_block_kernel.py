"""SM100 causal attention of prefill chunks over their paged history.

A prefill chunk of ``L_b`` query rows continues sequence ``b`` after its
paged history of ``P_b`` tokens: row ``i`` is absolute position
``P_b + i`` and attends to the keys ``[lower, P_b)`` of the history (the
whole history unless an explicit ``prefix_start`` column raises ``lower``)
and to the chunk's own keys ``[0, i]``. The kernel is the prefix-block
kernel of :mod:`._prefix_block_kernel` (the same two-CTA tiles, key-tile
streaming, page lookups, online softmax and output correction, which it
inherits) with two differences:

* Causal visibility. A work tile reads its block's key tiles only up to the
  one holding its last query position, and the softmax masks block tiles in
  block-key space, ``[tile start, min(position + 1, L))``, which covers the
  diagonal and the block's shifted last tile alike.
* Dynamic scheduling, the dynamic persistent varlen scheme of
  FlashAttention-4's SM100 forward kernel. The row blocks of a causal chunk
  and the chunks of a batch differ widely in cost, so a static stride
  leaves clusters idle. Ticket ``c`` of cluster ``c`` is its tile of a
  static first wave: tile ``c`` of the grid of (sequence, row block up to
  the host bound, KV head) tiles in sequence-major order, found without
  reading the other sequences' lengths; a grid tile past its sequence's
  length is empty. When every grid tile has its own cluster, that is the
  whole launch and no work item is exchanged. Otherwise the scheduler warp
  of a cluster's leader CTA ranks the sequences by cost (the keys their
  longest row block reads) into the workspace and claims each further
  ticket, the cluster count plus the previous value of a global counter,
  when the load warp requests the next tile (two key tiles before its
  current tile's loads end), so tickets go to clusters in the order they
  run out of work. Tickets past the first wave enumerate the remaining
  tiles that exist for the current lengths: sequences in decreasing cost,
  within a sequence the row blocks from last to first (longest first) with
  the KV heads interleaved. The scheduler decodes a ticket into ``(sequence,
  KV head, row block)`` and publishes it to the warps of both CTAs with
  asynchronous stores into their shared memory.

The counter must be zero when the kernel starts. Like FlashAttention-4's
tile-count semaphore, it is reset outside the kernel, by the launch that
precedes it on the stream, never by the kernel itself, whose clusters would
otherwise have to agree which of them makes the last claim. A captured
launch therefore replays together with that preceding launch, with changed
lengths and no host work.

Warp roles (per CTA, 12 warps): softmax (0-3), output correction and
epilogue (4-7), MMA issue (8, leader CTA only), TMA loads (9), the tile
scheduler (10, leader CTA only) and an idle warp.
"""

from typing import no_type_check

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import OperandMajorMode, cpasync

from ._prefix_block_kernel import (
    _LOG2_E,
    _TMEM_BARRIER,
    _TMEM_COLUMNS,
    PrefixBlockAttentionSm100,
    _group,
)

# Workspace int32 words: the ticket counter, zeroed by the launch that
# precedes the kernel on its stream, then per sequence its first dynamic
# ticket in the longest-first order and its tile count.
_WORKSPACE_TABLE = 1
# The load warp requests its cluster's next tile once this many key tiles
# of the current one remain to be issued, which hides the claim behind the
# tile's last key tiles.
_REQUEST_LEAD = 2


class CausalBlockAttentionSm100(PrefixBlockAttentionSm100):
    """Two-CTA SM100 kernel for causal prefill chunks over a paged history.

    Constructor arguments are compile-time specializations with the
    meanings of :class:`PrefixBlockAttentionSm100`; a causal chunk has no
    history window.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        tile_rows: int,
        group_size: int,
        page_tokens: int,
        has_prefix_start: bool,
        has_start_page: bool,
        has_lse: bool,
        lse_base2: bool,
    ) -> None:
        super().__init__(
            head_dim=head_dim,
            tile_rows=tile_rows,
            group_size=group_size,
            page_tokens=page_tokens,
            query_window=False,
            has_window=False,
            has_prefix_start=has_prefix_start,
            has_start_page=has_start_page,
            has_lse=has_lse,
            lse_base2=lse_base2,
            # The causal kernel launches in ordinary stream order.
            pdl=False,
        )
        # Warp 10 schedules the tiles; warp 11 stays idle.
        self.scheduler_warp = 10
        self.idle_warps = (11,)
        # Work-info stages between the scheduler and the other warps, each a
        # (sequence, KV head, row block) item. Warps 0-9 of both CTAs
        # consume every item; one lane of each releases it once read. The
        # scheduler claims a cluster's next ticket only when the leader's
        # load warp requests it, after issuing its tile's key loads, so
        # tickets go to clusters in the order they run out of work (list
        # scheduling). The second stage lets that item be written while the
        # slowest warps still read the current one.
        self.work_stages = 2
        self.work_words = 3
        self.work_consumers = 2 * (self.load_warp + 1)

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
        output: cute.Tensor,
        lse: cute.Tensor | None,
        counters: cute.Tensor,
        scale: Float32,
        num_clusters: Int32,
        row_blocks: Int32,
        single_wave: Int32,
        stream: cuda.CUstream,
    ):
        """Build tensor views and TMA descriptors, then launch the kernel.

        ``query``/``output`` are ``[tokens, Hq, D]``, ``key``/``value``
        ``[tokens, Hkv, D]``, caches ``[pages, page_tokens, Hkv, D]``,
        ``block_table`` ``[B, W]`` int32, ``query_offsets`` ``[B + 1]``,
        ``prefix_lengths``/``start_page``/``prefix_start`` ``[B]`` int32 and
        ``lse`` ``[tokens, Hq]`` FP32. ``counters`` is the scheduler's int32
        workspace: the ticket counter, which a preceding launch on the
        stream zeroes, and two words per sequence. ``num_clusters`` sizes
        the persistent grid and ``row_blocks`` is the host bound of row
        blocks per sequence and KV head; ``single_wave`` is nonzero when the
        grid has a cluster for every tile of that bound.
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
            work_full_empty: cute.struct.MemRange[Int64, self.work_stages * 2]
            request_full_empty: cute.struct.MemRange[Int64, 2]
            tmem_dealloc: Int64
            tmem_holding: Int32
            # Published work tiles, ``work_words`` int32 per stage; 16-byte
            # aligned for the scheduler's asynchronous cluster stores.
            work_items: cute.struct.Align[
                cute.struct.MemRange[Int32, self.work_stages * self.work_words],
                16,
            ]
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
            counters,
            block_table,
            query_offsets,
            prefix_lengths,
            start_page,
            prefix_start,
            key_cache.shape[0],
            scale,
            scale_log2,
            row_blocks,
            single_wave,
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
        )

    @no_type_check
    @cute.jit
    def schedule_tiles(
        self,
        schedule,
        counters,
        s_work,
        work_producer,
        request_consumer,
        first_item,
    ):
        """Claim, decode and publish the dynamic work tiles.

        Runs on the scheduler warp of the leader CTA. Ticket ``c`` of
        cluster ``c`` is its tile of the static first wave, which every role
        takes from the static map (:meth:`first_item`), so a single-wave
        launch (:meth:`single_wave`) publishes nothing. Otherwise the warp
        ranks the sequences into the workspace table and claims further
        tickets, the cluster count plus the counter's previous value, each
        when the load warp requests the next tile (at once when
        ``first_item``, the cluster's first tile decoded before the
        pipelines were set up, is empty), until one past the last tile
        yields the end marker (sequence -1). The counter starts at zero: the
        launch that precedes the kernel on its stream resets it, so no
        cluster of this launch writes it except by claiming.
        """
        num_clusters = schedule[1]
        lane = cute.arch.lane_idx()

        if not self.single_wave(schedule):
            self.rank_sequences(counters, schedule)
            # Whether the load warp will ask for the tile after its current
            # one; a cluster whose first grid tile is empty asks for none.
            pending = Int32(0)
            if first_item[0] >= 0:
                pending = Int32(1)
            claiming = Int32(1)
            while claiming != 0:
                if pending != 0:
                    request_consumer = self.await_request(request_consumer)
                handle, work_producer = self.acquire_stage(work_producer)
                taken = Int32(0)
                if lane == 0:
                    taken = cute.arch.atomic_add(
                        counters.iterator, Int32(1), sem="relaxed", scope="gpu"
                    )
                ticket = num_clusters + cute.arch.shuffle_sync(taken, 0)
                words = self.find_ticket(ticket, schedule, counters)
                self.store_item(s_work, handle, words)
                pending = Int32(1)
                if words[0] < 0:
                    claiming = Int32(0)

    @no_type_check
    @cute.jit
    def await_request(self, request_consumer):
        """Wait for the load warp's next-tile request and acknowledge it.

        Returns the advanced consumer, which the caller reassigns.
        """
        handle = request_consumer.wait_and_advance()
        if cute.arch.lane_idx() == 0:
            handle.release()
        return request_consumer

    @no_type_check
    @cute.jit
    def request_tile(self, request_producer):
        """Ask the scheduler warp for this cluster's next tile.

        Returns the advanced producer, which the caller reassigns.
        """
        handle = request_producer.acquire_and_advance()
        if cute.arch.lane_idx() == 0:
            handle.commit()
        return request_producer

    @no_type_check
    @cute.jit
    def store_item(self, s_work, handle, words):
        """Store an item's words in both CTAs' copies of an acquired stage.

        Lane r fills the stage of CTA r; each store completes its bytes on
        that CTA's full barrier, which the acquire armed.
        """
        lane = cute.arch.lane_idx()
        if lane < 2:
            for word in cutlass.range_constexpr(self.work_words):
                cute.arch.store_async_dsmem(
                    s_work[None, handle.index].iterator + word,
                    words[word],
                    handle.barrier,
                    lane,
                )

    @no_type_check
    @cute.jit
    def acquire_stage(self, work_producer):
        """Acquire the next free work-info stage.

        Returns the stage handle and the advanced producer, which the
        caller reassigns so that the scheduler loop carries its state.
        """
        handle = work_producer.acquire_and_advance()
        return handle, work_producer

    @no_type_check
    @cute.jit
    def single_wave(self, schedule):
        """Whether every possible work tile has its own cluster.

        The launcher sizes the grid by the host bounds and passes whether
        the sequences times KV heads times the row-block bound fit in it;
        each cluster then runs at most its first tile.
        """
        return schedule[9] != 0

    @no_type_check
    @cute.jit
    def first_item(self, schedule, check_length: bool):
        """Work item of the cluster's first tile, from the static first wave.

        Cluster ``c`` takes tile ``c`` of the sequence-major bound grid of
        (sequence, row block up to the host bound, KV head) tiles, found
        without a scan. The launcher never runs more clusters than the grid
        has tiles, so the tile's sequence exists. A grid tile past its
        sequence's length is empty: with ``check_length`` it yields sequence
        -1, otherwise the caller's :meth:`tile_info` finds it empty.
        """
        cluster_id, _, kv_heads, query_offsets = schedule[:4]
        per_row = kv_heads * schedule[8]
        row = cluster_id // per_row
        local = cluster_id - row * per_row
        m_block = local // kv_heads
        item = (row, local % kv_heads, m_block)
        if const_expr(check_length):
            length = query_offsets[row + 1] - query_offsets[row]
            packed = self.tile_rows // self.group_size
            if m_block * packed >= length:
                item = (Int32(-1), Int32(0), Int32(0))
        return item

    @no_type_check
    @cute.jit
    def sequence_tiles(self, row: Int32, schedule):
        """Tile counts and ranking key of sequence ``row``.

        Returns ``(tiles, dynamic, cost)``, zeros past the last sequence. A
        sequence has ``ceil(L * G / tile_rows)`` row blocks per KV head;
        ``dynamic`` counts those of its tiles the static first wave does
        not take (the first wave's grid covers each sequence's first tiles
        in row-block order). Its cost, the ranking key of the longest-first
        order, is the number of keys its longest row block reads: the whole
        block and the prefix its first query position sees.
        """
        num_clusters, kv_heads, query_offsets, prefix_lengths = schedule[1:5]
        prefix_start = schedule[6]
        rows = query_offsets.shape[0] - 1
        tiles = Int32(0)
        dynamic = Int32(0)
        cost = Int32(0)
        if row < rows:
            length = query_offsets[row + 1] - query_offsets[row]
            packed = self.tile_rows // self.group_size
            tiles = (length + packed - 1) // packed * kv_heads
            first_wave = num_clusters - row * kv_heads * schedule[8]
            dynamic = tiles - cutlass.min(
                tiles, cutlass.max(first_wave, Int32(0))
            )
            prefix = prefix_lengths[row]
            lower = Int32(0)
            if const_expr(self.has_prefix_start):
                lower = prefix_start[row]
            cost = cutlass.max(prefix - lower, Int32(0)) + length
        return tiles, dynamic, cost

    @no_type_check
    @cute.jit
    def rank_sequences(self, counters: cute.Tensor, schedule):
        """Store every sequence's first dynamic ticket and tile count.

        Sequences run in decreasing cost, ties in batch order, so a
        sequence's first dynamic ticket is the dynamic tile count of the
        sequences ranked before it. Each lane ranks its own sequences
        against all of them, 32 at a time. Every cluster's scheduler warp
        stores the same values, and each lane later reads back only the
        entries it stored (:meth:`find_ticket`).
        """
        rows = schedule[3].shape[0] - 1
        lane = cute.arch.lane_idx()
        for base in cutlass.range(0, rows, 32, unroll=1):
            row = base + lane
            tiles, _dynamic, cost = self.sequence_tiles(row, schedule)
            start = Int32(0)
            for other in cutlass.range(0, rows, 32, unroll=1):
                _t, other_dynamic, other_cost = self.sequence_tiles(
                    other + lane, schedule
                )
                for k in cutlass.range_constexpr(32):
                    count = cute.arch.shuffle_sync(other_dynamic, k)
                    key = cute.arch.shuffle_sync(other_cost, k)
                    if key > cost or (key == cost and other + k < row):
                        start = start + count
            if row < rows:
                counters[_WORKSPACE_TABLE + 2 * row] = start
                counters[_WORKSPACE_TABLE + 2 * row + 1] = tiles

    @no_type_check
    @cute.jit
    def find_ticket(self, ticket: Int32, schedule, table: cute.Tensor):
        """Return the work item of ticket ``ticket`` past the first wave.

        An item is ``(sequence, KV head, row block)``. The tickets after the
        cluster count enumerate the tiles the first wave does not take,
        sequences longest first (:meth:`rank_sequences`) and each sequence's
        tiles contiguous; a ballot over the table finds the sequence whose
        tickets hold ``ticket``. Within a sequence the row blocks run from
        last to first, since a causal row block sees more block keys than
        the ones before it, and the KV heads alternate. A ticket past the
        last tile returns sequence -1.
        """
        num_clusters, kv_heads, query_offsets = schedule[1:4]
        rows = query_offsets.shape[0] - 1
        lane = cute.arch.lane_idx()
        dynamic = ticket - num_clusters

        batch = Int32(-1)
        local = Int32(0)
        for base in cutlass.range(0, rows, 32, unroll=1):
            row = base + lane
            start = Int32(0)
            tiles = Int32(0)
            if row < rows:
                start = table[_WORKSPACE_TABLE + 2 * row]
                tiles = table[_WORKSPACE_TABLE + 2 * row + 1]
            # Tiles the first wave takes lead the sequence's row-block order.
            taken = cutlass.min(
                tiles,
                cutlass.max(
                    num_clusters - row * kv_heads * schedule[8], Int32(0)
                ),
            )
            hit = dynamic >= start and dynamic < start + tiles - taken
            found = cute.arch.vote_ballot_sync(hit)
            if found != 0:
                # Sequences' ticket ranges are disjoint: one lane holds it.
                source = 31 - cute.arch.clz(found)
                batch = cute.arch.shuffle_sync(row, source)
                offset = dynamic - cute.arch.shuffle_sync(start, source)
                local = cute.arch.shuffle_sync(tiles, source) - 1 - offset

        m_block = local // kv_heads
        kv_head = local % kv_heads
        return batch, kv_head, m_block

    @no_type_check
    @cute.jit
    def following_tile(self, work, schedule):
        """A role's work item after a tile and the advanced ``work``.

        It is the scheduler's next published item; a single-wave launch has
        none (the end marker, sequence -1).
        """
        tile = (Int32(-1), Int32(0), Int32(0))
        if not self.single_wave(schedule):
            work, tile = self.next_tile(work)
        return work, tile

    @no_type_check
    @cute.jit
    def next_tile(self, work):
        """Receive the next published work item.

        Every lane reads the stage; one lane releases it once the warp has
        read it. Returns ``(work, item)`` with the advanced consumer and the
        item's words (see :meth:`find_ticket`); sequence -1 marks the end
        of the work.
        """
        s_work, work_consumer = work
        handle = work_consumer.wait_and_advance()
        stage = s_work[None, handle.index]
        item = tuple(stage[word] for word in range(self.work_words))
        cute.arch.sync_warp()
        if cute.arch.lane_idx() == 0:
            handle.release()
        return (s_work, work_consumer), item

    @no_type_check
    @cute.jit
    def tile_info(self, tile, schedule):
        """Metadata and key ranges of a ``(sequence, KV head, row block)``.

        Returns:
            ``(batch, kv_head, m_block, query_start, query_len, prefix_len,
            first_page, prefix_base, prefix_tiles, key_tiles, lower_first,
            lower_last)``: ``prefix_base`` is the first token of the first
            prefix tile (page aligned); ``lower_first`` and ``lower_last``
            are the prefix lower bounds of the tile's first and last query
            positions. A causal tile's block key tiles end with the one
            holding its last query position. The end marker (sequence -1)
            reads sequence 0's metadata, which its caller ignores; a tile
            past its sequence's rows returns sequence -1 as well.
        """
        (
            _cluster_id,
            _num_clusters,
            _kv_heads,
            query_offsets,
            prefix_lengths,
            start_page,
            prefix_start,
            window,
            _row_blocks,
            _single_wave,
        ) = schedule
        batch, kv_head, m_block = tile
        row = cutlass.max(batch, Int32(0))

        query_start = query_offsets[row]
        query_len = query_offsets[row + 1] - query_start
        prefix_len = prefix_lengths[row]
        first_page = Int32(0)
        if const_expr(self.has_start_page):
            first_page = start_page[row]

        first_query = m_block * (self.tile_rows // self.group_size)
        if first_query >= query_len:
            batch = Int32(-1)
        last_query = cutlass.min(
            first_query + self.tile_rows // self.group_size, query_len
        )
        last_query = cutlass.max(last_query - 1, first_query)
        lower_first = self.lower_bound(
            first_query, prefix_len, row, prefix_start, window
        )
        lower_last = self.lower_bound(
            last_query, prefix_len, row, prefix_start, window
        )

        # Prefix tiles cover pages [lower_first // page, ceil(P / page)).
        prefix_base = (lower_first // self.page_tokens) * self.page_tokens
        prefix_tiles = Int32(0)
        if lower_first < prefix_len:
            page_end = (prefix_len + self.page_tokens - 1) // self.page_tokens
            prefix_span = page_end * self.page_tokens - prefix_base
            prefix_tiles = (prefix_span + self.tile_keys - 1) // self.tile_keys
        # Block tiles past the one holding the last query are masked, so
        # they are not read.
        current_tiles = cutlass.min(
            (query_len + self.tile_keys - 1) // self.tile_keys,
            last_query // self.tile_keys + 1,
        )
        key_tiles = prefix_tiles + current_tiles
        return (
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
        counters: cute.Tensor,
        block_table: cute.Tensor,
        query_offsets: cute.Tensor,
        prefix_lengths: cute.Tensor,
        start_page: cute.Tensor | None,
        prefix_start: cute.Tensor | None,
        num_pages: Int32,
        scale: Float32,
        scale_log2: Float32,
        row_blocks: Int32,
        single_wave: Int32,
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
        schedule = (
            cluster_id,
            num_clusters,
            o_packed.shape[2],
            query_offsets,
            prefix_lengths,
            start_page,
            prefix_start,
            # No history window bounds a causal chunk.
            Int32(0),
            row_blocks,
            single_wave,
        )
        # A cluster's first tile is its tile of the static first wave. The
        # scheduler warp decodes it, and the load warp decodes it again with
        # its first page lookups, before the barrier and tensor-memory
        # setup, which the other warps carry out meanwhile; the other warps
        # keep placeholder values and decode it when their roles start. The
        # first tile's loads then start as soon as the pipelines are ready.
        zero = Int32(0)
        first_item = (zero,) * self.work_words
        first_lookups = ((zero,) * 12, (zero, zero), (zero, zero))
        if warp == self.scheduler_warp:
            first_item = self.first_item(schedule, True)
        if warp == self.load_warp:
            first_lookups = self.tile_lookups(
                self.first_item(schedule, False),
                schedule,
                block_table,
                num_pages,
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
        # Work tiles: the leader's scheduler warp arrives on each CTA's full
        # barrier with the stage's byte count and fills the stage with
        # asynchronous cluster stores; consumers of both CTAs release a stage
        # on the leader's empty barrier. Every role receives each tile, so
        # each takes its own copy of the consumer; a shared participant would
        # carry one role's pipeline state into another role's code.
        work_producer, work_consumer = pipeline.PipelineClcFetchAsync.create(
            num_stages=self.work_stages,
            producer_group=_group(1),
            consumer_group=_group(self.work_consumers),
            tx_count=self.work_words * 4,
            barrier_storage=storage.work_full_empty.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_work = storage.work_items.get_tensor(
            cute.make_layout((self.work_words, self.work_stages))
        )
        # Next-tile requests from the leader's load warp to its scheduler
        # warp, one lane each.
        request_producer, request_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=_group(1),
            consumer_group=_group(1),
            barrier_storage=storage.request_full_empty.data_ptr(),
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

        # ------------------------------------------------------- scheduler
        if warp == self.scheduler_warp:
            cute.arch.setmaxregister_decrease(self.regs_other)
            if is_leader:
                self.schedule_tiles(
                    schedule,
                    counters,
                    s_work,
                    work_producer,
                    request_consumer,
                    first_item,
                )

        # ------------------------------------------------------------ load
        if warp == self.load_warp:
            cute.arch.setmaxregister_decrease(self.regs_other)
            self.load(
                schedule,
                (s_work, work_consumer.clone()),
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
                is_leader,
                request_producer,
                q_producer,
                kv_producer,
            )

        # ------------------------------------------------------------- mma
        if warp == self.mma_warp:
            cute.arch.setmaxregister_decrease(self.regs_other)
            self.mma(
                schedule,
                (s_work, work_consumer.clone()),
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
                (s_work, work_consumer.clone()),
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
                (s_work, work_consumer.clone()),
                cta_v,
                t_s,
                t_o,
                s_sum,
                s_out,
                o_packed,
                scale_log2,
                (stats_consumer, sum_consumer, o_consumer),
            )

        if warp > self.scheduler_warp:
            cute.arch.setmaxregister_decrease(self.regs_other)

        # Both CTAs must finish every TMEM access before the pair frees it.
        # Every tensor-memory access has completed through its pipeline by
        # now, so the arrival needs no release ordering; a releasing arrival
        # would wait for the epilogue's global stores to complete.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @no_type_check
    @cute.jit
    def load(
        self,
        schedule,
        work,
        cta_v: Int32,
        cta_coord_vmnk,
        cluster_layout_vmnk: cute.Layout,
        qk_thr,
        tma,
        smem_tensors,
        block_table: cute.Tensor,
        num_pages: Int32,
        first_lookups,
        is_leader,
        request_producer,
        q_producer,
        kv_producer,
    ):
        """Issue the TMA loads of Q and of every K/V stage of each tile.

        K tile ``j`` is loaded before V tile ``j - 1``, the order in which
        the MMA warp consumes the stages. Each tile's metadata and first
        page lookups (:meth:`tile_lookups`) are read as soon as its last
        loads are issued, before the stages it streams free up. The first
        tile's lookups, ``first_lookups``, were made before the pipelines
        were set up. Once a tile's key loads are issued, the leader's warp
        requests the next tile from the scheduler.
        """
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

        # The first tile, decoded before the setup, is the cluster's first
        # grid tile. An empty grid tile issues no loads.
        tile = self.first_item(schedule, False)
        info, slots, ahead = first_lookups
        while tile[0] >= 0:
            if info[0] >= 0:
                (
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
                # The next tile is requested once _REQUEST_LEAD key tiles remain
                # to be issued (at the start of a shorter tile), so the claim
                # completes while they stream in.
                request_at = cutlass.max(key_tiles - _REQUEST_LEAD, Int32(0))
                if is_leader and request_at == 0:
                    if not self.single_wave(schedule):
                        request_producer = self.request_tile(request_producer)
                kv_producer = self.load_k(
                    Int32(0), slots, sequence, sources, kv_producer
                )
                for key_tile in cutlass.range(1, key_tiles, unroll=1):
                    if is_leader and key_tile == request_at:
                        if not self.single_wave(schedule):
                            request_producer = self.request_tile(
                                request_producer
                            )
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
            work, tile = self.following_tile(work, schedule)
            info, slots, ahead = self.tile_lookups(
                tile, schedule, block_table, num_pages
            )
        kv_producer.tail()
        q_producer.tail()

    @no_type_check
    @cute.jit
    def tile_lookups(
        self,
        tile,
        schedule,
        block_table: cute.Tensor,
        num_pages: Int32,
    ):
        """Metadata of work tile ``tile`` and its first two prefix lookups.

        Returns ``(tile_info(tile), lane_slots(0), lane_slots(1))``. The end
        marker reads sequence 0's metadata (see :meth:`tile_info`), so the
        lookups stay inside the metadata tensors.
        """
        info = self.tile_info(tile, schedule)
        (
            _batch,
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
        batch = cutlass.max(info[0], Int32(0))
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
    def mma(
        self,
        schedule,
        work,
        is_leader,
        qk_mma,
        pv_mma,
        pv_thr,
        fragments,
        pipes,
    ):
        """Issue the QK and PV MMAs of every tile from the leader CTA.

        Scores of tile ``j`` are computed before the PV product of tile
        ``j - 1`` so that the softmax of one tile overlaps the MMAs of its
        neighbours; the two score stages alternate. The peer CTA's MMA warp
        only receives the tiles.
        """
        t_q, t_k, t_v, t_s, t_o, t_p, p_layout = fragments
        q_consumer, kv_consumer, s_producer, p_consumer, o_producer = pipes

        tile = self.first_item(schedule, False)
        while tile[0] >= 0:
            info = self.tile_info(tile, schedule)
            key_tiles = info[9]
            if info[0] >= 0 and is_leader:
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
            work, tile = self.following_tile(work, schedule)
        s_producer.tail()
        o_producer.tail()

    @no_type_check
    @cute.jit
    def softmax(
        self,
        schedule,
        work,
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
        window = schedule[7]
        prefix_start = schedule[6]
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
        tile = self.first_item(schedule, False)
        while tile[0] >= 0:
            (
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
            ) = self.tile_info(tile, schedule)
            if batch >= 0:
                packed_row = m_block * self.tile_rows + tile_row
                query_pos = packed_row // self.group_size
                first_query = m_block * (self.tile_rows // self.group_size)
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
                # Block keys this row sees: those up to its own position.
                block_end = cutlass.min(query_len, query_pos + 1)
                for key_tile in cutlass.range(key_tiles, unroll=1):
                    tile_start = prefix_base + key_tile * self.tile_keys
                    is_prefix = key_tile < prefix_tiles
                    # Visible intervals [first_start, first_end) and
                    # [second_start, second_end) of base + column. A block
                    # tile's column c holds block key row + c (row < 0 or
                    # below the tile's own start for the shifted last tile,
                    # whose leading columns repeat or precede the block);
                    # its visible keys are [block_tile * 128, block_end). A
                    # prefix tile's intervals are in column-token space.
                    # need_mask depends only on the work tile, never on the
                    # thread, because the TMEM load under it is collective.
                    block_tile = key_tile - prefix_tiles
                    row = self.current_row(block_tile, query_len)
                    base = row
                    first_start = block_tile * self.tile_keys
                    first_end = block_end
                    second_start = Int32(0)
                    second_end = Int32(0)
                    # The tile's first query row sees the fewest keys; the
                    # tile is masked when its leading columns repeat or
                    # precede the block or when that row's own position
                    # falls before the tile's last key.
                    need_mask = row < first_start or (
                        first_query + 1 < row + self.tile_keys
                    )
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
            work, tile = self.following_tile(work, schedule)
        p_producer.tail()
        stats_producer.tail()

    @no_type_check
    @cute.jit
    def correction(
        self,
        schedule,
        work,
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
        stats_consumer, sum_consumer, o_consumer = pipes

        tile = self.first_item(schedule, False)
        while tile[0] >= 0:
            (
                batch,
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
            ) = self.tile_info(tile, schedule)
            if batch >= 0:
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
            work, tile = self.following_tile(work, schedule)
