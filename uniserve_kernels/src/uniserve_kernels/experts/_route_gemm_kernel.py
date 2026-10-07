# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Derived from FlashInfer 0.6.18 flashinfer/fused_moe/cute_dsl/blackwell/
# blockscaled_contiguous_grouped_gemm_finalize_fusion.py (itself from
# TensorRT-LLM tensorrt_llm/_torch/cute_dsl_kernels/blackwell/), keeping its
# deterministic expanded-row store (use_fused_finalize=False) with the
# block-scaled MMA and its scale-factor loads replaced by BF16 tcgen05 MMA.

"""BF16 grouped GEMM storing every route's row for routed experts (SM100).

FC2 of a mixture of experts over FlashInfer's MoE sort metadata: every
tile of permuted rows of ``A [M, K]`` (the gated FC1 products, in sort
order) belongs to one expert; TMA loads the tile and the expert's BF16
``B [N, K]`` rows, tcgen05 MMA accumulates in FP32 in tensor memory, and
the epilogue rounds each row to BF16 once and bulk-copies it to row
``permuted_idx_to_expanded_idx[row]`` of ``out``: route ``k`` of token
``t`` lands at row ``t * K + k``, unweighted. Those rows with the route
weights are the experts' uncombined routes; nothing is reduced across
rows, so results repeat bit for bit whatever the tile schedule.

- Warp roles: epilogue (0-3), MMA (4), TMA A/B (5), scheduler (6) and
  a row-index loader (7) that stages each tile's output rows.
- Tiles at or past ``num_non_exiting_tiles`` are skipped and rows at or
  past a tile's ``mn_limit`` are not stored, so padded metadata of a
  larger capacity is safe under CUDA graphs.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from flashinfer.fused_moe.cute_dsl.blackwell.utils import (
    blk_copy,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)


class RouteGroupedGemmKernel:
    """Grouped GEMM of permuted rows storing each route's row (FC2).

    ``mma_tiler_mn`` is the MMA tile (M, N): M 128 runs one-CTA MMA, M 256
    two-CTA MMA over a cluster pair, and the MoE sort's routing tile equals
    M; N is 128 or 256 and divides the output width.
    """

    def __init__(
        self,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        raster_along_m: bool = False,
        enable_pdl: bool = True,
    ):
        self.enable_pdl = enable_pdl
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO
            if self.use_2cta_instrs
            else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.meta_load_warp_id = 7
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                self.meta_load_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.meta_load_warp_id,
            )
        )
        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    def _setup_attributes(self):
        """Derive the tiled MMA, tile shapes, stages and layouts from the
        operand dtypes and the kernel's tile configuration.
        """  # noqa: D205
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        # Four MMA instructions per K tile: 64 BF16 values (128 bytes).
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.gemm_output_layout,
            self.out_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
            self.num_meta_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.out_dtype,
            self.cta_tile_shape_mnk,
            self.num_smem_capacity,
            self.occupancy,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        # Row-major C tile with a 16-byte row pad, from which each output row
        # is bulk-copied to its expanded row.
        swizzled_pad = 16 // (self.out_dtype.width // 8)
        self.c_smem_layout_staged = cute.make_layout(
            (
                self.cta_tile_shape_mnk[0],
                self.cta_tile_shape_mnk[1],
                self.num_c_stage,
            ),
            stride=(
                self.cta_tile_shape_mnk[1] + swizzled_pad,
                1,
                self.cta_tile_shape_mnk[0]
                * (self.cta_tile_shape_mnk[1] + swizzled_pad),
            ),
        )

        # Two accumulator stages of N FP32 columns fit tensor memory's 512
        # columns for N <= 256.
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
        )
        self.num_tmem_alloc_cols = 512

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        out: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Launch the kernel on ``stream``.

        ``a`` is the ``(M, K, 1)`` permuted rows, ``b`` the ``(N, K, E)``
        expert weights, ``out`` the ``(R, N, 1)`` route rows written at
        ``permuted_idx_to_expanded_idx[row]``.
        """
        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        self.out_dtype: type[cutlass.Numeric] = out.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.gemm_output_layout = utils.LayoutEnum.ROW_MAJOR

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(
            self.a_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(
            self.b_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Compute the grid size
        self.tile_sched_params, grid = self._compute_grid(
            (a.shape[0], b.shape[0], a.shape[2]),
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        epi_tile_m = cute.size(self.epi_tile[0])
        epi_tile_n = cute.size(self.epi_tile[1])
        epi_tile_size = epi_tile_m * epi_tile_n
        num_epilogue_threads = 32 * len(self.epilog_warp_id)
        self.ttr_racc_size = epi_tile_size // num_epilogue_threads
        # BF16 rows: eight values per 16-byte vector.
        self.epi_layout = cute.make_layout(
            shape=(self.ttr_racc_size // 8, 4, 2), stride=(8, 2, 1)
        )
        self.epi_loop_size = self.ttr_racc_size // 8
        self.element_offset = 8

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # (bidx, bidy, bidz, valid, mn_limit)
            s_info: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            meta_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_meta_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (MMA, MMA_M, MMA_K, STAGE)
            s_a: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            s_b: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            s_c: cute.struct.Align[
                cute.struct.MemRange[
                    self.out_dtype,
                    cute.cosize(self.c_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            # Output row of each tile row, staged by the row-index loader.
            route_row: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    self.cta_tile_shape_mnk[0] * self.num_meta_stage,
                ],
                1,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            out,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.epi_layout,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        m_a_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        m_b_nkl: cute.Tensor,
        out: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        epi_layout: cute.Layout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """The persistent warp-specialized grouped GEMM (see the class)."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory
        # dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        num_acc_consumer_threads = (
            len(self.epilog_warp_id)
            * self.threads_per_warp
            * (2 if use_2cta_instrs else 1)
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Initialize metadata pipeline (meta loader warp -> epilogue warps)
        meta_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        meta_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        meta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.meta_mbar_ptr.data_ptr(),
            num_stages=self.num_meta_stage,
            producer_group=meta_pipeline_producer_group,
            consumer_group=meta_pipeline_consumer_group,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )
        tmem.allocate(self.num_tmem_alloc_cols)

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/C/Scale/ExpandedIdx
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        s_a = storage.s_a.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        s_b = storage.s_b.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        s_c = storage.s_c.get_tensor(c_smem_layout_staged)

        # (bidx, bidy, bidz, valid)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        s_info = storage.s_info.get_tensor(info_layout)

        # Output row of each tile row, staged by the meta loader warp:
        # (row, stage)
        meta_layout = cute.make_layout(
            (self.cta_tile_shape_mnk[0], self.num_meta_stage),
            stride=(1, self.cta_tile_shape_mnk[0]),
        )
        s_route_row = storage.route_row.get_tensor(meta_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(
            self.is_a_mcast or self.is_b_mcast or use_2cta_instrs
        ):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        g_a_mkl = cute.local_tile(
            m_a_mkl,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        g_b_nkl = cute.local_tile(
            m_b_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )

        k_tile_cnt = cutlass.Int32(cute.size(g_a_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        t_cg_a = thr_mma.partition_A(g_a_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        t_cg_b = thr_mma.partition_B(g_b_nkl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        t_as_a, t_ag_a = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(s_a, 0, 3),
            cute.group_modes(t_cg_a, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        t_bs_b, t_bg_b = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(s_b, 0, 3),
            cute.group_modes(t_cg_b, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        t_cr_a = tiled_mma.make_fragment_A(s_a)
        # (MMA, MMA_N, MMA_K, STAGE)
        t_cr_b = tiled_mma.make_fragment_B(s_b)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # (MMA, MMA_M, MMA_N, STAGE)
        t_ct_acc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        g_c_mnl = cute.local_tile(
            out,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )

        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        t_cg_c = thr_mma.partition_C(g_c_mnl)

        #
        # Persistent tile scheduler state. Emit the first tile before the
        # cluster/grid dependency wait so consumers can start as soon as the
        # wait completes; the main scheduler loop resumes from the next tile.
        #
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        tile_info_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_tile_stage
        )

        num_valid_tiles = num_non_exiting_tiles[0]
        is_continue = cutlass.Boolean(1)

        if warp_idx == self.sched_warp_id:
            if work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                    tiled_mma.thr_id.shape
                )
                expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                tile_idx = mma_tile_coord_m

                if tile_idx < num_valid_tiles:
                    tile_info_pipeline.producer_acquire(
                        tile_info_producer_state
                    )
                    mn_limit = tile_idx_to_mn_limit[tile_idx]
                    with cute.arch.elect_one():
                        s_info[(0, tile_info_producer_state.index)] = (
                            cur_tile_coord[0]
                        )
                        s_info[(1, tile_info_producer_state.index)] = (
                            cur_tile_coord[1]
                        )
                        s_info[(2, tile_info_producer_state.index)] = expert_idx
                        s_info[(3, tile_info_producer_state.index)] = (
                            cutlass.Int32(work_tile.is_valid_tile)
                        )
                        s_info[(4, tile_info_producer_state.index)] = mn_limit
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()
                else:
                    if cutlass.const_expr(not self.raster_along_m):
                        is_continue = cutlass.Boolean(0)

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        #
        # Cluster wait after early scheduler/TMEM setup
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        griddepcontrol_wait()

        #
        # Specialized Schedule warp
        #
        if warp_idx == self.sched_warp_id:
            #
            # Persistent tile scheduling loop, starting after the pre-emitted
            # first tile.
            #
            if cutlass.const_expr(self.raster_along_m):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                    tile_idx = mma_tile_coord_m
                    if tile_idx < num_valid_tiles:
                        tile_info_pipeline.producer_acquire(
                            tile_info_producer_state
                        )
                        mn_limit = tile_idx_to_mn_limit[tile_idx]
                        with cute.arch.elect_one():
                            s_info[(0, tile_info_producer_state.index)] = (
                                cur_tile_coord[0]
                            )
                            s_info[(1, tile_info_producer_state.index)] = (
                                cur_tile_coord[1]
                            )
                            s_info[(2, tile_info_producer_state.index)] = (
                                expert_idx
                            )
                            s_info[(3, tile_info_producer_state.index)] = (
                                cutlass.Int32(work_tile.is_valid_tile)
                            )
                            s_info[(4, tile_info_producer_state.index)] = (
                                mn_limit
                            )
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(
                            tile_info_producer_state
                        )
                        tile_info_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            else:
                while work_tile.is_valid_tile and is_continue:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                    tile_idx = mma_tile_coord_m
                    if tile_idx < num_valid_tiles:
                        tile_info_pipeline.producer_acquire(
                            tile_info_producer_state
                        )
                        mn_limit = tile_idx_to_mn_limit[tile_idx]
                        with cute.arch.elect_one():
                            s_info[(0, tile_info_producer_state.index)] = (
                                cur_tile_coord[0]
                            )
                            s_info[(1, tile_info_producer_state.index)] = (
                                cur_tile_coord[1]
                            )
                            s_info[(2, tile_info_producer_state.index)] = (
                                expert_idx
                            )
                            s_info[(3, tile_info_producer_state.index)] = (
                                cutlass.Int32(work_tile.is_valid_tile)
                            )
                            s_info[(4, tile_info_producer_state.index)] = (
                                mn_limit
                            )
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(
                            tile_info_producer_state
                        )
                        tile_info_producer_state.advance()

                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                s_info[(0, tile_info_producer_state.index)] = (
                    work_tile.tile_idx[0]
                )
                s_info[(1, tile_info_producer_state.index)] = (
                    work_tile.tile_idx[1]
                )
                s_info[(2, tile_info_producer_state.index)] = -1
                s_info[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                s_info[(4, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out
            # tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = s_info[(0, tile_info_consumer_state.index)]
            tile_info[1] = s_info[(1, tile_info_consumer_state.index)]
            tile_info[2] = s_info[(2, tile_info_consumer_state.index)]
            tile_info[3] = s_info[(3, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                t_ag_a_slice = t_ag_a[(None, mma_tile_coord_mnl[0], None, 0)]
                # ((atom_v, rest_v), loopK)
                t_bg_b_slice = t_bg_b[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile =
                # prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    t_ag_a_k = t_ag_a_slice[(None, ab_producer_state.count)]
                    t_bg_b_k = t_bg_b_slice[(None, ab_producer_state.count)]
                    t_as_a_pipe = t_as_a[(None, ab_producer_state.index)]
                    t_bs_b_pipe = t_bs_b[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(
                        ab_producer_state
                    )

                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        t_ag_a_k,
                        t_as_a_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        t_bg_b_k,
                        t_bs_b_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile =
                    # prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = s_info[(0, tile_info_consumer_state.index)]
                tile_info[1] = s_info[(1, tile_info_consumer_state.index)]
                tile_info[2] = s_info[(2, tile_info_consumer_state.index)]
                tile_info[3] = s_info[(3, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            t_ct_acc_base = cute.make_tensor(acc_tmem_ptr, t_ct_acc_fake.layout)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out
            # tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = s_info[(0, tile_info_consumer_state.index)]
            tile_info[1] = s_info[(1, tile_info_consumer_state.index)]
            tile_info[2] = s_info[(2, tile_info_consumer_state.index)]
            tile_info[3] = s_info[(3, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                t_ct_acc = t_ct_acc_base[
                    (None, None, None, acc_producer_state.index)
                ]

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in cutlass.range(k_tile_cnt):  # noqa: B007
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        # t_ct_acc += t_cr_a * t_cr_b
                        num_kblocks = cute.size(t_cr_a, mode=[2])

                        for kblock_idx in cutlass.range(
                            num_kblocks, unroll_full=True
                        ):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            cute.gemm(
                                tiled_mma,
                                t_ct_acc,
                                t_cr_a[kblock_coord],
                                t_cr_b[kblock_coord],
                                t_ct_acc,
                            )

                            # Enable accumulate on t_ct_acc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()
                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = s_info[(0, tile_info_consumer_state.index)]
                tile_info[1] = s_info[(1, tile_info_consumer_state.index)]
                tile_info[2] = s_info[(2, tile_info_consumer_state.index)]
                tile_info[3] = s_info[(3, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized metadata loader warp
        #
        if warp_idx == self.meta_load_warp_id:
            meta_lane = tidx % self.threads_per_warp

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            meta_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_meta_stage
            )
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = s_info[(0, tile_info_consumer_state.index)]
            tile_info[2] = s_info[(2, tile_info_consumer_state.index)]
            tile_info[3] = s_info[(3, tile_info_consumer_state.index)]
            tile_info[4] = s_info[(4, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]

                meta_pipeline.producer_acquire(meta_producer_state)
                meta_stage = meta_producer_state.index
                # Strided row assignment keeps the permuted_idx loads and smem
                # stores coalesced (each fixed j: 32 lanes touch 32 contiguous
                # rows). Rows at or past mn_limit are padding; the epilogue
                # never stores them, so their staged index is irrelevant.
                for j in cutlass.range(
                    self.cta_tile_shape_mnk[0] // self.threads_per_warp,
                    unroll_full=True,
                ):
                    r = meta_lane + j * self.threads_per_warp
                    permuted_row = tile_m_start + r
                    expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                    s_route_row[(r, meta_stage)] = cutlass.max(
                        expanded_idx, cutlass.Int32(0)
                    )
                cute.arch.fence_proxy("async.shared", space="cta")
                meta_pipeline.producer_commit(meta_producer_state)
                meta_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = s_info[(0, tile_info_consumer_state.index)]
                tile_info[2] = s_info[(2, tile_info_consumer_state.index)]
                tile_info[3] = s_info[(3, tile_info_consumer_state.index)]
                tile_info[4] = s_info[(4, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            meta_pipeline.producer_tail(meta_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            t_ct_acc_base = cute.make_tensor(tmem_ptr, t_ct_acc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                t_tr_t_acc_base,
                t_tr_r_acc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, t_ct_acc_base, t_cg_c, epi_tile, use_2cta_instrs
            )

            t_tr_r_c = cute.make_rmem_tensor(t_tr_r_acc.shape, self.out_dtype)
            tiled_copy_r2s, t_rs_r_c, t_rs_s_c = (
                self.epilog_smem_copy_and_partition(
                    epi_tidx, t_tr_r_c, s_c, tiled_copy_t2r
                )
            )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            meta_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_meta_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = s_info[(0, tile_info_consumer_state.index)]
            tile_info[1] = s_info[(1, tile_info_consumer_state.index)]
            tile_info[3] = s_info[(3, tile_info_consumer_state.index)]
            tile_info[4] = s_info[(4, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]
                permuted_row = tile_m_start + epi_tidx
                is_valid_row = permuted_row < tile_info[4]

                # Wait for the tile's output rows staged by the meta loader.
                meta_pipeline.consumer_wait(meta_consumer_state)
                acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                t_tr_t_acc = t_tr_t_acc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                t_tr_t_acc = cute.group_modes(
                    t_tr_t_acc, 3, cute.rank(t_tr_t_acc)
                )

                # Stage the whole tile in shared memory, one row per thread.
                subtile_cnt = cute.size(t_tr_t_acc.shape, mode=[3])

                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    t_tr_t_acc_mn = t_tr_t_acc[
                        (None, None, None, real_subtile_idx)
                    ]

                    cute.copy(tiled_copy_t2r, t_tr_t_acc_mn, t_tr_r_acc)

                    # Each route row rounds to BF16 once, unweighted.
                    acc_vec = t_tr_r_acc.load()
                    t_rs_r_c.store(acc_vec.to(self.out_dtype))
                    if is_valid_row:
                        cute.copy(
                            tiled_copy_r2s,
                            t_rs_r_c,
                            t_rs_s_c[(None, None, real_subtile_idx, None)],
                        )

                # Make all R2S smem writes visible to the async bulk-copy proxy.
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                is_partial_tile = (
                    tile_info[4] < tile_m_start + self.cta_tile_shape_mnk[0]
                )
                #
                # Async arrive accumulator buffer empty
                #
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                if is_partial_tile:
                    self.epilog_sync_barrier.arrive_and_wait()

                # Copy each valid row to its expanded row; a partial tile
                # redistributes rows so every warp copies contiguous rows.
                copy_row = epi_tidx
                if is_partial_tile:
                    copy_row = (epi_tidx % self.threads_per_warp) * len(
                        self.epilog_warp_id
                    ) + (epi_tidx // self.threads_per_warp)
                copy_permuted_row = tile_m_start + copy_row
                is_valid_copy_row = copy_permuted_row < tile_info[4]
                if is_valid_copy_row:
                    coord_n = tile_info[1] * self.cta_tile_shape_mnk[1]
                    valid_columns = cutlass.min(
                        cutlass.Int64(out.shape[1]) - coord_n,
                        cutlass.Int64(self.cta_tile_shape_mnk[1]),
                    )
                    if valid_columns > 0:
                        route_row = s_route_row[
                            (copy_row, meta_consumer_state.index)
                        ]
                        scatter_out_offset = cute.domain_offset(
                            (route_row, coord_n, 0), out
                        )
                        valid_copy_size = cutlass.Int32(
                            valid_columns * (self.out_dtype.width // 8)
                        )
                        # Each output row ends on a 16-byte boundary, as the
                        # bulk copy requires of its size and address.
                        blk_copy(
                            scatter_out_offset,
                            s_c[copy_row, None, 0],
                            valid_copy_size,
                        )

                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.epilog_sync_barrier.arrive_and_wait()

                # Release the prefetched metadata slot for this tile.
                meta_pipeline.consumer_release(meta_consumer_state)
                meta_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = s_info[(0, tile_info_consumer_state.index)]
                tile_info[1] = s_info[(1, tile_info_consumer_state.index)]
                tile_info[3] = s_info[(3, tile_info_consumer_state.index)]
                tile_info[4] = s_info[(4, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        griddepcontrol_launch_dependents()

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        t_acc: cute.Tensor,
        g_c_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: cutlass.Boolean | bool,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Partition the accumulators for tensor-memory-to-register loads.

        Returns the T2R tiled copy, this thread's partition of the staged
        accumulators and a register fragment for one epilogue subtile.
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.gemm_output_layout,
            self.out_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        t_acc_epi = cute.flat_divide(
            t_acc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, t_acc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        t_tr_t_acc = thr_copy_t2r.partition_S(t_acc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        g_c_mnl_epi = cute.flat_divide(
            g_c_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        t_tr_g_c = thr_copy_t2r.partition_D(g_c_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        t_tr_r_acc = cute.make_rmem_tensor(
            t_tr_g_c[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )

        return tiled_copy_t2r, t_tr_t_acc, t_tr_r_acc

    def epilog_smem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        t_tr_r_c: cute.Tensor,
        s_c: cute.Tensor,
        tiled_copy_t2r: cute.TiledCopy,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Create tiled copy for register to shared memory (R2S)."""
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.out_dtype,
        )

        tiled_copy_r2s = cute.make_tiled_copy_D(atom, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        t_rs_s_c = thr_copy_r2s.partition_D(s_c)
        t_rs_r_c = tiled_copy_r2s.retile(t_tr_r_c)
        return tiled_copy_r2s, t_rs_r_c, t_rs_s_c

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        cta_tile: cute.Tile,
        num_smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int, int, int, int]:
        """Return (accumulator, A/B, C, tile-info, row-index) stage counts.

        Two accumulator stages always fit tensor memory (N <= 256); A/B
        stages fill shared memory after one padded C tile, the row indices
        and the barriers.
        """
        num_acc_stage = 2
        num_c_stage = 1
        num_tile_stage = 2
        num_meta_stage = 2
        meta_smem_bytes = (
            cta_tile[0] * (cutlass.Int32.width // 8) * num_meta_stage
        )

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )
        # Rows padded to keep every row 16-byte aligned for the bulk copy.
        swizzled_pad = 16 // (out_dtype.width // 8)
        c_smem_layout_staged_one = cute.make_layout(
            (cta_tile[0], cta_tile[1]), stride=(cta_tile[1] + swizzled_pad, 1)
        )

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        c_bytes = (
            cute.size_in_bytes(out_dtype, c_smem_layout_staged_one)
            * num_c_stage
        )

        num_ab_stage = (
            num_smem_capacity // occupancy
            - (mbar_helpers_bytes + c_bytes + meta_smem_bytes)
        ) // ab_bytes_per_stage
        return (
            num_acc_stage,
            num_ab_stage,
            num_c_stage,
            num_tile_stage,
            num_meta_stage,
        )

    @staticmethod
    def _compute_grid(
        gemm_shape: tuple[int, int, int],
        cta_tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Return the persistent tile scheduler parameters and grid.

        ``gemm_shape`` is the GEMM's (M, N, L).
        """
        (m, n, l) = gemm_shape  # noqa: E741

        num_ctas_m = cute.ceil_div(m, cta_tile_shape_mnk[0])
        num_ctas_n = cute.ceil_div(n, cta_tile_shape_mnk[1])
        num_ctas_l = l

        num_ctas_mnl = (num_ctas_m, num_ctas_n, num_ctas_l)
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        tile_idx_to_expert_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        experts: cutlass.Int64,
        routes: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Bind row-major pointers and launch.

        ``a`` is ``[m, k]``, ``b`` ``[experts, n, k]`` and ``out``
        ``[routes, n]``; the tile metadata holds ``m / tile_size`` entries
        and ``permuted_idx_to_expanded_idx`` ``m``.
        """
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, experts), order=(1, 0, 2)),
        )
        out = cute.make_tensor(
            out_ptr,
            layout=cute.make_ordered_layout((routes, n, 1), order=(1, 0, 2)),
        )
        tile_idx_to_expert_idx = cute.make_tensor(
            tile_idx_to_expert_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        permuted_idx_to_expanded_idx = cute.make_tensor(
            permuted_idx_to_expanded_idx_ptr, layout=cute.make_layout((m,))
        )
        return self(
            a,
            b,
            out,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
