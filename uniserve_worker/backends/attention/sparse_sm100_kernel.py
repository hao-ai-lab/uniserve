# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 sparse attention with converged partial-output epilogues.

The block-64 mainloop and scheduler come from cuDNN frontend. This provider owns
its correction epilogue because context parallelism needs empty key partitions
and lane-dependent softmax underflow to obey the same warp collective protocol.
"""

import math
from typing import Optional, no_type_check

import cutlass
import cutlass.cute as cute
from cudnn.block_sparse_attention.csrc.fwd.sm100_blk64 import bsa_fwd_helpers
from cudnn.block_sparse_attention.csrc.fwd.sm100_blk64.bsa_fwd_sm100 import (
    BlockSparseAttnForwardSm100Blk64,
)
from cutlass import Float32, Int32, const_expr  # type: ignore[attr-defined]


class SparseAttentionSm100(BlockSparseAttnForwardSm100Blk64):
    """Persistent block-64 attention with warp-uniform TMEM participation."""

    # CuTe lowers this body as device IR, including pointer arithmetic and
    # dynamic predicates that its Python typing surface cannot represent.
    @no_type_check
    @cute.jit
    def correction_epilogue_combine_ws_raw(
        self,
        tmem_o0_addr: Int32,
        tmem_o1_addr: Int32,
        tidx: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        softmax_scale_log2: Float32,
        scale0: Float32,
        scale1: Float32,
        my_sum: Float32,
        my_max: Float32,
        sO: cute.Tensor,
        oStats: cute.Tensor,
        oExchange: cute.Tensor,
        reduce_mbar_addr: Int32,
        mLSE_cur: Optional[cute.Tensor] = None,
    ):
        """C++ blk64-style WS epilogue combine using raw TMEM/SMEM addressing."""
        corr_warp = tidx // cute.arch.WARP_SIZE
        lane_idx = tidx % cute.arch.WARP_SIZE
        partner_warp = corr_warp ^ 2

        # Exchange the two warp-pair row stats: (0,2) and (1,3).
        oStats[(partner_warp * 64) + lane_idx * 2 + 0] = my_sum
        oStats[(partner_warp * 64) + lane_idx * 2 + 1] = my_max
        bsa_fwd_helpers.mbar_arrive_and_wait(reduce_mbar_addr, Int32(0))

        partner_sum = oStats[(corr_warp * 64) + lane_idx * 2 + 0]
        partner_max = oStats[(corr_warp * 64) + lane_idx * 2 + 1]
        max_total = cutlass.max(my_max, partner_max)
        max_total_safe = max_total if max_total > -Float32.inf else Float32(0.0)
        my_rescale = (
            cute.math.exp2((my_max - max_total_safe) * softmax_scale_log2, fastmath=True)
            if my_sum > Float32(0.0)
            else Float32(0.0)
        )
        partner_rescale = (
            cute.math.exp2((partner_max - max_total_safe) * softmax_scale_log2, fastmath=True)
            if partner_sum > Float32(0.0)
            else Float32(0.0)
        )
        sum_total = my_sum * my_rescale + partner_sum * partner_rescale
        total_is_valid = sum_total > Float32(0.0)
        inv_sum_total = cute.arch.rcp_approx(sum_total) if total_is_valid else Float32(0.0)
        my_weight = my_rescale * inv_sum_total
        my_scale0 = scale0 * my_weight
        my_scale1 = scale1 * my_weight

        exchange_warp_base = corr_warp * 4 * 32 * 32
        exchange_addr = Int32((oExchange.iterator + exchange_warp_base + lane_idx * 4).toint())
        if const_expr(self.allow_empty_block_nums):
            is_zero_output = my_scale0 == Float32(0.0) and my_scale1 == Float32(0.0)
            # TMEM loads are warp collectives. Row-dependent softmax weights
            # can underflow on only some lanes, even for nonempty key tiles.
            # Skip TMEM only when the entire warp contributes zero output.
            if cute.arch.vote_any_sync(not is_zero_output):
                bsa_fwd_helpers.tmem_combine_store_exchange_4x32dp32b32x(
                    Int32(tmem_o0_addr),
                    Int32(tmem_o1_addr),
                    exchange_addr,
                    my_scale0,
                    my_scale1,
                )
            else:
                bsa_fwd_helpers.smem_zero_store_exchange_4x32dp32b32x(exchange_addr)
        else:
            bsa_fwd_helpers.tmem_combine_store_exchange_4x32dp32b32x(
                Int32(tmem_o0_addr),
                Int32(tmem_o1_addr),
                exchange_addr,
                my_scale0,
                my_scale1,
            )

        bsa_fwd_helpers.mbar_arrive_and_wait(reduce_mbar_addr, Int32(1))

        out_row = (corr_warp & 1) * cute.arch.WARP_SIZE + lane_idx
        if corr_warp < 2 and out_row < self.m_block_size:
            own_warp_base = corr_warp * 4 * 32 * 32
            partner_warp_base = partner_warp * 4 * 32 * 32
            lane_col_swizzle = lane_idx & 7
            for c in cutlass.range_constexpr(4):
                off = c * 32 * 32 + lane_idx * 4
                col0 = (((c * 4) + 0) ^ lane_col_swizzle) * 8
                col1 = (((c * 4) + 1) ^ lane_col_swizzle) * 8
                col2 = (((c * 4) + 2) ^ lane_col_swizzle) * 8
                col3 = (((c * 4) + 3) ^ lane_col_swizzle) * 8
                if const_expr(self.o_dtype == Float32):
                    col0 = (((c * 8) + 0) ^ lane_col_swizzle) * 4
                    col1 = (((c * 8) + 1) ^ lane_col_swizzle) * 4
                    col2 = (((c * 8) + 2) ^ lane_col_swizzle) * 4
                    col3 = (((c * 8) + 3) ^ lane_col_swizzle) * 4
                    col4 = (((c * 8) + 4) ^ lane_col_swizzle) * 4
                    col5 = (((c * 8) + 5) ^ lane_col_swizzle) * 4
                    col6 = (((c * 8) + 6) ^ lane_col_swizzle) * 4
                    col7 = (((c * 8) + 7) ^ lane_col_swizzle) * 4
                    bsa_fwd_helpers.smem_exchange_reduce_store_f32x32(
                        Int32((oExchange.iterator + own_warp_base + off).toint()),
                        Int32((oExchange.iterator + partner_warp_base + off).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col0))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col1))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col2))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col3))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col4))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col5))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col6))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col7))).toint()),
                    )
                else:
                    bsa_fwd_helpers.smem_exchange_reduce_store_bf16x32(
                        Int32((oExchange.iterator + own_warp_base + off).toint()),
                        Int32((oExchange.iterator + partner_warp_base + off).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col0))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col1))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col2))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col3))).toint()),
                    )

        cute.arch.fence_view_async_shared()

        # Keep every lane converged through the two warp-pair exchange barriers
        # above. On a partial final Q tile, predicating this store before the
        # raw inline-assembly mbarrier collectives introduces tail-dependent
        # lane divergence and can deadlock the CTA on SM100.
        if const_expr(mLSE_cur is not None):
            out_row = (corr_warp & 1) * cute.arch.WARP_SIZE + lane_idx
            valid_rows = seqlen_q - m_block * self.m_block_size
            if corr_warp < 2 and out_row < valid_rows:
                LN2 = math.log(2.0)
                lse = (
                    (max_total_safe * softmax_scale_log2 + cute.math.log2(sum_total, fastmath=True))
                    * LN2
                    if total_is_valid
                    else -Float32.inf
                )
                mLSE_cur[m_block * self.m_block_size + out_row] = lse
