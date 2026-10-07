#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t kGranKA, uint32_t kGranKB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kSwapAB,
          GemmType kGemmType, bool kWithAccumulation,
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t,
          typename epilogue_type_t>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_blockscaled_gemm_1d1d_impl(int* grouped_layout,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    // mxf4nvf4's K step is 64 elements and is MANDATORY -- issuing it at 32 traps
    // with an illegal instruction (measured).  The operands must therefore reach
    // it PACKED, two values per byte: 64 values in 32 bytes.  Feeding it UNPACKED
    // smem instead makes step 0 read only the first 32 values of its 64 and step
    // 1 read past them, which a B=identity probe confirmed exactly -- the first
    // 32 columns of every BLOCK_K=128 tile correct, the rest wrong.
    // See track.md: the packed-smem conversion is the remaining P6.1 work.
    constexpr bool kIsNvFp4 = (kGranKA == 16 and kGranKB == 16);
    constexpr uint32_t UMMA_K = kIsNvFp4 ? 64 : 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT(BLOCK_K == 128 or (kIsNvFp4 and BLOCK_K == 256),
                     "Invalid block K");   // packed NVFP4 takes 256 values = 128B
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    // SF configs
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    // Two orthogonal axes.  Coarser than group-32 spreads ONE word over several
    // stages; finer (group-16) packs SEVERAL words into one stage.  A uint32
    // holds four scales, so a BLOCK_K=128 tile spans 128/(gran*4) words.
    constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 128 ? 4 : 1;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 128 ? 4 : 1;
    // A uint32 holds four scales, so a BLOCK_K tile spans BLOCK_K/(gran*4) of
    // them: 1 at group-32/BLOCK_K=128, and 4 at group-16/BLOCK_K=256.
    constexpr uint32_t kNumSFAWordsPerStage = kGranKA == 128 ? 1 : BLOCK_K / (kGranKA * 4);
    // Gran-128 packs FOUR K-blocks' scales into ONE word (the 17th
    // expression of the packed/granularity invariant): the load fires every
    // fourth stage, sf_id selects the in-group byte, and the naive
    // BLOCK_K/(gran*4) FLOORS TO ZERO -- which silently zeroes the SF smem,
    // the copies, the TMEM columns and every product after them.
    constexpr uint32_t kNumSFBWordsPerStage = kGranKB == 128 ? 1 : BLOCK_K / (kGranKB * 4);
    // SF always reaches the kernel uint32-PACKED, four scales per element -- the
    // same container the UE8M0 path uses, only with an E4M3 payload.  That is
    // not cosmetic: one UTCCP chunk moves 128 rows x 4 BYTES and needs each
    // row's four scales CONTIGUOUS.  A raw e4m3 [mn, k/16] tensor is k-major in
    // smem, so a row's scales land strided by BLOCK_M and the chunk reads
    // garbage.  Packed, element count == word count and the whole layout
    // collapses onto the group-32 one.
    DG_STATIC_ASSERT(kGranKA == 16 or kGranKA == 32 or kGranKA == 128, "Invalid granularity K for A");
    DG_STATIC_ASSERT(kGranKB == 16 or kGranKB == 32 or kGranKB == 128, "Invalid granularity K for B");
    DG_STATIC_ASSERT(kIsNvFp4 or (kGranKA != 16 and kGranKB != 16),
                     "Group-16 needs mxf4nvf4, which scales BOTH operands per 16");
    DG_STATIC_ASSERT((kGemmType != GemmType::KGroupedContiguous) or kGranKA == kGranKB, "K-grouped SF requires kGranKA == kGranKB");

    // Epilogue configs
    // Always enable pipeline for better performance
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads: STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    // Packed FP4 stores two values per byte, so smem is half what `sizeof` implies.
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = mma::sm100::smem_bytes_of<a_dtype_t>(LOAD_BLOCK_M * BLOCK_K);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = mma::sm100::smem_bytes_of<b_dtype_t>(LOAD_BLOCK_N * BLOCK_K);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t) * kNumSFAWordsPerStage;
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t) * kNumSFBWordsPerStage;
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    // NOTES: Make sure we have enough shared memory for UMMA padding
    // Packed FP4 is half a byte per value here too -- this was the one
    // remaining raw `sizeof(a_dtype_t)` in the smem sizing.
    constexpr uint32_t UMMA_A_SIZE_PER_STAGE = mma::sm100::smem_bytes_of<a_dtype_t>(
        math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for UMMA");

    // Tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32 * kNumSFAWordsPerStage;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32 * kNumSFBWordsPerStage;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? cute::cluster_sync() : void();

    // Utils
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const auto shape_sfa_k = math::ceil_div(shape_k, kGranKA * 4);
    const auto shape_sfb_k = math::ceil_div(shape_k, kGranKB * 4);

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE); 
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SFA/SFB shared memory
    auto sf_start_ptr = reinterpret_cast<uint8_t*>(smem_b[kNumStages]);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_sfb[kNumStages]);;
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });
    auto tmem_ptr_in_smem  = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive at all CTAs
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // Arrive only at the leader CTA
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs, kGranKA * 4>(
        shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Use dynamic load block M, when swap-AB is enabled
            const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), sched::IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN> (
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, 1, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_b_idx, n_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, 1, batch_idx);
                // The transaction counts GMEM bytes.  Only the UNPACKED FP4 type
                // expands on the way into smem (half a byte in GMEM, a full byte in
                // smem); packed FP4 and FP8 are 1:1, so they must not be halved.
                constexpr uint32_t kADiv = std::is_same_v<a_dtype_t, cutlass::detail::float_e2m1_unpacksmem_t> ? 2 : 1;
                constexpr uint32_t kBDiv = std::is_same_v<b_dtype_t, cutlass::detail::float_e2m1_unpacksmem_t> ? 2 : 1;
                auto num_arrival_bytes = SMEM_A_SIZE_PER_STAGE / kADiv + SMEM_B_SIZE_PER_STAGE / kBDiv;

                // Issue SFA and SFB TMAs at certain stages
                // No swizzling, so one TMA for one SF is enough
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    uint32_t sfa_m_idx = m_block_idx * BLOCK_M;
                    uint32_t sfa_k_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::SF_K>(
                        shape_sfa_k, 1, math::ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad));
                    // One copy per SF word, each landing at its own 128-row chunk:
                    // the transposer/UTCCP/TMEM consume kw-major chunks of
                    // kNumUTCCPAlignedElems rows, and a compact multi-word box
                    // only matches that when SF_BLOCK == BLOCK (SFA at 128).
                    // Splitting is byte-identical when WordsPerStage == 1 (FP8).
                    #pragma unroll
                    for (uint32_t kw = 0; kw < kNumSFAWordsPerStage; ++ kw)
                        tma::copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[stage_idx],
                                                 smem_sfa[stage_idx] + kw * kNumUTCCPAlignedElems,
                                                 sfa_m_idx, sfa_k_idx * kNumSFAWordsPerStage + kw);
                    num_arrival_bytes += BLOCK_M * sizeof(uint32_t) * kNumSFAWordsPerStage;
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                    uint32_t sfb_k_idx = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(
                        shape_sfb_k, 1, math::ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad), m_block_idx);
                    #pragma unroll
                    for (uint32_t kw = 0; kw < kNumSFBWordsPerStage; ++ kw)
                        tma::copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[stage_idx],
                                                 smem_sfb[stage_idx] + kw * kNumUTCCPAlignedElems,
                                                 sfb_n_idx, sfb_k_idx * kNumSFBWordsPerStage + kw);
                    num_arrival_bytes += BLOCK_N * sizeof(uint32_t) * kNumSFBWordsPerStage;
                }

                // Arrive at full barriers
                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        // The descriptor's `scale_format_` carries the scale DTYPE, which differs
        // from the MX formats: NVFP4 scales are E4M3, MXFP8/MXFP4 are UE8M0.
        // (The scale GRANULARITY is a separate, instruction-level qualifier.)
        using sf_dtype_t = std::conditional_t<kIsNvFp4,
                                              cutlass::float_ue4m3_t,
                                              cutlass::float_ue8m0_t>;
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc_block_scaled<b_dtype_t, a_dtype_t, float, sf_dtype_t,
                                                                             UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, float, sf_dtype_t,
                                                                             UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Wait tensor memory empty barrier arrival
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                auto umma_arrive = [](const uint64_t* barrier) {
                    if constexpr (kNumMulticast == 1) {
                        cutlass::arch::umma_arrive(barrier);
                    } else {
                        constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    }
                };
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic update of UMMA N based on effective M, when swap-AB is enabled
            if constexpr (kSwapAB) {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }

            // Launch MMAs
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA and SF-transpose arrival
                with_sf_full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                if (cute::elect_one_sync()) {
                    // Do SF copy at certain stages
                    // TODO: process shared memory descriptor by addition
                    using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                        cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
                    const uint32_t sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
                    if (sfa_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems * kNumSFAWordsPerStage; ++ i) {
                            auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                    }
                    const uint32_t sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad;
                    if (sfb_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems * kNumSFBWordsPerStage; ++ i) {
                            auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                        }
                    }

                    // Issue UMMA
                    using mma_t = cute::conditional_t<kIsNvFp4,
                        cute::conditional_t<kNumMulticast == 1,
                            ptx::SM100_MMA_MXF4NVF4_SS, ptx::SM100_MMA_MXF4NVF4_2x1SM_SS>,
                        cute::conditional_t<kNumMulticast == 1,
                            ptx::SM100_MMA_MXF8F6F4_SS, ptx::SM100_MMA_MXF8F6F4_2x1SM_SS>>;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        // At group-32 one uint32 (four scales) feeds FOUR UMMA_K=32
                        // steps, so the column is fixed and sf_id picks the scale
                        // within it.  At group-16 one uint32 feeds exactly ONE
                        // UMMA_K=64 step, so consecutive steps must read
                        // consecutive UTCCP chunks -- which the loop above placed
                        // 4 columns apart -- and sf_id stays 0.
                        // One uint32 holds four group-16 scales = 64 elements = one
                        // UMMA_K=64 step, so step k reads UTCCP chunk k (placed 4
                        // columns apart) and sf_id stays 0.
                        const uint32_t sfa_id = kIsNvFp4 ? 0u : (kGranKA == 128 ? sfa_stage_in_group_idx : k);
                        const uint32_t sfb_id = kIsNvFp4 ? 0u : (kGranKB == 128 ? sfb_stage_in_group_idx : k);
                        const uint32_t sfa_col = kTmemStartColOfSFA + (kIsNvFp4 ? k * 4 : 0);
                        const uint32_t sfb_col = kTmemStartColOfSFB + (kIsNvFp4 ? k * 4 : 0);
                        const auto runtime_instr_desc = kSwapAB ?
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfb_id, sfa_id):
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                        if constexpr (kSwapAB) {
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc,
                                       sfb_col, sfa_col);
                        } else {
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc,
                                       sfa_col, sfb_col);
                        }
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }
    } else if (warp_idx == 2) {
        // UTCCP transposer
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);

                // Transpose for UTCCP at certain stages
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    #pragma unroll
                    // NVFP4 packs kNumSFAWordsPerStage words per stage; every 512B
                    // chunk feeds one UTCCP and must be transposed (12th expression
                    // of the granularity invariant -- the loop bound had missed it).
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems * kNumSFAWordsPerStage; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems * kNumSFBWordsPerStage; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }

                // Arrive
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            if constexpr (kSwapAB) {
                const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                epilogue::sm100_store_cd_swap_ab<
                    BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue_type_t>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 effective_m,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            } else {
                epilogue::sm100_store_cd<
                    BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue_type_t>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            }
        }
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
