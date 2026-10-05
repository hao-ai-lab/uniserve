// SPDX-License-Identifier: Apache-2.0
// Derived from FastVideo block-sparse SM100 attention. See ../LICENSE.

#ifndef BLOCK_SPARSE_VSA_LAUNCH_SM100A_CUH
#define BLOCK_SPARSE_VSA_LAUNCH_SM100A_CUH

// Native SM100 block-sparse attention launch with independent Q/K extents.
//
// Owns the host side of fmha_context_bf16_gen_kernel: the argument struct,
// geometry checks, TMA tensor-map encoding, the dynamic shared-memory size
// and the launch configuration. Two compile-time macros select the block size
// and tensor layout: VSA_BLK128 (64- or 128-token sparse blocks, defined in
// the kernel header) and VSA_BHSD (tensor maps addressed by explicit row and
// head strides, or a fixed contiguous [tokens][heads][dim] layout). The torch
// binding in attention.cu builds block 64 with VSA_BHSD.

#include "block_sparse_kernel_sm100a.cuh"

namespace VSA_NAMESPACE {

// Device pointers and geometry for one launch. The metadata arrays are read by
// the kernel only; their values are not validated on the host.
struct BlockSparseVsaArgs {
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  __nv_bfloat16* o;
  float* lse;                  // [batch, num_heads, seqlen] fp32 log2 of the softmax normalizer, or nullptr

  const int* q2k_idx;              // [batch*num_heads*num_blocks, max_kv] int32
  const int* q2k_num;              // [batch*num_heads*num_blocks] int32
  const int* variable_block_sizes; // [key_seqlen / BLOCK] int32, valid tokens per block

  int batch;
  int num_heads;
  int seqlen;
  int key_seqlen;
  // Element strides of the head-major tensor maps; Q, O, K and V may each use
  // their own. Row-major [rows, heads, dim] tensors pass head stride dim and
  // row stride heads * dim.
  int64_t query_row_stride;
  int64_t query_head_stride;
  int64_t output_row_stride;
  int64_t output_head_stride;
  int64_t key_row_stride;
  int64_t key_head_stride;
  int64_t value_row_stride;
  int64_t value_head_stride;
  int head_dim;
  int num_blocks;
  int max_kv;
  float sm_scale;
};

// Reject unsupported geometry before constructing tensor maps or launching.
// An odd num_blocks is accepted: each head's trailing query tile forms a work
// item with an absent partner (see decode_workitem).
__host__ inline cudaError_t block_sparse_supported(const BlockSparseVsaArgs& a) {
  if (a.head_dim != HEAD_DIM) return cudaErrorInvalidValue;      // compile-time in the kernel
  if (a.seqlen != a.num_blocks * BLOCK || a.key_seqlen < BLOCK || a.key_seqlen % BLOCK != 0) return cudaErrorInvalidValue;
  if (a.max_kv < 1 || a.num_blocks < 1) return cudaErrorInvalidValue;
  if (a.q == nullptr || a.k == nullptr || a.o == nullptr) return cudaErrorInvalidValue;
  if (a.q2k_idx == nullptr || a.q2k_num == nullptr) return cudaErrorInvalidValue;
  // Valid key extents prevent padded keys from contributing to attention.
  if (a.variable_block_sizes == nullptr) return cudaErrorInvalidValue;
  // V shares the key tensor layout.
  if (a.v == nullptr) return cudaErrorInvalidValue;
  return cudaSuccess;
}

__host__ inline cudaError_t launch_block_sparse_sm100a(const BlockSparseVsaArgs& a,
                                                           cudaStream_t stream) {
  const cudaError_t sup = block_sparse_supported(a);
  if (sup != cudaSuccess) return sup;

  const int B = a.batch, H = a.num_heads, S = a.seqlen, hd = a.head_dim;
  const int num_blocks = a.num_blocks, max_kv = a.max_kv;
  const long tq = (long)B * S;
  const int K = a.key_seqlen;
  const long tk = (long)B * K;
  // One work item is a pair of adjacent BLOCK-row query tiles of one head.
  const int packed_mtiles_per_seq = (num_blocks + 1) / 2;
  const int total_work = B * H * packed_mtiles_per_seq;
  constexpr bool BHSD = VSA_BHSD;

  // Tensor-map dimensions and boxes are listed innermost first; strides are
  // in bytes (element strides times 2) for dimensions 1 and up. The 128-byte
  // swizzle limits a box row to 128 bytes, so the 128 channels of a head are
  // addressed as Q_SUBTILES (or K_SUBTILES) subtiles of SUB_COLS_BF16.
  CUtensorMap tq_, tk_, tv_, to_;
  {
    // Q and O: [channel in subtile, head (batch * heads with BHSD), row,
    // subtile]. One box is one head's M_TILE rows with all subtiles. With
    // BHSD, rows past seqlen of an unpaired trailing tile load as zeros and
    // are not stored.
    uint64_t gd[4] = { (uint64_t)SUB_COLS_BF16, BHSD ? (uint64_t)((long)B * H) : (uint64_t)H,
                       BHSD ? (uint64_t)S : (uint64_t)tq, (uint64_t)Q_SUBTILES };
    uint64_t gs[3] = { BHSD ? (uint64_t)a.query_head_stride * 2u : (uint64_t)hd * 2u,
                       BHSD ? (uint64_t)a.query_row_stride * 2u : (uint64_t)((long)H * hd) * 2u,
                       (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, (uint32_t)Q_SUBTILES };
    uint32_t es[4] = { 1u, 1u, 1u, 1u };
    if (cuTensorMapEncodeTiled(&tq_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                               const_cast<__nv_bfloat16*>(a.q), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
    if constexpr (BHSD) {
      gs[0] = (uint64_t)a.output_head_stride * 2u;
      gs[1] = (uint64_t)a.output_row_stride * 2u;
    }
    if (cuTensorMapEncodeTiled(&to_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, a.o, gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }
  {
    // K and V with BHSD: [channel in subtile, key row, subtile, head]; the
    // contiguous layout folds the head into the subtile index. One box is one
    // BLOCK-row key block of one subtile (all subtiles in the 128-token build).
    uint64_t gd[4] = { (uint64_t)SUB_COLS_BF16,
                       BHSD ? (uint64_t)K : (uint64_t)tk,
                       BHSD ? (uint64_t)(hd / SUB_COLS_BF16)
                            : (uint64_t)((long)H * hd / SUB_COLS_BF16),
                       (uint64_t)((long)B * H) };
    uint64_t gs[3] = { BHSD ? (uint64_t)a.key_row_stride * 2u : (uint64_t)((long)H * hd) * 2u,
                       (uint64_t)SUB_COLS_BF16 * 2u,
                       (uint64_t)a.key_head_stride * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, (uint32_t)BLOCK,
                       BLK128 ? (uint32_t)K_SUBTILES : 1u, 1u };
    uint32_t es[4] = { 1u, 1u, 1u, 1u };
    if (cuTensorMapEncodeTiled(&tk_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, BHSD ? 4 : 3,
                               const_cast<__nv_bfloat16*>(a.k), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
    // K and V share logical geometry but can have independent physical strides.
    if constexpr (BHSD) {
      gs[0] = (uint64_t)a.value_row_stride * 2u;
      gs[2] = (uint64_t)a.value_head_stride * 2u;
    }
    if (cuTensorMapEncodeTiled(&tv_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, BHSD ? 4 : 3,
                               const_cast<__nv_bfloat16*>(a.v), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }
  // Mirrors the carve-up at the top of fmha_context_bf16_gen_kernel: two Q
  // tiles, the K/V ring, two BF16 O buffer tiles, the K/V ring mbarriers plus
  // 22 pipeline mbarriers, the CLC barriers and responses with 16 bytes of
  // response alignment, the TMEM address slot, the softmax statistics and 256
  // spare bytes. Adding a barrier or buffer to the kernel without updating
  // this sum overruns shared memory.
  const size_t smem =
        (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * KV_RING_SLOT_BYTES
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)
      + (2 * NUM_KV_STAGES + 22) * 8
      + (size_t)CLC_STAGES * (2 * 8 + 16) + 16
      + 8
      + (size_t)2 * STAT_REGIONS * STATS * sizeof(float)
      + 256;

  // Optional build knobs. VSA_NAMED_BAR moves the softmax-to-correction alpha
  // and l signals from mbarriers to named barriers, VSA_THROTTLE delays the
  // correction warps' acknowledgement of alpha until the O rescale finishes,
  // and VSA_USE_CLC selects cluster-launch-control scheduling.
#ifndef VSA_NAMED_BAR
#define VSA_NAMED_BAR false
#endif
#ifndef VSA_THROTTLE
#define VSA_THROTTLE false
#endif
#ifndef VSA_USE_CLC
#define VSA_USE_CLC true
#endif
  constexpr bool FULL_NAMED_BAR = VSA_NAMED_BAR, EX2_EMU = true, SPLIT_P = true,
                 SOFTMAX_THROTTLE = VSA_THROTTLE, USE_CLC = VSA_USE_CLC,
                 Q_RASTER = true, MHA = true;
  auto kfn = &fmha_context_bf16_gen_kernel<32, FULL_NAMED_BAR, EX2_EMU, SPLIT_P,
                                           SOFTMAX_THROTTLE, USE_CLC, Q_RASTER, MHA,
                                           /*RESCALE_THRESHOLD=*/8, /*BHSD=*/VSA_BHSD>;
  cudaError_t e = cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (e != cudaSuccess) return e;

  // Packed divisors (see make_magic) for the per-work-item divisions in
  // decode_workitem.
  const unsigned long long magic0 = make_magic((unsigned)(H * packed_mtiles_per_seq));
  const unsigned long long magic1 = make_magic((unsigned)H);
  const unsigned long long magic2 = make_magic((unsigned)packed_mtiles_per_seq);
  // The kernel evaluates exp2, so the softmax scale is folded with log2(e).
  const float scale_log2 = a.sm_scale * (float)M_LOG2E;

  int numSM = 0;
  e = cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
  if (e != cudaSuccess) return e;
  // With CLC, the grid holds one CTA per work item and running CTAs cancel
  // unlaunched ones to take over their work items. Without it, a persistent
  // grid of at most one CTA per SM strides over the work items.
  const int num_ctas = USE_CLC ? total_work : (total_work < numSM ? total_work : numSM);
  dim3 grid(num_ctas, 1, 1), block(N_WARPS * 32, 1, 1);

  if (USE_CLC) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cudaLaunchAttribute cfgAttr[1];
    cfgAttr[0].id = cudaLaunchAttributeClusterDimension;
    cfgAttr[0].val.clusterDim.x = 1; cfgAttr[0].val.clusterDim.y = 1;
    cfgAttr[0].val.clusterDim.z = 1;
    cfg.attrs = cfgAttr; cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kfn, tq_, tk_, tv_, to_, S, K, H, scale_log2, B,
                              num_blocks, packed_mtiles_per_seq, max_kv, magic0, magic1, magic2,
                              a.q2k_idx, a.q2k_num, a.variable_block_sizes, a.lse);
  }
  kfn<<<grid, block, smem, stream>>>(tq_, tk_, tv_, to_, S, K, H, scale_log2, B, num_blocks,
                                     packed_mtiles_per_seq, max_kv, magic0, magic1, magic2,
                                     a.q2k_idx, a.q2k_num, a.variable_block_sizes, a.lse);
  return cudaGetLastError();
}

}  // namespace VSA_NAMESPACE

// The binding translation unit selects one sparse-block configuration.
using namespace VSA_NAMESPACE;

#endif  // BLOCK_SPARSE_VSA_LAUNCH_SM100A_CUH
