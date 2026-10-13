// SPDX-License-Identifier: Apache-2.0

// SM90a block-sparse attention over 128-row query tiles and 128-row key
// blocks with a 128-wide BF16 head.
//
// One CTA evaluates one work item: the 128 query rows of one (head, query
// tile) against the key blocks its block map selects. The CTA has three
// warpgroups. Warpgroup 0 is the producer: one thread loads the query tile and
// streams the selected key and value blocks through two-stage TMA pipelines in
// the order the consumers read them. Warpgroups 1 and 2 are consumers, each
// owning 64 of the tile's query rows; per pipeline stage they compute
// S = Q K^T, an online softmax in FP32 registers, and O += P V. Both products
// take their A operand from registers: each consumer loads its query rows once.
//
// Within a consumer, the S GMMA of stage j and the P V GMMA of stage j - 1
// are issued together, so the softmax of stage j runs while the P V product
// of stage j - 1 executes; the two consumers' GMMA batches and softmaxes
// interleave on the SM without further ordering.
//
// Two shapes spare work on partial tiles. A key block whose valid rows lie
// within its first 64 shares a stage with another such block: their first 64
// rows fill the stage's two halves, so the pair costs one block's products.
// A query tile whose valid rows lie within the first consumer's 64 rows runs
// on that consumer alone; the second consumer stores zeros for the padding
// rows.
//
// A key block contributes only its first variable_block_sizes[block] rows:
// scores of the remaining columns are -inf before the softmax, so any finite
// payload past the valid size never contributes, whatever the softmax scale.
// A query tile whose count is zero, or whose selected blocks hold no valid
// key, produces zero output. Every query row is computed, padded rows
// included, except the rows of a single-consumer tile past its first 64,
// which are zero. Block indices and counts are device data read without range
// checks: counts must not exceed the selected width and indices must address
// existing key blocks.

#ifndef UNISERVE_VSA_BLOCK_SPARSE_SM90A_CUH
#define UNISERVE_VSA_BLOCK_SPARSE_SM90A_CUH

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

namespace uniserve::vsa_sm90 {

using namespace cute;
using cutlass::NumThreadsPerWarpGroup;

using Element = cutlass::bfloat16_t;

// Rows of a query tile, rows of a key block, and channels of a head.
inline constexpr int kBlockM = 128;
inline constexpr int kBlockN = 128;
inline constexpr int kHeadDim = 128;
// Rows of half a key block, and the channels one 128-byte swizzle row holds.
inline constexpr int kHalfN = kBlockN / 2;
inline constexpr int kChunk = 64;
// Key and value pipeline depth: two stages of each (128 KiB) hide the block
// loads; with the 32 KiB query tile the CTA occupies one SM.
inline constexpr int kStages = 2;
inline constexpr int kConsumerWarpgroups = 2;
inline constexpr int kConsumerThreads = kConsumerWarpgroups * NumThreadsPerWarpGroup;
inline constexpr int kThreads = (kConsumerWarpgroups + 1) * NumThreadsPerWarpGroup;
// Register split after reconfiguration: 128 * 24 + 256 * 240 <= 65536.
inline constexpr int kProducerRegisters = 24;
inline constexpr int kConsumerRegisters = 240;

inline constexpr uint32_t kTileBytes = kBlockN * kHeadDim * sizeof(Element);
static_assert(kBlockM * kHeadDim * sizeof(Element) == kTileBytes);

// The consumers' epilogue rendezvous, a named barrier numbered after the IDs
// CUTLASS reserves.
inline constexpr uint32_t kEpilogueBarrier = 0;

// Q and the epilogue's O: (rows, channels), channels contiguous.
using SmemLayoutAtomRow = decltype(GMMA::Layout_K_SW128_Atom<Element>{});
using SmemLayoutQ =
    decltype(tile_to_shape(SmemLayoutAtomRow{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
// K stages: (key rows, channels, stage), channels contiguous (K-major B of
// S = Q K^T). Each 64-channel chunk holds its 128 rows contiguously, so half
// a block's rows of one chunk are one contiguous 8 KiB box.
using SmemLayoutK = decltype(tile_to_shape(
    SmemLayoutAtomRow{}, Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));
// V stages viewed as V^T: (channels, key rows, stage), channels contiguous
// (MN-major B of O = P V). Keys are tiled first, so each 64-channel chunk
// holds its 128 keys contiguously, as for K.
using SmemLayoutVt = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<Element>{}, Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages>>{},
    Step<_2, _1, _3>{}));
// One box of half a block: 64 rows (keys) of one 64-channel chunk.
using SmemLayoutKBox = decltype(tile_to_shape(SmemLayoutAtomRow{}, Shape<Int<kHalfN>, Int<kChunk>>{}));
using SmemLayoutVtBox =
    decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{}, Shape<Int<kChunk>, Int<kHalfN>>{}));
// Element offsets inside a K or V^T stage: a chunk spans 128 rows, half a
// block 64.
inline constexpr int kStageElements = kBlockN * kHeadDim;
inline constexpr int kChunkElements = kBlockN * kChunk;
inline constexpr int kBoxElements = kHalfN * kChunk;

// Each GMMA atom covers 64 query rows; the two consumers tile the 128 rows.
using TiledMmaQK = decltype(make_tiled_mma(
    GMMA::rs_op_selector<Element, Element, float, Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>,
                         GMMA::Major::K, GMMA::Major::K>(),
    Layout<Shape<Int<kConsumerWarpgroups>, _1, _1>>{}));
using TiledMmaPV = decltype(make_tiled_mma(
    GMMA::rs_op_selector<Element, Element, float, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>,
                         GMMA::Major::K, GMMA::Major::MN>(),
    Layout<Shape<Int<kConsumerWarpgroups>, _1, _1>>{}));

// Global views: Q, K and O as (rows, channels, heads) and V as (channels,
// rows, heads), each with its own row and head strides in elements.
using ShapeRows = Shape<int32_t, int32_t, int32_t>;
using StrideRows = Stride<int64_t, _1, int64_t>;
using StrideColumns = Stride<_1, int64_t, int64_t>;

template <class T, class Stride>
using GmemTensor = decltype(make_tensor(make_gmem_ptr(static_cast<T*>(nullptr)), ShapeRows{}, Stride{}));

using TmaQ = decltype(make_tma_copy(SM90_TMA_LOAD{}, GmemTensor<Element const, StrideRows>{},
                                    SmemLayoutQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}, _1{}));
using TmaK = decltype(make_tma_copy(SM90_TMA_LOAD{}, GmemTensor<Element const, StrideRows>{},
                                    take<0, 2>(SmemLayoutK{}), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                    _1{}));
using TmaV = decltype(make_tma_copy(SM90_TMA_LOAD{}, GmemTensor<Element const, StrideColumns>{},
                                    take<0, 2>(SmemLayoutVt{}), Shape<Int<kHeadDim>, Int<kBlockN>>{},
                                    _1{}));
using TmaKBox = decltype(make_tma_copy(SM90_TMA_LOAD{}, GmemTensor<Element const, StrideRows>{},
                                       SmemLayoutKBox{}, Shape<Int<kHalfN>, Int<kChunk>>{}, _1{}));
using TmaVBox = decltype(make_tma_copy(SM90_TMA_LOAD{}, GmemTensor<Element const, StrideColumns>{},
                                       SmemLayoutVtBox{}, Shape<Int<kChunk>, Int<kHalfN>>{}, _1{}));
using TmaO = decltype(make_tma_copy(SM90_TMA_STORE{}, GmemTensor<Element, StrideRows>{},
                                    SmemLayoutQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}, _1{}));

using PipelineKV = cutlass::PipelineTmaAsync<kStages>;

struct SharedStorage {
  // The query tile; each consumer stages its O rows here once it holds its
  // query rows in registers (the consumers' rows are disjoint).
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 1024> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 1024> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, 1024> v;
  alignas(16) cutlass::arch::ClusterTransactionBarrier q_full;
  alignas(16) typename PipelineKV::SharedStorage k_pipeline;
  alignas(16) typename PipelineKV::SharedStorage v_pipeline;
};

struct Params {
  TmaQ tma_q;
  TmaK tma_k;
  TmaV tma_v;
  TmaKBox tma_k_box;
  TmaVBox tma_v_box;
  TmaO tma_o;
  int32_t query_rows;
  int32_t key_rows;
  int32_t heads;
  int32_t query_tiles;
  // Selected key blocks [heads, query_tiles, width] and their counts
  // [heads, query_tiles]; valid rows of each key block [key_rows / 128].
  int32_t const* indices;
  int32_t const* counts;
  int32_t const* valid_rows;
  int32_t width;
  // Valid rows of each query tile [query_tiles], or null when every query
  // row is valid; rows past them are padding.
  int32_t const* query_valid_rows;
  // Softmax scale folded with log2(e): probabilities are exp2(s * scale_log2).
  float scale_log2;
};

// --------------------------------------------------------------------------
// Stage schedule.
// --------------------------------------------------------------------------

// One pipeline stage: a whole key block, or the first 64 rows of two blocks.
// ``lower`` and ``upper`` are the valid keys among the stage's rows 0-63 and
// 64-127.
struct Stage {
  int first;
  int second;  // -1 for a whole block
  int lower;
  int upper;
};

// Walks a query tile's selected key blocks in the order the kernel evaluates
// them. A block with more than 64 valid rows fills a stage alone; blocks with
// at most 64 pair up in selection order, and a block left without a partner
// fills a stage alone once the selection is exhausted. The producer and every
// consumer thread walk the same selection and so agree on every stage.
struct StageWalk {
  int32_t const* selected;
  int32_t const* valid_rows;
  int count;
  int position = 0;
  int pending = -1;
  int pending_valid = 0;

  CUTLASS_DEVICE bool has_next() const { return position < count || pending >= 0; }

  // Returns the next stage; requires has_next().
  CUTLASS_DEVICE Stage next() {
    while (position < count) {
      int const block = __ldg(selected + position++);
      int const valid = __ldg(valid_rows + block);
      if (valid > kHalfN) {
        return {block, -1, kHalfN, valid - kHalfN};
      }
      if (pending < 0) {
        pending = block;
        pending_valid = valid;
        continue;
      }
      Stage const pair{pending, block, pending_valid, valid};
      pending = -1;
      return pair;
    }
    Stage const single{pending, -1, pending_valid, 0};
    pending = -1;
    return single;
  }
};

// --------------------------------------------------------------------------
// Register-fragment utilities.
// --------------------------------------------------------------------------

// Issue one GMMA batch over the K mode of the fragments; ``zero_init`` clears
// the accumulator first. The batch is committed but not awaited.
template <bool zero_init, class TiledMma, class TensorA, class TensorB, class TensorC>
CUTLASS_DEVICE void issue_gemm(TiledMma& tiled_mma, TensorA const& tA, TensorB const& tB, TensorC& tC) {
  warpgroup_fence_operand(const_cast<TensorA&>(tA));
  warpgroup_fence_operand(tC);
  warpgroup_arrive();
  if constexpr (zero_init) {
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  }
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < size<2>(tA); ++k) {
    cute::gemm(tiled_mma, tA(_, _, k), tB(_, _, k), tC);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_commit_batch();
  warpgroup_fence_operand(tC);
  warpgroup_fence_operand(const_cast<TensorA&>(tA));
}

// View a GMMA accumulator ((2, 2, N / 8), MMA_M, MMA_N) as (rows, columns):
// ((2, MMA_M), (2, N / 8, MMA_N)). A thread holds two rows, eight apart.
template <class Layout>
CUTLASS_DEVICE auto rows_columns(Layout layout) {
  static_assert(decltype(rank<0>(layout))::value == 3);
  static_assert(decltype(size<0, 0>(layout))::value == 2);
  static_assert(decltype(size<0, 1>(layout))::value == 2);
  return make_layout(make_layout(get<0, 1>(layout), get<1>(layout)),
                     make_layout(get<0, 0>(layout), get<0, 2>(layout), get<2>(layout)));
}

// View the S accumulator layout as the register A operand of the P V GMMA:
// ((2, 2, N / 8), MMA_M, MMA_N) -> ((2, 2, 2), MMA_M, N / 16). Both views
// address the same registers in the same order.
template <class Layout>
CUTLASS_DEVICE auto accumulator_as_a(Layout layout) {
  auto columns = logical_divide(get<0, 2>(layout), Tile<_2>{});
  return make_layout(make_layout(get<0, 0>(layout), get<0, 1>(layout), get<0, 0>(columns)),
                     get<1>(layout), coalesce(make_layout(get<0, 1>(columns), get<2>(layout))));
}

CUTLASS_DEVICE float quad_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1));
  return fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2));
}

CUTLASS_DEVICE float quad_sum(float value) {
  value += __shfl_xor_sync(0xffffffff, value, 1);
  return value + __shfl_xor_sync(0xffffffff, value, 2);
}

// Online softmax over the two query rows a consumer thread holds. The row
// maxima are of raw scores; ``row_sum`` holds this thread's partial sums of
// probabilities, which ``finalize`` reduces across the four threads of a row.
struct OnlineSoftmax {
  float row_max[2];
  float row_sum[2];

  CUTLASS_DEVICE OnlineSoftmax() {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < 2; ++r) {
      row_max[r] = -INFINITY;
      row_sum[r] = 0.f;
    }
  }

  // Replace the scores with exp2(s * scale_log2 - max * scale_log2) under
  // the updated row maxima and return each row's factor for rescaling the
  // output accumulated so far. A row that has seen only -inf scores keeps a
  // -inf maximum; its probabilities, sum and factor are zero.
  template <class Scores>
  CUTLASS_DEVICE void update(Scores& scores, float scale_log2, float (&rescale)[2]) {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < size<0>(scores); ++r) {
      float current = row_max[r];
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < size<1>(scores); ++c) {
        current = fmaxf(current, scores(r, c));
      }
      current = quad_max(current);

      // exp2 of an exact -inf difference is zero, which is the factor a row
      // with no earlier mass needs; a row still at -inf subtracts zero.
      float const safe = current == -INFINITY ? 0.f : current;
      rescale[r] = exp2f((row_max[r] - safe) * scale_log2);
      row_max[r] = current;

      float const offset = safe * scale_log2;
      float sum = 0.f;
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < size<1>(scores); ++c) {
        scores(r, c) = exp2f(scores(r, c) * scale_log2 - offset);
        sum += scores(r, c);
      }
      row_sum[r] = row_sum[r] * rescale[r] + sum;
    }
  }

  // Return each row's reciprocal normalizer, zero for a row without mass.
  CUTLASS_DEVICE void finalize(float (&reciprocal)[2]) {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < 2; ++r) {
      float const total = quad_sum(row_sum[r]);
      reciprocal[r] = total > 0.f ? 1.f / total : 0.f;
    }
  }
};

template <class Accumulator>
CUTLASS_DEVICE void scale_rows(Accumulator& accumulator, float const (&factor)[2]) {
  Tensor rows = make_tensor(accumulator.data(), rows_columns(accumulator.layout()));
  CUTLASS_PRAGMA_UNROLL
  for (int r = 0; r < size<0>(rows); ++r) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < size<1>(rows); ++c) {
      rows(r, c) *= factor[r];
    }
  }
}

// Convert the FP32 probabilities to the BF16 register operand of P V.
template <class Scores, class Probabilities>
CUTLASS_DEVICE void store_probabilities(Scores const& scores, Probabilities& probabilities) {
  constexpr int count = decltype(size(scores))::value;
  static_assert(count == decltype(size(probabilities))::value);
  cutlass::NumericArrayConverter<Element, float, count> convert;
  *reinterpret_cast<cutlass::Array<Element, count>*>(probabilities.data()) =
      convert(*reinterpret_cast<cutlass::Array<float, count> const*>(scores.data()));
}

// --------------------------------------------------------------------------
// Kernel.
// --------------------------------------------------------------------------

// Pipeline roles: warpgroup 0 produces, with thread 0 as the one thread that
// arms the full barriers; every thread of the active consumers releases each
// stage.
CUTLASS_DEVICE PipelineKV::Params pipeline_params(int warpgroup, int consumer_threads) {
  PipelineKV::Params params;
  params.transaction_bytes = kTileBytes;
  params.role = warpgroup == 0 ? PipelineKV::ThreadCategory::Producer : PipelineKV::ThreadCategory::Consumer;
  params.is_leader = threadIdx.x == 0;
  params.num_consumers = consumer_threads;
  return params;
}

__global__ void __launch_bounds__(kThreads, 1)
block_sparse_attention_kernel(CUTE_GRID_CONSTANT Params const params) {
  extern __shared__ __align__(1024) char shared_bytes[];
  auto& shared = *reinterpret_cast<SharedStorage*>(shared_bytes);

  int const warpgroup = cutlass::canonical_warp_group_idx();

  // Work item: heads outermost, so the CTAs resident at once read one head's
  // keys and values and share them in L2.
  int const head = blockIdx.x / params.query_tiles;
  int const tile = blockIdx.x - head * params.query_tiles;
  int64_t const map_row = int64_t(head) * params.query_tiles + tile;
  int32_t const* selected = params.indices + map_row * params.width;
  int const count = __ldg(params.counts + map_row);
  // Whether both consumers compute: a tile whose valid rows fit the first
  // consumer's rows runs on it alone, and the stages expect its threads only.
  int const query_valid = params.query_valid_rows == nullptr ? kBlockM : __ldg(params.query_valid_rows + tile);
  bool const paired = query_valid > kBlockM / kConsumerWarpgroups;
  int const consumer_threads = paired ? kConsumerThreads : NumThreadsPerWarpGroup;

  // Barrier initialization by one warp, visible to the CTA before use.
  if (threadIdx.x == 0) {
    shared.q_full.init(1);
  }
  PipelineKV k_pipeline(shared.k_pipeline, pipeline_params(warpgroup, consumer_threads), Shape<_1, _1, _1>{});
  PipelineKV v_pipeline(shared.v_pipeline, pipeline_params(warpgroup, consumer_threads), Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sQ = make_tensor(make_smem_ptr(shared.q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared.k.data()), SmemLayoutK{});
  Tensor sVt = make_tensor(make_smem_ptr(shared.v.data()), SmemLayoutVt{});

  if (warpgroup == 0) {
    // ---------------------------------------------------------------- producer
    cutlass::arch::warpgroup_reg_dealloc<kProducerRegisters>();
    // Thread 0 is the pipelines' leader, the one thread that issues loads.
    if (threadIdx.x != 0 || count == 0) {
      return;
    }
    // A zero count loads nothing: the consumers stage zeros in the query
    // tile's storage, which a query load would race.
    Tensor mQ = params.tma_q.get_tma_tensor(make_shape(params.query_rows, int32_t(kHeadDim), params.heads));
    Tensor mK = params.tma_k.get_tma_tensor(make_shape(params.key_rows, int32_t(kHeadDim), params.heads));
    Tensor mVt = params.tma_v.get_tma_tensor(make_shape(int32_t(kHeadDim), params.key_rows, params.heads));
    Tensor gQ = local_tile(mQ(_, _, head), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(tile, _0{}));
    Tensor gK = local_tile(mK(_, _, head), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));
    Tensor gVt = local_tile(mVt(_, _, head), Shape<Int<kHeadDim>, Int<kBlockN>>{}, make_coord(_0{}, _));
    // Boxes of half a block: (rows, channels, row half-block, channel chunk).
    Tensor gKBox = local_tile(mK(_, _, head), Shape<Int<kHalfN>, Int<kChunk>>{}, make_coord(_, _));
    Tensor gVtBox = local_tile(mVt(_, _, head), Shape<Int<kChunk>, Int<kHalfN>>{}, make_coord(_, _));

    auto tma_q = params.tma_q.get_slice(_0{});
    auto tma_k = params.tma_k.get_slice(_0{});
    auto tma_v = params.tma_v.get_slice(_0{});
    auto tma_k_box = params.tma_k_box.get_slice(_0{});
    auto tma_v_box = params.tma_v_box.get_slice(_0{});
    Tensor tQgQ = tma_q.partition_S(gQ);
    Tensor tQsQ = tma_q.partition_D(sQ);
    Tensor tKgK = tma_k.partition_S(gK);   // (TMA, TMA_N, TMA_K, blocks)
    Tensor tKsK = tma_k.partition_D(sK);   // (TMA, TMA_N, TMA_K, stages)
    Tensor tVgV = tma_v.partition_S(gVt);  // (TMA, TMA_K, TMA_N, blocks)
    Tensor tVsV = tma_v.partition_D(sVt);  // (TMA, TMA_K, TMA_N, stages)
    Tensor tKgKBox = tma_k_box.partition_S(gKBox);   // (TMA, 1, 1, half-blocks, chunks)
    Tensor tVgVBox = tma_v_box.partition_S(gVtBox);  // (TMA, 1, 1, chunks, half-blocks)

    // Queries are read by one CTA; key blocks are shared by the CTAs of the
    // head and stay in L2.
    constexpr auto evict_first = TMA::CacheHintSm90::EVICT_FIRST;
    constexpr auto evict_last = TMA::CacheHintSm90::EVICT_LAST;
    shared.q_full.arrive_and_expect_tx(kTileBytes);
    copy(params.tma_q.with(reinterpret_cast<uint64_t&>(shared.q_full), 0, evict_first), tQgQ, tQsQ);

    // Load a stage: a whole block in two boxes, or the first halves of two
    // blocks in four, each the same 32 KiB the stage expects.
    auto load_k = [&](auto& write, Stage const& stage) {
      k_pipeline.producer_acquire(write);
      auto* barrier = k_pipeline.producer_get_barrier(write);
      if (stage.second < 0) {
        copy(params.tma_k.with(*barrier, 0, evict_last), tKgK(_, _, _, stage.first),
             tKsK(_, _, _, write.index()));
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int half = 0; half < 2; ++half) {
          int const block = half == 0 ? stage.first : stage.second;
          CUTLASS_PRAGMA_UNROLL
          for (int chunk = 0; chunk < kHeadDim / kChunk; ++chunk) {
            Tensor box = make_tensor(
                make_smem_ptr(shared.k.data() + write.index() * kStageElements + chunk * kChunkElements +
                              half * kBoxElements),
                SmemLayoutKBox{});
            copy(params.tma_k_box.with(*barrier, 0, evict_last), tKgKBox(_, _, _, 2 * block, chunk),
                 tma_k_box.partition_D(box));
          }
        }
      }
      ++write;
    };
    auto load_v = [&](auto& write, Stage const& stage) {
      v_pipeline.producer_acquire(write);
      auto* barrier = v_pipeline.producer_get_barrier(write);
      if (stage.second < 0) {
        copy(params.tma_v.with(*barrier, 0, evict_last), tVgV(_, _, _, stage.first),
             tVsV(_, _, _, write.index()));
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int half = 0; half < 2; ++half) {
          int const block = half == 0 ? stage.first : stage.second;
          CUTLASS_PRAGMA_UNROLL
          for (int chunk = 0; chunk < kHeadDim / kChunk; ++chunk) {
            Tensor box = make_tensor(
                make_smem_ptr(shared.v.data() + write.index() * kStageElements + chunk * kChunkElements +
                              half * kBoxElements),
                SmemLayoutVtBox{});
            copy(params.tma_v_box.with(*barrier, 0, evict_last), tVgVBox(_, _, _, chunk, 2 * block),
                 tma_v_box.partition_D(box));
          }
        }
      }
      ++write;
    };

    // Consumers read K of stage j before V of stage j - 1, so the loads
    // follow that order: K 0, then (K j, V j - 1) for each later stage, then
    // the last V.
    StageWalk walk{selected, params.valid_rows, count};
    auto k_write = cutlass::make_producer_start_state<PipelineKV>();
    auto v_write = k_write;
    Stage previous = walk.next();
    load_k(k_write, previous);
    while (walk.has_next()) {
      Stage const stage = walk.next();
      load_k(k_write, stage);
      load_v(v_write, previous);
      previous = stage;
    }
    load_v(v_write, previous);
    return;
  }

  // ------------------------------------------------------------------ consumers
  cutlass::arch::warpgroup_reg_alloc<kConsumerRegisters>();
  int const consumer = warpgroup - 1;
  int const thread = threadIdx.x - NumThreadsPerWarpGroup;

  TiledMmaQK mma_qk;
  TiledMmaPV mma_pv;
  auto thread_qk = mma_qk.get_thread_slice(thread);
  auto thread_pv = mma_pv.get_thread_slice(thread);
  Tensor tSrQ = thread_qk.partition_fragment_A(sQ);
  Tensor tSrK = thread_qk.partition_fragment_B(sK);
  Tensor tOrV = thread_pv.partition_fragment_B(sVt);
  Tensor tOrO = partition_fragment_C(mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  Tensor tSrS = partition_fragment_C(mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
  Tensor tOrP = make_tensor<Element>(accumulator_as_a(tSrS.layout()));
  // Key column of each score this thread holds.
  Tensor tScS = thread_qk.partition_C(make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{}));
  Tensor scores = make_tensor(tSrS.data(), rows_columns(tSrS.layout()));
  Tensor columns = make_tensor(tScS.data(), rows_columns(tScS.layout()));

  // Mask the score columns past the valid keys of each half of the stage.
  auto mask_columns = [&](Stage const& stage) {
    if (stage.lower < kHalfN || stage.upper < kHalfN) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < size<1>(scores); ++c) {
        int const column = get<1>(columns(0, c));
        bool const masked = column < kHalfN ? column >= stage.lower : column - kHalfN >= stage.upper;
        if (masked) {
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < size<0>(scores); ++r) {
            scores(r, c) = -INFINITY;
          }
        }
      }
    }
  };

  clear(tOrO);
  if (count > 0 && !paired && consumer == 1) {
    // The idle consumer's zero rows overwrite the query tile's storage, so
    // they are staged after the query load lands.
    shared.q_full.wait(0);
  } else if (count > 0) {
    OnlineSoftmax softmax;
    float rescale[2];
    StageWalk walk{selected, params.valid_rows, count};

    // Stage 0: S only.
    PipelineKV::PipelineState k_read;
    PipelineKV::PipelineState v_read;
    Stage stage = walk.next();
    shared.q_full.wait(0);
    // Load this consumer's query rows into the register A operand once.
    auto load_q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, mma_qk);
    auto thread_load_q = load_q.get_thread_slice(thread);
    copy(load_q, thread_load_q.partition_S(sQ), thread_load_q.retile_D(tSrQ));
    k_pipeline.consumer_wait(k_read);
    issue_gemm<true>(mma_qk, tSrQ, tSrK(_, _, _, k_read.index()), tSrS);
    warpgroup_wait<0>();
    k_pipeline.consumer_release(k_read);
    mask_columns(stage);
    softmax.update(scores, params.scale_log2, rescale);
    store_probabilities(tSrS, tOrP);

    // Stage j: S of j with P V of j - 1, then the softmax of j while P V
    // executes.
    while (walk.has_next()) {
      stage = walk.next();
      ++k_read;
      k_pipeline.consumer_wait(k_read);
      issue_gemm<true>(mma_qk, tSrQ, tSrK(_, _, _, k_read.index()), tSrS);
      scale_rows(tOrO, rescale);
      v_pipeline.consumer_wait(v_read);
      issue_gemm<false>(mma_pv, tOrP, tOrV(_, _, _, v_read.index()), tOrO);
      warpgroup_wait<1>();
      k_pipeline.consumer_release(k_read);
      mask_columns(stage);
      softmax.update(scores, params.scale_log2, rescale);
      warpgroup_wait<0>();
      v_pipeline.consumer_release(v_read);
      ++v_read;
      store_probabilities(tSrS, tOrP);
    }

    // P V of the last stage, then normalization.
    scale_rows(tOrO, rescale);
    v_pipeline.consumer_wait(v_read);
    issue_gemm<false>(mma_pv, tOrP, tOrV(_, _, _, v_read.index()), tOrO);
    float reciprocal[2];
    softmax.finalize(reciprocal);
    warpgroup_wait<0>();
    v_pipeline.consumer_release(v_read);
    scale_rows(tOrO, reciprocal);
  }

  // ------------------------------------------------------------------ epilogue
  // Stage BF16 rows in the query tile's storage, then one thread stores the
  // tile. Each consumer writes only its own rows, which it no longer reads.
  Tensor tOrO_bf16 = make_tensor<Element>(tOrO.layout());
  {
    constexpr int values = decltype(size(tOrO))::value;
    cutlass::NumericArrayConverter<Element, float, values> convert;
    *reinterpret_cast<cutlass::Array<Element, values>*>(tOrO_bf16.data()) =
        convert(*reinterpret_cast<cutlass::Array<float, values> const*>(tOrO.data()));
  }
  auto store_o = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, mma_pv);
  auto thread_store = store_o.get_thread_slice(thread);
  Tensor sO = make_tensor(make_smem_ptr(shared.q.data()), SmemLayoutQ{});
  copy(store_o, thread_store.retile_S(tOrO_bf16), thread_store.partition_D(sO));
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::NamedBarrier::sync(kConsumerThreads, kEpilogueBarrier);

  if (thread == 0) {
    Tensor mO = params.tma_o.get_tma_tensor(make_shape(params.query_rows, int32_t(kHeadDim), params.heads));
    Tensor gO = local_tile(mO(_, _, head), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(tile, _0{}));
    auto tma_o = params.tma_o.get_slice(_0{});
    copy(params.tma_o, tma_o.partition_S(sO), tma_o.partition_D(gO));
    tma_store_arrive();
    tma_store_wait<0>();
  }
}

// --------------------------------------------------------------------------
// Host launch.
// --------------------------------------------------------------------------

// Device pointers and geometry of one call. Strides are in elements; channels
// are contiguous.
struct Arguments {
  Element const* q;
  Element const* k;
  Element const* v;
  Element* o;
  int64_t q_row_stride, q_head_stride;
  int64_t k_row_stride, k_head_stride;
  int64_t v_row_stride, v_head_stride;
  int64_t o_row_stride, o_head_stride;
  int32_t query_rows, key_rows, heads;
  int32_t const* indices;
  int32_t const* counts;
  int32_t const* valid_rows;
  int32_t width;
  int32_t const* query_valid_rows;
  float scale;
};

inline size_t shared_storage_bytes() { return sizeof(SharedStorage); }

inline cudaError_t launch(Arguments const& a, cudaStream_t stream) {
  auto rows = [](auto const* pointer, int32_t extent, int32_t heads, int64_t row_stride, int64_t head_stride) {
    return make_tensor(make_gmem_ptr(pointer), make_shape(extent, int32_t(kHeadDim), heads),
                       make_stride(row_stride, _1{}, head_stride));
  };
  Tensor q = rows(a.q, a.query_rows, a.heads, a.q_row_stride, a.q_head_stride);
  Tensor k = rows(a.k, a.key_rows, a.heads, a.k_row_stride, a.k_head_stride);
  Tensor o = rows(a.o, a.query_rows, a.heads, a.o_row_stride, a.o_head_stride);
  Tensor vt = make_tensor(make_gmem_ptr(a.v), make_shape(int32_t(kHeadDim), a.key_rows, a.heads),
                          make_stride(_1{}, a.v_row_stride, a.v_head_stride));

  Params params{
      make_tma_copy(SM90_TMA_LOAD{}, q, SmemLayoutQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}, _1{}),
      make_tma_copy(SM90_TMA_LOAD{}, k, take<0, 2>(SmemLayoutK{}), Shape<Int<kBlockN>, Int<kHeadDim>>{}, _1{}),
      make_tma_copy(SM90_TMA_LOAD{}, vt, take<0, 2>(SmemLayoutVt{}), Shape<Int<kHeadDim>, Int<kBlockN>>{}, _1{}),
      make_tma_copy(SM90_TMA_LOAD{}, k, SmemLayoutKBox{}, Shape<Int<kHalfN>, Int<kChunk>>{}, _1{}),
      make_tma_copy(SM90_TMA_LOAD{}, vt, SmemLayoutVtBox{}, Shape<Int<kChunk>, Int<kHalfN>>{}, _1{}),
      make_tma_copy(SM90_TMA_STORE{}, o, SmemLayoutQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}, _1{}),
      a.query_rows,
      a.key_rows,
      a.heads,
      a.query_rows / kBlockM,
      a.indices,
      a.counts,
      a.valid_rows,
      a.width,
      a.query_valid_rows,
      static_cast<float>(a.scale * M_LOG2E),
  };

  size_t const smem = shared_storage_bytes();
  cudaError_t error = cudaFuncSetAttribute(block_sparse_attention_kernel,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  if (error != cudaSuccess) {
    return error;
  }
  dim3 const grid(unsigned(a.heads) * unsigned(params.query_tiles));
  block_sparse_attention_kernel<<<grid, kThreads, smem, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace uniserve::vsa_sm90

#endif  // UNISERVE_VSA_BLOCK_SPARSE_SM90A_CUH
