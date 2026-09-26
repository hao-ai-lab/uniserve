// SPDX-License-Identifier: Apache-2.0

// Block-diffusion canvas sampling on SM100: one vocabulary sweep per canvas
// position, the per-row acceptance, re-noise and stop decisions, and the
// block-start canvas. uniserve_kernels.diffusion.canvas documents the
// contract; uniserve.diffusion.canvas defines the portable formulas these
// kernels follow. Every launch reads its per-row values (seed, block, step)
// from device memory, keeps static shapes and runs on the current stream, so
// a CUDA graph can capture it.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>
#include <vector>

namespace cg = cooperative_groups;

namespace {

// Philox4x32-10 multipliers and Weyl key increments (Salmon et al., SC'11).
constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;

// Random streams of one denoising block (uniserve.diffusion.tokens).
constexpr uint32_t kStreamInitial = 0;
constexpr uint32_t kStreamSample = 1;
constexpr uint32_t kStreamRenoise = 2;

// The largest perturbation -log(-log(u)) of any contract uniform is reached
// at u = 1 - 2^-24 and equals 16.63553. A token can beat a perturbed score B
// only if its processed logit is at least B - kGumbelReach; the 0.0045
// margin over 16.63553 exceeds the FP32 rounding of the threshold plus the
// 1.5 ulp by which a scaled logit x RN(1 / t) may undershoot x / t, for any
// |x / t| < 1024.
constexpr float kGumbelReach = 16.64f;

// A thread re-references its running exponential sums once a logit exceeds
// the reference by this much, so exp(q - reference) <= e^16 never overflows
// while re-referencing stays rare.
constexpr float kRebase = 16.0f;

constexpr float kLog2e = 1.4426950408889634f;
constexpr int kMaxEos = 8;
constexpr int kWarp = 32;

// A sweep CTA runs eight arithmetic warps and one loading warp. The loading
// warp streams the CTA's slice through kStages chunks of kChunkBytes; each
// arithmetic thread takes kGroups float4 groups of a chunk.
constexpr int kArithmeticThreads = 256;
constexpr int kSweepThreads = kArithmeticThreads + 32;
constexpr int kGroups = 4;
constexpr int kStages = 3;
constexpr int kChunkVectors = kArithmeticThreads * kGroups;
constexpr int kChunkFloats = kChunkVectors * 4;
constexpr int kChunkBytes = kChunkFloats * 4;

__device__ __forceinline__ uint4 philox(
    uint64_t seed, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) {
  uint32_t k0 = static_cast<uint32_t>(seed);
  uint32_t k1 = static_cast<uint32_t>(seed >> 32);
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    if (round) {
      k0 += kPhiloxW0;
      k1 += kPhiloxW1;
    }
    const uint32_t hi0 = __umulhi(c0, kPhiloxM0);
    const uint32_t lo0 = c0 * kPhiloxM0;
    const uint32_t hi1 = __umulhi(c2, kPhiloxM1);
    const uint32_t lo1 = c2 * kPhiloxM1;
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
  }
  return make_uint4(c0, c1, c2, c3);
}

// Counter word 3: the block index above two stream bits, modulo 2^32.
__device__ __forceinline__ uint32_t block_word(int64_t block, uint32_t stream) {
  return static_cast<uint32_t>((static_cast<uint64_t>(block) << 2) | stream);
}

// One of 2^23 cell midpoints (2k + 1) / 2^24, exact in FP32 and inside (0, 1).
__device__ __forceinline__ float uniform(uint32_t bits) {
  return __uint2float_rn(((bits >> 9) << 1) | 1u) * 0x1p-24f;
}

// Multiply-shift reduction of 32 random bits to [0, vocab).
__device__ __forceinline__ int64_t random_token(uint32_t bits, int64_t vocab) {
  return static_cast<int64_t>(
      (static_cast<uint64_t>(bits) * static_cast<uint64_t>(vocab)) >> 32);
}

// Gumbel-perturbed score in the operation order of the portable formula.
// logf is the accurate (1 ulp) libdevice logarithm: the file is compiled
// without fast math.
__device__ __forceinline__ float gumbel_score(float logit, uint32_t bits) {
  return logit - logf(-logf(uniform(bits)));
}

__device__ __forceinline__ float exp2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Transformers' linear temperature at `step` of `steps`, in FP32:
// t_min + (t_max - t_min) * ((steps - step) / steps). The explicit
// round-to-nearest intrinsics keep the three roundings uncontracted.
__device__ __forceinline__ float temperature(
    int64_t step, int steps, float t_min, float t_delta) {
  const float remaining = static_cast<float>(steps - step);
  return __fadd_rn(
      t_min, __fmul_rn(t_delta, __fdiv_rn(remaining, static_cast<float>(steps))));
}

__device__ __forceinline__ uint32_t pack_bf16(float2 value) {
  const __nv_bfloat162 packed = __float22bfloat162_rn(value);
  return *reinterpret_cast<const uint32_t*>(&packed);
}

__device__ __forceinline__ float2 unpack_bf16(uint32_t packed) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed));
}

// ---- Cluster exchange ------------------------------------------------------

__device__ __forceinline__ uint32_t smem_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void barrier_init(uint64_t* barrier, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(smem_address(barrier)), "r"(count)
               : "memory");
}

// Makes initialized barriers visible to the cluster and the async proxy.
__device__ __forceinline__ void barrier_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// The single arrival of a barrier phase, expecting `bytes` of posted data.
__device__ __forceinline__ void expect_bytes(uint64_t* barrier, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(smem_address(barrier)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void wait_barrier(uint64_t* barrier, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred done;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra WAIT_%=;\n"
      "}\n"
      :
      : "r"(smem_address(barrier)), "r"(parity)
      : "memory");
}

__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// The shared::cluster address of `local`'s counterpart in CTA `rank`.
__device__ __forceinline__ uint32_t peer_address(const void* local, uint32_t rank) {
  uint32_t address;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(address)
               : "r"(smem_address(local)), "r"(rank));
  return address;
}

// Posts 16 bytes into a peer CTA's shared memory; the peer's barrier counts
// the bytes when they land. The sender does not wait.
__device__ __forceinline__ void push(uint32_t slot, uint4 value, uint32_t barrier) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
      :
      : "r"(slot), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "r"(barrier)
      : "memory");
}

// ---- Reductions ------------------------------------------------------------

// Softmax moments of a set of processed logits q relative to `ref`:
// mass = sum exp(q - ref) and moment = sum exp(q - ref) (q - ref) <= 0. An
// empty set has zero mass.
struct Moments {
  float ref;
  float mass;
  float moment;
};

// The largest value and, among equal values, the lowest vocabulary index.
struct Best {
  float value;
  int32_t index;
};

// Rescales moments to a reference at least as large as their own.
__device__ __forceinline__ Moments rebase(const Moments& m, float ref) {
  if (m.mass == 0.f || m.ref == ref) {
    return {ref, m.mass, m.moment};
  }
  const float shift = m.ref - ref;  // < 0
  const float factor = exp2_approx(shift * kLog2e);
  // exp(q - ref) (q - ref) = factor exp(q - m.ref) ((q - m.ref) + shift):
  // both summands are non-positive, so nothing cancels.
  return {ref, m.mass * factor, factor * (m.moment + shift * m.mass)};
}

struct MomentsOp {
  __device__ Moments operator()(const Moments& a, const Moments& b) const {
    if (a.mass == 0.f) {
      return b;
    }
    if (b.mass == 0.f) {
      return a;
    }
    const float ref = fmaxf(a.ref, b.ref);
    const Moments x = rebase(a, ref);
    const Moments y = rebase(b, ref);
    return {ref, x.mass + y.mass, x.moment + y.moment};
  }
};

struct BestOp {
  __device__ Best operator()(const Best& a, const Best& b) const {
    return (b.value > a.value || (b.value == a.value && b.index < a.index)) ? b : a;
  }
};

// A CTA's first-pass partial: its softmax moments and first maximum.
struct Stats {
  Moments moments;
  Best top;
};

struct StatsOp {
  __device__ Stats operator()(const Stats& a, const Stats& b) const {
    return {MomentsOp{}(a.moments, b.moments), BestOp{}(a.top, b.top)};
  }
};

// A CTA's second-pass partial: its best perturbed score, its largest exact
// processed logit with the first index holding it, and its share of the
// self-conditioning normalizer.
struct Tally {
  Best sample;
  Best argmax;
  float normalizer;
};

struct TallyOp {
  __device__ Tally operator()(const Tally& a, const Tally& b) const {
    return {BestOp{}(a.sample, b.sample), BestOp{}(a.argmax, b.argmax),
            a.normalizer + b.normalizer};
  }
};

__device__ __forceinline__ Moments shuffle_down(const Moments& m, int offset) {
  return {__shfl_down_sync(0xffffffffu, m.ref, offset),
          __shfl_down_sync(0xffffffffu, m.mass, offset),
          __shfl_down_sync(0xffffffffu, m.moment, offset)};
}

__device__ __forceinline__ Best shuffle_down(const Best& b, int offset) {
  return {__shfl_down_sync(0xffffffffu, b.value, offset),
          __shfl_down_sync(0xffffffffu, b.index, offset)};
}

__device__ __forceinline__ Stats shuffle_down(const Stats& s, int offset) {
  return {shuffle_down(s.moments, offset), shuffle_down(s.top, offset)};
}

__device__ __forceinline__ Tally shuffle_down(const Tally& t, int offset) {
  return {shuffle_down(t.sample, offset), shuffle_down(t.argmax, offset),
          __shfl_down_sync(0xffffffffu, t.normalizer, offset)};
}

// Reduces the first `count` lanes of a warp into lane 0 along a fixed tree.
template <typename T, typename Op>
__device__ __forceinline__ T warp_reduce(T value, int count, Op op) {
  const int lane = threadIdx.x % kWarp;
#pragma unroll
  for (int offset = kWarp / 2; offset; offset >>= 1) {
    const T other = shuffle_down(value, offset);
    if (lane + offset < count) {
      value = op(value, other);
    }
  }
  return value;
}

// Lane 0's value in every lane of the warp.
__device__ __forceinline__ uint4 shuffle_first(uint4 value) {
  return make_uint4(__shfl_sync(0xffffffffu, value.x, 0), __shfl_sync(0xffffffffu, value.y, 0),
                    __shfl_sync(0xffffffffu, value.z, 0), __shfl_sync(0xffffffffu, value.w, 0));
}

// Stats and Tally travel between CTAs as two 16-byte words each.
__device__ __forceinline__ void encode(const Stats& s, uint4& a, uint4& b) {
  a = make_uint4(__float_as_uint(s.moments.ref), __float_as_uint(s.moments.mass),
                 __float_as_uint(s.moments.moment), __float_as_uint(s.top.value));
  b = make_uint4(static_cast<uint32_t>(s.top.index), 0u, 0u, 0u);
}

__device__ __forceinline__ Stats decode_stats(uint4 a, uint4 b) {
  return {{__uint_as_float(a.x), __uint_as_float(a.y), __uint_as_float(a.z)},
          {__uint_as_float(a.w), static_cast<int32_t>(b.x)}};
}

__device__ __forceinline__ void encode(const Tally& t, uint4& a, uint4& b) {
  a = make_uint4(__float_as_uint(t.sample.value), static_cast<uint32_t>(t.sample.index),
                 __float_as_uint(t.argmax.value), static_cast<uint32_t>(t.argmax.index));
  b = make_uint4(__float_as_uint(t.normalizer), 0u, 0u, 0u);
}

__device__ __forceinline__ Tally decode_tally(uint4 a, uint4 b) {
  return {{__uint_as_float(a.x), static_cast<int32_t>(a.y)},
          {__uint_as_float(a.z), static_cast<int32_t>(a.w)},
          __uint_as_float(b.x)};
}

// ---- Vocabulary sweep ----------------------------------------------------------

constexpr int kMaxCluster = 16;

// A float4 group of raw logits that may hold the Gumbel winner or the
// maximum, with the vocabulary index of its first token.
struct Candidate {
  float4 logits;
  int32_t index;
};

// Row values that score candidates of one position.
struct Evaluation {
  uint64_t seed;
  uint32_t position;     // canvas index
  uint32_t step;
  uint32_t sample_word;  // counter word 3 of the SAMPLE stream
  float temperature;
};

// Scores a candidate group: each token takes its exact IEEE quotient
// q = x / t, which competes for the argmax, and its Gumbel score from the
// Philox word of the group.
__device__ __forceinline__ void evaluate(
    const Candidate& candidate, const Evaluation& row, Best& sample, Best& argmax) {
  const uint4 bits = philox(row.seed, static_cast<uint32_t>(candidate.index) >> 2, row.position,
                            row.step, row.sample_word);
  const float values[4] = {candidate.logits.x, candidate.logits.y, candidate.logits.z,
                           candidate.logits.w};
  const uint32_t words[4] = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float exact = __fdiv_rn(values[k], row.temperature);
    const int32_t index = candidate.index + k;
    argmax = BestOp{}(argmax, Best{exact, index});
    sample = BestOp{}(sample, Best{gumbel_score(exact, words[k]), index});
  }
}

// Positions a CTA works on at once: a position's statistics exchange runs
// while the CTA streams the next position's first pass, and a posted
// partial lands in one of kSlots rotating slots (see score_kernel).
constexpr int kSlots = 4;

// Per-CTA shared state. After its first pass over item i, every CTA posts
// its partial into stats[i % kSlots][rank] of every peer, itself included;
// after the second pass, into tallies[i % kSlots][rank] of CTA 0. Local
// barriers count the landed bytes, and each receiver reduces its slots in
// rank order along one fixed tree, so all CTAs obtain identical bits
// without reading each other's shared memory. `full` and `empty` pace the
// ring of logits chunks between the loading warp and the arithmetic warps.
struct Exchange {
  uint4 stats[kSlots][kMaxCluster][2];
  uint4 tallies[kSlots][kMaxCluster][2];
  uint64_t full[kStages];
  uint64_t empty[kStages];
  uint64_t stats_landed[kSlots];
  uint64_t tallies_landed[kSlots];
  float maximum;
  float threshold;
  Stats stats_scratch[kWarp];
  Tally tally_scratch[kWarp];
  Candidate queue[kArithmeticThreads / kWarp][2 * kWarp];
};

struct SweepParams {
  const float* logits;             // [positions, vocab]
  __nv_bfloat16* weights;          // [positions, vocab], row stride weight_stride
  int64_t weight_stride;
  float* normalizer;               // [positions]
  float* entropy;                  // [positions]
  int64_t* argmax;                 // [positions]
  int64_t* sample;                 // [positions]
  const int64_t* seed;             // [rows]
  const int64_t* block;            // [rows]
  const int64_t* step;             // [rows]
  int64_t positions;
  int64_t vocab;
  int64_t canvas;
  int clusters;                    // clusters in the grid
  int steps;
  float t_min;
  float t_delta;
};

// L2 policies: the first pass keeps its lines for the second, which
// releases them.
__device__ __forceinline__ uint64_t l2_policy_keep() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

__device__ __forceinline__ uint64_t l2_policy_release() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

// Bulk copy of `bytes` from global memory into this CTA's shared memory;
// `barrier` counts the bytes when they land.
__device__ __forceinline__ void bulk_load(
    void* destination, const void* source, uint32_t bytes, uint64_t* barrier, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1], %2, [%3], %4;"
      :
      : "r"(smem_address(destination)), "l"(source), "r"(bytes), "r"(smem_address(barrier)),
        "l"(policy)
      : "memory");
}

__device__ __forceinline__ void arrive(uint64_t* barrier) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :
               : "r"(smem_address(barrier))
               : "memory");
}

// Synchronizes the arithmetic warps only; the loading warp runs ahead.
__device__ __forceinline__ void arithmetic_sync() {
  asm volatile("bar.sync 1, %0;" : : "r"(kArithmeticThreads) : "memory");
}

// Reduces every arithmetic thread's value into thread 0: lanes, then warps,
// in a fixed order, so equal inputs reduce to equal bits on every launch.
template <typename T, typename Op>
__device__ __forceinline__ T arithmetic_reduce(T value, T* scratch, Op op) {
  constexpr int warps = kArithmeticThreads / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  value = warp_reduce(value, kWarp, op);
  if (lane == 0) {
    scratch[warp] = value;
  }
  arithmetic_sync();
  if (warp == 0) {
    value = warp_reduce(scratch[lane < warps ? lane : 0], warps, op);
  }
  return value;
}

// Waits for chunk `n` of the ring, copies this thread's kGroups float4 groups
// out of it (group u sits kArithmeticThreads float4 after group u - 1) and
// returns the chunk's stage to the loading warp.
__device__ __forceinline__ void take_chunk(
    const float4* ring, uint64_t* full, uint64_t* empty, int n, float4 (&x)[kGroups]) {
  const int stage = n % kStages;
  wait_barrier(&full[stage], (n / kStages) & 1);
  const float4* data = ring + stage * kChunkVectors + threadIdx.x;
#pragma unroll
  for (int u = 0; u < kGroups; ++u) {
    x[u] = data[u * kArithmeticThreads];
  }
  __syncwarp();
  if (threadIdx.x % kWarp == 0) {
    arrive(&empty[stage]);
  }
}

// Scaled logits q' = x RN(1 / t), pairs (2u, 2u + 1) holding group u. Each
// is within 1.5 ulp of the exact quotient x / t and ordered like it.
__device__ __forceinline__ void scale(
    const float4 (&x)[kGroups], float2 reciprocal, float2 (&q)[2 * kGroups]) {
#pragma unroll
  for (int u = 0; u < kGroups; ++u) {
    q[2 * u] = __fmul2_rn(make_float2(x[u].x, x[u].y), reciprocal);
    q[2 * u + 1] = __fmul2_rn(make_float2(x[u].z, x[u].w), reciprocal);
  }
}

__device__ __forceinline__ float largest(const float2 (&q)[2 * kGroups]) {
  float value = fmaxf(q[0].x, q[0].y);
#pragma unroll
  for (int k = 1; k < 2 * kGroups; ++k) {
    value = fmaxf(value, fmaxf(q[k].x, q[k].y));
  }
  return value;
}

// The CTA's tasks in order. A cluster works through positions c, c + G,
// c + 2G, ... (G clusters in the grid) as items 0, 1, 2, ... in the order
//   first(0), first(1), second(0), first(2), second(1), ..., second(n - 1),
// so the exchange of an item's first-pass statistics overlaps the next
// item's first pass.
struct Schedule {
  int count;    // items of this cluster
  int cluster;  // this cluster's index c
  int clusters; // G

  __device__ int64_t position(int item) const {
    return static_cast<int64_t>(cluster) + static_cast<int64_t>(item) * clusters;
  }

  __device__ int length() const { return 2 * count; }

  // Whether task t is a first pass, and of which item.
  __device__ bool first_pass(int t) const {
    return t == 0 || (t < 2 * count - 1 && t % 2 == 1);
  }

  __device__ int item(int t) const {
    if (t == 0) {
      return 0;
    }
    if (t == 2 * count - 1) {
      return count - 1;
    }
    return t % 2 == 1 ? (t + 1) / 2 : t / 2 - 1;
  }
};

// CTA 0's warp 0 reduces the tallies of `item` once every peer's post has
// landed, re-arms the slot and writes the position's argmax, sample and
// normalizer.
template <int CLUSTER>
__device__ __forceinline__ void collect(
    Exchange& exchange, int item, const SweepParams& params, int64_t position) {
  const int lane = threadIdx.x % kWarp;
  const int slot = item % kSlots;
  wait_barrier(&exchange.tallies_landed[slot], (item / kSlots) & 1);
  Tally peer{{-INFINITY, INT_MAX}, {-INFINITY, INT_MAX}, 0.f};
  if (lane < CLUSTER) {
    peer = decode_tally(exchange.tallies[slot][lane][0], exchange.tallies[slot][lane][1]);
  }
  const Tally total = warp_reduce(peer, CLUSTER, TallyOp{});
  if (lane == 0) {
    expect_bytes(&exchange.tallies_landed[slot], CLUSTER * 2 * sizeof(uint4));
    params.sample[position] = total.sample.index;
    params.argmax[position] = total.argmax.index;
    params.normalizer[position] = total.normalizer;
  }
}

// Persistent sweep: a cluster of CLUSTER CTAs scores canvas positions one
// after another; CTA `rank` owns the vocabulary slice
// [rank * slice, (rank + 1) * slice) of each. A loading warp streams every
// task's chunks through a shared-memory ring with bulk copies, running
// ahead of the eight arithmetic warps. Per position:
//
// 1. First pass (HBM; lines kept in L2) over q' = x RN(1 / t): each thread
//    tracks its first maximum and the running softmax moments. Warp 0 posts
//    the CTA's partial to every peer.
// 2. At the start of the second pass every CTA reduces the posted partials
//    to the maximum M' of q', the entropy log(mass) - moment / mass, and a
//    Gumbel threshold: B, the exact perturbed score of the token holding
//    M', bounds the winning score from below, so only tokens with
//    q' >= B - kGumbelReach can win.
// 3. Second pass (L2): those candidates take their exact IEEE quotient
//    q = x / t, draw their Philox word and score; they also contain every
//    token holding the exact maximum, so the first of them is the exact
//    argmax. Every token's self-conditioning weight bf16(exp(bf16(q') - M'))
//    is written and summed into the normalizer. Warp 0 posts the CTA's
//    partial to CTA 0, which reduces argmax, sample and normalizer after
//    its next task.
//
// Slot reuse: a peer posts item i + kSlots only after its second pass of
// item i + 2, which needs this CTA's first-pass partial of item i + 2; this
// CTA consumes item i's slot (second pass of i) before that first pass. The
// same argument bounds CTA 0's tally slots. Every CTA consumes all posts
// addressed to it before it exits.
//
// The weight rows must not overlap the logits: peers may still be reading
// their slices when a CTA writes.
template <int CLUSTER>
__global__ void __launch_bounds__(kSweepThreads, 3)
    score_kernel(const SweepParams params) {
  extern __shared__ __align__(128) float4 ring[];
  __shared__ Exchange exchange;

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_index = static_cast<int>(blockIdx.x) / CLUSTER;
  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;

  const int64_t slice = params.vocab / CLUSTER;
  const int chunks = static_cast<int>(slice / kChunkFloats);
  const int64_t slice_start = static_cast<int64_t>(rank) * slice;
  const Schedule schedule{
      static_cast<int>((params.positions - cluster_index + params.clusters - 1) / params.clusters),
      cluster_index, params.clusters};

  // Each exchange barrier takes one local arrival per use, registered with
  // the bytes it expects; a use completes once every partial has landed.
  constexpr uint32_t exchange_bytes = CLUSTER * 2 * sizeof(uint4);
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      barrier_init(&exchange.full[s], 1);
      barrier_init(&exchange.empty[s], kArithmeticThreads / kWarp);
    }
    for (int s = 0; s < kSlots; ++s) {
      barrier_init(&exchange.stats_landed[s], 1);
      barrier_init(&exchange.tallies_landed[s], 1);
    }
    barrier_init_fence();
    for (int s = 0; s < kSlots; ++s) {
      expect_bytes(&exchange.stats_landed[s], exchange_bytes);
      if (rank == 0) {
        expect_bytes(&exchange.tallies_landed[s], exchange_bytes);
      }
    }
  }
  __syncthreads();
  // Peers post into these barriers only after the cluster barrier, which
  // every CTA arrives at once its barriers exist.
  cluster_arrive();

  if (warp == kArithmeticThreads / kWarp) {
    // ---- Loading warp. ------------------------------------------------------
    if (lane == 0) {
      const uint64_t keep = l2_policy_keep();
      const uint64_t release = l2_policy_release();
      int n = 0;
      for (int t = 0; t < schedule.length(); ++t) {
        const bool first = schedule.first_pass(t);
        const float* source =
            params.logits + schedule.position(schedule.item(t)) * params.vocab + slice_start;
        for (int c = 0; c < chunks; ++c, ++n) {
          const int stage = n % kStages;
          const int round = n / kStages;
          if (round) {
            wait_barrier(&exchange.empty[stage], (round - 1) & 1);
          }
          expect_bytes(&exchange.full[stage], kChunkBytes);
          bulk_load(ring + stage * kChunkVectors, source + c * kChunkFloats, kChunkBytes,
                    &exchange.full[stage], first ? keep : release);
        }
      }
    }
    return;
  }

  // ---- Arithmetic warps. ------------------------------------------------------
  const float2 log2e2 = make_float2(kLog2e, kLog2e);
  int n = 0;  // chunks taken so far, which fixes each chunk's stage and phase
  bool posted = false;

  for (int t = 0; t < schedule.length(); ++t) {
    const int item = schedule.item(t);
    const int slot = item % kSlots;
    const int64_t position = schedule.position(item);
    const int64_t row = position / params.canvas;
    const uint32_t canvas_index = static_cast<uint32_t>(position % params.canvas);
    const int64_t step = params.step[row];
    const uint64_t seed = static_cast<uint64_t>(params.seed[row]);
    const uint32_t sample_word = block_word(params.block[row], kStreamSample);
    const float temp = temperature(step, params.steps, params.t_min, params.t_delta);
    const float reciprocal = __frcp_rn(temp);
    const float2 reciprocal2 = make_float2(reciprocal, reciprocal);

    if (schedule.first_pass(t)) {
      // ---- First pass. ------------------------------------------------------
      // Independent accumulators (A for even pairs, B for odd) shorten the
      // dependency chains; they merge before the reduction.
      float ref = -INFINITY;
      float2 mass_a = make_float2(0.f, 0.f), mass_b = make_float2(0.f, 0.f);
      float2 moment_a = make_float2(0.f, 0.f), moment_b = make_float2(0.f, 0.f);
      Best top{-INFINITY, INT_MAX};
      for (int c = 0; c < chunks; ++c) {
        float4 x[kGroups];
        take_chunk(ring, exchange.full, exchange.empty, n++, x);
        float2 q[2 * kGroups];
        scale(x, reciprocal2, q);
        const float peak = largest(q);
        if (peak > top.value) {
          // Rare: a new maximum. The first occurrence in index order wins.
          const int first = c * kChunkVectors + tid;
          int index = INT_MAX;
#pragma unroll
          for (int k = 2 * kGroups - 1; k >= 0; --k) {
            const int group = first + (k / 2) * kArithmeticThreads;
            if (q[k].y == peak) {
              index = 4 * group + 2 * (k % 2) + 1;
            }
            if (q[k].x == peak) {
              index = 4 * group + 2 * (k % 2);
            }
          }
          top = {peak, static_cast<int32_t>(slice_start + index)};
          if (peak > ref + kRebase) {
            if (ref != -INFINITY) {
              const float shift = ref - peak;
              const float factor = exp2_approx(shift * kLog2e);
              const float2 factor2 = make_float2(factor, factor);
              const float2 shift2 = make_float2(shift, shift);
              moment_a = __fmul2_rn(factor2, __ffma2_rn(shift2, mass_a, moment_a));
              moment_b = __fmul2_rn(factor2, __ffma2_rn(shift2, mass_b, moment_b));
              mass_a = __fmul2_rn(mass_a, factor2);
              mass_b = __fmul2_rn(mass_b, factor2);
            }
            ref = peak;
          }
        }
        const float2 negated_ref2 = make_float2(-ref, -ref);
#pragma unroll
        for (int k = 0; k < 2 * kGroups; ++k) {
          const float2 d = __fadd2_rn(q[k], negated_ref2);
          const float2 a = __fmul2_rn(d, log2e2);
          const float2 e = make_float2(exp2_approx(a.x), exp2_approx(a.y));
          if (k % 2 == 0) {
            mass_a = __fadd2_rn(mass_a, e);
            moment_a = __ffma2_rn(e, d, moment_a);
          } else {
            mass_b = __fadd2_rn(mass_b, e);
            moment_b = __ffma2_rn(e, d, moment_b);
          }
        }
      }
      const float2 mass = __fadd2_rn(mass_a, mass_b);
      const float2 moment = __fadd2_rn(moment_a, moment_b);
      const Stats cta = arithmetic_reduce(
          Stats{{ref, mass.x + mass.y, moment.x + moment.y}, top}, exchange.stats_scratch,
          StatsOp{});
      if (warp == 0) {
        // Lane p posts this CTA's partial into slot [rank] of peer p.
        uint4 a, b;
        encode(cta, a, b);
        a = shuffle_first(a);
        b = shuffle_first(b);
        if (!posted) {
          cluster_wait();
          posted = true;
        }
        if (lane < CLUSTER) {
          const uint32_t landed = peer_address(&exchange.stats_landed[slot], lane);
          const uint32_t address = peer_address(&exchange.stats[slot][rank][0], lane);
          push(address, a, landed);
          push(address + sizeof(uint4), b, landed);
        }
      }
      continue;
    }

    // ---- Second pass. -------------------------------------------------------
    if (warp == 0) {
      wait_barrier(&exchange.stats_landed[slot], (item / kSlots) & 1);
      Stats peer{{-INFINITY, 0.f, 0.f}, {-INFINITY, INT_MAX}};
      if (lane < CLUSTER) {
        peer = decode_stats(exchange.stats[slot][lane][0], exchange.stats[slot][lane][1]);
      }
      const Stats total = warp_reduce(peer, CLUSTER, StatsOp{});
      if (lane == 0) {
        expect_bytes(&exchange.stats_landed[slot], exchange_bytes);
        // Every thread's reference is at most its maximum, so the moments
        // rebase to M' with a factor of at most one.
        const Best top = total.top;
        const Moments moments = rebase(total.moments, top.value);
        const float x = params.logits[position * params.vocab + top.index];
        const uint4 bits = philox(seed, static_cast<uint32_t>(top.index) >> 2, canvas_index,
                                  static_cast<uint32_t>(step), sample_word);
        const uint32_t lanes[4] = {bits.x, bits.y, bits.z, bits.w};
        exchange.maximum = top.value;
        exchange.threshold = gumbel_score(__fdiv_rn(x, temp), lanes[top.index & 3]) - kGumbelReach;
        if (rank == 0) {
          // Entropy is non-negative, as the reference computes it. Rebased
          // moments round without preserving that sign where a near one-hot
          // position's entropy approaches zero; the clamp keeps every entropy
          // and mean entropy >= 0, so a confidence of zero stops no row.
          params.entropy[position] =
              fmaxf(0.f, logf(moments.mass) - moments.moment / moments.mass);
        }
      }
    }
    arithmetic_sync();
    const float maximum = exchange.maximum;
    const float threshold = exchange.threshold;
    const float2 negated_max2 = make_float2(-maximum * kLog2e, -maximum * kLog2e);
    uint2* destination = reinterpret_cast<uint2*>(
        params.weights + position * params.weight_stride + slice_start);
    Best sample{-INFINITY, INT_MAX};
    Best argmax{-INFINITY, INT_MAX};
    float2 normalizer_a = make_float2(0.f, 0.f), normalizer_b = make_float2(0.f, 0.f);
    // Candidate groups queue up per warp and are scored 32 at a time, one per
    // lane, so the rare Philox and logarithm work never diverges.
    Candidate* queue = exchange.queue[warp];
    int queued = 0;
    const Evaluation evaluation{seed, canvas_index, static_cast<uint32_t>(step), sample_word,
                                temp};
    for (int c = 0; c < chunks; ++c) {
      float4 x[kGroups];
      take_chunk(ring, exchange.full, exchange.empty, n++, x);
      const int first = c * kChunkVectors + tid;
      float2 q[2 * kGroups];
      scale(x, reciprocal2, q);
      if (__any_sync(0xffffffffu, largest(q) >= threshold)) {
        // Tokens that can win the Gumbel race, and every token holding the
        // maximum.
#pragma unroll
        for (int u = 0; u < kGroups; ++u) {
          const bool hit = fmaxf(fmaxf(q[2 * u].x, q[2 * u].y),
                                 fmaxf(q[2 * u + 1].x, q[2 * u + 1].y)) >= threshold;
          const uint32_t ballot = __ballot_sync(0xffffffffu, hit);
          if (ballot) {
            if (hit) {
              const int at = queued + __popc(ballot & ((1u << lane) - 1u));
              queue[at] = {x[u], static_cast<int32_t>(slice_start +
                                                      4 * (first + u * kArithmeticThreads))};
            }
            queued += __popc(ballot);
            if (queued >= kWarp) {
              __syncwarp();
              evaluate(queue[lane], evaluation, sample, argmax);
              __syncwarp();
              if (lane < queued - kWarp) {
                queue[lane] = queue[lane + kWarp];
              }
              queued -= kWarp;
              __syncwarp();
            }
          }
        }
      }
#pragma unroll
      for (int u = 0; u < kGroups; ++u) {
        uint2 packed;
        const float2 b_lo = unpack_bf16(pack_bf16(q[2 * u]));
        const float2 b_hi = unpack_bf16(pack_bf16(q[2 * u + 1]));
        const float2 a_lo = __ffma2_rn(b_lo, log2e2, negated_max2);
        const float2 a_hi = __ffma2_rn(b_hi, log2e2, negated_max2);
        const float2 w_lo = make_float2(exp2_approx(a_lo.x), exp2_approx(a_lo.y));
        const float2 w_hi = make_float2(exp2_approx(a_hi.x), exp2_approx(a_hi.y));
        normalizer_a = __fadd2_rn(normalizer_a, w_lo);
        normalizer_b = __fadd2_rn(normalizer_b, w_hi);
        packed.x = pack_bf16(w_lo);
        packed.y = pack_bf16(w_hi);
        __stcs(destination + first + u * kArithmeticThreads, packed);
      }
    }
    __syncwarp();
    if (lane < queued) {
      evaluate(queue[lane], evaluation, sample, argmax);
    }
    const float2 normalizer = __fadd2_rn(normalizer_a, normalizer_b);
    const Tally tally = arithmetic_reduce(Tally{sample, argmax, normalizer.x + normalizer.y},
                                          exchange.tally_scratch, TallyOp{});
    if (warp == 0) {
      uint4 a, b;
      encode(tally, a, b);
      a = shuffle_first(a);
      b = shuffle_first(b);
      if (lane == 0) {
        const uint32_t landed = peer_address(&exchange.tallies_landed[slot], 0);
        const uint32_t address = peer_address(&exchange.tallies[slot][rank][0], 0);
        push(address, a, landed);
        push(address + sizeof(uint4), b, landed);
      }
      // CTA 0 collects the previous item's tallies, whose posts have usually
      // landed by now, and the last item's before it exits.
      if (rank == 0) {
        if (item > 0) {
          collect<CLUSTER>(exchange, item - 1, params, schedule.position(item - 1));
        }
        if (item == schedule.count - 1) {
          collect<CLUSTER>(exchange, item, params, position);
        }
      }
    }
  }
}

// ---- Row decisions -----------------------------------------------------------

struct EosIds {
  int64_t ids[kMaxEos];
  int count;
};

struct AdvanceParams {
  const float* entropy;     // [rows, canvas]
  const int64_t* argmax;    // [rows, canvas]
  const int64_t* sample;    // [rows, canvas]
  const int64_t* seed;      // [rows]
  const int64_t* block;     // [rows]
  const int64_t* step;      // [rows]
  int64_t* history;         // [rows, stability, canvas]
  int64_t* canvas_tokens;   // [rows, canvas]
  int64_t* tokens;          // [rows, canvas]
  bool* finished;           // [rows, 2]
  int64_t vocab;
  int canvas;
  int stability;
  int steps;
  float bound;
  float confidence;
  int64_t pad;
  EosIds eos;
};

// One CTA per canvas row, one thread per canvas position. The kernel is
// latency-bound: every global read issues up front, work that does not
// depend on the acceptance prefix runs before it, and the one sequential
// part, the FP64 prefix chain, reads pre-widened values.
__global__ void advance_kernel(const AdvanceParams params) {
  extern __shared__ __align__(16) float row_smem[];
  float* entropies = row_smem;                                          // [canvas]
  double* prefixes = reinterpret_cast<double*>(entropies + params.canvas);  // [canvas]
  __shared__ double partial[kWarp];
  __shared__ double total;
  __shared__ int first_eos;

  const int row = blockIdx.x;
  const int c = threadIdx.x;
  const int canvas = params.canvas;
  const int64_t offset = static_cast<int64_t>(row) * canvas + c;
  const float h = params.entropy[offset];
  const int64_t argmax = params.argmax[offset];
  const int64_t sample = params.sample[offset];
  const int64_t seed = params.seed[row];
  const int64_t block = params.block[row];
  const int64_t step = params.step[row];
  entropies[c] = h;
  if (c == 0) {
    first_eos = canvas;
  }

  // Stable: the argmax canvas equals each of the previous `stability` ones.
  // Each thread owns its history column, oldest entry first.
  int64_t* column = params.history + static_cast<int64_t>(row) * params.stability * canvas + c;
  bool same = true;
  for (int s = 0; s < params.stability; ++s) {
    same &= column[static_cast<int64_t>(s) * canvas] == argmax;
  }
  for (int s = 0; s + 1 < params.stability; ++s) {
    column[static_cast<int64_t>(s) * canvas] = column[static_cast<int64_t>(s + 1) * canvas];
  }
  if (params.stability) {
    column[static_cast<int64_t>(params.stability - 1) * canvas] = argmax;
  }

  bool eos = false;
  for (int e = 0; e < params.eos.count; ++e) {
    eos |= argmax == params.eos.ids[e];
  }
  const uint4 noise = philox(static_cast<uint64_t>(seed), 0u, static_cast<uint32_t>(c),
                             static_cast<uint32_t>(step), block_word(block, kStreamRenoise));

  // Confident: the FP64 mean entropy, rounded to FP32, is below the bound.
  // Lanes reduce here; one thread adds the warps' partials in order below.
  double value = static_cast<double>(h);
  for (int shift = kWarp / 2; shift; shift >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, shift);
  }
  if (c % kWarp == 0) {
    partial[c / kWarp] = value;
  }
  __syncthreads();
  if (eos) {
    atomicMin(&first_eos, c);
  }

  // Stable ascending rank: smaller entropies first, equal ones by position.
  // Four entries per shared-memory read and two counts keep the scan
  // throughput-bound.
  const float4* entropies4 = reinterpret_cast<const float4*>(entropies);
  int rank_even = 0;
  int rank_odd = 0;
#pragma unroll 8
  for (int j = 0; j < canvas / 4; ++j) {
    const float4 other = entropies4[j];
    const int k = 4 * j;
    rank_even += (other.x < h) || (other.x == h && k < c);
    rank_odd += (other.y < h) || (other.y == h && k + 1 < c);
    rank_even += (other.z < h) || (other.z == h && k + 2 < c);
    rank_odd += (other.w < h) || (other.w == h && k + 3 < c);
  }
  const int rank = rank_even + rank_odd;
  prefixes[rank] = static_cast<double>(h);
  const bool stable = __syncthreads_and(same);

  // Prefix sums accumulate sequentially in FP64 in sorted order, which is
  // how the CPU cumsum of the portable formula evaluates FP32 input; each
  // prefix rounds to FP32 below. The mean's partials sum on another thread
  // meanwhile.
  if (c == 0) {
    // Eight values load ahead of their additions, off the FP64 chain.
    double running = 0.0;
    for (int k = 0; k < canvas; k += 8) {
      double terms[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        terms[u] = prefixes[k + u];
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        running += terms[u];
        prefixes[k + u] = running;
      }
    }
  }
  if (c == canvas - 1) {
    double sum = 0.0;
    for (int w = 0; w < canvas / kWarp; ++w) {
      sum += partial[w];
    }
    total = sum;
  }
  __syncthreads();

  const bool accepted =
      __fsub_rn(static_cast<float>(prefixes[rank]), h) <= params.bound;
  params.canvas_tokens[offset] = accepted ? sample : random_token(noise.x, params.vocab);

  const float mean = static_cast<float>(total / canvas);
  const bool stop = stable && mean < params.confidence;
  const bool done = stop || step == params.steps - 1;

  // Pad every token after the first end-of-sequence token.
  const int first = first_eos;
  params.tokens[offset] = c > first ? params.pad : argmax;
  if (c == 0) {
    params.finished[2 * row] = done;
    params.finished[2 * row + 1] = first < canvas;
  }
}

struct StartParams {
  const int64_t* seed;
  const int64_t* block;
  const int64_t* step;
  int64_t* canvas_tokens;          // [rows, canvas]
  int64_t* history;                // [rows, stability, canvas]
  uint4* self_conditioning;        // [rows * canvas, hidden] as 16-byte words
  int64_t vocab;
  int canvas;
  int stability;
  int64_t row_words;               // 16-byte words per self-conditioning row
};

// Rows at step 0 begin a block: an initial canvas from the INITIAL stream,
// a history that no argmax canvas matches, and zero self-conditioning.
__global__ void start_kernel(const StartParams params) {
  const int row = blockIdx.x;
  if (params.step[row] != 0) {
    return;
  }
  const int canvas = params.canvas;
  const uint64_t seed = static_cast<uint64_t>(params.seed[row]);
  const uint32_t initial = block_word(params.block[row], kStreamInitial);
  for (int c = threadIdx.x; c < canvas; c += blockDim.x) {
    const uint4 bits = philox(seed, 0u, static_cast<uint32_t>(c), 0u, initial);
    params.canvas_tokens[static_cast<int64_t>(row) * canvas + c] =
        random_token(bits.x, params.vocab);
    for (int s = 0; s < params.stability; ++s) {
      params.history[(static_cast<int64_t>(row) * params.stability + s) * canvas + c] = -1;
    }
  }
  const int64_t words = params.row_words * canvas;
  uint4* rows = params.self_conditioning + static_cast<int64_t>(row) * words;
  for (int64_t w = threadIdx.x; w < words; w += blockDim.x) {
    rows[w] = make_uint4(0u, 0u, 0u, 0u);
  }
}

struct ConditionParams {
  const float4* product;      // [positions, hidden] FP32, weights @ embedding
  const float* normalizer;    // [positions]
  uint2* output;              // [positions, hidden] BF16 as groups of four
  int64_t groups;             // float4 groups per row
  int64_t total;              // float4 groups overall
  float scale;
};

// Self-conditioning embedding: bf16(product * scale / normalizer), one
// rounding per element; grid-stride over float4 groups.
__global__ void condition_kernel(const ConditionParams params) {
  for (int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; g < params.total;
       g += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float factor = __fdiv_rn(params.scale, params.normalizer[g / params.groups]);
    const float4 value = params.product[g];
    const float2 two = make_float2(factor, factor);
    uint2 packed;
    packed.x = pack_bf16(__fmul2_rn(make_float2(value.x, value.y), two));
    packed.y = pack_bf16(__fmul2_rn(make_float2(value.z, value.w), two));
    params.output[g] = packed;
  }
}

// ---- Host entry points ---------------------------------------------------------

template <int CLUSTER>
void launch_score(SweepParams params, cudaStream_t stream) {
  auto kernel = score_kernel<CLUSTER>;
  constexpr int ring_bytes = kStages * kChunkBytes;
  cudaLaunchConfig_t config = {};
  config.blockDim = dim3(kSweepThreads);
  config.dynamicSmemBytes = ring_bytes;
  config.stream = stream;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = CLUSTER;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;

  // Host-side function attributes and the number of clusters the device
  // holds at once, fixed on the first launch before any CUDA graph capture.
  // The persistent grid launches no more clusters than that.
  static int resident = 0;
  if (resident == 0) {
    if (CLUSTER > 8) {
      C10_CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    }
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ring_bytes));
    cudaLaunchConfig_t probe = config;
    probe.gridDim = dim3(CLUSTER);
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&resident, kernel, &probe));
    TORCH_CHECK(resident > 0, "the device holds no cluster of the vocabulary sweep");
  }
  params.clusters = static_cast<int>(std::min<int64_t>(resident, params.positions));
  config.gridDim = dim3(static_cast<unsigned>(params.clusters * CLUSTER));
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params));
}

void check_rows(const torch::Tensor& tensor, int64_t rows, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == at::kLong && tensor.dim() == 1 &&
                  tensor.size(0) == rows && tensor.is_contiguous(),
              name, " must be a contiguous CUDA int64 [rows] tensor");
}

}  // namespace

// Scores every canvas position; see uniserve_kernels.diffusion.canvas.score.
void score(torch::Tensor logits, torch::Tensor weights, torch::Tensor normalizer,
           torch::Tensor entropy, torch::Tensor argmax, torch::Tensor sample, torch::Tensor seed,
           torch::Tensor block, torch::Tensor step, int64_t steps, double t_min, double t_delta,
           int64_t cluster) {
  const c10::cuda::CUDAGuard guard(logits.device());
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 3 && logits.scalar_type() == at::kFloat &&
                  logits.is_contiguous() &&
                  reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
              "logits must be contiguous, 16-byte aligned FP32 [rows, canvas, vocab]");
  const int64_t rows = logits.size(0);
  const int64_t canvas = logits.size(1);
  const int64_t vocab = logits.size(2);
  const int64_t positions = rows * canvas;
  TORCH_CHECK(positions > 0 && positions * cluster <= INT_MAX, "canvas positions out of range");
  TORCH_CHECK(vocab % (kChunkFloats * cluster) == 0 && vocab <= INT_MAX,
              "the vocabulary must split into whole 4096-token chunks per CTA");
  TORCH_CHECK(weights.is_cuda() && weights.dim() == 2 &&
                  weights.scalar_type() == at::kBFloat16 && weights.size(0) == positions &&
                  weights.size(1) == vocab && weights.stride(1) == 1 &&
                  weights.stride(0) % 4 == 0 && weights.stride(0) >= vocab &&
                  reinterpret_cast<uintptr_t>(weights.data_ptr()) % 8 == 0,
              "weights must be BF16 [positions, vocab] rows with 8-byte alignment");
  {
    // Peers of a cluster may still read their logits when a CTA writes its
    // weights, so the two must not share storage.
    const auto logits_begin = reinterpret_cast<uintptr_t>(logits.data_ptr());
    const auto logits_end = logits_begin + logits.numel() * sizeof(float);
    const auto begin = reinterpret_cast<uintptr_t>(weights.data_ptr());
    const auto end = begin + ((positions - 1) * weights.stride(0) + vocab) * 2;
    TORCH_CHECK(end <= logits_begin || begin >= logits_end,
                "weights must not overlap the logits");
  }
  for (const auto& output : {normalizer, entropy, argmax, sample}) {
    TORCH_CHECK(output.is_cuda() && output.is_contiguous() && output.numel() == positions,
                "per-position outputs must be contiguous CUDA [rows, canvas] tensors");
  }
  TORCH_CHECK(normalizer.scalar_type() == at::kFloat && entropy.scalar_type() == at::kFloat,
              "normalizer and entropy must be FP32");
  TORCH_CHECK(argmax.scalar_type() == at::kLong && sample.scalar_type() == at::kLong,
              "argmax and sample must be int64");
  check_rows(seed, rows, "seed");
  check_rows(block, rows, "block");
  check_rows(step, rows, "step");

  SweepParams params;
  params.logits = logits.data_ptr<float>();
  params.weights = reinterpret_cast<__nv_bfloat16*>(weights.data_ptr());
  params.weight_stride = weights.stride(0);
  params.normalizer = normalizer.data_ptr<float>();
  params.entropy = entropy.data_ptr<float>();
  params.argmax = argmax.data_ptr<int64_t>();
  params.sample = sample.data_ptr<int64_t>();
  params.seed = seed.data_ptr<int64_t>();
  params.block = block.data_ptr<int64_t>();
  params.step = step.data_ptr<int64_t>();
  params.positions = positions;
  params.vocab = vocab;
  params.canvas = canvas;
  params.clusters = 0;
  params.steps = static_cast<int>(steps);
  params.t_min = static_cast<float>(t_min);
  params.t_delta = static_cast<float>(t_delta);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (cluster) {
    case 1: launch_score<1>(params, stream); break;
    case 2: launch_score<2>(params, stream); break;
    case 4: launch_score<4>(params, stream); break;
    case 8: launch_score<8>(params, stream); break;
    case 16: launch_score<16>(params, stream); break;
    default: TORCH_CHECK(false, "cluster size must be 1, 2, 4, 8 or 16");
  }
}

// Accepts, re-noises and decides every canvas row; see
// uniserve_kernels.diffusion.canvas.advance.
void advance(torch::Tensor entropy, torch::Tensor argmax, torch::Tensor sample,
             torch::Tensor seed, torch::Tensor block, torch::Tensor step, torch::Tensor history,
             torch::Tensor canvas_tokens, torch::Tensor tokens, torch::Tensor finished,
             int64_t vocab, int64_t steps, double bound, double confidence,
             std::vector<int64_t> eos, int64_t pad) {
  const c10::cuda::CUDAGuard guard(canvas_tokens.device());
  TORCH_CHECK(canvas_tokens.is_cuda() && canvas_tokens.dim() == 2 &&
                  canvas_tokens.scalar_type() == at::kLong && canvas_tokens.is_contiguous(),
              "canvas must be contiguous CUDA int64 [rows, canvas]");
  const int64_t rows = canvas_tokens.size(0);
  const int64_t canvas = canvas_tokens.size(1);
  TORCH_CHECK(canvas > 0 && canvas <= 1024 && canvas % kWarp == 0,
              "canvas length must be a positive multiple of 32 up to 1024");
  TORCH_CHECK(history.dim() == 3 && history.size(0) == rows && history.size(2) == canvas &&
                  history.scalar_type() == at::kLong && history.is_contiguous(),
              "history must be contiguous int64 [rows, stability, canvas]");
  TORCH_CHECK(entropy.scalar_type() == at::kFloat && entropy.is_contiguous() &&
                  entropy.numel() == rows * canvas,
              "entropy must be contiguous FP32 [rows, canvas]");
  for (const auto& tensor : {argmax, sample, tokens}) {
    TORCH_CHECK(tensor.scalar_type() == at::kLong && tensor.is_contiguous() &&
                    tensor.numel() == rows * canvas,
                "argmax, sample and tokens must be contiguous int64 [rows, canvas]");
  }
  TORCH_CHECK(finished.scalar_type() == at::kBool && finished.is_contiguous() &&
                  finished.numel() == 2 * rows,
              "finished must be contiguous bool [rows, 2]");
  TORCH_CHECK(static_cast<int>(eos.size()) <= kMaxEos, "at most 8 end-of-sequence ids");
  check_rows(seed, rows, "seed");
  check_rows(block, rows, "block");
  check_rows(step, rows, "step");
  if (rows == 0) {
    return;
  }

  AdvanceParams params;
  params.entropy = entropy.data_ptr<float>();
  params.argmax = argmax.data_ptr<int64_t>();
  params.sample = sample.data_ptr<int64_t>();
  params.seed = seed.data_ptr<int64_t>();
  params.block = block.data_ptr<int64_t>();
  params.step = step.data_ptr<int64_t>();
  params.history = history.data_ptr<int64_t>();
  params.canvas_tokens = canvas_tokens.data_ptr<int64_t>();
  params.tokens = tokens.data_ptr<int64_t>();
  params.finished = finished.data_ptr<bool>();
  params.vocab = vocab;
  params.canvas = static_cast<int>(canvas);
  params.stability = static_cast<int>(history.size(1));
  params.steps = static_cast<int>(steps);
  params.bound = static_cast<float>(bound);
  params.confidence = static_cast<float>(confidence);
  params.pad = pad;
  params.eos.count = static_cast<int>(eos.size());
  for (int e = 0; e < kMaxEos; ++e) {
    params.eos.ids[e] = e < params.eos.count ? eos[e] : -1;
  }
  const size_t smem = static_cast<size_t>(canvas) * (sizeof(float) + sizeof(double));
  advance_kernel<<<static_cast<unsigned>(rows), static_cast<unsigned>(canvas), smem,
                   at::cuda::getCurrentCUDAStream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Begins a block for every row at step 0; see
// uniserve_kernels.diffusion.canvas.start.
void start(torch::Tensor seed, torch::Tensor block, torch::Tensor step,
           torch::Tensor canvas_tokens, torch::Tensor history, torch::Tensor self_conditioning,
           int64_t vocab) {
  const c10::cuda::CUDAGuard guard(canvas_tokens.device());
  TORCH_CHECK(canvas_tokens.is_cuda() && canvas_tokens.dim() == 2 &&
                  canvas_tokens.scalar_type() == at::kLong && canvas_tokens.is_contiguous(),
              "canvas must be contiguous CUDA int64 [rows, canvas]");
  const int64_t rows = canvas_tokens.size(0);
  const int64_t canvas = canvas_tokens.size(1);
  TORCH_CHECK(history.dim() == 3 && history.size(0) == rows && history.size(2) == canvas &&
                  history.scalar_type() == at::kLong && history.is_contiguous(),
              "history must be contiguous int64 [rows, stability, canvas]");
  TORCH_CHECK(self_conditioning.dim() == 2 && self_conditioning.size(0) == rows * canvas &&
                  self_conditioning.is_contiguous() &&
                  (self_conditioning.size(1) * self_conditioning.element_size()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(self_conditioning.data_ptr()) % 16 == 0,
              "self-conditioning must be contiguous [rows * canvas, hidden] with 16-byte rows");
  check_rows(seed, rows, "seed");
  check_rows(block, rows, "block");
  check_rows(step, rows, "step");
  if (rows == 0) {
    return;
  }

  StartParams params;
  params.seed = seed.data_ptr<int64_t>();
  params.block = block.data_ptr<int64_t>();
  params.step = step.data_ptr<int64_t>();
  params.canvas_tokens = canvas_tokens.data_ptr<int64_t>();
  params.history = history.data_ptr<int64_t>();
  params.self_conditioning = reinterpret_cast<uint4*>(self_conditioning.data_ptr());
  params.vocab = vocab;
  params.canvas = static_cast<int>(canvas);
  params.stability = static_cast<int>(history.size(1));
  params.row_words = self_conditioning.size(1) * self_conditioning.element_size() / 16;
  start_kernel<<<static_cast<unsigned>(rows), 256, 0, at::cuda::getCurrentCUDAStream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Normalizes and scales the self-conditioning product; see
// uniserve_kernels.diffusion.canvas.condition.
void condition(torch::Tensor product, torch::Tensor normalizer, double scale,
               torch::Tensor output) {
  const c10::cuda::CUDAGuard guard(product.device());
  TORCH_CHECK(product.is_cuda() && product.dim() == 2 && product.scalar_type() == at::kFloat &&
                  product.is_contiguous() && product.size(1) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(product.data_ptr()) % 16 == 0,
              "product must be contiguous FP32 [positions, hidden], hidden divisible by 4");
  TORCH_CHECK(output.sizes() == product.sizes() && output.scalar_type() == at::kBFloat16 &&
                  output.is_contiguous() && reinterpret_cast<uintptr_t>(output.data_ptr()) % 8 == 0,
              "output must be contiguous BF16 shaped like the product");
  TORCH_CHECK(normalizer.scalar_type() == at::kFloat && normalizer.is_contiguous() &&
                  normalizer.numel() == product.size(0),
              "normalizer must be contiguous FP32 [positions]");
  ConditionParams params;
  params.product = reinterpret_cast<const float4*>(product.data_ptr<float>());
  params.normalizer = normalizer.data_ptr<float>();
  params.output = reinterpret_cast<uint2*>(output.data_ptr());
  params.groups = product.size(1) / 4;
  params.total = product.numel() / 4;
  params.scale = static_cast<float>(scale);
  if (params.total == 0) {
    return;
  }
  const int threads = 256;
  const int64_t blocks = std::min<int64_t>((params.total + threads - 1) / threads, 8192);
  condition_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                     at::cuda::getCurrentCUDAStream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Defined in product.cpp, which the host compiler builds against cuBLASLt.
void product(torch::Tensor weights, torch::Tensor table, torch::Tensor output,
             torch::Tensor scratch);
std::vector<int64_t> product_algorithm(torch::Tensor weights, torch::Tensor table,
                                       torch::Tensor output, torch::Tensor scratch);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, binding) {
  binding.def("score", &score, "Score canvas positions over the vocabulary");
  binding.def("product", &product, "Multiply self-conditioning weights by the embedding table");
  binding.def("product_algorithm", &product_algorithm,
              "Configuration of the algorithm chosen for a product shape");
  binding.def("condition", &condition, "Normalize the self-conditioning product");
  binding.def("advance", &advance, "Accept, re-noise and decide canvas rows");
  binding.def("start", &start, "Begin a block on rows at step 0");
}
