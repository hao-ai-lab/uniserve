// SPDX-License-Identifier: Apache-2.0

// Expert routing: softmax over each token's router scores, the top-k
// experts, optional renormalization and an optional per-expert scale.
//
// uniserve_kernels.routing documents the contract; the composition in
// uniserve.nn.functional.topk_softmax defines the numbers. One warp owns
// one token and reproduces that composition's FP32 arithmetic order:
//
// - softmax as PyTorch's warp softmax evaluates a row: lane l holds scores
//   l + 32 * i, the maximum and the sum of expf(score - max) reduce by xor
//   butterflies, and each probability is exp / sum;
// - the k largest probabilities in descending order, equal ones in the
//   order torch.topk gives them;
// - their sum by the xor butterfly over k lanes that PyTorch's row sum of a
//   [tokens, k] tensor uses, clamped below by FLT_EPSILON, divides them;
// - a per-expert scale multiplies the result in FP32.
//
// Launches synchronize nothing, so CUDA graphs capture them.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cfloat>
#include <climits>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 4;

// Storage kinds of the scores and of the per-expert scale.
constexpr int kAbsent = -1;
constexpr int kFloat = 0;
constexpr int kBFloat16 = 1;
constexpr int kHalf = 2;

__device__ __forceinline__ float load_value(const void* data, int kind, int64_t index) {
  if (kind == kFloat) return static_cast<const float*>(data)[index];
  if (kind == kHalf) return __half2float(static_cast<const __half*>(data)[index]);
  return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
}

// kIterations = next_power_of_two(experts) / 32 scores per lane (at least
// one); kTop = next_power_of_two(top) picks, the width of the renormalizing
// butterfly (picks past `top` enter it as zeros).
template <int kIterations, int kTop>
__global__ void topk_softmax_kernel(const void* scores, int score_kind, int64_t row_stride,
                                    int rows, int experts, int top, int renormalize,
                                    const void* scale, int scale_kind, int32_t* ids,
                                    float* weights) {
  __shared__ float shared_value[kWarpsPerBlock][kWarp];
  __shared__ int shared_expert[kWarpsPerBlock][kWarp];
  __shared__ bool shared_valid[kWarpsPerBlock][kWarp];
  const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) return;  // uniform per warp

  // Scores past the expert count are -inf, as PyTorch pads a row.
  float values[kIterations];
#pragma unroll
  for (int it = 0; it < kIterations; ++it) {
    const int expert = lane + it * kWarp;
    values[it] = expert < experts
                     ? load_value(scores, score_kind, static_cast<int64_t>(row) * row_stride + expert)
                     : -INFINITY;
  }

  float maximum = values[0];
#pragma unroll
  for (int it = 0; it < kIterations; ++it) maximum = maximum > values[it] ? maximum : values[it];
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    const float other = __shfl_xor_sync(0xffffffffu, maximum, offset);
    maximum = maximum < other ? other : maximum;
  }
  float sum = 0.f;
#pragma unroll
  for (int it = 0; it < kIterations; ++it) {
    values[it] = expf(values[it] - maximum);
    sum += values[it];
  }
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    sum = sum + __shfl_xor_sync(0xffffffffu, sum, offset);
  }
#pragma unroll
  for (int it = 0; it < kIterations; ++it) values[it] = values[it] / sum;

  // Top-k by repeated warp arg-max; every lane ends with all k picks.
  float top_value[kTop];
  float kth = 0.f;
  unsigned taken = 0u;
#pragma unroll
  for (int j = 0; j < kTop; ++j) {
    if (j >= top) {
      top_value[j] = 0.f;
      continue;
    }
    float best = -1.f;
    int best_expert = INT_MAX;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const int expert = lane + it * kWarp;
      if (expert < experts && !(taken >> it & 1u) && values[it] > best) {
        best = values[it];
        best_expert = expert;
      }
    }
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      const float other = __shfl_xor_sync(0xffffffffu, best, offset);
      const int other_expert = __shfl_xor_sync(0xffffffffu, best_expert, offset);
      if (other > best || (other == best && other_expert < best_expert)) {
        best = other;
        best_expert = other_expert;
      }
    }
    top_value[j] = best;
    if (j == top - 1) kth = best;
    if (best_expert % kWarp == lane) taken |= 1u << (best_expert / kWarp);
  }

  // The order of the picks. Equal probabilities keep the order that the
  // composition's torch.topk gives them: its gather lists the probabilities
  // above the k-th largest in expert order, then those equal to it in
  // expert order, and a bitonic network over 32 slots (16 threads, two
  // slots each, strict comparison) sorts them in descending order.
  float* slot_value = shared_value[threadIdx.x / kWarp];
  int* slot_expert = shared_expert[threadIdx.x / kWarp];
  bool* slot_valid = shared_valid[threadIdx.x / kWarp];
  slot_valid[lane] = false;
  slot_value[lane] = 0.f;
  slot_expert[lane] = 0;
  __syncwarp();
  int placed = 0;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const int expert = lane + it * kWarp;
      const bool listed = expert < experts &&
                          (pass == 0 ? values[it] > kth : values[it] == kth);
      const unsigned ballot = __ballot_sync(0xffffffffu, listed);
      const int position = placed + __popc(ballot & ((1u << lane) - 1u));
      if (listed && position < top) {
        slot_value[position] = values[it];
        slot_expert[position] = expert;
        slot_valid[position] = true;
      }
      placed += __popc(ballot);
    }
  }
  __syncwarp();
  auto exchange = [&](int stride, bool descending_half) {
    if (lane < kWarp / 2) {
      const int a = 2 * lane - (lane & (stride - 1));
      const int b = a + stride;
      const bool swap = (slot_value[a] > slot_value[b] && slot_valid[a]) || !slot_valid[b];
      if (swap == descending_half) {
        const float value = slot_value[a];
        const int expert = slot_expert[a];
        const bool valid = slot_valid[a];
        slot_value[a] = slot_value[b];
        slot_expert[a] = slot_expert[b];
        slot_valid[a] = slot_valid[b];
        slot_value[b] = value;
        slot_expert[b] = expert;
        slot_valid[b] = valid;
      }
    }
    __syncwarp();
  };
#pragma unroll
  for (int size = 2; size < kWarp; size *= 2) {
#pragma unroll
    for (int stride = size / 2; stride > 0; stride /= 2) exchange(stride, (lane & (size / 2)) != 0);
  }
#pragma unroll
  for (int stride = kWarp / 2; stride > 0; stride /= 2) exchange(stride, false);

  // Lane j of the first k lanes stores pick j.
  if (lane >= top) return;
  float weight = slot_value[lane];
  const int expert = slot_expert[lane];
  if (renormalize) {
    // The butterfly over k lanes: at each step lane j adds lane j ^ offset.
    float partial[kTop];
#pragma unroll
    for (int j = 0; j < kTop; ++j) partial[j] = top_value[j];
#pragma unroll
    for (int offset = kTop / 2; offset > 0; offset /= 2) {
      float next[kTop];
#pragma unroll
      for (int j = 0; j < kTop; ++j) next[j] = partial[j] + partial[j ^ offset];
#pragma unroll
      for (int j = 0; j < kTop; ++j) partial[j] = next[j];
    }
    const float total = partial[0] < FLT_EPSILON ? FLT_EPSILON : partial[0];
    weight = weight / total;
  }
  if (scale_kind != kAbsent) weight = weight * load_value(scale, scale_kind, expert);
  ids[static_cast<int64_t>(row) * top + lane] = expert;
  weights[static_cast<int64_t>(row) * top + lane] = weight;
}

int kind_of(const torch::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case torch::kFloat32:
      return kFloat;
    case torch::kFloat16:
      return kHalf;
    default:
      return kBFloat16;
  }
}

struct Launch {
  const void* scores;
  int score_kind;
  int64_t row_stride;
  int rows;
  int experts;
  int top;
  int renormalize;
  const void* scale;
  int scale_kind;
  int32_t* ids;
  float* weights;
};

template <int kIterations, int kTop>
void launch(const Launch& a, cudaStream_t stream) {
  const dim3 grid((a.rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  topk_softmax_kernel<kIterations, kTop><<<grid, kWarpsPerBlock * kWarp, 0, stream>>>(
      a.scores, a.score_kind, a.row_stride, a.rows, a.experts, a.top, a.renormalize, a.scale,
      a.scale_kind, a.ids, a.weights);
}

template <int kIterations>
void dispatch_top(const Launch& a, int top, cudaStream_t stream) {
  if (top <= 1) return launch<kIterations, 1>(a, stream);
  if (top <= 2) return launch<kIterations, 2>(a, stream);
  if (top <= 4) return launch<kIterations, 4>(a, stream);
  return launch<kIterations, 8>(a, stream);
}

// Python binding. uniserve_kernels.routing.unsupported validated the
// operands: `scores` rows have unit expert stride and at most 256 experts,
// and `top` is 1..8.
void topk_softmax(const torch::Tensor& scores, int64_t top, bool renormalize,
                  const c10::optional<torch::Tensor>& scale, const torch::Tensor& ids,
                  const torch::Tensor& weights) {
  Launch a{};
  a.experts = static_cast<int>(scores.size(-1));
  a.rows = static_cast<int>(scores.numel() / a.experts);
  a.top = static_cast<int>(top);
  if (a.rows == 0) return;
  a.scores = scores.data_ptr();
  a.score_kind = kind_of(scores);
  a.row_stride = a.rows > 1 ? scores.stride(-2) : a.experts;
  a.renormalize = renormalize ? 1 : 0;
  a.scale = scale.has_value() ? scale->data_ptr() : nullptr;
  a.scale_kind = scale.has_value() ? kind_of(*scale) : kAbsent;
  a.ids = ids.data_ptr<int32_t>();
  a.weights = weights.data_ptr<float>();

  const c10::cuda::CUDAGuard guard(scores.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int per_lane = a.experts <= 32 ? 1 : a.experts <= 64 ? 2 : a.experts <= 128 ? 4 : 8;
  switch (per_lane) {
    case 1:
      dispatch_top<1>(a, static_cast<int>(top), stream);
      break;
    case 2:
      dispatch_top<2>(a, static_cast<int>(top), stream);
      break;
    case 4:
      dispatch_top<4>(a, static_cast<int>(top), stream);
      break;
    default:
      dispatch_top<8>(a, static_cast<int>(top), stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("topk_softmax", &topk_softmax, "Softmax top-k expert routing");
}
