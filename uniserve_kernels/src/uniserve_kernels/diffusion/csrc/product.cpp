// Dense BF16 product with FP32 accumulation and output through cuBLASLt,
// with the algorithm measured once per problem shape.
//
// The self-conditioning product multiplies [positions, vocab] weights by the
// [vocab, hidden] embedding table: a long reduction (vocab = 262144) with a
// short output. Which of the algorithms cuBLASLt proposes for such a shape
// runs fastest depends on the number of positions and is often not its
// first proposal, so the first product of each shape times the proposals on
// the caller's stream and keeps the fastest. Every candidate accumulates in
// FP32, so all choices satisfy the same dot-product rounding bound; they
// differ only in summation order.
//
// The products run on PyTorch's cuBLASLt handle in the caller's scratch
// workspace, whose size bounds the split-K partial sums an algorithm may
// keep. Measuring synchronizes with the device and therefore must happen
// outside stream capture; later launches capture into CUDA graphs.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

#define CHECK_CUBLASLT(expression)                                                 \
  do {                                                                             \
    const cublasStatus_t status = (expression);                                    \
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, #expression, " failed with status ", \
                static_cast<int>(status));                                         \
  } while (0)

// Candidates the heuristic proposes and the timed launches per candidate.
constexpr int kCandidates = 8;
constexpr int kRounds = 5;

// One problem shape: device, output rows and columns, reduction length, the
// operands' row strides, the byte alignment all operand addresses share and
// the workspace size.
using Shape = std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, uint32_t,
                         size_t>;

// The descriptors of one shape and the algorithm measured fastest for it.
// Plans live for the process: graphs replay their launches.
struct Plan {
  cublasLtMatmulDesc_t operation;
  cublasLtMatrixLayout_t table;
  cublasLtMatrixLayout_t weights;
  cublasLtMatrixLayout_t output;
  cublasLtMatmulAlgo_t algorithm;
};

std::mutex plans_mutex;
std::map<Shape, Plan> plans;

uint32_t alignment(const void* address) {
  const auto value = reinterpret_cast<uintptr_t>(address);
  uint32_t bytes = 256;
  while (bytes > 1 && value % bytes) {
    bytes /= 2;
  }
  return bytes;
}

cublasStatus_t launch(const Plan& plan, const cublasLtMatmulAlgo_t& algorithm,
                      const torch::Tensor& weights, const torch::Tensor& table,
                      torch::Tensor& output, void* workspace, size_t workspace_bytes,
                      cudaStream_t stream) {
  const float one = 1.f;
  const float zero = 0.f;
  return cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), plan.operation, &one,
                        table.data_ptr(), plan.table, weights.data_ptr(), plan.weights, &zero,
                        output.data_ptr(), plan.output, output.data_ptr(), plan.output,
                        &algorithm, workspace, workspace_bytes, stream);
}

// Builds the plan of `shape`: the heuristic's candidates within the
// workspace, each launched once to warm up and kRounds times interleaved with
// the others; the lowest median time wins.
Plan measure(const Shape& shape, const torch::Tensor& weights, const torch::Tensor& table,
             torch::Tensor& output, void* workspace, size_t workspace_bytes,
             cudaStream_t stream) {
  const int64_t positions = weights.size(0);
  const int64_t vocab = weights.size(1);
  const int64_t hidden = output.size(1);

  // Column-major view of the row-major product:
  // output^T [hidden, positions] = table^T [hidden, vocab] x weights^T [vocab, positions].
  Plan plan{};
  CHECK_CUBLASLT(cublasLtMatmulDescCreate(&plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t identity = CUBLAS_OP_N;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &identity, sizeof(identity)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &identity, sizeof(identity)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&plan.table, CUDA_R_16BF, hidden, vocab,
                                            table.stride(0)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&plan.weights, CUDA_R_16BF, vocab, positions,
                                            weights.stride(0)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&plan.output, CUDA_R_32F, hidden, positions,
                                            output.stride(0)));

  cublasLtMatmulPreference_t preference;
  CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
  const uint64_t limit = workspace_bytes;
  CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
  // Candidates may assume the alignment every operand address has.
  const uint32_t aligned = std::get<7>(shape);
  for (const auto attribute :
       {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
    CHECK_CUBLASLT(
        cublasLtMatmulPreferenceSetAttribute(preference, attribute, &aligned, sizeof(aligned)));
  }

  cublasLtMatmulHeuristicResult_t results[kCandidates];
  int found = 0;
  CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
      at::cuda::getCurrentCUDABlasLtHandle(), plan.operation, plan.table, plan.weights,
      plan.output, plan.output, preference, kCandidates, results, &found));
  cublasLtMatmulPreferenceDestroy(preference);
  TORCH_CHECK(found > 0, "cuBLASLt proposes no algorithm for the self-conditioning product");

  // Warm-up launches drop candidates the device rejects.
  std::vector<int> candidates;
  for (int i = 0; i < found; ++i) {
    if (launch(plan, results[i].algo, weights, table, output, workspace, workspace_bytes,
               stream) == CUBLAS_STATUS_SUCCESS) {
      candidates.push_back(i);
    }
  }
  TORCH_CHECK(!candidates.empty(),
              "no cuBLASLt algorithm runs the self-conditioning product");

  std::vector<cudaEvent_t> events(2 * kRounds * candidates.size());
  for (auto& event : events) {
    C10_CUDA_CHECK(cudaEventCreate(&event));
  }
  for (int round = 0; round < kRounds; ++round) {
    for (size_t c = 0; c < candidates.size(); ++c) {
      const size_t slot = 2 * (round * candidates.size() + c);
      C10_CUDA_CHECK(cudaEventRecord(events[slot], stream));
      CHECK_CUBLASLT(launch(plan, results[candidates[c]].algo, weights, table, output,
                            workspace, workspace_bytes, stream));
      C10_CUDA_CHECK(cudaEventRecord(events[slot + 1], stream));
    }
  }
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  float best_time = 0.f;
  int best = -1;
  for (size_t c = 0; c < candidates.size(); ++c) {
    std::vector<float> times(kRounds);
    for (int round = 0; round < kRounds; ++round) {
      const size_t slot = 2 * (round * candidates.size() + c);
      C10_CUDA_CHECK(cudaEventElapsedTime(&times[round], events[slot], events[slot + 1]));
    }
    std::nth_element(times.begin(), times.begin() + kRounds / 2, times.end());
    const float median = times[kRounds / 2];
    if (best < 0 || median < best_time) {
      best_time = median;
      best = candidates[c];
    }
  }
  for (auto& event : events) {
    C10_CUDA_CHECK(cudaEventDestroy(event));
  }
  plan.algorithm = results[best].algo;
  return plan;
}


// Checks the operands of a product and returns its plan key.
Shape shape_of(const torch::Tensor& weights, const torch::Tensor& table,
               const torch::Tensor& output, const torch::Tensor& scratch) {
  TORCH_CHECK(weights.is_cuda() && table.is_cuda() && output.is_cuda(),
              "the self-conditioning product needs CUDA operands");
  TORCH_CHECK(weights.device() == table.device() && weights.device() == output.device(),
              "the self-conditioning operands must share a device");
  TORCH_CHECK(weights.scalar_type() == at::kBFloat16 && table.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kFloat,
              "the self-conditioning product multiplies BF16 operands into FP32");
  TORCH_CHECK(weights.dim() == 2 && table.dim() == 2 && output.dim() == 2 &&
                  weights.size(1) == table.size(0) && weights.size(0) == output.size(0) &&
                  table.size(1) == output.size(1),
              "the self-conditioning product needs [P, V] x [V, H] -> [P, H]");
  TORCH_CHECK(weights.stride(1) == 1 && table.stride(1) == 1 && output.stride(1) == 1,
              "the self-conditioning operands need unit column stride");
  TORCH_CHECK(scratch.device() == weights.device() && scratch.scalar_type() == at::kByte &&
                  scratch.is_contiguous(),
              "the self-conditioning scratch must be contiguous uint8 on the operands' device");
  return Shape{weights.get_device(),
               weights.size(0),
               output.size(1),
               weights.size(1),
               weights.stride(0),
               table.stride(0),
               output.stride(0),
               std::min({alignment(weights.data_ptr()), alignment(table.data_ptr()),
                         alignment(output.data_ptr())}),
               static_cast<size_t>(scratch.numel())};
}

}  // namespace

// output [positions, hidden] (FP32) = weights [positions, vocab] (BF16) x
// table [vocab, hidden] (BF16), every dot product accumulated in FP32, with
// the contiguous uint8 `scratch` as cuBLASLt workspace.
void product(torch::Tensor weights, torch::Tensor table, torch::Tensor output,
             torch::Tensor scratch) {
  const Shape shape = shape_of(weights, table, output, scratch);
  if (weights.numel() == 0 || output.numel() == 0) {
    return;
  }

  const c10::cuda::CUDAGuard guard(weights.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  void* workspace = scratch.data_ptr();
  const size_t workspace_bytes = static_cast<size_t>(scratch.numel());

  std::unique_lock<std::mutex> lock(plans_mutex);
  auto found = plans.find(shape);
  if (found == plans.end()) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
                "the self-conditioning product measures its algorithm on the first product "
                "of each shape; run one eager step before capturing a CUDA graph");
    found = plans
                .emplace(shape, measure(shape, weights, table, output, workspace,
                                        workspace_bytes, stream))
                .first;
  }
  const Plan plan = found->second;
  lock.unlock();
  CHECK_CUBLASLT(
      launch(plan, plan.algorithm, weights, table, output, workspace, workspace_bytes, stream));
}

// The configuration of the algorithm chosen for these operands' shape, or an
// empty list before their first product: cuBLASLt version, algorithm id,
// tile, stages, split-K count, reduction scheme, CTA swizzling, custom
// option, inner shape and cluster shape.
std::vector<int64_t> product_algorithm(torch::Tensor weights, torch::Tensor table,
                                       torch::Tensor output, torch::Tensor scratch) {
  const Shape shape = shape_of(weights, table, output, scratch);
  cublasLtMatmulAlgo_t algorithm;
  {
    const std::lock_guard<std::mutex> lock(plans_mutex);
    const auto found = plans.find(shape);
    if (found == plans.end()) {
      return {};
    }
    algorithm = found->second.algorithm;
  }
  const auto read = [&](cublasLtMatmulAlgoConfigAttributes_t attribute, auto value) {
    size_t written = 0;
    CHECK_CUBLASLT(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attribute, &value,
                                                        sizeof(value), &written));
    return static_cast<int64_t>(value);
  };
  return {static_cast<int64_t>(cublasLtGetVersion()),
          read(CUBLASLT_ALGO_CONFIG_ID, int32_t{}),
          read(CUBLASLT_ALGO_CONFIG_TILE_ID, uint32_t{}),
          read(CUBLASLT_ALGO_CONFIG_STAGES_ID, uint32_t{}),
          read(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, int32_t{}),
          read(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, uint32_t{}),
          read(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, uint32_t{}),
          read(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, uint32_t{}),
          read(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, uint16_t{}),
          read(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, uint16_t{})};
}
