// Dense BF16 product with FP32 accumulation and output through cuBLASLt, with
// one algorithm per problem shape.
//
// The self-conditioning product multiplies [positions, vocab] weights by the
// [vocab, hidden] embedding table: a long reduction (vocab = 262144) with a
// short output. Which of the algorithms cuBLASLt proposes for such a shape
// runs fastest depends on the number of positions and is often not its
// first proposal. Every candidate accumulates in FP32, so all choices
// satisfy the same dot-product rounding bound; they differ in summation
// order, and so in the last bits of the product.
//
// The caller passes the configuration a shipped table names for the shape
// (measured once per device model and cuBLASLt version); the first product
// of the shape takes the proposal with that configuration, which makes the
// rounding reproducible across processes. Without one, or when cuBLASLt no
// longer proposes it, the first product times the proposals on the caller's
// stream and keeps the fastest; that choice can differ between processes.
//
// The products run on PyTorch's cuBLASLt handle in the caller's scratch
// workspace, whose size bounds the split-K partial sums an algorithm may
// keep. Choosing synchronizes with the device and therefore must happen
// outside stream capture; later launches capture into CUDA graphs.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

#define CHECK_CUBLASLT(expression)                                                    \
  do {                                                                                \
    const cublasStatus_t status = (expression);                                       \
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, #expression, " failed with status ", \
                static_cast<int>(status));                                            \
  } while (0)

// Candidates requested from the heuristic, and interleaved timing rounds of
// the choice a process makes without a shipped configuration.
constexpr int kCandidates = 8;
constexpr int kStartupRounds = 5;

// An algorithm configuration: id, tile, stages, split-K count, reduction
// scheme, CTA swizzling, custom option, inner shape and cluster shape.
constexpr int kConfigFields = 9;
using Config = std::array<int64_t, kConfigFields>;

// One problem shape: device, output rows and columns, reduction length, the
// operands' row strides, the byte alignment all operand addresses share and
// the workspace size.
using Shape = std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, uint32_t,
                         size_t>;

// Matrix descriptors of one shape.
struct Descriptors {
  cublasLtMatmulDesc_t operation;
  cublasLtMatrixLayout_t table;
  cublasLtMatrixLayout_t weights;
  cublasLtMatrixLayout_t output;
};

// The descriptors of one shape, its algorithm and whether a shipped
// configuration chose it. Plans live for the process: graphs replay their
// launches.
struct Plan {
  Descriptors descriptors;
  cublasLtMatmulAlgo_t algorithm;
  bool shipped;
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

template <typename T>
int64_t attribute(const cublasLtMatmulAlgo_t& algorithm,
                  cublasLtMatmulAlgoConfigAttributes_t name) {
  T value{};
  size_t written = 0;
  CHECK_CUBLASLT(
      cublasLtMatmulAlgoConfigGetAttribute(&algorithm, name, &value, sizeof(value), &written));
  return static_cast<int64_t>(value);
}

Config config_of(const cublasLtMatmulAlgo_t& algorithm) {
  return {attribute<int32_t>(algorithm, CUBLASLT_ALGO_CONFIG_ID),
          attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID),
          attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID),
          attribute<int32_t>(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
          attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
          attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
          attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
          attribute<uint16_t>(algorithm, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID),
          attribute<uint16_t>(algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)};
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

// Column-major view of the row-major product:
// output^T [hidden, positions] = table^T [hidden, vocab] x weights^T [vocab, positions].
Descriptors describe(const torch::Tensor& weights, const torch::Tensor& table,
                     const torch::Tensor& output) {
  const int64_t positions = weights.size(0);
  const int64_t vocab = weights.size(1);
  const int64_t hidden = output.size(1);
  Descriptors descriptors{};
  CHECK_CUBLASLT(
      cublasLtMatmulDescCreate(&descriptors.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t identity = CUBLAS_OP_N;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA, &identity, sizeof(identity)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB, &identity, sizeof(identity)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&descriptors.table, CUDA_R_16BF, hidden, vocab,
                                            table.stride(0)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&descriptors.weights, CUDA_R_16BF, vocab, positions,
                                            weights.stride(0)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&descriptors.output, CUDA_R_32F, hidden, positions,
                                            output.stride(0)));
  return descriptors;
}

void release(const Descriptors& descriptors) {
  cublasLtMatrixLayoutDestroy(descriptors.output);
  cublasLtMatrixLayoutDestroy(descriptors.weights);
  cublasLtMatrixLayoutDestroy(descriptors.table);
  cublasLtMatmulDescDestroy(descriptors.operation);
}

// The heuristic's proposals within the workspace, for operands aligned to the
// shape's shared alignment.
std::vector<cublasLtMatmulHeuristicResult_t> propose(const Descriptors& descriptors,
                                                     const Shape& shape) {
  cublasLtMatmulPreference_t preference;
  CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
  const uint64_t limit = std::get<8>(shape);
  CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
  const uint32_t aligned = std::get<7>(shape);
  for (const auto name :
       {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
    CHECK_CUBLASLT(
        cublasLtMatmulPreferenceSetAttribute(preference, name, &aligned, sizeof(aligned)));
  }
  std::vector<cublasLtMatmulHeuristicResult_t> results(kCandidates);
  int found = 0;
  CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
      at::cuda::getCurrentCUDABlasLtHandle(), descriptors.operation, descriptors.table,
      descriptors.weights, descriptors.output, descriptors.output, preference, kCandidates,
      results.data(), &found));
  cublasLtMatmulPreferenceDestroy(preference);
  results.resize(found);
  TORCH_CHECK(found > 0, "cuBLASLt proposes no algorithm for the self-conditioning product");
  return results;
}

cublasStatus_t launch(const Descriptors& descriptors, const cublasLtMatmulAlgo_t& algorithm,
                      const torch::Tensor& weights, const torch::Tensor& table,
                      const torch::Tensor& output, const torch::Tensor& scratch,
                      cudaStream_t stream) {
  const float one = 1.f;
  const float zero = 0.f;
  return cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), descriptors.operation, &one,
                        table.data_ptr(), descriptors.table, weights.data_ptr(),
                        descriptors.weights, &zero, output.data_ptr(), descriptors.output,
                        output.data_ptr(), descriptors.output, &algorithm, scratch.data_ptr(),
                        static_cast<size_t>(scratch.numel()), stream);
}

// Median milliseconds of every proposal: one warm-up launch each, then
// `rounds` rounds that launch every proposal once, in the same order. When
// `flush_bytes` is positive, a device memset of that many bytes precedes
// every timed launch, outside the timed interval, to evict the L2. Proposals
// the device rejects get NaN.
std::vector<float> time_proposals(const Descriptors& descriptors,
                                  const std::vector<cublasLtMatmulHeuristicResult_t>& proposals,
                                  const torch::Tensor& weights, const torch::Tensor& table,
                                  const torch::Tensor& output, const torch::Tensor& scratch,
                                  int64_t rounds, int64_t flush_bytes, cudaStream_t stream) {
  std::vector<size_t> runnable;
  for (size_t i = 0; i < proposals.size(); ++i) {
    if (launch(descriptors, proposals[i].algo, weights, table, output, scratch, stream) ==
        CUBLAS_STATUS_SUCCESS) {
      runnable.push_back(i);
    }
  }
  TORCH_CHECK(!runnable.empty(), "no cuBLASLt algorithm runs the self-conditioning product");

  torch::Tensor flush;
  if (flush_bytes > 0) {
    flush = torch::empty({flush_bytes}, weights.options().dtype(torch::kUInt8));
  }
  std::vector<cudaEvent_t> events(2 * rounds * runnable.size());
  for (auto& event : events) {
    C10_CUDA_CHECK(cudaEventCreate(&event));
  }
  for (int64_t round = 0; round < rounds; ++round) {
    for (size_t r = 0; r < runnable.size(); ++r) {
      if (flush.defined()) {
        C10_CUDA_CHECK(cudaMemsetAsync(flush.data_ptr(), static_cast<int>(round & 0xff),
                                       static_cast<size_t>(flush_bytes), stream));
      }
      const size_t slot = 2 * (round * runnable.size() + r);
      C10_CUDA_CHECK(cudaEventRecord(events[slot], stream));
      CHECK_CUBLASLT(launch(descriptors, proposals[runnable[r]].algo, weights, table, output,
                            scratch, stream));
      C10_CUDA_CHECK(cudaEventRecord(events[slot + 1], stream));
    }
  }
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> medians(proposals.size(), NAN);
  for (size_t r = 0; r < runnable.size(); ++r) {
    std::vector<float> times(rounds);
    for (int64_t round = 0; round < rounds; ++round) {
      const size_t slot = 2 * (round * runnable.size() + r);
      C10_CUDA_CHECK(cudaEventElapsedTime(&times[round], events[slot], events[slot + 1]));
    }
    std::nth_element(times.begin(), times.begin() + rounds / 2, times.end());
    medians[runnable[r]] = times[rounds / 2];
  }
  for (auto& event : events) {
    C10_CUDA_CHECK(cudaEventDestroy(event));
  }
  return medians;
}

// The plan of a shape: the proposal whose configuration equals `preferred`
// when there is one, otherwise the fastest proposal over kStartupRounds.
Plan choose(const Shape& shape, const std::vector<int64_t>& preferred,
            const torch::Tensor& weights, const torch::Tensor& table,
            const torch::Tensor& output, const torch::Tensor& scratch, cudaStream_t stream) {
  const Descriptors descriptors = describe(weights, table, output);
  const auto proposals = propose(descriptors, shape);
  if (preferred.size() == kConfigFields) {
    for (const auto& proposal : proposals) {
      const Config config = config_of(proposal.algo);
      if (std::equal(config.begin(), config.end(), preferred.begin())) {
        return Plan{descriptors, proposal.algo, true};
      }
    }
  }
  const std::vector<float> medians = time_proposals(descriptors, proposals, weights, table,
                                                    output, scratch, kStartupRounds, 0, stream);
  // time_proposals guarantees one runnable proposal; rejected ones are NaN.
  int best = -1;
  for (size_t i = 0; i < medians.size(); ++i) {
    if (!std::isnan(medians[i]) && (best < 0 || medians[i] < medians[best])) {
      best = static_cast<int>(i);
    }
  }
  return Plan{descriptors, proposals[best].algo, false};
}

}  // namespace

// output [positions, hidden] (FP32) = weights [positions, vocab] (BF16) x
// table [vocab, hidden] (BF16), every dot product accumulated in FP32, with
// the contiguous uint8 `scratch` as cuBLASLt workspace. `preferred` is the
// shipped configuration for the shape, or empty.
void product(torch::Tensor weights, torch::Tensor table, torch::Tensor output,
             torch::Tensor scratch, std::vector<int64_t> preferred) {
  const Shape shape = shape_of(weights, table, output, scratch);
  if (weights.numel() == 0 || output.numel() == 0) {
    return;
  }

  const c10::cuda::CUDAGuard guard(weights.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  std::unique_lock<std::mutex> lock(plans_mutex);
  auto found = plans.find(shape);
  if (found == plans.end()) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
                "the self-conditioning product chooses its algorithm on the first product "
                "of each shape; run one eager step before capturing a CUDA graph");
    found = plans.emplace(shape, choose(shape, preferred, weights, table, output, scratch, stream))
                .first;
  }
  const Plan plan = found->second;
  lock.unlock();
  CHECK_CUBLASLT(launch(plan.descriptors, plan.algorithm, weights, table, output, scratch, stream));
}

// The algorithm chosen for these operands' shape, or an empty list before
// its first product: 1 when a shipped configuration chose it (0 when timed),
// then the configuration fields.
std::vector<int64_t> product_algorithm(torch::Tensor weights, torch::Tensor table,
                                       torch::Tensor output, torch::Tensor scratch) {
  const Shape shape = shape_of(weights, table, output, scratch);
  const std::lock_guard<std::mutex> lock(plans_mutex);
  const auto found = plans.find(shape);
  if (found == plans.end()) {
    return {};
  }
  const Config config = config_of(found->second.algorithm);
  std::vector<int64_t> values{found->second.shipped ? 1 : 0};
  values.insert(values.end(), config.begin(), config.end());
  return values;
}

// Every proposal for these operands with its median milliseconds over
// `rounds` interleaved rounds (NaN when the device rejects it), each launch
// preceded by an L2-evicting memset of `flush_bytes`: rows of the
// configuration fields followed by the median. Synchronizes the device.
std::vector<std::vector<double>> product_proposals(torch::Tensor weights, torch::Tensor table,
                                                   torch::Tensor output, torch::Tensor scratch,
                                                   int64_t rounds, int64_t flush_bytes) {
  const Shape shape = shape_of(weights, table, output, scratch);
  TORCH_CHECK(rounds > 0, "timing needs at least one round");
  const c10::cuda::CUDAGuard guard(weights.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const Descriptors descriptors = describe(weights, table, output);
  const auto proposals = propose(descriptors, shape);
  const std::vector<float> medians = time_proposals(descriptors, proposals, weights, table,
                                                    output, scratch, rounds, flush_bytes, stream);
  std::vector<std::vector<double>> rows;
  for (size_t i = 0; i < proposals.size(); ++i) {
    const Config config = config_of(proposals[i].algo);
    std::vector<double> row(config.begin(), config.end());
    row.push_back(medians[i]);
    rows.push_back(row);
  }
  release(descriptors);
  return rows;
}

int64_t cublaslt_version() { return static_cast<int64_t>(cublasLtGetVersion()); }
