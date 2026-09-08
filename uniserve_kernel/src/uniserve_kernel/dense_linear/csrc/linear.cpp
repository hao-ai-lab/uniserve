// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>
#include <optional>

namespace {

class MathModeGuard {
 public:
  explicit MathModeGuard(cublasHandle_t handle) : handle_(handle) {
    TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle_, &previous_));
    // A split-K algorithm must reduce its partial sums in the accumulator
    // type, independently of the lower-precision output storage format.
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(
        handle_, static_cast<cublasMath_t>(
                     previous_ | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  }

  ~MathModeGuard() { cublasSetMathMode(handle_, previous_); }

 private:
  cublasHandle_t handle_;
  cublasMath_t previous_;
};

torch::Tensor linear_fp32_accum(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && weight.dim() == 2 &&
                  input.is_contiguous() && weight.is_contiguous() &&
                  input.device() == weight.device() &&
                  input.scalar_type() == weight.scalar_type() &&
                  input.size(1) == weight.size(1),
              "dense projection requires compatible contiguous CUDA matrices");
  const auto dtype = input.scalar_type();
  TORCH_CHECK(dtype == torch::kBFloat16 || dtype == torch::kFloat16,
              "FP32 accumulation requires FP16 or BF16 input and weights");
  const auto rows = input.size(0), features = input.size(1), columns = weight.size(0);
  for (const auto extent : {rows, features, columns}) {
    TORCH_CHECK(extent > 0 && extent <= std::numeric_limits<int>::max(),
                "dense projection extents must be positive and fit cuBLAS dimensions");
  }
  if (bias.has_value()) {
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == columns &&
                    bias->device() == input.device() && bias->scalar_type() == dtype,
                "dense projection bias must match output columns, dtype and device");
  }
  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty({rows, columns}, input.options());
  if (bias.has_value()) {
    output.copy_(*bias);
  }
  const auto handle = at::cuda::getCurrentCUDABlasHandle();
  const MathModeGuard math_guard(handle);
  const at::cuda::blas::PointerModeGuard pointer_guard(handle, CUBLAS_POINTER_MODE_HOST);
  const float alpha = 1.0f, beta = bias.has_value() ? 1.0f : 0.0f;
  const auto storage = dtype == torch::kBFloat16 ? CUDA_R_16BF : CUDA_R_16F;
  // Column-major cuBLAS computes Y^T = W * X^T from row-major X and W.
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, features,
      &alpha, weight.data_ptr(), storage, features,
      input.data_ptr(), storage, features, &beta,
      output.data_ptr(), storage, columns, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &linear_fp32_accum);
}
