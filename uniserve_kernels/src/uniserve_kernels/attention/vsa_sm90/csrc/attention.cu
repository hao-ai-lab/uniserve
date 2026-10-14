// SPDX-License-Identifier: Apache-2.0

// PyTorch binding of the SM90a block-sparse attention forward. It validates
// tensor geometry on the host and launches on the current CUDA stream. The
// metadata values (block indices, counts, valid sizes) stay on the device and
// are neither read nor range-checked here.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cmath>
#include <optional>

#include "block_sparse_sm90a.cuh"

namespace {

using uniserve::vsa_sm90::kBlockM;
using uniserve::vsa_sm90::kHeadDim;

void sparse_attention(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output,
                      torch::Tensor indices, torch::Tensor counts, torch::Tensor valid_sizes,
                      std::optional<torch::Tensor> query_valid_rows, double scale) {
  TORCH_CHECK(query.is_cuda(), "sparse attention requires CUDA tensors");
  const c10::cuda::CUDAGuard guard(query.device());
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3, "sparse attention requires [rows, heads, 128] tensors");
  const int64_t query_rows = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t key_rows = key.size(0);
  TORCH_CHECK(heads > 0 && query_rows > 0 && key_rows > 0 && heads <= INT_MAX && query_rows <= INT_MAX &&
                  key_rows <= INT_MAX && heads * (query_rows / kBlockM) <= INT_MAX,
              "sparse attention extents must fit positive int32 geometry");
  // TMA addresses rows and heads by independent strides; channels are
  // contiguous and every stride and base preserves 16-byte alignment.
  for (const auto& tensor : {query, key, value, output}) {
    TORCH_CHECK(tensor.dim() == 3 && tensor.device() == query.device() && tensor.stride(2) == 1 &&
                    tensor.scalar_type() == at::kBFloat16 && tensor.size(1) == heads &&
                    tensor.size(2) == kHeadDim,
                "Q/K/V/O must be CUDA BF16 tensors with matching heads and contiguous channels");
    TORCH_CHECK(tensor.stride(0) > 0 && tensor.stride(1) > 0 && tensor.stride(0) % 8 == 0 &&
                    tensor.stride(1) % 8 == 0 && reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                "Q/K/V/O strides and base addresses must preserve 16-byte alignment");
  }
  TORCH_CHECK(output.sizes() == query.sizes(), "sparse attention output must match the query shape");
  TORCH_CHECK(key.sizes() == value.sizes(), "K/V geometry must agree");
  TORCH_CHECK(query_rows % kBlockM == 0 && key_rows % kBlockM == 0,
              "query and key rows must be whole 128-row tiles");
  const int64_t query_tiles = query_rows / kBlockM;
  for (const auto& tensor : {indices, counts, valid_sizes}) {
    TORCH_CHECK(tensor.device() == query.device() && tensor.is_contiguous() && tensor.scalar_type() == at::kInt,
                "sparse metadata must be contiguous CUDA int32");
  }
  TORCH_CHECK(indices.dim() == 3 && indices.size(0) == heads && indices.size(1) == query_tiles &&
                  indices.size(2) > 0 && indices.size(2) <= INT_MAX,
              "sparse indices must have shape [heads, query tiles, selected blocks]");
  TORCH_CHECK(counts.dim() == 2 && counts.size(0) == heads && counts.size(1) == query_tiles &&
                  valid_sizes.dim() == 1 && valid_sizes.numel() * kBlockM == key_rows,
              "sparse counts and valid key sizes must match Q/K geometry");
  if (query_valid_rows) {
    TORCH_CHECK(query_valid_rows->device() == query.device() && query_valid_rows->is_contiguous() &&
                    query_valid_rows->scalar_type() == at::kInt && query_valid_rows->dim() == 1 &&
                    query_valid_rows->numel() == query_tiles,
                "query valid rows must hold one contiguous CUDA int32 entry per query tile");
  }
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "softmax scale must be positive and finite");

  using uniserve::vsa_sm90::Element;
  uniserve::vsa_sm90::Arguments arguments{
      reinterpret_cast<const Element*>(query.data_ptr()),
      reinterpret_cast<const Element*>(key.data_ptr()),
      reinterpret_cast<const Element*>(value.data_ptr()),
      reinterpret_cast<Element*>(output.data_ptr()),
      query.stride(0),
      query.stride(1),
      key.stride(0),
      key.stride(1),
      value.stride(0),
      value.stride(1),
      output.stride(0),
      output.stride(1),
      static_cast<int32_t>(query_rows),
      static_cast<int32_t>(key_rows),
      static_cast<int32_t>(heads),
      indices.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(),
      valid_sizes.data_ptr<int32_t>(),
      static_cast<int32_t>(indices.size(2)),
      query_valid_rows ? query_valid_rows->data_ptr<int32_t>() : nullptr,
      static_cast<float>(scale),
  };
  const cudaError_t result = uniserve::vsa_sm90::launch(arguments, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(result == cudaSuccess, "SM90 sparse attention: ", cudaGetErrorString(result));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, binding) {
  binding.def("forward", &sparse_attention);
}
