// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>

#define VSA_BHSD true
#include "block_sparse_launch_sm100a.cuh"

torch::Tensor sparse_attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    torch::Tensor valid_sizes) {
  TORCH_CHECK(query.is_cuda(), "sparse attention requires CUDA tensors");
  const c10::cuda::CUDAGuard guard(query.device());
  TORCH_CHECK(query.dim() == 4 && query.size(0) == 1,
              "sparse attention requires [1, heads, rows, 128] tensors");
  const int64_t heads = query.size(1);
  const int64_t query_rows = query.size(2);
  TORCH_CHECK(key.dim() == 4, "sparse attention keys require four dimensions");
  const int64_t key_rows = key.size(2);
  TORCH_CHECK(heads > 0 && query_rows > 0 && key_rows > 0 &&
                  heads <= INT_MAX && query_rows <= INT_MAX && key_rows <= INT_MAX,
              "sparse attention extents must fit positive int32 geometry");
  for (const auto& tensor : {query, key, value}) {
    TORCH_CHECK(tensor.dim() == 4 && tensor.device() == query.device() && tensor.stride(3) == 1 &&
                    tensor.scalar_type() == at::kBFloat16 &&
                    tensor.size(0) == 1 && tensor.size(1) == heads && tensor.size(3) == 128,
                "Q/K/V must be CUDA BF16 tensors with matching heads and contiguous channels");
    TORCH_CHECK(tensor.stride(1) > 0 && tensor.stride(2) > 0 &&
                    tensor.stride(1) % 8 == 0 && tensor.stride(2) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                "Q/K/V strides and base addresses must preserve 16-byte alignment");
  }
  TORCH_CHECK(query.is_contiguous(), "sparse queries require contiguous head-major storage");
  TORCH_CHECK(key.sizes() == value.sizes(), "K/V geometry must agree");
  TORCH_CHECK(query_rows % 64 == 0 && key_rows % 64 == 0,
              "query and key rows must divide 64");
  for (const auto& tensor : {indices, counts, valid_sizes}) {
    TORCH_CHECK(tensor.device() == query.device() && tensor.is_contiguous() &&
                    tensor.scalar_type() == at::kInt,
                "sparse metadata must be contiguous CUDA int32");
  }
  const int64_t query_tiles = query_rows / 64;
  TORCH_CHECK(indices.dim() == 3 && indices.size(0) == heads &&
                  indices.size(1) == query_tiles && indices.size(2) > 0 &&
                  indices.size(2) <= INT_MAX,
              "sparse indices must have shape [heads, query tiles, selected blocks]");
  TORCH_CHECK(counts.dim() == 2 && counts.size(0) == heads &&
                  counts.size(1) == query_tiles && valid_sizes.dim() == 1 &&
                  valid_sizes.numel() * 64 == key_rows,
              "sparse counts and valid key sizes must match Q/K geometry");

  auto output = torch::empty_like(query);
  BlockSparseVsaArgs args{};
  args.q = reinterpret_cast<const __nv_bfloat16*>(query.data_ptr());
  args.k = reinterpret_cast<const __nv_bfloat16*>(key.data_ptr());
  args.v = reinterpret_cast<const __nv_bfloat16*>(value.data_ptr());
  args.o = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  args.q2k_idx = indices.data_ptr<int>();
  args.q2k_num = counts.data_ptr<int>();
  args.variable_block_sizes = valid_sizes.data_ptr<int>();
  args.batch = 1;
  args.num_heads = static_cast<int>(heads);
  args.seqlen = static_cast<int>(query_rows);
  args.key_seqlen = static_cast<int>(key_rows);
  args.key_row_stride = key.stride(2);
  args.key_head_stride = key.stride(1);
  args.value_row_stride = value.stride(2);
  args.value_head_stride = value.stride(1);
  args.head_dim = 128;
  args.num_blocks = static_cast<int>(query_tiles);
  args.max_kv = static_cast<int>(indices.size(2));
  args.sm_scale = 1.0f / std::sqrt(128.0f);
  const auto result = launch_block_sparse_sm100a(args, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(result == cudaSuccess, "SM100 sparse attention: ", cudaGetErrorString(result));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, binding) {
  binding.def("forward", &sparse_attention);
}
