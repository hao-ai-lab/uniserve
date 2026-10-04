#pragma once

#include <torch/python.h>
#include <deep_ep/common/exception.cuh>

#include "../utils/system.hpp"

namespace deep_ep::elastic {

static at::cuda::CUDAStream get_elastic_comm_stream() {
    // Independent microbatch buffers must not serialize their blocking
    // dispatch/combine kernels on one process-wide communication stream.
    return at::cuda::getStreamFromPool(true);
}

template <int kNumDims>
static auto get_shape(const torch::Tensor& t) {
    EP_HOST_ASSERT(t.dim() == kNumDims);
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<kNumDims>());
}

template <typename dtype_t = void>
static dtype_t* get_data_ptr(const std::optional<torch::Tensor>& t) {
    return t.has_value() ? t->data_ptr<dtype_t>() : nullptr;
}

}  // deep_ep::elastic
