#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega.hpp"
#include "apis/mega_m2n.hpp"
#include "apis/runtime.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    deep_gemm::gemm::register_apis(module);
    deep_gemm::layout::register_apis(module);
    deep_gemm::mega::register_apis(module);
    deep_gemm::mega_m2n::register_apis(module);
    deep_gemm::runtime::register_apis(module);
}
