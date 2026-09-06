// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>

#include <limits>
#include <memory>
#include <vector>

namespace {

void check_cuda(CUresult result, const char* operation) {
  const char* message = nullptr;
  if (result != CUDA_SUCCESS) {
    cuGetErrorString(result, &message);
    TORCH_CHECK(false, operation, ": ", message ? message : "CUDA driver error");
  }
}

CUmemAllocationProp allocation_properties(int device) {
  CUmemAllocationProp properties{};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = device;
  properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  return properties;
}

size_t allocation_granularity(int device) {
  const c10::cuda::CUDAGuard guard(device);
  auto properties = allocation_properties(device);
  size_t granularity = 0;
  check_cuda(cuMemGetAllocationGranularity(
                 &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
             "query peer allocation granularity");
  return granularity;
}

struct PeerMapping {
  CUdeviceptr address = 0;
  size_t total_bytes = 0;
  size_t segment_bytes = 0;
  size_t mapped_segments = 0;
  int device = 0;
  std::vector<CUmemGenericAllocationHandle> handles;

  ~PeerMapping() {
    // The distributed runtime retires the tensor after dependent streams and
    // graphs complete. Every imported handle retains its physical allocation.
    const c10::cuda::CUDAGuard guard(device);
    for (size_t index = 0; index < mapped_segments; ++index) {
      cuMemUnmap(address + index * segment_bytes, segment_bytes);
    }
    if (address) {
      cuMemAddressFree(address, total_bytes);
    }
    for (const auto handle : handles) {
      cuMemRelease(handle);
    }
  }
};

class PeerAllocation {
 public:
  PeerAllocation(torch::Tensor prototype, std::vector<int64_t> shape)
      : options_(prototype.options()), shape_(std::move(shape)), device_(prototype.get_device()) {
    TORCH_CHECK(prototype.is_cuda() && !shape_.empty(),
                "peer allocation requires CUDA storage with a leading dimension");
    const c10::cuda::CUDAGuard guard(device_);
    bytes_ = prototype.element_size();
    for (const auto extent : shape_) {
      TORCH_CHECK(extent > 0 && bytes_ <= std::numeric_limits<size_t>::max() / extent,
                  "peer tensor extents must be positive and fit size_t");
      bytes_ *= extent;
    }
    TORCH_CHECK(bytes_ % allocation_granularity(device_) == 0,
                "peer allocation must occupy an integral number of CUDA pages");
    auto properties = allocation_properties(device_);
    check_cuda(cuMemCreate(&handle_, bytes_, &properties, 0), "allocate peer tensor pages");
  }

  ~PeerAllocation() {
    if (handle_) {
      const c10::cuda::CUDAGuard guard(device_);
      cuMemRelease(handle_);
    }
  }

  int export_fd() const {
    const c10::cuda::CUDAGuard guard(device_);
    int descriptor = -1;
    check_cuda(cuMemExportToShareableHandle(
                   &descriptor, handle_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
               "export peer tensor allocation");
    return descriptor;
  }

  torch::Tensor map_peers(const std::vector<int>& descriptors) const {
    const c10::cuda::CUDAGuard guard(device_);
    TORCH_CHECK(!descriptors.empty() &&
                    bytes_ <= std::numeric_limits<size_t>::max() / descriptors.size() &&
                    shape_[0] <= std::numeric_limits<int64_t>::max() / descriptors.size(),
                "peer tensor requires a nonempty, representable owner extent");
    auto mapping = std::make_shared<PeerMapping>();
    mapping->device = device_;
    mapping->segment_bytes = bytes_;
    mapping->total_bytes = bytes_ * descriptors.size();
    for (const auto descriptor : descriptors) {
      CUmemGenericAllocationHandle imported;
      check_cuda(cuMemImportFromShareableHandle(
                     &imported, reinterpret_cast<void*>(static_cast<uintptr_t>(descriptor)),
                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                 "import peer tensor allocation");
      mapping->handles.push_back(imported);
    }
    check_cuda(cuMemAddressReserve(&mapping->address, mapping->total_bytes,
                                  allocation_granularity(device_), 0, 0),
               "reserve peer tensor address range");
    for (const auto handle : mapping->handles) {
      const auto destination = mapping->address + mapping->mapped_segments * bytes_;
      check_cuda(cuMemMap(destination, bytes_, 0, handle, 0), "map peer tensor rows");
      ++mapping->mapped_segments;
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check_cuda(cuMemSetAccess(mapping->address, mapping->total_bytes, &access, 1),
               "enable peer tensor access");
    auto shape = shape_;
    shape[0] *= descriptors.size();
    // The first segment can physically belong to a remote GPU. Tensor execution
    // belongs to the current mapping's device, not CUDA's pointer-owner query.
    return at::for_blob(reinterpret_cast<void*>(mapping->address), shape)
        .deleter([mapping](void*) {})
        .options(options_)
        .target_device(c10::Device(c10::kCUDA, device_))
        .make_tensor();
  }

 private:
  at::TensorOptions options_;
  std::vector<int64_t> shape_;
  int device_;
  size_t bytes_ = 0;
  CUmemGenericAllocationHandle handle_ = 0;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, binding) {
  binding.def("allocation_granularity", &allocation_granularity);
  pybind11::class_<PeerAllocation>(binding, "PeerAllocation")
      .def(pybind11::init<torch::Tensor, std::vector<int64_t>>())
      .def("export_fd", &PeerAllocation::export_fd)
      .def("map_peers", &PeerAllocation::map_peers);
}
