// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>

#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

namespace {

void check_cuda(CUresult result, const char* operation) {
  const char* message = nullptr;
  if (result != CUDA_SUCCESS) {
    cuGetErrorString(result, &message);
    TORCH_CHECK(false, operation, ": ", message ? message : "CUDA driver error");
  }
}

void ensure_current_context(int device) {
  CUcontext context = nullptr;
  check_cuda(cuCtxGetCurrent(&context), "query CUDA context");
  // A tensor returned by the caching allocator need not initialize this host
  // thread's driver context. A device guard can also keep the same ordinal
  // without calling cudaSetDevice. Driver memory operations require a current
  // context even when the device's primary context already exists elsewhere.
  if (context == nullptr) {
    C10_CUDA_CHECK(cudaSetDevice(device));
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

std::tuple<pybind11::bytes, size_t, size_t> export_ipc(torch::Tensor tensor) {
  TORCH_CHECK(tensor.is_cuda() && tensor.numel() > 0,
              "CUDA IPC export requires a nonempty CUDA tensor");
  const c10::cuda::CUDAGuard guard(tensor.get_device());
  ensure_current_context(tensor.get_device());
  const auto pointer = reinterpret_cast<CUdeviceptr>(tensor.data_ptr());
  CUdeviceptr base = 0;
  size_t size = 0;
  check_cuda(cuMemGetAddressRange(&base, &size, pointer), "query IPC allocation bounds");
  CUipcMemHandle handle{};
  check_cuda(cuIpcGetMemHandle(&handle, base), "export CUDA IPC allocation");
  return {pybind11::bytes(reinterpret_cast<const char*>(&handle), sizeof(handle)),
          size, pointer - base};
}

struct IpcMapping {
  CUdeviceptr address = 0;
  int device = 0;

  ~IpcMapping() {
    // The public transfer lease holds this tensor through the last device
    // read. Closing the mapping precedes acknowledgement to the producer.
    if (address) {
      const c10::cuda::CUDAGuard guard(device);
      ensure_current_context(device);
      cuIpcCloseMemHandle(address);
    }
  }
};

torch::Tensor import_ipc(torch::Tensor prototype, const pybind11::bytes& opaque,
                         size_t allocation_bytes, size_t byte_offset,
                         const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& strides) {
  TORCH_CHECK(prototype.is_cuda() && !shape.empty() && shape.size() == strides.size(),
              "CUDA IPC import requires matching tensor shape and strides");
  const std::string bytes = opaque;
  TORCH_CHECK(bytes.size() == sizeof(CUipcMemHandle), "invalid CUDA IPC allocation handle");
  size_t extent = 1;
  for (size_t index = 0; index < shape.size(); ++index) {
    TORCH_CHECK(shape[index] > 0 && strides[index] >= 0,
                "CUDA IPC tensor extents and strides are invalid");
    const auto rows = static_cast<size_t>(shape[index] - 1);
    const auto stride = static_cast<size_t>(strides[index]);
    TORCH_CHECK(!rows || stride <= (std::numeric_limits<size_t>::max() - extent) / rows,
                "CUDA IPC tensor span overflows size_t");
    extent += rows * stride;
  }
  TORCH_CHECK(byte_offset % prototype.element_size() == 0 && byte_offset < allocation_bytes &&
                  extent <= (allocation_bytes - byte_offset) / prototype.element_size(),
              "CUDA IPC tensor exceeds its exported allocation");
  const auto device = prototype.get_device();
  const c10::cuda::CUDAGuard guard(device);
  ensure_current_context(device);
  auto mapping = std::make_shared<IpcMapping>();
  mapping->device = device;
  CUipcMemHandle handle{};
  std::memcpy(&handle, bytes.data(), sizeof(handle));
  check_cuda(cuIpcOpenMemHandle(&mapping->address, handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
              "import CUDA IPC allocation");
  CUdeviceptr allocation_base = 0;
  size_t mapped_bytes = 0;
  check_cuda(cuMemGetAddressRange(&allocation_base, &mapped_bytes, mapping->address),
              "query imported CUDA allocation bounds");
  TORCH_CHECK(allocation_base == mapping->address && allocation_bytes == mapped_bytes,
              "CUDA IPC allocation size disagrees with the exported handle");
  return at::for_blob(reinterpret_cast<void*>(mapping->address + byte_offset), shape)
      .strides(strides)
      .deleter([mapping](void*) {})
      .options(prototype.options())
      .target_device(c10::Device(c10::kCUDA, device))
      .make_tensor();
}

void copy_host_device(torch::Tensor destination, torch::Tensor source, uint64_t stream_handle) {
  TORCH_CHECK(source.is_cuda() != destination.is_cuda() &&
                  (source.is_cpu() || destination.is_cpu()),
              "host/device copy requires one CUDA tensor and one host tensor");
  TORCH_CHECK(source.sizes() == destination.sizes() && source.dtype() == destination.dtype(),
              "host/device copy cannot change tensor shape or dtype");
  const auto& host = source.is_cuda() ? destination : source;
  const auto& device_tensor = source.is_cuda() ? source : destination;
  TORCH_CHECK(host.is_pinned(), "asynchronous host/device copy requires pinned host storage");
  const c10::cuda::CUDAGuard guard(device_tensor.device());
  ensure_current_context(device_tensor.get_device());
  if (source.numel() == 0) {
    return;
  }

  // Combine dimensions contiguous in both views, then use one pitched DMA
  // dimension. Only the remaining outer coordinates need separate copies.
  // No device tensor is allocated to pack a strided source or destination.
  std::vector<int64_t> axes;
  for (int64_t axis = 0; axis < source.dim(); ++axis) {
    TORCH_CHECK(source.stride(axis) >= 0 && destination.stride(axis) >= 0,
                "host/device copy requires nonnegative strides");
    if (source.size(axis) > 1) {
      axes.push_back(axis);
    }
  }
  int64_t width = 1;
  bool combined = true;
  while (combined) {
    combined = false;
    for (auto it = axes.begin(); it != axes.end(); ++it) {
      if (source.stride(*it) == width && destination.stride(*it) == width) {
        width *= source.size(*it);
        axes.erase(it);
        combined = true;
        break;
      }
    }
  }
  const auto item_bytes = source.element_size();
  int max_pitch = 0;
  check_cuda(cuDeviceGetAttribute(&max_pitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH,
                                 device_tensor.get_device()),
             "query host/device copy pitch bound");
  int64_t row_axis = -1;
  for (const auto axis : axes) {
    if (source.stride(axis) >= width && destination.stride(axis) >= width &&
        source.stride(axis) <= max_pitch / item_bytes &&
        destination.stride(axis) <= max_pitch / item_bytes &&
        (row_axis < 0 || source.size(axis) > source.size(row_axis))) {
      row_axis = axis;
    }
  }
  std::vector<int64_t> outer_axes;
  int64_t copies = 1;
  for (const auto axis : axes) {
    if (axis != row_axis) {
      outer_axes.push_back(axis);
      copies *= source.size(axis);
    }
  }
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = source.is_cuda() ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
  copy.dstMemoryType = destination.is_cuda() ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
  copy.WidthInBytes = width * item_bytes;
  copy.Height = row_axis < 0 ? 1 : source.size(row_axis);
  copy.srcPitch = (row_axis < 0 ? width : source.stride(row_axis)) * item_bytes;
  copy.dstPitch = (row_axis < 0 ? width : destination.stride(row_axis)) * item_bytes;
  for (int64_t index = 0; index < copies; ++index) {
    int64_t position = index;
    int64_t source_offset = 0;
    int64_t destination_offset = 0;
    for (const auto axis : outer_axes) {
      const auto coordinate = position % source.size(axis);
      position /= source.size(axis);
      source_offset += coordinate * source.stride(axis);
      destination_offset += coordinate * destination.stride(axis);
    }
    const auto* source_address =
        static_cast<const char*>(source.data_ptr()) + source_offset * item_bytes;
    auto* destination_address =
        static_cast<char*>(destination.data_ptr()) + destination_offset * item_bytes;
    if (source.is_cuda()) {
      copy.srcDevice = reinterpret_cast<CUdeviceptr>(source_address);
      copy.dstHost = destination_address;
    } else {
      copy.srcHost = source_address;
      copy.dstDevice = reinterpret_cast<CUdeviceptr>(destination_address);
    }
    const auto stream = reinterpret_cast<CUstream>(stream_handle);
    if (row_axis >= 0) {
      check_cuda(cuMemcpy2DAsync(&copy, stream), "copy strided host/device tensor");
    } else if (source.is_cuda()) {
      check_cuda(cuMemcpyDtoHAsync(copy.dstHost, copy.srcDevice, copy.WidthInBytes, stream),
                 "copy contiguous device/host span");
    } else {
      check_cuda(cuMemcpyHtoDAsync(copy.dstDevice, copy.srcHost, copy.WidthInBytes, stream),
                 "copy contiguous host/device span");
    }
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, binding) {
  binding.def("allocation_granularity", &allocation_granularity);
  binding.def("export_ipc", &export_ipc);
  binding.def("import_ipc", &import_ipc);
  binding.def("copy_host_device", &copy_host_device);
  pybind11::class_<PeerAllocation>(binding, "PeerAllocation")
      .def(pybind11::init<torch::Tensor, std::vector<int64_t>>())
      .def("export_fd", &PeerAllocation::export_fd)
      .def("map_peers", &PeerAllocation::map_peers);
}
