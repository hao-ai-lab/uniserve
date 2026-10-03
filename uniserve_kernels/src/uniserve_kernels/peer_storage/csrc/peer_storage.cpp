// SPDX-License-Identifier: Apache-2.0

// CUDA driver virtual memory primitives for sharing tensor storage across
// processes, plus strided asynchronous copies between pinned host and device
// memory. The Python package uniserve_kernels.peer_storage compiles this file
// on first use and wraps each binding.
//
// PeerAllocation owns one physical allocation created with cuMemCreate and
// exports its shareable handle. PeerMapping owns one reserved virtual address
// range, the segments mapped into it, and the allocation handles that keep the
// mapped physical memory alive; every tensor this file returns holds its
// PeerMapping through the tensor deleter, so the mapping lives exactly as long
// as the tensor's storage. Handle transport between processes, grants, and
// retirement ordering belong to the Python callers.

#include <torch/extension.h>
#include <ATen/core/CachingHostAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <tuple>
#include <vector>

namespace {

// Raises a TORCH_CHECK failure (RuntimeError in Python) naming the operation
// when a driver call fails.
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
  // without calling cudaSetDevice. Driver storage operations require a current
  // context even when the device's primary context already exists elsewhere.
  if (context == nullptr) {
    C10_CUDA_CHECK(cudaSetDevice(device));
  }
}

// A device's shareable handle type, probed once and cached.
//
// A fabric handle is importable from another host inside the fabric domain; a
// descriptor handle reaches only processes on this one. Fabric is probed first
// so an instance that can span hosts does, and the result is cached because
// the probe allocates to establish it: granularity alone does not say whether
// the driver will export the type.
CUmemAllocationHandleType shareable_handle_type(int device) {
  static std::mutex probe_mutex;
  static std::unordered_map<int, CUmemAllocationHandleType> probed;

  const std::lock_guard<std::mutex> lock(probe_mutex);
  const auto cached = probed.find(device);
  if (cached != probed.end()) {
    return cached->second;
  }

  const c10::cuda::CUDAGuard guard(device);
  CUmemAllocationProp properties{};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = device;

  auto usable = [&](CUmemAllocationHandleType type) {
    properties.requestedHandleTypes = type;
    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &properties,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
            CUDA_SUCCESS ||
        granularity == 0) {
      return false;
    }
    CUmemGenericAllocationHandle handle = 0;
    if (cuMemCreate(&handle, granularity, &properties, 0) != CUDA_SUCCESS) {
      return false;
    }
    cuMemRelease(handle);
    return true;
  };

  const auto selected = usable(CU_MEM_HANDLE_TYPE_FABRIC)
                            ? CU_MEM_HANDLE_TYPE_FABRIC
                            : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  probed.emplace(device, selected);
  return selected;
}

// Bytes of one exported handle of the given type.
//
// A descriptor is an int; a fabric handle is an opaque struct. Both travel as
// bytes so one publication shape carries either.
size_t handle_bytes(CUmemAllocationHandleType type) {
  return type == CU_MEM_HANDLE_TYPE_FABRIC ? sizeof(CUmemFabricHandle)
                                           : sizeof(int);
}

// Reports whether this device's probed handle type is importable from
// another host, which is what decides if a transfer edge may cross one.
bool exports_fabric_handles(int device) {
  return shareable_handle_type(device) == CU_MEM_HANDLE_TYPE_FABRIC;
}

// Properties of a device-resident CU_MEM_ALLOCATION_TYPE_PINNED allocation
// requesting the device's probed handle type, shared by PeerAllocation and
// allocation_granularity.
CUmemAllocationProp allocation_properties(int device) {
  CUmemAllocationProp properties{};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = device;
  properties.requestedHandleTypes = shareable_handle_type(device);
  return properties;
}

// Exports one allocation's shareable handle as bytes.
std::string export_handle_bytes(CUmemGenericAllocationHandle handle, int device) {
  const auto type = shareable_handle_type(device);
  std::string exported(handle_bytes(type), '\0');
  check_cuda(cuMemExportToShareableHandle(exported.data(), handle, type, 0),
             "export peer allocation handle");
  return exported;
}

// Imports one allocation from the bytes a producing rank exported.
CUmemGenericAllocationHandle import_handle_bytes(const std::string& exported,
                                                 int device) {
  const auto type = shareable_handle_type(device);
  TORCH_CHECK(exported.size() == handle_bytes(type),
              "peer allocation handle does not match this device's handle type");
  CUmemGenericAllocationHandle handle = 0;
  // The two handle types reach the driver differently. A fabric handle is an
  // opaque structure the driver reads through a pointer, while a POSIX
  // descriptor is the operating system handle itself and travels by value.
  // Export writes both into the same byte buffer, so the descriptor has to be
  // read back out of it here.
  void* os_handle = const_cast<char*>(exported.data());
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    int descriptor = 0;
    std::memcpy(&descriptor, exported.data(), sizeof(descriptor));
    os_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(descriptor));
  }
  check_cuda(cuMemImportFromShareableHandle(&handle, os_handle, type),
             "import peer allocation handle");
  return handle;
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

// One virtual address range holding `mapped_segments` mapped segments of
// `segment_bytes` each, plus the allocation handles that keep their physical
// memory alive. The destructor also unwinds a partially built mapping: it
// unmaps only the segments counted as mapped, frees the range only when one was
// reserved, and releases every handle collected so far. Driver errors during
// teardown are ignored.
struct PeerMapping {
  CUdeviceptr address = 0;
  size_t total_bytes = 0;
  size_t segment_bytes = 0;
  size_t mapped_segments = 0;
  // Composite mappings contain segments of unequal size. Ordinary peer
  // mappings retain their uniform segment description above.
  std::vector<std::pair<CUdeviceptr, size_t>> ranges;
  int device = 0;
  std::vector<CUmemGenericAllocationHandle> handles;

  ~PeerMapping() {
    // Teardown issues no synchronization of its own: the owner must drop the
    // last tensor reference only after every stream and graph reading it has
    // completed. Each handle keeps its physical allocation alive until it is
    // released, which happens after the range is unmapped and freed.
    const c10::cuda::CUDAGuard guard(device);
    for (size_t index = 0; index < mapped_segments; ++index) {
      cuMemUnmap(address + index * segment_bytes, segment_bytes);
    }
    for (const auto& [start, length] : ranges) {
      cuMemUnmap(start, length);
    }
    if (address) {
      cuMemAddressFree(address, total_bytes);
    }
    for (const auto handle : handles) {
      cuMemRelease(handle);
    }
  }
};

// Concatenate physical allocations in virtual space, without copying bytes.
// Each input exposes one complete allocation, starting at its beginning.
// Fabric handles do not support partial mappings. Retaining the CUDA handle
// lets the result outlive the input tensor's original virtual mapping.
torch::Tensor map_segments(const std::vector<torch::Tensor>& parts) {
  TORCH_CHECK(!parts.empty(), "composite mapping needs at least one segment");
  TORCH_CHECK(parts.front().is_cuda(), "composite mapping requires CUDA allocations");
  const auto device = parts.front().get_device();
  const c10::cuda::CUDAGuard guard(device);
  const auto granularity = allocation_granularity(device);
  auto mapping = std::make_shared<PeerMapping>();
  mapping->device = device;
  for (const auto& part : parts) {
    const auto size = part.nbytes();
    TORCH_CHECK(part.is_cuda() && part.get_device() == device &&
                    part.scalar_type() == at::kByte && part.is_contiguous() &&
                    size > 0 && size % granularity == 0 &&
                    reinterpret_cast<uintptr_t>(part.data_ptr()) % granularity == 0,
                "composite segments must be page-aligned CUDA byte spans on one device");
    TORCH_CHECK(mapping->total_bytes <= std::numeric_limits<size_t>::max() - size,
                "composite mapping size exceeds size_t");
    mapping->total_bytes += size;
    CUmemGenericAllocationHandle handle;
    check_cuda(cuMemRetainAllocationHandle(&handle, part.data_ptr()),
               "retain composite segment allocation");
    mapping->handles.push_back(handle);
  }
  check_cuda(cuMemAddressReserve(&mapping->address, mapping->total_bytes, 0, 0, 0),
             "reserve composite tensor address range");
  auto address = mapping->address;
  for (size_t index = 0; index < parts.size(); ++index) {
    const auto length = parts[index].nbytes();
    check_cuda(cuMemMap(address, length, 0, mapping->handles[index], 0),
               "map composite tensor segment");
    mapping->ranges.emplace_back(address, length);
    address += length;
  }
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  check_cuda(cuMemSetAccess(mapping->address, mapping->total_bytes, &access, 1),
             "enable composite tensor access");
  return at::for_blob(reinterpret_cast<void*>(mapping->address),
                     {static_cast<int64_t>(mapping->total_bytes)})
      .deleter([mapping](void*) {})
      .options(parts.front().options())
      .target_device(c10::Device(c10::kCUDA, device))
      .make_tensor();
}

// One physical allocation of `shape` elements of the prototype's dtype on the
// prototype's device. The constructor requires the byte size to be an exact
// multiple of the allocation granularity rather than rounding it up. The object
// holds one reference to the allocation; mappings it creates hold their own,
// so they outlive it.
class PeerAllocation : public std::enable_shared_from_this<PeerAllocation> {
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

  pybind11::bytes export_handle() const {
    const c10::cuda::CUDAGuard guard(device_);
    // A shareable handle is opaque bytes, not text, so it crosses to Python as
    // bytes rather than a string the binding would decode as UTF-8.
    return pybind11::bytes(export_handle_bytes(handle_, device_));
  }

  // Maps this allocation into a new address range as a tensor of `shape_`,
  // accessible from this device only.
  torch::Tensor map_local() const {
    const c10::cuda::CUDAGuard guard(device_);
    auto mapping = std::make_shared<PeerMapping>();
    mapping->device = device_;
    mapping->total_bytes = bytes_;
    mapping->segment_bytes = bytes_;
    check_cuda(cuMemAddressReserve(&mapping->address, bytes_, 0, 0, 0),
                "reserve shared tensor address range");
    check_cuda(cuMemMap(mapping->address, bytes_, 0, handle_, 0),
                "map shared tensor allocation");
    mapping->mapped_segments = 1;

    // The mapping retains the originating allocation handle rather than one
    // imported from its exported handle: CUDA cannot re-export an imported
    // handle, and export_handle() on this tensor must reach another reader.
    // The retained reference also keeps the storage alive after this
    // PeerAllocation is destroyed.
    CUmemGenericAllocationHandle retained;
    check_cuda(cuMemRetainAllocationHandle(
                   &retained, reinterpret_cast<void*>(mapping->address)),
               "retain shared tensor allocation");
    mapping->handles.push_back(retained);

    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check_cuda(cuMemSetAccess(mapping->address, bytes_, &access, 1),
                "enable shared tensor access");
    return at::for_blob(reinterpret_cast<void*>(mapping->address), shape_)
        .deleter([mapping](void*) {})
        .options(options_)
        .target_device(c10::Device(c10::kCUDA, device_))
        .make_tensor();
  }

  // Imports one exported handle per owner and maps them back to back, in list
  // order, into one tensor whose leading extent is shape_[0] times the number
  // of handles. Each handle is mapped as one `bytes_` segment, so every owner
  // must allocate the same shape and dtype. Access is enabled for this device
  // only. A POSIX descriptor must stay open in this process until the call
  // returns.
  torch::Tensor map_peers(const std::vector<std::string>& descriptors) const {
    const c10::cuda::CUDAGuard guard(device_);
    TORCH_CHECK(!descriptors.empty() &&
                    bytes_ <= std::numeric_limits<size_t>::max() / descriptors.size() &&
                    shape_[0] <= std::numeric_limits<int64_t>::max() / descriptors.size(),
                "peer tensor requires a nonempty, representable owner extent");
    auto mapping = std::make_shared<PeerMapping>();
    mapping->device = device_;
    mapping->segment_bytes = bytes_;
    mapping->total_bytes = bytes_ * descriptors.size();
    for (const auto& exported : descriptors) {
      mapping->handles.push_back(import_handle_bytes(exported, device_));
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
    // Imported handles keep pages alive but do not keep the original fabric
    // export importable. A slower peer may still be importing that export
    // after this call returns, so retain its owner through tensor retirement.
    return at::for_blob(reinterpret_cast<void*>(mapping->address), shape)
        .deleter([mapping, owner = shared_from_this()](void*) {})
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

// Exports the allocation backing `tensor`'s storage as (handle bytes, storage
// bytes, byte offset of the tensor's first element within the storage), or
// nullopt when the driver rejects the storage base as a VMM allocation
// (CUDA_ERROR_INVALID_VALUE) or the allocation was not created with this
// device's probed handle type.
std::optional<std::tuple<pybind11::bytes, size_t, size_t>> export_handle(
    torch::Tensor tensor) {
  TORCH_CHECK(tensor.is_cuda() && tensor.numel() > 0,
              "shared allocation export requires a nonempty CUDA tensor");
  const c10::cuda::CUDAGuard guard(tensor.get_device());
  const auto base = tensor.storage().data_ptr().get();
  CUmemGenericAllocationHandle handle;
  const auto status = cuMemRetainAllocationHandle(&handle, base);
  // Ordinary caching-allocator storage needs an exportable materialization.
  // Driver failures other than an unsupported allocation remain observable.
  if (status == CUDA_ERROR_INVALID_VALUE) {
    return std::nullopt;
  }
  check_cuda(status, "retain shared allocation");
  try {
    CUmemAllocationProp properties{};
    check_cuda(cuMemGetAllocationPropertiesFromHandle(&properties, handle),
                "query shared allocation properties");
    const auto device = tensor.get_device();
    // Only allocations created with the probed type can export it.
    if (!(properties.requestedHandleTypes & shareable_handle_type(device))) {
      cuMemRelease(handle);
      return std::nullopt;
    }
    auto exported = export_handle_bytes(handle, device);
    cuMemRelease(handle);
    const auto offset = static_cast<const char*>(tensor.data_ptr()) -
        static_cast<const char*>(base);
    return std::make_tuple(pybind11::bytes(exported), tensor.storage().nbytes(),
                           offset);
  } catch (...) {
    cuMemRelease(handle);
    throw;
  }
}

// Maps the whole of one exported allocation as a flat tensor of the
// prototype's dtype on the prototype's device. The caller applies the exported
// byte offset.
torch::Tensor import_handle(torch::Tensor prototype, const std::string& exported,
                        size_t allocation_bytes) {
  TORCH_CHECK(prototype.is_cuda() && allocation_bytes > 0 &&
                  allocation_bytes % prototype.element_size() == 0,
              "shared allocation import requires a representable CUDA extent");
  const auto device = prototype.get_device();
  const c10::cuda::CUDAGuard guard(device);
  auto mapping = std::make_shared<PeerMapping>();
  mapping->device = device;
  mapping->total_bytes = allocation_bytes;
  mapping->segment_bytes = allocation_bytes;
  const auto imported = import_handle_bytes(exported, device);
  mapping->handles.push_back(imported);
  check_cuda(cuMemAddressReserve(&mapping->address, allocation_bytes, 0, 0, 0),
              "reserve shared allocation address range");
  check_cuda(cuMemMap(mapping->address, allocation_bytes, 0, imported, 0),
              "map shared allocation");
  mapping->mapped_segments = 1;
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  check_cuda(cuMemSetAccess(mapping->address, allocation_bytes, &access, 1),
              "enable shared allocation access");
  return at::for_blob(reinterpret_cast<void*>(mapping->address),
                     {static_cast<int64_t>(allocation_bytes / prototype.element_size())})
      .deleter([mapping](void*) {})
      .options(prototype.options())
      .target_device(c10::Device(c10::kCUDA, device))
      .make_tensor();
}

// Records `stream` on the PyTorch pinned host allocator block backing `tensor`,
// so the allocator does not reuse the block until work already enqueued on the
// stream completes. Fails unless `tensor` is pinned host storage from that
// allocator.
void record_host_usage(torch::Tensor tensor, int64_t device, uint64_t stream_handle) {
  TORCH_CHECK(tensor.is_cpu() && tensor.is_pinned(),
              "native DMA lifetime tracking requires pinned host storage");
  const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  const auto stream = c10::cuda::getStreamFromExternal(
      reinterpret_cast<cudaStream_t>(stream_handle), static_cast<c10::DeviceIndex>(device));
  const auto& allocation = tensor.storage().data_ptr();
  TORCH_CHECK(at::getHostAllocator(at::kCUDA)->record_event(
                  tensor.data_ptr(), allocation.get_context(), stream.unwrap()),
              "native DMA source must belong to the PyTorch pinned allocator");
}

// Enqueues `source` -> `destination` on the raw CUDA stream `stream_handle`
// with driver DMA calls, one per outer coordinate left after the layout is
// reduced to a contiguous span and at most one pitched row axis. The copy does
// not record either tensor for lifetime tracking.
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
  // Axes of extent one address nothing and are dropped.
  std::vector<int64_t> axes;
  for (int64_t axis = 0; axis < source.dim(); ++axis) {
    TORCH_CHECK(source.stride(axis) >= 0 && destination.stride(axis) >= 0,
                "host/device copy requires nonnegative strides");
    if (source.size(axis) > 1) {
      axes.push_back(axis);
    }
  }

  // Grow the contiguous span `width` (in elements) by absorbing any axis whose
  // stride equals the current width in both views, until none qualifies.
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

  // The pitched row axis must step at least one span in both views, since a
  // 2D copy's pitch cannot be smaller than its row width, and its byte pitch
  // must not exceed the device's maximum memcpy pitch. Among qualifying axes
  // the largest extent is chosen, which minimizes the number of copies. With no
  // qualifying axis each copy is one contiguous span.
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

  // Every remaining axis is iterated on the host, one DMA call per coordinate.
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
    // Decode `index` into outer coordinates, first outer axis fastest, and
    // turn them into element offsets in each view.
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
  binding.def("map_segments", &map_segments);
  binding.def("exports_fabric_handles", &exports_fabric_handles);
  binding.def("export_handle", &export_handle);
  binding.def("import_handle", &import_handle);
  binding.def("copy_host_device", &copy_host_device);
  binding.def("record_host_usage", &record_host_usage);
  pybind11::class_<PeerAllocation, std::shared_ptr<PeerAllocation>>(binding, "PeerAllocation")
      .def(pybind11::init<torch::Tensor, std::vector<int64_t>>())
      .def("export_handle", &PeerAllocation::export_handle)
      .def("map_local", &PeerAllocation::map_local)
      .def("map_peers", &PeerAllocation::map_peers);
}
