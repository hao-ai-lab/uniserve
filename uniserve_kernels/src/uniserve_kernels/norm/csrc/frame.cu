// Per-frame group normalization, SiLU and causal padding in two kernels.
//
// A causal video convolution of a pre-activation block reads
// pad(silu(group_norm(frame))) for every frame of [batch, channels, frames,
// height, width] values. PyTorch evaluates that as a frame-folding copy, its
// group-norm moments and apply kernels, a SiLU kernel and the padding, five
// passes over the activation. Here `frame_moments` computes the moments of
// every (frame, group) straight from the values, and `frame_norm_pad` writes
// the padded, normalized and activated input in one pass.
//
// Both reproduce PyTorch 2.13's CUDA arithmetic exactly, so the result is
// bit for bit the composition's: the moments kernel is PyTorch's
// RowwiseMomentsCUDAKernel (aten/src/ATen/native/cuda/group_norm_kernel.cu)
// with the same Welford operator, block reduction and launch width, visiting
// each row's elements in the folded order; the apply kernel forms PyTorch's
// fused parameters (ComputeFusedParamsCUDAKernel), its `a * x + b`, and
// SiLU's `x / (1 + exp(-x))` (ActivationSiluKernel.cu) with the same
// expressions. The extension compiles without fast math, as PyTorch does.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/extension.h>

#include <cstdint>
#include <utility>

namespace {

using at::native::WelfordData;
using at::native::WelfordOps;

// Elements each thread of the moments kernel loads ahead of reducing them.
constexpr int kMomentLoads = 8;

// Mean and reciprocal standard deviation of row i = frame * groups + group:
// the group's `channels_per_group * pixels` values of one frame, visited in
// the order of the frame-folded [frames, channels, pixels] tensor PyTorch
// normalizes. A frame's pixels of one channel are contiguous.
__global__ void frame_moments_kernel(
    const float* values,
    int64_t stride_batch,
    int64_t stride_channel,
    int64_t stride_frame,
    int64_t frames,
    int64_t pixels,
    int64_t channels_per_group,
    int64_t groups,
    float eps,
    float* mean,
    float* rstd) {
  using WelfordType = WelfordData<float, int64_t>;
  using WelfordOp = WelfordOps<float, float, int64_t, std::pair<float, float>>;

  const int64_t i = blockIdx.x;
  const int64_t folded = i / groups;
  const int64_t group = i - folded * groups;
  const int64_t sample = folded / frames;
  const int64_t frame = folded - sample * frames;
  const float* row = values + sample * stride_batch + frame * stride_frame +
      group * channels_per_group * stride_channel;
  const int64_t N = channels_per_group * pixels;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  // Element j of the row is pixel j % pixels of channel j / pixels; both
  // advance by a fixed stride between a thread's elements, so they are
  // stepped rather than divided out. A thread loads `kMomentLoads` of its
  // elements before reducing them in order, keeping several loads in flight
  // without changing the order of its Welford updates.
  const int64_t step = blockDim.x;
  int64_t channel = threadIdx.x / pixels;
  int64_t pixel = threadIdx.x - channel * pixels;
  const int64_t channel_step = step / pixels;
  const int64_t pixel_step = step - channel_step * pixels;
  for (int64_t j0 = threadIdx.x; j0 < N; j0 += step * kMomentLoads) {
    float loaded[kMomentLoads];
#pragma unroll
    for (int u = 0; u < kMomentLoads; ++u) {
      if (j0 + u * step < N) {
        loaded[u] = static_cast<float>(row[channel * stride_channel + pixel]);
      }
      channel += channel_step;
      pixel += pixel_step;
      if (pixel >= pixels) {
        pixel -= pixels;
        channel += 1;
      }
    }
#pragma unroll
    for (int u = 0; u < kMomentLoads; ++u) {
      const int64_t j = j0 + u * step;
      if (j < N) {
        val = welford_op.reduce(val, loaded[u], i * N + j);
      }
    }
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    val = at::native::cuda_utils::WarpReduce(val, welford_op);
  } else {
    alignas(WelfordType) __shared__ char
        val_shared[sizeof(WelfordType) * C10_WARP_SIZE_UPPER_BOUND];
    WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
    val = at::native::cuda_utils::BlockReduce(
        val, welford_op, WelfordType(0, 0, 0, 0), val_shared_ptr);
  }
  if (threadIdx.x == 0) {
    auto [m2, m1] = welford_op.project(val);
    mean[i] = m1;
    rstd[i] = c10::cuda::compat::rsqrt(m2 + eps);
  }
}

// One output row segment: `kPixels` output columns by `kChannels` channels.
constexpr int kPixels = 64;
constexpr int kChannels = 32;
constexpr int kRows = 8;
constexpr int kThreads = 32 * kRows;

__device__ __forceinline__ int64_t source_index(int64_t index, int64_t extent, bool reflect) {
  if (reflect) {
    index = index < 0 ? -index : index;
    return index >= extent ? 2 * extent - 2 - index : index;
  }
  return index < 0 ? 0 : (index >= extent ? extent - 1 : index);
}

// How a frame's values normalize, following the branch PyTorch's
// GroupNormKernelImplInternal takes for the parameters and frame size.
enum Normalization : int {
  // ComputeFusedParamsCUDAKernel's a = rstd * gamma and b = -a * mean + beta,
  // then the apply's `a * x + b`; per channel (first, second) = (a, b).
  kFused = 0,
  // No affine parameters: `(x - mean) * rstd`; per channel (mean, rstd).
  kCentered = 1,
  // One-pixel frames (GroupNorm1dForward); per channel (mean, rstd).
  kSinglePixel = 2,
};

// GroupNorm1dForward's expressions for one-pixel frames.
__device__ __forceinline__ float normalize_single(
    float x, float mean, float rstd, const float* weight, const float* bias, int channel) {
  if (weight != nullptr && bias != nullptr) {
    return (x - mean) * rstd * weight[channel] + bias[channel];
  }
  if (weight != nullptr) {
    return (x - mean) * rstd * weight[channel];
  }
  if (bias != nullptr) {
    return (x - mean) * rstd + bias[channel];
  }
  return (x - mean) * rstd;
}

// Writes pad(silu(group_norm(values))) for one (sample, output frame, output
// row) and a run of `kPixels` output columns, every channel in blocks of
// `kChannels`. Output frames before `front` are zero; every other element
// normalizes and activates its reflected or replicated source pixel with
// its channel's parameters, which the block forms once for its frame.
// Channels-last output is transposed through shared memory so both the
// column-major reads and the channel-major writes coalesce.
template <bool kChannelsLast>
__global__ void __launch_bounds__(kThreads) frame_norm_pad_kernel(
    const float* values,
    int64_t stride_batch,
    int64_t stride_channel,
    int64_t stride_frame,
    const float* mean,
    const float* rstd,
    const float* weight,
    const float* bias,
    float* out,
    int64_t out_batch,
    int64_t out_channel,
    int64_t out_frame_stride,
    int64_t out_row,
    int64_t out_column,
    int channels,
    int frames,
    int height,
    int width,
    int out_frames,
    int out_height,
    int out_width,
    int channels_per_group,
    int groups,
    int top,
    int left,
    int front,
    bool reflect,
    int normalization) {
  extern __shared__ float parameters[];  // [2, channels]
  __shared__ float tile[kChannels][kPixels + 1];

  const int row_id = blockIdx.x;
  const int row = row_id % out_height;
  const int out_frame = (row_id / out_height) % out_frames;
  const int sample = row_id / (out_height * out_frames);
  const int column0 = blockIdx.y * kPixels;
  const int frame = out_frame - front;
  const int thread = threadIdx.y * 32 + threadIdx.x;
  float* destination = out + sample * out_batch + out_frame * out_frame_stride + row * out_row;

  if (frame < 0) {
    // A leading zero frame of the causal padding.
    for (int index = thread; index < channels * kPixels; index += kThreads) {
      const int channel = kChannelsLast ? index % channels : index / kPixels;
      const int column = column0 + (kChannelsLast ? index / channels : index % kPixels);
      if (column < out_width) {
        destination[channel * out_channel + column * out_column] = 0.0f;
      }
    }
    return;
  }

  const int group_base = (sample * frames + frame) * groups;
  for (int channel = thread; channel < channels; channel += kThreads) {
    const int ng = group_base + channel / channels_per_group;
    if (normalization == kFused) {
      const float scale = weight == nullptr ? rstd[ng] : rstd[ng] * weight[channel];
      const float shift = -scale * mean[ng] + (bias == nullptr ? 0 : bias[channel]);
      parameters[channel] = scale;
      parameters[channels + channel] = shift;
    } else {
      parameters[channel] = mean[ng];
      parameters[channels + channel] = rstd[ng];
    }
  }
  __syncthreads();

  // Each thread reads columns threadIdx.x and threadIdx.x + 32 of the source
  // row; a channel's pixels are contiguous within a frame.
  const int source_row = static_cast<int>(source_index(row - top, height, reflect));
  const float* source = values + sample * stride_batch + frame * stride_frame +
      static_cast<int64_t>(source_row) * width;
  int source_columns[kPixels / 32];
  bool live[kPixels / 32];
#pragma unroll
  for (int k = 0; k < kPixels / 32; ++k) {
    const int column = column0 + threadIdx.x + 32 * k;
    live[k] = column < out_width;
    source_columns[k] = static_cast<int>(source_index(column - left, width, reflect));
  }

  for (int channel0 = 0; channel0 < channels; channel0 += kChannels) {
    // All of a thread's loads of this channel block are issued before any
    // is used, so they are in flight together.
    float loaded[kChannels / kRows][kPixels / 32];
#pragma unroll
    for (int r = 0; r < kChannels / kRows; ++r) {
      const int channel = channel0 + threadIdx.y + r * kRows;
#pragma unroll
      for (int k = 0; k < kPixels / 32; ++k) {
        if (channel < channels && live[k]) {
          loaded[r][k] = source[channel * stride_channel + source_columns[k]];
        }
      }
    }
#pragma unroll
    for (int r = 0; r < kChannels / kRows; ++r) {
      const int c = threadIdx.y + r * kRows;
      const int channel = channel0 + c;
      if (channel >= channels) {
        continue;
      }
      const float first = parameters[channel];
      const float second = parameters[channels + channel];
#pragma unroll
      for (int k = 0; k < kPixels / 32; ++k) {
        if (!live[k]) {
          continue;
        }
        const float x = loaded[r][k];
        float y;
        if (normalization == kFused) {
          y = first * x + second;
        } else if (normalization == kCentered) {
          y = (x - first) * second;
        } else {
          y = normalize_single(x, first, second, weight, bias, channel);
        }
        // SiLU (ActivationSiluKernel.cu).
        const float value = y / (1.0f + ::exp(-y));
        const int column_offset = threadIdx.x + 32 * k;
        if constexpr (kChannelsLast) {
          tile[c][column_offset] = value;
        } else {
          destination[channel * out_channel + (column0 + column_offset) * out_column] = value;
        }
      }
    }
    if constexpr (kChannelsLast) {
      __syncthreads();
      // Writes run along channels, the innermost storage axis.
      const int channel = channel0 + threadIdx.x;
      for (int p = threadIdx.y; p < kPixels; p += kRows) {
        const int column = column0 + p;
        if (column < out_width && channel < channels) {
          destination[column * out_column + channel * out_channel] = tile[threadIdx.x][p];
        }
      }
      __syncthreads();
    }
  }
}

void check_values(const torch::Tensor& values) {
  TORCH_CHECK(values.is_cuda() && values.scalar_type() == torch::kFloat32 && values.dim() == 5,
              "frame normalization takes CUDA float32 [batch, channels, frames, height, width]");
  TORCH_CHECK(values.stride(4) == 1 && values.stride(3) == values.size(4),
              "frame normalization reads contiguous rows of height * width pixels");
}

}  // namespace

// Returns [batch * frames, groups] float32 mean and reciprocal standard
// deviation of every frame's channel groups.
std::vector<torch::Tensor> frame_moments(const torch::Tensor& values, int64_t groups, double eps) {
  check_values(values);
  const int64_t batch = values.size(0), channels = values.size(1), frames = values.size(2);
  const int64_t pixels = values.size(3) * values.size(4);
  TORCH_CHECK(groups > 0 && channels % groups == 0, "channels must divide into groups");
  const c10::cuda::CUDAGuard guard(values.device());
  auto mean = torch::empty({batch * frames, groups}, values.options());
  auto rstd = torch::empty({batch * frames, groups}, values.options());
  const int64_t channels_per_group = channels / groups;
  // PyTorch's launch width, which fixes the reduction tree.
  const int64_t threads = channels_per_group * pixels < at::native::cuda_utils::kCUDABlockReduceNumThreads
      ? at::cuda::warp_size()
      : at::native::cuda_utils::kCUDABlockReduceNumThreads;
  if (batch * frames * groups > 0) {
    frame_moments_kernel<<<batch * frames * groups, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        values.data_ptr<float>(), values.stride(0), values.stride(1), values.stride(2), frames, pixels,
        channels_per_group, groups, static_cast<float>(eps), mean.data_ptr<float>(), rstd.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {mean, rstd};
}

// Writes pad(silu(group_norm(values))) into `out`, contiguous or
// channels-last [batch, channels, frames + front, height + top + bottom,
// width + left + right] float32 storage.
void frame_norm_pad(
    const torch::Tensor& values,
    const torch::Tensor& mean,
    const torch::Tensor& rstd,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    torch::Tensor& out,
    int64_t groups,
    int64_t top,
    int64_t left,
    int64_t front,
    bool reflect) {
  check_values(values);
  const int64_t batch = values.size(0), channels = values.size(1), frames = values.size(2);
  const int64_t height = values.size(3), width = values.size(4);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.dim() == 5 &&
                  out.size(0) == batch && out.size(1) == channels,
              "the padded output must be float32 with the values' batch and channels");
  const bool channels_last = !out.is_contiguous() && out.is_contiguous(at::MemoryFormat::ChannelsLast3d);
  TORCH_CHECK(out.is_contiguous() || channels_last, "the padded output is neither contiguous nor channels-last");
  for (const auto& parameter : {weight, bias}) {
    TORCH_CHECK(!parameter.has_value() || (parameter->is_contiguous() && parameter->numel() == channels &&
                                           parameter->scalar_type() == torch::kFloat32),
                "normalization parameters must be float32 per channel");
  }
  const int64_t out_frames = out.size(2), out_height = out.size(3), out_width = out.size(4);
  const c10::cuda::CUDAGuard guard(values.device());
  TORCH_CHECK(batch * out_frames * out_height < (int64_t{1} << 31) && channels < (int64_t{1} << 31),
              "frame normalization addresses rows and channels with 32-bit indices");
  const dim3 grid(batch * out_frames * out_height, (out_width + kPixels - 1) / kPixels);
  const dim3 block(32, kRows);
  if (grid.x == 0 || grid.y == 0) {
    return;
  }
  // The branch PyTorch's group norm takes for these parameters and frames.
  const int normalization = height * width == 1
      ? kSinglePixel
      : (!weight.has_value() && !bias.has_value() ? kCentered : kFused);
  const size_t parameter_bytes = 2 * channels * sizeof(float);
  auto launch = [&](auto kernel) {
    if (parameter_bytes > 48 * 1024) {
      AT_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(parameter_bytes)));
    }
    kernel<<<grid, block, parameter_bytes, at::cuda::getCurrentCUDAStream()>>>(
        values.data_ptr<float>(), values.stride(0), values.stride(1), values.stride(2), mean.data_ptr<float>(),
        rstd.data_ptr<float>(), weight.has_value() ? weight->data_ptr<float>() : nullptr,
        bias.has_value() ? bias->data_ptr<float>() : nullptr, out.data_ptr<float>(), out.stride(0), out.stride(1),
        out.stride(2), out.stride(3), out.stride(4), channels, frames, height, width, out_frames, out_height,
        out_width, channels / groups, groups, top, left, front, reflect, normalization);
  };
  if (channels_last) {
    launch(frame_norm_pad_kernel<true>);
  } else {
    launch(frame_norm_pad_kernel<false>);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("frame_moments", &frame_moments);
  module.def("frame_norm_pad", &frame_norm_pad);
}
