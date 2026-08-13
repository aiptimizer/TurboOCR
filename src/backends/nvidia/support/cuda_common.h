#pragma once

// cuda_common.h — small, header-only glue between the device-agnostic backend
// interfaces (include/turbo_ocr/backend/*) and the existing NVIDIA
// classes (include/turbo_ocr/**). NOTHING here is a re-implementation; it only
// translates the vocabulary at the seam:
//
//   backend::ImageView            <->  decode::GpuImage      (add/drop `kind`)
//   backend::DeviceQueue::native  ->   cudaStream_t          (opaque handle cast)
//   backend::DType                <->  nvinfer1::DataType / ORT i64 flag
//   backend::DeviceTensor.shape   <->  nvinfer1::Dims
//
// This header pulls in CUDA + TensorRT, so it (and every TU that includes it)
// compiles ONLY on a CUDA host. See README.md.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include "turbo_ocr/backend/device_queue.h" // backend::DeviceQueue, DeviceKind
#include "turbo_ocr/backend/engine.h"        // backend::DeviceTensor, DType
#include "turbo_ocr/backend/image_view.h"    // backend::ImageView
#include "nvidia/support/gpu_image.h"      // decode::GpuImage

namespace turbo_ocr::nvidia {

// --- ImageView <-> GpuImage -------------------------------------------------
// A GpuImage is byte-for-byte an ImageView minus `kind`; the NVIDIA space is
// always Cuda. These are the ONLY two conversions the stage adapters need.

[[nodiscard]] inline decode::GpuImage to_gpu_image(const backend::ImageView &v) noexcept {
  return decode::GpuImage{.data = v.data, .step = v.step, .rows = v.rows, .cols = v.cols};
}

[[nodiscard]] inline backend::ImageView to_image_view(const decode::GpuImage &g) noexcept {
  return backend::ImageView{.data = g.data,
                            .step = g.step,
                            .rows = g.rows,
                            .cols = g.cols,
                            .kind = backend::DeviceKind::Cuda};
}

// --- DeviceQueue -> cudaStream_t --------------------------------------------
// A CUDA DeviceQueue's native_handle() is exactly its cudaStream_t (see
// cuda_device_queue.h). A null queue handle degrades to the default stream 0,
// matching every existing signature's `cudaStream_t stream = 0` default.

[[nodiscard]] inline cudaStream_t cuda_stream(backend::DeviceQueue &q) noexcept {
  return static_cast<cudaStream_t>(q.native_handle());
}

// --- DType helpers ----------------------------------------------------------

[[nodiscard]] inline bool is_i64(backend::DType d) noexcept {
  return d == backend::DType::I64;
}

[[nodiscard]] inline nvinfer1::Dims to_trt_dims(const std::vector<int64_t> &shape) noexcept {
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int32_t>(shape.size());
  for (int i = 0; i < d.nbDims && i < nvinfer1::Dims::MAX_DIMS; ++i)
    d.d[i] = static_cast<int64_t>(shape[static_cast<std::size_t>(i)]);
  return d;
}

[[nodiscard]] inline std::vector<int64_t> from_trt_dims(const nvinfer1::Dims &d) {
  std::vector<int64_t> out;
  out.reserve(static_cast<std::size_t>(d.nbDims));
  for (int i = 0; i < d.nbDims; ++i)
    out.push_back(d.d[i]);
  return out;
}

// A Cuda-space tensor guard used by the engine adapters before binding a
// caller pointer (mirrors the audit's "validate the pointer against your own
// allocator/MemoryInfo" rule). Cheap: only checks the declared space.
[[nodiscard]] inline bool is_cuda_space(const backend::DeviceTensor &t) noexcept {
  return t.space == backend::DeviceKind::Cuda;
}

} // namespace turbo_ocr::nvidia
