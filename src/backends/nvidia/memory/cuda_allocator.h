#pragma once

// CudaAllocator — the NVIDIA IDeviceAllocator (backend/backend.h). A direct
// wrap of cudaMalloc / cudaFree / cudaMallocHost / cudaFreeHost and the
// cudaMemcpyAsync family, i.e. exactly what CudaPtr / CudaHostPtr
// (common/cuda/cuda_ptr.h) do today — surfaced behind the interface so the ONE
// OcrPipeline can allocate device buffers without naming CUDA.
//
// A single shared instance per device is fine (all methods are stateless
// wrappers over the CUDA runtime, which is itself thread-safe). Copies are
// ORDERED on the passed DeviceQueue's stream, so they interleave with kernels
// exactly as the existing hand-written cudaMemcpyAsync calls do.

#include <cstddef>

#include "turbo_ocr/backend/backend.h" // IDeviceAllocator, DeviceKind

namespace turbo_ocr::nvidia {

class CudaAllocator final : public backend::IDeviceAllocator {
public:
  [[nodiscard]] backend::DeviceKind device() const noexcept override {
    return backend::DeviceKind::Cuda;
  }

  [[nodiscard]] void *allocate(std::size_t bytes) override;
  void free(void *p) noexcept override;

  [[nodiscard]] void *allocate_host(std::size_t bytes) override; // pinned
  void free_host(void *p) noexcept override;

  void copy_h2d(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
  void copy_d2h(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
  void copy_d2d(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
};

} // namespace turbo_ocr::nvidia
