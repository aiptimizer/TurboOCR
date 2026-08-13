#pragma once

// MetalAllocator — Apple IDeviceAllocator (backend.h). Allocates
// MTLResourceStorageModeShared buffers: on Apple silicon that is UNIFIED memory,
// so a buffer's .contents pointer is valid on both the CPU and the GPU and D2H/
// H2D "copies" are coherent memcpys with no PCIe transfer. Each allocation is
// registered (metal_common.h) so the bare .contents void* the Backend seam hands
// around can be resolved back to its id<MTLBuffer> when bound to an
// MPSGraphTensorData or a Metal encoder.
//
// A single shared allocator per device is fine (Metal buffer creation is
// thread-safe); the pipeline holds it via shared_ptr (Backend::allocator()).

#include <cstddef>
#include <memory>

#include "turbo_ocr/backend/backend.h" // IDeviceAllocator, DeviceBuffer

namespace turbo_ocr::apple {

class MetalAllocator final : public backend::IDeviceAllocator {
public:
  MetalAllocator() = default;
  ~MetalAllocator() override = default;

  [[nodiscard]] backend::DeviceKind device() const noexcept override {
    return backend::DeviceKind::Metal;
  }

  [[nodiscard]] void *allocate(std::size_t bytes) override;
  void free(void *p) noexcept override;

  [[nodiscard]] void *allocate_host(std::size_t bytes) override;
  void free_host(void *p) noexcept override;

  // Unified memory => coherent memcpy. Ordered on `queue` only in the sense that
  // the caller stages inputs before enqueuing GPU reads / reads outputs after
  // synchronize(); no blit encoder is needed for Shared buffers.
  void copy_h2d(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
  void copy_d2h(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
  void copy_d2d(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
};

// Shared per-process allocator (one GPU on this Mac).
std::shared_ptr<MetalAllocator> shared_allocator();

} // namespace turbo_ocr::apple
