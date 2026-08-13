#pragma once

// HipAllocator — backend::IDeviceAllocator over hipMalloc / hipHostMalloc /
// hipMemcpyAsync. One shared instance per device (RocmBackend holds a shared_ptr).
// All device buffers the pipeline uses (stage scratch, batched input tensors,
// CCL label maps, MIGraphX I/O) come from here, so a pointer's space is always
// DeviceKind::Hip and validated against this allocator.

#include <cstddef>

#include "turbo_ocr/backend/backend.h" // IDeviceAllocator, DeviceQueue, DeviceKind

namespace turbo_ocr::amd {

using backend::DeviceKind;
using backend::DeviceQueue;
using backend::IDeviceAllocator;

class HipAllocator final : public IDeviceAllocator {
public:
  explicit HipAllocator(int device_id = 0) : device_id_(device_id) {}

  [[nodiscard]] DeviceKind device() const noexcept override {
    return DeviceKind::Hip;
  }

  [[nodiscard]] void *allocate(std::size_t bytes) override;
  void free(void *p) noexcept override;

  // hipHostMalloc pinned staging (fast async H2D/D2H, matching the CUDA path's
  // CudaHostPtr pinned mirrors).
  [[nodiscard]] void *allocate_host(std::size_t bytes) override;
  void free_host(void *p) noexcept override;

  void copy_h2d(void *dst, const void *src, std::size_t bytes,
                DeviceQueue &queue) override;
  void copy_d2h(void *dst, const void *src, std::size_t bytes,
                DeviceQueue &queue) override;
  void copy_d2d(void *dst, const void *src, std::size_t bytes,
                DeviceQueue &queue) override;

  [[nodiscard]] int device_id() const noexcept { return device_id_; }

private:
  int device_id_ = 0;
};

} // namespace turbo_ocr::amd
