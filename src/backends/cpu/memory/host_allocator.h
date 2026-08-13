#pragma once

// HostAllocator — the CpuBackend IDeviceAllocator (backend/backend.h). On the
// host there is one address space, so "device" memory and "pinned host" staging
// memory are the SAME plain host allocation, and every H2D/D2H/D2D copy is a
// std::memcpy on the calling thread (the Host queue is synchronous, so the copy
// has completed by the time the call returns — matching copy_h2d's "immediate
// on Host" contract).
//
// A single shared instance is fine: the methods are stateless wrappers over the
// C++ allocator / memcpy. This is the CPU analogue of nvidia/memory/cuda_allocator.h.

#include <cstddef>

#include <opencv2/core.hpp>

#include "turbo_ocr/backend/backend.h"    // IDeviceAllocator, DeviceKind
#include "turbo_ocr/backend/image_view.h" // ImageView

namespace turbo_ocr::cpu {

class HostAllocator final : public backend::IDeviceAllocator {
public:
  [[nodiscard]] backend::DeviceKind device() const noexcept override {
    return backend::DeviceKind::Host;
  }

  [[nodiscard]] void *allocate(std::size_t bytes) override;
  void free(void *p) noexcept override;

  // Pinned/page-locked staging is a device-transfer optimization; on the host
  // it is identical to a plain allocation.
  [[nodiscard]] void *allocate_host(std::size_t bytes) override;
  void free_host(void *p) noexcept override;

  void copy_h2d(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
  void copy_d2h(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
  void copy_d2d(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
};

// Wrap a host BGR8 cv::Mat as a Host ImageView (zero-copy, no upload). The
// helper the plan asks the allocator to expose; the pixels stay owned by the
// cv::Mat, so keep it alive for the ImageView's lifetime.
[[nodiscard]] backend::ImageView image_view_from_mat(const cv::Mat &m) noexcept;

} // namespace turbo_ocr::cpu
