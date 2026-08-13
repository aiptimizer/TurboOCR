#pragma once

// L0Allocator — the Intel backend's IDeviceAllocator (backend.h), backed by
// SYCL USM allocations on a Level Zero context.
//
// This is what makes the Intel path device-resident: page image, warped crops,
// logits and argmax outputs live in `sycl::malloc_device` USM and the OpenVINO
// engine binds those SAME pointers as ov::RemoteTensor, so inference reads and
// writes them in place.
//
//   device()       -> DeviceKind::L0
//   allocate()     -> sycl::malloc_device        (USM device)
//   allocate_host()-> sycl::malloc_host          (pinned staging for H2D/D2H)
//   copy_*()       -> memcpy enqueued on the CALLER'S DeviceQueue lane, so a
//                     copy interleaves correctly with kernel/inference work
//                     (never on a private queue — that would silently reorder).
//
// One allocator instance is shared per device (a shared_ptr held by the
// backend), mirroring how the CUDA path shares one device context. It owns the
// sycl::context/device purely as the USM binding target and to hand OpenVINO the
// native L0 handles for a shared ov::RemoteContext.
//
// SIZE POLICY: allocations here are made at load()/warmup by the stages and then
// REUSED; nothing in this class is expected on the hot path (the performance
// gate forbids per-request device allocation).

#include <cstddef>
#include <memory>

#include "turbo_ocr/backend/backend.h" // IDeviceAllocator, DeviceKind

namespace turbo_ocr::intel {

class L0Allocator final : public backend::IDeviceAllocator {
public:
  explicit L0Allocator(int device_id = -1);
  ~L0Allocator() override;

  [[nodiscard]] backend::DeviceKind device() const noexcept override {
    return backend::DeviceKind::L0;
  }

  [[nodiscard]] void *allocate(std::size_t bytes) override;
  void free(void *p) noexcept override;

  [[nodiscard]] void *allocate_host(std::size_t bytes) override;
  void free_host(void *p) noexcept override;

  void copy_h2d(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
  void copy_d2h(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;
  void copy_d2d(void *dst, const void *src, std::size_t bytes,
                backend::DeviceQueue &queue) override;

  // True when a real USM/Level-Zero context was created (i.e. built with SYCL
  // and an Intel GPU was found). False => the host-malloc parse/degraded path;
  // the engine must then bind HOST tensors, which caps() reports honestly.
  [[nodiscard]] bool has_device() const noexcept;

  // Native Level Zero handles (ze_context_handle_t / ze_device_handle_t) so
  // OpenVINO can build an ov::RemoteContext over the SAME L0 context and accept
  // our USM pointers as RemoteTensors. Returned opaque; nullptr when absent.
  [[nodiscard]] void *native_l0_context() const noexcept;
  [[nodiscard]] void *native_l0_device() const noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace turbo_ocr::intel
