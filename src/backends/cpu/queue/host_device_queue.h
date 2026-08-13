#pragma once

// HostDeviceQueue / HostDeviceEvent — the CpuBackend implementation of the
// device-agnostic ordering primitives (backend/device_queue.h).
//
// The Host backend is SYNCHRONOUS: every engine/kernel/stage call runs to
// completion before it returns, so there is nothing to order and nothing to
// wait for. is_async() is false; record/wait/synchronize and begin/end_batch
// are no-ops; a HostDeviceEvent is always already-signalled (query()==true,
// synchronize() returns immediately). This is the degenerate lane the seam
// documents for DeviceKind::Host — it exists only so the ONE pipeline can hold
// a DeviceQueue& uniformly across every backend.

#include <memory>

#include "turbo_ocr/backend/device_queue.h"

namespace turbo_ocr::cpu {

class HostDeviceEvent final : public backend::DeviceEvent {
public:
  HostDeviceEvent() = default;
  ~HostDeviceEvent() override = default;

  [[nodiscard]] backend::DeviceKind device() const noexcept override {
    return backend::DeviceKind::Host;
  }
  // No native timeline handle on the host.
  [[nodiscard]] void *native_handle() const noexcept override {
    return nullptr;
  }
  // Work already completed synchronously, so waiting is instantaneous.
  void synchronize() override {}
  [[nodiscard]] bool query() const noexcept override { return true; }
};

class HostDeviceQueue final : public backend::DeviceQueue {
public:
  HostDeviceQueue() = default;
  ~HostDeviceQueue() override = default;

  [[nodiscard]] backend::DeviceKind device() const noexcept override {
    return backend::DeviceKind::Host;
  }
  [[nodiscard]] bool is_async() const noexcept override { return false; }
  [[nodiscard]] void *native_handle() const noexcept override {
    return nullptr;
  }

  // Ordering is meaningless on a synchronous lane — every op has already
  // finished by the time the next one is issued.
  void record(backend::DeviceEvent & /*ev*/) override {}
  void wait(const backend::DeviceEvent & /*ev*/) override {}
  void synchronize() override {}
  [[nodiscard]] std::unique_ptr<backend::DeviceEvent> make_event() override {
    return std::make_unique<HostDeviceEvent>();
  }

  // Batching is a residency lever for async command-buffer backends (Metal);
  // on the host there is nothing to group, so this only tracks the flag for
  // symmetry with BatchScope.
  void begin_batch() override { batch_open_ = true; }
  void end_batch() override { batch_open_ = false; }
  [[nodiscard]] bool batch_open() const noexcept override {
    return batch_open_;
  }

private:
  bool batch_open_ = false;
};

} // namespace turbo_ocr::cpu
