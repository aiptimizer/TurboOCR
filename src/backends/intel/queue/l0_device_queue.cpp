// L0DeviceQueue / L0DeviceEvent — SYCL (oneAPI/DPC++, Level Zero) impl.
//
// TOOLCHAIN: `icpx -fsycl` (Intel oneAPI DPC++) + the Level Zero loader + an
// Intel GPU driver. NOT compilable on the dev Mac; guarded by
// TURBO_OCR_HAS_SYCL. The guarded-IN path is the authoritative implementation;
// the guarded-out stub exists only so the tree parses (and its `override` set is
// checked against the seam) under a plain host compiler.

#include "intel/queue/l0_device_queue.h"

#if defined(TURBO_OCR_HAS_SYCL)
#include <sycl/sycl.hpp>
#if defined(TURBO_OCR_HAS_SYCL_GRAPH)
#include <sycl/ext/oneapi/experimental/graph.hpp>
#endif
#endif

namespace turbo_ocr::intel {

#if defined(TURBO_OCR_HAS_SYCL)

// ---- L0DeviceEvent ---------------------------------------------------------

struct L0DeviceEvent::Impl {
  sycl::event ev; // default-constructed == already complete
};

L0DeviceEvent::L0DeviceEvent() : impl_(std::make_unique<Impl>()) {}
L0DeviceEvent::~L0DeviceEvent() = default;

backend::DeviceKind L0DeviceEvent::device() const noexcept {
  return backend::DeviceKind::L0;
}
void *L0DeviceEvent::native_handle() const noexcept {
  return const_cast<sycl::event *>(&impl_->ev);
}
void L0DeviceEvent::synchronize() { impl_->ev.wait(); }
bool L0DeviceEvent::query() const noexcept {
  return impl_->ev.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

// ---- L0DeviceQueue ---------------------------------------------------------

struct L0DeviceQueue::Impl {
  sycl::queue q;
  sycl::event last;   // tail of this lane, for record()
  bool batch = false; // a one-submit region is open

  explicit Impl(int device_id) {
    // IN-ORDER: submissions execute in program order, so the pipeline's stage
    // ordering (warp -> infer -> argmax) holds without per-call events, and the
    // L0 backend is free to coalesce consecutive appends into one command-list
    // submission. This is what makes begin/end_batch cheap here.
    const sycl::property_list props{sycl::property::queue::in_order()};
    if (device_id < 0) {
      q = sycl::queue(sycl::gpu_selector_v, props);
    } else {
      auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
      const auto idx = static_cast<std::size_t>(device_id);
      q = (idx < gpus.size()) ? sycl::queue(gpus[idx], props)
                              : sycl::queue(sycl::gpu_selector_v, props);
    }
  }
};

L0DeviceQueue::L0DeviceQueue(int device_id)
    : impl_(std::make_unique<Impl>(device_id)) {}
L0DeviceQueue::~L0DeviceQueue() = default;

backend::DeviceKind L0DeviceQueue::device() const noexcept {
  return backend::DeviceKind::L0;
}
void *L0DeviceQueue::native_handle() const noexcept { return &impl_->q; }

void L0DeviceQueue::note_submission(void *sycl_event) noexcept {
  if (sycl_event)
    impl_->last = *static_cast<sycl::event *>(sycl_event);
}

void L0DeviceQueue::record(backend::DeviceEvent &ev) {
  // In-order lane: the tail event completing implies every prior submission
  // completed, so snapshotting the tail is a faithful cudaEventRecord.
  static_cast<L0DeviceEvent &>(ev).impl()->ev = impl_->last;
}

void L0DeviceQueue::wait(const backend::DeviceEvent &ev) {
  const auto *e = static_cast<const L0DeviceEvent &>(ev).impl();
  // Device-side barrier — no host round-trip (cudaStreamWaitEvent analogue).
  impl_->last = impl_->q.ext_oneapi_submit_barrier({e->ev});
}

void L0DeviceQueue::synchronize() { impl_->q.wait(); }

std::unique_ptr<backend::DeviceEvent> L0DeviceQueue::make_event() {
  return std::make_unique<L0DeviceEvent>();
}

void L0DeviceQueue::begin_batch() {
  // Mark the region only. Deliberately NO flush/sync here: the in-order queue
  // is already free to coalesce the region's appends into one L0 command-list
  // submission, and forcing anything would defeat that. See the header for why
  // the Metal analogue needs an explicit command buffer and SYCL does not.
  impl_->batch = true;
}
void L0DeviceQueue::end_batch() {
  // Per the seam: commit, do NOT synchronize. On an in-order SYCL queue the
  // work is already submitted; the driver flushes on its own schedule and the
  // caller waits via an event or synchronize(). Calling q.wait() here would
  // turn every BatchScope into a host round-trip — the exact anti-pattern the
  // performance gate calls out.
  impl_->batch = false;
}
bool L0DeviceQueue::batch_open() const noexcept { return impl_->batch; }

#else // !TURBO_OCR_HAS_SYCL — parse-only stubs (no DPC++ toolchain).

struct L0DeviceEvent::Impl {};
L0DeviceEvent::L0DeviceEvent() : impl_(std::make_unique<Impl>()) {}
L0DeviceEvent::~L0DeviceEvent() = default;
backend::DeviceKind L0DeviceEvent::device() const noexcept {
  return backend::DeviceKind::L0;
}
void *L0DeviceEvent::native_handle() const noexcept { return nullptr; }
void L0DeviceEvent::synchronize() {}
bool L0DeviceEvent::query() const noexcept { return true; }

struct L0DeviceQueue::Impl {
  bool batch = false;
};
L0DeviceQueue::L0DeviceQueue(int) : impl_(std::make_unique<Impl>()) {}
L0DeviceQueue::~L0DeviceQueue() = default;
backend::DeviceKind L0DeviceQueue::device() const noexcept {
  return backend::DeviceKind::L0;
}
void *L0DeviceQueue::native_handle() const noexcept { return nullptr; }
void L0DeviceQueue::note_submission(void *) noexcept {}
void L0DeviceQueue::record(backend::DeviceEvent &) {}
void L0DeviceQueue::wait(const backend::DeviceEvent &) {}
void L0DeviceQueue::synchronize() {}
std::unique_ptr<backend::DeviceEvent> L0DeviceQueue::make_event() {
  return std::make_unique<L0DeviceEvent>();
}
void L0DeviceQueue::begin_batch() { impl_->batch = true; }
void L0DeviceQueue::end_batch() { impl_->batch = false; }
bool L0DeviceQueue::batch_open() const noexcept { return impl_->batch; }

#endif

} // namespace turbo_ocr::intel
