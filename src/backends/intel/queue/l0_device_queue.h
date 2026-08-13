#pragma once

// L0DeviceQueue / L0DeviceEvent — the Intel backend's DeviceQueue/DeviceEvent
// (include/turbo_ocr/backend/device_queue.h), backed by an IN-ORDER
// sycl::queue on the Level Zero backend (Intel iGPU / Arc / NPU-adjacent GPUs).
//
// Mapping to the seam contract:
//   device()        -> DeviceKind::L0
//   is_async()      -> true   (SYCL submissions are asynchronous)
//   native_handle() -> sycl::queue*  (down-cast by L0Allocator / SyclKernels /
//                      OpenVINOEngine; portable code never touches it)
//   record(ev)      -> snapshot this lane's tail sycl::event into `ev`
//   wait(ev)        -> q.ext_oneapi_submit_barrier({ev}) — a DEVICE-side wait,
//                      no host round-trip (the cudaStreamWaitEvent analogue)
//   synchronize()   -> q.wait()
//
// --- begin_batch()/end_batch(): the one-submit lever, and why it is (almost)
//     a no-op here -----------------------------------------------------------
// On Metal, BatchScope is load-bearing: it opens ONE MTLCommandBuffer that every
// encoder appends to, so a whole image's warp+rec+argmax costs one commit. The
// SYCL equivalent of "one submission" is NOT a manual command buffer: an
// in-order sycl::queue already
//   (a) guarantees program-order execution without any event choreography, and
//   (b) lets the Level Zero backend batch consecutive kernel appends into a
//       single command-list submission (the driver flushes on host sync, on a
//       barrier, or when its append budget fills).
// So the correct Intel implementation of the batch region is: *do not force a
// flush inside it*. That means begin_batch() only marks the region, and
// end_batch() must NOT call q.wait() (a flush there would destroy exactly the
// coalescing the seam is asking for) — which matches the seam's own wording:
// "end_batch() flushes/commits; it does NOT synchronize".
//
// The one thing that WOULD make this a true single submission is
// sycl_ext_oneapi_graph (`command_graph`): record every kernel in the region
// into a graph, finalize it once, and execute it per image. That is compiled in
// under TURBO_OCR_HAS_SYCL_GRAPH and is UNVALIDATED (no Intel hardware here) —
// see the bring-up checklist in README.md. The default path is the in-order
// queue, which is correct in both cases; the graph path is a throughput lever
// to measure on hardware, not a correctness requirement.
//
// The public header carries NO SYCL types (pImpl), so the device-neutral shared
// pipeline can hold a DeviceQueue without a DPC++ toolchain in scope.

#include <memory>

#include "turbo_ocr/backend/device_queue.h"

namespace turbo_ocr::intel {

// A recordable point in an L0DeviceQueue's timeline (wraps a sycl::event).
class L0DeviceEvent final : public backend::DeviceEvent {
public:
  L0DeviceEvent();
  ~L0DeviceEvent() override;

  [[nodiscard]] backend::DeviceKind device() const noexcept override;
  [[nodiscard]] void *native_handle() const noexcept override; // sycl::event*
  void synchronize() override;
  [[nodiscard]] bool query() const noexcept override;

  // Internal: L0DeviceQueue::record() overwrites the wrapped sycl::event.
  struct Impl;
  [[nodiscard]] Impl *impl() noexcept { return impl_.get(); }
  [[nodiscard]] const Impl *impl() const noexcept { return impl_.get(); }

private:
  std::unique_ptr<Impl> impl_;
};

// One ordered lane of Intel GPU work.
class L0DeviceQueue final : public backend::DeviceQueue {
public:
  // In-order sycl::queue on the default GPU selector; `device_id` >= 0 selects
  // among multiple Intel GPUs.
  explicit L0DeviceQueue(int device_id = -1);
  ~L0DeviceQueue() override;

  [[nodiscard]] backend::DeviceKind device() const noexcept override;
  [[nodiscard]] bool is_async() const noexcept override { return true; }
  [[nodiscard]] void *native_handle() const noexcept override; // sycl::queue*

  void record(backend::DeviceEvent &ev) override;
  void wait(const backend::DeviceEvent &ev) override;
  void synchronize() override;
  [[nodiscard]] std::unique_ptr<backend::DeviceEvent> make_event() override;

  void begin_batch() override;
  void end_batch() override;
  [[nodiscard]] bool batch_open() const noexcept override;

  // Called by kernels/allocator right after they enqueue, so record() can
  // snapshot an accurate lane tail. (SYCL has no "query the queue's last event"
  // API; the lane must be told.)
  void note_submission(void *sycl_event) noexcept;

  struct Impl;
  [[nodiscard]] Impl *impl() noexcept { return impl_.get(); }

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace turbo_ocr::intel
