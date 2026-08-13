#pragma once

// MetalDeviceQueue / MetalDeviceEvent — Apple implementation of the
// device-agnostic ordering + one-submit batching primitives (device_queue.h).
//
// A DeviceQueue is "one ordered lane of device work". On Metal that is an
// MTLCommandQueue plus, while a batch is open, the CURRENTLY OPEN MPSCommandBuffer
// that every encoder appends to. begin_batch() opens one command buffer;
// end_batch() commits it (without waiting) — that single command buffer spanning
// warp -> MPSGraph rec -> argmax IS the residency guarantee the POC proved
// (tools/probes/apple/mps_ocr.mm:151-154, "FUSED one cmd buffer"). Outside a batch each op
// acquires its own command buffer and submits it immediately.
//
// The Apple-internal acquire_cb()/submit_cb() pair is how stages/engine/kernels
// enqueue work uniformly in both modes: inside a batch they share the open
// buffer (submit is a no-op, commit deferred to end_batch); outside one they get
// a fresh buffer that submit commits. Portable callers only ever see the base
// DeviceQueue interface; Apple TUs down-cast via as_metal().
//
// Events are MTLSharedEvent-based, giving true device-side cross-queue waits
// (record/wait) as well as host synchronize()/query().

#include <atomic>
#include <cstdint>
#include <memory>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif

#include "turbo_ocr/backend/device_queue.h"

namespace turbo_ocr::apple {

#ifdef __OBJC__
// Attach the "did this command buffer actually succeed?" completion handler. A
// failed MTLCommandBuffer executes NONE of its encoded work — including the
// event signal — so without this a GPU fault is indistinguishable from success
// except that the destination buffers silently keep their previous contents.
//
// `sink` receives the failure count and MUST outlive the command buffer, which
// is why it is a shared_ptr captured by value: the completion handler can fire
// after the MetalDeviceQueue that committed the buffer has been destroyed.
void attach_error_watch(id<MTLCommandBuffer> cb,
                        std::shared_ptr<std::atomic<unsigned long long>> sink);
#endif
// Total command-buffer failures observed in this process (all queues). Kept for
// coarse reporting; correctness decisions use the PER-QUEUE count via
// MetalDeviceQueue::sync_ok(), because a process-wide counter cannot tell you
// whether *your* page's work failed.
unsigned long long command_buffer_error_count();

class MetalDeviceEvent final : public backend::DeviceEvent {
public:
#ifdef __OBJC__
  MetalDeviceEvent(id<MTLSharedEvent> ev) : event_(ev) {}
  [[nodiscard]] id<MTLSharedEvent> raw() const noexcept { return event_; }
#endif
  ~MetalDeviceEvent() override;

  [[nodiscard]] backend::DeviceKind device() const noexcept override {
    return backend::DeviceKind::Metal;
  }
  [[nodiscard]] void *native_handle() const noexcept override;
  void synchronize() override;
  [[nodiscard]] bool query() const noexcept override;

  // The value the recording queue will signal (and the waiter awaits). Set when
  // the event is recorded; monotonically increasing per queue timeline.
  std::uint64_t signal_value = 0;

private:
#ifdef __OBJC__
  id<MTLSharedEvent> event_ = nil;
#else
  void *event_ = nullptr;
#endif
};

class MetalDeviceQueue final : public backend::DeviceQueue {
public:
  MetalDeviceQueue();
  ~MetalDeviceQueue() override;

  [[nodiscard]] backend::DeviceKind device() const noexcept override {
    return backend::DeviceKind::Metal;
  }
  [[nodiscard]] bool is_async() const noexcept override { return true; }
  [[nodiscard]] void *native_handle() const noexcept override;

  void record(backend::DeviceEvent &ev) override;
  void wait(const backend::DeviceEvent &ev) override;
  void synchronize() override;

  // CHECKED synchronize. Returns false if the timeline wait timed out OR any
  // command buffer committed on THIS queue failed since the matching
  // sync_mark(). Callers that are about to read device scratch the GPU was
  // supposed to overwrite MUST use this, not synchronize().
  //
  // Why it exists: a failed MTLCommandBuffer executes none of its encoded work,
  // so every buffer it was to write silently keeps the PREVIOUS page's bytes.
  // Reading them yields the previous page's complete, correct transcript
  // attributed to this page — a whole-page mix-up, not garbled output. The
  // detector for this already existed (attach_error_watch) but its counter was
  // consumed by nothing; this is what consumes it.
  [[nodiscard]] unsigned long long sync_mark() const noexcept {
    return cb_errors_ ? cb_errors_->load(std::memory_order_relaxed) : 0;
  }
  [[nodiscard]] bool sync_ok(unsigned long long mark);
  [[nodiscard]] std::unique_ptr<backend::DeviceEvent> make_event() override;

  void begin_batch() override;
  void end_batch() override;
  void flush() override;
  [[nodiscard]] bool batch_open() const noexcept override { return batch_open_; }

#ifdef __OBJC__
  // --- Apple-internal encoding API (used by MpsEngine / MetalKernels) --------
  // Acquire the command buffer to encode onto: the open batch buffer if a batch
  // is active, otherwise a fresh MPSCommandBuffer the caller must submit_cb().
  MPSCommandBuffer *acquire_cb();
  // Finish a command buffer obtained from acquire_cb(): a no-op for the batch
  // buffer (committed at end_batch), else commit it now (does NOT wait).
  void submit_cb(MPSCommandBuffer *cb);

  [[nodiscard]] id<MTLCommandQueue> raw() const noexcept { return q_; }
#endif

private:
#ifdef __OBJC__
  // Every commit funnels through here so `submitted_value_` covers ALL work ever
  // submitted on this queue, not just the most recent command buffer.
  void commit_(MPSCommandBuffer *cb);

  id<MTLCommandQueue> q_ = nil;
  MPSCommandBuffer *open_ = nil;             // the batch command buffer, or nil
  id<MTLCommandBuffer> last_committed_ = nil; // most recent (diagnostics only)
  id<MTLSharedEvent> timeline_ = nil;         // record/wait + synchronize token
#endif
  std::uint64_t timeline_value_ = 0;   // next value to hand out
  std::uint64_t submitted_value_ = 0;  // highest value actually COMMITTED
  bool batch_open_ = false;
  // shared_ptr, not a member value: a completion handler can fire after this
  // queue is destroyed, and it captures the sink by value to stay valid.
  std::shared_ptr<std::atomic<unsigned long long>> cb_errors_;
};

// Down-cast a portable DeviceQueue& known to be Metal (every Apple stage owns a
// Metal queue). Mirrors the NVIDIA backend's stream down-cast.
inline MetalDeviceQueue &as_metal(backend::DeviceQueue &q) {
  return static_cast<MetalDeviceQueue &>(q);
}

} // namespace turbo_ocr::apple
