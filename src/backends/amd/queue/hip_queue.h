#pragma once

// HipStreamQueue / HipEvent — the AMD realization of backend::DeviceQueue /
// backend::DeviceEvent. Maps 1:1 onto the CUDA design (wf: "Hip — DeviceQueue =
// hipStream_t; DeviceEvent = hipEvent_t (1:1 with CUDA)"):
//   record()      -> hipEventRecord
//   wait()        -> hipStreamWaitEvent   (device-side, no host round-trip)
//   synchronize() -> hipStreamSynchronize
//   make_event()  -> hipEventCreateWithFlags(hipEventDisableTiming)
//
// is_async() == true.
//
// ---------------------------------------------------------------------------
// BatchScope / begin_batch()/end_batch() — WHY THIS IS A NO-OP ON HIP
// ---------------------------------------------------------------------------
// The seam defines begin_batch/end_batch as "group every submission enqueued
// between these calls into a SINGLE device submission". That verb only has
// teeth on an API where the default is MANY submissions: Metal, where every
// encoder pass would otherwise be its own MTLCommandBuffer with its own commit
// and its own host round-trip. There, BatchScope is the residency lever — one
// command buffer spanning warp + rec + argmax for a whole image.
//
// HIP has the opposite default. A hipStream_t IS a single ordered submission
// lane: back-to-back kernel launches and hipMemcpyAsync calls on one stream are
// already queued without any host round-trip between them, and nothing is
// "committed" — the driver submits continuously. So the property BatchScope
// EXISTS TO GUARANTEE is already unconditionally true here, and the honest
// implementation of "make it true" is to do nothing. This matches CUDA.
//
// This is therefore NOT an unimplemented stub. Two invariants make the no-op
// correct rather than merely convenient:
//   * end_batch() must not synchronize — and it does not. The seam is explicit
//     that end_batch flushes but does NOT wait; callers order with events or
//     synchronize(). A HIP stream needs no flush.
//   * Work enqueued inside the scope must be ordered with work enqueued before
//     it — which stream semantics give for free.
// batch_open() still tracks the flag so BatchScope's contract (and any future
// nesting assertion) is well-defined and observable.
//
// TODO(on-hardware, optional): begin_batch/end_batch is the natural place to
// open a hipGraph capture region (hipStreamBeginCapture/hipStreamEndCapture +
// hipGraphLaunch) to collapse per-image LAUNCH overhead — a different win from
// residency, and the HIP analogue of the TRT CUDA-graph bake. It is a pure
// optimization: the plain-stream path is already correct and already resident,
// so this should only be attempted once the pipeline is stable and profiling
// shows launch overhead actually matters. Note that capture forbids
// synchronizing calls inside the region, which today's db_postprocess (it
// syncs to read the component count) violates — that would have to be lifted
// first.

#include <memory>

#include <hip/hip_runtime_api.h>

#include "turbo_ocr/backend/device_queue.h"

namespace turbo_ocr::amd {

using backend::DeviceEvent;
using backend::DeviceKind;
using backend::DeviceQueue;

class HipEvent final : public DeviceEvent {
public:
  HipEvent();
  ~HipEvent() override;

  [[nodiscard]] DeviceKind device() const noexcept override {
    return DeviceKind::Hip;
  }
  [[nodiscard]] void *native_handle() const noexcept override {
    return reinterpret_cast<void *>(event_);
  }
  void synchronize() override;
  [[nodiscard]] bool query() const noexcept override;

  [[nodiscard]] hipEvent_t raw() const noexcept { return event_; }

private:
  hipEvent_t event_ = nullptr;
};

class HipStreamQueue final : public DeviceQueue {
public:
  // owns_stream == true: created via hipStreamCreateWithFlags and destroyed here
  // (a pipeline entry owns its queue). Pass an existing stream + owns=false to
  // wrap a foreign stream (e.g. one MIGraphX or an external caller supplies).
  explicit HipStreamQueue(int device_id = 0);
  HipStreamQueue(hipStream_t stream, bool owns_stream) noexcept;
  ~HipStreamQueue() override;

  [[nodiscard]] DeviceKind device() const noexcept override {
    return DeviceKind::Hip;
  }
  [[nodiscard]] bool is_async() const noexcept override { return true; }
  [[nodiscard]] void *native_handle() const noexcept override {
    return reinterpret_cast<void *>(stream_);
  }

  void record(DeviceEvent &ev) override;
  void wait(const DeviceEvent &ev) override;
  void synchronize() override;
  [[nodiscard]] std::unique_ptr<DeviceEvent> make_event() override;

  void begin_batch() override { batch_open_ = true; }
  void end_batch() override { batch_open_ = false; }
  [[nodiscard]] bool batch_open() const noexcept override { return batch_open_; }

  [[nodiscard]] hipStream_t raw() const noexcept { return stream_; }
  [[nodiscard]] int device_id() const noexcept { return device_id_; }

private:
  hipStream_t stream_ = nullptr;
  int device_id_ = 0;
  bool owns_stream_ = true;
  bool batch_open_ = false;
};

// Down-cast helper for stage/kernel code that has a DeviceQueue& but needs the
// raw hipStream_t. Asserts the queue is actually a HIP queue.
[[nodiscard]] hipStream_t hip_stream_of(DeviceQueue &q);

} // namespace turbo_ocr::amd
