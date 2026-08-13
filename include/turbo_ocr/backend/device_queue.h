#pragma once

// DeviceQueue + DeviceEvent — device-agnostic ordering / barrier / one-submit
// batching primitives.
//
// These replace the raw cudaStream_t / cudaEvent_t handles that are currently
// woven through OcrPipeline (rec_stream_/det_event_/layout_stream_/… — see
// pipeline/ocr/ocr_pipeline.h:286-370) and passed on every stage, engine, and
// kernel call. One DeviceQueue is the single "ordered lane of device work" the
// whole pipeline schedules against; a DeviceEvent is a cross-lane dependency /
// completion token.
//
// Per-vendor mapping:
//   Cuda   — DeviceQueue = cudaStream_t; DeviceEvent = cudaEvent_t.
//            record()=cudaEventRecord, wait()=cudaStreamWaitEvent,
//            synchronize()=cudaStreamSynchronize. Async (is_async()==true).
//   Metal  — DeviceQueue wraps an MTLCommandQueue and the CURRENTLY OPEN
//            MTLCommandBuffer; DeviceEvent = MTLSharedEvent (or a command-buffer
//            completion handler). begin_batch() opens one MTLCommandBuffer that
//            every encoder appends to; end_batch() commits it — that IS the
//            residency guarantee (all stages of one image in one command
//            buffer, zero host round-trips), the pattern the MPSGraph/Metal POC
//            proved (tools/probes/apple/mps_*.mm).
//   Hip    — DeviceQueue = hipStream_t; DeviceEvent = hipEvent_t (1:1 with CUDA).
//   L0     — DeviceQueue = ze_command_queue_t + a command list (or a SYCL queue);
//            DeviceEvent = ze_event_t / SYCL event.
//   Host   — synchronous no-op: is_async()==false, record/wait/synchronize do
//            nothing, work already completed on return (CpuBackend).
//
// Ownership: a DeviceQueue and DeviceEvent are created by the Backend's
// factories (see backend.h) and owned by whoever holds them (a pipeline entry
// owns its queue, exactly as a GpuPipelineEntry owns its cudaStream_t today).
// The queue never blocks or syncs implicitly — the caller orders work with
// events and calls synchronize() before reading results host-side.

#include <memory>

#include "turbo_ocr/backend/image_view.h" // DeviceKind

namespace turbo_ocr::backend {

// A recordable completion / dependency token in one device's timeline.
// Non-copyable; owned via unique_ptr handed out by DeviceQueue::make_event().
class DeviceEvent {
public:
  virtual ~DeviceEvent() = default;

  [[nodiscard]] virtual DeviceKind device() const noexcept = 0;

  // Opaque native handle (cudaEvent_t / MTLSharedEvent* / hipEvent_t / ze_event_t).
  // A backend down-casts its own queue's events; callers treat it as opaque.
  [[nodiscard]] virtual void *native_handle() const noexcept = 0;

  // Block the HOST until the work that recorded this event has completed.
  // No-op / immediate for synchronous (Host) backends.
  virtual void synchronize() = 0;

  // True once the recorded work has completed, without blocking. On a
  // synchronous backend this is always true.
  [[nodiscard]] virtual bool query() const noexcept = 0;

protected:
  DeviceEvent() = default;
  DeviceEvent(const DeviceEvent &) = delete;
  DeviceEvent &operator=(const DeviceEvent &) = delete;
};

// One ordered lane of device work. All engine::run / kernel / stage calls take
// a DeviceQueue& and enqueue onto it; execution is asynchronous for device
// backends (results are NOT ready until an event fires or synchronize() returns)
// and immediate for the Host backend.
class DeviceQueue {
public:
  virtual ~DeviceQueue() = default;

  [[nodiscard]] virtual DeviceKind device() const noexcept = 0;

  // false for the Host backend (synchronous; every call blocks to completion and
  // events are already-signalled). true for Cuda/Metal/Hip/L0. Callers use this
  // to decide whether cross-queue event choreography is meaningful.
  [[nodiscard]] virtual bool is_async() const noexcept = 0;

  // Opaque native handle:
  //   Cuda -> cudaStream_t, Hip -> hipStream_t, L0 -> ze_command_list/queue,
  //   Metal -> the open MTLCommandBuffer (or MTLCommandQueue when no batch is
  //   open). Consumers that must reach the raw handle down-cast to the vendor
  //   queue; portable code never touches this.
  [[nodiscard]] virtual void *native_handle() const noexcept = 0;

  // --- Ordering / barriers --------------------------------------------------

  // Record `ev` at the current point of THIS queue's timeline.
  virtual void record(DeviceEvent &ev) = 0;

  // Make THIS queue wait (device-side, not host-side) until `ev` has been
  // reached on whichever queue recorded it. This is how the pipeline expresses
  // "rec waits for det" without a host round-trip (today: cudaStreamWaitEvent).
  virtual void wait(const DeviceEvent &ev) = 0;

  // Block the HOST until every submission on this queue has completed.
  //
  // CONTRACT: calling synchronize() while a batch is OPEN is a LOGIC ERROR. The
  // work accumulated in the open batch has not been submitted, so the wait can
  // only cover EARLIER submissions and the caller would then read stale results
  // — a silent wrong-output bug, not a crash. Close the BatchScope first, or
  // call flush() if the batch must stay open. An implementation should diagnose
  // this rather than return quietly.
  virtual void synchronize() = 0;

  // Mint an event that can be recorded on this queue and waited on from any
  // queue of the same device.
  [[nodiscard]] virtual std::unique_ptr<DeviceEvent> make_event() = 0;

  // --- One-submit batch -----------------------------------------------------
  //
  // Groups every submission enqueued between begin_batch() and end_batch() into
  // a SINGLE device submission. This is the residency lever: on Metal it is one
  // MTLCommandBuffer spanning warp+normalize+MPSGraph+argmax for a whole image
  // (the ~100 img/s POC path); on CUDA it is a natural no-op (already one
  // stream) or an optional graph-capture region; on Host it is a no-op.
  //
  // Batches do not nest (begin while open is a logic error). end_batch()
  // flushes/commits; it does NOT synchronize — use an event or synchronize()
  // to wait. batch_open() reports whether a batch is currently accumulating.
  virtual void begin_batch() = 0;
  virtual void end_batch() = 0;
  [[nodiscard]] virtual bool batch_open() const noexcept = 0;

  // Submit everything accumulated so far WITHOUT closing the batch. Required
  // before any host read of results produced inside an open batch (see the
  // synchronize() contract above): flush() then synchronize() is well-defined,
  // synchronize() alone under an open batch is not. On Metal this commits the
  // open MTLCommandBuffer and opens a fresh one that later encoders append to;
  // on CUDA/Host it is a no-op because work is submitted as it is enqueued.
  // Does NOT wait.
  virtual void flush() {}

protected:
  DeviceQueue() = default;
  DeviceQueue(const DeviceQueue &) = delete;
  DeviceQueue &operator=(const DeviceQueue &) = delete;
};

// RAII helper: opens a one-submit batch for the lifetime of the scope and
// commits it on destruction. Use for exception-safe residency around a full
// per-image pipeline run.
//
//   {
//     BatchScope batch(queue);      // begin_batch()
//     detector.run(img, h, w, queue);
//     recognizer.run(img, boxes, queue);
//   }                               // end_batch() — one submission
class BatchScope {
public:
  explicit BatchScope(DeviceQueue &q) : q_(&q) { q_->begin_batch(); }
  ~BatchScope() {
    if (q_)
      q_->end_batch();
  }
  BatchScope(const BatchScope &) = delete;
  BatchScope &operator=(const BatchScope &) = delete;
  BatchScope(BatchScope &&o) noexcept : q_(o.q_) { o.q_ = nullptr; }

private:
  DeviceQueue *q_;
};

} // namespace turbo_ocr::backend
