#pragma once

// CudaDeviceQueue / CudaDeviceEvent — the NVIDIA implementation of the
// device-agnostic ordering primitives (backend/device_queue.h).
//
// This is a DIRECT wrap of the raw cudaStream_t / cudaEvent_t handles that
// OcrPipeline weaves through today (rec_stream_/det_event_/…, ocr_pipeline.h
// :286-370). No behaviour changes: record==cudaEventRecord,
// wait==cudaStreamWaitEvent, synchronize==cudaStreamSynchronize, and the
// one-submit batch is a NO-OP because CUDA is already one ordered stream (the
// residency it expresses is intrinsic to a stream — begin/end_batch exist only
// so Metal can group a command buffer; on CUDA there is nothing to group).
//
// Ownership mirrors GpuPipelineEntry: a queue owns its stream and destroys it.
// A queue may optionally NOT own the stream (owns_==false) when adapting a
// stream the pipeline already created (e.g. wrapping OcrPipeline's rec_stream_
// during incremental migration).

#include <memory>

#include <cuda_runtime.h>

#include "turbo_ocr/backend/device_queue.h"

namespace turbo_ocr::nvidia {

class CudaDeviceEvent final : public backend::DeviceEvent {
public:
  // Creates a timing-disabled event (cudaEventDisableTiming) — the pipeline
  // uses events only for ordering, never for elapsed-time queries, and the
  // disabled-timing flag makes record/wait cheaper.
  CudaDeviceEvent();
  ~CudaDeviceEvent() override;

  [[nodiscard]] backend::DeviceKind device() const noexcept override {
    return backend::DeviceKind::Cuda;
  }
  [[nodiscard]] void *native_handle() const noexcept override {
    return static_cast<void *>(event_);
  }
  void synchronize() override;
  [[nodiscard]] bool query() const noexcept override;

  [[nodiscard]] cudaEvent_t raw() const noexcept { return event_; }

private:
  cudaEvent_t event_ = nullptr;
};

class CudaDeviceQueue final : public backend::DeviceQueue {
public:
  // owns==true => the queue creates (cudaStreamCreateWithFlags,
  // cudaStreamNonBlocking) and destroys the stream. owns==false => it adapts an
  // existing stream and never destroys it (migration aid).
  explicit CudaDeviceQueue(bool owns = true);
  explicit CudaDeviceQueue(cudaStream_t adopt, bool owns = false)
      : stream_(adopt), owns_(owns) {}
  ~CudaDeviceQueue() override;

  [[nodiscard]] backend::DeviceKind device() const noexcept override {
    return backend::DeviceKind::Cuda;
  }
  [[nodiscard]] bool is_async() const noexcept override { return true; }
  [[nodiscard]] void *native_handle() const noexcept override {
    return static_cast<void *>(stream_);
  }

  void record(backend::DeviceEvent &ev) override;
  void wait(const backend::DeviceEvent &ev) override;
  void synchronize() override;
  [[nodiscard]] std::unique_ptr<backend::DeviceEvent> make_event() override;

  // CUDA is already one ordered lane; batching is intrinsic, so these are
  // no-ops. batch_open() tracks the RAII BatchScope state only for symmetry
  // with the async backends (Metal), where end_batch commits a command buffer.
  void begin_batch() override { batch_open_ = true; }
  void end_batch() override { batch_open_ = false; }
  [[nodiscard]] bool batch_open() const noexcept override { return batch_open_; }

  [[nodiscard]] cudaStream_t raw() const noexcept { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
  bool owns_ = true;
  bool batch_open_ = false;
};

} // namespace turbo_ocr::nvidia
