// CudaDeviceQueue / CudaDeviceEvent implementation — thin cudaStream_t /
// cudaEvent_t forwarding. Every call is the SAME CUDA call the existing
// pipeline already makes; the only added value is the interface vtable.

#include "nvidia/queue/cuda_device_queue.h"

#include "nvidia/support/cuda_check.h" // CUDA_CHECK

namespace turbo_ocr::nvidia {

// ---- CudaDeviceEvent -------------------------------------------------------

CudaDeviceEvent::CudaDeviceEvent() {
  CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}

CudaDeviceEvent::~CudaDeviceEvent() {
  if (event_)
    cudaEventDestroy(event_); // best-effort; ignore error in destructor
}

void CudaDeviceEvent::synchronize() {
  if (event_)
    CUDA_CHECK(cudaEventSynchronize(event_));
}

bool CudaDeviceEvent::query() const noexcept {
  if (!event_)
    return true;
  return cudaEventQuery(event_) == cudaSuccess;
}

// ---- CudaDeviceQueue -------------------------------------------------------

CudaDeviceQueue::CudaDeviceQueue(bool owns) : owns_(owns) {
  if (owns_)
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

CudaDeviceQueue::~CudaDeviceQueue() {
  if (owns_ && stream_)
    cudaStreamDestroy(stream_); // best-effort in destructor
}

void CudaDeviceQueue::record(backend::DeviceEvent &ev) {
  auto *ce = static_cast<CudaDeviceEvent *>(&ev);
  CUDA_CHECK(cudaEventRecord(ce->raw(), stream_));
}

void CudaDeviceQueue::wait(const backend::DeviceEvent &ev) {
  // Device-side wait: this stream stalls until `ev` is reached on whichever
  // stream recorded it — the exact "rec waits for det" ordering the pipeline
  // expresses today via cudaStreamWaitEvent (no host round-trip).
  const auto *ce = static_cast<const CudaDeviceEvent *>(&ev);
  CUDA_CHECK(cudaStreamWaitEvent(stream_, ce->raw(), 0));
}

void CudaDeviceQueue::synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

std::unique_ptr<backend::DeviceEvent> CudaDeviceQueue::make_event() {
  return std::make_unique<CudaDeviceEvent>();
}

} // namespace turbo_ocr::nvidia
