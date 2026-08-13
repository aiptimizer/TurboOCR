#include "amd/queue/hip_queue.h"

#include "amd/support/hip_check.h"

#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>

namespace turbo_ocr::amd {

// --- HipEvent ---------------------------------------------------------------

HipEvent::HipEvent() {
  // DisableTiming: we use events purely for ordering/queries, not profiling —
  // matches the CUDA path's cudaEventDisableTiming for lower record overhead.
  HIP_CHECK(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
}

HipEvent::~HipEvent() {
  if (event_)
    hipEventDestroy(event_); // best-effort; never throw from a dtor
}

void HipEvent::synchronize() { HIP_CHECK(hipEventSynchronize(event_)); }

bool HipEvent::query() const noexcept {
  return hipEventQuery(event_) == hipSuccess;
}

// --- HipStreamQueue ---------------------------------------------------------

HipStreamQueue::HipStreamQueue(int device_id) : device_id_(device_id) {
  HIP_CHECK(hipSetDevice(device_id_));
  // NonBlocking so this stream never implicitly serializes against the legacy
  // default stream — every stage lane is independent (mirrors the CUDA entry's
  // cudaStreamCreateWithFlags(cudaStreamNonBlocking)).
  HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
}

HipStreamQueue::HipStreamQueue(hipStream_t stream, bool owns_stream) noexcept
    : stream_(stream), owns_stream_(owns_stream) {}

HipStreamQueue::~HipStreamQueue() {
  if (owns_stream_ && stream_)
    hipStreamDestroy(stream_);
}

void HipStreamQueue::record(DeviceEvent &ev) {
  HIP_CHECK(hipEventRecord(reinterpret_cast<hipEvent_t>(ev.native_handle()),
                           stream_));
}

void HipStreamQueue::wait(const DeviceEvent &ev) {
  // Device-side wait: this stream stalls on the GPU until `ev` is reached on
  // whichever stream recorded it — no host round-trip (this is how "rec waits
  // for det" is expressed, exactly like cudaStreamWaitEvent).
  HIP_CHECK(hipStreamWaitEvent(
      stream_, reinterpret_cast<hipEvent_t>(ev.native_handle()), 0));
}

void HipStreamQueue::synchronize() { HIP_CHECK(hipStreamSynchronize(stream_)); }

std::unique_ptr<DeviceEvent> HipStreamQueue::make_event() {
  return std::make_unique<HipEvent>();
}

hipStream_t hip_stream_of(DeviceQueue &q) {
  // Portable code never calls this; stage/kernel code that must reach the raw
  // stream down-casts. A non-HIP queue here is a build/wiring bug.
  auto *hq = dynamic_cast<HipStreamQueue *>(&q);
  if (!hq) {
    // Returning nullptr would be WORSE than crashing: nullptr is the HIP NULL
    // (default) stream, so every kernel would silently run on a different lane
    // from the one the caller is synchronizing and event-ordering against —
    // a race that shows up as intermittent garbage output, not as an error.
    // A DeviceKind::Hip queue that is not a HipStreamQueue means two AMD queue
    // implementations are linked into one binary. Fail loudly.
    std::fprintf(stderr,
                 "[HIP] hip_stream_of(): queue is not a HipStreamQueue "
                 "(device=%d). Two queue implementations linked?\n",
                 static_cast<int>(q.device()));
    std::abort();
  }
  return hq->raw();
}

} // namespace turbo_ocr::amd
