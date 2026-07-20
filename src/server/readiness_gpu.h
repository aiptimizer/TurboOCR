#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"

// GPU readiness probe, extracted from the server main: a bounded real-
// inference probe whose verdict is cached for the gRPC cache-only view.
namespace turbo_ocr::server {

struct GpuProbeState {
  std::atomic<bool> ok{true};
  std::atomic<long long> last_check_ms{0};
  std::mutex mu;  // single-flight: one real probe at a time under spikes
};

// The HTTP /health/ready probe: may run a real (bounded) GPU pass; refreshes
// state->ok for bootstrap::make_cached_readiness consumers.
[[nodiscard]] std::function<bool()>
make_gpu_readiness(pipeline::PipelineDispatcher &dispatcher,
                   bool layout_available,
                   std::shared_ptr<GpuProbeState> state);

} // namespace turbo_ocr::server
