#include "readiness_gpu.h"

#include <chrono>

#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/server/bootstrap/server_bootstrap.h"

namespace turbo_ocr::server {

std::function<bool()>
make_gpu_readiness(pipeline::PipelineDispatcher &dispatcher,
                   bool layout_available,
                   std::shared_ptr<GpuProbeState> state) {
  auto *disp = &dispatcher;
  auto probe = std::move(state);
  return [disp, probe, layout_available]() -> bool {
    // Fail readiness the instant a drain begins so k8s stops routing NEW traffic
    // to a shutting-down pod (previously it reported Ready throughout drain).
    if (bootstrap::g_shutdown_requested.load(std::memory_order_acquire))
      return false;
    using namespace std::chrono;
    const int64_t now_ms = duration_cast<milliseconds>(
        steady_clock::now().time_since_epoch()).count();
    if (now_ms - probe->last_check_ms.load(std::memory_order_acquire) < 5000)
      return probe->ok.load(std::memory_order_acquire);

    std::lock_guard lock(probe->mu);
    // Recheck under lock — another thread may have refreshed the cache
    // while we waited. Avoids stampedes during probe spikes.
    if (now_ms - probe->last_check_ms.load(std::memory_order_acquire) < 5000)
      return probe->ok.load(std::memory_order_acquire);

    bool ok = false;
    try {
      // submit_for_default honours the request timeout, so a wedged worker
      // flips readiness to not-ready (TimeoutError, caught below) instead of
      // blocking this probe thread forever. The dummy Mat is created inside the
      // task, so there is nothing request-scoped to abandon on timeout.
      disp->submit_for_default([layout_available](auto &e) {
        cv::Mat dummy(48, 48, CV_8UC3, cv::Scalar(255, 255, 255));
        (void)e.pipeline->run_with_layout(dummy, e.stream,
                                          /*want_layout=*/layout_available,
                                          /*want_reading_order=*/false);
      });
      ok = true;
    } catch (const turbo_ocr::PoolExhaustedError &) {
      // Queue full = the server is BUSY but healthy. Flipping to not-ready here
      // would pull a fine-but-loaded pod out of rotation under a burst (and shed
      // load exactly when it's needed). Keep the last verdict; reserve not-ready
      // for a wedged worker (TimeoutError) or a genuine inference fault.
      ok = probe->ok.load(std::memory_order_acquire);
    } catch (...) {
      ok = false;
    }
    probe->ok.store(ok, std::memory_order_release);
    probe->last_check_ms.store(now_ms, std::memory_order_release);
    return ok;
  };
}

} // namespace turbo_ocr::server
