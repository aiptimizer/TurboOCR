#pragma once

// make_infer_func — the ONE server::InferFunc builder over a pool of
// UnifiedOcrPipeline entries.
//
// This replaces every per-backend make_infer_func (AppleBackend::make_infer_
// func's hand-rolled det->cls->rec closure, CudaBackend's dispatcher submit,
// and the CPU server's make_cpu_infer_func). Backends keep only load_stages +
// make_image_decoder + make_orient_func; the merged server_main builds a pool
// of UnifiedOcrPipeline over the backend's StageSet and calls this once.
//
// Each UnifiedPipelineEntry is single-thread-per-instance (the pipeline contract), so
// the pool leases one free entry per request and the InferFunc is safe to call
// concurrently from N request threads as long as the pool has >= N entries
// available (it blocks for a free one otherwise).

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "turbo_ocr/backend/backend.h"  // backend::Backend
#include "turbo_ocr/core/service_fns.h" // server::InferFunc
#include "turbo_ocr/pipeline/unified/stage_batcher.h"               // DetBatchConfig / DetectionBatcher
#include "turbo_ocr/pipeline/unified/unified_ocr_pipeline.h"

namespace turbo_ocr::pipeline {

// One pool slot. The pipeline owns its DeviceQueue (moved in at construction);
// the queue member is kept for symmetry / explicit lifetime when a caller wants
// to hold the queue separately, and is null when the pipeline owns it.
struct UnifiedPipelineEntry {
  std::unique_ptr<UnifiedOcrPipeline> pipeline;
  std::unique_ptr<backend::DeviceQueue> queue;
};

// NAME: the "Unified" prefix is now VESTIGIAL. It exists because a
// turbo_ocr::pipeline::PipelinePool<Pipeline> template once shared this
// namespace, so the two names collided in the merged server_main. That template
// was the CUDA orchestration's pool; it went unreferenced when that pipeline was
// deleted — never instantiated anywhere, its own header included by nobody — and
// has now been deleted with it. The prefix stays only because renaming a type
// this widely used is a change of its own, not because anything still collides.
//
class UnifiedPipelinePool {
public:
  explicit UnifiedPipelinePool(std::vector<UnifiedPipelineEntry> entries);
  UnifiedPipelinePool(std::vector<UnifiedPipelineEntry> entries,
                      std::chrono::milliseconds acquire_timeout,
                      std::size_t max_waiters);

  // RAII lease of one free pipeline. Returns the slot on destruction.
  class Lease {
  public:
    Lease(UnifiedPipelinePool &pool, std::size_t idx) noexcept
        : pool_(&pool), idx_(idx) {}
    ~Lease() { if (pool_) pool_->release_(idx_); }
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    Lease(Lease &&o) noexcept : pool_(o.pool_), idx_(o.idx_) { o.pool_ = nullptr; }

    [[nodiscard]] UnifiedOcrPipeline &pipeline() const noexcept {
      return *pool_->entries_[idx_].pipeline;
    }

  private:
    UnifiedPipelinePool *pool_;
    std::size_t idx_;
  };

  // Lease one replica. Throws PoolExhaustedError when the wait cap is already
  // full or the deadline elapses — never queues without bound.
  [[nodiscard]] Lease acquire();

  // Requests currently blocked in acquire(). Exposed so /metrics can report
  // saturation of the pool that ACTUALLY serialises on the device, rather than
  // only the WorkPool queue in front of it (the gap work_pool.h names).
  [[nodiscard]] std::size_t waiting() const;
  [[nodiscard]] std::size_t available() const;

  // ---- Stuck-lease detection (the dispatcher's watchdog, folded in) ---------
  //
  // The pool's DEADLINE bounds QUEUEING only: once a replica is leased, a wedged
  // stage runs unbounded and the request never returns. That is invisible today
  // — the slot simply never comes back, throughput silently drops by 1/N, and
  // nothing says why. These make it observable.
  //
  // WHY DETECTION AND NOT RECOVERY. Rebuilding a wedged replica is deliberately
  // NOT done here. A stuck lease is usually stuck INSIDE a device call, and
  // destroying a pipeline whose kernels/command buffers are still in flight is
  // not a recovery, it is a crash with extra steps — the owning thread is still
  // executing in it. Safe recycling needs a cancellation point in the stage
  // seam (IEngine/DeviceQueue) that does not exist yet. Reporting the condition
  // loudly is worth having now; pretending to fix it is not.

  // Age of the longest-held ACTIVE lease. Zero when nothing is leased.
  [[nodiscard]] std::chrono::milliseconds oldest_lease_age() const;

  // Monotonic count of leases observed held past the stuck threshold. Each one
  // is a request that is almost certainly never returning — a nonzero, growing
  // value is the wedged-worker signal, and it is what an alert should watch.
  // A lease is counted at most once no matter how long it stays stuck.
  [[nodiscard]] std::uint64_t stuck_leases() const;

  // Threshold from TURBO_POOL_STUCK_LEASE_MS. Default 0 = detection disabled;
  // set it to a small multiple of your request timeout. Exposed so /metrics can
  // report whether the check is even armed — a zero counter means nothing if
  // the detector is off.
  [[nodiscard]] std::chrono::milliseconds stuck_threshold() const noexcept {
    return stuck_threshold_;
  }

  // Sweep for leases past the threshold: logs each newly-stuck slot once and
  // bumps stuck_leases(). Called from the metrics scrape rather than from a
  // dedicated thread — the pool must not own a thread whose only job is to
  // observe, and a scrape is exactly when someone is asking.
  void check_stuck_leases();

  // Bounded acquire — returns nullopt if no slot frees up within `timeout`.
  // This is what a readiness probe must use: the unbounded acquire() above can
  // never answer NOT-ready and would block the probe forever under saturation.
  // Shared here (not per-server) so every backend's /health/ready behaves the
  // same way.
  [[nodiscard]] std::optional<Lease>
  try_acquire_for(std::chrono::milliseconds timeout);

  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

  // NOTE (removed): pipeline_at(), the "pool-wide configuration" accessor. It
  // had no caller, and the configuration it existed for already happens without
  // it — every replica installs pipeline::shared_detection_batcher() in its own
  // constructor, so there is one batcher across the pool by construction rather
  // than by a bootstrap walking the entries.

private:
  friend class Lease;
  void release_(std::size_t idx);

  std::vector<UnifiedPipelineEntry> entries_;
  mutable std::mutex mtx_;
  std::condition_variable cv_;
  std::vector<std::size_t> free_;
  std::chrono::milliseconds acquire_timeout_{0};  // 0 = wait without a deadline
  std::size_t max_waiters_ = 0;                   // 0 = no cap
  std::size_t waiters_ = 0;                       // guarded by mtx_

  // Per-slot lease start. steady_clock::time_point{} (epoch) means "not leased"
  // — a sentinel rather than a parallel bool vector, so the two cannot disagree
  // about whether a slot is out. All guarded by mtx_.
  std::vector<std::chrono::steady_clock::time_point> leased_since_;
  // Whether a slot has ALREADY been counted stuck, so a lease wedged for an
  // hour contributes 1 to the counter and one log line, not one per scrape.
  std::vector<bool> stuck_reported_;
  std::chrono::milliseconds stuck_threshold_{0};  // 0 = detection disabled
  std::uint64_t stuck_leases_ = 0;                // guarded by mtx_
};

// Build the shared InferFunc over `pool`. Every downstream route (HTTP/gRPC)
// consumes the returned std::function unchanged.
//
// CROSS-REQUEST DETECTION BATCHING is applied here, and this is the only place
// it CAN be applied: `pool` is the object that knows about more than one
// in-flight request. Each lease still gets a whole replica (slot isolation is
// unchanged — one request, one pipeline, one queue, one set of scratch
// buffers), but the replicas now share ONE DetectionBatcher, so two requests
// that are simultaneously inside `run_with_layout` meet in a single
// IDetector::run_batch submission instead of issuing two batch-1 forward passes.
//
// It is a no-op unless the backend opted in (BackendCaps::preferred_batch_size /
// IDetector::max_batch_size, or the TURBO_DET_BATCH env override): with no
// batcher installed every pipeline calls det_->run() directly, as it always did.
[[nodiscard]] server::InferFunc make_infer_func(std::shared_ptr<UnifiedPipelinePool> pool);

// Encoded-bytes twin — see server::EncodedInferFunc. Prefer it in any route that
// still holds the undecoded body; it is the only way a backend's on-device
// decoder (nvJPEG / vImage) is reachable at all.
[[nodiscard]] server::EncodedInferFunc
make_encoded_infer_func(std::shared_ptr<UnifiedPipelinePool> pool,
                        backend::Backend &backend);

// Single-crop, single-backend inference over `pool` — the seam POST /infer
// needs. It leases a replica exactly like make_infer_func and calls
// UnifiedOcrPipeline::infer_one, which exists on every backend (its own
// diagnostic names "/infer"; the route is registered — infer_route.cpp).
//
// `inline_spec` may be null (use the named backend); when non-null the callee
// builds a transient recognizer from it. Ownership stays with the caller for the
// duration of the call.
[[nodiscard]] server::InferOneFunc
make_infer_one_func(std::shared_ptr<UnifiedPipelinePool> pool);

// NOTE (removed): there used to be a `make_infer_func(pool, cfg)` overload and an
// `install_detection_batching(pool, cfg)` helper here. Both were dead (no caller
// in src/, tests/ or tools/ — server_main.cpp:115 uses the one-argument form) AND
// both were wrong: they called configure_detection_batching(cfg), which RESETS
// the installed batcher, and then constructed a SECOND, unregistered
// DetectionBatcher to push onto the pool's entries. Any replica built afterwards
// asked shared_detection_batcher() and got a THIRD object with the same config —
// two coalescing cohorts where stage_batcher.h promises one — while
// current_detection_batcher() (the only stats consumer) saw null.
//
// To configure batching explicitly, call
// pipeline::configure_detection_batching(cfg) BEFORE the replicas are built; the
// UnifiedOcrPipeline constructor then picks the single installed batcher up.

} // namespace turbo_ocr::pipeline
