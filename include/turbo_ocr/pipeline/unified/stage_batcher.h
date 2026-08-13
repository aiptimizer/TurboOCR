#pragma once

// stage_batcher.h — CROSS-REQUEST DYNAMIC BATCHING for the detection stage.
//
// ============================================================================
// WHY THIS EXISTS
// ============================================================================
// `IDetector::run_batch()` has existed in the seam since day one
// (include/turbo_ocr/backend/stages.h) and NO CALLER COULD EVER REACH IT
// WITH TWO DIFFERENT REQUESTS' IMAGES. The reason is structural and identical on
// every vendor: pipeline::make_infer_func leases one whole UnifiedOcrPipeline
// replica per request, each replica owns its own IDetector and DeviceQueue, and
// the replicas never talk to each other. So `run_batch` is only ever fed the N
// images of ONE caller's explicit batch call, and the server — where the
// concurrency actually is — always runs detection at batch 1.
//
// That is expensive in a way that does not show up as idle time. On the Apple
// M3 Max the GPU reads 98.5% "Device Utilization" under load, yet a 992x800
// DBNet forward takes 5.6-5.7 ms where the arithmetic says 1-2 ms: the device is
// BUSY but badly OCCUPIED, because a batch-1 convolution cannot fill the ALUs.
// There is no idle to reclaim (which is why async pipelining and extra replicas
// both measured flat) — the only remaining lever is MORE IMAGES PER SUBMISSION.
//
// ============================================================================
// THE ALGORITHM (Triton's dynamic batcher, and its default)
// ============================================================================
// Triton's dynamic batcher is exactly two knobs: `preferred_batch_size` and
// `max_queue_delay_microseconds`, and its DEFAULT DELAY IS ZERO — it is purely
// opportunistic coalescing of whatever is ALREADY queued. Under sustained load
// that is near-free, because there is always something queued. A nonzero linger
// trades latency for batch size and must be opt-in: an earlier ad-hoc "2 ms
// batch-fill linger" on this machine measured 40 img/s against a 105 img/s
// baseline. So: `max_queue_delay_us` defaults to 0 and stays there.
//
// ============================================================================
// THE SHAPE: LEADER/FOLLOWER RENDEZVOUS, NOT A WORKER POOL
// ============================================================================
// src/backends/apple/engine/ane_rec_engine.mm's AneBatchService (worth 2.9x on the Neural
// Engine) drains a queue into one pinned-shape predict from dedicated worker
// threads that own their own MLModel replicas. That works there because a CoreML
// model is cheap to duplicate. A detector is not: giving the batcher its OWN
// IDetector would mean another DBNet graph, another canvas buffer, another
// model load, and — worse — it would have to be wired in by every vendor's
// bootstrap.
//
// So this generalizes the AneBatchService's SHAPE (a queue + one submission +
// scatter, zero added delay) with a LEADER/FOLLOWER rendezvous instead of worker
// threads:
//
//   * every request thread enqueues its own (view, dims, out) slot and blocks;
//   * whichever thread finds no leader BECOMES the leader: it drains up to
//     `preferred_batch_size` pending slots (its own included) and runs them
//     through ITS OWN replica's IDetector::run_batch on ITS OWN DeviceQueue;
//   * it scatters the results back into each waiting slot and wakes them.
//
// Consequences, all of which matter:
//   - ZERO extra detectors, queues, threads, or device memory.
//   - No bootstrap wiring: any pool of UnifiedOcrPipeline replicas (the server's
//     pool AND turbo_bench's K replicas) gets it from one shared_ptr.
//   - When the queue holds exactly ONE slot the leader calls plain
//     `IDetector::run(...)` — byte-for-byte today's call, on today's queue, with
//     no extra copy and no added latency. At K=1 that is EVERY call.
//   - A backend that has not ASKED for coalescing (`preferred_batch_size == 1`,
//     which is every vendor today) never builds a coalescing batcher at all (see
//     resolve_det_batching()), so CPU, NVIDIA and today's Apple are bit-unchanged
//     until a vendor opts in or TURBO_DET_BATCH is set.
//
// ============================================================================
// ERROR ISOLATION (the requirement that shapes the code)
// ============================================================================
// One request's failure must not corrupt or fail its neighbours. Two mechanisms:
//   1. Every slot owns its own output vector; nothing is shared between slots.
//   2. If `run_batch` throws, the leader does NOT fail the batch — it re-runs
//      every slot INDIVIDUALLY through `run()`, so only the genuinely bad image
//      fails. The exception is captured per slot as an exception_ptr and
//      rethrown IN THE REQUESTER'S OWN THREAD, so a caller sees exactly the
//      exception it would have seen unbatched, and its neighbours see none.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "turbo_ocr/backend/backend.h"    // BackendCaps
#include "turbo_ocr/backend/device_queue.h"
#include "turbo_ocr/backend/image_view.h"
#include "turbo_ocr/backend/stages.h"     // IDetector
#include "turbo_ocr/base/geometry/box.h"

namespace turbo_ocr::pipeline {

// The two Triton knobs, plus the enable switch.
struct DetBatchConfig {
  // Max images coalesced into one IDetector::run_batch submission. 0 => ask the
  // backend (BackendCaps::preferred_batch_size, then IDetector::max_batch_size).
  // 1 => INSTRUMENT-ONLY: no rendezvous at all, detection runs inline exactly as
  //      it does today, and only the per-call timing is recorded. This is what
  //      makes the "batcher off" arm of an A/B measurable on the same footing.
  int preferred_batch_size = 0;
  // Triton's max_queue_delay_microseconds. 0 (the default, and Triton's) means
  // purely opportunistic: never wait for a batch to fill.
  int max_queue_delay_us = 0;
  // false => no batcher object is created at all (the absolute default).
  bool enabled = false;
};

// Resolve the effective config from (explicit config, env, backend advice).
//
// Precedence: an explicitly configured DetBatchConfig (see
// configure_detection_batching) wins; otherwise the env knobs
//   TURBO_DET_BATCH            off|0 | 1 (instrument-only) | N | auto
//   TURBO_DET_BATCH_DELAY_US   default 0
// win; otherwise the backend's own advice
//   BackendCaps::preferred_batch_size, capped by IDetector::max_batch_size()
// — POLICY enables, CAPABILITY caps; see resolve_det_batching's body for why
// that asymmetry matters. preferred_batch_size is 1 for every vendor that has
// not opted in, i.e. disabled.
[[nodiscard]] DetBatchConfig
resolve_det_batching(const backend::BackendCaps &caps,
                     const backend::IDetector *det);

class DetectionBatcher {
public:
  DetectionBatcher(int preferred_batch_size, int max_queue_delay_us);

  // true when the coalescing rendezvous is live (preferred_batch_size > 1).
  // false => instrument-only: detect() runs inline, lock-free.
  [[nodiscard]] bool coalescing() const noexcept { return max_batch_ > 1; }
  [[nodiscard]] int preferred_batch_size() const noexcept { return max_batch_; }
  [[nodiscard]] int max_queue_delay_us() const noexcept { return delay_us_; }

  // Detect on `view` (dims `orig_h` x `orig_w`), coalescing with whatever other
  // requests are pending. `det` / `queue` are THIS caller's own replica stage and
  // lane; they are used only if this caller becomes the batch leader.
  //
  // Rethrows, in the caller's thread, whatever the detector threw for THIS
  // image. Neighbours in the same batch are unaffected.
  [[nodiscard]] std::vector<turbo_ocr::Box>
  detect(backend::IDetector &det, backend::DeviceQueue &queue,
         const backend::ImageView &view, int orig_h, int orig_w);

  // Counters. `sum_batch / batches` is the achieved mean batch size — the number
  // that says whether coalescing actually happened; `det_ns / images` is det
  // ms/call amortized per image.
  struct Stats {
    std::uint64_t images = 0;     // detections requested
    std::uint64_t batches = 0;    // run_batch/run submissions made
    std::uint64_t sum_batch = 0;  // sum of submission sizes (== images)
    std::uint64_t max_batch = 0;  // largest submission seen
    std::uint64_t det_ns = 0;     // device+post time inside submissions
    std::uint64_t wait_ns = 0;    // time followers spent blocked
    std::uint64_t failures = 0;   // images whose detection threw
    // Submissions where the BATCHED call did not deliver (it threw, or it
    // returned the wrong number of results) and the batch was re-run one image
    // at a time. A detector whose batched path is permanently broken degrades to
    // batch-1 forever while every other counter still looks healthy, so this is
    // the number that says "coalescing is configured but not working".
    std::uint64_t batch_fallbacks = 0;
  };
  [[nodiscard]] Stats stats() const noexcept;
  [[nodiscard]] std::string stats_line() const;
  [[nodiscard]] std::string stats_json() const;
  void reset_stats() noexcept;

private:
  struct Slot {
    const backend::ImageView *view = nullptr;
    int orig_h = 0, orig_w = 0;
    std::vector<turbo_ocr::Box> out;
    std::exception_ptr err;
    bool done = false;
  };

  // Runs `batch` through `det` on `queue`, filling each slot. `noexcept` is the
  // CONTRACT, not a hope: the leader runs these with mu_ unlocked and with every
  // slot in `batch` already removed from the queue, so an escaping exception
  // would strand those slots (done never set) and leadership (leader_ never
  // cleared). Enforced by the compiler rather than by a comment; every failure
  // path inside is captured per slot instead.
  void submit_(backend::IDetector &det, backend::DeviceQueue &queue,
               const std::vector<Slot *> &batch) noexcept;
  void submit_one_each_(backend::IDetector &det, backend::DeviceQueue &queue,
                        const std::vector<Slot *> &batch) noexcept;

  const int max_batch_;
  const int delay_us_;

  std::mutex mu_;
  std::condition_variable cv_;     // wakes followers (done, or leadership free)
  std::condition_variable arrive_; // wakes a lingering leader on new arrivals
  std::deque<Slot *> q_;
  bool leader_ = false;

  std::atomic<std::uint64_t> n_images_{0}, n_batches_{0}, n_sum_{0}, n_max_{0},
      n_det_ns_{0}, n_wait_ns_{0}, n_fail_{0}, n_batch_fallback_{0};
};

// ---------------------------------------------------------------------------
// Process-wide installation
// ---------------------------------------------------------------------------
// One batcher is shared by every UnifiedOcrPipeline replica of one backend —
// which is the entire point (cross-REQUEST batching cannot come from an object
// that a single replica owns). UnifiedOcrPipeline's constructor asks for it, so
// both the server's pool and turbo_bench's K replicas pick it up with no wiring.
//
// ONE SLOT PER backend::DeviceKind, NOT one per process. A single binary CAN
// hold two backends (src/backend/backend_registry.cpp says linking several
// vendor registrars together "is not just legal but the point", and the Python
// bindings build a fresh Backend per Pipeline), and ImageView::data is a pointer
// valid only in the address space named by ImageView::kind. A process-global
// batcher would let a leader on one backend gather a foreign backend's views and
// hand them to its own detector. This is the same fix src/pipeline/unified/vlm_factory.cpp
// applies to the device readback table, for the same reason.
//
// configure_detection_batching() must be called BEFORE the replicas are
// constructed; passing {enabled=false} tears the batcher down again (the next
// generation of replicas gets nullptr). Not calling it at all leaves the env /
// backend-advice path in charge, whose default is disabled.
void configure_detection_batching(const DetBatchConfig &cfg);

// The batcher for `caps`/`det`, creating it on first use FOR caps.device.
// nullptr when batching resolves to disabled — in which case UnifiedOcrPipeline
// keeps calling det_->run() directly and nothing at all changes.
[[nodiscard]] std::shared_ptr<DetectionBatcher>
shared_detection_batcher(const backend::BackendCaps &caps,
                         const backend::IDetector *det);

// The currently-installed batcher (nullptr when none). For reporting. With more
// than one backend live this returns the first installed slot; pass a device to
// name one exactly.
[[nodiscard]] std::shared_ptr<DetectionBatcher> current_detection_batcher();
[[nodiscard]] std::shared_ptr<DetectionBatcher>
current_detection_batcher(backend::DeviceKind device);

} // namespace turbo_ocr::pipeline
