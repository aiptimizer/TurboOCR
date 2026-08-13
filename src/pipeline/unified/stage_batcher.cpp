// stage_batcher — see the header for the design and for why it is a
// leader/follower rendezvous rather than a worker pool.

#include "turbo_ocr/pipeline/unified/stage_batcher.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"

namespace turbo_ocr::pipeline {
namespace {

using clk = std::chrono::steady_clock;

std::uint64_t ns_since(clk::time_point t) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count());
}

// The explicit override, when a caller configured one. Guarded because the
// server bootstrap and the bench both set it before spawning replicas.
std::mutex &cfg_mu() {
  static std::mutex m;
  return m;
}
DetBatchConfig &explicit_cfg() {
  static DetBatchConfig c{};
  return c;
}
bool &explicit_set() {
  static bool b = false;
  return b;
}
// ONE SLOT PER backend::DeviceKind — see the header. A process-global slot would
// hand the second backend in a multi-backend binary a batcher keyed on nothing,
// and its leader would gather views from a foreign address space.
using InstallTable = std::array<std::shared_ptr<DetectionBatcher>, 8>;
InstallTable &install_table() {
  static InstallTable t;
  return t;
}
std::size_t kind_index(backend::DeviceKind k) {
  const auto i = static_cast<std::size_t>(k);
  return i < std::tuple_size<InstallTable>::value ? i : 0;
}
std::shared_ptr<DetectionBatcher> &installed(backend::DeviceKind k) {
  return install_table()[kind_index(k)];
}

} // namespace

// ---------------------------------------------------------------------------
// Config resolution
// ---------------------------------------------------------------------------

DetBatchConfig resolve_det_batching(const backend::BackendCaps &caps,
                                    const backend::IDetector *det) {
  {
    std::lock_guard<std::mutex> lk(cfg_mu());
    if (explicit_set()) return explicit_cfg();
  }

  // What the BACKEND advises. The two inputs are NOT interchangeable and the
  // difference decides which one may switch batching on:
  //
  //   max_batch_size()      CAPABILITY — the hard ceiling the stage can
  //                         physically accept (a TRT profile, a pre-sized batch
  //                         canvas). Saying 8 means "8 will not crash".
  //   preferred_batch_size  POLICY — the vendor's judgement that coalescing to
  //                         N is actually FASTER on its device. stages.h says
  //                         sizing it is "a per-device judgement the shared
  //                         layer must not make".
  //
  // So POLICY enables and CAPABILITY caps. This used to be the other way round:
  // any detector that reported a ceiling above 1 switched the rendezvous on by
  // itself, which made an honest capability report a scheduling change. NVIDIA's
  // detector is the case in point — detection::PaddleDet has had a working
  // batched path with kMaxBatchSize == 8 since before the seam, and the wrapper
  // could not forward that number without silently rerouting every request on a
  // vendor arm through the coalescer. The ceiling is now safe to state.
  const int det_cap = det ? det->max_batch_size() : 1;
  const int advised = caps.preferred_batch_size > 1
                          ? std::min(caps.preferred_batch_size, det_cap)
                          : 1;

  DetBatchConfig c;
  // Upper bound is one second: this is a coalescing window held by a request
  // thread, so anything larger is a hang expressed as a tuning knob.
  c.max_queue_delay_us = env::env_int("TURBO_DET_BATCH_DELAY_US", 0, 0, 1'000'000);

  const std::string e = env::env_or("TURBO_DET_BATCH", "auto");
  if (e == "auto") {
    c.preferred_batch_size = advised;
    c.enabled = advised > 1;
    return c;
  }
  if (e == "off") return c; // enabled stays false
  const int n = env::env_int("TURBO_DET_BATCH", 0, 0, 4096);
  if (n <= 0) return c;
  // An explicit N is honoured even when the detector reports max_batch_size()==1:
  // that is how the coalescer's own cost is measured on a backend whose detector
  // cannot yet accept a real batch (run_batch then loops run(), so the RESULTS
  // are identical and only the scheduling differs). N==1 is instrument-only.
  c.preferred_batch_size = n;
  c.enabled = true;
  return c;
}

void configure_detection_batching(const DetBatchConfig &cfg) {
  // The old generation, kept alive until after the lock is dropped (see below).
  InstallTable old;
  {
    std::lock_guard<std::mutex> lk(cfg_mu());
    explicit_cfg() = cfg;
    explicit_set() = true;
    // Drop the previous generation (EVERY device's) so the next replica built
    // picks up `cfg`. UNDER cfg_mu(): every other access to the table takes that
    // lock, and clearing it unlocked races shared_detection_batcher, which reads
    // installed(dev), decides it is empty and then assigns to the same slot.
    old.swap(install_table());
  }
  // `old` — and therefore the last reference to each retired batcher — dies HERE,
  // with cfg_mu() released. ~DetectionBatcher runs arbitrary teardown (a mutex,
  // two condition variables, a deque of slots), and running it under the global
  // config lock would block every other backend's shared_detection_batcher for
  // the duration for no reason.
}

std::shared_ptr<DetectionBatcher>
shared_detection_batcher(const backend::BackendCaps &caps,
                         const backend::IDetector *det) {
  const backend::DeviceKind dev = caps.device;
  std::unique_lock<std::mutex> lk(cfg_mu());
  if (installed(dev)) return installed(dev);
  const bool have_override = explicit_set();
  DetBatchConfig c = explicit_cfg();
  if (!have_override) {
    // resolve_det_batching takes cfg_mu() itself — drop it across the call.
    lk.unlock();
    c = resolve_det_batching(caps, det);
    lk.lock();
    if (installed(dev)) return installed(dev);
  }
  if (!c.enabled || c.preferred_batch_size < 1) return nullptr;
  installed(dev) = std::make_shared<DetectionBatcher>(c.preferred_batch_size,
                                                      c.max_queue_delay_us);
  return installed(dev);
}

std::shared_ptr<DetectionBatcher> current_detection_batcher() {
  std::lock_guard<std::mutex> lk(cfg_mu());
  for (const auto &b : install_table())
    if (b) return b;
  return nullptr;
}

std::shared_ptr<DetectionBatcher>
current_detection_batcher(backend::DeviceKind device) {
  std::lock_guard<std::mutex> lk(cfg_mu());
  return installed(device);
}

// ---------------------------------------------------------------------------
// DetectionBatcher
// ---------------------------------------------------------------------------

DetectionBatcher::DetectionBatcher(int preferred_batch_size,
                                   int max_queue_delay_us)
    : max_batch_(std::max(1, preferred_batch_size)),
      delay_us_(std::max(0, max_queue_delay_us)) {}

void DetectionBatcher::submit_one_each_(backend::IDetector &det,
                                        backend::DeviceQueue &queue,
                                        const std::vector<Slot *> &batch) noexcept {
  for (Slot *s : batch) {
    try {
      s->out = det.run(*s->view, s->orig_h, s->orig_w, queue);
    } catch (...) {
      s->err = std::current_exception();
      s->out.clear();
      n_fail_.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

void DetectionBatcher::submit_(backend::IDetector &det,
                               backend::DeviceQueue &queue,
                               const std::vector<Slot *> &batch) noexcept {
  const std::size_t n = batch.size();
  if (n == 0) return;
  const auto t0 = clk::now();

  if (n == 1) {
    // EXACTLY today's call: one image, the caller's own detector, the caller's
    // own queue, no gather, no scatter, no copy. This is the whole degrade-safely
    // requirement, and at K=1 it is every single call.
    submit_one_each_(det, queue, batch);
  } else {
    bool ok = false;
    bool threw = false;
    std::size_t got = 0;
    std::string why;
    std::vector<std::vector<turbo_ocr::Box>> res;
    try {
      // The GATHER IS INSIDE THE TRY on purpose. A bad_alloc from these
      // reserves/push_backs is exactly the "not attributable to one image" case
      // the per-slot fallback below already handles — and this function is
      // noexcept, so letting it escape would terminate rather than degrade.
      std::vector<backend::ImageView> views;
      std::vector<std::pair<int, int>> dims;
      views.reserve(n);
      dims.reserve(n);
      for (const Slot *s : batch) {
        views.push_back(*s->view);
        dims.emplace_back(s->orig_h, s->orig_w);
      }
      res = det.run_batch(views, dims, queue);
      got = res.size();
      ok = got == n;
    } catch (const std::exception &e) {
      threw = true;
      // what() itself may throw (the copy into `why` can allocate). Losing the
      // message is acceptable — `threw` is already set, so the failure is still
      // reported; losing the exception path is not.
      try { why = e.what(); } catch (...) { // NOLINT(bugprone-empty-catch)
      }
      ok = false; // fall through to per-slot isolation below
    } catch (...) {
      threw = true;
      ok = false;
    }
    if (ok) {
      for (std::size_t i = 0; i < n; ++i) batch[i]->out = std::move(res[i]);
    } else {
      // ERROR ISOLATION. A throw out of run_batch is not attributable to any one
      // image, and the default run_batch (a loop over run()) loses the results of
      // the images that already succeeded. Re-run each slot on its own so the
      // failure lands on — and only on — the image that caused it.
      //
      // Report it. A detector whose batched path fails on EVERY call degrades to
      // batch-1 forever, which is the entire point of this class silently gone —
      // and n_fail_ never moves, because the per-slot re-run succeeds.
      n_batch_fallback_.fetch_add(1, std::memory_order_relaxed);
      if (threw) {
        TOCR_LOG_WARN_RL("detector run_batch threw; falling back to per-slot",
                         "batch", static_cast<long long>(n), "error",
                         why.empty() ? "unknown exception" : why.c_str());
      } else {
        // NOT an exception: a contract violation. stages.h says run_batch
        // returns EXACTLY imgs.size() entries.
        TOCR_LOG_WARN_RL("detector run_batch returned the wrong result count; "
                         "falling back to per-slot",
                         "batch", static_cast<long long>(n), "returned",
                         static_cast<long long>(got));
      }
      submit_one_each_(det, queue, batch);
    }
  }

  const std::uint64_t dt = ns_since(t0);
  n_det_ns_.fetch_add(dt, std::memory_order_relaxed);
  n_batches_.fetch_add(1, std::memory_order_relaxed);
  n_sum_.fetch_add(n, std::memory_order_relaxed);
  std::uint64_t prev = n_max_.load(std::memory_order_relaxed);
  while (n > prev &&
         !n_max_.compare_exchange_weak(prev, n, std::memory_order_relaxed)) {}
}

std::vector<turbo_ocr::Box>
DetectionBatcher::detect(backend::IDetector &det, backend::DeviceQueue &queue,
                         const backend::ImageView &view, int orig_h, int orig_w) {
  n_images_.fetch_add(1, std::memory_order_relaxed);

  // INSTRUMENT-ONLY: no queue, no mutex, no rendezvous — the call is inline and
  // identical to det.run(), and only its duration is recorded. This exists so the
  // "batching off" arm of an A/B reports det ms/call on the same footing as the
  // "on" arm without paying for a scheduler it isn't using.
  if (max_batch_ <= 1) {
    const auto t0 = clk::now();
    std::vector<turbo_ocr::Box> out;
    std::exception_ptr err;
    try {
      out = det.run(view, orig_h, orig_w, queue);
    } catch (...) {
      err = std::current_exception();
      n_fail_.fetch_add(1, std::memory_order_relaxed);
    }
    n_det_ns_.fetch_add(ns_since(t0), std::memory_order_relaxed);
    n_batches_.fetch_add(1, std::memory_order_relaxed);
    n_sum_.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t prev = n_max_.load(std::memory_order_relaxed);
    while (prev < 1 &&
           !n_max_.compare_exchange_weak(prev, 1, std::memory_order_relaxed)) {}
    if (err) std::rethrow_exception(err);
    return out;
  }

  Slot me;
  me.view = &view;
  me.orig_h = orig_h;
  me.orig_w = orig_w;

  const auto t_enq = clk::now();
  std::unique_lock<std::mutex> lk(mu_);
  q_.push_back(&me);
  arrive_.notify_all(); // a lingering leader may now have enough to go

  // `&me` is THIS stack frame, just published into the shared queue. On any
  // exceptional exit while it is still queued — a leader-path throw before
  // this slot was drained into `batch` (batch.reserve/arrive_.wait_for), or a
  // follower's cv_.wait throwing — the frame dies but the pointer stays, and
  // a LATER leader drains it and writes into freed stack memory. Unlink on
  // unwind, under the lock; dismissed on the normal exit (a done slot was
  // already drained by its leader).
  struct UnlinkGuard {
    DetectionBatcher *self;
    std::unique_lock<std::mutex> *lk;
    Slot *me;
    bool dismissed = false;
    ~UnlinkGuard() {
      if (dismissed) return;
      try {
        if (!lk->owns_lock()) lk->lock();
        auto &q = self->q_;
        q.erase(std::remove(q.begin(), q.end(), me), q.end());
      } catch (...) { // NOLINT(bugprone-empty-catch)
        // A destructor may not throw; if re-locking mu_ failed the process is
        // already unrecoverable.
      }
    }
  } unlink_guard{this, &lk, &me};

  for (;;) {
    if (me.done) break;

    if (!leader_) {
      leader_ = true;

      std::vector<Slot *> batch;

      // LEADERSHIP IS RAII. Between here and the handover below the leader
      // allocates (batch.reserve, the gather inside submit_) and re-locks mu_,
      // and any of that can throw. If it escaped with leader_ still true and the
      // claimed slots' `done` still false, every follower would be parked on
      // `me.done || !leader_` — a predicate that can never become true again —
      // and every future caller would join them. So: on ANY exceptional exit,
      // hand leadership back and release every slot this thread claimed, with
      // the in-flight exception attached so each caller rethrows it in its own
      // thread (exactly what it would have seen unbatched).
      struct LeaderGuard {
        DetectionBatcher *self;
        std::unique_lock<std::mutex> *lk;
        std::vector<Slot *> *batch;
        bool handed_over = false;
        ~LeaderGuard() {
          if (handed_over) return;
          try {
            std::exception_ptr err = std::current_exception();
            if (!err)
              err = std::make_exception_ptr(std::runtime_error(
                  "detection batch leader aborted before submission"));
            if (!lk->owns_lock()) lk->lock();
            for (Slot *s : *batch) {
              if (s->done) continue;
              if (!s->err) s->err = err;
              s->out.clear();
              s->done = true;
            }
            self->leader_ = false;
            self->cv_.notify_all();
          } catch (...) { // NOLINT(bugprone-empty-catch) — reason below
            // A destructor may not throw. Nothing left to do but leave the
            // slots as they are; the process is already in an unrecoverable
            // state if re-locking mu_ failed.
          }
        }
      } leader_guard{this, &lk, &batch};

      // Triton's max_queue_delay_microseconds. DEFAULT 0 => this block never
      // runs and the batch is whatever was ALREADY queued. A nonzero linger is
      // opt-in and measured harmful on this machine; see the header.
      if (delay_us_ > 0 && static_cast<int>(q_.size()) < max_batch_) {
        arrive_.wait_for(lk, std::chrono::microseconds(delay_us_), [this] {
          return static_cast<int>(q_.size()) >= max_batch_;
        });
      }

      // Take only slots whose view lives in the SAME address space as this
      // leader's own — ImageView::data is valid only in the space named by
      // ImageView::kind, so a leader must never hand a foreign backend's pointer
      // to its own detector. Slots of another kind stay queued and are served by
      // a leader of that kind. `me` always matches, so progress is guaranteed,
      // and in a single-backend process every slot matches and this is exactly
      // the old front-to-back drain.
      const backend::DeviceKind my_kind =
          me.view ? me.view->kind : backend::DeviceKind::Host;
      batch.reserve(std::min<std::size_t>(q_.size(), max_batch_));
      for (auto it = q_.begin();
           it != q_.end() && static_cast<int>(batch.size()) < max_batch_;) {
        if ((*it)->view && (*it)->view->kind != my_kind) {
          ++it;
          continue;
        }
        batch.push_back(*it);
        it = q_.erase(it);
      }

      lk.unlock();
      submit_(det, queue, batch); // noexcept — see the header
      lk.lock();

      for (Slot *s : batch) s->done = true;
      leader_guard.handed_over = true;
      leader_ = false;
      cv_.notify_all();
      // If the queue was deeper than max_batch_ this thread's own slot may not
      // have been in that batch — loop and either take leadership again or wait.
      continue;
    }

    // Follower: wake when my slot is done, or when leadership frees up (in which
    // case I take it, so a batch is never left un-driven).
    cv_.wait(lk, [this, &me] { return me.done || !leader_; });
  }
  unlink_guard.dismissed = true; // done => a leader already drained this slot
  lk.unlock();

  n_wait_ns_.fetch_add(ns_since(t_enq), std::memory_order_relaxed);
  if (me.err) std::rethrow_exception(me.err);
  return std::move(me.out);
}

DetectionBatcher::Stats DetectionBatcher::stats() const noexcept {
  Stats s;
  s.images = n_images_.load(std::memory_order_relaxed);
  s.batches = n_batches_.load(std::memory_order_relaxed);
  s.sum_batch = n_sum_.load(std::memory_order_relaxed);
  s.max_batch = n_max_.load(std::memory_order_relaxed);
  s.det_ns = n_det_ns_.load(std::memory_order_relaxed);
  s.wait_ns = n_wait_ns_.load(std::memory_order_relaxed);
  s.failures = n_fail_.load(std::memory_order_relaxed);
  s.batch_fallbacks = n_batch_fallback_.load(std::memory_order_relaxed);
  return s;
}

void DetectionBatcher::reset_stats() noexcept {
  n_images_ = 0;
  n_batches_ = 0;
  n_sum_ = 0;
  n_max_ = 0;
  n_det_ns_ = 0;
  n_wait_ns_ = 0;
  n_fail_ = 0;
  n_batch_fallback_ = 0;
}

std::string DetectionBatcher::stats_line() const {
  const Stats s = stats();
  const double mean_batch =
      s.batches ? static_cast<double>(s.sum_batch) / s.batches : 0.0;
  const double det_ms_per_call =
      s.batches ? s.det_ns / 1e6 / s.batches : 0.0;
  const double det_ms_per_img = s.images ? s.det_ns / 1e6 / s.images : 0.0;
  const double wait_ms_per_img = s.images ? s.wait_ns / 1e6 / s.images : 0.0;
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "det-batch: mode=%s preferred=%d delay_us=%d | images=%llu "
                "submissions=%llu mean_batch=%.2f max_batch=%llu | "
                "det=%.2f ms/submission %.2f ms/image | queue_wait=%.2f ms/image "
                "| failures=%llu batch_fallbacks=%llu",
                coalescing() ? "coalescing" : "instrument-only", max_batch_,
                delay_us_, (unsigned long long)s.images,
                (unsigned long long)s.batches, mean_batch,
                (unsigned long long)s.max_batch, det_ms_per_call, det_ms_per_img,
                wait_ms_per_img, (unsigned long long)s.failures,
                (unsigned long long)s.batch_fallbacks);
  return buf;
}

std::string DetectionBatcher::stats_json() const {
  const Stats s = stats();
  const double mean_batch =
      s.batches ? static_cast<double>(s.sum_batch) / s.batches : 0.0;
  char buf[640];
  std::snprintf(buf, sizeof buf,
                "{\"mode\":\"%s\",\"preferred_batch_size\":%d,"
                "\"max_queue_delay_us\":%d,\"images\":%llu,\"submissions\":%llu,"
                "\"mean_batch\":%.4f,\"max_batch\":%llu,"
                "\"det_ms_per_submission\":%.4f,\"det_ms_per_image\":%.4f,"
                "\"queue_wait_ms_per_image\":%.4f,\"failures\":%llu,"
                "\"batch_fallbacks\":%llu}",
                coalescing() ? "coalescing" : "instrument-only", max_batch_,
                delay_us_, (unsigned long long)s.images,
                (unsigned long long)s.batches, mean_batch,
                (unsigned long long)s.max_batch,
                s.batches ? s.det_ns / 1e6 / s.batches : 0.0,
                s.images ? s.det_ns / 1e6 / s.images : 0.0,
                s.images ? s.wait_ns / 1e6 / s.images : 0.0,
                (unsigned long long)s.failures,
                (unsigned long long)s.batch_fallbacks);
  return buf;
}

} // namespace turbo_ocr::pipeline
