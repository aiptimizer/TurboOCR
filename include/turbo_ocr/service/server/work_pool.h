#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string_view>
#include <thread>
#include <vector>

#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/base/log/logger.h"

namespace turbo_ocr::server {

/// Thread pool for offloading blocking HTTP handler work from Drogon's
/// event-loop threads.
///
/// submit() is non-blocking: it always enqueues and returns immediately.
///
/// BACKPRESSURE — READ THIS BEFORE RELYING ON IT.
/// This class's queue depth is the FIRST bound on in-flight work; the second is
/// UnifiedPipelinePool::acquire(), which the unified path calls to lease a
/// replica. That acquire() used to be an unbounded condition-variable wait, so a
/// saturated device parked a WorkPool thread per queued request instead of
/// shedding; it now enforces a waiter cap and a deadline and rejects with
/// PoolExhaustedError (see the class comment in make_infer_func.h), which is the
/// same rejection PipelineDispatcher made before it was deleted with the CUDA
/// pipeline.
///
/// The stuck-worker WATCHDOG is now folded in: UnifiedPipelinePool timestamps
/// every lease, and check_stuck_leases() (run on each /metrics scrape) logs and
/// counts any lease held past TURBO_POOL_STUCK_LEASE_MS. /metrics reports the
/// device-side pool alongside this queue — turbo_ocr_pool_{waiting,available,
/// oldest_lease_ms,stuck_leases} — so a server whose replicas are all wedged no
/// longer looks idle here.
///
/// Still NOT done, deliberately: REBUILDING a wedged replica. A stuck lease is
/// normally stuck inside a device call, and destroying a pipeline whose kernels
/// are still in flight is a crash rather than a recovery — safe recycling needs
/// a cancellation point in the stage seam that does not exist yet.
///
/// The pool's deadline still covers QUEUEING only, not execution: once a replica
/// is leased, a wedged stage runs unbounded. That is now observable, not fixed.
///
/// SHUTDOWN: the worker loop drains the queue before exiting, so destroying
/// the pool with a backlog would run the whole backlog. The graceful-shutdown
/// path therefore calls discard_pending() when wait_drain() times out — drop
/// what never started, finish what did — which is what makes the grace period
/// a real upper bound on teardown (minus a wedged in-flight task, which is the
/// observable-not-fixed case above).
///
/// Queue depth is bounded (default 8192) as a safety net against memory
/// exhaustion.  When full, submit() throws PoolExhaustedError.
class WorkPool {
public:
  explicit WorkPool(int num_threads, size_t max_depth = 8192)
      : max_depth_(max_depth) {
    workers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock lock(mutex_);
            // Predicate form is only safe against lost wakeups when the
            // notifier mutates state under the same mutex — see dtor.
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
              if (stop_) return;
              continue;
            }
            task = std::move(queue_.front());
            queue_.pop();
            ++inflight_;
          }
          // RAII inflight decrement: even if `task()` escapes with an
          // exception, inflight_ MUST drop back to 0 — otherwise
          // wait_drain() hangs forever (the graceful-shutdown path
          // depends on inflight_ reaching 0). This is a FIRST line of
          // defence, not a spare one: submit_work() wraps the CALLBACK in a
          // shared_ptr and catches only the PoolExhaustedError that submit()
          // itself throws — the run_with_error_handling around the task body
          // is a convention each route opts into, not something submit_work()
          // supplies. Call sites that submit raw lambdas exist today
          // (/health/ready and both /ocr/stream submits), and the readiness
          // one carries no exception handler of its own at all.
          struct InflightGuard {
            WorkPool *self;
            ~InflightGuard() noexcept {
              std::lock_guard lock(self->mutex_);
              --self->inflight_;
              if (self->inflight_ == 0 && self->queue_.empty())
                self->drain_cv_.notify_all();
            }
          } guard{this};
          // A task escaping is a bug at the call site, but we cannot let it
          // kill this worker (the pool would slowly bleed threads until
          // wait_drain() hangs) — so it is caught here. It must still be
          // LOGGED: an escaped task means its DrogonCallback was never
          // invoked, so the HTTP client blocks until its own timeout with no
          // server-side trace. Rate-limited because a systematically broken
          // call site would otherwise emit one line per request; the window
          // rollup still names this site.
          //
          // Rate-limiting is LOSSY by construction, so the log alone cannot
          // answer "how many clients did we wedge?". escaped_tasks_ counts
          // every one — that counter, not the log, is what /metrics scrapes
          // and what an alert can be built on.
          try {
            task();
          } catch (const std::exception &e) {
            escaped_tasks_.fetch_add(1, std::memory_order_relaxed);
            TOCR_LOG_ERROR_RL("WorkPool task escaped (callback may never fire)",
                              "error", std::string_view(e.what()));
          } catch (...) {
            escaped_tasks_.fetch_add(1, std::memory_order_relaxed);
            TOCR_LOG_ERROR_RL("WorkPool task escaped with a non-standard "
                              "exception (callback may never fire)");
          }
        }
      });
    }
  }

  ~WorkPool() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &w : workers_)
      if (w.joinable()) w.join();
  }

  WorkPool(const WorkPool &) = delete;
  WorkPool &operator=(const WorkPool &) = delete;

  void submit(std::function<void()> fn) {
    {
      std::lock_guard lock(mutex_);
      // After discard_pending() the pool no longer accepts work: a submit
      // that slipped in behind the discard would be silently re-queued and
      // drained by the destructor — reopening exactly the unbounded-teardown
      // hole discard_pending() exists to close. Rejecting throws the same
      // error the full-queue path throws, so the caller's 503 machinery
      // already handles it (the process is quitting; a shed request is the
      // correct answer).
      if (discarding_)
        throw turbo_ocr::PoolExhaustedError(
            "Server is shutting down (grace period expired); request not "
            "accepted.");
      if (queue_.size() >= max_depth_)
        throw turbo_ocr::PoolExhaustedError(
            "Server at capacity (work queue full). Use persistent connections "
            "(HTTP keep-alive) instead of opening a new connection per request.");
      queue_.push(std::move(fn));
    }
    cv_.notify_one();
  }

  /// Shutdown backstop: drop every QUEUED (not yet started) task and refuse
  /// all further submits. In-flight tasks are untouched — they run to
  /// completion and the destructor still joins them.
  ///
  /// This is what makes SHUTDOWN_GRACE_SECONDS a real bound instead of an
  /// advisory one: the worker loop only exits on {stop_, queue empty}, so a
  /// destructor reached with a deep backlog used to RUN the entire backlog —
  /// up to max_depth_ tasks of teardown, far outside the grace window, until
  /// the orchestrator's SIGKILL cut the process mid-request. After this call
  /// the destructor has nothing to drain, so teardown time is bounded by the
  /// in-flight tail alone.
  ///
  /// A discarded task's HTTP callback never fires; the client sees the
  /// connection close when the process exits — the same outcome SIGKILL gave,
  /// minus the mid-request kill of the in-flight work. Each drop is counted
  /// (discarded_tasks(), scraped by /metrics) and the caller logs the total,
  /// so shed load at shutdown is observable, never silent.
  ///
  /// Returns the number of tasks dropped. Idempotent; safe to race with
  /// submit()/workers (single mutex).
  size_t discard_pending() {
    std::queue<std::function<void()>> dropped;
    {
      std::lock_guard lock(mutex_);
      discarding_ = true;
      dropped.swap(queue_);
      if (inflight_ == 0) drain_cv_.notify_all();  // nothing left at all
    }
    // Destroy the dropped closures OUTSIDE the lock: a closure owns request
    // state (body buffers, callbacks) whose destructors need no queue mutex
    // and must not stall submitters or the worker loop while they run.
    const size_t n = dropped.size();
    discarded_tasks_.fetch_add(n, std::memory_order_relaxed);
    return n;
  }

  /// Saturation snapshot for /metrics. Both reads take the same mutex as
  /// submit()/the worker loop so the values are consistent (no torn reads of
  /// inflight_/queue_, which are guarded by mutex_, not atomic).
  [[nodiscard]] size_t queue_depth() const {
    std::lock_guard lock(mutex_);
    return queue_.size();
  }

  [[nodiscard]] size_t inflight() const {
    std::lock_guard lock(mutex_);
    return inflight_;
  }

  /// Worker-thread count this pool was built with — the derived number the
  /// startup log reports, so it comes from the pool rather than being recomputed
  /// by the caller and risking a rule change that only updates one of the two.
  [[nodiscard]] size_t num_threads() const { return workers_.size(); }

  /// Configured ceiling for queue_depth() (the point at which submit()
  /// rejects with PoolExhaustedError).
  [[nodiscard]] size_t max_depth() const { return max_depth_; }

  /// Monotonic count of tasks that escaped with an exception. Each one is an
  /// HTTP callback that was never invoked, so the client blocks until its own
  /// timeout — a nonzero, growing value here is the wedged-client signal.
  ///
  /// A counter rather than a straight Metrics::instance() call because
  /// server/metrics.h includes THIS header: the dependency has to point
  /// outward, so the metrics layer polls this the same way it polls
  /// queue_depth()/inflight() at scrape time.
  ///
  /// Deliberately a plain atomic and NOT guarded by mutex_ — the worker's
  /// catch block must not have to take the queue lock, and unlike
  /// inflight_/queue_ this value is never read together with another field,
  /// so there is nothing to tear.
  [[nodiscard]] uint64_t escaped_tasks() const {
    return escaped_tasks_.load(std::memory_order_relaxed);
  }

  /// Monotonic count of queued tasks dropped by discard_pending() at
  /// shutdown. Same atomic-not-mutex rationale as escaped_tasks(). A nonzero
  /// value on a scrape just before termination says how much load was shed
  /// past the grace window.
  [[nodiscard]] uint64_t discarded_tasks() const {
    return discarded_tasks_.load(std::memory_order_relaxed);
  }

  /// Block until queue is empty and no task is in flight, OR timeout
  /// elapses. Returns true on full drain, false on timeout. Used by the
  /// graceful-shutdown path: caller stops admitting new work first, then
  /// waits here for inflight to finish before tearing down Drogon.
  bool wait_drain(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return drain_cv_.wait_for(lock, timeout, [this] {
      return queue_.empty() && inflight_ == 0;
    });
  }

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable drain_cv_;
  bool stop_{false};        // guarded by mutex_
  bool discarding_{false};  // guarded by mutex_; set once by discard_pending()
  size_t inflight_{0};      // guarded by mutex_
  size_t max_depth_;
  std::atomic<uint64_t> escaped_tasks_{0};    // lock-free; see escaped_tasks()
  std::atomic<uint64_t> discarded_tasks_{0};  // lock-free; see discarded_tasks()
};

} // namespace turbo_ocr::server
