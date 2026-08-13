// UnifiedPipelinePool — the replica lease pool: admission, timeout, shedding
// and the wedged-lease detector.
//
// Split out of make_infer_func.cpp, which held BOTH this class and the InferFunc
// factories. They are separate concerns — the pool bounds access to device
// replicas, the factories decide what to run on one — and keeping them in one TU
// meant anything linking the pool also pulled in finalize_deferred and the whole
// result-serialization stack. That is what stopped tests/cpp/pipeline/
// test_pipeline_pool.cpp from linking against it.

#include "turbo_ocr/pipeline/unified/make_infer_func.h"

#include <cstdlib>

#include "turbo_ocr/base/env_utils.h" // env::env_or — records the read
#include <numeric>
#include <string>
#include <utility>

#include "turbo_ocr/base/errors.h"      // PoolExhaustedError
#include "turbo_ocr/base/log/logger.h"  // stuck-lease reporting

namespace turbo_ocr::pipeline {

namespace {

// Env-tunable admission bounds. Read once; see the class comment for why these
// two and why the defaults are generous.
std::chrono::milliseconds default_acquire_timeout() {
  static const long long ms = [] {
    const std::string sv = env::env_or("TURBO_POOL_ACQUIRE_TIMEOUT_MS", "");
    const char *v = sv.empty() ? nullptr : sv.c_str();
    if (!v || !v[0]) return 30000LL;
    char *end = nullptr;
    const long long parsed = std::strtoll(v, &end, 10);
    return (end == v || parsed < 0) ? 30000LL : parsed;
  }();
  return std::chrono::milliseconds(ms);
}

std::size_t default_max_waiters(std::size_t pool_size) {
  static const long long cfg = [] {
    const std::string sv = env::env_or("TURBO_POOL_MAX_WAITERS", "");
    const char *v = sv.empty() ? nullptr : sv.c_str();
    if (!v || !v[0]) return -1LL;  // -1 => derive from pool size
    char *end = nullptr;
    const long long parsed = std::strtoll(v, &end, 10);
    return (end == v || parsed < 0) ? -1LL : parsed;
  }();
  if (cfg >= 0) return static_cast<std::size_t>(cfg);
  return pool_size * 8;
}

// TURBO_POOL_STUCK_LEASE_MS — how long a lease may be held before it is
// reported stuck. Default 0 (off): the right value depends on the slowest
// legitimate request a deployment serves, and a threshold guessed too low turns
// a slow 400-page PDF into a false alarm.
std::chrono::milliseconds default_stuck_threshold() {
  const std::string sv = env::env_or("TURBO_POOL_STUCK_LEASE_MS", "");
  const char *v = sv.empty() ? nullptr : sv.c_str();
  if (!v || !*v) return std::chrono::milliseconds{0};
  char *end = nullptr;
  const long long parsed = std::strtoll(v, &end, 10);
  if (end == v || parsed < 0) return std::chrono::milliseconds{0};
  return std::chrono::milliseconds{parsed};
}

}  // namespace

UnifiedPipelinePool::UnifiedPipelinePool(std::vector<UnifiedPipelineEntry> entries)
    : UnifiedPipelinePool(std::move(entries), default_acquire_timeout(),
                          std::size_t{0}) {
  // The waiter cap depends on the pool size, which is only known after the
  // delegating constructor has moved `entries` in.
  max_waiters_ = default_max_waiters(entries_.size());
}

UnifiedPipelinePool::UnifiedPipelinePool(std::vector<UnifiedPipelineEntry> entries,
                                         std::chrono::milliseconds acquire_timeout,
                                         std::size_t max_waiters)
    : entries_(std::move(entries)), acquire_timeout_(acquire_timeout),
      max_waiters_(max_waiters) {
  free_.resize(entries_.size());
  std::iota(free_.begin(), free_.end(), std::size_t{0});
  // Epoch = "not leased"; see the member comment.
  leased_since_.assign(entries_.size(), std::chrono::steady_clock::time_point{});
  stuck_reported_.assign(entries_.size(), false);
  stuck_threshold_ = default_stuck_threshold();
}

UnifiedPipelinePool::Lease UnifiedPipelinePool::acquire() {
  std::unique_lock<std::mutex> lk(mtx_);
  if (free_.empty()) {
    // SHED BEFORE PARKING. Rejecting here costs the caller one 503; admitting it
    // costs a blocked WorkPool thread for the whole deadline and still ends in a
    // failure the client sees as a timeout.
    if (max_waiters_ > 0 && waiters_ >= max_waiters_)
      throw turbo_ocr::PoolExhaustedError(
          "Server at capacity (" + std::to_string(waiters_) +
          " request(s) already waiting for " + std::to_string(entries_.size()) +
          " pipeline replica(s)); retry with backoff");
    ++waiters_;
    // Decrement on EVERY exit — timeout, notification, or a throw out of wait.
    struct WaiterGuard {
      std::size_t *n;
      ~WaiterGuard() { --*n; }
    } guard{&waiters_};
    if (acquire_timeout_.count() > 0) {
      if (!cv_.wait_for(lk, acquire_timeout_, [this] { return !free_.empty(); }))
        throw turbo_ocr::PoolExhaustedError(
            "Timed out after " + std::to_string(acquire_timeout_.count()) +
            "ms waiting for a free pipeline replica");
    } else {
      cv_.wait(lk, [this] { return !free_.empty(); });
    }
  }
  const std::size_t idx = free_.back();
  free_.pop_back();
  // Stamp under the SAME lock that hands out the slot, so a scrape can never
  // observe a leased slot with no start time (which would read as age 0 and
  // hide exactly the lease that is stuck).
  leased_since_[idx] = std::chrono::steady_clock::now();
  stuck_reported_[idx] = false;
  return Lease{*this, idx};
}

std::size_t UnifiedPipelinePool::waiting() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return waiters_;
}

std::size_t UnifiedPipelinePool::available() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return free_.size();
}

std::optional<UnifiedPipelinePool::Lease>
UnifiedPipelinePool::try_acquire_for(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lk(mtx_);
  if (!cv_.wait_for(lk, timeout, [this] { return !free_.empty(); }))
    return std::nullopt;
  const std::size_t idx = free_.back();
  free_.pop_back();
  // Stamp under the SAME lock that hands out the slot, so a scrape can never
  // observe a leased slot with no start time (which would read as age 0 and
  // hide exactly the lease that is stuck).
  leased_since_[idx] = std::chrono::steady_clock::now();
  stuck_reported_[idx] = false;
  return Lease{*this, idx};
}

void UnifiedPipelinePool::release_(std::size_t idx) {
  {
    std::lock_guard<std::mutex> lk(mtx_);
    free_.push_back(idx);
    // Clear the stamp under the same lock. A slot that comes back is by
    // definition no longer stuck — including one already counted: the counter is
    // monotonic (it records that it HAPPENED), while oldest_lease_age() reflects
    // only what is outstanding now.
    leased_since_[idx] = std::chrono::steady_clock::time_point{};
    stuck_reported_[idx] = false;
  }
  cv_.notify_one();
}

std::chrono::milliseconds UnifiedPipelinePool::oldest_lease_age() const {
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(mtx_);
  std::chrono::milliseconds oldest{0};
  for (const auto &since : leased_since_) {
    if (since == std::chrono::steady_clock::time_point{}) continue;  // not leased
    const auto age =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - since);
    if (age > oldest) oldest = age;
  }
  return oldest;
}

std::uint64_t UnifiedPipelinePool::stuck_leases() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return stuck_leases_;
}

void UnifiedPipelinePool::check_stuck_leases() {
  if (stuck_threshold_.count() <= 0) return;  // detector disarmed
  const auto now = std::chrono::steady_clock::now();
  // Collect under the lock, LOG outside it: the logger can block on I/O, and
  // holding the pool mutex across that would stall every acquire() and release()
  // — turning a diagnostic for one wedged slot into a stall of the whole pool.
  std::vector<std::pair<std::size_t, long long>> newly_stuck;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    for (std::size_t i = 0; i < leased_since_.size(); ++i) {
      if (leased_since_[i] == std::chrono::steady_clock::time_point{}) continue;
      if (stuck_reported_[i]) continue;  // counted once per lease, not per scrape
      const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
          now - leased_since_[i]);
      if (age < stuck_threshold_) continue;
      stuck_reported_[i] = true;
      ++stuck_leases_;
      newly_stuck.emplace_back(i, age.count());
    }
  }
  for (const auto &[idx, age_ms] : newly_stuck)
    TOCR_LOG_ERROR("pipeline replica appears wedged (lease held past the stuck "
                   "threshold; the request is unlikely to return and this "
                   "replica is effectively lost until the process restarts)",
                   "replica", idx, "held_ms", age_ms, "threshold_ms",
                   static_cast<long long>(stuck_threshold_.count()),
                   "pool_size", entries_.size());
}

} // namespace turbo_ocr::pipeline
