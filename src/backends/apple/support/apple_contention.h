#pragma once

// Lock-free contention counters for the Apple backend.
//
// The stage profiler in apple_profile.h takes a PROCESS-GLOBAL std::mutex per
// scope, which is itself a serialization point — using it to hunt for
// serialization at K=24 measures the profiler. These counters are plain relaxed
// atomics on a per-site cache-line-padded struct, so the measurement does not
// create the phenomenon.
//
// Enabled by TURBO_APPLE_CONTENTION=1 (one static bool read per site when off).
// Dumped to stderr at process exit.
//
// Three shapes:
//   TURBO_APPLE_STAT(site)            — wall time + call count for a scope
//   TURBO_APPLE_STAT_N(site, k)       — same, plus an integer accumulator (rows)
//   TURBO_APPLE_LOCK(site, mutex)     — a std::mutex guard that separates
//                                       UNCONTENDED acquisition from time spent
//                                       BLOCKED waiting for another thread.
//
// Apple-local: it measures Metal/CoreML mechanics, not shared pipeline policy.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace turbo_ocr::apple {

class Stat {
public:
  explicit Stat(const char *name);
  void add(long long ns, long long units = 0) noexcept {
    ns_.fetch_add(ns, std::memory_order_relaxed);
    n_.fetch_add(1, std::memory_order_relaxed);
    if (units) units_.fetch_add(units, std::memory_order_relaxed);
  }
  void add_blocked(long long ns) noexcept {
    blocked_ns_.fetch_add(ns, std::memory_order_relaxed);
    blocked_n_.fetch_add(1, std::memory_order_relaxed);
  }
  void bump(long long units) noexcept {
    n_.fetch_add(1, std::memory_order_relaxed);
    units_.fetch_add(units, std::memory_order_relaxed);
  }

  const char *name_;
  std::atomic<long long> ns_{0};
  std::atomic<long long> n_{0};
  std::atomic<long long> units_{0};
  std::atomic<long long> blocked_ns_{0};
  std::atomic<long long> blocked_n_{0};
  char pad_[64];
};

bool contention_enabled();
void contention_dump();
std::vector<Stat *> &contention_registry();

using cclock = std::chrono::steady_clock;

struct StatScope {
  Stat *s;
  long long units;
  cclock::time_point t0;
  StatScope(Stat &st, long long u = 0)
      : s(contention_enabled() ? &st : nullptr), units(u) {
    if (s) t0 = cclock::now();
  }
  ~StatScope() {
    if (!s) return;
    s->add((long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
               cclock::now() - t0)
               .count(),
           units);
  }
};

// A std::mutex guard that measures how long it BLOCKED (i.e. real contention),
// distinct from the cost of an uncontended acquire.
struct StatLock {
  std::mutex &m;
  StatLock(Stat &st, std::mutex &mu) : m(mu) {
    if (!contention_enabled()) { m.lock(); return; }
    if (m.try_lock()) { st.add(0, 0); return; }
    const auto t0 = cclock::now();
    m.lock();
    const auto dt = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        cclock::now() - t0)
                        .count();
    st.add(0, 0);
    st.add_blocked(dt);
  }
  ~StatLock() { m.unlock(); }
};

} // namespace turbo_ocr::apple

#define TURBO_APPLE_STAT(site)                                                 \
  static ::turbo_ocr::apple::Stat _cstat_##site(#site);                        \
  ::turbo_ocr::apple::StatScope _cscope_##site(_cstat_##site)

#define TURBO_APPLE_STAT_N(site, k)                                            \
  static ::turbo_ocr::apple::Stat _cstat_##site(#site);                        \
  ::turbo_ocr::apple::StatScope _cscope_##site(_cstat_##site, (long long)(k))

#define TURBO_APPLE_LOCK(site, mu)                                             \
  static ::turbo_ocr::apple::Stat _cstat_##site(#site);                        \
  ::turbo_ocr::apple::StatLock _clock_##site(_cstat_##site, (mu))

#define TURBO_APPLE_BUMP(site, k)                                              \
  do {                                                                         \
    static ::turbo_ocr::apple::Stat _cstat_##site(#site);                      \
    if (::turbo_ocr::apple::contention_enabled())                              \
      _cstat_##site.bump((long long)(k));                                      \
  } while (0)
