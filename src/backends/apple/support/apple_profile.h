#pragma once

// Tiny env-gated stage profiler for the Apple backend.
//
// TURBO_APPLE_PROFILE=1 accumulates wall time under named counters and dumps a
// table at process exit. Zero cost when the env var is unset (one atomic bool
// read per scope). Used to answer "where did the 3x vs the standalone harness
// go" without dragging in a real tracing dependency.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "turbo_ocr/base/env_utils.h"

namespace turbo_ocr::apple {

class Profiler {
public:
  static Profiler &get() {
    static Profiler p;
    return p;
  }
  static bool enabled() {
    static const bool on = [] {
      const std::string e = env::env_or("TURBO_APPLE_PROFILE", "");
      return !e.empty() && e != "0";
    }();
    return on;
  }
  void add(const char *name, double ms, long count = 1) {
    std::lock_guard<std::mutex> lk(mu_);
    auto &e = acc_[name];
    e.ms += ms;
    e.n += count;
  }
  void dump() {
    std::lock_guard<std::mutex> lk(mu_);
    if (acc_.empty()) return;
    std::vector<std::pair<std::string, Entry>> v(acc_.begin(), acc_.end());
    std::sort(v.begin(), v.end(),
              [](auto &a, auto &b) { return a.second.ms > b.second.ms; });
    std::fprintf(stderr, "\n=== apple profile (total ms / calls / ms-per-call) ===\n");
    for (auto &[k, e] : v)
      std::fprintf(stderr, "  %-28s %9.1f  %7ld  %8.3f\n", k.c_str(), e.ms, e.n,
                   e.n ? e.ms / e.n : 0.0);
    std::fprintf(stderr, "======================================================\n");
    acc_.clear();
  }

private:
  Profiler() = default;
  ~Profiler() { dump(); }
  struct Entry { double ms = 0; long n = 0; };
  std::mutex mu_;
  std::unordered_map<std::string, Entry> acc_;
};

struct ProfScope {
  const char *name;
  std::chrono::steady_clock::time_point t0;
  bool on;
  explicit ProfScope(const char *n)
      : name(n), t0(std::chrono::steady_clock::now()), on(Profiler::enabled()) {}
  ~ProfScope() {
    if (!on) return;
    Profiler::get().add(
        name, std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count());
  }
};

#define TURBO_APPLE_PROF(name) ::turbo_ocr::apple::ProfScope _prof_##__LINE__(name)

} // namespace turbo_ocr::apple
