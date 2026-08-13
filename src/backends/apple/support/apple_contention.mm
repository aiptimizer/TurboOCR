// Contention counter registry + exit dump (see apple_contention.h).

#include "apple/support/apple_contention.h"

#include "turbo_ocr/base/env_utils.h"

#include <algorithm>
#include <string>

namespace turbo_ocr::apple {

namespace {
std::mutex &reg_mu() {
  static std::mutex m;
  return m;
}
struct Dumper {
  ~Dumper() { contention_dump(); }
};
Dumper &dumper() {
  static Dumper d;
  return d;
}
} // namespace

std::vector<Stat *> &contention_registry() {
  static std::vector<Stat *> v;
  return v;
}

bool contention_enabled() {
  static const bool on = [] {
    const std::string e = env::env_or("TURBO_APPLE_CONTENTION", "");
    return !e.empty() && e != "0";
  }();
  return on;
}

Stat::Stat(const char *name) : name_(name) {
  std::lock_guard<std::mutex> lk(reg_mu());
  contention_registry().push_back(this);
  (void)dumper(); // make sure the exit dump is registered after us
}

void contention_dump() {
  if (!contention_enabled()) return;
  std::lock_guard<std::mutex> lk(reg_mu());
  auto &v = contention_registry();
  if (v.empty()) return;
  std::vector<Stat *> s(v.begin(), v.end());
  std::sort(s.begin(), s.end(), [](Stat *a, Stat *b) {
    return (a->ns_ + a->blocked_ns_) > (b->ns_ + b->blocked_ns_);
  });
  std::fprintf(stderr,
               "\n=== apple contention (ms total / calls / us-per-call / "
               "blocked-ms / blocked-calls / units) ===\n");
  for (Stat *st : s) {
    const long long n = st->n_.load();
    const long long ns = st->ns_.load();
    if (n == 0 && ns == 0 && st->blocked_ns_.load() == 0) continue;
    std::fprintf(stderr, "  %-30s %10.1f %9lld %9.2f %10.1f %9lld %10lld\n",
                 st->name_, ns / 1e6, n, n ? (ns / 1e3) / (double)n : 0.0,
                 st->blocked_ns_.load() / 1e6, st->blocked_n_.load(),
                 st->units_.load());
    st->ns_ = 0; st->n_ = 0; st->units_ = 0;
    st->blocked_ns_ = 0; st->blocked_n_ = 0;
  }
  std::fprintf(stderr,
               "=========================================================="
               "==========================\n");
}

} // namespace turbo_ocr::apple
