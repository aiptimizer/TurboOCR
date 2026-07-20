#include <catch_amalgamated.hpp>

#include "turbo_ocr/common/log/logger.h"

#include <chrono>
#include <thread>

using turbo_ocr::log::RateLimitSlot;
using turbo_ocr::log::ratelimit_check;
using turbo_ocr::log::ratelimit_config;

TEST_CASE("ratelimit allows up to N then suppresses", "[logger][ratelimit]") {
  // Default config: 10 logs / 1000 ms.
  RateLimitSlot slot;
  int passed = 0, suppressed_seen = 0, drained = 0;
  for (int i = 0; i < 100; ++i) {
    int d = 0;
    if (ratelimit_check(slot, d)) ++passed;
    else                          ++suppressed_seen;
    drained += d;
  }
  CHECK(passed == ratelimit_config().max_logs_per_window);
  CHECK(suppressed_seen == 100 - ratelimit_config().max_logs_per_window);
  // No window roll inside this tight loop, so no rollup is reported here.
  CHECK(drained == 0);
}

TEST_CASE("ratelimit emits suppressed count on window roll", "[logger][ratelimit]") {
  RateLimitSlot slot;
  // Burst past the cap.
  for (int i = 0; i < 50; ++i) {
    int d = 0;
    ratelimit_check(slot, d);
  }
  // Wait out the window.
  std::this_thread::sleep_for(
      std::chrono::milliseconds(ratelimit_config().window_ms + 50));
  int drained = 0;
  bool pass = ratelimit_check(slot, drained);
  CHECK(pass);
  CHECK(drained == 50 - ratelimit_config().max_logs_per_window);
}

TEST_CASE("ratelimit independent slots track separately", "[logger][ratelimit]") {
  RateLimitSlot a, b;
  for (int i = 0; i < 20; ++i) {
    int d = 0;
    ratelimit_check(a, d);
  }
  // Slot b should still admit a fresh batch.
  int passed_b = 0;
  for (int i = 0; i < 5; ++i) {
    int d = 0;
    if (ratelimit_check(b, d)) ++passed_b;
  }
  CHECK(passed_b == 5);
}
