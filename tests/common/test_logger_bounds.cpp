#include <catch_amalgamated.hpp>

#include <cstdio>
#include <string>

#include "turbo_ocr/common/log/logger.h"

using turbo_ocr::log::detail::advance;

TEST_CASE("advance clamps a truncating snprintf return", "[logger][bounds]") {
  char buf[8];
  char *p = buf;
  size_t rem = sizeof(buf) - 1;  // 7 usable

  // snprintf reports the WOULD-BE length (11) though only 7 chars + NUL fit.
  int n = std::snprintf(p, rem, "%s", "hello world");  // returns 11
  REQUIRE(n == 11);
  advance(p, rem, n);
  // Cursor must not pass the buffer: advanced by at most rem-1 (6), leaving
  // rem == 1 so the caller's reserved NUL still fits.
  CHECK(p <= buf + sizeof(buf));
  CHECK(rem >= 1);
  CHECK(static_cast<size_t>(p - buf) <= sizeof(buf) - 1);
}

TEST_CASE("advance is a no-op on encoding error", "[logger][bounds]") {
  char buf[8];
  char *p = buf;
  size_t rem = 7;
  advance(p, rem, -1);
  CHECK(p == buf);
  CHECK(rem == 7);
}

TEST_CASE("advance on an exactly-fitting write consumes it", "[logger][bounds]") {
  char buf[8];
  char *p = buf;
  size_t rem = 7;
  int n = std::snprintf(p, rem, "%s", "abc");  // 3
  advance(p, rem, n);
  CHECK(p == buf + 3);
  CHECK(rem == 4);
}

TEST_CASE("log_msg survives an oversized message and KV values", "[logger][bounds]") {
  // Drives the full formatter path with inputs far larger than the 4KB
  // thread-local buffer. Before the clamped-advance fix this wrote past the
  // buffer (ASan/stack-protector would trip); it must now simply clip.
  const std::string huge_msg(10000, 'M');
  const std::string huge_val(10000, 'V');
  // Any active LOG_FORMAT exercises the same advance() guard.
  turbo_ocr::log::log_msg(turbo_ocr::log::Level::Error, huge_msg,
                          std::string_view("key"), huge_val,
                          std::string_view("k2"), 123456789);
  SUCCEED("no out-of-bounds write / crash");
}
