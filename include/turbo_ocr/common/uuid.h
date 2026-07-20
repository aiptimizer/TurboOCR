#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

namespace turbo_ocr {

// ── UUID v7 (timestamp-ordered, ~50ns) ──────────────────────────────────
//
// Request-id only — used for log correlation / X-Request-Id, never as a
// security token. mt19937_64 is fast but predictable; do NOT reuse these IDs
// to gate access or authorize anything.
[[nodiscard]] inline std::string generate_uuid_v7() {
  auto ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  thread_local std::mt19937_64 rng(std::random_device{}());
  uint64_t rand_a = rng();
  uint64_t rand_b = rng();

  uint8_t u[16];
  u[0]  = (ms >> 40) & 0xFF;
  u[1]  = (ms >> 32) & 0xFF;
  u[2]  = (ms >> 24) & 0xFF;
  u[3]  = (ms >> 16) & 0xFF;
  u[4]  = (ms >> 8)  & 0xFF;
  u[5]  = ms & 0xFF;
  std::memcpy(u + 6, &rand_a, 2);
  std::memcpy(u + 8, &rand_b, 8);
  u[6] = (u[6] & 0x0F) | 0x70;   // version 7
  u[8] = (u[8] & 0x3F) | 0x80;   // variant 10

  char buf[37];
  std::snprintf(buf, sizeof(buf),
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],
      u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  return std::string(buf, 36);
}

} // namespace turbo_ocr
