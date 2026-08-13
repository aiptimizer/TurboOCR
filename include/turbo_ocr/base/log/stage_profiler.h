#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

#include "turbo_ocr/base/env_utils.h"

// Per-stage wall-clock accumulators for the CPU OCR path. Zero cost unless
// PROFILE_STAGES=1 (checked once). Read+reset the JSON snapshot via the
// /profile HTTP route so a benchmark can attribute time across det/cls/rec
// sub-stages without per-request log scraping.
namespace turbo_ocr::prof {

enum Stage {
  DET,
  DET_PRE,
  DET_INFER,
  DET_POST,
  CLS,
  REC_PRE,
  REC_INFER,
  REC_DECODE,
  REC,
  SORT,
  COMBINE,
  LAYOUT,
  FORMULA,
  TABLE,
  N_STAGES
};

inline const char *name(Stage s) {
  switch (s) {
  case DET: return "det";
  case DET_PRE: return "det_pre";
  case DET_INFER: return "det_infer";
  case DET_POST: return "det_post";
  case CLS: return "cls";
  case REC_PRE: return "rec_pre";
  case REC_INFER: return "rec_infer";
  case REC_DECODE: return "rec_decode";
  case REC: return "rec";
  case SORT: return "sort";
  case COMBINE: return "combine";
  case LAYOUT: return "layout";
  case FORMULA: return "formula";
  case TABLE: return "table";
  default: return "?";
  }
}

struct Acc {
  std::atomic<uint64_t> ns{0};
  std::atomic<uint64_t> count{0};
};

inline std::array<Acc, N_STAGES> &accs() {
  static std::array<Acc, N_STAGES> a;
  return a;
}

inline bool enabled() {
  static const bool e = env::env_enabled("PROFILE_STAGES");
  return e;
}

inline void add(Stage s, uint64_t ns) {
  accs()[s].ns.fetch_add(ns, std::memory_order_relaxed);
  accs()[s].count.fetch_add(1, std::memory_order_relaxed);
}

// RAII scope timer. Skips clock reads entirely when profiling is off.
struct Scope {
  Stage s;
  bool on;
  std::chrono::steady_clock::time_point t0;
  explicit Scope(Stage st) : s(st), on(enabled()) {
    if (on) t0 = std::chrono::steady_clock::now();
  }
  ~Scope() {
    if (on)
      add(s, static_cast<uint64_t>(
                 std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t0)
                     .count()));
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
};

inline std::string dump_json_and_reset() {
  std::string out = "{";
  bool first = true;
  for (int i = 0; i < N_STAGES; ++i) {
    uint64_t ns = accs()[i].ns.exchange(0, std::memory_order_relaxed);
    uint64_t cnt = accs()[i].count.exchange(0, std::memory_order_relaxed);
    if (!first) out += ",";
    first = false;
    out += "\"";
    out += name(static_cast<Stage>(i));
    out += "\":{\"ms\":";
    out += std::to_string(static_cast<double>(ns) / 1e6);
    out += ",\"count\":";
    out += std::to_string(cnt);
    out += "}";
  }
  out += "}";
  return out;
}

} // namespace turbo_ocr::prof
