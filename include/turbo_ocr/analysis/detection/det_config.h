#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <utility>

#include "turbo_ocr/base/env_utils.h"

namespace turbo_ocr::detection {

// Bounds for the engine optimization-profile MAX side. Kept symmetric with the
// integer-overflow safeguard in paddle_det.cpp (`max_pixels = kMaxSideLen_²`).
inline constexpr int kDetMaxSideMin = 32;
inline constexpr int kDetMaxSideMax = 4096;

// Per-model detection resize policy (PaddleOCR DetResizeForTest, resize_image_type0).
// PaddleX OCR pipeline default for every PP-OCRv6 det tier (inference.yml has
// "DetResizeForTest: null"): limit_type="min", limit_side_len=64, max_side_limit=4000.
//   - "min": grow the SHORTER side up to limit_side_len.
//   - "max": shrink the LONGER side down to limit_side_len.
// Both clamp the final output so max(resize_h,resize_w) <= max_side_limit.
struct DetResizeParams {
  const char* limit_type;  // "min" or "max"
  int limit_side_len;
  int max_side_limit;
};

// PaddleOCR's official OCR pipeline default is {"min", 64, 4000}. The 4000 cap
// is sized for single-image CLI use; this server pre-allocates det buffers at
// max_side_limit² per pooled pipeline, so 4000²×pool OOMs a 32 GB card. 1280
// runs the vast majority of documents at native resolution (the part that
// matters) while fitting the pool. Raise via DET_MAX_SIDE_LIMIT/DET_MAX_SIDE.
inline constexpr DetResizeParams kDetResizeDefault{"min", 64, 1280};

// DB post-processing parameters (PP-OCRv6 detection defaults). Per-model: only
// box_thresh differs across tiers (0.45 medium/small, 0.40 tiny). Shared by both
// detector paths so the GPU and CPU detectors can never disagree.
struct DbParams {
  float thresh;        // pixel-map binarization threshold
  float box_thresh;    // per-box mean-score cutoff
  float unclip_ratio;  // polygon expansion ratio
};

inline constexpr DbParams kDbDefaults{0.2f, 0.45f, 1.4f};

// ---------------------------------------------------------------------------
// PER-MODEL BASE, installed once at bootstrap. The model registry pairs each
// detector with its official config (tiny's box_thresh is 0.40, not the 0.45
// default — model_catalog.h), and that pairing must reach every backend's
// detector. The unified merge dropped the plumbing: det_cfg reached
// ServerConfig and then NOTHING read it, so every tier ran kDbDefaults unless
// DET_* env was set — and the Apple arm grew a private tier-from-model-PATH
// sniff to compensate. Installing the base here lets the no-arg
// read_det_resize()/read_db_params() calls every stage already makes inherit
// the tier config with zero per-backend threading. Env overrides still win
// (read_* applies them on top of whatever base).
//
// NOT thread-safe by design: call set_det_config_base() at bootstrap, before
// any stage loads (ServerConfig::load_or_die / the Python binding do).
namespace detail {
inline DetResizeParams &det_resize_base_slot() {
  static DetResizeParams v = kDetResizeDefault;
  return v;
}
inline DbParams &det_db_base_slot() {
  static DbParams v = kDbDefaults;
  return v;
}
} // namespace detail

[[nodiscard]] inline DetResizeParams det_resize_base() {
  return detail::det_resize_base_slot();
}
[[nodiscard]] inline DbParams det_db_base() { return detail::det_db_base_slot(); }
inline void set_det_config_base(const DetResizeParams &resize,
                                const DbParams &db) {
  detail::det_resize_base_slot() = resize;
  detail::det_db_base_slot() = db;
}

// Round to the nearest multiple of 32 (det engine requires /32 input dims),
// with a floor of 32. Matches PaddleOCR resize_image_type0's rounding.
[[nodiscard]] inline int round32_floor(double v) {
  return std::max(static_cast<int>(std::round(v / 32.0) * 32), 32);
}

// Shared resize computation for all three call sites (CPU + GPU single + GPU
// batch). Implements PaddleOCR resize_image_type0 for both limit policies plus
// the max_side_limit cap, returning the /32-rounded output dims. Returns
// (resize_h, resize_w).
[[nodiscard]] inline std::pair<int, int> compute_det_resize(int h, int w,
                                                            const DetResizeParams& p) {
  const int L = p.limit_side_len;
  float ratio = 1.0f;
  if (p.limit_type[0] == 'm' && p.limit_type[1] == 'a') {  // "max"
    const int longest = std::max(h, w);
    if (longest > L) ratio = static_cast<float>(L) / longest;
  } else {  // "min"
    const int shortest = std::min(h, w);
    if (shortest < L) ratio = static_cast<float>(L) / shortest;
  }

  int resize_h = round32_floor(h * ratio);
  int resize_w = round32_floor(w * ratio);

  // Cap the longer output side at max_side_limit, then re-round to /32.
  const int longest_out = std::max(resize_h, resize_w);
  if (longest_out > p.max_side_limit) {
    const double rescale = static_cast<double>(p.max_side_limit) / longest_out;
    resize_h = round32_floor(resize_h * rescale);
    resize_w = round32_floor(resize_w * rescale);
  }
  return {resize_h, resize_w};
}

// Apply env overrides to a per-model resize base. Each field is overridden ONLY
// when its env var is present, so a per-model value is never clobbered by a
// default. DET_MAX_SIDE (the single-knob effective-engine-max override consumed
// by effective_det_max_side()) is also folded in here so the resize cap can
// never exceed the engine profile / pinned buffers.
//   DET_LIMIT_TYPE / DET_LIMIT_SIDE_LEN / DET_MAX_SIDE_LIMIT / DET_MAX_SIDE
[[nodiscard]] inline DetResizeParams read_det_resize(DetResizeParams base = det_resize_base()) {
  const bool base_was_min = (base.limit_type[1] == 'i');
  bool side_len_from_env = false;
  const std::string limit_type = env::env_or("DET_LIMIT_TYPE", "");
  if (limit_type.size() >= 2)
    base.limit_type =
        (std::tolower(static_cast<unsigned char>(limit_type[0])) == 'm' &&
         std::tolower(static_cast<unsigned char>(limit_type[1])) == 'a') ? "max" : "min";
  // Numeric envs are clamped to [kDetMaxSideMin, kDetMaxSideMax]: an unclamped
  // parse turns garbage/negative input into values that would otherwise
  // thumbnail every image (a 0 cap resizes everything to 32px) — same
  // silent-empty-results failure class as GitHub #23.
  if (env::env_present("DET_LIMIT_SIDE_LEN")) {
    base.limit_side_len =
        env::env_int("DET_LIMIT_SIDE_LEN", base.limit_side_len, kDetMaxSideMin, kDetMaxSideMax);
    side_len_from_env = true;
  }
  base.max_side_limit =
      env::env_int("DET_MAX_SIDE_LIMIT", base.max_side_limit, kDetMaxSideMin, kDetMaxSideMax);
  // DET_MAX_SIDE sizes the TRT profile MAX + pinned buffers via
  // effective_det_max_side(); it must ALSO cap the resize output, or a
  // DET_MAX_SIDE below max_side_limit (e.g. 640 < 1280) lets compute_det_resize
  // emit up to max_side_limit px into a buffer/profile sized for the smaller
  // value — overrun. Fold it in as a shrink-only min (enlarging is safe: the
  // buffer is sized to the larger effective max).
  if (env::env_present("DET_MAX_SIDE"))
    base.max_side_limit = std::min(
        base.max_side_limit,
        env::env_int("DET_MAX_SIDE", base.max_side_limit, kDetMaxSideMin, kDetMaxSideMax));
  // DET_LIMIT_TYPE=max without DET_LIMIT_SIDE_LEN: the min-policy base default
  // (64 = "grow the SHORTER side to at least 64") would flip meaning to
  // "shrink the LONGER side down to 64px", thumbnailing every image into zero
  // detections (GitHub #23). The only sane implicit max-policy target is the
  // resize cap itself: native resolution up to max_side_limit.
  if (base_was_min && !side_len_from_env && base.limit_type[1] == 'a') {
    base.limit_side_len = base.max_side_limit;
    std::cerr << std::format(
        "[DetConfig] DET_LIMIT_TYPE=max set without DET_LIMIT_SIDE_LEN; "
        "defaulting limit_side_len to the max-side cap ({})\n", base.max_side_limit);
  }
  return base;
}

// Apply env overrides to a per-model DB base. Each field is overridden ONLY when
// its env var is present, so a per-model value is never clobbered by a default.
//   DET_DB_THRESH / DET_BOX_THRESH / DET_UNCLIP
//
// Clamped to their physically meaningful domains, for the same reason the
// resize knobs above are: both thresholds cut a probability map, so a value
// outside [0,1] means "keep everything" or "keep nothing" — two ways to get
// empty or unusable output from a typo — and an unclip ratio at or below zero
// collapses every polygon.
[[nodiscard]] inline DbParams read_db_params(DbParams base = det_db_base()) {
  base.thresh = env::env_float("DET_DB_THRESH", base.thresh, 0.0f, 1.0f);
  base.box_thresh = env::env_float("DET_BOX_THRESH", base.box_thresh, 0.0f, 1.0f);
  base.unclip_ratio = env::env_float("DET_UNCLIP", base.unclip_ratio, 0.1f, 10.0f);
  return base;
}

// Effective engine optimization-profile MAX side. The TRT profile MAX (and the
// buffers sized against it) must be >= the largest possible resize output, i.e.
// >= the model's max_side_limit. DET_MAX_SIDE, when set, overrides it (the
// single-knob override). Clamped to [kDetMaxSideMin, kDetMaxSideMax].
//
// Rounded UP to the next multiple of 32: compute_det_resize caps the longer
// side at max_side_limit then re-rounds to /32 (round-to-nearest), which can
// nudge the output above a non-/32 max_side_limit by up to 16px. Sizing the
// profile/buffers at the /32-ceil guarantees they always cover that output.
//
// Read by:
//   - paddle_det.cpp (GPU detector — sizes pinned input buffers)
//   - ort_paddle_det.cpp (CPU detector — same role)
//   - onnx_to_trt.cpp (TRT engine builder — sizes the optimization profile
//     MAX, and is included in the engine cache key)
// All three MUST agree or the engine and runtime silently disagree.
[[nodiscard]] inline int effective_det_max_side(const DetResizeParams& p = det_resize_base()) {
  const int v = env::env_int("DET_MAX_SIDE", std::clamp(p.max_side_limit, kDetMaxSideMin,
                                                        kDetMaxSideMax),
                             kDetMaxSideMin, kDetMaxSideMax);
  return ((v + 31) / 32) * 32;  // /32-ceil stays within [kDetMaxSideMin, Max]
}

// Close the detection canvas set for shape-specialized runtimes.
//
// compute_det_resize emits a /32-rounded canvas per image, so the (h,w) set is
// OPEN. That is fine for runtimes where dynamic shapes are free (ORT/MLAS,
// OpenVINO CPU) and hostile to per-shape-JIT runtimes: measured on a UHD 770
// (OpenVINO GPU plugin), det_tiny ran at 121 ms/img in-pipeline against
// 15.4 ms for the same fixed shape in benchmark_app, because nearly every
// FUNSD page brought a NEW canvas and each one paid a kernel JIT. Snapping
// both sides UP to a 128 grid (4 /32-steps) collapses the open set to <=10
// values per side — a handful in practice, each compiled once (and persisted
// across runs with OV_CACHE_DIR / the TRT engine cache).
//
// The caller letterboxes the resized content into the snapped canvas
// (top-left, zero-fill in NORMALIZED space = per-channel mean pixel, which DB
// scores as background) and keeps rescaling boxes by the CONTENT dims, so
// detection geometry is unchanged. Gate use on EngineCaps::per_shape_jit —
// padding costs compute, which is pure waste where dynamic shapes are cheap.
// NAMING: takes ALREADY-RESIZED dims and rounds up onto a grid. The aspect
// picker that takes ORIGINAL dims and a finite export set is
// detection::pick_det_canvas (db_post_config.h) — they used to share one name
// with opposite coordinate-space contracts.
inline constexpr int kDetCanvasSnap = 128;
[[nodiscard]] inline std::pair<int, int> snap_det_canvas_grid(
    int resize_h, int resize_w, const DetResizeParams& p = det_resize_base()) {
  const int cap = effective_det_max_side(p);
  const auto snap = [cap](int v) {
    const int s = ((v + kDetCanvasSnap - 1) / kDetCanvasSnap) * kDetCanvasSnap;
    // v <= cap always (compute_det_resize caps at max_side_limit and
    // effective_det_max_side /32-ceils it), so the clamp keeps snap(v) >= v.
    return std::clamp(s, kDetMaxSideMin, cap);
  };
  return {snap(resize_h), snap(resize_w)};
}

} // namespace turbo_ocr::detection
