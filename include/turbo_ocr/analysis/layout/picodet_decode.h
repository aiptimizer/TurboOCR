#pragma once

// THE shared PP-DocLayoutV3 (PicoDet + multiclass_nms3) detection-row decoder.
//
// WHY THIS FILE EXISTS: three backends decoded these rows independently and two
// of the three were wrong. Intel's (intel_stages.cpp) was correct; AMD's
// (rocm_stages.cpp) decoded only 6 columns, refused any tensor with `cols < 6`
// (so it would drop the real 7-column output's read_order and, on a 6-column
// export, silently return nothing on a 7-column one), never read the
// authoritative `count` tensor — using the rows tensor's first dim, which is
// DATA-DEPENDENT and documented to go stale across repeated requests, silently
// dropping layout from every consecutive response — and did no class-id range
// check, so a garbage id indexes kLayoutLabels out of range downstream.
//
// This is Intel's decoder, lifted verbatim. Every backend calls it.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "turbo_ocr/base/log/logger.h"   // TOCR_LOG_ERROR_RL
#include "turbo_ocr/core/layout_types.h" // LayoutBox, kLayoutLabels

namespace turbo_ocr::layout {

// NMS output budget of the exported graph. The rows tensor is allocated at this
// size regardless of how many detections the page actually produced, so an
// unclamped row count reads uninitialized memory.
inline constexpr int kPicodetMaxDet = 300;

// Decode `rows` [N, stride] into LayoutBoxes.
//   rows        — float rows {class_id, score, x0, y0, x1, y1[, read_order]} in
//                 ORIGINAL image coordinates (the PicoDet head applies im_shape
//                 + scale_factor internally).
//   rows_dim0   — the rows tensor's first dimension (the FALLBACK count).
//   stride      — the rows tensor's second dimension (6 or 7).
//   count       — the model's own count tensor, or nullptr when the graph has
//                 none. AUTHORITATIVE when present: rows_dim0 is data-dependent
//                 and goes stale across repeated requests on at least the TRT
//                 path, silently zeroing layout on every response after the
//                 first.
//
// Coordinate handling is a TRUNCATING cast (not lround) to match the reference
// decoder — lround would shift boxes by a pixel in a cross-backend golden diff.
[[nodiscard]] inline std::vector<LayoutBox>
decode_picodet_rows(const float *rows, int rows_dim0, int stride,
                    const std::int32_t *count, float score_threshold,
                    int orig_h, int orig_w) {
  std::vector<LayoutBox> out;
  if (!rows || stride < 6 || orig_h <= 0 || orig_w <= 0) return out;

  int n = rows_dim0;
  if (count) n = static_cast<int>(*count);
  // Clamp against the rows tensor's REAL first dimension too, not only the
  // assumed graph budget: `count` is authoritative for how many rows carry
  // data, but a graph whose count exceeds the rows it actually emitted would
  // otherwise read past the buffer (latent — every current caller passes
  // count=nullptr or allocates kPicodetMaxDet rows; this is the guard the
  // header comment already claimed).
  if (rows_dim0 > 0) n = std::min(n, rows_dim0);
  n = std::clamp(n, 0, kPicodetMaxDet);
  out.reserve(static_cast<std::size_t>(n));

  // FAIL LOUD ON NON-FINITE OUTPUT. A NaN score compares false against the
  // threshold, so a numerically broken graph or execution provider drops every
  // row and produces an empty layout indistinguishable from a clean page —
  // fast, silent, and wrong. Not hypothetical: the CoreML EP on ORT 1.24.4
  // returns NaN for every score/box on this model. Hoisted HERE so every
  // backend gets the guard — it briefly lived only in the CPU copy, leaving
  // Intel/AMD (which call this function) unguarded against the same class of
  // EP failure.
  //
  // Checked PER ROW, not just on row 0: an EP that NaNs only part of the output
  // passed a first-row-only test and then had its bad rows dropped silently by
  // `score < score_threshold`. And the guard covers EVERY field the loop below
  // consumes, not just the score — the class id r[0], the score r[1], the box
  // r[2]..r[5], and the read order r[6] on a 7-column export. All of those but
  // the score go through `static_cast<int>`, which is undefined behaviour for a
  // NaN/inf source, so a row with finite score and non-finite anything else is
  // worse than the case the guard was originally written for.
  //
  // Logged through TOCR_LOG (rate-limited: this is a per-request path), not
  // std::cerr — unstructured stderr carries no level and no request id and is
  // invisible to the log pipeline that consumes the server's output.
  int non_finite = 0;

  for (int i = 0; i < n; ++i) {
    const float *r = rows + static_cast<std::size_t>(i) * stride;
    bool finite = std::isfinite(r[0]) && std::isfinite(r[1]) &&
                  std::isfinite(r[2]) && std::isfinite(r[3]) &&
                  std::isfinite(r[4]) && std::isfinite(r[5]);
    // r[6] only exists on a 7-column export, and only then is it read.
    if (finite && stride >= 7) finite = std::isfinite(r[6]);
    if (!finite) {
      ++non_finite;
      continue;
    }
    const float score = r[1];
    if (score < score_threshold) continue;
    const int cls = static_cast<int>(r[0]);
    if (cls < 0 || cls >= static_cast<int>(kLayoutLabels.size())) continue;
    const int x0 = std::clamp(static_cast<int>(r[2]), 0, orig_w - 1);
    const int y0 = std::clamp(static_cast<int>(r[3]), 0, orig_h - 1);
    const int x1 = std::clamp(static_cast<int>(r[4]), 0, orig_w - 1);
    const int y1 = std::clamp(static_cast<int>(r[5]), 0, orig_h - 1);
    if (x1 <= x0 || y1 <= y0) continue;
    LayoutBox lb{};
    lb.class_id = cls;
    lb.score = score;
    if (stride >= 7) lb.read_order = static_cast<int>(r[6]);
    lb.box[0] = {x0, y0};
    lb.box[1] = {x1, y0};
    lb.box[2] = {x1, y1};
    lb.box[3] = {x0, y1};
    out.push_back(lb);
  }

  if (non_finite == n) {
    // EVERY row bad — the failure this guard was written for (CoreML EP on
    // ORT 1.24.4 NaN'd every score/box, so an empty layout was indistinguish-
    // able from a blank page). Still an error, still returns empty.
    TOCR_LOG_ERROR_RL(
        "layout model returned non-finite rows — the execution provider is "
        "producing garbage; do NOT treat this as a blank page",
        "non_finite_rows", non_finite, "rows", n);
    return {};
  }
  if (non_finite > 0) {
    // SOME rows bad is a different, benign thing, and reporting it at error
    // level cried wolf on every page. Root cause, traced through the graph on
    // the shipped layout export: the mask->box subgraph marks an EMPTY mask
    // with a literal 1e+08 sentinel — `where(mask, xs, 1e8)` — then zeroes the
    // box with `box * has_any`. In fp32 that is 0 * 1e8 == 0. The CoreML EP
    // computes this branch in FLOAT16 when it runs on the Metal GPU, where
    // 1e8 overflows to +inf (fp16 max is 65504), so it becomes 0 * inf == NaN
    // and the NaN then propagates through the six sigmoid/log refinement
    // layers into the final box.
    //
    // So these rows are NOT uninitialized memory and NOT an unwritten tail:
    // they are real, deterministic, bit-identical across runs, sitting
    // mid-buffer (indices 156-241 of 300 on one measured page), and their
    // class id / score / read order are all finite — only the four box
    // columns are NaN. They are exactly the EMPTY-MASK queries, so they carry
    // ~0.003 scores against a 0.3 threshold, and the CPU provider's rows at
    // the same indices are degenerate boxes (x1 <= x0) that this loop drops
    // anyway. No detection is lost: across an 83-page document CoreML and the
    // CPU provider produced 601 regions each.
    //
    // Triggered only by MLComputeUnits=CPUAndGPU, which is what this repo's
    // default COREML_FLAGS=0x020 selects; ORT's own default (ALL), CPUOnly,
    // ANE, and ONLY_ALLOW_STATIC_INPUT_SHAPES all compute it cleanly. Dropping
    // the rows (above) is a complete fix, so this is a debug note.
    TOCR_LOG_DEBUG_RL(
        "layout: dropped rows whose box is non-finite (fp16 overflow of the "
        "empty-mask sentinel on the CoreML GPU path; sub-threshold queries)",
        "non_finite_rows", non_finite, "rows", n);
  }
  return out;
}

} // namespace turbo_ocr::layout
