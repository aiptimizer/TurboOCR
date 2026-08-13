#pragma once

// Shared DB post-process geometry limits + the fixed-canvas det decision.
//
// The main-tree turbo_ocr/analysis/detection/det_config.h owns the resize policy and the
// three DB thresholds (kDbDefaults = 0.2 bin / 0.45 box / 1.4 unclip) and is
// already honoured by every backend. What it did NOT own were the four geometry
// limits fed to extract_boxes_from_bitmap, so those were retyped per call site
// and drifted: an earlier Apple metal_kernels.mm passed min_unclipped_side = 2.0
// while host_kernels.cpp and the Apple detector itself both pass 5.0 — Apple was
// internally inconsistent with its own detector.
//
// This header is included BY the seam (backend/kernels.h) so DbPostParams
// can default from it. It therefore must NOT include kernels.h.

#include <algorithm>
#include <cmath>
#include <span>
#include <utility>

#include "turbo_ocr/analysis/detection/det_config.h" // kDbDefaults, compute_det_resize, DbParams

namespace turbo_ocr::detection {

// Geometry limits applied per connected component, in the DB map's coordinate
// space. Values are PaddleOCR DBPostProcess's, matching
// src/analysis/detection/det_postprocess.cpp's call sites.
inline constexpr float kMinBoxSide = 3.0f;       // drop pre-unclip slivers
inline constexpr float kMinUnclippedSide = 5.0f; // drop post-unclip slivers
inline constexpr float kMinExpand = 2.0f;        // floor on the unclip radius
inline constexpr float kMaxExpand = 24.0f;       // ceiling on the unclip radius

// Candidate component budget (PP-OCRv6 DB). Above this the map is pathological
// (usually a mis-thresholded photo) and the extra components are noise.
inline constexpr int kMaxDbComponents = 3000;

// ---------------------------------------------------------------------------
// Fixed-canvas detectors (Apple/MPSGraph exports a graph at ONE static input
// shape; the graph cannot be re-shaped per page). Such a backend must still make
// the aspect decision the SHARED way rather than stretching every page onto its
// canvas: run the normal policy, then map the result onto the nearest export the
// backend actually has. With a non-empty `available` the returned canvas is one
// of its entries; an EMPTY `available` returns the unsnapped policy result
// (the caller has no exports to choose from, so the policy answer stands).
//
// `available` is the list of (h, w) export canvases, in any order. The chosen
// canvas is the one whose aspect ratio is closest to the policy's, breaking ties
// toward the larger area (more pixels, never fewer, for the same aspect).
//
// Returning a canvas rather than silently stretching makes the constraint
// visible and keeps DET_LIMIT_TYPE / DET_LIMIT_SIDE_LEN / DET_MAX_SIDE
// meaningful: they still choose WHICH canvas a page lands on.
//
// NAMING: this takes ORIGINAL page dims and PICKS from a finite export set.
// Its former name, snap_det_canvas, collided with det_config.h's grid
// snapper — a different function taking ALREADY-RESIZED dims — and both
// overloads were visible in most TUs with each file's comment calling its own
// "the SHARED one". The contract now lives in the name; see
// detection::snap_det_canvas_grid for the other coordinate space.
[[nodiscard]] inline std::pair<int, int>
pick_det_canvas(int orig_h, int orig_w,
                std::span<const std::pair<int, int>> available,
                const DetResizeParams &policy = kDetResizeDefault) {
  const auto [want_h, want_w] = compute_det_resize(orig_h, orig_w, policy);
  if (available.empty()) return {want_h, want_w};
  const double want_ar =
      static_cast<double>(want_w) / std::max(1, want_h);
  std::pair<int, int> best = available.front();
  double best_err = 1e18;
  for (const auto &c : available) {
    if (c.first <= 0 || c.second <= 0) continue;
    const double ar = static_cast<double>(c.second) / c.first;
    const double err = std::abs(std::log(ar / want_ar));
    const double best_area = static_cast<double>(best.first) * best.second;
    const double area = static_cast<double>(c.first) * c.second;
    if (err < best_err - 1e-9 || (std::abs(err - best_err) <= 1e-9 && area > best_area)) {
      best_err = err;
      best = c;
    }
  }
  return best;
}

} // namespace turbo_ocr::detection
