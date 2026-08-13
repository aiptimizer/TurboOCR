#pragma once

// SHARED recognition batching policy — device-agnostic.
//
// Every recognizer backend that must pre-build per-SHAPE artefacts (Apple
// MPSGraph executables, TensorRT optimization profiles, an ORT IOBinding pool)
// faces the SAME two policy questions:
//
//   1. which WIDTH bucket does this line go to?              (rec_geometry.h)
//   2. which static BATCH size do I run that bucket at?      (here)
//
// Question 1 already has one shared answer in
// include/turbo_ocr/analysis/recognition/rec_geometry.h (rec_input_width /
// kRecWidthBuckets / snap_width_bucket). Question 2 used to be answered
// per-backend, which is exactly how the Apple ladder-clamping bug happened: a
// backend-private ladder drifts from the shared one, and a bug fixed on one path
// silently persists on the other.
//
// So both halves live here, once:
//   * kRecBatchLadder          — geometric static-batch rungs (<=2x pad waste).
//   * batch_ladder_for_width() — the rungs a given width may use under a fixed
//                                per-submission element budget (a wide bucket
//                                gets small rungs, a narrow one large rungs).
//   * snap_batch()             — tightest rung >= demand.
//   * group_by_width_bucket()  — the ONE routing rule (smallest fitting bucket,
//                                overflow clamped to the widest).
//   * plan_rec_batches()       — routing + chunking + rung choice in one call:
//                                a backend just iterates the returned plan.
//   * rec_shape_matrix()       — every (width,batch) shape a backend must
//                                pre-build at load() so nothing is compiled in
//                                the hot path.
//
// Nothing here names a device. A backend with genuinely dynamic shapes (the
// Host/ORT path) can use the routing half and ignore the batch half.

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "turbo_ocr/core/types.h"            // turbo_ocr::Box
#include "turbo_ocr/analysis/recognition/rec_geometry.h" // rec_input_width, kRecWidthBuckets

namespace turbo_ocr::recognition {

// Static batch rungs, geometric (x2) so a chunk never pads more than ~2x its
// real demand while keeping the number of pre-built artefacts small.
inline constexpr std::array kRecBatchLadder = {4, 8, 16, 32, 64, 128, 256};

// Per-submission input budget in ELEMENTS (float32 crop pixels). Bounds both the
// scratch buffer a backend must allocate per (width,batch) and the latency of a
// single submission. 8M elements = 32 MB of F32 crops — a batch of 128 at width
// 320, or 8 at width 4000. Chosen to match the largest batch the measured Apple
// MPSGraph ladder ran profitably; identical arithmetic serves a TRT profile set.
inline constexpr std::size_t kRecBatchElemBudget = 8u << 20;

// Rungs usable at `width`: those whose batch fits the element budget. Always at
// least one rung (the smallest) so a pathologically wide bucket still runs.
[[nodiscard]] inline std::vector<int>
batch_ladder_for_width(int width, int rec_image_h,
                       std::size_t budget = kRecBatchElemBudget,
                       std::span<const int> ladder = kRecBatchLadder) {
  const std::size_t per_row =
      static_cast<std::size_t>(3) * rec_image_h * std::max(width, 1);
  std::vector<int> out;
  for (int r : ladder)
    if (static_cast<std::size_t>(r) * per_row <= budget) out.push_back(r);
  if (out.empty() && !ladder.empty()) out.push_back(ladder.front());
  return out;
}

// Tightest ladder rung >= demand; demands above the top rung return the top rung
// (callers chunk — see plan_rec_batches).
[[nodiscard]] inline int snap_batch(int demand, std::span<const int> ladder) {
  if (ladder.empty()) return demand;
  for (int r : ladder)
    if (r >= demand) return r;
  return ladder.back();
}

// Route each box to the SMALLEST width bucket that fits its natural rec width,
// clamping overflow to the widest bucket. One index list per bucket, in the
// caller's bucket order (which MUST be ascending).
inline void group_by_width_bucket(const std::vector<turbo_ocr::Box> &boxes,
                                  int rec_image_h,
                                  std::span<const int> bucket_widths,
                                  std::vector<std::vector<int>> &out_lists) {
  const int nb = static_cast<int>(bucket_widths.size());
  out_lists.assign(nb, {});
  if (nb == 0) return;
  for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
    const int nat = rec_input_width(boxes[i], rec_image_h);
    int b = nb - 1;
    for (int k = 0; k < nb; ++k)
      if (nat <= bucket_widths[k]) { b = k; break; }
    out_lists[b].push_back(i);
  }
}

// One unit of recognizer work: `count` crops taken from bucket `bucket`'s index
// list starting at `offset`, executed at static batch size `batch` (>= count).
struct RecBatchChunk {
  int bucket = 0;
  int offset = 0;
  int count = 0;
  int batch = 0;
};

// Full per-page plan: route the boxes, split each bucket's demand into chunks no
// larger than that width's top rung, and snap each chunk to its tightest rung.
// `out_lists` receives the routing so the caller maps results back to original
// box indices.
// `bucket_rungs`, when non-empty, gives the batch sizes each bucket's artefact
// ACTUALLY supports — a hardware fact, not a policy: a CoreML mlprogram carries
// a fixed enumerated shape set, a TRT engine carries its profile min/opt/max.
// The policy (tightest supported batch >= demand, chunk above the top) is the
// same either way; empty falls back to batch_ladder_for_width.
[[nodiscard]] inline std::vector<RecBatchChunk>
plan_rec_batches(const std::vector<turbo_ocr::Box> &boxes, int rec_image_h,
                 std::span<const int> bucket_widths,
                 std::vector<std::vector<int>> &out_lists,
                 std::span<const std::vector<int>> bucket_rungs = {},
                 std::size_t budget = kRecBatchElemBudget) {
  group_by_width_bucket(boxes, rec_image_h, bucket_widths, out_lists);
  std::vector<RecBatchChunk> plan;
  for (int b = 0; b < static_cast<int>(out_lists.size()); ++b) {
    const int n = static_cast<int>(out_lists[b].size());
    if (n == 0) continue;
    // A per-bucket rung list can be EMPTY even when bucket_rungs is populated:
    // a backend may prebuild only some widths (Intel's
    // OV_REC_MAX_PREBUILD_WIDTH defaults to 1600 and skips every wider shape)
    // while still sizing bucket_rungs for the whole kRecWidthBuckets ladder. Any
    // page with a line of aspect ratio above ~33:1 routes into one of those
    // buckets, and `rungs.back()` on an empty vector is UB — and if it survives,
    // the caller then sizes its logits/index buffers from the (absent) prebuilt
    // shape and writes past them. Fall back to the shared ladder so the plan is
    // always well-formed; the backend still has to cope with a shape it did not
    // prebuild, but it does so with correct sizes instead of corrupting memory.
    const bool have_rungs =
        !bucket_rungs.empty() && !bucket_rungs[b].empty();
    const std::vector<int> fallback =
        have_rungs ? std::vector<int>{}
                   : batch_ladder_for_width(bucket_widths[b], rec_image_h, budget);
    const std::vector<int> &rungs = have_rungs ? bucket_rungs[b] : fallback;
    if (rungs.empty()) continue; // degenerate ladder: nothing schedulable
    const int cap = rungs.back();
    for (int off = 0; off < n;) {
      const int c = std::min(cap, n - off);
      plan.push_back({b, off, c, snap_batch(c, rungs)});
      off += c;
    }
  }
  return plan;
}

// Every (width, batch) shape a backend must have an artefact for. Feed this to
// eager pre-build at load() so the hot path never compiles.
[[nodiscard]] inline std::vector<std::pair<int, int>>
rec_shape_matrix(std::span<const int> bucket_widths, int rec_image_h,
                 std::size_t budget = kRecBatchElemBudget) {
  std::vector<std::pair<int, int>> v;
  for (int w : bucket_widths)
    for (int b : batch_ladder_for_width(w, rec_image_h, budget))
      v.emplace_back(w, b);
  return v;
}

} // namespace turbo_ocr::recognition
