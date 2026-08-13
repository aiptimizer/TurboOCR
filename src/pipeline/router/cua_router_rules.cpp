#include "turbo_ocr/pipeline/router/cua_router.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace turbo_ocr::router {

namespace {

[[nodiscard]] inline int64_t aabb_area(const std::array<int, 4> &r) noexcept {
  return static_cast<int64_t>(std::max(0, r[2] - r[0])) *
         static_cast<int64_t>(std::max(0, r[3] - r[1]));
}

[[nodiscard]] inline int64_t aabb_intersect_area(
    const std::array<int, 4> &a,
    const std::array<int, 4> &b) noexcept {
  int x0 = std::max(a[0], b[0]);
  int y0 = std::max(a[1], b[1]);
  int x1 = std::min(a[2], b[2]);
  int y1 = std::min(a[3], b[3]);
  if (x1 <= x0 || y1 <= y0) return 0;
  return static_cast<int64_t>(x1 - x0) * static_cast<int64_t>(y1 - y0);
}


// Class priority for the IoU tie-breaker (higher wins). Mirrors plan
// 05 §9: Table > display_formula > inline_formula > Text-classes >
// Skip-classes.
[[nodiscard]] inline int class_priority(int class_id) noexcept {
  switch (class_id) {
    case 21: return 5; // table
    case 5:  return 4; // display_formula
    case 15: return 3; // inline_formula
    case 9:  case 13: case 20:
      // explicit Skip classes
      return 1;
    case 3:  case 14:
      // image / chart — Skip-by-default but content-bearing
      return 1;
    default:
      return 2; // text-bound
  }
}

} // namespace

// -- seam 1 ------------------------------------------------------------------
Destination default_destination(int class_id) noexcept {
  return destination_for(class_id, 0.0f);
}

// -- seam 2 ------------------------------------------------------------------
ConfidenceTier tier_for(int class_id, float score,
                        const RouterConfig &cfg) noexcept {
  float tau_trust  = 0.40f;
  float tau_verify = 0.20f;
  if (class_id >= 0 && class_id < kNumLayoutClasses) {
    tau_trust  = cfg.tau_trust[class_id];
    tau_verify = cfg.tau_verify[class_id];
  }
  if (score >= tau_trust)  return ConfidenceTier::Trust;
  if (score >= tau_verify) return ConfidenceTier::Verify;
  return ConfidenceTier::Fallback;
}

// -- seam 3 ------------------------------------------------------------------
FormulaWrap formula_wrap_for(int class_id) noexcept {
  if (class_id == 5)  return FormulaWrap::Display;
  if (class_id == 15) return FormulaWrap::Inline;
  return FormulaWrap::None;
}

// -- seam 5 ------------------------------------------------------------------
void build_overlap_stats(
    const std::vector<turbo_ocr::layout::LayoutBox> &layout,
    const std::vector<turbo_ocr::Box> &det_boxes,
    const std::vector<std::array<int, 4>> &layout_aabbs,
    OverlapStats &out) {
  const std::size_t n = layout.size();
  out.resize(n);
  if (n == 0) return;

  std::vector<float> aspect_sum(n, 0.0f);

  for (const auto &b : det_boxes) {
    auto db = turbo_ocr::aabb(b);
    const float bw = static_cast<float>(std::max(0, db[2] - db[0]));
    const float bh = static_cast<float>(std::max(0, db[3] - db[1]));
    const float det_area = bw * bh;
    const float ar = (bh > 0.0f && bw > 0.0f) ? (bw / bh) : 0.0f;

    for (std::size_t i = 0; i < n; ++i) {
      if (turbo_ocr::centroid_in_aabb(b, layout_aabbs[i])) {
        out.det_count[i] += 1;
        out.det_coverage[i] += det_area;
        if (ar > 0.0f) aspect_sum[i] += ar;
        break; // first match wins
      }
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto &r = layout_aabbs[i];
    const float la = static_cast<float>(aabb_area(r));
    if (la > 0.0f) {
      out.det_coverage[i] = std::min(1.0f, out.det_coverage[i] / la);
    } else {
      out.det_coverage[i] = 0.0f;
    }
    out.mean_aspect_ratio[i] = (out.det_count[i] > 0)
        ? aspect_sum[i] / static_cast<float>(out.det_count[i])
        : 0.0f;
    out.symbol_density_hint[i] =
        region_symbol_density_hint(r, out.det_count[i]);
  }
}

// -- seam 4 ------------------------------------------------------------------
bool is_formula_salvage_candidate(
    const turbo_ocr::layout::LayoutBox &lb,
    int layout_idx,
    const OverlapStats &ov,
    const PageStats &page,
    const RouterConfig &cfg) noexcept {
  if (!cfg.enable_formula_salvage) return false;
  if (!page.has_confident_formula) return false;
  if (lb.class_id != 22) return false; // only "text" is salvageable
  if (ov.empty()) return false;
  if (layout_idx < 0 ||
      static_cast<std::size_t>(layout_idx) >= ov.det_count.size())
    return false;

  const int n_det = ov.det_count[layout_idx];
  if (n_det > 3) return false;

  const float ar_mean = ov.mean_aspect_ratio[layout_idx];
  const float density = ov.symbol_density_hint[layout_idx];

  // §5 predicate 2: aspect ≥ 6.0 OR ≤ 1/6 (vertical formula). 1/6 ≈ 0.1667.
  const bool flat_or_thin = ar_mean >= 6.0f ||
                            (ar_mean > 0.0f && ar_mean <= (1.0f / 6.0f));

  // §5 predicate 2 alt: area < 1.5 × median text-line area.
  const auto ab = turbo_ocr::aabb(lb.box);
  const float lb_area = static_cast<float>(aabb_area(ab));
  const bool small_area = page.median_text_line_area > 0.0f &&
                          lb_area < 1.5f * page.median_text_line_area;

  return (flat_or_thin || small_area) && density > 0.4f;
}

// -- seam 7 ------------------------------------------------------------------
RouterDecision route(int layout_idx,
                     const turbo_ocr::layout::LayoutBox &lb,
                     const OverlapStats &ov,
                     const PageStats &page,
                     const RouterConfig &cfg) noexcept {
  RouterDecision d{};
  d.layout_idx = layout_idx;
  d.tier = tier_for(lb.class_id, lb.score, cfg);

  Destination base = default_destination(lb.class_id);
  d.dest = base;
  d.wrap = formula_wrap_for(lb.class_id);
  d.reason = (lb.class_id == turbo_ocr::layout::kSupplementaryRegionClassId)
                 ? RouterReason::SupplementaryRegion
                 : RouterReason::ClassDefault;

  // Image/chart Text-fallback gates (plan 05 §2). Only applies in
  // Trust/Verify tiers — Fallback already demotes everything.
  if (cfg.image_text_fallback && !ov.empty() && layout_idx >= 0 &&
      static_cast<std::size_t>(layout_idx) < ov.det_count.size()) {
    const int n_det = ov.det_count[layout_idx];
    const float cov = ov.det_coverage[layout_idx];
    if (lb.class_id == 3 /* chart */ && n_det >= 2) {
      d.dest = Destination::Text;
      d.reason = RouterReason::SkipWithDetPassthrough;
    } else if (lb.class_id == 14 /* image */ && n_det >= 3 && cov >= 0.15f) {
      d.dest = Destination::Text;
      d.reason = RouterReason::SkipWithDetPassthrough;
    }
  }

  // Confidence-tier policy (plan 05 §3).
  switch (d.tier) {
    case ConfidenceTier::Trust:
      // pass through the class-default decision above.
      break;
    case ConfidenceTier::Verify:
      // PASS-THROUGH TODAY. The comment here used to say table verification
      // "fires later via VerifyTablePath::should_invoke" — nothing in
      // production calls should_invoke or apply_verify_result; the gate is
      // built and unit-tested (tests/cpp/router) but was never wired into
      // UnifiedOcrPipeline::dispatch_router_, so a Verify-tier region takes its
      // class-default exactly like a Trust-tier one. Wiring it in is a routing
      // behaviour change and belongs in its own commit; until then this says
      // what the code does. The formula geometry spot-check IS live, through
      // the salvage path.
      break;
    case ConfidenceTier::Fallback:
      if (d.dest == Destination::Table) {
        d.dest = Destination::Text;
        d.reason = RouterReason::TableFallback;
      } else if (d.dest == Destination::Formula) {
        d.dest = Destination::Text;
        d.wrap = FormulaWrap::None;
        d.reason = RouterReason::FormulaFallback;
      } else if (d.dest == Destination::Skip) {
        d.dest = Destination::Text;
        d.reason = RouterReason::SkipWithDetPassthrough;
      }
      break;
  }

  // Formula salvage (plan 05 §5) — only flips text-default cells.
  if (d.dest == Destination::Text &&
      is_formula_salvage_candidate(lb, layout_idx, ov, page, cfg)) {
    d.dest = Destination::Formula;
    d.wrap = FormulaWrap::Inline;
    d.reason = RouterReason::FormulaSalvage;
    d.also_text = true; // dual-routing — merge picks the winner
  }

  return d;
}

// -- seam 6 ------------------------------------------------------------------
void resolve_tie_breakers(
    std::vector<RouterDecision> &decisions,
    const std::vector<turbo_ocr::layout::LayoutBox> &layout,
    const std::vector<std::array<int, 4>> &layout_aabbs) noexcept {
  const std::size_t n = decisions.size();
  if (n < 2) return;

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const auto &ra = layout_aabbs[i];
      const auto &rb = layout_aabbs[j];
      const int64_t inter = aabb_intersect_area(ra, rb);
      if (inter == 0) continue; // early exit on no overlap

      const int64_t aa = aabb_area(ra);
      const int64_t ab = aabb_area(rb);
      if (aa == 0 || ab == 0) continue;
      const int64_t uni = aa + ab - inter;

      // 1) Containment ≥ 80% — the inner cell wins, outer goes Text.
      const float frac_b_in_a = static_cast<float>(inter) /
                                static_cast<float>(ab);
      const float frac_a_in_b = static_cast<float>(inter) /
                                static_cast<float>(aa);
      // A table is SUPPOSED to enclose cells/text/captions, so the containment
      // rule must never demote it — otherwise large tables that contain other
      // layout regions get dropped to Text and never reach the table
      // recognizer (class 21 == table).
      constexpr int kTableClassId = 21;
      if (frac_b_in_a >= 0.80f && aa > ab) {
        // B contained in A → demote A (unless A is a table).
        if (decisions[i].dest != Destination::Text &&
            layout[i].class_id != kTableClassId) {
          decisions[i].dest = Destination::Text;
          decisions[i].wrap = FormulaWrap::None;
          decisions[i].reason = RouterReason::ContainmentLoser;
        }
        continue;
      }
      if (frac_a_in_b >= 0.80f && ab > aa) {
        if (decisions[j].dest != Destination::Text &&
            layout[j].class_id != kTableClassId) {
          decisions[j].dest = Destination::Text;
          decisions[j].wrap = FormulaWrap::None;
          decisions[j].reason = RouterReason::ContainmentLoser;
        }
        continue;
      }

      // 2) IoU ∈ [0.30, 0.80] — class priority decides.
      const float iou = static_cast<float>(inter) /
                        static_cast<float>(uni);
      if (iou < 0.30f || iou > 0.80f) continue;

      const int pi = class_priority(layout[i].class_id);
      const int pj = class_priority(layout[j].class_id);

      // Among equal priority and same class, fall through to score
      // then id (§9 step 3-4).
      int loser = -1;
      if (pi > pj) {
        loser = static_cast<int>(j);
      } else if (pj > pi) {
        loser = static_cast<int>(i);
      } else {
        // Same priority. Same-class → higher score wins (0.01 tie → larger box).
        const float si = layout[i].score;
        const float sj = layout[j].score;
        if (std::fabs(si - sj) > 0.01f) {
          loser = si > sj ? static_cast<int>(j) : static_cast<int>(i);
        } else if (aa != ab) {
          loser = aa > ab ? static_cast<int>(j) : static_cast<int>(i);
        } else {
          // 4) smaller id wins.
          loser = layout[i].id < layout[j].id
                      ? static_cast<int>(j)
                      : static_cast<int>(i);
        }
      }
      if (loser >= 0 && decisions[loser].dest != Destination::Text) {
        decisions[loser].dest = Destination::Text;
        decisions[loser].wrap = FormulaWrap::None;
        decisions[loser].reason = RouterReason::IoUOverlapLoser;
      }
    }
  }
}

// -- seam 8 ------------------------------------------------------------------
bool VerifyTablePath::should_invoke(const RouterDecision &d) noexcept {
  return d.dest == Destination::Table && d.tier == ConfidenceTier::Verify;
}

// -- seam 9 ------------------------------------------------------------------
RouterDecision apply_verify_result(RouterDecision d,
                                   TableClsVerdict v) noexcept {
  if (!v.is_table) {
    d.dest = Destination::Text;
    d.wrap = FormulaWrap::None;
    d.reason = RouterReason::TableVerifyDemoted;
    d.wired_hint.reset();
  } else {
    d.wired_hint = v.cls;
  }
  return d;
}

} // namespace turbo_ocr::router
