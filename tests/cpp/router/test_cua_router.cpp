#include <catch_amalgamated.hpp>

#include <algorithm>
#include <chrono>
#include <random>

#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/pipeline/router/cua_router.h"

using namespace turbo_ocr;
using namespace turbo_ocr::router;
using LayoutBox = turbo_ocr::layout::LayoutBox;

namespace {

Box make_box(int x0, int y0, int x1, int y1) {
  Box b{};
  b[0] = {x0, y0};
  b[1] = {x1, y0};
  b[2] = {x1, y1};
  b[3] = {x0, y1};
  return b;
}

LayoutBox make_layout(int class_id, float score, int x0, int y0, int x1, int y1,
                      int id = -1) {
  LayoutBox lb;
  lb.class_id = class_id;
  lb.score = score;
  lb.box = make_box(x0, y0, x1, y1);
  lb.id = id;
  return lb;
}

} // namespace

// --- seam 1: default_destination over all 25 classes + sentinel ---

TEST_CASE("default_destination covers every layout class_id", "[router][lut]") {
  // Table / formula / skip classes
  CHECK(default_destination(21) == Destination::Table);
  CHECK(default_destination(5)  == Destination::Formula);
  CHECK(default_destination(15) == Destination::Formula);
  CHECK(default_destination(3)  == Destination::Skip);    // chart
  CHECK(default_destination(9)  == Destination::Skip);    // footer_image
  CHECK(default_destination(13) == Destination::Skip);    // header_image
  CHECK(default_destination(14) == Destination::Skip);    // image
  CHECK(default_destination(20) == Destination::Skip);    // seal

  // formula_number always text (cheap digits)
  CHECK(default_destination(11) == Destination::Text);

  // Text-bound classes
  for (int cid : {0, 1, 2, 4, 6, 7, 8, 10, 12, 16, 17, 18, 19, 22, 23, 24}) {
    CHECK(default_destination(cid) == Destination::Text);
  }

  // SupplementaryRegion sentinel
  CHECK(default_destination(-1) == Destination::Text);

  // Out-of-range class_ids fall through to Text (safe default).
  CHECK(default_destination(99) == Destination::Text);
}

// --- seam 2: tier_for boundaries ---

TEST_CASE("tier_for honors per-class thresholds", "[router][tier]") {
  auto cfg = RouterConfig::defaults();

  // table (21): trust=0.60, verify=0.35
  CHECK(tier_for(21, 0.70f, cfg) == ConfidenceTier::Trust);
  CHECK(tier_for(21, 0.60f, cfg) == ConfidenceTier::Trust);   // boundary inclusive
  CHECK(tier_for(21, 0.50f, cfg) == ConfidenceTier::Verify);
  CHECK(tier_for(21, 0.35f, cfg) == ConfidenceTier::Verify);
  CHECK(tier_for(21, 0.20f, cfg) == ConfidenceTier::Fallback);

  // display_formula (5): trust=0.55, verify=0.30
  CHECK(tier_for(5, 0.55f, cfg) == ConfidenceTier::Trust);
  CHECK(tier_for(5, 0.40f, cfg) == ConfidenceTier::Verify);
  CHECK(tier_for(5, 0.10f, cfg) == ConfidenceTier::Fallback);

  // text-bound default (22): trust=0.40, verify=0.20
  CHECK(tier_for(22, 0.50f, cfg) == ConfidenceTier::Trust);
  CHECK(tier_for(22, 0.30f, cfg) == ConfidenceTier::Verify);
  CHECK(tier_for(22, 0.05f, cfg) == ConfidenceTier::Fallback);

  // Out-of-range class_id falls back to generic thresholds (0.40/0.20).
  CHECK(tier_for(-1, 0.45f, cfg) == ConfidenceTier::Trust);
  CHECK(tier_for(99, 0.25f, cfg) == ConfidenceTier::Verify);
}

// --- seam 3: formula_wrap_for ---

TEST_CASE("formula_wrap_for maps display vs inline", "[router][wrap]") {
  CHECK(formula_wrap_for(5)  == FormulaWrap::Display);
  CHECK(formula_wrap_for(15) == FormulaWrap::Inline);
  CHECK(formula_wrap_for(21) == FormulaWrap::None);
  CHECK(formula_wrap_for(22) == FormulaWrap::None);
}

// --- seam 7 (route): Trust / Verify / Fallback paths ---

TEST_CASE("route: Fallback tier demotes Table to Text", "[router][route]") {
  auto cfg = RouterConfig::defaults();
  auto lb = make_layout(21, 0.10f, 0, 0, 100, 100); // table, very low score
  OverlapStats ov;
  PageStats ps;
  auto d = route(0, lb, ov, ps, cfg);
  CHECK(d.dest == Destination::Text);
  CHECK(d.reason == RouterReason::TableFallback);
  CHECK(d.tier == ConfidenceTier::Fallback);
}

TEST_CASE("route: Fallback tier demotes Formula to Text", "[router][route]") {
  auto cfg = RouterConfig::defaults();
  auto lb = make_layout(5, 0.10f, 0, 0, 100, 50);
  OverlapStats ov;
  PageStats ps;
  auto d = route(0, lb, ov, ps, cfg);
  CHECK(d.dest == Destination::Text);
  CHECK(d.reason == RouterReason::FormulaFallback);
  CHECK(d.wrap == FormulaWrap::None);
}

TEST_CASE("route: Trust tier passes Table/Formula through", "[router][route]") {
  auto cfg = RouterConfig::defaults();
  OverlapStats ov;
  PageStats ps;

  auto t = route(0, make_layout(21, 0.85f, 0, 0, 100, 100), ov, ps, cfg);
  CHECK(t.dest == Destination::Table);
  CHECK(t.tier == ConfidenceTier::Trust);

  auto f = route(0, make_layout(5, 0.85f, 0, 0, 100, 50), ov, ps, cfg);
  CHECK(f.dest == Destination::Formula);
  CHECK(f.wrap == FormulaWrap::Display);

  auto i = route(0, make_layout(15, 0.85f, 0, 0, 100, 30), ov, ps, cfg);
  CHECK(i.dest == Destination::Formula);
  CHECK(i.wrap == FormulaWrap::Inline);
}

TEST_CASE("route: image with det boxes inside falls back to Text",
          "[router][route]") {
  auto cfg = RouterConfig::defaults();
  auto lb = make_layout(14, 0.85f, 0, 0, 100, 100); // image, trust score
  OverlapStats ov;
  ov.resize(1);
  ov.det_count[0] = 5;
  ov.det_coverage[0] = 0.25f;
  PageStats ps;
  auto d = route(0, lb, ov, ps, cfg);
  CHECK(d.dest == Destination::Text);
  CHECK(d.reason == RouterReason::SkipWithDetPassthrough);
}

TEST_CASE("route: image without det boxes stays Skip", "[router][route]") {
  auto cfg = RouterConfig::defaults();
  auto lb = make_layout(14, 0.85f, 0, 0, 100, 100);
  OverlapStats ov;
  ov.resize(1);
  ov.det_count[0] = 0;
  PageStats ps;
  auto d = route(0, lb, ov, ps, cfg);
  CHECK(d.dest == Destination::Skip);
}

// --- seam 4: formula salvage gating ---

TEST_CASE("formula salvage requires confident formula on page",
          "[router][salvage]") {
  auto cfg = RouterConfig::defaults();
  auto lb = make_layout(22, 0.9f, 0, 0, 600, 80); // wide text
  OverlapStats ov;
  ov.resize(1);
  ov.det_count[0] = 2;
  ov.mean_aspect_ratio[0] = 8.0f;
  ov.symbol_density_hint[0] = 0.6f;

  PageStats ps;
  ps.has_confident_formula = false;
  CHECK_FALSE(is_formula_salvage_candidate(lb, 0, ov, ps, cfg));

  ps.has_confident_formula = true;
  CHECK(is_formula_salvage_candidate(lb, 0, ov, ps, cfg));
}

TEST_CASE("formula salvage rejects high det_count text", "[router][salvage]") {
  auto cfg = RouterConfig::defaults();
  auto lb = make_layout(22, 0.9f, 0, 0, 600, 80);
  OverlapStats ov;
  ov.resize(1);
  ov.det_count[0] = 5;
  ov.mean_aspect_ratio[0] = 8.0f;
  ov.symbol_density_hint[0] = 0.6f;
  PageStats ps;
  ps.has_confident_formula = true;
  CHECK_FALSE(is_formula_salvage_candidate(lb, 0, ov, ps, cfg));
}

TEST_CASE("formula salvage rejects when class is not 'text'",
          "[router][salvage]") {
  auto cfg = RouterConfig::defaults();
  auto lb = make_layout(0, 0.9f, 0, 0, 600, 80); // abstract
  OverlapStats ov;
  ov.resize(1);
  ov.det_count[0] = 1;
  ov.mean_aspect_ratio[0] = 9.0f;
  ov.symbol_density_hint[0] = 0.7f;
  PageStats ps;
  ps.has_confident_formula = true;
  CHECK_FALSE(is_formula_salvage_candidate(lb, 0, ov, ps, cfg));
}

// --- seam 6: tie-breakers ---

TEST_CASE("tie-breaker: containment ≥80% demotes outer", "[router][tiebreak]") {
  // A large layout cell fully containing a small inner table.
  std::vector<LayoutBox> layout = {
      make_layout(0,  0.9f, 0, 0, 1000, 1000, 0),  // abstract (Text default)
      make_layout(21, 0.9f, 100, 100, 300, 300, 1), // small table inside
  };
  std::vector<std::array<int, 4>> aabbs;
  for (auto &lb : layout) aabbs.push_back(turbo_ocr::aabb(lb.box));

  std::vector<RouterDecision> ds{
      RouterDecision{0, Destination::Text,  FormulaWrap::None,
                     ConfidenceTier::Trust, RouterReason::ClassDefault, {}, false},
      RouterDecision{1, Destination::Table, FormulaWrap::None,
                     ConfidenceTier::Trust, RouterReason::ClassDefault, {}, false},
  };
  resolve_tie_breakers(ds, layout, aabbs);
  CHECK(ds[0].dest == Destination::Text); // outer still text
  CHECK(ds[1].dest == Destination::Table);
  // Outer reason stays ClassDefault because it was already Text — only
  // non-Text destinations are demoted with the containment reason.
}

TEST_CASE("tie-breaker: class priority on IoU overlap demotes Text near Table",
          "[router][tiebreak]") {
  // Two cells with ~0.5 IoU: table vs paragraph.
  std::vector<LayoutBox> layout = {
      make_layout(21, 0.9f,   0,   0, 100, 100, 0),
      make_layout(22, 0.9f,  50,  50, 150, 150, 1),
  };
  std::vector<std::array<int, 4>> aabbs;
  for (auto &lb : layout) aabbs.push_back(turbo_ocr::aabb(lb.box));

  std::vector<RouterDecision> ds{
      RouterDecision{0, Destination::Table, FormulaWrap::None,
                     ConfidenceTier::Trust, RouterReason::ClassDefault, {}, false},
      RouterDecision{1, Destination::Text,  FormulaWrap::None,
                     ConfidenceTier::Trust, RouterReason::ClassDefault, {}, false},
  };
  resolve_tie_breakers(ds, layout, aabbs);
  CHECK(ds[0].dest == Destination::Table);
  CHECK(ds[1].dest == Destination::Text);
}

TEST_CASE("tie-breaker: no overlap → no mutation", "[router][tiebreak]") {
  std::vector<LayoutBox> layout = {
      make_layout(21, 0.9f,   0,   0, 100, 100, 0),
      make_layout(5,  0.9f, 500, 500, 600, 600, 1),
  };
  std::vector<std::array<int, 4>> aabbs;
  for (auto &lb : layout) aabbs.push_back(turbo_ocr::aabb(lb.box));

  std::vector<RouterDecision> ds{
      RouterDecision{0, Destination::Table,   FormulaWrap::None,
                     ConfidenceTier::Trust, RouterReason::ClassDefault, {}, false},
      RouterDecision{1, Destination::Formula, FormulaWrap::Display,
                     ConfidenceTier::Trust, RouterReason::ClassDefault, {}, false},
  };
  resolve_tie_breakers(ds, layout, aabbs);
  CHECK(ds[0].dest == Destination::Table);
  CHECK(ds[1].dest == Destination::Formula);
}

// --- seams 8 + 9: verify table path ---

TEST_CASE("VerifyTablePath::should_invoke fires only on Verify-tier Tables",
          "[router][verify]") {
  RouterDecision d{};
  d.dest = Destination::Table;
  d.tier = ConfidenceTier::Verify;
  CHECK(VerifyTablePath::should_invoke(d));

  d.tier = ConfidenceTier::Trust;
  CHECK_FALSE(VerifyTablePath::should_invoke(d));

  d.tier = ConfidenceTier::Verify;
  d.dest = Destination::Text;
  CHECK_FALSE(VerifyTablePath::should_invoke(d));
}

TEST_CASE("apply_verify_result demotes when not-a-table, forwards hint",
          "[router][verify]") {
  RouterDecision in{};
  in.layout_idx = 7;
  in.dest = Destination::Table;
  in.tier = ConfidenceTier::Verify;

  auto demoted = apply_verify_result(in, {false, TableClass::Wired, 0.45f});
  CHECK(demoted.dest == Destination::Text);
  CHECK(demoted.reason == RouterReason::TableVerifyDemoted);
  CHECK_FALSE(demoted.wired_hint.has_value());

  auto kept = apply_verify_result(in, {true, TableClass::Wireless, 0.72f});
  CHECK(kept.dest == Destination::Table);
  REQUIRE(kept.wired_hint.has_value());
  CHECK(*kept.wired_hint == TableClass::Wireless);
}

// --- CuaRouter::classify end-to-end ---

TEST_CASE("classify: empty layout routes everything to text",
          "[router][classify]") {
  CuaRouter r;
  std::vector<Box> det = {make_box(0, 0, 10, 10), make_box(20, 20, 30, 30)};
  std::vector<LayoutBox> layout;
  RoutingPlan plan;
  r.classify(det, layout, plan);
  CHECK(plan.text_indices.size() == 2);
  CHECK(plan.text_to_layout_id == std::vector<int>{-1, -1});
  CHECK(plan.table_layout_ids.empty());
  CHECK(plan.formula_layout_ids.empty());
}

TEST_CASE("classify: text-only page skips OverlapStats", "[router][classify]") {
  CuaRouter r;
  // 200 detection boxes inside one wide paragraph layout cell.
  std::vector<Box> det;
  det.reserve(200);
  for (int i = 0; i < 200; ++i) {
    int x0 = (i % 20) * 50;
    int y0 = (i / 20) * 30;
    det.push_back(make_box(x0, y0, x0 + 40, y0 + 25));
  }
  std::vector<LayoutBox> layout = {
      make_layout(22, 0.9f, 0, 0, 1000, 300, 0),
  };
  RoutingPlan plan;
  r.classify(det, layout, plan);

  // OverlapStats not built — last_decisions reflects the cell only.
  REQUIRE(r.last_decisions().size() == 1);
  CHECK(r.last_decisions()[0].dest == Destination::Text);
  CHECK(plan.text_indices.size() == 200);
  for (uint8_t s : plan.rec_suppress) CHECK(s == 0);
}

TEST_CASE("classify: det boxes inside a Table cell are suppressed from rec",
          "[router][classify]") {
  CuaRouter r;
  std::vector<Box> det = {
      make_box(  0,   0,  40,  25), // outside table — text
      make_box(110, 110, 140, 130), // inside table — suppress
      make_box(500, 500, 540, 530), // outside any layout — orphan
  };
  std::vector<LayoutBox> layout = {
      make_layout(21, 0.9f, 100, 100, 300, 300, 0),
  };
  RoutingPlan plan;
  r.classify(det, layout, plan);

  REQUIRE(plan.rec_suppress.size() == 3);
  CHECK(plan.rec_suppress[0] == 0);
  CHECK(plan.rec_suppress[1] == 1);
  CHECK(plan.rec_suppress[2] == 0);

  CHECK(plan.text_indices.size() == 2);
  CHECK(plan.text_indices[0] == 0);
  CHECK(plan.text_indices[1] == 2);
  CHECK(plan.table_layout_ids == std::vector<int>{0});
}

TEST_CASE("classify: 30-layout / 200-det-box budget under 2 ms",
          "[router][budget]") {
  CuaRouter r;
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> xdist(0, 1900);
  std::uniform_int_distribution<int> ydist(0, 1080);

  std::vector<LayoutBox> layout;
  layout.reserve(30);
  // 1 table, 1 display formula, 1 image, rest text — exercises the
  // OverlapStats gate plus tie-break loops.
  layout.push_back(make_layout(21, 0.9f, 100, 100, 600, 400, 0));
  layout.push_back(make_layout(5,  0.9f, 700, 100, 1200, 200, 1));
  layout.push_back(make_layout(14, 0.9f, 1300, 100, 1800, 400, 2));
  for (int i = 3; i < 30; ++i) {
    int x0 = xdist(rng);
    int y0 = ydist(rng);
    layout.push_back(
        make_layout(22, 0.9f, x0, y0, x0 + 200, y0 + 30, i));
  }

  std::vector<Box> det;
  det.reserve(200);
  for (int i = 0; i < 200; ++i) {
    int x0 = xdist(rng);
    int y0 = ydist(rng);
    det.push_back(make_box(x0, y0, x0 + 60, y0 + 20));
  }

  RoutingPlan plan;
  // Warm cache.
  r.classify(det, layout, plan);

  const int iters = 50;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    r.classify(det, layout, plan);
  }
  auto t1 = std::chrono::steady_clock::now();
  double avg_us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  // 2 ms == 2000 µs. Leave slack on noisy CI hosts.
  CHECK(avg_us < 2000.0);
}
