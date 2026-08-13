// Unit tests for the recursive XY-cut reading-order algorithm.
//
// Exercises projection_by_bboxes, split_projection_profile, and
// assign_reading_order on synthetic layouts: single column, two
// columns, header + two columns, single box, and empty input.

#include <catch_amalgamated.hpp>
#include <algorithm>
#include <vector>

#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/analysis/layout/blocks/child_blocks.h"
#include "turbo_ocr/analysis/layout/blocks/match_unsorted.h"
#include "turbo_ocr/analysis/layout/order/reading_order.h"
#include "turbo_ocr/analysis/layout/blocks/text_line_cluster.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::layout::assign_reading_order;
using turbo_ocr::layout::assign_reading_order_for_results;
using turbo_ocr::layout::LayoutBox;
using turbo_ocr::layout::projection_by_bboxes;
using turbo_ocr::layout::recursive_xy_cut;
using turbo_ocr::layout::split_projection_profile;

namespace {

// Build a 4-corner Box from (x0, y0, x1, y1).
Box make_box(int x0, int y0, int x1, int y1) {
  return Box{{{{{x0, y0}}, {{x1, y0}}, {{x1, y1}}, {{x0, y1}}}}};
}

LayoutBox make_layout(int x0, int y0, int x1, int y1, int class_id = 22) {
  LayoutBox lb;
  lb.class_id = class_id;
  lb.score = 0.99f;
  lb.box = make_box(x0, y0, x1, y1);
  return lb;
}

} // namespace

namespace {
// Make an OCRResultItem with center inside the given AABB and a given
// layout_id. Used to test results-level reading-order grouping.
OCRResultItem make_result(int x0, int y0, int x1, int y1, int layout_id) {
  OCRResultItem r;
  r.text = "x";
  r.confidence = 0.9f;
  r.box = make_box(x0, y0, x1, y1);
  r.layout_id = layout_id;
  return r;
}
} // namespace



// ---- Class-aware bucketing (header → body → footer/reference) -----------

TEST_CASE("assign_reading_order hoists header above body and sinks footer",
          "[xy_cut][bucket]") {
  // Geometric position alone would order: header, body1, body2, footer
  // (top-to-bottom). We also add a malformed layout where the footer is
  // accidentally placed mid-page — the class-aware bucket sort must
  // still push it to the end.
  std::vector<LayoutBox> layout = {
      make_layout(50, 200, 450, 280, /*class_id=*/22),  // body text 1
      make_layout(50, 50,  450, 100, /*class_id=*/12),  // header
      make_layout(50, 320, 450, 380, /*class_id=*/8),   // footer (mis-placed mid-body)
      make_layout(50, 290, 450, 310, /*class_id=*/22),  // body text 2 (after footer y)
  };
  auto order = assign_reading_order(layout);
  REQUIRE(order.size() == 4);
  CHECK(order[0] == 1);  // header
  // body bucket: text 1 above text 2
  CHECK(order[1] == 0);
  CHECK(order[2] == 3);
  CHECK(order[3] == 2);  // footer last regardless of geometric position
}

TEST_CASE("assign_reading_order sinks reference/footnote/vision_footnote to bottom",
          "[xy_cut][bucket]") {
  // Reference should land after body. Multi-line footnote keeps internal
  // top-to-bottom order within the bottom bucket.
  std::vector<LayoutBox> layout = {
      make_layout(50, 200, 450, 240, /*class_id=*/18),  // reference (early)
      make_layout(50,  60, 450, 100, /*class_id=*/22),  // body 1
      make_layout(50, 110, 450, 150, /*class_id=*/22),  // body 2
      make_layout(50, 260, 450, 280, /*class_id=*/10),  // footnote (lower)
      make_layout(50, 290, 450, 310, /*class_id=*/24),  // vision_footnote
  };
  auto order = assign_reading_order(layout);
  REQUIRE(order.size() == 5);
  CHECK(order[0] == 1);  // body 1
  CHECK(order[1] == 2);  // body 2
  // bottom bucket order is XY-cut by y_min: reference (y=200), footnote (260), vision_footnote (290)
  CHECK(order[2] == 0);  // reference
  CHECK(order[3] == 3);  // footnote
  CHECK(order[4] == 4);  // vision_footnote
}

TEST_CASE("assign_reading_order_for_results: header text reads first across buckets",
          "[xy_cut][bucket]") {
  // Real-world shape: header line + two body paragraphs + footnote.
  // Each layout region holds one OCR result.
  std::vector<LayoutBox> layout = {
      make_layout(50, 200, 450, 240, /*class_id=*/22),  // body 1
      make_layout(50,  60, 450, 100, /*class_id=*/12),  // header
      make_layout(50, 250, 450, 290, /*class_id=*/22),  // body 2
      make_layout(50, 320, 450, 360, /*class_id=*/10),  // footnote
  };
  std::vector<OCRResultItem> results = {
      make_result(50, 250, 450, 290, /*layout_id=*/2),  // body 2 line
      make_result(50,  60, 450, 100, /*layout_id=*/1),  // header line
      make_result(50, 320, 450, 360, /*layout_id=*/3),  // footnote line
      make_result(50, 200, 450, 240, /*layout_id=*/0),  // body 1 line
  };
  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 4);
  CHECK(order[0] == 1);  // header
  CHECK(order[1] == 3);  // body 1
  CHECK(order[2] == 0);  // body 2
  CHECK(order[3] == 2);  // footnote (bottom bucket)
}

TEST_CASE("assign_reading_order_for_results: row tolerance handles table cell jitter",
          "[xy_cut][table]") {
  // A 3-column × 2-row table inside one layout box. OCR detection
  // produces a few pixels of y-jitter per cell — strict (y, x) sort
  // would interleave columns. The within-block sort must bucket by row
  // first, then sort x within each row.
  std::vector<LayoutBox> layout = {make_layout(50, 100, 950, 250, /*class_id=*/21)};  // table
  std::vector<OCRResultItem> results = {
      // Row 1, with 1-3 px y-jitter per cell.
      make_result( 60, 110, 200, 140, /*layout_id=*/0),  // R1-LEFT  cy ≈ 125
      make_result(360, 112, 500, 142, /*layout_id=*/0),  // R1-MID   cy ≈ 127
      make_result(660, 113, 800, 143, /*layout_id=*/0),  // R1-RIGHT cy ≈ 128
      // Row 2.
      make_result( 60, 200, 200, 230, /*layout_id=*/0),  // R2-LEFT  cy ≈ 215
      make_result(360, 201, 500, 231, /*layout_id=*/0),  // R2-MID   cy ≈ 216
      make_result(660, 202, 800, 232, /*layout_id=*/0),  // R2-RIGHT cy ≈ 217
  };
  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 6);
  // Expected row-major: R1-LEFT, R1-MID, R1-RIGHT, R2-LEFT, R2-MID, R2-RIGHT
  CHECK(order[0] == 0);
  CHECK(order[1] == 1);
  CHECK(order[2] == 2);
  CHECK(order[3] == 3);
  CHECK(order[4] == 4);
  CHECK(order[5] == 5);
}

TEST_CASE("assign_reading_order_for_results: orphan stays in body even near header band",
          "[xy_cut][bucket]") {
  // An orphan with no layout match goes into the body bucket. If it
  // happens to sit at the very top of the page (y=10) the body XY-cut
  // places it above the body region, but it still reads AFTER any
  // explicit header.
  std::vector<LayoutBox> layout = {
      make_layout(50,  60, 450, 100, /*class_id=*/12),   // header
      make_layout(50, 200, 450, 280, /*class_id=*/22),   // body
  };
  std::vector<OCRResultItem> results = {
      make_result(50, 220, 200, 240, /*layout_id=*/1),    // body line
      make_result(60,  10, 200,  30, /*layout_id=*/-1),   // orphan near top
      make_result(50,  60, 450, 100, /*layout_id=*/0),    // header line
  };
  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 3);
  CHECK(order[0] == 2);  // header line first (top bucket)
  // Body bucket: orphan (y=10) above body line (y=220).
  CHECK(order[1] == 1);  // orphan
  CHECK(order[2] == 0);  // body line
}

TEST_CASE("assign_layout_ids synthesises SupplementaryRegion for orphans",
          "[layout_ids][supplementary]") {
  // Two real layout boxes; result #1 falls inside layout[0], result #2
  // falls inside layout[1], result #0 has its centroid OUTSIDE both —
  // that's the orphan case. After assign_layout_ids:
  //   - layout vector grows by one entry (index 2) tagged
  //     class_id == kSupplementaryRegionClassId
  //   - the synthetic block's bbox encloses the orphan's bbox
  //   - the orphan's layout_id points at the synthetic block
  //   - the matched results keep their original layout_id
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(100, 100, 200, 200),   // idx 0
      make_layout(300, 300, 400, 400),   // idx 1
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(500, 500, 540, 520, /*layout_id=*/-1), // orphan
      make_result(110, 110, 190, 190, /*layout_id=*/-1), // → layout[0]
      make_result(310, 310, 390, 390, /*layout_id=*/-1), // → layout[1]
  };

  turbo_ocr::assign_layout_ids(results, layout);

  REQUIRE(layout.size() == 3);
  CHECK(layout[2].class_id ==
        turbo_ocr::layout::kSupplementaryRegionClassId);
  CHECK(layout[2].id == 2);
  CHECK(turbo_ocr::layout::label_name(layout[2].class_id) ==
        "SupplementaryRegion");

  // Synthetic bbox covers the orphan's AABB exactly (single orphan).
  auto [sx0, sy0, sx1, sy1] = turbo_ocr::aabb(layout[2].box);
  CHECK(sx0 == 500);
  CHECK(sy0 == 500);
  CHECK(sx1 == 540);
  CHECK(sy1 == 520);

  CHECK(results[0].layout_id == 2);  // orphan → SupplementaryRegion
  CHECK(results[1].layout_id == 0);  // matched
  CHECK(results[2].layout_id == 1);  // matched
}

TEST_CASE("assign_layout_ids: SupplementaryRegion encloses ALL orphans",
          "[layout_ids][supplementary]") {
  // Multiple scattered orphans → one SupplementaryRegion whose bbox
  // is the minimum-enclosing rectangle of all of them.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(100, 100, 200, 200),
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result( 50,  60,  80,  80, /*layout_id=*/-1),  // top-left orphan
      make_result(110, 110, 190, 190, /*layout_id=*/-1),  // matched
      make_result(500, 500, 540, 540, /*layout_id=*/-1),  // bottom-right orphan
      make_result(300,  20, 320,  40, /*layout_id=*/-1),  // top-right orphan
  };

  turbo_ocr::assign_layout_ids(results, layout);

  REQUIRE(layout.size() == 2);
  auto [sx0, sy0, sx1, sy1] = turbo_ocr::aabb(layout[1].box);
  // Min-enclosing of {50,60,80,80}, {500,500,540,540}, {300,20,320,40}
  CHECK(sx0 == 50);
  CHECK(sy0 == 20);
  CHECK(sx1 == 540);
  CHECK(sy1 == 540);

  CHECK(results[0].layout_id == 1);  // orphan → SupplementaryRegion
  CHECK(results[1].layout_id == 0);  // matched
  CHECK(results[2].layout_id == 1);
  CHECK(results[3].layout_id == 1);
}

TEST_CASE("assign_layout_ids: no orphans → no SupplementaryRegion appended",
          "[layout_ids][supplementary]") {
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(100, 100, 200, 200),
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(110, 110, 190, 190, /*layout_id=*/-1),
  };
  turbo_ocr::assign_layout_ids(results, layout);
  REQUIRE(layout.size() == 1);   // unchanged
  CHECK(results[0].layout_id == 0);
}

TEST_CASE("assign_layout_ids: empty layout stays empty (backward-compat)",
          "[layout_ids][supplementary]") {
  // When the caller did not request layout (empty input) we DO NOT
  // synthesise a SupplementaryRegion. The serializer then omits the
  // layout key + per-result layout_id keys entirely, keeping responses
  // byte-identical to pre-layout clients.
  std::vector<turbo_ocr::layout::LayoutBox> layout;
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(10, 20, 30, 40, /*layout_id=*/-1),
      make_result(50, 60, 70, 80, /*layout_id=*/-1),
  };
  turbo_ocr::assign_layout_ids(results, layout);
  REQUIRE(layout.empty());
  CHECK(results[0].layout_id == -1);
  CHECK(results[1].layout_id == -1);
}

TEST_CASE("assign_reading_order_for_results: orphans inside SupplementaryRegion "
          "still placed individually by XY-cut",
          "[layout_ids][supplementary][xy_cut]") {
  // Real layout that misses both result boxes — both become orphans,
  // get assigned to a synthesised SupplementaryRegion, and the
  // reading-order code must still emit them in geometric order rather
  // than treating the synthetic region as one indivisible block.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(800, 800, 900, 900),  // far-away real layout, contains nothing
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(50, 200, 100, 220, /*layout_id=*/-1),  // bottom
      make_result(50,  20, 100,  40, /*layout_id=*/-1),  // top
  };
  turbo_ocr::assign_layout_ids(results, layout);
  REQUIRE(layout.size() == 2);
  CHECK(results[0].layout_id == 1);
  CHECK(results[1].layout_id == 1);

  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 2);
  // Top result should come first geometrically.
  CHECK(order[0] == 1);
  CHECK(order[1] == 0);
}
