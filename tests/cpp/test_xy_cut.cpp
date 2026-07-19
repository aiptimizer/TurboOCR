// Unit tests for the recursive XY-cut reading-order algorithm.
//
// Exercises projection_by_bboxes, split_projection_profile, and
// assign_reading_order on synthetic layouts: single column, two
// columns, header + two columns, single box, and empty input.

#include <catch_amalgamated.hpp>
#include <algorithm>
#include <vector>

#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/layout/child_blocks.h"
#include "turbo_ocr/layout/match_unsorted.h"
#include "turbo_ocr/layout/reading_order.h"
#include "turbo_ocr/layout/text_line_cluster.h"

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

TEST_CASE("projection_by_bboxes simple X projection", "[xy_cut]") {
  std::vector<std::array<int, 4>> rects = {
      {0, 0, 10, 5},
      {20, 0, 30, 5},
  };
  auto p = projection_by_bboxes(rects, 0);
  REQUIRE(p.size() == 30);
  CHECK(p[0] == 1);
  CHECK(p[5] == 1);
  CHECK(p[15] == 0);  // gap between rects
  CHECK(p[25] == 1);
}

TEST_CASE("projection_by_bboxes caps histogram at 4096 bins for large pages",
          "[xy_cut][cap]") {
  // Adversarial: a 10000-pixel-wide page with two boxes at the extremes.
  // Without the cap the histogram would be 10000 ints (~40 KB); with the
  // cap the histogram never exceeds 4096 bins regardless of page extent.
  std::vector<std::array<int, 4>> rects = {
      {0,    0, 100,  10},
      {9900, 0, 10000, 10},
  };
  auto p = projection_by_bboxes(rects, 0);
  CHECK(p.size() <= 4096);
  // Both bands must still be representable: the first 100 px and the last
  // 100 px each contribute > 0 to at least one bin.
  bool low_set = false, high_set = false;
  for (size_t i = 0; i < p.size() / 2; ++i) {
    if (p[i] > 0) { low_set = true; break; }
  }
  for (size_t i = p.size() / 2; i < p.size(); ++i) {
    if (p[i] > 0) { high_set = true; break; }
  }
  CHECK(low_set);
  CHECK(high_set);
  // And the gap between them remains visible: there exists at least one
  // empty bin between the two populated regions.
  bool gap_seen = false;
  for (size_t i = 0; i < p.size(); ++i) {
    if (p[i] == 0) { gap_seen = true; break; }
  }
  CHECK(gap_seen);
}

TEST_CASE("projection_by_bboxes preserves resolution for small pages",
          "[xy_cut][cap]") {
  // Below the cap threshold the histogram must remain pixel-accurate so
  // that small-input behaviour (and the rest of the test suite) is
  // unchanged.
  std::vector<std::array<int, 4>> rects = {
      {0, 0, 10, 5},
      {20, 0, 30, 5},
  };
  auto p = projection_by_bboxes(rects, 0);
  REQUIRE(p.size() == 30);
  CHECK(p[0] == 1);
  CHECK(p[15] == 0);
  CHECK(p[25] == 1);
}

TEST_CASE("recursive_xy_cut splits large-page two-column layout correctly",
          "[xy_cut][cap]") {
  // 10000x10000 page with a clean two-column layout; each column has two
  // stacked paragraphs. Even with the projection histogram downsampled,
  // XY-cut must still split into 2 columns × 2 rows in reading order.
  std::vector<std::array<int, 4>> rects = {
      {6000, 5000, 9500, 6000},   // right-bottom (idx 0)
      {500,   500, 4000, 1500},   // left-top     (idx 1)
      {6000,  500, 9500, 1500},   // right-top    (idx 2)
      {500,  5000, 4000, 6000},   // left-bottom  (idx 3)
  };
  std::vector<int> indices = {0, 1, 2, 3};
  std::vector<int> order;
  recursive_xy_cut(rects, indices, order);
  REQUIRE(order.size() == 4);
  CHECK(order[0] == 1);  // left-top
  CHECK(order[1] == 3);  // left-bottom
  CHECK(order[2] == 2);  // right-top
  CHECK(order[3] == 0);  // right-bottom
}

TEST_CASE("split_projection_profile finds gaps", "[xy_cut]") {
  std::vector<int> proj = {1, 1, 0, 0, 0, 1, 1, 0, 1};
  auto seg = split_projection_profile(proj, 0, 1);
  // Sig indices: 0,1,5,6,8. Index gaps: 1, 4, 1, 2.
  // Gaps strictly greater than min_gap=1 split the run: positions
  // (1→5) gap=4 and (6→8) gap=2 both qualify, yielding 3 segments.
  REQUIRE(seg.size() == 3);
  CHECK(seg[0].start == 0); CHECK(seg[0].end == 2);
  CHECK(seg[1].start == 5); CHECK(seg[1].end == 7);
  CHECK(seg[2].start == 8); CHECK(seg[2].end == 9);
}

TEST_CASE("split_projection_profile single segment when min_gap large",
          "[xy_cut]") {
  std::vector<int> proj = {1, 1, 0, 0, 0, 1, 1};
  auto seg = split_projection_profile(proj, 0, 5);
  REQUIRE(seg.size() == 1);
  CHECK(seg[0].start == 0);
  CHECK(seg[0].end == 7);
}

TEST_CASE("assign_reading_order empty layout", "[xy_cut]") {
  std::vector<LayoutBox> layout;
  auto order = assign_reading_order(layout);
  CHECK(order.empty());
}

TEST_CASE("assign_reading_order single box", "[xy_cut]") {
  std::vector<LayoutBox> layout = {make_layout(10, 10, 100, 50)};
  auto order = assign_reading_order(layout);
  REQUIRE(order.size() == 1);
  CHECK(order[0] == 0);
}

TEST_CASE("assign_reading_order single column top-to-bottom", "[xy_cut]") {
  // Three stacked paragraphs in a single column.
  std::vector<LayoutBox> layout = {
      make_layout(50, 300, 550, 400),  // bottom (idx 0)
      make_layout(50, 100, 550, 200),  // top    (idx 1)
      make_layout(50, 220, 550, 290),  // middle (idx 2)
  };
  auto order = assign_reading_order(layout);
  REQUIRE(order.size() == 3);
  CHECK(order[0] == 1);  // top first
  CHECK(order[1] == 2);  // middle next
  CHECK(order[2] == 0);  // bottom last
}

TEST_CASE("assign_reading_order two columns left-then-right", "[xy_cut]") {
  // Two-column page: left column has two paragraphs stacked, right
  // column has two stacked. Reading order is left-top, left-bottom,
  // right-top, right-bottom.
  std::vector<LayoutBox> layout = {
      make_layout(420, 300, 780, 400),  // right-bottom (idx 0)
      make_layout(20, 100, 380, 200),   // left-top     (idx 1)
      make_layout(420, 100, 780, 200),  // right-top    (idx 2)
      make_layout(20, 300, 380, 400),   // left-bottom  (idx 3)
  };
  auto order = assign_reading_order(layout);
  REQUIRE(order.size() == 4);
  CHECK(order[0] == 1);
  CHECK(order[1] == 3);
  CHECK(order[2] == 2);
  CHECK(order[3] == 0);
}

TEST_CASE("assign_reading_order header spanning two columns", "[xy_cut]") {
  // Page-wide header on top, then a two-column body underneath. The
  // header overlaps both columns, so the top-level X-projection sees a
  // single column. Recursion then Y-splits header from body, and the
  // body's per-row Y bands each split into left/right cells. Result:
  // header, then row1 (left-of-row1, right-of-row1), then row2.
  std::vector<LayoutBox> layout = {
      make_layout(20, 200, 380, 300),    // body row1 left  (idx 0)
      make_layout(20, 20, 780, 80),      // header (full)   (idx 1)
      make_layout(420, 200, 780, 300),   // body row1 right (idx 2)
      make_layout(20, 320, 380, 420),    // body row2 left  (idx 3)
      make_layout(420, 320, 780, 420),   // body row2 right (idx 4)
  };
  auto order = assign_reading_order(layout);
  REQUIRE(order.size() == 5);
  CHECK(order[0] == 1);  // header first
  CHECK(order[1] == 0);  // row1 left
  CHECK(order[2] == 2);  // row1 right
  CHECK(order[3] == 3);  // row2 left
  CHECK(order[4] == 4);  // row2 right
}

TEST_CASE("assign_reading_order returns complete permutation on overlap",
          "[xy_cut]") {
  // Two heavily overlapping boxes: the algorithm may bail out of the
  // recursion early, but assign_reading_order's defense-in-depth must
  // still emit every input index exactly once.
  std::vector<LayoutBox> layout = {
      make_layout(100, 100, 500, 400),
      make_layout(110, 110, 490, 390),
  };
  auto order = assign_reading_order(layout);
  REQUIRE(order.size() == 2);
  std::vector<int> sorted = order;
  std::sort(sorted.begin(), sorted.end());
  CHECK(sorted[0] == 0);
  CHECK(sorted[1] == 1);
}

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

TEST_CASE("assign_reading_order_for_results groups by layout XY-cut order",
          "[xy_cut]") {
  // Two-column layout: layout[0] is left column, layout[1] is right.
  // XY-cut puts left first, then right.
  std::vector<LayoutBox> layout = {
      make_layout(20, 100, 380, 400),    // idx 0 = left column
      make_layout(420, 100, 780, 400),   // idx 1 = right column
  };

  // Results in arbitrary input order, with layout_id pointing at their
  // owning region. Within each region, the y-tiebreak orders top-to-
  // bottom and x-tiebreak orders left-to-right.
  std::vector<OCRResultItem> results = {
      make_result(420, 300, 600, 320, /*layout_id=*/1),  // R-bottom
      make_result(20,  150, 200, 170, /*layout_id=*/0),  // L-top
      make_result(420, 150, 600, 170, /*layout_id=*/1),  // R-top
      make_result(20,  300, 200, 320, /*layout_id=*/0),  // L-bottom
  };

  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 4);
  // Expected: L-top (1), L-bottom (3), R-top (2), R-bottom (0)
  CHECK(order[0] == 1);
  CHECK(order[1] == 3);
  CHECK(order[2] == 2);
  CHECK(order[3] == 0);
}

TEST_CASE("assign_reading_order_for_results places orphan ABOVE layout via XY-cut",
          "[xy_cut]") {
  // A page number / header the layout model missed sits above the
  // body paragraph. Augmented XY-cut feeds both into the cut and
  // partitions them as two stacked rows — the orphan must be read
  // first, not appended to the end.
  std::vector<LayoutBox> layout = {make_layout(50, 200, 450, 500)};  // body
  std::vector<OCRResultItem> results = {
      make_result(50, 250, 200, 270, /*layout_id=*/0),   // first body line
      make_result(60,  20, 180,  40, /*layout_id=*/-1),  // orphan above
      make_result(50, 300, 200, 320, /*layout_id=*/0),   // second body line
  };
  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 3);
  CHECK(order[0] == 1);  // orphan (page header) first — read before body
  CHECK(order[1] == 0);  // first body line
  CHECK(order[2] == 2);  // second body line
}

TEST_CASE("assign_reading_order_for_results places orphan BELOW layout",
          "[xy_cut]") {
  // Footer that the layout model missed: must come AFTER the body even
  // though the legacy code already happened to put orphans at the end
  // — here we assert it lands by XY-cut position, not by accident.
  std::vector<LayoutBox> layout = {make_layout(50, 50, 450, 350)};
  std::vector<OCRResultItem> results = {
      make_result(60, 400, 200, 420, /*layout_id=*/-1),  // orphan footer
      make_result(60,  60, 200,  80, /*layout_id=*/0),   // body line 1
      make_result(60, 100, 200, 120, /*layout_id=*/0),   // body line 2
  };
  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 3);
  CHECK(order[0] == 1);  // body line 1
  CHECK(order[1] == 2);  // body line 2
  CHECK(order[2] == 0);  // orphan footer last
}

TEST_CASE("assign_reading_order_for_results orphan between two columns",
          "[xy_cut]") {
  // Two-column doc with an orphan between the columns vertically. The
  // augmented XY-cut splits into three columns horizontally: left col,
  // orphan-only middle col, right col.
  std::vector<LayoutBox> layout = {
      make_layout(20, 100, 200, 400),    // left col
      make_layout(420, 100, 600, 400),   // right col
  };
  std::vector<OCRResultItem> results = {
      make_result(420, 150, 580, 170, /*layout_id=*/1),   // right line
      make_result(20,  150, 180, 170, /*layout_id=*/0),   // left line
      make_result(260, 200, 380, 220, /*layout_id=*/-1),  // orphan in gutter
  };
  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 3);
  // Three columns L→C→R: left line, orphan, right line.
  CHECK(order[0] == 1);  // left
  CHECK(order[1] == 2);  // orphan
  CHECK(order[2] == 0);  // right
}

TEST_CASE("assign_reading_order_for_results empty layout falls back to y/x",
          "[xy_cut]") {
  std::vector<LayoutBox> layout;
  std::vector<OCRResultItem> results = {
      make_result(200, 50,  300, 70,  /*layout_id=*/-1),  // top-right
      make_result(10,  10,  100, 30,  /*layout_id=*/-1),  // top-left
      make_result(10,  100, 100, 130, /*layout_id=*/-1),  // bottom-left
  };
  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 3);
  // y-then-x sort: top-left (1), top-right (0), bottom-left (2)
  CHECK(order[0] == 1);
  CHECK(order[1] == 0);
  CHECK(order[2] == 2);
}

TEST_CASE("assign_reading_order_for_results empty results", "[xy_cut]") {
  std::vector<LayoutBox> layout = {make_layout(0, 0, 100, 100)};
  std::vector<OCRResultItem> results;
  auto order = assign_reading_order_for_results(results, layout);
  CHECK(order.empty());
}

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

// =====  Layer 2: label-aware match_unsorted_blocks  =====

TEST_CASE("match_unsorted_blocks: doc_title pinned to top via weighted insert",
          "[match_unsorted]") {
  // Body has two text blocks at y=200 and y=400. A doc_title (class 6)
  // appears at y=50 — should land at position 0.
  std::vector<turbo_ocr::layout::UnsortedBlock> sorted = {
      {/*idx=*/0, {{50, 200, 750, 240}}, turbo_ocr::layout::OrderLabel::kBody, 22},
      {/*idx=*/1, {{50, 400, 750, 440}}, turbo_ocr::layout::OrderLabel::kBody, 22},
  };
  std::vector<turbo_ocr::layout::UnsortedBlock> unsorted = {
      {/*idx=*/2, {{200, 50, 600, 90}},
       turbo_ocr::layout::OrderLabel::kDocTitle, /*class_id=*/6},
  };
  turbo_ocr::layout::match_unsorted_blocks(sorted, unsorted, /*text_line_width=*/700, turbo_ocr::layout::Direction::kHorizontal, /*layout=*/{});
  REQUIRE(sorted.size() == 3);
  CHECK(sorted[0].layout_idx == 2);  // doc_title first
  CHECK(sorted[1].layout_idx == 0);
  CHECK(sorted[2].layout_idx == 1);
}

TEST_CASE("match_unsorted_blocks: cross_reference appended via reference_insert",
          "[match_unsorted]") {
  // A reference block at the bottom of the page. reference_insert
  // should place it AFTER the highest sorted block whose y2 ≤
  // reference y1.
  std::vector<turbo_ocr::layout::UnsortedBlock> sorted = {
      {/*idx=*/0, {{50,  60, 750, 100}}, turbo_ocr::layout::OrderLabel::kBody, 22},
      {/*idx=*/1, {{50, 200, 750, 240}}, turbo_ocr::layout::OrderLabel::kBody, 22},
      {/*idx=*/2, {{50, 400, 750, 440}}, turbo_ocr::layout::OrderLabel::kBody, 22},
  };
  std::vector<turbo_ocr::layout::UnsortedBlock> unsorted = {
      {/*idx=*/3, {{50, 500, 750, 540}},
       turbo_ocr::layout::OrderLabel::kCrossReference, /*class_id=*/18},
  };
  turbo_ocr::layout::match_unsorted_blocks(sorted, unsorted, 700, turbo_ocr::layout::Direction::kHorizontal, /*layout=*/{});
  REQUIRE(sorted.size() == 4);
  // Reference is below all three sorted blocks; goes after index 2.
  CHECK(sorted[3].layout_idx == 3);
}

TEST_CASE("match_unsorted_blocks: unordered (page number) via manhattan_insert",
          "[match_unsorted]") {
  // A `number` block (page number, class_id=16) at the bottom-left.
  // manhattan_insert places it after the nearest sorted block by L1
  // distance between top-left corners.
  std::vector<turbo_ocr::layout::UnsortedBlock> sorted = {
      {/*idx=*/0, {{50, 100, 750, 140}}, turbo_ocr::layout::OrderLabel::kBody, 22},
      {/*idx=*/1, {{50, 500, 200, 530}}, turbo_ocr::layout::OrderLabel::kBody, 22},
  };
  std::vector<turbo_ocr::layout::UnsortedBlock> unsorted = {
      {/*idx=*/2, {{60, 540, 110, 560}},
       turbo_ocr::layout::OrderLabel::kUnordered, /*class_id=*/16},
  };
  turbo_ocr::layout::match_unsorted_blocks(sorted, unsorted, 700, turbo_ocr::layout::Direction::kHorizontal, /*layout=*/{});
  REQUIRE(sorted.size() == 3);
  // Page number should land after the closer sorted block (idx 1 at y=500),
  // not after the far one (idx 0 at y=100).
  CHECK(sorted[2].layout_idx == 2);
}

TEST_CASE("match_unsorted_blocks: vision below text bound via weighted_insert",
          "[match_unsorted]") {
  // Two text columns; an image sits below the left column. The
  // weighted-distance insert should land it adjacent to its nearest
  // text neighbor.
  std::vector<turbo_ocr::layout::UnsortedBlock> sorted = {
      {/*idx=*/0, {{ 50, 100, 350, 140}}, turbo_ocr::layout::OrderLabel::kBody, 22},
      {/*idx=*/1, {{450, 100, 750, 140}}, turbo_ocr::layout::OrderLabel::kBody, 22},
      {/*idx=*/2, {{450, 200, 750, 240}}, turbo_ocr::layout::OrderLabel::kBody, 22},
  };
  std::vector<turbo_ocr::layout::UnsortedBlock> unsorted = {
      {/*idx=*/3, {{ 50, 200, 350, 380}},
       turbo_ocr::layout::OrderLabel::kVision, /*class_id=*/14},
  };
  turbo_ocr::layout::match_unsorted_blocks(sorted, unsorted, 300, turbo_ocr::layout::Direction::kHorizontal, /*layout=*/{});
  REQUIRE(sorted.size() == 4);
  // Vision block should be inserted somewhere in the sequence; the
  // important property is that it ended up next to its nearest text.
  // Find its position and check the neighbor is sensible.
  size_t vision_pos = 0;
  for (size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i].layout_idx == 3) { vision_pos = i; break; }
  }
  CHECK(vision_pos > 0);  // never first since left column header is above
}

TEST_CASE("order_label_for: PP-DocLayoutV3 class_id mapping",
          "[match_unsorted]") {
  using turbo_ocr::layout::order_label_for;
  using turbo_ocr::layout::OrderLabel;
  CHECK(order_label_for(6)  == OrderLabel::kDocTitle);
  CHECK(order_label_for(17) == OrderLabel::kParagraphTitle);
  CHECK(order_label_for(7)  == OrderLabel::kVisionTitle);
  CHECK(order_label_for(14) == OrderLabel::kVision);
  CHECK(order_label_for(21) == OrderLabel::kVision);  // table
  CHECK(order_label_for(3)  == OrderLabel::kVision);  // chart
  CHECK(order_label_for(18) == OrderLabel::kCrossReference);
  CHECK(order_label_for(10) == OrderLabel::kCrossReference);  // footnote
  CHECK(order_label_for(16) == OrderLabel::kUnordered);  // page number
  CHECK(order_label_for(20) == OrderLabel::kUnordered);  // seal
  CHECK(order_label_for(22) == OrderLabel::kBody);       // text — XY-cut
  CHECK(order_label_for(4)  == OrderLabel::kBody);       // content
  CHECK(order_label_for(-1) == OrderLabel::kBody);       // SupplementaryRegion
}

// =====  Child-block detection + splice (PaddleX layer 3)  =====

TEST_CASE("detect_child_blocks: doc_title attaches small adjacent text",
          "[child_blocks]") {
  // doc_title with a short subtitle line right underneath. The
  // subtitle's short side is well under 80% of the title's, edge
  // distance is under 2× text_line_height — should attach.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(100, 100, 700, 160, /*class_id=*/6),   // doc_title
      make_layout(150, 180, 650, 210, /*class_id=*/22),  // text subtitle
      make_layout( 50, 400, 750, 800, /*class_id=*/22),  // body text
  };
  auto links = turbo_ocr::layout::detect_child_blocks(layout, /*tlh=*/30);
  CHECK(links.size() == 3);
  // doc_title (idx 0) should claim subtitle (idx 1).
  CHECK(links[0].child_indices.size() == 1);
  CHECK(links[0].child_indices[0] == 1);
  CHECK(links[1].child_indices.empty());
  CHECK(links[2].child_indices.empty());
}

TEST_CASE("detect_child_blocks: vision attaches a single-line caption below",
          "[child_blocks]") {
  // image with a "Figure 1: …" single-line caption right underneath.
  // Caption is left-aligned with image's left edge AND vertical edge
  // distance is small.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(100, 100, 600, 400, /*class_id=*/14),  // image
      make_layout(100, 410, 580, 440, /*class_id=*/22),  // caption text
  };
  auto links = turbo_ocr::layout::detect_child_blocks(layout, /*tlh=*/30);
  REQUIRE(links.size() == 2);
  // image (idx 0) should claim caption (idx 1).
  CHECK(links[0].child_indices.size() == 1);
  CHECK(links[0].child_indices[0] == 1);
}

TEST_CASE("detect_child_blocks: paragraph_title attaches sub-headings",
          "[child_blocks]") {
  // A paragraph_title at y=100, then another paragraph_title at y=140
  // with the same left-edge — the second is a sub-heading and should
  // attach to the first.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(100, 100, 500, 130, /*class_id=*/17),  // top
      make_layout(100, 140, 450, 165, /*class_id=*/17),  // sub-heading
  };
  auto links = turbo_ocr::layout::detect_child_blocks(layout, /*tlh=*/30);
  REQUIRE(links.size() == 2);
  CHECK(links[0].child_indices.size() == 1);
  CHECK(links[0].child_indices[0] == 1);
  CHECK(links[1].child_indices.empty());
}

TEST_CASE("detect_child_blocks: no candidates → empty links",
          "[child_blocks]") {
  // text-only layout: nothing has a parent role.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout( 50, 100, 750, 200, /*class_id=*/22),
      make_layout( 50, 220, 750, 320, /*class_id=*/22),
  };
  auto links = turbo_ocr::layout::detect_child_blocks(layout, 25);
  REQUIRE(links.size() == 2);
  CHECK(links[0].child_indices.empty());
  CHECK(links[1].child_indices.empty());
}

TEST_CASE("assign_reading_order_for_results: vision + caption emit contiguously",
          "[child_blocks][reading_order]") {
  // Three layout cells: a body paragraph (text), an image, and a
  // single-line caption text right below the image. The vision should
  // attract the caption as a child so caption emits IMMEDIATELY after
  // image's results, before the body paragraph's that comes later.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout( 50, 800, 750, 900, /*class_id=*/22),  // body text (later)
      make_layout(100, 100, 600, 400, /*class_id=*/14),  // image
      make_layout(100, 410, 580, 440, /*class_id=*/22),  // caption
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(100, 410, 580, 440, /*layout_id=*/-1),  // caption text
      make_result( 50, 820, 750, 850, /*layout_id=*/-1),  // body text
  };
  // assign_layout_ids first (mutates layout_id and may add SupplementaryRegion)
  turbo_ocr::assign_layout_ids(results, layout);
  auto order = turbo_ocr::layout::assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 2);
  // Caption (results[0]) should come BEFORE body (results[1]) because
  // it's emitted under the image's slot via child splice, and the
  // image sits above the body.
  CHECK(order[0] == 0);  // caption
  CHECK(order[1] == 1);  // body
}

// =====  Text-line clustering pre-pass + direction inference  =====

TEST_CASE("cluster_text_lines: groups boxes on the same y-band into one line",
          "[cluster][text_lines]") {
  // Two y-bands, three boxes per band — each band should become one
  // TextLine, so num_of_lines = 2.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(0, 0, 1000, 200, /*class_id=*/22),
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result( 50, 50, 200, 80, /*layout_id=*/0),
      make_result(220, 50, 400, 80, /*layout_id=*/0),
      make_result(420, 50, 600, 80, /*layout_id=*/0),
      make_result( 50,150, 200,180, /*layout_id=*/0),
      make_result(220,150, 400,180, /*layout_id=*/0),
      make_result(420,150, 600,180, /*layout_id=*/0),
  };
  turbo_ocr::layout::cluster_text_lines(results, layout);
  CHECK(layout[0].num_of_lines == 2);
  CHECK(layout[0].direction == turbo_ocr::layout::Direction::kHorizontal);
  CHECK(layout[0].text_line_height >  0);
  CHECK(layout[0].text_line_width >  0);
  // First line starts at x=50, last line ends at x=600.
  CHECK(layout[0].seg_start_coordinate == 50);
  CHECK(layout[0].seg_end_coordinate == 600);
}

TEST_CASE("cluster_text_lines: vertical-text cell gets vertical direction",
          "[cluster][text_lines][vertical]") {
  // 4 single-column tall narrow boxes — taller than wide, so each
  // box votes "vertical". Cluster sorts by descending x and groups
  // by x-projection overlap.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(0, 0, 200, 1000, /*class_id=*/22),
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(80,  50,  120, 250, /*layout_id=*/0),
      make_result(80, 270,  120, 470, /*layout_id=*/0),
      make_result(80, 490,  120, 690, /*layout_id=*/0),
      make_result(80, 710,  120, 910, /*layout_id=*/0),
  };
  turbo_ocr::layout::cluster_text_lines(results, layout);
  CHECK(layout[0].direction == turbo_ocr::layout::Direction::kVertical);
  CHECK(layout[0].num_of_lines == 1);  // all 4 spans share an x-band
}

TEST_CASE("infer_page_direction: majority vote over text cells",
          "[cluster][direction]") {
  using turbo_ocr::layout::Direction;
  // 3 text cells: 2 horizontal, 1 vertical → page is horizontal.
  std::vector<turbo_ocr::layout::LayoutBox> layout(3);
  for (auto &lb : layout) lb.class_id = 22;
  layout[0].direction = Direction::kHorizontal;
  layout[1].direction = Direction::kHorizontal;
  layout[2].direction = Direction::kVertical;
  CHECK(turbo_ocr::layout::infer_page_direction(layout) ==
        Direction::kHorizontal);

  // Flip: 1H 2V → vertical.
  layout[0].direction = Direction::kVertical;
  CHECK(turbo_ocr::layout::infer_page_direction(layout) ==
        Direction::kVertical);

  // Empty layout → horizontal default.
  std::vector<turbo_ocr::layout::LayoutBox> empty;
  CHECK(turbo_ocr::layout::infer_page_direction(empty) ==
        Direction::kHorizontal);
}

TEST_CASE("get_seg_flag: continuing paragraph signals seg_start_flag = false",
          "[seg_flag]") {
  using turbo_ocr::layout::Direction;
  using turbo_ocr::layout::get_seg_flag;
  // prev: multi-line block ending flush right (seg_end at x1).
  // current: starts flush left (seg_start at x0).
  turbo_ocr::layout::LayoutBox prev;
  prev.box = make_box(50, 100, 750, 300);
  prev.direction = Direction::kHorizontal;
  prev.num_of_lines = 4;
  prev.text_line_height = 25;
  prev.seg_start_coordinate = 50;
  prev.seg_end_coordinate = 745;  // close to x1=750

  turbo_ocr::layout::LayoutBox cur;
  cur.box = make_box(50, 320, 750, 500);
  cur.direction = Direction::kHorizontal;
  cur.num_of_lines = 3;
  cur.text_line_height = 25;
  cur.seg_start_coordinate = 52;  // close to x0=50
  cur.seg_end_coordinate = 600;

  auto sf = get_seg_flag(cur, prev, Direction::kHorizontal);
  CHECK(sf.seg_start_flag == false);  // continues prev's paragraph
}

TEST_CASE("get_seg_flag: clean break signals seg_start_flag = true",
          "[seg_flag]") {
  using turbo_ocr::layout::Direction;
  using turbo_ocr::layout::get_seg_flag;
  // prev: ends MID-LINE (seg_end is far from x1) — clean paragraph break.
  turbo_ocr::layout::LayoutBox prev;
  prev.box = make_box(50, 100, 750, 300);
  prev.direction = Direction::kHorizontal;
  prev.num_of_lines = 3;
  prev.text_line_height = 25;
  prev.seg_start_coordinate = 50;
  prev.seg_end_coordinate = 400;  // far from x1=750 → paragraph end

  turbo_ocr::layout::LayoutBox cur;
  cur.box = make_box(50, 320, 750, 500);
  cur.direction = Direction::kHorizontal;
  cur.num_of_lines = 3;
  cur.text_line_height = 25;
  cur.seg_start_coordinate = 50;
  cur.seg_end_coordinate = 600;

  auto sf = get_seg_flag(cur, prev, Direction::kHorizontal);
  CHECK(sf.seg_start_flag == true);
}

TEST_CASE("vertical reading order: right column emits before left column",
          "[xy_cut][vertical]") {
  // Two columns of vertical text. PaddleX/CJK convention: rightmost
  // column reads first. Build text-class cells with vertical
  // direction signal so infer_page_direction picks vertical.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(  50, 50, 250, 800, /*class_id=*/22),  // LEFT col (idx 0)
      make_layout( 350, 50, 550, 800, /*class_id=*/22),  // RIGHT col (idx 1)
  };
  // Synthesise vertical-text result boxes inside each column so
  // cluster_text_lines votes vertical. Each box: width 50, height 200
  // → height > width → "vertical".
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result( 80,  80, 130, 280, /*layout_id=*/0),  // left col span 1
      make_result( 80, 300, 130, 500, /*layout_id=*/0),  // left col span 2
      make_result(380,  80, 430, 280, /*layout_id=*/1),  // right col span 1
      make_result(380, 300, 430, 500, /*layout_id=*/1),  // right col span 2
  };
  // assign_layout_ids first to set ids stably.
  turbo_ocr::assign_layout_ids(results, layout);
  auto order =
      turbo_ocr::layout::assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 4);
  // After cluster: both cells voted vertical, page direction = vertical.
  // Reading order: right column (layout idx 1) comes before left (idx 0).
  // Within each column: top-to-bottom (the 2 spans inside).
  // Result order should be: right col span 1, right col span 2, left col span 1, left col span 2.
  // Mapping result indices to expected positions in `order`:
  //   results[2] (right col, top)    → order[0]
  //   results[3] (right col, bottom) → order[1]
  //   results[0] (left col, top)     → order[2]
  //   results[1] (left col, bottom)  → order[3]
  CHECK(order[0] == 2);
  CHECK(order[1] == 3);
  CHECK(order[2] == 0);
  CHECK(order[3] == 1);
}

// =====  flatten_descendants — nested child trees  =====

TEST_CASE("flatten_descendants: linear A → B → C chain emits in depth order",
          "[child_blocks][nested]") {
  // Manually build a chain: layout[0] (A) parent of layout[1] (B);
  // layout[1] (B) parent of layout[2] (C). Expected order: B, C.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout( 50,  10, 750,  60),  // A
      make_layout(100, 100, 700, 200),  // B
      make_layout(200, 220, 600, 280),  // C
  };
  std::vector<turbo_ocr::layout::ChildLinks> links(3);
  links[0].child_indices = {1};
  links[1].child_indices = {2};

  auto desc = turbo_ocr::layout::flatten_descendants(0, links, layout);
  REQUIRE(desc.size() == 2);
  CHECK(desc[0] == 1);
  CHECK(desc[1] == 2);
}

TEST_CASE("flatten_descendants: branching A → [B, C], B → D",
          "[child_blocks][nested]") {
  // A has two children B and C; B has child D. Walk emits B before
  // its descendants, then C. Expected: B, D, C.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout( 50,  10, 750,  60),  // 0 = A
      make_layout(100, 100, 350, 200),  // 1 = B (top-left)
      make_layout(400, 100, 750, 200),  // 2 = C (top-right)
      make_layout(150, 220, 300, 260),  // 3 = D (under B)
  };
  std::vector<turbo_ocr::layout::ChildLinks> links(4);
  links[0].child_indices = {1, 2};
  links[1].child_indices = {3};

  auto desc = turbo_ocr::layout::flatten_descendants(0, links, layout);
  REQUIRE(desc.size() == 3);
  CHECK(desc[0] == 1);  // B
  CHECK(desc[1] == 3);  // D (B's child)
  CHECK(desc[2] == 2);  // C
}

TEST_CASE("flatten_descendants: cycle A ↔ B is broken by visited set",
          "[child_blocks][nested]") {
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(50, 10, 750, 60),
      make_layout(50, 80, 750, 130),
  };
  std::vector<turbo_ocr::layout::ChildLinks> links(2);
  links[0].child_indices = {1};
  links[1].child_indices = {0};  // cycle back to A

  auto desc = turbo_ocr::layout::flatten_descendants(0, links, layout);
  REQUIRE(desc.size() == 1);
  CHECK(desc[0] == 1);  // visit B; do not recurse back into A
}

TEST_CASE("flatten_descendants: self-loop A → A is silently skipped",
          "[child_blocks][nested]") {
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(50, 10, 750, 60),
      make_layout(50, 80, 750, 130),
  };
  std::vector<turbo_ocr::layout::ChildLinks> links(2);
  links[0].child_indices = {0, 1};  // includes self
  auto desc = turbo_ocr::layout::flatten_descendants(0, links, layout);
  REQUIRE(desc.size() == 1);
  CHECK(desc[0] == 1);  // self-reference dropped, sibling kept
}

TEST_CASE("flatten_descendants: out-of-bounds parent → empty",
          "[child_blocks][nested]") {
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(50, 10, 750, 60),
  };
  std::vector<turbo_ocr::layout::ChildLinks> links(1);
  CHECK(turbo_ocr::layout::flatten_descendants(-1, links, layout).empty());
  CHECK(turbo_ocr::layout::flatten_descendants(99, links, layout).empty());
}

TEST_CASE("flatten_descendants: out-of-bounds child indices skipped",
          "[child_blocks][nested]") {
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(50, 10, 750, 60),
      make_layout(50, 80, 750, 130),
  };
  std::vector<turbo_ocr::layout::ChildLinks> links(2);
  links[0].child_indices = {1, 99, -1};  // 99 / -1 are bogus
  auto desc = turbo_ocr::layout::flatten_descendants(0, links, layout);
  REQUIRE(desc.size() == 1);
  CHECK(desc[0] == 1);
}

TEST_CASE("flatten_descendants: empty children → empty",
          "[child_blocks][nested]") {
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(50, 10, 750, 60),
  };
  std::vector<turbo_ocr::layout::ChildLinks> links(1);
  auto desc = turbo_ocr::layout::flatten_descendants(0, links, layout);
  CHECK(desc.empty());
}

TEST_CASE("flatten_descendants: deep chain hits depth limit, doesn't infinite-loop",
          "[child_blocks][nested]") {
  // Pathological: every node points to itself + the next, forming
  // a self-loop tree. The visited set short-circuits each self-loop
  // and the depth limit guards against any cycle the visited set
  // somehow misses.
  const int N = 50;
  std::vector<turbo_ocr::layout::LayoutBox> layout;
  layout.reserve(N);
  for (int i = 0; i < N; ++i) {
    layout.push_back(make_layout(i * 10, 100 + i, i * 10 + 50, 130 + i));
  }
  std::vector<turbo_ocr::layout::ChildLinks> links(N);
  for (int i = 0; i < N - 1; ++i) {
    links[i].child_indices = {i, i + 1};  // self + next
  }
  auto desc = turbo_ocr::layout::flatten_descendants(0, links, layout);
  // Visits 1..N-1 (N-1 entries). 0 was the parent, doesn't appear.
  CHECK(desc.size() == static_cast<size_t>(N - 1));
}

TEST_CASE("assign_reading_order_for_results: nested children A→B→C emit "
          "in depth order with manually-built links via splice_child_blocks",
          "[child_blocks][nested]") {
  // Build a UnsortedBlock sequence for splice_child_blocks where
  // layout[0] has child layout[1], and layout[1] has child layout[2].
  // After splice: only layout[0] remains as a top-level entry, with
  // layout[1] then layout[2] inserted after it.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(  0,   0, 100,  20),  // 0 root
      make_layout(  0,  30, 100,  50),  // 1
      make_layout(  0,  60, 100,  80),  // 2
  };
  std::vector<turbo_ocr::layout::ChildLinks> links(3);
  links[0].child_indices = {1};
  links[1].child_indices = {2};

  std::vector<turbo_ocr::layout::UnsortedBlock> sorted = {
      {0, {{0,  0, 100, 20}}, turbo_ocr::layout::OrderLabel::kBody, 22},
      {1, {{0, 30, 100, 50}}, turbo_ocr::layout::OrderLabel::kBody, 22},
      {2, {{0, 60, 100, 80}}, turbo_ocr::layout::OrderLabel::kBody, 22},
  };
  turbo_ocr::layout::splice_child_blocks(sorted, links, layout);
  REQUIRE(sorted.size() == 3);
  CHECK(sorted[0].layout_idx == 0);
  CHECK(sorted[1].layout_idx == 1);  // child of 0
  CHECK(sorted[2].layout_idx == 2);  // grandchild via 1
}

// =====  ?as_blocks=1 — paragraph-level aggregate  =====

TEST_CASE("results_with_blocks: short lines (mid-cell end) join with newline",
          "[blocks]") {
  // Two layout cells with two text lines each. Lines END WELL SHORT
  // of the cell's right margin → smart-join detects "paragraph end"
  // and emits '\n' between them. (A line that DID reach the right
  // margin would be a wrap and get joined with ' '; covered by the
  // dedicated "long lines" test below.)
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      // Cell right margin at x=350; lines end at x=200 → 150px short.
      make_layout(50,  50, 350, 200, /*class_id=*/22),
      make_layout(400, 50, 700, 200, /*class_id=*/22),
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(50,  60,  200,  90, /*layout_id=*/0),
      make_result(50, 110,  200, 140, /*layout_id=*/0),
      make_result(400, 60,  550,  90, /*layout_id=*/1),
      make_result(400,110,  550, 140, /*layout_id=*/1),
  };
  results[0].text = "left top";
  results[1].text = "left bottom";
  results[2].text = "right top";
  results[3].text = "right bottom";
  layout[0].text_line_height = 35;
  layout[1].text_line_height = 35;
  std::vector<int> reading_order = {0, 1, 2, 3};

  auto json = turbo_ocr::results_with_blocks(results, layout, reading_order);
  REQUIRE(json.find("\"blocks\":[") != std::string::npos);
  CHECK(json.find("\"content\":\"left top\\nleft bottom\"") != std::string::npos);
  CHECK(json.find("\"content\":\"right top\\nright bottom\"") != std::string::npos);
}

TEST_CASE("results_with_blocks: long lines (right-margin) join with space",
          "[blocks]") {
  // Lines that EXTEND to within text_line_height of the cell's right
  // margin are paragraph wraps — smart-join emits ' ' instead of '\n'
  // so the multi-line paragraph reads as one flowing string.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(50, 50, 350, 200, /*class_id=*/22),  // x1=350
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      // Both lines end close enough to right margin (340 vs 350) that
      // smart-join treats them as a paragraph wrap.
      make_result(50,  60, 340,  90, /*layout_id=*/0),
      make_result(50, 110, 340, 140, /*layout_id=*/0),
  };
  results[0].text = "the quick brown fox jumps over";
  results[1].text = "the lazy dog";
  layout[0].text_line_height = 35;
  std::vector<int> reading_order = {0, 1};

  auto json = turbo_ocr::results_with_blocks(results, layout, reading_order);
  CHECK(json.find("\"content\":\"the quick brown fox jumps over the lazy dog\"")
        != std::string::npos);
}

TEST_CASE("results_with_blocks: same-line texts join with space",
          "[blocks]") {
  // Two text spans on the same y-band → joined with ' ', not '\n'.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(50, 50, 700, 100, /*class_id=*/22),
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result( 50, 60, 200, 80, /*layout_id=*/0),
      make_result(220, 60, 400, 80, /*layout_id=*/0),
      make_result(420, 60, 690, 80, /*layout_id=*/0),
  };
  results[0].text = "alpha";
  results[1].text = "beta";
  results[2].text = "gamma";
  layout[0].text_line_height = 25;
  std::vector<int> reading_order = {0, 1, 2};

  auto json = turbo_ocr::results_with_blocks(results, layout, reading_order);
  CHECK(json.find("\"content\":\"alpha beta gamma\"") != std::string::npos);
}

TEST_CASE("results_with_blocks: omits blocks key when no layout/reading_order",
          "[blocks]") {
  std::vector<turbo_ocr::layout::LayoutBox> layout;
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(10, 20, 30, 40, /*layout_id=*/-1),
  };
  results[0].text = "x";
  std::vector<int> reading_order;
  auto json = turbo_ocr::results_with_blocks(results, layout, reading_order);
  CHECK(json.find("\"blocks\"") == std::string::npos);
}

TEST_CASE("results_with_blocks: escapes JSON-special chars in content",
          "[blocks]") {
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(0, 0, 200, 100, /*class_id=*/22),
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(10, 20, 190, 60, /*layout_id=*/0),
  };
  results[0].text = "with \"quotes\" and \\backslash";
  layout[0].text_line_height = 25;
  std::vector<int> reading_order = {0};
  auto json = turbo_ocr::results_with_blocks(results, layout, reading_order);
  CHECK(json.find("\"content\":\"with \\\"quotes\\\" and \\\\backslash\"")
        != std::string::npos);
}

TEST_CASE("recursive_xy_cut is lossless on a large input", "[xy_cut]") {
  // Stress the recursion over many boxes and assert the core invariant: the
  // output is a permutation of the input (every index exactly once, nothing
  // lost or duplicated). This does NOT reach the depth cap — well-formed
  // splits keep the recursion shallow — so the cap remains purely defensive
  // against a pathological input that can't arise here; it is not claimed to
  // be exercised by this test.
  const int N = 500;
  std::vector<std::array<int, 4>> rects;
  rects.reserve(N);
  for (int i = 0; i < N; ++i) {
    const int y = i * 100;  // well-separated rows
    rects.push_back({0, y, 50, y + 10});
  }
  std::vector<int> indices(N);
  for (int i = 0; i < N; ++i) indices[i] = i;

  std::vector<int> order;
  recursive_xy_cut(rects, indices, order);

  REQUIRE(order.size() == static_cast<size_t>(N));
  std::vector<int> sorted = order;
  std::sort(sorted.begin(), sorted.end());
  for (int i = 0; i < N; ++i) CHECK(sorted[i] == i);
}
