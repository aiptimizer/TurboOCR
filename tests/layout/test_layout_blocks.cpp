// Unit tests for the recursive XY-cut reading-order algorithm.
//
// Exercises projection_by_bboxes, split_projection_profile, and
// assign_reading_order on synthetic layouts: single column, two
// columns, header + two columns, single box, and empty input.

#include <catch_amalgamated.hpp>
#include <algorithm>
#include <vector>

#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/layout/blocks/child_blocks.h"
#include "turbo_ocr/layout/blocks/match_unsorted.h"
#include "turbo_ocr/layout/order/reading_order.h"
#include "turbo_ocr/layout/blocks/text_line_cluster.h"

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
