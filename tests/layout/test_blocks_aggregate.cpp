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
