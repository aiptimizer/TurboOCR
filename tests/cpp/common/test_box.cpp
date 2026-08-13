#include <catch_amalgamated.hpp>

#include "turbo_ocr/base/geometry/box.h"

using turbo_ocr::Box;

TEST_CASE("Box default construction is zeroed", "[box]") {
  Box b{};
  for (int i = 0; i < 4; ++i) {
    CHECK(b[i][0] == 0);
    CHECK(b[i][1] == 0);
  }
}

TEST_CASE("Box equality and comparison", "[box]") {
  Box a{{{{{10, 20}}, {{30, 20}}, {{30, 40}}, {{10, 40}}}}};
  Box b = a;
  CHECK(a == b);

  Box c{{{{{10, 20}}, {{30, 20}}, {{30, 40}}, {{10, 41}}}}};
  CHECK(a != c);
  CHECK(a < c); // a[3][1]=40 < c[3][1]=41
}

TEST_CASE("sorted_boxes orders top-to-bottom, left-to-right", "[box]") {
  // Box A at y=100, x=200
  Box a{{{{{200, 100}}, {{300, 100}}, {{300, 130}}, {{200, 130}}}}};
  // Box B at y=100, x=50 (same line, left of A)
  Box b{{{{{50, 105}}, {{150, 105}}, {{150, 130}}, {{50, 130}}}}};
  // Box C at y=300, x=10 (lower line)
  Box c{{{{{10, 300}}, {{110, 300}}, {{110, 330}}, {{10, 330}}}}};

  std::vector<Box> boxes = {a, c, b};
  turbo_ocr::sorted_boxes(boxes);

  // A and B are within kSameLineThreshold of each other, so B.x < A.x decides
  CHECK(boxes[0] == b);
  CHECK(boxes[1] == a);
  CHECK(boxes[2] == c);
}

TEST_CASE("sorted_boxes same-line tolerance does not depend on absolute Y",
          "[box]") {
  // REGRESSION: this used to quantize y/10 into FIXED bands, so whether two
  // boxes counted as one line depended on where they sat relative to a
  // multiple of 10 rather than on the gap between them. Tops of 29 and 34 (5px
  // apart) landed in different bands while 30 and 34 (4px apart) did not —
  // real symptom: two words on one line came back in opposite orders on two
  // backends, purely from a 1px difference in the detected top edge.
  //
  // Every pair below is 5px apart and must therefore sort left-to-right,
  // wherever the band edge happens to fall.
  for (int top = 25; top <= 35; ++top) {
    INFO("right-hand box top = " << top);
    // right-hand word, higher up
    Box right{{{{{400, top}}, {{500, top}}, {{500, top + 26}}, {{400, top + 26}}}}};
    // left-hand word, 5px lower
    const int lt = top + 5;
    Box left{{{{{50, lt}}, {{150, lt}}, {{150, lt + 26}}, {{50, lt + 26}}}}};

    std::vector<Box> boxes = {right, left};
    turbo_ocr::sorted_boxes(boxes);
    CHECK(boxes[0] == left); // same line => leftmost first
    CHECK(boxes[1] == right);
  }
}

TEST_CASE("sorted_boxes does not chain a staircase into one line", "[box]") {
  // Each box sits 9px below the previous — within tolerance of its NEIGHBOUR,
  // but the last is 45px below the first. Measuring the gap from the current
  // line's own top (rather than from the previous box) stops the whole run from
  // collapsing into one arbitrarily tall "line" that would then sort purely by
  // X. x DECREASES as y increases, so a collapse would return them reversed.
  std::vector<Box> boxes;
  for (int i = 0; i < 6; ++i) {
    const int y = 100 + i * 9;
    const int x = 500 - i * 50;
    boxes.push_back(Box{{{{{x, y}}, {{x + 40, y}}, {{x + 40, y + 20}}, {{x, y + 20}}}}});
  }
  auto shuffled = boxes;
  std::swap(shuffled[0], shuffled[4]);
  std::swap(shuffled[1], shuffled[3]);
  turbo_ocr::sorted_boxes(shuffled);

  // THE INVARIANT: a box may only precede one that is higher up when the two
  // are within the same-line tolerance. Anything further out of Y order means
  // lines were merged that should not have been.
  //
  // Deliberately not asserting an exact permutation: with a 9px step and a 10px
  // tolerance, consecutive boxes ARE legitimately same-line, so they pair up —
  // that is the defined behaviour, not a defect, and pinning the exact sequence
  // would just re-encode the implementation.
  for (std::size_t i = 0; i + 1 < shuffled.size(); ++i) {
    INFO("index " << i);
    CHECK(shuffled[i][0][1] <= shuffled[i + 1][0][1] + 10);
  }
  // And the run must NOT have become a single line (which would fully reverse
  // it, putting the bottom-left box first).
  CHECK(shuffled.front()[0][1] < shuffled.back()[0][1]);
}

TEST_CASE("sorted_boxes is deterministic regardless of input order", "[box]") {
  // The comparator must be a strict weak ordering: a tolerance-based one is not
  // transitive (a~b, b~c, but a<c), which is UB in std::sort. Grouping into
  // lines first and then sorting by (line, x) keeps it well-defined — this
  // checks the observable consequence, that permuting the input cannot change
  // the output.
  std::vector<Box> boxes;
  for (int i = 0; i < 12; ++i) {
    const int y = 40 + (i % 4) * 7;   // clusters that straddle band edges
    const int x = 10 + ((i * 37) % 400);
    boxes.push_back(Box{{{{{x, y}}, {{x + 30, y}}, {{x + 30, y + 18}}, {{x, y + 18}}}}});
  }
  auto expected = boxes;
  turbo_ocr::sorted_boxes(expected);

  auto rotated = boxes;
  for (std::size_t r = 1; r < boxes.size(); ++r) {
    std::rotate(rotated.begin(), rotated.begin() + 1, rotated.end());
    auto got = rotated;
    turbo_ocr::sorted_boxes(got);
    INFO("rotation " << r);
    CHECK(got == expected);
  }
}

TEST_CASE("sorted_boxes empty vector", "[box]") {
  std::vector<Box> boxes;
  turbo_ocr::sorted_boxes(boxes);
  CHECK(boxes.empty());
}

TEST_CASE("sorted_boxes single element", "[box]") {
  Box a{{{{{10, 20}}, {{30, 20}}, {{30, 40}}, {{10, 40}}}}};
  std::vector<Box> boxes = {a};
  turbo_ocr::sorted_boxes(boxes);
  CHECK(boxes.size() == 1);
  CHECK(boxes[0] == a);
}

TEST_CASE("is_vertical_box detects vertical text", "[box]") {
  // Horizontal box: width=100, height=30
  Box horiz{{{{{0, 0}}, {{100, 0}}, {{100, 30}}, {{0, 30}}}}};
  CHECK_FALSE(turbo_ocr::is_vertical_box(horiz));

  // Vertical box: width=30, height=100 (h >= w * 1.5)
  Box vert{{{{{0, 0}}, {{30, 0}}, {{30, 100}}, {{0, 100}}}}};
  CHECK(turbo_ocr::is_vertical_box(vert));

  // Square box: width=100, height=100 (NOT vertical, h < w*1.5)
  Box square{{{{{0, 0}}, {{100, 0}}, {{100, 100}}, {{0, 100}}}}};
  CHECK_FALSE(turbo_ocr::is_vertical_box(square));
}

TEST_CASE("is_vertical_box edge case at boundary ratio", "[box]") {
  // width=20, height=30 -> ratio = 1.5 exactly
  // is_vertical_box uses h*h >= w*w*225/100 -> 900 >= 400*2.25 = 900 -> true
  Box boundary{{{{{0, 0}}, {{20, 0}}, {{20, 30}}, {{0, 30}}}}};
  CHECK(turbo_ocr::is_vertical_box(boundary));

  // width=20, height=29 -> 841 >= 900 -> false
  Box just_below{{{{{0, 0}}, {{20, 0}}, {{20, 29}}, {{0, 29}}}}};
  CHECK_FALSE(turbo_ocr::is_vertical_box(just_below));
}
