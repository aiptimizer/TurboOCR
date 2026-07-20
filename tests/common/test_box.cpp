#include <catch_amalgamated.hpp>

#include "turbo_ocr/common/geometry/box.h"

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

  // B and A are on the same line band (y/10 == 10), B.x < A.x
  CHECK(boxes[0] == b);
  CHECK(boxes[1] == a);
  CHECK(boxes[2] == c);
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
