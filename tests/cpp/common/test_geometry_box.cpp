#include <catch_amalgamated.hpp>

#include "turbo_ocr/base/geometry/box.h"

using turbo_ocr::Box;
using turbo_ocr::aabb;
using turbo_ocr::is_vertical_box;
using turbo_ocr::sorted_boxes;
using turbo_ocr::kVerticalAspectRatio;

namespace {
// pts order is [tl, tr, br, bl].
Box make_box(int x0, int y0, int x1, int y1, int x2, int y2, int x3, int y3) {
  return Box{{{{x0, y0}, {x1, y1}, {x2, y2}, {x3, y3}}}};
}
Box axis_box(int x0, int y0, int x1, int y1) {
  return make_box(x0, y0, x1, y0, x1, y1, x0, y1);
}
} // namespace

TEST_CASE("aabb on an axis-aligned box: corner0/corner2 IS the diagonal", "[geometry][aabb]") {
  // Sanity baseline before the rotated case below.
  Box b = axis_box(10, 20, 110, 70);
  CHECK(aabb(b) == std::array<int, 4>{10, 20, 110, 70});
}

TEST_CASE("aabb on a rotated quad: corner0/corner2 is NOT the diagonal",
          "[geometry][aabb]") {
  // Diamond inscribed in [10,10]-[90,90]: tl=top, tr=right, br=bottom, bl=left.
  // corner[0]=(50,10) and corner[2]=(50,90) share the same x, so a naive
  // "aabb = min/max of corner0 and corner2" would collapse the box to zero
  // width (x0==x1==50) and silently drop the true 10..90 x-extent that comes
  // from corner[1]/corner[3]. This is exactly the case box.h's comment warns
  // about; aabb() must fold in all 4 corners to get it right.
  Box diamond = make_box(50, 10, 90, 50, 50, 90, 10, 50);
  CHECK(aabb(diamond) == std::array<int, 4>{10, 10, 90, 90});
}

TEST_CASE("aabb on a slanted (non-90-degree) quad", "[geometry][aabb]") {
  // A sheared parallelogram: still corner0/corner2 not axis-aligned with the
  // true extent. tl=(50,0) tr=(100,20) br=(80,70) bl=(30,50).
  Box b = make_box(50, 0, 100, 20, 80, 70, 30, 50);
  CHECK(aabb(b) == std::array<int, 4>{30, 0, 100, 70});
}

TEST_CASE("is_vertical_box boundary at kVerticalAspectRatio (1.5)", "[geometry][vertical]") {
  REQUIRE(kVerticalAspectRatio == 1.5f);

  // h == w * 1.5 exactly: the comparison is >=, so this must read as vertical.
  // A strict-> implementation would flip this one case.
  Box exact = axis_box(0, 0, 10, 15);
  CHECK(is_vertical_box(exact));

  // Just under the ratio (h=14 vs w=10, threshold is h=15): must NOT be vertical.
  Box just_under = axis_box(0, 0, 10, 14);
  CHECK_FALSE(is_vertical_box(just_under));

  // Just over: must be vertical.
  Box just_over = axis_box(0, 0, 10, 16);
  CHECK(is_vertical_box(just_over));
}

TEST_CASE("is_vertical_box takes the max width/height across both edge pairs",
          "[geometry][vertical]") {
  // Trapezoid: top edge width 10 (tr.x-tl.x), bottom edge width 5 (br.x-bl.x).
  // w must be max(10,5)=10, not the bottom edge or an average — otherwise a
  // slightly skewed box would misjudge orientation. Height fixed at 20, well
  // past 10*1.5=15, so the call is unambiguous regardless of which edge wins.
  Box trapezoid = make_box(0, 0, 10, 0, 5, 20, 0, 20);
  CHECK(is_vertical_box(trapezoid));

  // Same box but check it would read as NOT vertical if the narrower (5px)
  // edge were used as w with a height that only clears the narrow threshold:
  // h=8 is >= 5*1.5=7.5 (narrow-edge-wins) but < 10*1.5=15 (wide-edge-wins).
  // Correct behavior (wide edge wins) must report false here.
  Box narrow_h = make_box(0, 0, 10, 0, 5, 8, 0, 8);
  CHECK_FALSE(is_vertical_box(narrow_h));
}

TEST_CASE("is_vertical_box does not overflow on large coordinates",
          "[geometry][vertical]") {
  // w*w and h*h are computed in the several-hundred-thousand-squared range,
  // which overflows a 32-bit int (max ~2.1e9); the header explicitly casts to
  // int64_t to avoid this. w=100000 -> w*w=1e10, already past INT32_MAX.
  Box big = axis_box(0, 0, 100000, 160000); // ratio 1.6 > 1.5
  CHECK(is_vertical_box(big));
  Box big_not = axis_box(0, 0, 100000, 140000); // ratio 1.4 < 1.5
  CHECK_FALSE(is_vertical_box(big_not));
}

TEST_CASE("sorted_boxes is a no-op below 2 boxes", "[geometry][sort]") {
  std::vector<Box> empty;
  sorted_boxes(empty);
  CHECK(empty.empty());

  std::vector<Box> one{axis_box(5, 5, 15, 15)};
  sorted_boxes(one);
  REQUIRE(one.size() == 1);
  CHECK(one[0] == axis_box(5, 5, 15, 15));
}

TEST_CASE("sorted_boxes orders by x within a single line, tie-break on exact-equal tops",
          "[geometry][sort]") {
  // Two boxes with the identical top (a tie, not just "close") must still
  // land in left-to-right x order.
  std::vector<Box> boxes{
      axis_box(50, 0, 60, 10), // x=50
      axis_box(10, 0, 20, 10), // x=10
  };
  sorted_boxes(boxes);
  REQUIRE(boxes.size() == 2);
  CHECK(boxes[0][0][0] == 10);
  CHECK(boxes[1][0][0] == 50);
}

TEST_CASE("sorted_boxes groups tops within kSameLineThreshold as one line",
          "[geometry][sort]") {
  // Regression for the band-quantization bug the header calls out: tops of
  // 29 and 34 differ by 5px (well under the 10px threshold) and must be
  // treated as ONE line. The old `y / 10` quantization put 29 in band 2 and
  // 34 in band 3, so it wrongly split them into two lines. x order within
  // the (single) line must still be respected.
  std::vector<Box> boxes{
      axis_box(80, 29, 90, 39), // top=29, x=80
      axis_box(20, 34, 30, 44), // top=34, x=20
  };
  sorted_boxes(boxes);
  REQUIRE(boxes.size() == 2);
  // Same line -> sorted by x, not by the (incidental) top-order they came in.
  CHECK(boxes[0][0][0] == 20);
  CHECK(boxes[1][0][0] == 80);
}

TEST_CASE("sorted_boxes measures each new line's gap from the LINE'S top, not the previous box",
          "[geometry][sort]") {
  // The chaining bug the header warns about: 4 boxes at tops 0, 9, 18, 27,
  // each only 9px below the previous one (under the 10px threshold pairwise).
  // If grouping chained off the previous box, all 4 would collapse into one
  // arbitrarily tall "line". Anchoring the comparison to the current line's
  // own top instead splits them into two real lines: {0,9} and {18,27},
  // because 18-0=18>10 opens a new line even though 18-9=9<=10.
  std::vector<Box> boxes{
      axis_box(50, 0, 60, 10),  // line A, x=50
      axis_box(10, 9, 20, 19),  // line A, x=10
      axis_box(80, 18, 90, 28), // line B, x=80
      axis_box(20, 27, 30, 37), // line B, x=20
  };
  sorted_boxes(boxes);
  REQUIRE(boxes.size() == 4);
  // Line A (tops 0,9) sorted by x: x=10 then x=50.
  CHECK(boxes[0][0][0] == 10);
  CHECK(boxes[0][0][1] == 9);
  CHECK(boxes[1][0][0] == 50);
  CHECK(boxes[1][0][1] == 0);
  // Line B (tops 18,27) sorted by x: x=20 then x=80.
  CHECK(boxes[2][0][0] == 20);
  CHECK(boxes[2][0][1] == 27);
  CHECK(boxes[3][0][0] == 80);
  CHECK(boxes[3][0][1] == 18);
}

TEST_CASE("sorted_boxes separates lines further apart than the threshold",
          "[geometry][sort]") {
  std::vector<Box> boxes{
      axis_box(10, 100, 20, 110), // second line
      axis_box(10, 0, 20, 10),    // first line
  };
  sorted_boxes(boxes);
  REQUIRE(boxes.size() == 2);
  CHECK(boxes[0][0][1] == 0);
  CHECK(boxes[1][0][1] == 100);
}
