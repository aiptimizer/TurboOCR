#include <catch_amalgamated.hpp>

#include "turbo_ocr/recognition/rec_geometry.h"

using namespace turbo_ocr::recognition;
using turbo_ocr::Box;

namespace {

Box axis_aligned_box(int x, int y, int w, int h) {
  Box b{};
  b[0] = {x, y};
  b[1] = {x + w, y};
  b[2] = {x + w, y + h};
  b[3] = {x, y + h};
  return b;
}

} // namespace

TEST_CASE("box_aspect uses edge lengths, not the AABB", "[rec_geometry]") {
  CHECK(box_aspect(axis_aligned_box(0, 0, 100, 50)) == Catch::Approx(2.0f));
  CHECK(box_aspect(axis_aligned_box(10, 20, 48, 48)) == Catch::Approx(1.0f));

  // 3-4-5 rotated rectangle: width edge (30,40) has length 50, height edge
  // (-8,6) has length 10 -> aspect 5.
  Box rot{};
  rot[0] = {0, 0};
  rot[1] = {30, 40};
  rot[2] = {22, 46};
  rot[3] = {-8, 6};
  CHECK(box_aspect(rot) == Catch::Approx(5.0f));
}

TEST_CASE("box_aspect degenerate height yields zero", "[rec_geometry]") {
  CHECK(box_aspect(axis_aligned_box(0, 0, 100, 0)) == 0.0f);
}

TEST_CASE("natural_rec_width clamps to [floor, kMaxRecWidth]", "[rec_geometry]") {
  // aspect 2 at h=48 -> 96
  CHECK(natural_rec_width(2.0f, 48, 32) == 96);
  // below the floor -> floor
  CHECK(natural_rec_width(0.1f, 48, 32) == 32);
  CHECK(natural_rec_width(0.1f, 48, 320) == 320);
  // beyond the ceiling -> kMaxRecWidth
  CHECK(natural_rec_width(1000.0f, 48, 32) == kMaxRecWidth);
  // exactly at the ceiling
  CHECK(natural_rec_width(kMaxRecWidth / 48.0f, 48, 32) == kMaxRecWidth);
}

TEST_CASE("snap_width_bucket picks the smallest covering bucket", "[rec_geometry]") {
  CHECK(snap_width_bucket(1) == 320);
  CHECK(snap_width_bucket(320) == 320);
  CHECK(snap_width_bucket(321) == 480);
  CHECK(snap_width_bucket(480) == 480);
  CHECK(snap_width_bucket(2001) == 2500);
  CHECK(snap_width_bucket(kMaxRecWidth) == kMaxRecWidth);
  // Larger-than-cap inputs must still land in the last bucket (never past
  // the table end).
  CHECK(snap_width_bucket(kMaxRecWidth + 1000) == kMaxRecWidth);
}

TEST_CASE("every clamped width has a bucket", "[rec_geometry]") {
  // The invariant that makes the lower_bound in snap_width_bucket safe: no
  // natural_rec_width result can exceed the last bucket.
  for (int w : {32, 100, 319, 320, 321, 799, 800, 3999, 4000}) {
    const int bucket = snap_width_bucket(w);
    CHECK(bucket >= w);
    CHECK(bucket <= kMaxRecWidth);
  }
}

TEST_CASE("snap_width_step pads at most step-1 and never shrinks", "[rec_geometry]") {
  CHECK(snap_width_step(100, 16) == 112);
  CHECK(snap_width_step(112, 16) == 112);
  CHECK(snap_width_step(100, 1) == 100);   // step 1: identity
  CHECK(snap_width_step(100, 0) == 100);   // disabled: identity
  // Clamped to the ceiling but never below the crop's own content width.
  CHECK(snap_width_step(kMaxRecWidth, 64) == kMaxRecWidth);
  CHECK(snap_width_step(kMaxRecWidth - 1, 64) == kMaxRecWidth);
  for (int w = 1; w < 200; ++w) {
    const int b = snap_width_step(w, 16);
    CHECK(b >= w);
    CHECK(b - w < 16);
  }
}
