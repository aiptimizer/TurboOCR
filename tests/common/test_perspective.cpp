#include <catch_amalgamated.hpp>

#include "turbo_ocr/common/geometry/perspective.h"

using turbo_ocr::Box;
using turbo_ocr::compute_crop_transform;
using turbo_ocr::compute_perspective_inv;

TEST_CASE("compute_crop_transform horizontal box", "[perspective]") {
  // Horizontal box 100x30
  Box box{{{{{0, 0}}, {{100, 0}}, {{100, 30}}, {{0, 30}}}}};
  auto ct = compute_crop_transform(box, 48, 4000);

  CHECK_FALSE(ct.vertical);
  // AR = 100/30 = 3.33, crop_width = ceil(48 * 3.33) = 160
  CHECK(ct.crop_width == 160);
  // M_inv should be non-zero (valid transform)
  bool any_nonzero = false;
  for (int i = 0; i < 9; ++i) {
    if (ct.M_inv[i] != 0.0f)
      any_nonzero = true;
  }
  CHECK(any_nonzero);
}

TEST_CASE("compute_crop_transform vertical box", "[perspective]") {
  // Vertical box: width=20, height=100 (h >= w * 1.5)
  Box box{{{{{0, 0}}, {{20, 0}}, {{20, 100}}, {{0, 100}}}}};
  auto ct = compute_crop_transform(box, 48, 4000);

  CHECK(ct.vertical);
  // After rotation, effective w=100, h=20, AR=5.0
  // crop_width = ceil(48 * 5.0) = 240
  CHECK(ct.crop_width == 240);
}

TEST_CASE("compute_crop_transform clamps to max width", "[perspective]") {
  // Very wide box: width=10000, height=10
  Box box{{{{{0, 0}}, {{10000, 0}}, {{10000, 10}}, {{0, 10}}}}};
  auto ct = compute_crop_transform(box, 48, 320);

  CHECK(ct.crop_width <= 320);
}

TEST_CASE("compute_crop_transform minimum width is 1", "[perspective]") {
  // Degenerate box: near-zero width
  Box box{{{{{0, 0}}, {{0, 0}}, {{0, 100}}, {{0, 100}}}}};
  auto ct = compute_crop_transform(box, 48, 4000);

  CHECK(ct.crop_width >= 1);
}

TEST_CASE("compute_perspective_inv identity-like transform", "[perspective]") {
  // Map unit square to itself
  float dst[8] = {0, 0, 1, 0, 1, 1, 0, 1};
  float src[8] = {0, 0, 1, 0, 1, 1, 0, 1};
  float M[9];
  compute_perspective_inv(dst, src, M);

  // Should be close to identity: M = [[1,0,0],[0,1,0],[0,0,1]]
  CHECK(M[0] == Catch::Approx(1.0f).margin(1e-4f));
  CHECK(M[1] == Catch::Approx(0.0f).margin(1e-4f));
  CHECK(M[2] == Catch::Approx(0.0f).margin(1e-4f));
  CHECK(M[3] == Catch::Approx(0.0f).margin(1e-4f));
  CHECK(M[4] == Catch::Approx(1.0f).margin(1e-4f));
  CHECK(M[5] == Catch::Approx(0.0f).margin(1e-4f));
  CHECK(M[6] == Catch::Approx(0.0f).margin(1e-4f));
  CHECK(M[7] == Catch::Approx(0.0f).margin(1e-4f));
  CHECK(M[8] == Catch::Approx(1.0f).margin(1e-4f));
}

TEST_CASE("compute_perspective_inv scaled transform", "[perspective]") {
  // Map (0,0)-(100,0)-(100,50)-(0,50) to (0,0)-(200,0)-(200,100)-(0,100)
  // This is a 2x scale
  float dst[8] = {0, 0, 100, 0, 100, 50, 0, 50};
  float src[8] = {0, 0, 200, 0, 200, 100, 0, 100};
  float M[9];
  compute_perspective_inv(dst, src, M);

  // Applying M to dst point (50, 25) should yield src point (100, 50)
  float x = 50, y = 25;
  float denom = M[6] * x + M[7] * y + M[8];
  float sx = (M[0] * x + M[1] * y + M[2]) / denom;
  float sy = (M[3] * x + M[4] * y + M[5]) / denom;
  CHECK(sx == Catch::Approx(100.0f).margin(0.1f));
  CHECK(sy == Catch::Approx(50.0f).margin(0.1f));
}
