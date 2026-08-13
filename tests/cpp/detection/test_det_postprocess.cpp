#include <catch_amalgamated.hpp>
#include <cstdlib>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/analysis/detection/det_config.h"
#include "turbo_ocr/analysis/detection/det_postprocess.h"

using turbo_ocr::Box;
using turbo_ocr::detection::box_score_fast;
using turbo_ocr::detection::compute_det_resize;
using turbo_ocr::detection::effective_det_max_side;
using turbo_ocr::detection::get_mini_boxes;
using turbo_ocr::detection::kDetResizeDefault;
using turbo_ocr::detection::read_det_resize;
using turbo_ocr::detection::unclip;

TEST_CASE("get_mini_boxes returns ordered corners", "[det_postprocess]") {
  // A simple rectangle contour
  std::vector<cv::Point> contour = {{10, 10}, {50, 10}, {50, 30}, {10, 30}};
  float min_side = 0;
  Box box = get_mini_boxes(contour, min_side);

  // min_side should be the shorter dimension (height=20)
  CHECK(min_side == Catch::Approx(20.0f).margin(1.0f));

  // top-left should have smallest y among left pair, smallest x among top pair
  // Verify ordering: tl.y <= bl.y, tr.y <= br.y, tl.x <= tr.x
  CHECK(box[0][1] <= box[3][1]); // tl.y <= bl.y
  CHECK(box[1][1] <= box[2][1]); // tr.y <= br.y
  CHECK(box[0][0] <= box[1][0]); // tl.x <= tr.x
}

TEST_CASE("get_mini_boxes handles tilted contour", "[det_postprocess]") {
  // Slightly rotated rectangle
  std::vector<cv::Point> contour = {{15, 5}, {55, 10}, {53, 35}, {13, 30}};
  float min_side = 0;
  Box box = get_mini_boxes(contour, min_side);

  // Should still produce a valid 4-corner box
  CHECK(min_side > 0);
  // All corners should be close to the input contour bounding region
  for (int i = 0; i < 4; ++i) {
    CHECK(box[i][0] >= 0);
    CHECK(box[i][1] >= 0);
  }
}

TEST_CASE("unclip expands polygon", "[det_postprocess]") {
  std::vector<cv::Point> polygon = {{10, 10}, {50, 10}, {50, 30}, {10, 30}};
  float unclip_ratio = 1.5f;
  auto expanded = unclip(polygon, unclip_ratio);

  // Expanded polygon should have at least 3 points
  REQUIRE(expanded.size() >= 3);

  // The bounding rect of the expanded polygon should be larger
  cv::Rect orig_br = cv::boundingRect(polygon);
  cv::Rect exp_br = cv::boundingRect(expanded);
  CHECK(exp_br.width >= orig_br.width);
  CHECK(exp_br.height >= orig_br.height);
}

TEST_CASE("unclip with zero perimeter returns original", "[det_postprocess]") {
  // Degenerate polygon (single point repeated)
  std::vector<cv::Point> polygon = {{10, 10}, {10, 10}, {10, 10}};
  auto result = unclip(polygon, 1.5f);
  // Should return original (no crash)
  CHECK(result.size() == polygon.size());
}

TEST_CASE("box_score_fast computes mean within polygon", "[det_postprocess]") {
  // Create a small prediction map filled with 0.8
  cv::Mat pred_map(100, 100, CV_32F, cv::Scalar(0.8f));

  // A rectangle covering part of the image
  std::vector<cv::Point> contour = {{20, 20}, {60, 20}, {60, 50}, {20, 50}};

  std::vector<cv::Point> shifted_buf;
  cv::Mat mask_buf;
  float score = box_score_fast(pred_map, contour, shifted_buf, mask_buf);

  // Should be approximately 0.8 (uniform fill)
  CHECK(score == Catch::Approx(0.8f).margin(0.01f));
}

TEST_CASE("box_score_fast returns zero for out-of-bounds contour", "[det_postprocess]") {
  cv::Mat pred_map(50, 50, CV_32F, cv::Scalar(0.9f));

  // Contour outside image bounds (negative coords clamped to 0)
  // All points at origin => xmax <= xmin => returns 0
  std::vector<cv::Point> contour = {{0, 0}, {0, 0}, {0, 0}};

  std::vector<cv::Point> shifted_buf;
  cv::Mat mask_buf;
  float score = box_score_fast(pred_map, contour, shifted_buf, mask_buf);

  CHECK(score == Catch::Approx(0.0f).margin(0.01f));
}

// Regression: DET_MAX_SIDE must clamp BOTH the engine-profile/buffer size
// (effective_det_max_side) AND the runtime resize cap (read_det_resize) so the
// resize output can never exceed the allocated buffer. A DET_MAX_SIDE below the
// model's max_side_limit used to size buffers at the smaller value while the
// resize still emitted up to max_side_limit px -> device overrun.
TEST_CASE("DET_MAX_SIDE shrinks both the buffer and the resize cap together", "[det_config]") {
  ::unsetenv("DET_MAX_SIDE");
  ::unsetenv("DET_MAX_SIDE_LIMIT");
  ::unsetenv("DET_LIMIT_TYPE");
  ::unsetenv("DET_LIMIT_SIDE_LEN");

  SECTION("shrink below max_side_limit") {
    ::setenv("DET_MAX_SIDE", "640", 1);
    auto p = read_det_resize();
    const int buf = effective_det_max_side(p);
    CHECK(p.max_side_limit == 640);  // resize cap shrank to 640
    CHECK(buf == 640);               // buffer/profile sized to 640
    // A large input must resize to <= the buffer side on every axis.
    auto [rh, rw] = compute_det_resize(4000, 3000, p);
    CHECK(std::max(rh, rw) <= buf);
    ::unsetenv("DET_MAX_SIDE");
  }

  SECTION("enlarge above max_side_limit leaves the resize cap untouched") {
    ::setenv("DET_MAX_SIDE", "2048", 1);
    auto p = read_det_resize();
    const int buf = effective_det_max_side(p);
    CHECK(p.max_side_limit == kDetResizeDefault.max_side_limit);  // unchanged (1280)
    CHECK(buf == 2048);  // buffer grows; resize stays <= 1280 < buf
    auto [rh, rw] = compute_det_resize(4000, 3000, p);
    CHECK(std::max(rh, rw) <= buf);
    ::unsetenv("DET_MAX_SIDE");
  }
}

// Regression (GitHub #23): DET_LIMIT_TYPE=max alone used to inherit the
// min-policy default limit_side_len=64, which under max semantics means
// "shrink the LONGEST side to 64px" — every image became a thumbnail and OCR
// silently returned zero results.
TEST_CASE("DET_LIMIT_TYPE=max without DET_LIMIT_SIDE_LEN targets the max-side cap",
          "[det_config]") {
  ::unsetenv("DET_MAX_SIDE");
  ::unsetenv("DET_MAX_SIDE_LIMIT");
  ::unsetenv("DET_LIMIT_TYPE");
  ::unsetenv("DET_LIMIT_SIDE_LEN");

  SECTION("bare max policy keeps native resolution up to the cap") {
    ::setenv("DET_LIMIT_TYPE", "max", 1);
    auto p = read_det_resize();
    CHECK(p.limit_side_len == p.max_side_limit);
    auto [rh, rw] = compute_det_resize(1000, 800, p);
    CHECK(std::max(rh, rw) >= 960);  // near-native, NOT a 64px thumbnail
    ::unsetenv("DET_LIMIT_TYPE");
  }

  SECTION("issue #23 env combo: max policy + DET_MAX_SIDE_LIMIT=2560") {
    ::setenv("DET_LIMIT_TYPE", "max", 1);
    ::setenv("DET_MAX_SIDE_LIMIT", "2560", 1);
    auto p = read_det_resize();
    CHECK(p.limit_side_len == 2560);
    auto [rh, rw] = compute_det_resize(4000, 3000, p);
    CHECK(std::max(rh, rw) == 2560);  // capped, not thumbnailed
    ::unsetenv("DET_LIMIT_TYPE");
    ::unsetenv("DET_MAX_SIDE_LIMIT");
  }

  SECTION("explicit DET_LIMIT_SIDE_LEN under max policy is honored") {
    ::setenv("DET_LIMIT_TYPE", "max", 1);
    ::setenv("DET_LIMIT_SIDE_LEN", "960", 1);
    auto p = read_det_resize();
    CHECK(p.limit_side_len == 960);
    auto [rh, rw] = compute_det_resize(4000, 3000, p);
    CHECK(std::max(rh, rw) == 960);
    ::unsetenv("DET_LIMIT_TYPE");
    ::unsetenv("DET_LIMIT_SIDE_LEN");
  }

  SECTION("garbage/zero numeric envs clamp instead of thumbnailing to 0") {
    ::setenv("DET_MAX_SIDE_LIMIT", "0", 1);
    ::setenv("DET_LIMIT_SIDE_LEN", "junk", 1);
    auto p = read_det_resize();
    CHECK(p.max_side_limit >= 32);
    CHECK(p.limit_side_len >= 32);
    ::unsetenv("DET_MAX_SIDE_LIMIT");
    ::unsetenv("DET_LIMIT_SIDE_LEN");
  }
}
