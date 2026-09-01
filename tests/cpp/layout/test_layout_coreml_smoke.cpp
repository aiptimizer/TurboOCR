// The accelerated layout path's ONLY automated gate.
//
// Every accuracy/golden/conformance ctest runs with DISABLE_COREML=1 (see
// _turbo_test_env in CMakeLists.txt — the CoreML EP is non-deterministic
// across macOS versions, which is fatal for exact-match gates). Deliberate and
// correct — but it means a CoreML EP regression on the layout stage is
// INVISIBLE to the whole suite. That is not hypothetical: on ORT 1.24.4 the
// CoreML EP returned NaN for every score and box, the decoder dropped every
// row, and an EMPTY layout shipped as a fast HTTP 200 for months of dev time.
//
// This test is that missing gate. Hidden tag ([.]) keeps it OUT of the plain
// `turbo_ocr_tests` run — it is registered as its own ctest entry
// (layout_coreml_smoke), Apple-only, with NO DISABLE_COREML in its
// environment, and must run in its OWN process: the latch test in
// test_picodet_decode.cpp wedges CoreML process-wide when it runs first.

#include <catch_amalgamated.hpp>

#ifdef __APPLE__

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/analysis/layout/ort_paddle_layout.h"

namespace {

// A document-looking page: a title bar and two paragraph columns of "text
// lines". PP-DocLayoutV3 reliably finds regions on this shape, which the
// test asserts — a model that finds nothing here is itself a regression.
cv::Mat synthetic_page() {
  cv::Mat img(800, 800, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::rectangle(img, {80, 60}, {720, 110}, cv::Scalar(20, 20, 20), cv::FILLED);
  for (int y = 180; y < 460; y += 34)
    cv::rectangle(img, {80, y}, {700, y + 16}, cv::Scalar(40, 40, 40),
                  cv::FILLED);
  for (int y = 520; y < 740; y += 34)
    cv::rectangle(img, {80, y}, {520, y + 16}, cv::Scalar(40, 40, 40),
                  cv::FILLED);
  return img;
}

// Box coords are ints (the decoder's finite-guard runs BEFORE the int cast,
// so a NaN coordinate can never reach a LayoutBox) — the float to check is
// the score, and the coords are checked against the page bounds the decoder
// clamps to.
bool sane(const std::vector<turbo_ocr::layout::LayoutBox> &boxes) {
  for (const auto &b : boxes) {
    if (!std::isfinite(b.score) || b.score <= 0.0f || b.score > 1.0f)
      return false;
    for (const auto &p : b.box.pts)
      if (p[0] < 0 || p[0] >= 800 || p[1] < 0 || p[1] >= 800) return false;
  }
  return true;
}

} // namespace

TEST_CASE("layout CoreML smoke: the accelerated path attaches and agrees "
          "with the CPU provider",
          "[.][coreml-smoke]") {
  // Its own ctest entry sets no DISABLE_COREML; clear it defensively in case
  // this hidden test is invoked by hand from an environment that exports it
  // (this test's whole purpose is the path the rest of the suite disables).
  unsetenv("DISABLE_COREML");

  if (turbo_ocr::layout::coreml_layout_wedged())
    FAIL("CoreML latch already set — this test must run in its own process "
         "(the picodet latch test wedges it)");

  const char *model = "models/layout/layout.onnx";
  if (FILE *f = std::fopen(model, "rb")) {
    std::fclose(f);
  } else {
    SKIP("models/layout/layout.onnx not present (run from the source dir "
         "with models provisioned)");
  }

  const cv::Mat page = synthetic_page();

  turbo_ocr::layout::OrtPaddleLayout cpu;
  REQUIRE(cpu.load_model(model));
  const auto cpu_boxes = cpu.run(page);
  // The synthetic page is DESIGNED to be detected; zero regions from the
  // reference provider means the fixture or the model broke, and every
  // assertion below would pass vacuously.
  REQUIRE(!cpu_boxes.empty());
  REQUIRE(sane(cpu_boxes));

  turbo_ocr::layout::OrtPaddleLayout coreml;
  coreml.set_use_coreml(true);
  REQUIRE(coreml.load_model(model));
  // On a Mac with a working CoreML the session must actually ATTACH — a
  // fallback here means the accelerated path is broken even though the
  // library would keep working (that graceful degradation is exactly why no
  // other test can catch this).
  REQUIRE_FALSE(coreml.coreml_dropped());
  REQUIRE_FALSE(turbo_ocr::layout::coreml_layout_wedged());

  const auto ml_boxes = coreml.run(page);
  REQUIRE(sane(ml_boxes));
  // The historical failure mode is EMPTY-while-CPU-finds-regions (every row
  // NaN'd away). Exact equality is not promised across providers (measured
  // 602 vs 601 regions over 83 real pages), so allow a one-region skew.
  REQUIRE(!ml_boxes.empty());
  const auto diff = static_cast<int>(ml_boxes.size()) -
                    static_cast<int>(cpu_boxes.size());
  REQUIRE(std::abs(diff) <= 1);
}

#else

TEST_CASE("layout CoreML smoke is Apple-only", "[.][coreml-smoke]") {
  SUCCEED("no CoreML off Apple; the CPU provider is the only layout path");
}

#endif
