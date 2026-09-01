// decode_picodet_rows — THE shared PP-DocLayoutV3 row decoder
// (layout/picodet_decode.h). Until recently this loop existed as three private
// copies (shared header / CPU / TRT) and the fail-loud non-finite guard lived
// in only one of them; these tests pin the now-single policy, guard included.

#include <catch_amalgamated.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "turbo_ocr/analysis/layout/ort_paddle_layout.h"
#include "turbo_ocr/analysis/layout/picodet_decode.h"

using turbo_ocr::layout::decode_picodet_rows;
using turbo_ocr::layout::kPicodetMaxDet;

namespace {

// One [class_id, score, x0, y0, x1, y1, read_order] row.
void push_row(std::vector<float> &rows, float cls, float score, float x0,
              float y0, float x1, float y1, float order = 0.0f) {
  rows.insert(rows.end(), {cls, score, x0, y0, x1, y1, order});
}

} // namespace

TEST_CASE("picodet decode: basic row, clamping, read_order", "[layout][picodet]") {
  std::vector<float> rows;
  push_row(rows, 2, 0.9f, 10, 20, 110, 220, 3);
  // coordinates beyond the page are clamped into it
  push_row(rows, 1, 0.8f, -5, -5, 5000, 5000, 1);

  const auto out = decode_picodet_rows(rows.data(), 2, 7, nullptr, 0.5f, 1000, 800);
  REQUIRE(out.size() == 2);
  CHECK(out[0].class_id == 2);
  CHECK(out[0].read_order == 3);
  CHECK(out[0].box[0][0] == 10);
  CHECK(out[1].box[0][0] == 0);        // clamped
  CHECK(out[1].box[2][0] == 800 - 1);  // clamped to orig_w-1
}

TEST_CASE("picodet decode: non-finite rows fail loud, never silently blank",
          "[layout][picodet]") {
  // The guard exists because a broken execution provider (CoreML EP on ORT
  // 1.24.4) returned NaN for every score: NaN < threshold is false-y in a way
  // that silently drops all rows, which is indistinguishable from a blank
  // page. The decoder must make that case visible rather than return a
  // plausible-looking empty layout.
  //
  // The rule is PER ROW: a non-finite row is dropped and COUNTED, finite rows
  // survive. Dropping only the bad rows is deliberate — one corrupt row should
  // not discard an otherwise good page, and it matches how the detection
  // postprocess treats malformed output.
  //
  // The two cases are logged DIFFERENTLY, which is behaviour this test pins by
  // proxy (the return value) rather than by capturing the log:
  //   * ALL rows non-finite -> ERROR + empty result. This is the whole-page
  //     failure that motivated the guard.
  //   * SOME rows non-finite -> DEBUG, page kept. The export is a fixed-
  //     capacity candidate buffer, and an execution provider may write only
  //     the detections it found and leave the rest uninitialized; reporting
  //     that at ERROR cried wolf on every page.
  const float nan_v = std::numeric_limits<float>::quiet_NaN();

  // All rows garbage (the real CoreML failure) -> empty.
  std::vector<float> all_bad;
  push_row(all_bad, 2, nan_v, 10, 20, 110, 220);
  push_row(all_bad, 1, std::numeric_limits<float>::infinity(), 10, 20, 110, 220);
  CHECK(decode_picodet_rows(all_bad.data(), 2, 7, nullptr, 0.5f, 1000, 800).empty());

  // A non-finite COORDINATE is as fatal to a row as a non-finite score:
  // static_cast<int> of NaN is UB, so the row must never reach the clamp.
  std::vector<float> bad_coord;
  push_row(bad_coord, 2, 0.9f, nan_v, 20, 110, 220);
  CHECK(decode_picodet_rows(bad_coord.data(), 1, 7, nullptr, 0.5f, 1000, 800).empty());

  // Mixed: the finite row survives, the garbage one is dropped (not the page).
  std::vector<float> mixed;
  push_row(mixed, 2, nan_v, 10, 20, 110, 220);
  push_row(mixed, 1, 0.9f, 10, 20, 110, 220);
  const auto out = decode_picodet_rows(mixed.data(), 2, 7, nullptr, 0.5f, 1000, 800);
  REQUIRE(out.size() == 1);
  CHECK(out[0].class_id == 1);
}

TEST_CASE("picodet decode: count tensor overrides rows_dim0", "[layout][picodet]") {
  // The count output is authoritative: shape[0] is data-dependent and can go
  // stale. The CALLER owns making the buffer big enough for *count (the AMD
  // path copies the full NMS budget for exactly this reason).
  std::vector<float> rows;
  push_row(rows, 2, 0.9f, 10, 20, 110, 220);
  push_row(rows, 3, 0.9f, 10, 20, 110, 220);
  push_row(rows, 4, 0.9f, 10, 20, 110, 220);

  const std::int32_t count = 2; // fewer than rows_dim0: only 2 decoded
  const auto out = decode_picodet_rows(rows.data(), 3, 7, &count, 0.5f, 1000, 800);
  CHECK(out.size() == 2);

  const std::int32_t over = kPicodetMaxDet + 50; // runaway count: clamped to budget
  std::vector<float> big(static_cast<std::size_t>(kPicodetMaxDet) * 7, 0.0f);
  for (int i = 0; i < kPicodetMaxDet; ++i) {
    big[static_cast<std::size_t>(i) * 7 + 0] = 2;
    big[static_cast<std::size_t>(i) * 7 + 1] = 0.9f;
    big[static_cast<std::size_t>(i) * 7 + 2] = 0;
    big[static_cast<std::size_t>(i) * 7 + 3] = 0;
    big[static_cast<std::size_t>(i) * 7 + 4] = 10;
    big[static_cast<std::size_t>(i) * 7 + 5] = 10;
  }
  CHECK(decode_picodet_rows(big.data(), kPicodetMaxDet, 7, &over, 0.5f, 1000, 800)
            .size() == static_cast<std::size_t>(kPicodetMaxDet));
}

TEST_CASE("picodet decode: malformed rows are dropped", "[layout][picodet]") {
  std::vector<float> rows;
  push_row(rows, 999, 0.9f, 10, 20, 110, 220); // class id out of label range
  push_row(rows, -1, 0.9f, 10, 20, 110, 220);  // negative class id
  push_row(rows, 2, 0.9f, 110, 220, 10, 20);   // inverted box (x1<=x0 after clamp)
  push_row(rows, 2, 0.1f, 10, 20, 110, 220);   // below threshold
  push_row(rows, 2, 0.9f, 10, 20, 110, 220);   // the one good row
  const auto out = decode_picodet_rows(rows.data(), 5, 7, nullptr, 0.5f, 1000, 800);
  REQUIRE(out.size() == 1);
  CHECK(out[0].class_id == 2);
}

TEST_CASE("picodet decode: degenerate inputs return empty", "[layout][picodet]") {
  std::vector<float> rows;
  push_row(rows, 2, 0.9f, 10, 20, 110, 220);
  CHECK(decode_picodet_rows(nullptr, 1, 7, nullptr, 0.5f, 1000, 800).empty());
  CHECK(decode_picodet_rows(rows.data(), 0, 7, nullptr, 0.5f, 1000, 800).empty());
  CHECK(decode_picodet_rows(rows.data(), 1, 5, nullptr, 0.5f, 1000, 800).empty()); // stride < 6
  CHECK(decode_picodet_rows(rows.data(), 1, 7, nullptr, 0.5f, 0, 800).empty());
}

TEST_CASE("layout CoreML latch: one-way, so a replica pool never mixes providers",
          "[layout][coreml]") {
  // Replicas each build their OWN layout session (python pipeline.py builds N
  // Pipelines under one construct_lock hold). If a transient CoreML failure hit
  // replica 3 of 4, a per-session fallback would leave a pool that answers the
  // same page differently depending on which replica served it — while
  // src/service/server/unified/backend_stages.cpp asserts the opposite
  // invariant ("All entries load identically, so the last wins").
  //
  // The latch makes the pool homogeneous by construction: once ANY layout
  // session drops CoreML, every later load in the process skips it. It is
  // deliberately ONE-WAY — silently re-acquiring the accelerator mid-pool is
  // the very inhomogeneity it exists to prevent.
  using turbo_ocr::layout::coreml_layout_wedged;
  using turbo_ocr::layout::set_coreml_layout_wedged;

#ifdef __APPLE__
  const bool before = coreml_layout_wedged();
  set_coreml_layout_wedged();
  CHECK(coreml_layout_wedged());
  // Idempotent: latching twice is not an error and does not unset.
  set_coreml_layout_wedged();
  CHECK(coreml_layout_wedged());
  (void)before;  // no un-latch API on purpose; the state is process-scoped
#else
  // Off Apple there is no CoreML to drop, so the latch is inert and the
  // accelerated path is never taken.
  set_coreml_layout_wedged();
  CHECK_FALSE(coreml_layout_wedged());
#endif
}
