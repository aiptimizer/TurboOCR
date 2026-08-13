#include <catch_amalgamated.hpp>

#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/pipeline/ocr_pipeline_detail.h"

using turbo_ocr::Box;
using turbo_ocr::pipeline::OcrPipelineResult;
using turbo_ocr::pipeline::detail::combine_recognition;
using turbo_ocr::pipeline::detail::flag_dropped_crops;
using turbo_ocr::pipeline::detail::flag_text_degraded;

namespace {

Box box_at(int x) {
  Box b{};
  b[0] = {x, 0};
  b[1] = {x + 10, 0};
  b[2] = {x + 10, 10};
  b[3] = {x, 10};
  return b;
}

} // namespace

TEST_CASE("combine_recognition keeps confident non-empty results in box order",
          "[pipeline_detail]") {
  std::vector<Box> boxes{box_at(0), box_at(20), box_at(40)};
  std::vector<std::pair<std::string, float>> rec{
      {"first", 0.9f}, {"second", 0.8f}, {"third", 0.7f}};
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec);
  REQUIRE(out.results.size() == 3);
  CHECK(out.results[0].text == "first");
  CHECK(out.results[1].text == "second");
  CHECK(out.results[2].text == "third");
  CHECK(out.results[1].box[0][0] == 20);
  CHECK_FALSE(out.text_degraded);
}

TEST_CASE("combine_recognition drops empty and low-confidence entries",
          "[pipeline_detail]") {
  std::vector<Box> boxes{box_at(0), box_at(20), box_at(40)};
  std::vector<std::pair<std::string, float>> rec{
      {"keep", 0.9f}, {"", 0.9f}, {"low", 0.1f}};
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec);
  REQUIRE(out.results.size() == 1);
  CHECK(out.results[0].text == "keep");
}

TEST_CASE("combine_recognition tolerates a short rec vector", "[pipeline_detail]") {
  std::vector<Box> boxes{box_at(0), box_at(20)};
  std::vector<std::pair<std::string, float>> rec{{"only", 0.9f}};
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec);
  REQUIRE(out.results.size() == 1);
  CHECK(out.results[0].text == "only");
}

TEST_CASE("all-empty recognition on detected boxes flags text_degraded",
          "[pipeline_detail]") {
  std::vector<Box> boxes{box_at(0), box_at(20)};
  std::vector<std::pair<std::string, float>> rec{{"", 0.0f}, {"", 0.0f}};
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec);
  CHECK(out.results.empty());
  CHECK(out.text_degraded);
  CHECK_FALSE(out.text_warning.empty());
}

TEST_CASE("zero detections is a clean page, not a degraded one",
          "[pipeline_detail]") {
  std::vector<Box> boxes;
  std::vector<std::pair<std::string, float>> rec;
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec);
  CHECK_FALSE(out.text_degraded);
}

TEST_CASE("flag_dropped_crops marks partial drops loud", "[pipeline_detail]") {
  OcrPipelineResult out;
  out.results.push_back({.text = "survivor", .confidence = 0.9f, .box = box_at(0)});
  flag_dropped_crops(out, 3);
  CHECK(out.text_degraded);
  CHECK(out.text_warning.find("3") != std::string::npos);
  // Zero drops must not touch the flags.
  OcrPipelineResult clean;
  flag_dropped_crops(clean, 0);
  CHECK_FALSE(clean.text_degraded);
}

TEST_CASE("flag_dropped_crops appends to an existing warning", "[pipeline_detail]") {
  OcrPipelineResult out;
  flag_text_degraded(out, 2); // no results + 2 boxes -> sets first warning
  flag_dropped_crops(out, 1);
  CHECK(out.text_degraded);
  CHECK(out.text_warning.find(';') != std::string::npos);
}

// ---------------------------------------------------------------------------
// Warning ACCUMULATION. These pin the fix for the defect that let the two
// orchestrations drift: every writer of text_warning must APPEND, never assign.
//
// The failure this prevents: a recognizer under-returns (so the correct
// diagnosis "N of M region(s) were not recognized" is computed), and the
// surviving results are then empty (so flag_text_degraded also fires). If
// flag_text_degraded ASSIGNS, it replaces a true message with a FALSE one —
// "all crops decoded empty/blank" is wrong, the recognizer never ran. The
// correct cause is computed and then thrown away.
// ---------------------------------------------------------------------------

TEST_CASE("flag_text_degraded appends, never overwrites an existing warning",
          "[pipeline_detail]") {
  OcrPipelineResult out;
  out.text_degraded = true;
  out.text_warning = "text stage degraded: 2 of 3 region(s) were not recognized";

  flag_text_degraded(out, 3); // results empty -> fires

  CHECK(out.text_degraded);
  // the prior, CORRECT diagnosis must survive
  CHECK(out.text_warning.find("were not recognized") != std::string::npos);
  // and the new one is added, not substituted
  CHECK(out.text_warning.find("all crops decoded empty/blank") != std::string::npos);
  CHECK(out.text_warning.find("; ") != std::string::npos);
}

TEST_CASE("under-return + empty results keeps BOTH causes, in order",
          "[pipeline_detail]") {
  // The end-to-end shape of the bug: combine_recognition detects the
  // under-return, then flag_text_degraded fires on the empty result set.
  std::vector<Box> boxes{box_at(0), box_at(20), box_at(40)};
  std::vector<std::pair<std::string, float>> rec{{"", 0.99f}}; // 1 of 3, and empty
  OcrPipelineResult out;

  combine_recognition(out, boxes, rec);

  REQUIRE(out.results.empty());
  CHECK(out.text_degraded);
  CHECK(out.text_warning.find("under-returned") != std::string::npos);
  CHECK(out.text_warning.find("all crops decoded empty/blank") != std::string::npos);
}

TEST_CASE("combine_recognition folds dropped-crop accounting in",
          "[pipeline_detail]") {
  // The 4th parameter exists so no call site can forget the drop accounting.
  // Defaulted to 0, so the CUDA pipeline's existing 3-arg calls are unchanged.
  std::vector<Box> boxes{box_at(0), box_at(20)};
  std::vector<std::pair<std::string, float>> rec{{"a", 0.9f}, {"b", 0.9f}};
  OcrPipelineResult out;

  combine_recognition(out, boxes, rec, /*dropped_crops=*/2);

  CHECK(out.results.size() == 2);       // the good results survive
  CHECK(out.text_degraded);             // but the page is NOT clean
  CHECK(out.text_warning.find("dropped 2 crop(s)") != std::string::npos);
}

TEST_CASE("a clean page stays clean and unwarned", "[pipeline_detail]") {
  std::vector<Box> boxes{box_at(0)};
  std::vector<std::pair<std::string, float>> rec{{"ok", 0.95f}};
  OcrPipelineResult out;

  combine_recognition(out, boxes, rec, /*dropped_crops=*/0);

  CHECK(out.results.size() == 1);
  CHECK_FALSE(out.text_degraded);
  CHECK(out.text_warning.empty());
}
