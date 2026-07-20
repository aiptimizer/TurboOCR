#include <catch_amalgamated.hpp>

#include <string>
#include <utility>
#include <vector>

#include "../../src/pipeline/ocr/ocr_pipeline_detail.h"

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
