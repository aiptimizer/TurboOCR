#include <catch_amalgamated.hpp>

#include "turbo_ocr/common/serialization.h"

using turbo_ocr::OCRResultItem;
using turbo_ocr::Box;
using turbo_ocr::results_to_json;

TEST_CASE("results_to_json empty results", "[serialization]") {
  std::vector<OCRResultItem> results;
  auto json = results_to_json(results);
  CHECK(json == R"({"results":[]})");
}

TEST_CASE("results_to_json single result", "[serialization]") {
  Box box{{{{{10, 20}}, {{30, 20}}, {{30, 40}}, {{10, 40}}}}};
  std::vector<OCRResultItem> results = {
      {.text = "Hello", .confidence = 0.95f, .box = box}};
  auto json = results_to_json(results);

  // Check structure -- contains the text and bounding box
  CHECK(json.find("\"text\":\"Hello\"") != std::string::npos);
  CHECK(json.find("\"bounding_box\":[[10,20],[30,20],[30,40],[10,40]]") !=
        std::string::npos);
  // Starts and ends correctly
  CHECK(json.substr(0, 13) == "{\"results\":[{");
  CHECK(json.back() == '}');
}

TEST_CASE("results_to_json escapes special characters", "[serialization]") {
  Box box{};
  std::vector<OCRResultItem> results = {
      {.text = "He said \"hello\" \\ world", .confidence = 0.9f, .box = box}};
  auto json = results_to_json(results);

  CHECK(json.find(R"(He said \"hello\" \\ world)") != std::string::npos);
}

TEST_CASE("results_to_json escapes control characters", "[serialization]") {
  Box box{};
  std::string text_with_controls = "line1\nline2\ttab\rreturn";
  std::vector<OCRResultItem> results = {
      {.text = text_with_controls, .confidence = 0.8f, .box = box}};
  auto json = results_to_json(results);

  CHECK(json.find(R"(line1\nline2\ttab\rreturn)") != std::string::npos);
}

TEST_CASE("results_to_json escapes low control chars as unicode", "[serialization]") {
  Box box{};
  // \x01 should be escaped as \u0001
  std::string text = "a";
  text += '\x01';
  text += "b";
  std::vector<OCRResultItem> results = {
      {.text = text, .confidence = 0.8f, .box = box}};
  auto json = results_to_json(results);

  CHECK(json.find(R"(a\u0001b)") != std::string::npos);
}

TEST_CASE("results_to_json preserves raw UTF-8 multi-byte text", "[serialization]") {
  // Regression: sign-extension bug previously escaped UTF-8 continuation
  // bytes (0x80+) as \u00xx via the control-char branch, producing mojibake
  // on the wire. RFC 8259 permits raw UTF-8 in JSON strings; only control
  // chars < 0x20 need escaping.
  Box box{};
  std::string text = "\xe4\xbd\xa0\xe5\xa5\xbd"; // "你好" (nǐ hǎo)
  std::vector<OCRResultItem> results = {
      {.text = text, .confidence = 0.9f, .box = box}};
  auto json = results_to_json(results);

  // Raw UTF-8 bytes must appear unmodified in the output.
  CHECK(json.find(text) != std::string::npos);
  // And no \u00XX escape sequence for any of those bytes.
  CHECK(json.find("\\u00") == std::string::npos);
}

TEST_CASE("results_to_json multiple results separated by commas", "[serialization]") {
  Box box{};
  std::vector<OCRResultItem> results = {
      {.text = "A", .confidence = 0.9f, .box = box},
      {.text = "B", .confidence = 0.8f, .box = box},
      {.text = "C", .confidence = 0.7f, .box = box},
  };
  auto json = results_to_json(results);

  // Count commas between result objects (should be 2)
  int comma_count = 0;
  bool in_results = false;
  for (size_t i = 0; i < json.size(); ++i) {
    if (json[i] == '[' && i > 0 && json[i - 1] == ':')
      in_results = true;
    if (in_results && json[i] == '}' && i + 1 < json.size() && json[i + 1] == ',')
      comma_count++;
  }
  CHECK(comma_count == 2);
}
