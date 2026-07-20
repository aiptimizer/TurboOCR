#include <catch_amalgamated.hpp>

#include "turbo_ocr/recognition/ctc_decode.h"

using turbo_ocr::recognition::ctc_greedy_decode;
using turbo_ocr::recognition::ctc_greedy_decode_raw;

// Helper: build a label list with blank at index 0
static std::vector<std::string> make_labels() {
  // index 0 = blank, 1 = "a", 2 = "b", 3 = "c", 4 = " "
  return {"blank", "a", "b", "c", " "};
}

TEST_CASE("ctc_greedy_decode normal sequence", "[ctc]") {
  auto labels = make_labels();
  // Sequence: blank, a, a, blank, b, c  -> "abc" (repeated a collapsed)
  int indices[] = {0, 1, 1, 0, 2, 3};
  float scores[] = {0.9f, 0.8f, 0.7f, 0.9f, 0.85f, 0.95f};
  auto [text, score] = ctc_greedy_decode(indices, scores, 6, labels);
  CHECK(text == "abc");
  // Score = (0.8 + 0.85 + 0.95) / 3
  CHECK(score == Catch::Approx((0.8f + 0.85f + 0.95f) / 3.0f));
}

TEST_CASE("ctc_greedy_decode all blanks", "[ctc]") {
  auto labels = make_labels();
  int indices[] = {0, 0, 0, 0};
  float scores[] = {0.9f, 0.9f, 0.9f, 0.9f};
  auto [text, score] = ctc_greedy_decode(indices, scores, 4, labels);
  CHECK(text.empty());
  CHECK(score == 0.0f);
}

TEST_CASE("ctc_greedy_decode empty sequence", "[ctc]") {
  auto labels = make_labels();
  auto [text, score] = ctc_greedy_decode(nullptr, nullptr, 0, labels);
  CHECK(text.empty());
  CHECK(score == 0.0f);
}

TEST_CASE("ctc_greedy_decode repeated chars with blanks between", "[ctc]") {
  auto labels = make_labels();
  // a, blank, a -> "aa" (blank separator allows repeated char)
  int indices[] = {1, 0, 1};
  float scores[] = {0.9f, 0.8f, 0.7f};
  auto [text, score] = ctc_greedy_decode(indices, scores, 3, labels);
  CHECK(text == "aa");
  CHECK(score == Catch::Approx((0.9f + 0.7f) / 2.0f));
}

TEST_CASE("ctc_greedy_decode repeated chars without blanks", "[ctc]") {
  auto labels = make_labels();
  // a, a, a -> "a" (collapsed)
  int indices[] = {1, 1, 1};
  float scores[] = {0.9f, 0.8f, 0.7f};
  auto [text, score] = ctc_greedy_decode(indices, scores, 3, labels);
  CHECK(text == "a");
  // Only the first 'a' (index change from -1 to 1) counts
  CHECK(score == Catch::Approx(0.9f));
}

TEST_CASE("ctc_greedy_decode single character", "[ctc]") {
  auto labels = make_labels();
  int indices[] = {2};
  float scores[] = {0.75f};
  auto [text, score] = ctc_greedy_decode(indices, scores, 1, labels);
  CHECK(text == "b");
  CHECK(score == Catch::Approx(0.75f));
}

TEST_CASE("ctc_greedy_decode_raw normal sequence", "[ctc]") {
  auto labels = make_labels();
  // 5 classes, 3 timesteps
  // timestep 0: class 1 (a) wins
  // timestep 1: class 0 (blank) wins
  // timestep 2: class 2 (b) wins
  float logits[] = {
      // t=0: blank=0.1, a=0.9, b=0.0, c=0.0, space=0.0
      0.1f, 0.9f, 0.0f, 0.0f, 0.0f,
      // t=1: blank=0.8, ...
      0.8f, 0.1f, 0.0f, 0.0f, 0.1f,
      // t=2: b=0.95
      0.0f, 0.0f, 0.95f, 0.0f, 0.05f,
  };
  auto [text, score] = ctc_greedy_decode_raw(logits, 3, 5, labels);
  CHECK(text == "ab");
  CHECK(score == Catch::Approx((0.9f + 0.95f) / 2.0f));
}
