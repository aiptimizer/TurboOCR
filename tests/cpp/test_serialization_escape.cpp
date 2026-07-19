#include <catch_amalgamated.hpp>

#include <cmath>
#include <string>

#include "turbo_ocr/common/serialization.h"

using turbo_ocr::detail::append_escaped_string;
using turbo_ocr::detail::append_score;

TEST_CASE("append_score never emits nan/inf (invalid JSON)", "[serialization]") {
  std::string j;
  append_score(j, std::nan(""));
  CHECK(j == "0");
  j.clear();
  append_score(j, std::numeric_limits<double>::infinity());
  CHECK(j == "0");
  j.clear();
  append_score(j, -std::numeric_limits<double>::infinity());
  CHECK(j == "0");
  j.clear();
  append_score(j, 0.98564);
  CHECK(j == "0.98564");
}

TEST_CASE("append_escaped_string escapes JSON control/special chars", "[serialization]") {
  std::string j;
  append_escaped_string(j, "a\"b\\c\nd\te");
  CHECK(j == "a\\\"b\\\\c\\nd\\te");
}

TEST_CASE("append_escaped_string replaces malformed UTF-8 with U+FFFD", "[serialization]") {
  std::string j;
  // Lone 0xFF is not valid UTF-8 → must not ship raw (RFC-8259 invalid).
  append_escaped_string(j, std::string("x\xFF""y"));
  CHECK(j.find('\xFF') == std::string::npos);
  CHECK(j.find("\xEF\xBF\xBD") != std::string::npos); // U+FFFD
  CHECK(j.front() == 'x');
  CHECK(j.back() == 'y');
}

TEST_CASE("append_escaped_string passes valid multibyte UTF-8 verbatim", "[serialization]") {
  std::string j;
  const std::string s = "\xE2\x9C\x93 \xC3\xA9"; // ✓ é
  append_escaped_string(j, s);
  CHECK(j == s);
}
