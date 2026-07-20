#include <catch_amalgamated.hpp>

#include "turbo_ocr/common/encoding.h"

using turbo_ocr::base64_decode;

TEST_CASE("base64_decode valid input", "[encoding]") {
  auto result = base64_decode(std::string_view{"SGVsbG8sIFdvcmxkIQ=="});
  CHECK(result == "Hello, World!");
}

TEST_CASE("base64_decode no padding", "[encoding]") {
  auto result = base64_decode(std::string_view{"YWJj"});
  CHECK(result == "abc");
}

TEST_CASE("base64_decode single char padding", "[encoding]") {
  auto result = base64_decode(std::string_view{"YWI="});
  CHECK(result == "ab");
}

TEST_CASE("base64_decode double padding", "[encoding]") {
  auto result = base64_decode(std::string_view{"YQ=="});
  CHECK(result == "a");
}

TEST_CASE("base64_decode empty input", "[encoding]") {
  auto result = base64_decode(std::string_view{""});
  CHECK(result.empty());
}

TEST_CASE("base64_decode with newlines stripped", "[encoding]") {
  auto result = base64_decode(std::string_view{"YWJj\n\r"});
  CHECK(result == "abc");
}

TEST_CASE("base64_decode string_view overload", "[encoding]") {
  std::string_view sv = "SGVsbG8=";
  auto result = base64_decode(sv);
  CHECK(result == "Hello");
}

TEST_CASE("base64_decode string overload", "[encoding]") {
  std::string s = "SGVsbG8=";
  auto result = base64_decode(s);
  CHECK(result == "Hello");
}

TEST_CASE("base64_decode binary data", "[encoding]") {
  auto result = base64_decode(std::string_view{"AAEC/w=="});
  REQUIRE(result.size() == 4);
  CHECK(static_cast<unsigned char>(result[0]) == 0x00);
  CHECK(static_cast<unsigned char>(result[1]) == 0x01);
  CHECK(static_cast<unsigned char>(result[2]) == 0x02);
  CHECK(static_cast<unsigned char>(result[3]) == 0xFF);
}
