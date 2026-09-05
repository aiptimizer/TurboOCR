#include <catch_amalgamated.hpp>

#include <string>

#include "turbo_ocr/decode/json_image_field.h"

using turbo_ocr::decode::find_json_image_field;

TEST_CASE("the plain /ocr body is scanned without a JSON document", "[json_image_field]") {
  auto r = find_json_image_field(R"({"image": "AAAA/+=="})");
  REQUIRE(r.has_value());
  CHECK(r->base64 == "AAAA/+==");
  CHECK_FALSE(r->has_routing);

  auto r2 = find_json_image_field("  \n{\"image\":\"QUJD\" , \"layout\": true, \"n\": 12.5, \"tags\": [\"a\", {\"b\": [1,2]}] }\n");
  REQUIRE(r2.has_value());
  CHECK(r2->base64 == "QUJD");
  CHECK_FALSE(r2->has_routing);

  auto r3 = find_json_image_field(R"({"routing": {"table": "slanext"}, "image": "QUJD"})");
  REQUIRE(r3.has_value());
  CHECK(r3->base64 == "QUJD");
  CHECK(r3->has_routing);
}

TEST_CASE("anything the scanner is not sure about is refused, never misread", "[json_image_field]") {
  // escapes inside the image string (\/ is legal base64 '/' escaped)
  CHECK_FALSE(find_json_image_field(R"({"image": "AA\/BB"})").has_value());
  // empty image, missing image, non-string image
  CHECK_FALSE(find_json_image_field(R"({"image": ""})").has_value());
  CHECK_FALSE(find_json_image_field(R"({"layout": true})").has_value());
  CHECK_FALSE(find_json_image_field(R"({"image": 42})").has_value());
  // duplicate key, malformed syntax, trailing garbage, not an object
  CHECK_FALSE(find_json_image_field(R"({"image": "QQ==", "image": "QQ=="})").has_value());
  CHECK_FALSE(find_json_image_field(R"({"image": "QQ==")").has_value());
  CHECK_FALSE(find_json_image_field(R"({"image": "QQ=="} x)").has_value());
  CHECK_FALSE(find_json_image_field(R"(["image", "QQ=="])").has_value());
  CHECK_FALSE(find_json_image_field("").has_value());
  // an escaped key is left to the parser
  CHECK_FALSE(find_json_image_field(R"({"im\u0061ge": "QQ=="})").has_value());
}

TEST_CASE("large payloads are handled without copying", "[json_image_field]") {
  std::string big(8 << 20, 'A');
  std::string body = "{\"image\":\"" + big + "\",\"layout\":false}";
  auto r = find_json_image_field(body);
  REQUIRE(r.has_value());
  CHECK(r->base64.size() == big.size());
  CHECK(r->base64.data() == body.data() + 10);  // a view into the body, not a copy
}
