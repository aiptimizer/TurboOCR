#include <catch_amalgamated.hpp>

#include <string>

#include "turbo_ocr/decode/jpeg_codec.h"

using turbo_ocr::decode::classify_nvjpeg_status;
using turbo_ocr::decode::JpegDecodeStatus;
using turbo_ocr::decode::looks_like_jpeg;
using turbo_ocr::decode::to_string;
namespace ns = turbo_ocr::decode::nvjpeg_status;

TEST_CASE("only bitstream problems count as unsupported; every other nvJPEG status is a fault", "[jpeg_codec]") {
  CHECK(classify_nvjpeg_status(ns::kSuccess) == JpegDecodeStatus::Ok);
  CHECK(classify_nvjpeg_status(ns::kBadJpeg) == JpegDecodeStatus::Unsupported);
  CHECK(classify_nvjpeg_status(ns::kJpegNotSupported) == JpegDecodeStatus::Unsupported);
  CHECK(classify_nvjpeg_status(ns::kIncompleteBitstream) == JpegDecodeStatus::Unsupported);
  // NOT_INITIALIZED, INVALID_PARAMETER, ALLOCATOR_FAILURE, EXECUTION_FAILED,
  // ARCH_MISMATCH, INTERNAL_ERROR, IMPLEMENTATION_NOT_SUPPORTED: device/runtime.
  for (int st : {1, 2, 5, 6, 7, 8, 9, 42, -1})
    CHECK(classify_nvjpeg_status(st) == JpegDecodeStatus::Failed);
  static_assert(classify_nvjpeg_status(5) == JpegDecodeStatus::Failed, "constexpr");
}

TEST_CASE("statuses have stable names for logs", "[jpeg_codec]") {
  CHECK(std::string(to_string(JpegDecodeStatus::Ok)) == "ok");
  CHECK(std::string(to_string(JpegDecodeStatus::Unsupported)) == "unsupported");
  CHECK(std::string(to_string(JpegDecodeStatus::Failed)) == "failed");
}

TEST_CASE("the JPEG sniff is the SOI marker and nothing else", "[jpeg_codec]") {
  const unsigned char soi[] = {0xFF, 0xD8, 0xFF, 0xE0};
  const unsigned char png[] = {0x89, 'P', 'N', 'G'};
  const unsigned char one[] = {0xFF};
  CHECK(looks_like_jpeg(soi, sizeof soi));
  CHECK_FALSE(looks_like_jpeg(png, sizeof png));
  CHECK_FALSE(looks_like_jpeg(one, sizeof one));
  CHECK_FALSE(looks_like_jpeg(nullptr, 2));
  CHECK_FALSE(looks_like_jpeg(soi, 0));
}
