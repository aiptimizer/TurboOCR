// decode_cpu_fallback's contract: empty Mat on any undecodable input, NEVER a
// throw or an abort(). Both cases here are ones a fuzzer actually found against
// the network-reachable image path (tests/fuzz/fuzz_image_decode.cpp) and that
// crashed the process before the guard existed.
#include <catch_amalgamated.hpp>

#include <cstdint>
#include <vector>

#include "turbo_ocr/image/cpu_image_decode.h"

using turbo_ocr::decode::decode_cpu_fallback;
using turbo_ocr::decode::is_supported_image_magic;

TEST_CASE("decode: empty input returns empty, never throws", "[decode][fuzz]") {
  // cv::imdecode ASSERTS (throws cv::Exception) on a zero-length buffer; on the
  // gRPC path decode runs before the .empty() guard, so that threw out of the
  // handler. Must be an empty Mat now.
  REQUIRE(decode_cpu_fallback(nullptr, 0).empty());
  const uint8_t one = 0x00;
  REQUIRE(decode_cpu_fallback(&one, 0).empty());
}

TEST_CASE("decode: unsupported format never reaches an aborting codec",
          "[decode][fuzz]") {
  // A 142-byte blob whose content OpenCV routed into GDCM (DICOM), which hit
  // `assert(0 && "Should not happen")` and abort()ed — uncatchable by any
  // try/catch. The magic allowlist must reject it before imdecode.
  std::vector<uint8_t> dicomish(142, 0x00);
  dicomish[0] = 'D'; dicomish[1] = 'I'; dicomish[2] = 'C'; dicomish[3] = 'M';
  REQUIRE_FALSE(is_supported_image_magic(dicomish.data(), dicomish.size()));
  REQUIRE(decode_cpu_fallback(dicomish.data(), dicomish.size()).empty());

  // Random noise: no valid magic, must be rejected, must not throw.
  std::vector<uint8_t> noise = {0x1f, 0x8b, 0x08, 0x00, 0x42, 0x99, 0x00, 0x11};
  REQUIRE(decode_cpu_fallback(noise.data(), noise.size()).empty());
}

TEST_CASE("decode: the served formats pass the magic gate", "[decode]") {
  // The allowlist must not reject formats the server actually decodes. Magic
  // prefixes only — a full valid file is exercised by the round-trip suite.
  auto ok = [](std::vector<uint8_t> m) {
    return is_supported_image_magic(m.data(), m.size());
  };
  CHECK(ok({0xFF, 0xD8, 0xFF, 0xE0}));                       // JPEG
  CHECK(ok({'B', 'M', 0, 0}));                               // BMP
  CHECK(ok({'G', 'I', 'F', '8', '9', 'a'}));                 // GIF89a
  CHECK(ok({'I', 'I', 0x2A, 0x00}));                         // TIFF LE
  CHECK(ok({'M', 'M', 0x00, 0x2A}));                         // TIFF BE
  CHECK(ok({'R','I','F','F',0,0,0,0,'W','E','B','P'}));      // WebP
  // A too-short prefix of a real magic must not read out of bounds (ASan) and
  // must simply say "not yet a known format".
  CHECK_FALSE(ok({0xFF}));                                   // 1 byte of JPEG
  CHECK_FALSE(ok({'R', 'I', 'F', 'F'}));                     // RIFF, no WEBP tag
}
