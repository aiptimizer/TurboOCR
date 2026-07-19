#include <catch_amalgamated.hpp>

#include <cstdint>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/decode/fast_png_decoder.h"

using turbo_ocr::decode::FastPngDecoder;

namespace {

// Big-endian u32 write at a byte offset (PNG IHDR layout).
void put_be32(std::vector<uint8_t> &buf, std::size_t off, uint32_t v) {
  buf[off + 0] = static_cast<uint8_t>(v >> 24);
  buf[off + 1] = static_cast<uint8_t>(v >> 16);
  buf[off + 2] = static_cast<uint8_t>(v >> 8);
  buf[off + 3] = static_cast<uint8_t>(v);
}

// CRC-32 (PNG polynomial). The IHDR chunk CRC covers the type + 13 data
// bytes (offsets 12..24); recomputing it after forging the dimensions makes
// the chunk VALID again, so the only thing left to reject the file is the
// dimension guard under test — a stale CRC would let any decoder bail for
// the wrong reason and mask a deleted guard.
uint32_t crc32_png(const uint8_t *data, std::size_t len) {
  static uint32_t table[256];
  static bool init = [] {
    for (uint32_t n = 0; n < 256; ++n) {
      uint32_t c = n;
      for (int k = 0; k < 8; ++k)
        c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[n] = c;
    }
    return true;
  }();
  (void)init;
  uint32_t c = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < len; ++i)
    c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

// Forge the IHDR dimensions and re-seal the chunk with a valid CRC.
void forge_ihdr_dims(std::vector<uint8_t> &png, uint32_t w, uint32_t h) {
  put_be32(png, 16, w);
  put_be32(png, 20, h);
  put_be32(png, 29, crc32_png(png.data() + 12, 17));
}

std::vector<uint8_t> encode_png(const cv::Mat &img) {
  std::vector<uint8_t> bytes;
  REQUIRE(cv::imencode(".png", img, bytes));
  return bytes;
}

} // namespace

TEST_CASE("8-bit PNG decodes through the fast path", "[png_guards]") {
  cv::Mat img(20, 30, CV_8UC3, cv::Scalar(10, 20, 30));
  auto bytes = encode_png(img);
  REQUIRE(FastPngDecoder::is_png(bytes.data(), bytes.size()));
  cv::Mat out = FastPngDecoder::decode(bytes.data(), bytes.size());
  REQUIRE_FALSE(out.empty());
  CHECK(out.rows == 20);
  CHECK(out.cols == 30);
  CHECK(out.type() == CV_8UC3);
}

TEST_CASE("16-bit PNG decodes via the OpenCV fallback", "[png_guards]") {
  cv::Mat img(12, 17, CV_16UC1, cv::Scalar(40000));
  auto bytes = encode_png(img);
  // IHDR bit-depth byte must actually say 16 for this to test the fallback.
  REQUIRE(bytes.size() > 25);
  REQUIRE(bytes[24] == 16);
  cv::Mat out = FastPngDecoder::decode(bytes.data(), bytes.size());
  REQUIRE_FALSE(out.empty());
  CHECK(out.rows == 12);
  CHECK(out.cols == 17);
  CHECK(out.type() == CV_8UC3); // downconverted like the fast path emits
}

TEST_CASE("oversized 8-bit PNG dimensions are rejected before allocation",
          "[png_guards]") {
  cv::Mat img(8, 8, CV_8UC3, cv::Scalar(0, 0, 0));
  auto bytes = encode_png(img);
  // Forge the IHDR to declare an absurd raster, CRC re-sealed so ONLY the
  // dimension guard can reject it — no multi-GB buffer may ever be requested.
  forge_ihdr_dims(bytes, 0x40000000u, 0x40000000u);
  CHECK(FastPngDecoder::decode(bytes.data(), bytes.size()).empty());
}

TEST_CASE("oversized 16-bit PNG dimensions are rejected before OpenCV sees them",
          "[png_guards]") {
  cv::Mat img(8, 8, CV_16UC1, cv::Scalar(1));
  auto bytes = encode_png(img);
  REQUIRE(bytes[24] == 16);
  forge_ihdr_dims(bytes, 0x40000000u, 0x40000000u);
  CHECK(FastPngDecoder::decode(bytes.data(), bytes.size()).empty());
}

TEST_CASE("16-bit PNG with a huge single side is rejected", "[png_guards]") {
  cv::Mat img(8, 8, CV_16UC1, cv::Scalar(1));
  auto bytes = encode_png(img);
  forge_ihdr_dims(bytes, 1u << 20, 8); // width far over MAX_IMAGE_DIM
  CHECK(FastPngDecoder::decode(bytes.data(), bytes.size()).empty());
}

TEST_CASE("zero-dimension PNG is rejected", "[png_guards]") {
  cv::Mat img(8, 8, CV_16UC1, cv::Scalar(1));
  auto bytes = encode_png(img);
  forge_ihdr_dims(bytes, 0, 8);
  CHECK(FastPngDecoder::decode(bytes.data(), bytes.size()).empty());
}
