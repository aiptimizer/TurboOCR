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
  // Forge the IHDR to declare an absurd raster. The decode must refuse from
  // the header alone (CRC/stream errors would also produce empty, but the
  // point is that no multi-GB buffer is ever requested).
  put_be32(bytes, 16, 0x40000000u); // width
  put_be32(bytes, 20, 0x40000000u); // height
  CHECK(FastPngDecoder::decode(bytes.data(), bytes.size()).empty());
}

TEST_CASE("oversized 16-bit PNG dimensions are rejected before OpenCV sees them",
          "[png_guards]") {
  cv::Mat img(8, 8, CV_16UC1, cv::Scalar(1));
  auto bytes = encode_png(img);
  REQUIRE(bytes[24] == 16);
  put_be32(bytes, 16, 0x40000000u);
  put_be32(bytes, 20, 0x40000000u);
  CHECK(FastPngDecoder::decode(bytes.data(), bytes.size()).empty());
}

TEST_CASE("16-bit PNG with a huge single side is rejected", "[png_guards]") {
  cv::Mat img(8, 8, CV_16UC1, cv::Scalar(1));
  auto bytes = encode_png(img);
  put_be32(bytes, 16, 1u << 20); // width far over MAX_IMAGE_DIM
  CHECK(FastPngDecoder::decode(bytes.data(), bytes.size()).empty());
}

TEST_CASE("zero-dimension PNG is rejected", "[png_guards]") {
  cv::Mat img(8, 8, CV_16UC1, cv::Scalar(1));
  auto bytes = encode_png(img);
  put_be32(bytes, 16, 0);
  CHECK(FastPngDecoder::decode(bytes.data(), bytes.size()).empty());
}
