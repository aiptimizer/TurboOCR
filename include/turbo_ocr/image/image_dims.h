#pragma once

#include <climits>
#include <cstdint>
#include <cstring>
#include <optional>

// Format-aware image-dimension sniffers.
//
// The /ocr/raw and /ocr endpoints accept PNG / JPEG / WEBP / BMP / etc. The
// pixel grid for any of those is bounded only by the encoded body cap until
// the decoder runs — and a ~1 KB PNG of a solid colour can claim 100 000 ×
// 100 000 dimensions, allocating 30 GB during decode. To refuse such a
// "decompression bomb" before the decoder allocates anything, we parse the
// width/height directly from the file header.
//
// This is the standard pattern used by every public image API (AWS Rekognition,
// Google Vision, Cloudinary, ImageMagick MAGICK_AREA_LIMIT). We sniff every
// container we decode whose header carries the pixel grid — PNG, JPEG, WebP,
// TIFF, GIF, and BMP — so the per-image and aggregate batch-pixel caps can
// bound a decompression bomb before the decoder allocates anything. BMP is
// uncompressed, but that does NOT make MAX_BODY_MB a bound: cv::imdecode sizes
// its output Mat from the BMP header's declared dimensions, not from the body
// length, so a tiny BMP header can still drive a multi-GB allocation. It must
// be sniffed like the rest. Formats whose header this cannot parse fall through
// to the route handler's post-decode check as a safety net.
namespace turbo_ocr::decode {

struct ImageDims {
  int width;
  int height;
};

// PNG: signature (8) + IHDR chunk header (length=13, type="IHDR") + data.
// Width is 4 big-endian bytes at offset 16, height at offset 20.
inline std::optional<ImageDims>
peek_png_dims(const unsigned char *data, size_t len) {
  if (len < 24) return std::nullopt;
  static constexpr unsigned char sig[8] = {0x89, 0x50, 0x4E, 0x47,
                                            0x0D, 0x0A, 0x1A, 0x0A};
  if (std::memcmp(data, sig, 8) != 0) return std::nullopt;
  if (std::memcmp(data + 12, "IHDR", 4) != 0) return std::nullopt;
  auto rd = [](const unsigned char *p) -> uint32_t {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8)  |  uint32_t(p[3]);
  };
  uint32_t w = rd(data + 16);
  uint32_t h = rd(data + 20);
  if (w == 0 || h == 0 || w > INT_MAX || h > INT_MAX) return std::nullopt;
  return ImageDims{static_cast<int>(w), static_cast<int>(h)};
}

// JPEG: SOI (0xFFD8) + segments. Walk segments until we hit the first SOFn
// marker (0xC0..0xCF except 0xC4/0xC8/0xCC). The frame header has the
// canonical width/height as 16-bit big-endian fields.
//
// Bounded scan: real JPEGs put SOFn within the first ~64 KB (after JFIF/Exif
// thumbnails). A malformed file with thousands of fill bytes or oversized
// segment lengths could otherwise drag this loop across an entire 100 MB
// upload. Cap at 64 KB or the full body, whichever is smaller.
inline std::optional<ImageDims>
peek_jpeg_dims(const unsigned char *data, size_t len) {
  if (len < 4 || data[0] != 0xFF || data[1] != 0xD8) return std::nullopt;
  constexpr size_t kJpegScanLimit = 64 * 1024;
  const size_t scan_end = (len < kJpegScanLimit) ? len : kJpegScanLimit;
  size_t i = 2;
  while (i + 9 < scan_end) {
    if (data[i] != 0xFF) return std::nullopt;
    // Skip fill bytes (some encoders emit multiple 0xFF before the marker).
    while (i < scan_end && data[i] == 0xFF) ++i;
    if (i >= scan_end) return std::nullopt;
    uint8_t marker = data[i++];
    // Standalone markers (no length): TEM (0x01), RSTm (0xD0–0xD7),
    // SOI (0xD8), EOI (0xD9). Skip them.
    if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD9)) continue;
    if (i + 1 >= scan_end) return std::nullopt;
    uint16_t seg_len = (uint16_t(data[i]) << 8) | data[i + 1];
    // SOFn (n in 0..15) carries dimensions, except DHT/JPG/DAC.
    if (marker >= 0xC0 && marker <= 0xCF &&
        marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
      if (i + 6 >= scan_end || seg_len < 7) return std::nullopt;
      // Layout after seg_len: precision(1), height(2), width(2), components(1)
      uint16_t h = (uint16_t(data[i + 3]) << 8) | data[i + 4];
      uint16_t w = (uint16_t(data[i + 5]) << 8) | data[i + 6];
      if (w == 0 || h == 0) return std::nullopt;
      return ImageDims{int(w), int(h)};
    }
    if (seg_len < 2) return std::nullopt;
    i += seg_len;
  }
  return std::nullopt;
}

// WebP: RIFF container with one of three sub-formats.
//   - VP8  (lossy):    width/height at bytes 26-29 as 14-bit LE values
//                      following the 0x9D 0x01 0x2A start code
//   - VP8L (lossless): a packed 32-bit LE word at byte 21 carrying
//                      (width-1, height-1) as 14-bit fields
//   - VP8X (extended): width-1 (24-bit LE) at 24, height-1 at 27
inline std::optional<ImageDims>
peek_webp_dims(const unsigned char *data, size_t len) {
  if (len < 30) return std::nullopt;
  if (std::memcmp(data, "RIFF", 4) != 0) return std::nullopt;
  if (std::memcmp(data + 8, "WEBP", 4) != 0) return std::nullopt;

  if (std::memcmp(data + 12, "VP8 ", 4) == 0) {
    // Look for the keyframe start code 0x9D 0x01 0x2A at bytes 23-25.
    if (data[23] != 0x9D || data[24] != 0x01 || data[25] != 0x2A)
      return std::nullopt;
    uint16_t w = (uint16_t(data[27]) << 8 | data[26]) & 0x3FFF;
    uint16_t h = (uint16_t(data[29]) << 8 | data[28]) & 0x3FFF;
    if (w == 0 || h == 0) return std::nullopt;
    return ImageDims{int(w), int(h)};
  }
  if (std::memcmp(data + 12, "VP8L", 4) == 0) {
    if (data[20] != 0x2F) return std::nullopt;
    uint32_t bits = uint32_t(data[21]) | (uint32_t(data[22]) << 8) |
                    (uint32_t(data[23]) << 16) | (uint32_t(data[24]) << 24);
    int w = static_cast<int>((bits & 0x3FFF) + 1);
    int h = static_cast<int>(((bits >> 14) & 0x3FFF) + 1);
    return ImageDims{w, h};
  }
  if (std::memcmp(data + 12, "VP8X", 4) == 0) {
    uint32_t w_minus = uint32_t(data[24]) | (uint32_t(data[25]) << 8) |
                       (uint32_t(data[26]) << 16);
    uint32_t h_minus = uint32_t(data[27]) | (uint32_t(data[28]) << 8) |
                       (uint32_t(data[29]) << 16);
    return ImageDims{int(w_minus + 1), int(h_minus + 1)};
  }
  return std::nullopt;
}

// GIF: "GIF87a" / "GIF89a" magic (6 bytes) + logical-screen descriptor.
// Width is 2 LE bytes at offset 6, height 2 LE bytes at offset 8. GIF's LZW
// stream is highly compressible, so a tiny body can claim a huge canvas —
// sniff it so the per-image and aggregate caps bound it like PNG/JPEG.
inline std::optional<ImageDims>
peek_gif_dims(const unsigned char *data, size_t len) {
  if (len < 10) return std::nullopt;
  if (std::memcmp(data, "GIF87a", 6) != 0 &&
      std::memcmp(data, "GIF89a", 6) != 0)
    return std::nullopt;
  uint32_t w = uint32_t(data[6]) | (uint32_t(data[7]) << 8);
  uint32_t h = uint32_t(data[8]) | (uint32_t(data[9]) << 8);
  if (w == 0 || h == 0) return std::nullopt;
  return ImageDims{int(w), int(h)};
}

// TIFF: "II\x2A\x00" (little-endian) or "MM\x00\x2A" (big-endian) header, then
// a 4-byte offset to the first IFD. The IFD is a 2-byte entry count followed
// by 12-byte entries; ImageWidth (tag 256) and ImageLength (tag 257) carry the
// pixel grid as SHORT (type 3, inlined) or LONG (type 4). LZW/deflate/PackBits
// TIFF is a compressible bomb vector, so sniff the first IFD's dims. All reads
// are bounds-checked against len; a malformed/truncated header returns nullopt
// and falls through to the post-decode safety net.
inline std::optional<ImageDims>
peek_tiff_dims(const unsigned char *data, size_t len) {
  if (len < 8) return std::nullopt;
  bool le;
  if (std::memcmp(data, "II\x2A\x00", 4) == 0) le = true;
  else if (std::memcmp(data, "MM\x00\x2A", 4) == 0) le = false;
  else return std::nullopt;
  auto rd16 = [le](const unsigned char *p) -> uint32_t {
    return le ? (uint32_t(p[0]) | (uint32_t(p[1]) << 8))
              : (uint32_t(p[1]) | (uint32_t(p[0]) << 8));
  };
  auto rd32 = [le](const unsigned char *p) -> uint32_t {
    return le ? (uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
                 (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24))
              : (uint32_t(p[3]) | (uint32_t(p[2]) << 8) |
                 (uint32_t(p[1]) << 16) | (uint32_t(p[0]) << 24));
  };
  uint32_t ifd_off = rd32(data + 4);
  if (ifd_off < 8 || static_cast<uint64_t>(ifd_off) + 2 > len)
    return std::nullopt;
  uint32_t count = rd16(data + ifd_off);
  const uint64_t entries_start = static_cast<uint64_t>(ifd_off) + 2;
  if (entries_start >= len) return std::nullopt;
  // Clamp to the entries that actually fit; never read past len.
  const uint64_t max_fit = (len - entries_start) / 12;
  if (count > max_fit) count = static_cast<uint32_t>(max_fit);
  uint32_t w = 0, h = 0;
  for (uint32_t e = 0; e < count; ++e) {
    const unsigned char *ent = data + entries_start + static_cast<uint64_t>(e) * 12;
    uint16_t tag = static_cast<uint16_t>(rd16(ent));
    if (tag != 256 && tag != 257) continue;
    uint16_t type = static_cast<uint16_t>(rd16(ent + 2));
    uint32_t val;
    if (type == 3) val = rd16(ent + 8);       // SHORT (value inlined)
    else if (type == 4) val = rd32(ent + 8);  // LONG
    else continue;
    if (tag == 256) w = val;
    else h = val;
  }
  if (w == 0 || h == 0 || w > INT_MAX || h > INT_MAX) return std::nullopt;
  return ImageDims{static_cast<int>(w), static_cast<int>(h)};
}

// BMP: "BM" + a 14-byte BITMAPFILEHEADER, then a DIB header whose first field
// (u32 LE at offset 14) is its own size. Two shapes carry the pixel grid:
//   * BITMAPCOREHEADER (size 12): width/height are u16 LE at offsets 18/20.
//   * BITMAPINFOHEADER and every later variant (size >= 40, plus the rare OS/2
//     size-16/64 forms): width/height are i32 LE at offsets 18/22; a negative
//     height marks a top-down image, so its magnitude is the real height.
// BMP is uncompressed, but that does NOT make it safe to skip: cv::imdecode
// allocates its output Mat from these HEADER dimensions, not from the body
// length, so a ~50-byte BMP declaring 60000x60000 makes libopencv attempt a
// ~10 GB allocation before any body-size or post-decode check runs. MAX_BODY_MB
// therefore does not bound it. Sniff it like the compressible formats so the
// shared per-image and aggregate caps reject it before the decoder is reached.
// All reads are bounds-checked; a truncated header returns nullopt and falls
// through to the post-decode net (a body too short to hold a valid header also
// can't drive a large allocation).
inline std::optional<ImageDims>
peek_bmp_dims(const unsigned char *data, size_t len) {
  if (len < 18 || data[0] != 'B' || data[1] != 'M') return std::nullopt;
  auto rd32 = [](const unsigned char *p) -> uint32_t {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
           (uint32_t(p[3]) << 24);
  };
  const uint32_t dib = rd32(data + 14);
  int64_t w, h;
  if (dib == 12) { // BITMAPCOREHEADER: u16 dims at 18/20
    if (len < 22) return std::nullopt;
    w = uint32_t(data[18]) | (uint32_t(data[19]) << 8);
    h = uint32_t(data[20]) | (uint32_t(data[21]) << 8);
  } else { // BITMAPINFOHEADER and later: i32 dims at 18/22
    if (len < 26) return std::nullopt;
    w = static_cast<int32_t>(rd32(data + 18));
    h = static_cast<int32_t>(rd32(data + 22));
    if (h < 0) h = -h; // top-down bitmap
  }
  if (w <= 0 || h <= 0 || w > INT_MAX || h > INT_MAX) return std::nullopt;
  return ImageDims{static_cast<int>(w), static_cast<int>(h)};
}

// Try every supported format. Returns nullopt if format is unrecognised
// (callers should fall through to post-decode validation, which still
// catches the rest as a safety net).
inline std::optional<ImageDims>
peek_image_dimensions(const unsigned char *data, size_t len) {
  if (auto d = peek_png_dims(data, len)) return d;
  if (auto d = peek_jpeg_dims(data, len)) return d;
  if (auto d = peek_webp_dims(data, len)) return d;
  if (auto d = peek_gif_dims(data, len)) return d;
  if (auto d = peek_tiff_dims(data, len)) return d;
  if (auto d = peek_bmp_dims(data, len)) return d;
  return std::nullopt;
}

} // namespace turbo_ocr::decode
