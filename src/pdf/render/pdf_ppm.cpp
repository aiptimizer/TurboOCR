// PPM parsing + decode: header parse, completeness probe, mmap decode.
#include "turbo_ocr/pdf/render/pdf_renderer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

// decode_ppm mmaps on POSIX and reads on Windows — see the branch there. The
// completeness probe above it also uses open/fstat, so both need the POSIX
// headers only on the platforms that have them.
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "pdf_renderer_internal.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"

using namespace turbo_ocr::render;

namespace {

// PPM RGB→BGR swap implementation selector (TURBO_PPM_SWAP=scalar forces the
// old byte loop; validated to {simd, scalar} at startup by ServerConfig).
// Default uses OpenCV's SIMD cvtColor, markedly faster on a single core.
// CPU-only path — no GPU required.
bool ppm_swap_use_simd() {
  static const bool simd = [] {
    std::string e = turbo_ocr::env::env_or("TURBO_PPM_SWAP", "");
    return !(!e.empty() && e == "scalar");
  }();
  return simd;
}

// Max rendered pixels per page (width*height). The per-side 16384 cap below
// still bounds a single dimension, but a 16384x16384 page is ~268MP → ~768MB
// raster + a same-size encoded image held in the response: this area cap
// rejects such pages (decode_ppm returns empty → the route reports a decode
// failure). Reads MAX_PDF_PAGE_PIXELS_MP (megapixels); ServerConfig validates
// it to [1,268] at startup. Default 40 MP (e.g. 5000x8000 at ~600 DPI A4).
} // namespace

// Shared with the darwin in-process renderer via pdf_renderer_internal.h —
// both platforms must read ONE area cap.
int64_t turbo_ocr::render::pdfrdetail::ppm_max_pixels() {
  static const int64_t px = [] {
    // env_int clamps to [1,268] and returns the default (40) on any malformed
    // value; a garbage value would previously revert to 40 with no diagnostic.
    const int def = 40;
    const int mp = turbo_ocr::env::env_int("MAX_PDF_PAGE_PIXELS_MP", def, 1, 268);
    if (std::string e = turbo_ocr::env::env_or("MAX_PDF_PAGE_PIXELS_MP", "");
        !e.empty() && mp == def) {
      // Distinguish "explicitly set to 40" from "malformed -> default".
      char *end = nullptr;
      std::strtol(e.c_str(), &end, 10);
      if (end == e.c_str() || *end != '\0')
        TOCR_LOG_WARN("MAX_PDF_PAGE_PIXELS_MP malformed; using default",
                      "value", e, "default", def);
    }
    return static_cast<int64_t>(mp) * 1000000;
  }();
  return px;
}

// PpmHeader + parse_ppm_header now live in the pdfrdetail namespace (declared in
// pdf_renderer_internal.h) so the fuzz target and unit tests can reach the pure
// parser directly. The body is unchanged.
namespace turbo_ocr::render::pdfrdetail {

PpmHeader parse_ppm_header(const unsigned char *base, size_t len) {
  PpmHeader hdr;
  const unsigned char *end = base + len;
  const unsigned char *p = base;
  if (len < 3 || p[0] != 'P' || (p[1] != '5' && p[1] != '6')) return hdr;
  hdr.gray = (p[1] == '5');
  p += 2;

  // Consume one header token (int), skipping whitespace and '#'-comments.
  auto next_int = [&](int &out) -> bool {
    while (p < end) {
      unsigned char c = *p;
      if (c == '#') { while (p < end && *p != '\n') ++p; continue; }
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++p; continue; }
      break;
    }
    if (p >= end || *p < '0' || *p > '9') return false;
    int v = 0;
    while (p < end && *p >= '0' && *p <= '9') {
      v = v * 10 + (*p - '0');
      if (v > 100000) return false;
      ++p;
    }
    out = v;
    return true;
  };

  int maxval = 0;
  if (!next_int(hdr.w) || !next_int(hdr.h) || !next_int(maxval)) return hdr;
  if (hdr.w <= 0 || hdr.h <= 0 || hdr.w > 16384 || hdr.h > 16384 || maxval != 255)
    return hdr;
  if (static_cast<int64_t>(hdr.w) * hdr.h > ppm_max_pixels())
    return hdr;  // area bomb guard
  // Exactly one whitespace byte separates maxval from the payload.
  if (p >= end) return hdr;
  ++p;

  hdr.payload_offset = static_cast<size_t>(p - base);
  hdr.payload_bytes =
      static_cast<size_t>(hdr.w) * hdr.h * (hdr.gray ? 1 : 3);
  hdr.valid = true;
  return hdr;
}

} // namespace

namespace turbo_ocr::render::pdfrdetail {

bool ppm_is_complete(const std::string &path) {
  // Reads the first 128 bytes (enough for any PPM header) and compares the
  // declared payload against the file size — the writer may still be flushing.
  // Same logic either way; only the file access differs.
#if defined(_WIN32)
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamoff sz = f.tellg();
  if (sz <= 0) return false;
  const auto file_size = static_cast<size_t>(sz);
  unsigned char buf[128];
  f.seekg(0);
  f.read(reinterpret_cast<char *>(buf), sizeof(buf));
  const std::streamsize got = f.gcount();
  if (got <= 0) return false;
#else
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  struct stat st{};
  if (::fstat(fd, &st) < 0) { ::close(fd); return false; }
  const auto file_size = static_cast<size_t>(st.st_size);
  unsigned char buf[128];
  const ssize_t got = ::read(fd, buf, sizeof(buf));
  ::close(fd);
  if (got <= 0) return false;
#endif
  const PpmHeader hdr = parse_ppm_header(buf, static_cast<size_t>(got));
  return hdr.valid && file_size >= hdr.payload_offset + hdr.payload_bytes;
}

} // namespace turbo_ocr::render::pdfrdetail

// PPM → BGR decoder. mmap the file, copy pixels into a cv::Mat with a
// single-pass RGB→BGR swap, then unlink the file. Unlinking immediately
// after mmap keeps /dev/shm usage bounded by the number of in-flight
// workers rather than the total page count — critical for large PDFs
// where N × ~3 MB/page would exhaust the default 64 MB Docker shm.
cv::Mat PdfRenderer::decode_ppm(const std::string &path) {
  // Everything below the acquisition is platform-neutral: it wants `base` and
  // `file_size` for a read-only PPM whose file has already been deleted. Only
  // HOW those bytes are obtained differs.
#if defined(_WIN32)
  // No mmap. The mapping is a zero-copy optimization whose second purpose —
  // freeing /dev/shm the instant the page is claimed — has no Windows analogue,
  // so a plain read costs one memcpy of a page-sized buffer and nothing else.
  // The buffer must outlive the parse below, hence function scope.
  std::vector<unsigned char> owned;
  {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamoff sz = f.tellg();
    if (sz < 3) return {};
    owned.resize(static_cast<size_t>(sz));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(owned.data()),
                static_cast<std::streamsize>(sz)))
      return {};
  }
  std::remove(path.c_str());
  const size_t file_size = owned.size();
  const unsigned char *base = owned.data();
#else
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return {};
  struct stat st{};
  if (::fstat(fd, &st) < 0 || st.st_size < 3) {
    ::close(fd);
    return {};
  }
  const size_t file_size = static_cast<size_t>(st.st_size);
  void *map = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map == MAP_FAILED) return {};
  // Unlink now: MAP_PRIVATE made a CoW snapshot so the mapping survives.
  // This frees /dev/shm space immediately instead of after StreamHandle cleanup.
  std::remove(path.c_str());

  struct Unmap {
    void *p;
    size_t n;
    ~Unmap() noexcept { if (p && p != MAP_FAILED) ::munmap(p, n); }
  } guard{map, file_size};

  const unsigned char *base = static_cast<const unsigned char *>(map);
#endif
  const pdfrdetail::PpmHeader hdr = pdfrdetail::parse_ppm_header(base, file_size);
  if (!hdr.valid) return {};
  const bool gray = hdr.gray;
  const int w = hdr.w, h = hdr.h;
  const unsigned char *p = base + hdr.payload_offset;
  const size_t expected = hdr.payload_bytes;
  if (file_size - hdr.payload_offset < expected) return {};

  if (gray) {
    cv::Mat g(h, w, CV_8UC1);
    std::memcpy(g.data, p, expected);
    cv::Mat bgr;
    cv::cvtColor(g, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
  }

  // Color: RGB (PPM) → BGR. OpenCV's cvtColor is a SIMD-vectorized channel
  // swap — meaningfully faster than a scalar byte loop on the single core that
  // bottlenecks PDF page-image throughput. The scalar loop is kept as a
  // fallback selectable with TURBO_PPM_SWAP=scalar.
  cv::Mat bgr(h, w, CV_8UC3);
  if (ppm_swap_use_simd()) {
    // Header over the mmap'd RGB payload — no copy; cvtColor reads it and
    // writes the owned `bgr`, both completing before the munmap at return.
    cv::Mat rgb(h, w, CV_8UC3, const_cast<unsigned char *>(p));
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  } else {
    const unsigned char *src = p;
    unsigned char *dst = bgr.data;
    const size_t n_px = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n_px; ++i) {
      dst[0] = src[2];
      dst[1] = src[1];
      dst[2] = src[0];
      src += 3;
      dst += 3;
    }
  }
  return bgr;
}
